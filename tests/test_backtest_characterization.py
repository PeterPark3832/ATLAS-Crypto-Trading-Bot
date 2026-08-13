"""
ATLAS — 백테스트 특성화(characterization) 테스트
================================
`backtest_strategy`는 순환복잡도 81짜리 함수다(권장 10 이하). 분해하려면
**결과가 한 비트도 달라지지 않았다**는 것을 먼저 증명할 수단이 필요하다.

이 파일은 기능을 검증하지 않는다. **현재 동작을 그대로 사진 찍는다.**
리팩터링 전후로 이 해시가 같으면 동작이 보존된 것이고, 다르면 어디선가
의미가 바뀐 것이다. 리팩터링이 끝나도 남겨 두면 이후 회귀도 잡는다.

(골든값은 코드에 박지 않고 **같은 실행 안에서 두 경로를 비교**하지 않는다.
 대신 파일에 저장된 스냅샷과 비교한다 — 그래야 '리팩터링 후'에도 의미가
 있다. 스냅샷이 없으면 처음 실행에서 만들고 통과시킨다)

실행:
  pytest tests/test_backtest_characterization.py -v
  ATLAS_UPDATE_SNAPSHOT=1 pytest tests/test_backtest_characterization.py  # 의도적 갱신
"""

import hashlib
import json
import os
from pathlib import Path


import numpy as np
import pytest

import atlas_spot_backtest as bt

SNAPSHOT = Path(__file__).parent / 'data' / 'backtest_snapshot.json'


# ══════════════════════════════════════════════════════════════
#  결정론적 합성 시장
# ══════════════════════════════════════════════════════════════

def _series(seed: int, n: int = 700, regime_period: int = 25) -> list:
    """시드 고정 합성 OHLCV. 상승/하락 구간을 번갈아 만들어 각 전략이
    진입·청산·시간청산을 모두 겪게 한다."""
    rng = np.random.default_rng(seed)
    px, out, ts = 100.0, [], 1577836800000      # 2020-01-01
    for i in range(n):
        o = px
        drift = 0.015 if (i // regime_period) % 2 == 0 else -0.008
        px = max(px * (1 + drift + rng.normal(0, 0.018)), 1.0)
        vol = 1e6 * (2.2 if drift > 0 else 0.9)
        out.append([ts + i * 86400000, o, max(o, px) * 1.012,
                    min(o, px) * 0.988, px, vol])
    return out


# 각 전략이 **실제로 진입할 수 있는** 레짐을 준다. 레짐이 맞지 않으면
# regime_block만 쌓여 진입·사이징·청산 경로를 하나도 밟지 못하고,
# 그러면 특성화가 아무것도 고정하지 못한다.
_FAVOURABLE = {
    'S3': ['WEAK_TREND'],
    'S4': ['RANGING', 'TRENDING_DOWN'],
    'S5': ['RANGING', 'WEAK_TREND'],
    'S6': ['TRENDING_UP', 'WEAK_TREND'],
}


def _regime_map(kind: str, sid: str = 'S6') -> dict:
    from datetime import datetime, timedelta, timezone
    base = datetime(2019, 12, 1, tzinfo=timezone.utc)
    if kind == 'empty':
        return {}
    if kind == 'favourable':
        seq = _FAVOURABLE[sid]
    elif kind == 'mixed':
        # 유리한 레짐 + 차단 레짐을 섞어 regime_block 경로도 함께 고정
        seq = _FAVOURABLE[sid] + ['CRISIS', 'MICRO_RANGING']
    else:
        seq = ['TRENDING_UP']
    out = {}
    for i in range(1200):
        out[(base + timedelta(days=i)).strftime('%Y-%m-%d')] = seq[i // 40 % len(seq)]
    return out


CASES = [
    ('S3', 'BTCUSDT',  1, 'favourable'),
    ('S3', 'DOTUSDT',  7, 'mixed'),
    ('S4', 'ETHUSDT',  2, 'favourable'),
    ('S4', 'XRPUSDT',  6, 'mixed'),
    ('S5', 'BNBUSDT',  3, 'favourable'),
    ('S5', 'LINKUSDT', 8, 'mixed'),
    ('S6', 'SOLUSDT',  4, 'favourable'),
    ('S6', 'ADAUSDT',  5, 'mixed'),
    ('S6', 'AVAXUSDT', 9, 'empty'),
    ('S4', 'ATOMUSDT', 10, 'empty'),
]


def _run_all() -> dict:
    """모든 케이스를 돌려 (거래 + 진단)을 정규화한 dict로 반환."""
    result = {}
    for sid, sym, seed, rk in CASES:
        trades, diag = bt.backtest_strategy(
            sid, sym, _series(seed), _regime_map(rk, sid),
            '2020-01-01', '2021-12-31')
        result[f'{sid}/{sym}/{rk}'] = {
            'n': len(trades),
            'trades': [
                {
                    'entry':   round(float(t.entry), 8),
                    'exit':    round(float(t.exit_px), 8),
                    'pnl_r':   round(float(t.pnl_r), 8),
                    'risk':    round(float(t.risk_pct), 10),
                    'reason':  t.reason,
                    'regime':  t.regime,
                    'bars':    [int(t.entry_bar), int(t.exit_bar)],
                    'hold_h':  round(float(t.hold_hours), 4),
                    'cost':    round(float(t.cost_usdt), 6),
                }
                for t in trades
            ],
            'diag': {k: int(v) for k, v in sorted(dict(diag).items())},
        }
    return result


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ══════════════════════════════════════════════════════════════
#  스냅샷 비교
# ══════════════════════════════════════════════════════════════

class TestCharacterization:
    def test_matches_snapshot(self):
        """리팩터링 전후로 **완전히 동일한 결과**여야 한다."""
        current = _run_all()
        if os.getenv('ATLAS_UPDATE_SNAPSHOT') == '1' or not SNAPSHOT.exists():
            SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT.write_text(
                json.dumps(current, indent=1, sort_keys=True, ensure_ascii=False))
            pytest.skip(f'스냅샷 생성: {SNAPSHOT.name} — 다시 실행하면 비교한다')

        saved = json.loads(SNAPSHOT.read_text())
        if _digest(saved) == _digest(current):
            return

        # 어디가 달라졌는지 짚어 준다 — 해시만 다르면 원인을 못 찾는다
        diffs = []
        for key in sorted(set(saved) | set(current)):
            a, b = saved.get(key), current.get(key)
            if a == b:
                continue
            if a is None or b is None:
                diffs.append(f'{key}: 한쪽에만 존재'); continue
            if a['n'] != b['n']:
                diffs.append(f'{key}: 거래수 {a["n"]} → {b["n"]}')
            if a['diag'] != b['diag']:
                diffs.append(f'{key}: 진단 {a["diag"]} → {b["diag"]}')
            for i, (ta, tb) in enumerate(zip(a['trades'], b['trades'], strict=False)):
                if ta != tb:
                    changed = {k: (ta[k], tb[k]) for k in ta if ta[k] != tb[k]}
                    diffs.append(f'{key} 거래#{i}: {changed}')
                    break
        pytest.fail('백테스트 동작이 바뀌었다:\n  ' + '\n  '.join(diffs[:15]))

    def test_snapshot_is_not_trivially_empty(self):
        """전 케이스가 0건이면 스냅샷이 아무것도 고정하지 못한다."""
        cur = _run_all()
        total = sum(v['n'] for v in cur.values())
        assert total >= 5, f'전체 거래 {total}건 — 표본이 너무 적어 의미가 없다'

    def test_deterministic_within_run(self):
        assert _digest(_run_all()) == _digest(_run_all())

    def test_covers_multiple_exit_reasons(self):
        """SL/TP/CROSS/TIME이 골고루 나와야 청산 경로가 실제로 고정된다."""
        reasons = {t['reason'] for v in _run_all().values() for t in v['trades']}
        assert len(reasons) >= 2, f'청산 사유가 {reasons} 뿐 — 경로 커버리지 부족'


class TestSnapshotIsVersioned:
    """스냅샷이 저장소에 없으면 이 테스트 파일 전체가 무의미해진다.

    test_matches_snapshot 은 파일이 없으면 새로 만들고 **skip** 한다.
    즉 스냅샷이 커밋되지 않으면 CI는 매번 새로 만들고 건너뛰므로,
    백테스트 동작이 바뀌어도 아무도 모른다 — 잠금 장치가 죽은 채로
    초록 불만 켜진다.

    실제로 그런 상태였다. .gitignore 의 `data/` 패턴이 (선행 슬래시가 없어)
    모든 깊이에서 매칭돼 tests/data/ 까지 무시했고, 스냅샷은 한 번도
    커밋되지 않았다.
    """

    def test_snapshots_are_not_gitignored(self):
        import subprocess
        for snap in (SNAPSHOT,
                     SNAPSHOT.parent / 'metrics_snapshot.json'):
            r = subprocess.run(['git', 'check-ignore', str(snap)],
                               capture_output=True, text=True,
                               cwd=str(Path(__file__).parent.parent))
            assert r.returncode != 0, (
                f'{snap.name} 이 .gitignore로 무시된다 — 커밋되지 않으면 '
                f'CI에서 특성화 테스트가 조용히 skip 되어 잠금이 풀린다')

    def test_snapshot_file_exists(self):
        assert SNAPSHOT.exists(), (
            '스냅샷이 없으면 test_matches_snapshot 이 생성 후 skip 한다 — '
            '동작 변화를 감지하지 못한다')

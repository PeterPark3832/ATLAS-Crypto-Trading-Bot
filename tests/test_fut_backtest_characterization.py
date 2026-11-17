"""
ATLAS 선물 — 백테스트 특성화(characterization) 테스트
=====================================================
현물 쪽 test_backtest_characterization.py 와 같은 방식이다. 기능을 검증하지 않고
**지금 동작을 그대로 사진 찍는다.** 수정이 결과를 한 비트라도 바꾸면 여기서 드러나고,
의도한 변경이면 스냅샷을 갱신하면서 그 차이를 리뷰에서 본다.

합성 시장은 시드를 고정하고 **사칙연산만으로** 만든다(random.gauss 는 cos/log 를 써서
OS마다 마지막 자리가 달라질 수 있다). BTC가 오르내리게 해서 장세 판정, S6 롱·SX 숏,
손절·익절·추세 이탈, 지정가·시장가 전환, 펀딩비 정산을 모두 한 번 이상 지나가게 한다.

실제 데이터 수치(FUTURES.md 의 연 9.7% / MDD 11.7%)는 데이터가 저장소에 없어서(data/ 무시)
CI에서 돌지 않는다. 그 재현은 맨 아래 test_real_data_reference 가 맡는다:
  ATLAS_FUT_DATA_DIR=data/futures pytest tests/test_fut_backtest_characterization.py -k reference

의도적 갱신 (스냅샷 파일이 없을 때 자동으로 만들지 않는다 — 없으면 실패):
  ATLAS_UPDATE_SNAPSHOT=1 pytest tests/test_fut_backtest_characterization.py
"""

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

import atlas_fut_backtest as bt
import atlas_fut_config as cfg
from atlas_fut_core import DAY_MS, H4_MS, Bar

DATA = Path(__file__).parent / 'data'
SNAPSHOT = DATA / 'fut_backtest_snapshot.json'
REFERENCE = DATA / 'fut_backtest_reference.json'
T0 = 1577836800000            # 2020-01-01
DAYS = 560
START = T0 + 230 * DAY_MS     # EMA200 이 자리 잡은 뒤부터 매매

# 60일 단위 일간 추세 (BTC는 장세를 돌리고, 알트는 서로 다른 순서로 돌파·이탈을 만든다)
PHASES = {
    'BTCUSDT': (0.012, 0.009, -0.006, 0.0, -0.010, 0.010, 0.0, -0.006),
    'ETHUSDT': (0.008, -0.002, -0.009, 0.006, 0.0, -0.008, 0.007, 0.0),
    'SOLUSDT': (-0.008, 0.009, 0.0, -0.007, 0.008, -0.003, -0.009, 0.006),
    'XRPUSDT': (0.0, -0.009, 0.009, 0.004, -0.008, 0.0, 0.008, -0.006),
}
SEEDS = {'BTCUSDT': 11, 'ETHUSDT': 22, 'SOLUSDT': 33, 'XRPUSDT': 44}


def _noise(rng: random.Random) -> float:
    """평균 0, 표준편차 1에 가까운 잡음 (균등분포 12개 합 − 6, 사칙연산만)."""
    return sum(rng.random() for _ in range(12)) - 6.0


def _market(symbol: str) -> list[Bar]:
    rng = random.Random(SEEDS[symbol])
    px, bars = 100.0, []
    for d in range(DAYS):
        drift = PHASES[symbol][(d // 60) % len(PHASES[symbol])]
        spike = 3.0 if rng.random() < 0.12 else 1.0          # 거래량이 터지는 날 (S6 조건: 평균의 2배)
        for k in range(6):
            o = px
            px = round(max(px * (1 + drift / 6 + 0.012 * _noise(rng)), 1.0), 6)
            hi = round(max(o, px) * (1 + 0.004 * rng.random()), 6)
            lo = round(min(o, px) * (1 - 0.004 * rng.random()), 6)
            v = round(1000.0 * (0.5 + rng.random()) * spike, 3)
            bars.append(Bar(T0 + d * DAY_MS + k * H4_MS, o, hi, lo, px, v))
    return bars


def _funding(symbol: str) -> tuple[list[int], list[float]]:
    rng = random.Random(SEEDS[symbol] + 1000)
    ts = [T0 + i * 8 * 3600_000 for i in range(DAYS * 3)]
    # 대부분 +0.01% 근처, 가끔 과밀(±0.06%) — 펀딩 필터 경로도 지나가게
    rates = [round(0.0001 + 0.0002 * (rng.random() - 0.5) + (0.0006 if rng.random() < 0.03 else 0.0)
                   - (0.0006 if rng.random() < 0.03 else 0.0), 8) for _ in ts]
    return ts, rates


# 실행에만 관여하고 백테스트 수치에는 영향이 없는 설정. 나머지 설정은 전부 기준값(reference)에 잠근다 —
# 새 상수를 추가하면 둘 중 하나로 분류해야 테스트가 통과한다.
RUNTIME_ONLY = {
    'FUT_EQUITY_JUMP_PCT', 'ENTRY_MAX_REPEG', 'DAY_CLOSE_DELAY_SEC', 'POLL_SEC', 'ALERT_COOLDOWN_SEC',
    'FUT_LIVE_BASE_URL', 'FUT_TESTNET_BASE_URL', 'FUT_DB_FILE', 'FUT_LOG_DIR', 'FUT_KILL_SWITCH',
    'FUT_LIVE_CONFIRM_PHRASE',
}


def _locked_params() -> dict:
    """FUTURES.md·기준값의 백테스트 수치를 만든 설정 (JSON 형태로 정규화)."""
    names = sorted(k for k in vars(cfg) if k.isupper() and k not in RUNTIME_ONLY)
    return {k: json.loads(json.dumps(getattr(cfg, k))) for k in names}


def _universe() -> dict:
    now = T0 + DAYS * DAY_MS
    out = {}
    for s in PHASES:
        ft, fr = _funding(s)
        out[s] = bt.build_symbol(_market(s), ft, fr, now)
    return out


def _run_all() -> dict:
    r = bt.simulate(_universe(), bt.BTConfig(start_ms=START))
    return {
        'n': len(r.trades),
        'final': round(r.final, 6),
        'max_dd': round(r.max_dd, 8),
        'orders': dict(sorted(r.orders.items())),
        'costs': {k: round(v, 6) for k, v in sorted(r.costs.items())},
        'trades': [
            {'s': t.symbol, 'st': t.st, 'side': t.side, 'in': t.entry_ts, 'out': t.exit_ts,
             'entry': round(t.entry, 6), 'exit': round(t.exit_px, 6), 'reason': t.reason,
             'net': round(t.net, 6), 'fund': round(t.fund, 6), 'regime': t.regime}
            for t in r.trades
        ],
    }


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TestCharacterization:
    def test_matches_snapshot(self):
        current = _run_all()
        if os.getenv('ATLAS_UPDATE_SNAPSHOT') == '1':
            SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True, ensure_ascii=False), encoding='utf-8')
            pytest.skip(f'스냅샷 갱신: {SNAPSHOT.name} — 바뀐 거래를 리뷰에서 확인할 것')
        # 파일이 없을 때 여기서 새로 만들면 '지금 동작'이 기준이 되어 회귀를 놓친다. 게다가 이 테스트가
        # 존재 검사(TestSnapshotIsVersioned)보다 먼저 돌아서 방금 만든 파일로 존재 검사까지 통과했다.
        if not SNAPSHOT.exists():
            pytest.fail(f'{SNAPSHOT.name} 이 없다 — 병합 등으로 빠졌으면 저장소에서 되살리고, '
                        f'처음 만드는 것이면 ATLAS_UPDATE_SNAPSHOT=1 로 의도적으로 만들 것')
        saved = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
        if _digest(saved) == _digest(current):
            return
        diffs = [f'{k}: {saved.get(k)} → {current.get(k)}'
                 for k in ('n', 'final', 'max_dd', 'orders', 'costs') if saved.get(k) != current.get(k)]
        for i, (a, b) in enumerate(zip(saved['trades'], current['trades'], strict=False)):
            if a != b:
                diffs.append(f'거래#{i}: ' + str({k: (a[k], b.get(k)) for k in a if a[k] != b.get(k)}))
                break
        pytest.fail('선물 백테스트 동작이 바뀌었다 (의도한 변경이면 ATLAS_UPDATE_SNAPSHOT=1):\n  '
                    + '\n  '.join(diffs[:12]))

    def test_deterministic_within_run(self):
        assert _digest(_run_all()) == _digest(_run_all())

    def test_snapshot_exercises_both_sides_and_exit_paths(self):
        """표본이 한쪽 전략·한 가지 청산만 지나가면 스냅샷이 고정하는 게 거의 없다."""
        cur = _run_all()
        assert cur['n'] >= 8, f'거래 {cur["n"]}건 — 표본이 너무 적다'
        assert {t['st'] for t in cur['trades']} == {'S6', 'SX'}
        assert len({t['reason'] for t in cur['trades']}) >= 3
        assert cur['orders']['filled'] > 0 and cur['orders']['fallback'] > 0


def test_drawdown_halt_is_sticky_like_live():
    """낙폭 20% 정지는 라이브처럼 한 번 걸리면 풀리지 않는다(라이브는 사람이 --resume 해야 재개).
    예전 백테스트는 보유 포지션이 회복해 낙폭이 20% 아래로 돌아오면 저절로 다시 진입했고, 정지 자체가
    '수량 > 0' 비교 하나로만 구현돼 있었는데 어떤 테스트도 낙폭 20%까지 가지 않았다(2026-11-17 재검증).
    위험을 10%로 올려 일부러 넘긴다."""
    r = bt.simulate(_universe(), bt.BTConfig(start_ms=START, risk=0.10))
    assert r.max_dd >= 0.20 and r.halted_at is not None
    assert all(t.risk_usd > 0 for t in r.trades)                  # 수량 0 거래가 끼지 않는다
    assert all(t.entry_ts <= r.halted_at for t in r.trades)       # 정지 뒤 새로 산 거래가 없다
    assert bt.summarize(r)['halted_at'] == r.halted_at
    calm = bt.simulate(_universe(), bt.BTConfig(start_ms=START))
    assert calm.halted_at is None


class TestSnapshotIsVersioned:
    def test_snapshot_file_exists(self):
        assert SNAPSHOT.exists() and REFERENCE.exists()

    def test_snapshots_are_not_gitignored(self):
        for snap in (SNAPSHOT, REFERENCE):
            try:
                r = subprocess.run(['git', 'check-ignore', str(snap)], capture_output=True, text=True,
                                   cwd=str(Path(__file__).parent.parent), timeout=20)
            except (OSError, subprocess.SubprocessError) as e:
                pytest.skip(f'git 을 실행할 수 없다: {e}')
            if r.returncode == 128:
                pytest.skip(f'git 저장소를 읽을 수 없다: {r.stderr.strip()[:80]}')
            assert r.returncode != 0, f'{snap.name} 이 .gitignore 로 무시된다 — 커밋되지 않으면 잠금이 풀린다'


class TestReferenceParams:
    """설정값이 바뀌면 문서·기준값의 수치는 더 이상 그 설정의 결과가 아니다.
    2026-11-17 변이 시험에서 설정 상수 44개 중 27개(장세 임계값, 펀딩 필터, 손절폭 상한, 동시 보유 수 등)를
    바꿔도 선물 테스트가 전부 통과했다. 여기서 한꺼번에 잠근다. 의도한 변경이면 실제 데이터로 다시 돌려 갱신하고
      ATLAS_FUT_DATA_DIR=data/futures ATLAS_UPDATE_REFERENCE=1 pytest tests/test_fut_backtest_characterization.py -k reference
    FUTURES.md 숫자를 같이 고친다."""

    def test_params_match_reference(self):
        ref = json.loads(REFERENCE.read_text(encoding='utf-8'))['params']
        cur = _locked_params()
        changed = {k: (ref.get(k), cur.get(k)) for k in sorted(set(ref) | set(cur)) if ref.get(k) != cur.get(k)}
        assert not changed, (
            '백테스트 수치를 만든 설정이 바뀌었다(기준값 → 지금): '
            + ', '.join(f'{k} {a} → {b}' for k, (a, b) in changed.items())
            + ' — 의도한 변경이면 실제 데이터로 다시 돌려 기준값과 FUTURES.md를 갱신하고, '
              '실행에만 쓰는 설정이면 RUNTIME_ONLY 에 넣을 것')

    def test_runtime_only_names_exist(self):
        missing = sorted(n for n in RUNTIME_ONLY if not hasattr(cfg, n))
        assert not missing, f'RUNTIME_ONLY 에 없는 설정이 있다(이름이 바뀌었나?): {missing}'


def test_real_data_reference():
    """FUTURES.md 숫자 재현. 데이터는 공개 API에서 받는다(키 불필요):
      python atlas_fut_backtest.py --data-dir data/futures --fetch --now-ms <reference.now_ms>
    """
    d = os.getenv('ATLAS_FUT_DATA_DIR')
    if not d:
        pytest.skip('ATLAS_FUT_DATA_DIR 미설정 — 실제 데이터 재현은 수동으로 돌린다')
    ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
    data = bt.load_universe(Path(d), cfg.FUT_UNIVERSE, ref['now_ms'])
    s = bt.summarize(bt.simulate(data, bt.BTConfig(start_ms=ref['start_ms'], risk=ref['risk'])))
    got = {'final': s['final'], 'trades': s['trades'], 'max_dd': round(s['max_dd'], 4),
           'cagr': round(s['cagr'], 4), 'pf': round(s['pf'], 2), 'win_rate': round(s['win_rate'], 2)}
    if os.getenv('ATLAS_UPDATE_REFERENCE') == '1':
        ref.update(expect=got, params=_locked_params())
        REFERENCE.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding='utf-8')
        pytest.skip(f'기준값 갱신: {got} — FUTURES.md 숫자도 같이 고칠 것')
    assert got == ref['expect'], f'실제 데이터 결과가 문서와 다르다: {got} ≠ {ref["expect"]}'

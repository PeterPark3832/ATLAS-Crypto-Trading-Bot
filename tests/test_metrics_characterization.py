"""
ATLAS — 성과지표 특성화 (calc_spot_metrics)
================================
순환복잡도 34짜리 함수를 분해하기 전에 현재 동작을 스냅샷으로 고정한다.
지표는 WFO 합격/불합격과 재최적화 제안을 좌우하므로, 값이 조금만 달라져도
"어떤 파라미터를 쓸 것인가"가 바뀐다.

기존 테스트는 4개뿐이었고 Sharpe·Sortino·Calmar·CAGR·드로다운 지속은
전혀 고정돼 있지 않았다.

실행:
  pytest tests/test_metrics_characterization.py -v
  ATLAS_UPDATE_SNAPSHOT=1 pytest tests/test_metrics_characterization.py
"""

import json
import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import atlas_spot_backtest as bt

SNAPSHOT = Path(__file__).parent / 'data' / 'metrics_snapshot.json'


def _trades(pattern: str, n: int, seed: int = 3):
    """지표의 각 분기를 밟는 거래 묶음."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if pattern == 'winning':
            r = abs(rng.normal(1.2, 0.5))
        elif pattern == 'losing':
            r = -abs(rng.normal(0.9, 0.3))
        elif pattern == 'mixed':
            r = rng.normal(0.15, 1.1)
        elif pattern == 'all_wins':
            r = 1.5
        elif pattern == 'all_losses':
            r = -1.0
        elif pattern == 'flat':
            r = 0.0
        else:                       # drawdown_then_recover
            r = -1.0 if i < n // 2 else 2.0
        reason = ('SL' if r <= 0 else
                  ['TP', 'BB_MID', 'DON_LOW', 'TIME'][i % 4])
        out.append(bt.SpotTrade(
            symbol='X', strategy='S6', direction='LONG', mode='',
            entry=100.0, exit_px=100.0 + r, pnl_r=round(float(r), 6),
            risk_pct=0.02, reason=reason, entry_bar=i, exit_bar=i + 1,
            regime='RANGING', hold_hours=round(4.0 * (i % 7 + 1), 1),
            cost_usdt=50.0))
    return out


CASES = [
    ('winning-40',   'winning', 40, '2021-01-01', '2022-12-31'),
    ('losing-40',    'losing', 40, '2021-01-01', '2022-12-31'),
    ('mixed-120',    'mixed', 120, '2021-01-01', '2023-12-31'),
    ('all_wins-30',  'all_wins', 30, '2021-01-01', '2021-12-31'),
    ('all_losses-30', 'all_losses', 30, '2021-01-01', '2021-12-31'),
    ('flat-20',      'flat', 20, '2021-01-01', '2021-12-31'),
    ('dd_recover-60', 'dd', 60, '2021-01-01', '2022-06-30'),
    ('no-dates',     'mixed', 50, '', ''),
    ('bad-dates',    'mixed', 50, 'not-a-date', '2022-01-01'),
    ('single',       'winning', 1, '2021-01-01', '2021-06-30'),
]


def _run_all() -> dict:
    return {
        name: bt.calc_spot_metrics(_trades(pat, n), 10_000.0, s, e)
        for name, pat, n, s, e in CASES
    }


class TestMetricsCharacterization:
    def test_matches_snapshot(self):
        current = _run_all()
        if os.getenv('ATLAS_UPDATE_SNAPSHOT') == '1' or not SNAPSHOT.exists():
            SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True))
            pytest.skip('스냅샷 생성 — 다시 실행하면 비교한다')
        saved = json.loads(SNAPSHOT.read_text())
        diffs = []
        for k in sorted(set(saved) | set(current)):
            a, b = saved.get(k, {}), current.get(k, {})
            for key in sorted(set(a) | set(b)):
                if a.get(key) != b.get(key):
                    diffs.append(f'{k}.{key}: {a.get(key)} → {b.get(key)}')
        assert not diffs, '지표 계산이 바뀌었다:\n  ' + '\n  '.join(diffs[:20])

    def test_all_metric_keys_present(self):
        """키가 사라지면 대시보드·WFO 리포트가 조용히 0을 표시한다."""
        need = {'total_trades', 'win_rate', 'avg_win_r', 'avg_loss_r',
                'profit_factor', 'max_dd_pct', 'sharpe', 'sortino', 'calmar',
                'cagr_pct', 'total_pnl_pct', 'final_equity', 'sl_count',
                'tp_count', 'time_count', 'avg_hold_hours'}
        assert need <= set(_run_all()['mixed-120'])

    def test_snapshot_exercises_all_branches(self):
        """전 케이스가 같은 값이면 스냅샷이 아무것도 고정하지 못한다."""
        cur = _run_all()
        assert len({m['sharpe'] for m in cur.values()}) >= 4
        assert any(m['max_dd_pct'] > 0 for m in cur.values())
        assert any(m['cagr_pct'] != 0 for m in cur.values())
        assert any(m['sortino'] != 0 for m in cur.values())

    def test_empty_is_special_cased(self):
        assert bt.calc_spot_metrics([]) == {'total_trades': 0}

    def test_deterministic(self):
        assert _run_all() == _run_all()

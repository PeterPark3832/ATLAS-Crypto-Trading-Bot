"""
ATLAS — 파라미터 최적화 도구 (과최적화 방지 로직)
================================
백테스트 성적을 올리는 파라미터는 언제나 찾을 수 있다. 이 도구의 가치는
**찾는 능력이 아니라 거절하는 능력**에 있으므로, 거절 조건을 고정한다.

실행:
  pytest tests/test_optimize.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_optimize as opt
from atlas_spot_config import WF_OOS_MIN_PF, WF_OOS_MIN_SHARPE


def _m(trades=100, pf=2.0, sharpe=1.0, mdd=10.0, pnl=20.0):
    return {'total_trades': trades, 'profit_factor': pf, 'sharpe': sharpe,
            'max_dd_pct': mdd, 'total_pnl_pct': pnl}


# ══════════════════════════════════════════════════════════════
#  점수 — 노이즈 설정을 배제하는가
# ══════════════════════════════════════════════════════════════

class TestScore:
    def test_thin_sample_scores_zero(self):
        """표본이 적으면 성적이 아니라 운이다."""
        assert opt._score(_m(trades=opt.MIN_TRADES - 1, pf=9.0, sharpe=9.0)) == 0.0

    def test_losing_config_scores_zero(self):
        assert opt._score(_m(pf=0.9)) == 0.0
        assert opt._score(_m(sharpe=-0.5)) == 0.0

    def test_higher_pf_scores_higher(self):
        assert opt._score(_m(pf=2.5)) > opt._score(_m(pf=1.5))

    def test_deeper_drawdown_penalised(self):
        assert opt._score(_m(mdd=30.0)) < opt._score(_m(mdd=10.0))

    def test_diverging_pf_is_capped(self):
        """PF 999 같은 발산값이 점수를 지배하지 않도록 절단한다."""
        assert opt._score(_m(pf=999.0)) == opt._score(_m(pf=5.0))

    def test_zero_trades_safe(self):
        assert opt._score({'total_trades': 0}) == 0.0


# ══════════════════════════════════════════════════════════════
#  탐색 공간
# ══════════════════════════════════════════════════════════════

class TestSearchSpace:
    def test_full_grid_when_small(self):
        grid = {'a': [1, 2], 'b': [3, 4]}
        assert len(opt._combos(grid, trials=99, seed=1)) == 4

    def test_sampled_when_large(self):
        grid = {'a': list(range(10)), 'b': list(range(10))}
        c = opt._combos(grid, trials=15, seed=1)
        assert len(c) == 15

    def test_sampling_is_reproducible(self):
        """시드를 고정하면 같은 후보 집합이 나와야 결과를 재현할 수 있다."""
        grid = {'a': list(range(10)), 'b': list(range(10))}
        assert opt._combos(grid, 15, seed=7) == opt._combos(grid, 15, seed=7)
        assert opt._combos(grid, 15, seed=7) != opt._combos(grid, 15, seed=8)

    def test_neighbors_move_one_axis(self):
        grid = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        ns = opt._neighbors({'a': 2, 'b': 20}, grid)
        assert {'a': 1, 'b': 20} in ns and {'a': 3, 'b': 20} in ns
        assert {'a': 2, 'b': 10} in ns and {'a': 2, 'b': 30} in ns
        assert len(ns) == 4

    def test_neighbors_at_edge(self):
        grid = {'a': [1, 2, 3]}
        assert opt._neighbors({'a': 1}, grid) == [{'a': 2}]

    def test_every_strategy_grid_axes_are_lists(self):
        for s, g in opt.PARAM_GRID.items():
            assert g, f'{s} 탐색 공간이 비어 있다'
            for k, v in g.items():
                assert isinstance(v, list) and len(v) >= 2, f'{s}.{k}'


# ══════════════════════════════════════════════════════════════
#  판정 — 거절해야 할 때 거절하는가
# ══════════════════════════════════════════════════════════════

def _res(oos_best, is_best=None, oos_base=None):
    return {
        'strategy': 'S3', 'trials': 20, 'evaluated': 40,
        'best_params': {'S3_ADX_MIN': 25},
        'is_best': is_best or _m(pf=3.0, sharpe=2.0),
        'oos_best': oos_best,
        'is_baseline': _m(pf=2.0, sharpe=1.0),
        'oos_baseline': oos_base or _m(pf=1.5, sharpe=0.5),
        'plateau': 1.0, 'is_score': 2.0,
    }


class TestVerdict:
    def test_accepts_when_all_gates_pass(self, capsys):
        assert opt.report(_res(_m(trades=80, pf=2.0, sharpe=1.5))) is True

    def test_rejects_on_thin_oos_sample(self):
        """OOS 성적이 좋아도 표본이 얇으면 채택하지 않는다."""
        assert opt.report(_res(_m(trades=opt.MIN_TRADES - 1,
                                  pf=5.0, sharpe=5.0))) is False

    def test_rejects_when_below_sharpe_gate(self):
        assert opt.report(_res(_m(sharpe=WF_OOS_MIN_SHARPE - 0.01))) is False

    def test_rejects_when_below_pf_gate(self):
        assert opt.report(_res(_m(pf=WF_OOS_MIN_PF - 0.01))) is False

    def test_rejects_when_not_better_than_current(self):
        """현재 설정보다 못하면 바꿀 이유가 없다."""
        assert opt.report(_res(_m(sharpe=1.0), oos_base=_m(sharpe=2.0))) is False

    def test_empty_result_rejected(self):
        assert opt.report({}) is False

    def test_reports_multiple_testing_count(self, capsys):
        opt.report(_res(_m(trades=80, pf=2.0, sharpe=1.5)))
        out = capsys.readouterr().out
        assert '다중검정' in out and '40개' in out, '평가 횟수를 반드시 알려야 한다'

    def test_reports_is_oos_degradation(self, capsys):
        opt.report(_res(_m(trades=80, pf=2.0, sharpe=1.0),
                        is_best=_m(pf=4.0, sharpe=4.0)))
        out = capsys.readouterr().out
        assert '열화율' in out
        assert '과최적화 의심' in out, 'Sharpe 4.0 → 1.0(75% 열화)은 경고 대상'

    def test_no_overfit_warning_when_stable(self, capsys):
        opt.report(_res(_m(trades=80, pf=2.0, sharpe=1.8),
                        is_best=_m(pf=2.2, sharpe=2.0)))
        assert '과최적화 의심' not in capsys.readouterr().out

    def test_handles_zero_trade_oos(self):
        assert opt.report(_res({'total_trades': 0})) is False

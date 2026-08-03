"""
ATLAS — 재최적화 과최적화 방지 장치
================================
`reoptimize.py`는 이미 IS 전용 탐색 / OOS 1회 검증 / 최소 거래수 게이트를
갖추고 있다. 여기서는 그 위에 추가된 두 가지를 고정한다:

  ① 고원(plateau) 확인 — 이웃 파라미터도 함께 좋아야 제안한다.
     한 점만 튀는 IS 최고점은 거의 항상 과거 노이즈다.
  ② 과최적화 진단 노출 — 평가한 조합 수(다중검정)와 IS→OOS 열화율을
     제안과 같은 화면에 보여준다. 따로 두면 사람이 안 본다.

실행:
  pytest tests/test_reoptimize_guards.py -v
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import reoptimize as ro


# ══════════════════════════════════════════════════════════════
#  이웃 탐색
# ══════════════════════════════════════════════════════════════

class TestGlobalParamOverride:
    """추적 손절 같은 **실행 규칙**은 전략 모듈이 아니라 라이브·백테스트
    양쪽 전역에 있다. 한쪽만 바꾸면 검증이 실제 동작과 어긋난다."""

    def test_patches_both_live_and_backtest(self):
        import atlas_spot_backtest as bt
        import atlas_spot_main as sm
        before = (sm.SPOT_TRAIL_ENABLED, bt.SPOT_TRAIL_ENABLED)
        with ro.override_params({'SPOT_TRAIL_ENABLED': True}):
            assert sm.SPOT_TRAIL_ENABLED is True
            assert bt.SPOT_TRAIL_ENABLED is True
        assert (sm.SPOT_TRAIL_ENABLED, bt.SPOT_TRAIL_ENABLED) == before

    def test_strategy_constants_still_work(self):
        before = ro.strat.S3_ADX_MIN
        with ro.override_params({'S3_ADX_MIN': 99}):
            assert ro.strat.S3_ADX_MIN == 99
        assert ro.strat.S3_ADX_MIN == before

    def test_restores_on_exception(self):
        before = ro.strat.S3_ADX_MIN
        with pytest.raises(RuntimeError):
            with ro.override_params({'S3_ADX_MIN': 77}):
                raise RuntimeError('중단')
        assert ro.strat.S3_ADX_MIN == before, '예외가 나도 원복돼야 한다'

    def test_unknown_param_raises(self):
        with pytest.raises(KeyError):
            with ro.override_params({'NOT_A_REAL_PARAM': 1}):
                pass

    def test_current_params_reads_right_module(self):
        cur = ro.current_params('S3')
        assert 'SPOT_TRAIL_ENABLED' in cur
        import atlas_spot_main as sm
        assert cur['SPOT_TRAIL_ENABLED'] == sm.SPOT_TRAIL_ENABLED


class TestPlateauExclusion:
    """고원 개념은 '파라미터 표면이 매끄러운가'라서 순서 있는 수치 축에만
    의미가 있다. ON/OFF 토글은 이웃이 항상 반대값이라, 효과가 클수록
    오히려 '고립된 피크'로 오판된다."""

    def test_toggle_axis_excluded_from_neighbors(self):
        combo = {'S3_ADX_MIN': 25, 'S3_ATR_SL': 2.5, 'SPOT_TRAIL_ENABLED': True}
        ns = ro._neighbors(combo, ro.GRIDS['S3'])
        assert ns, '수치 축 이웃은 있어야 한다'
        assert all(n['SPOT_TRAIL_ENABLED'] is True for n in ns), (
            '토글 축을 흔들면 그 기능의 효과가 곧 고립 피크로 오판된다')

    def test_excluded_axis_declared(self):
        assert 'SPOT_TRAIL_ENABLED' in ro.PLATEAU_EXCLUDE

    def test_trailing_only_in_trend_strategies(self):
        """평균회귀(S4·S5)는 되돌림을 먹는 구조라 조기 청산 위험이 크고,
        축을 늘릴수록 다중검정 인플레만 커진다."""
        assert 'SPOT_TRAIL_ENABLED' in ro.GRIDS['S3']
        assert 'SPOT_TRAIL_ENABLED' in ro.GRIDS['S6']
        assert 'SPOT_TRAIL_ENABLED' not in ro.GRIDS['S4']
        assert 'SPOT_TRAIL_ENABLED' not in ro.GRIDS['S5']


class TestNeighbors:
    def test_moves_one_axis_at_a_time(self):
        grid = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        ns = ro._neighbors({'a': 2, 'b': 20}, grid)
        assert {'a': 1, 'b': 20} in ns and {'a': 3, 'b': 20} in ns
        assert {'a': 2, 'b': 10} in ns and {'a': 2, 'b': 30} in ns
        assert len(ns) == 4

    def test_edge_value_has_one_neighbor(self):
        assert ro._neighbors({'a': 1}, {'a': [1, 2, 3]}) == [{'a': 2}]

    def test_value_missing_from_grid_is_skipped(self):
        """현재값이 그리드에 없어도 죽지 않아야 한다."""
        assert ro._neighbors({'a': 99}, {'a': [1, 2, 3]}) == []

    def test_real_grids_have_neighbors(self):
        for sid, grid in ro.GRIDS.items():
            mid = {k: v[len(v) // 2] for k, v in grid.items()}
            assert ro._neighbors(mid, grid), f'{sid} 이웃이 없다'


# ══════════════════════════════════════════════════════════════
#  IS 점수
# ══════════════════════════════════════════════════════════════

def _m(trades=50, pf=2.0, sharpe=1.0):
    return {'total_trades': trades, 'profit_factor': pf, 'sharpe': sharpe}


class TestIsScore:
    def test_thin_sample_scores_zero(self):
        assert ro._is_score(_m(trades=ro.MIN_TRADES - 1, pf=9, sharpe=9)) == 0.0

    def test_losing_config_scores_zero(self):
        assert ro._is_score(_m(pf=0.0)) == 0.0
        assert ro._is_score(_m(sharpe=-1.0)) == 0.0

    def test_diverging_pf_capped(self):
        """PF 999 같은 발산값이 점수를 지배하면 안 된다."""
        assert ro._is_score(_m(pf=999.0)) == ro._is_score(_m(pf=5.0))

    def test_better_metrics_score_higher(self):
        assert ro._is_score(_m(pf=3.0)) > ro._is_score(_m(pf=1.5))
        assert ro._is_score(_m(sharpe=2.0)) > ro._is_score(_m(sharpe=1.0))

    def test_empty_metrics_safe(self):
        assert ro._is_score({}) == 0.0


# ══════════════════════════════════════════════════════════════
#  제안 메시지에 진단이 드러나는가
# ══════════════════════════════════════════════════════════════

def _prop(**kw):
    base = {
        'sid': 'S4', 'name': 'RSI Reversal', 'accepted': True, 'reason': '',
        'current': {'S4_RSI_ENTRY': 30}, 'proposed': {'S4_RSI_ENTRY': 25},
        'baseline': {'is': _m(), 'oos': _m(pf=1.2, sharpe=0.5)},
        'candidate': {'is': _m(pf=3.0, sharpe=2.0), 'oos': _m(pf=1.8, sharpe=1.2)},
        'n_combos': 27, 'evaluated': 31,
        'peak_is_score': 6.0, 'plateau_is_score': 5.0,
        'isolated_peak': False, 'is_oos_degrade_pct': 40.0,
    }
    base.update(kw)
    return base


class TestProposalDiagnostics:
    def test_shows_evaluated_count(self):
        msg = ro.format_proposal([_prop()], datetime.now(timezone.utc))
        assert '31개 조합' in msg, '다중검정 규모를 알려야 한다'

    def test_shows_plateau_and_peak(self):
        msg = ro.format_proposal([_prop()], datetime.now(timezone.utc))
        assert '이웃평균/최고점' in msg and '5.00/6.00' in msg

    def test_shows_degradation(self):
        msg = ro.format_proposal([_prop()], datetime.now(timezone.utc))
        assert 'IS→OOS 열화 +40%' in msg

    def test_flags_severe_degradation(self):
        msg = ro.format_proposal([_prop(is_oos_degrade_pct=75.0)],
                                 datetime.now(timezone.utc))
        assert '+75%' in msg and '⚠️' in msg

    def test_missing_degradation_is_safe(self):
        msg = ro.format_proposal([_prop(is_oos_degrade_pct=None)],
                                 datetime.now(timezone.utc))
        assert '열화' not in msg and 'S4' in msg

    def test_rejected_proposal_shows_reason(self):
        msg = ro.format_proposal(
            [_prop(accepted=False, reason='고립된 피크 — 이웃 IS 점수가 최고점의 20%뿐')],
            datetime.now(timezone.utc))
        assert '개선 제안 없음' in msg and '고립된 피크' in msg


# ══════════════════════════════════════════════════════════════
#  고립된 피크 판정 기준
# ══════════════════════════════════════════════════════════════

class TestPlateauThreshold:
    def test_ratio_constant_is_sane(self):
        assert 0 < ro.PLATEAU_MIN_RATIO < 1

    @pytest.mark.parametrize('plateau,peak,expected', [
        (5.0, 6.0, False),   # 이웃도 좋음 → 고원
        (1.0, 6.0, True),    # 이웃이 크게 낮음 → 고립된 피크
        (3.0, 6.0, False),   # 정확히 50% → 경계(포함)
        (2.9, 6.0, True),
    ])
    def test_isolation_rule(self, plateau, peak, expected):
        assert bool(peak > 0 and plateau < peak * ro.PLATEAU_MIN_RATIO) is expected

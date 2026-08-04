"""
ATLAS — 동적 RR (calc_dynamic_rr_ma)
================================
정적 분석(ruff F401)이 "쓰지 않는 import"를 지적한 데서 출발해 세 개의
결함이 나왔다. 셋 다 **코드는 멀쩡히 돌고 테스트도 통과하는** 종류다.

  ① 범위 상수가 죽어 있었다 — 1.5~3.0이 함수 안에 하드코딩돼 있고,
     config의 S3/S6/S7_RR_MIN·MAX 6개는 그 값과 **우연히 같아서**
     동작하는 것처럼 보였다. S6_RR_MAX를 5.0으로 바꿔도 TP는 그대로였다.
     WFO 그리드에 넣었다면 모든 후보가 동점이 되는 그 실패다.

  ② 정적 폴백 상수 3개(S3/S6/S7_RR)에 폴백 경로가 없었다.
     "동적 RR 미계산 시 폴백"이라 문서화됐지만 쓰이는 곳이 없었다.

  ③ **NaN이 최강 추세로 둔갑했다.** 파이썬 min/max는 NaN 비교가 전부
     False라 첫 인자를 돌려준다 → `max(0, min(1, nan))` == 1.0.
     즉 지표가 없을 때 **가장 공격적인 TP**가 잡혔다. 데이터가 없을수록
     크게 베팅하는 것은 정확히 반대로 가는 실패다.

실행:
  pytest tests/test_dynamic_rr.py -v
"""

import math
import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_indicators as ind
import atlas_spot_config as cfg
import atlas_spot_strategies as st

RR = ind.calc_dynamic_rr_ma


# ══════════════════════════════════════════════════════════════
#  ① 회귀 없음 — 기본값은 기존 동작과 완전히 동일해야 한다
# ══════════════════════════════════════════════════════════════

class TestNoRegression:
    @pytest.mark.parametrize('adx', [0, 10, 15, 20, 30, 45, 60, 100])
    @pytest.mark.parametrize('gap', [0.0, 0.5, 1.0, 2.0, 3.0, 10.0])
    def test_defaults_match_explicit_bounds(self, adx, gap):
        assert RR(adx, gap) == RR(adx, gap, 1.5, 3.0)

    def test_known_values_unchanged(self):
        """수정 전 실측값 — 하나라도 달라지면 기존 전략 동작이 바뀐 것이다."""
        assert RR(15, 0.5) == 1.5
        assert RR(45, 3.0) == 3.0
        assert RR(50, 5.0) == 3.0
        assert RR(30, 0.5) == 2.0

    def test_weights_preserve_original_split(self):
        """ADX 2/3, 갭 1/3 — 원래 (1.0, 0.5) 배분과 같아야 한다."""
        assert ind.RR_ADX_WEIGHT * 1.5 == pytest.approx(1.0)
        assert ind.RR_GAP_WEIGHT * 1.5 == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════
#  ② 범위 상수가 실제로 살아 있는가
# ══════════════════════════════════════════════════════════════

class TestBoundsAreLive:
    def test_max_actually_raises_ceiling(self):
        """이게 실패하면 config 상수가 다시 죽은 것이다."""
        assert RR(50, 5.0, 1.5, 5.0) == 5.0
        assert RR(50, 5.0, 1.5, 3.0) == 3.0

    def test_min_actually_raises_floor(self):
        assert RR(0, 0.0, 2.5, 4.0) == 2.5

    def test_output_always_within_bounds(self):
        for lo, hi in [(1.0, 2.0), (1.5, 3.0), (2.0, 6.0), (1.2, 1.3)]:
            for adx in (0, 15, 25, 45, 99):
                for gap in (0.0, 1.0, 5.0):
                    v = RR(adx, gap, lo, hi)
                    assert lo <= v <= hi, f'{v} ∉ [{lo},{hi}]'

    def test_inverted_bounds_do_not_explode(self):
        """오설정(min>max)에도 발산하지 않는다."""
        v = RR(30, 1.0, 3.0, 1.5)
        assert 1.5 <= v <= 3.0

    def test_zero_span_is_constant(self):
        assert RR(0, 0.0, 2.0, 2.0) == 2.0
        assert RR(99, 9.0, 2.0, 2.0) == 2.0

    def test_monotonic_in_adx(self):
        vals = [RR(a, 0.0, 1.5, 3.0) for a in range(0, 60, 5)]
        assert vals == sorted(vals)

    def test_monotonic_in_gap(self):
        vals = [RR(30, g, 1.5, 3.0) for g in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)]
        assert vals == sorted(vals)


# ══════════════════════════════════════════════════════════════
#  ③ NaN이 최강 추세로 둔갑하지 않는다
# ══════════════════════════════════════════════════════════════

NAN = float('nan')


class TestMissingDataIsNotStrongTrend:
    def test_python_minmax_really_does_this(self):
        """왜 가드가 필요한지 — 언어 동작 자체를 고정해 둔다."""
        assert max(0.0, min(1.0, NAN)) == 1.0

    def test_nan_adx_uses_fallback_not_max(self):
        assert RR(NAN, 0.0, 1.5, 3.0, 2.0) == 2.0
        assert RR(NAN, 0.0, 1.5, 3.0, 2.0) != RR(99, 9.0, 1.5, 3.0)

    def test_nan_gap_uses_fallback(self):
        assert RR(30.0, NAN, 1.5, 3.0, 2.0) == 2.0

    def test_inf_is_treated_as_missing(self):
        assert RR(float('inf'), 0.0, 1.5, 3.0, 2.0) == 2.0
        assert RR(float('-inf'), 0.0, 1.5, 3.0, 2.0) == 2.0

    def test_none_is_treated_as_missing(self):
        assert RR(None, 0.0, 1.5, 3.0, 2.0) == 2.0

    def test_without_fallback_uses_midpoint_not_max(self):
        """폴백을 안 주더라도 **최댓값으로 튀면 안 된다.**"""
        v = RR(NAN, NAN, 1.5, 3.0)
        assert v == pytest.approx(2.25)
        assert v < 3.0

    def test_fallback_is_clamped_to_bounds(self):
        assert RR(NAN, 0.0, 1.5, 3.0, 99.0) == 3.0
        assert RR(NAN, 0.0, 1.5, 3.0, -5.0) == 1.5

    def test_result_is_never_nan(self):
        for a in (NAN, 0, 30, float('inf')):
            for g in (NAN, 0.0, 2.0):
                assert math.isfinite(RR(a, g, 1.5, 3.0, 2.0))


# ══════════════════════════════════════════════════════════════
#  ④ 전략 배선 — 상수가 실제로 전달되는가
# ══════════════════════════════════════════════════════════════

class TestStrategyWiring:
    SRC = None

    @classmethod
    def setup_class(cls):
        cls.SRC = Path(st.__file__).read_text()

    @pytest.mark.parametrize('sid', ['S3', 'S6', 'S7'])
    def test_passes_own_bounds(self, sid):
        assert f'{sid}_RR_MIN, {sid}_RR_MAX, {sid}_RR' in self.SRC, (
            f'{sid}가 자기 RR 상수를 넘기지 않는다 — config 값이 죽는다')

    def test_no_hardcoded_range_left(self):
        """함수 안에 1.5~3.0이 다시 박히면 상수가 또 죽는다."""
        src = Path(ind.__file__).read_text()
        body = src[src.index('def calc_dynamic_rr_ma'):]
        assert 'min(3.0,' not in body and 'max(1.5,' not in body

    @pytest.mark.parametrize('sid', ['S3', 'S6', 'S7'])
    def test_config_constants_exist(self, sid):
        for suffix in ('', '_MIN', '_MAX'):
            assert hasattr(cfg, f'{sid}_RR{suffix}')

    def test_adx_missing_reaches_rr_as_nan(self, monkeypatch):
        """결측 ADX를 0.0으로 바꿔치기하면 '추세 없음'으로 **위장**돼
        폴백이 돌지 않고 조용히 rr_min이 쓰인다.

        소스 문자열이 아니라 **실제로 넘어간 인자**를 잡는다 —
        문자열 검사는 표현만 바뀌어도 통과해 버린다.
        """
        seen = {}

        def spy(adx, gap, lo=1.5, hi=3.0, fb=None):
            seen['adx'] = adx
            return 2.0

        monkeypatch.setattr(st, 'calc_dynamic_rr_ma', spy)
        df = _s6_frame(adx=float('nan'))
        sig = st.get_signal_s6(df, len(df) - 1)
        assert sig['signal'] == 1, '테스트용 진입 신호가 만들어지지 않았다'
        assert 'adx' in seen, 'calc_dynamic_rr_ma가 호출되지 않았다'
        assert math.isnan(seen['adx']), (
            f'결측 ADX가 {seen["adx"]}로 바뀌어 전달됐다 — 폴백이 무력화된다')

    def test_adx_present_is_passed_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(st, 'calc_dynamic_rr_ma',
                            lambda adx, gap, lo=1.5, hi=3.0, fb=None:
                            (seen.__setitem__('adx', adx), 2.0)[1])
        st.get_signal_s6(_s6_frame(adx=33.0), len(_s6_frame()) - 1)
        assert seen['adx'] == pytest.approx(33.0)


def _s6_frame(adx=30.0, n=80):
    """S6 진입 조건을 만족하는 최소 DataFrame."""
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        'ts':       pd.date_range('2024-01-01', periods=n, freq='1D', tz='UTC'),
        'open':     [100.0] * n,
        'high':     [101.0] * n,
        'low':      [99.0] * n,
        'close':    [100.0] * n,
        'volume':   [1000.0] * n,
        'don_high': [100.0] * n,
        'don_low':  [95.0] * n,
        'vol_ma':   [1000.0] * n,
        'atr':      [2.0] * n,
        'vwap':     [99.0] * n,
        'adx':      [adx] * n,
    })
    # 마지막 봉에서 돌파 + 거래량 스파이크
    df.loc[n - 1, 'close']  = 110.0
    df.loc[n - 1, 'volume'] = 1000.0 * (cfg.S6_VOL_MULT + 1.0)
    return df


# ══════════════════════════════════════════════════════════════
#  ⑤ 통합 — 상수를 바꾸면 전략 TP가 실제로 움직이는가
# ══════════════════════════════════════════════════════════════

class TestConstantsChangeBehaviour:
    def test_s6_rr_max_moves_tp(self, monkeypatch):
        """이 테스트가 통과해야 'WFO로 튜닝 가능한 값'이라고 말할 수 있다."""
        base = RR(50, 0.0, st.S6_RR_MIN, st.S6_RR_MAX, st.S6_RR)
        monkeypatch.setattr(st, 'S6_RR_MAX', 5.0)
        wider = RR(50, 0.0, st.S6_RR_MIN, st.S6_RR_MAX, st.S6_RR)
        assert wider > base

    def test_s7_rr_min_moves_tp(self, monkeypatch):
        base = RR(0, 0.0, st.S7_RR_MIN, st.S7_RR_MAX, st.S7_RR)
        monkeypatch.setattr(st, 'S7_RR_MIN', 2.5)
        higher = RR(0, 0.0, st.S7_RR_MIN, st.S7_RR_MAX, st.S7_RR)
        assert higher > base

    def test_static_rr_now_has_a_path(self, monkeypatch):
        """S*_RR은 '폴백'이라 문서화됐지만 경로가 없었다. 이제 있다."""
        monkeypatch.setattr(st, 'S6_RR', 2.7)
        assert RR(NAN, 0.0, st.S6_RR_MIN, st.S6_RR_MAX, st.S6_RR) == 2.7

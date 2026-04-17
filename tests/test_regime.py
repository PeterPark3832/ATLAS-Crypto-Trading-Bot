"""
ATLAS — 장세 분류기 단위 테스트
================================
atlas_regime.py의 classify_regime 및 라우팅 함수를 검증합니다.

실행:
  pytest tests/test_regime.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from atlas_regime import (
    classify_regime,
    REGIME_TRENDING_UP, REGIME_TRENDING_DOWN,
    REGIME_RANGING, REGIME_WEAK_TREND, REGIME_CRISIS,
    get_risk_scale, RegimeState,
)
from atlas_config import (
    REGIME_ADX_TREND, REGIME_ADX_WEAK, REGIME_CRISIS_ATR,
)


# ══════════════════════════════════════════════════════════════
#  classify_regime — 순수 함수 테스트
# ══════════════════════════════════════════════════════════════

class TestClassifyRegime:
    """classify_regime(adx, btc_price, ema200, atr_pct) → 레짐 문자열"""

    # ── CRISIS ────────────────────────────────────────────────
    def test_crisis_high_atr(self):
        """ATR/Price ≥ 7% → CRISIS (ADX·방향 무관)."""
        assert classify_regime(30, 50000, 45000, 0.08) == REGIME_CRISIS

    def test_crisis_atr_exactly_at_threshold(self):
        """ATR/Price = 7% → CRISIS (경계값 포함)."""
        assert classify_regime(30, 50000, 45000, REGIME_CRISIS_ATR) == REGIME_CRISIS

    def test_crisis_overrides_strong_trend(self):
        """강한 추세이더라도 ATR 위기 시 CRISIS 반환."""
        assert classify_regime(50, 60000, 40000, 0.10) == REGIME_CRISIS

    # ── TRENDING_UP ───────────────────────────────────────────
    def test_trending_up(self):
        """ADX ≥ 25, BTC > EMA200, ATR 정상 → TRENDING_UP."""
        assert classify_regime(REGIME_ADX_TREND, 50000, 40000, 0.02) == REGIME_TRENDING_UP

    def test_trending_up_exact_adx(self):
        """ADX = 25 (경계값) → TRENDING_UP (btc > ema200)."""
        assert classify_regime(25, 50000, 40000, 0.02) == REGIME_TRENDING_UP

    def test_trending_up_btc_far_above_ema(self):
        """BTC가 EMA200보다 훨씬 위 → TRENDING_UP."""
        assert classify_regime(40, 100000, 50000, 0.01) == REGIME_TRENDING_UP

    # ── TRENDING_DOWN ─────────────────────────────────────────
    def test_trending_down(self):
        """ADX ≥ 25, BTC < EMA200, ATR 정상 → TRENDING_DOWN."""
        assert classify_regime(REGIME_ADX_TREND, 40000, 50000, 0.02) == REGIME_TRENDING_DOWN

    def test_trending_down_exact_adx(self):
        """ADX = 25 경계값, btc < ema200 → TRENDING_DOWN."""
        assert classify_regime(25, 30000, 40000, 0.02) == REGIME_TRENDING_DOWN

    # ── RANGING ───────────────────────────────────────────────
    def test_ranging_low_adx(self):
        """ADX < 20 → RANGING."""
        assert classify_regime(REGIME_ADX_WEAK - 1, 50000, 50000, 0.02) == REGIME_RANGING

    def test_ranging_adx_zero(self):
        """ADX = 0 → RANGING."""
        assert classify_regime(0, 50000, 50000, 0.02) == REGIME_RANGING

    def test_ranging_regardless_of_price_direction(self):
        """ADX < 20이면 BTC vs EMA200 방향 무관하게 RANGING."""
        assert classify_regime(10, 60000, 40000, 0.02) == REGIME_RANGING
        assert classify_regime(10, 40000, 60000, 0.02) == REGIME_RANGING

    # ── WEAK_TREND ────────────────────────────────────────────
    def test_weak_trend_between_adx_thresholds(self):
        """20 ≤ ADX < 25 → WEAK_TREND."""
        assert classify_regime(22, 50000, 45000, 0.02) == REGIME_WEAK_TREND

    def test_weak_trend_lower_bound(self):
        """ADX = 20 → WEAK_TREND."""
        assert classify_regime(REGIME_ADX_WEAK, 50000, 45000, 0.02) == REGIME_WEAK_TREND

    def test_weak_trend_upper_bound_exclusive(self):
        """ADX = 25 → TRENDING_UP (WEAK_TREND 아님)."""
        result = classify_regime(25, 50000, 45000, 0.02)
        assert result != REGIME_WEAK_TREND
        assert result == REGIME_TRENDING_UP

    # ── 경계값 추가 검증 ───────────────────────────────────────
    def test_btc_exactly_at_ema200(self):
        """BTC = EMA200 → TRENDING_UP (btc_price >= ema200)."""
        assert classify_regime(30, 50000, 50000, 0.02) == REGIME_TRENDING_UP

    def test_atr_just_below_crisis(self):
        """ATR/Price = 6.99% → CRISIS 아님."""
        result = classify_regime(30, 50000, 45000, 0.0699)
        assert result != REGIME_CRISIS


# ══════════════════════════════════════════════════════════════
#  RegimeState
# ══════════════════════════════════════════════════════════════

class TestRegimeState:
    def test_default_regime_unknown(self):
        state = RegimeState()
        assert state.regime == 'UNKNOWN'

    def test_summary_contains_regime(self):
        state = RegimeState(
            regime=REGIME_TRENDING_UP,
            adx=30.0, btc_price=50000, ema200=45000,
            atr_pct=0.02, updated_at=0.0)
        summary = state.summary()
        assert REGIME_TRENDING_UP in summary
        assert 'ADX' in summary

    def test_all_regimes_have_emoji_in_summary(self):
        """모든 레짐에 대해 summary()가 정상 작동."""
        for regime in (REGIME_TRENDING_UP, REGIME_TRENDING_DOWN,
                       REGIME_RANGING, REGIME_WEAK_TREND, REGIME_CRISIS):
            state = RegimeState(regime=regime, adx=20.0,
                                btc_price=50000, ema200=50000,
                                atr_pct=0.02, updated_at=0.0)
            summary = state.summary()
            assert regime in summary


# ══════════════════════════════════════════════════════════════
#  전략 라우터 (get_risk_scale 등은 캐시 의존 — 간접 테스트)
# ══════════════════════════════════════════════════════════════

class TestRouterFunctions:
    """
    is_phoenix_long_allowed 등은 _regime_cache에 의존하므로
    여기서는 get_risk_scale 반환값과 classify_regime 조합만 검증.
    """

    def test_risk_scale_returns_float(self):
        """get_risk_scale()은 float 반환 (캐시 UNKNOWN일 때 1.0)."""
        scale = get_risk_scale()
        assert isinstance(scale, float)
        assert scale > 0

    @pytest.mark.parametrize('adx,btc,ema,atr_pct,expected_regime', [
        (30,  50000, 45000, 0.02, REGIME_TRENDING_UP),
        (30,  45000, 50000, 0.02, REGIME_TRENDING_DOWN),
        (10,  50000, 45000, 0.02, REGIME_RANGING),
        (22,  50000, 45000, 0.02, REGIME_WEAK_TREND),
        (30,  50000, 45000, 0.08, REGIME_CRISIS),
    ])
    def test_classify_parametrized(self, adx, btc, ema, atr_pct, expected_regime):
        assert classify_regime(adx, btc, ema, atr_pct) == expected_regime

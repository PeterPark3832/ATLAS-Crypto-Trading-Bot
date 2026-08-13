"""
ATLAS — 장세 분류기 보강 단위 테스트
================================
test_regime.py가 다루지 않는 classify_regime의 ADX 기울기 다운그레이드/
MICRO_RANGING 분기, update_regime(거래소 모킹), is_crisis,
캐시 설정된 get_risk_scale을 검증합니다.

실행:
  pytest tests/test_regime_extra.py -v
"""



import pytest

import atlas_regime as rg
from atlas_regime import (
    classify_regime, update_regime, is_crisis, get_risk_scale,
    RegimeState,
    REGIME_TRENDING_UP, REGIME_TRENDING_DOWN, REGIME_WEAK_TREND,
    REGIME_MICRO_RANGING, REGIME_CRISIS, REGIME_RANGING,
)


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr(rg, '_regime_cache', RegimeState())


# ══════════════════════════════════════════════════════════════
#  classify_regime — ADX 기울기 다운그레이드 / MICRO_RANGING
# ══════════════════════════════════════════════════════════════

class TestClassifyRegimeAdvanced:
    def test_falling_adx_slope_downgrades_to_weak_trend(self):
        """ADX>=25이지만 3봉 연속 하락(slope<0) + ADX<30 → WEAK_TREND로 다운그레이드."""
        result = classify_regime(28, 50000, 45000, 0.02, adx_slope=-2.0)
        assert result == REGIME_WEAK_TREND

    def test_falling_slope_but_adx_30_or_above_keeps_trend(self):
        """ADX>=30이면 기울기 하락 중이어도 다운그레이드하지 않음."""
        result = classify_regime(30, 50000, 45000, 0.02, adx_slope=-2.0)
        assert result == REGIME_TRENDING_UP

    def test_rising_slope_keeps_trend(self):
        result = classify_regime(28, 50000, 45000, 0.02, adx_slope=1.5)
        assert result == REGIME_TRENDING_UP

    def test_uptrend_with_weak_4h_adx_is_micro_ranging(self):
        """1D TRENDING_UP + 4H ADX<20 → MICRO_RANGING."""
        result = classify_regime(30, 50000, 45000, 0.02, adx_4h=15.0)
        assert result == REGIME_MICRO_RANGING

    def test_uptrend_with_strong_4h_adx_stays_trending(self):
        result = classify_regime(30, 50000, 45000, 0.02, adx_4h=25.0)
        assert result == REGIME_TRENDING_UP

    def test_downtrend_ignores_4h_micro_ranging_check(self):
        """MICRO_RANGING은 TRENDING_UP에만 적용 — DOWN에는 영향 없음."""
        result = classify_regime(30, 40000, 50000, 0.02, adx_4h=10.0)
        assert result == REGIME_TRENDING_DOWN

    def test_adx_4h_zero_skips_micro_ranging_check(self):
        """adx_4h=0.0 (조회 실패 기본값)이면 MICRO_RANGING 판정을 건너뜀."""
        result = classify_regime(30, 50000, 45000, 0.02, adx_4h=0.0)
        assert result == REGIME_TRENDING_UP


# ══════════════════════════════════════════════════════════════
#  is_crisis / get_risk_scale — 캐시 의존 함수
# ══════════════════════════════════════════════════════════════

class TestCacheDependentRouting:
    def test_is_crisis_false_by_default(self):
        assert is_crisis() is False

    def test_is_crisis_true_when_cached(self, monkeypatch):
        monkeypatch.setattr(rg, '_regime_cache', RegimeState(regime=REGIME_CRISIS))
        assert is_crisis() is True

    def test_risk_scale_micro_ranging(self, monkeypatch):
        monkeypatch.setattr(rg, '_regime_cache', RegimeState(regime=REGIME_MICRO_RANGING))
        assert get_risk_scale() == 0.50

    def test_risk_scale_weak_trend(self, monkeypatch):
        monkeypatch.setattr(rg, '_regime_cache', RegimeState(regime=REGIME_WEAK_TREND))
        assert get_risk_scale() == 0.70

    def test_risk_scale_normal_regime(self, monkeypatch):
        monkeypatch.setattr(rg, '_regime_cache', RegimeState(regime=REGIME_TRENDING_UP))
        assert get_risk_scale() == 1.00


# ══════════════════════════════════════════════════════════════
#  update_regime — 거래소 모킹
# ══════════════════════════════════════════════════════════════

def _make_ohlcv(n: int, start: float = 100.0, drift: float = 0.0,
                 interval_ms: int = 86_400_000) -> list:
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    result = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99
        result.append([i * interval_ms, o, h, l, c, 1000.0])
    return result


class _FakeRegimeExchange:
    def __init__(self, ohlcv_1d, ohlcv_4h=None, raise_4h=False):
        self._ohlcv_1d = ohlcv_1d
        self._ohlcv_4h = ohlcv_4h or []
        self._raise_4h = raise_4h

    def fetch_ohlcv(self, sym, timeframe, limit=None):
        if timeframe == '4h':
            if self._raise_4h:
                raise RuntimeError('4h fetch failed')
            return self._ohlcv_4h
        return self._ohlcv_1d


class TestUpdateRegime:
    def test_too_few_candles_returns_cached(self, monkeypatch):
        cached = RegimeState(regime=REGIME_RANGING)
        monkeypatch.setattr(rg, '_regime_cache', cached)
        ex = _FakeRegimeExchange(ohlcv_1d=_make_ohlcv(10))
        result = update_regime(ex)
        assert result is cached

    def test_uptrend_data_classified_and_cached(self, monkeypatch):
        ex = _FakeRegimeExchange(ohlcv_1d=_make_ohlcv(90, start=100.0, drift=0.02))
        result = update_regime(ex)
        assert result.regime in (REGIME_TRENDING_UP, REGIME_MICRO_RANGING)
        assert result.btc_price > 0
        assert result.ema200 > 0
        cached = rg.get_cached_regime()
        assert cached.regime == result.regime

    def test_4h_fetch_failure_does_not_abort_update(self, monkeypatch):
        ex = _FakeRegimeExchange(ohlcv_1d=_make_ohlcv(90, start=100.0, drift=0.02),
                                  raise_4h=True)
        result = update_regime(ex)
        assert result.adx_4h == 0.0
        assert result.regime != REGIME_MICRO_RANGING

    def test_exchange_exception_returns_cached(self, monkeypatch):
        cached = RegimeState(regime=REGIME_TRENDING_DOWN)
        monkeypatch.setattr(rg, '_regime_cache', cached)

        class _RaisingExchange:
            def fetch_ohlcv(self, sym, timeframe, limit=None):
                raise RuntimeError('network error')

        result = update_regime(_RaisingExchange())
        assert result is cached

    def test_regime_loop_exits_on_stop_event(self, monkeypatch):
        """regime_loop이 stop_event로 종료 가능해야 함 (과거: while True로 종료 불가 +
        main이 stop_event를 ex 자리에 넘겨 조용히 버려지던 배선 실수)."""
        import threading

        calls = []
        monkeypatch.setattr(rg, '_make_regime_ex', lambda: object())
        monkeypatch.setattr(rg, 'update_regime',
                             lambda ex, cc=None: calls.append(1) or RegimeState())

        class _SetOnWaitEvent(threading.Event):
            def wait(self, timeout=None):
                self.set()
                return True

        ev = _SetOnWaitEvent()
        rg.regime_loop(ev)          # 1회 실행 후 종료해야 함 (행 없음)
        assert calls == [1]

    def test_regime_loop_preset_event_skips_body(self, monkeypatch):
        import threading

        calls = []
        monkeypatch.setattr(rg, '_make_regime_ex', lambda: object())
        monkeypatch.setattr(rg, 'update_regime',
                             lambda ex, cc=None: calls.append(1) or RegimeState())
        ev = threading.Event()
        ev.set()
        rg.regime_loop(ev)
        assert calls == []

    def test_uses_candle_cache_when_provided(self, monkeypatch):
        calls = []

        class _FakeCandleCache:
            def get(self, ex, sym, timeframe, limit):
                calls.append((sym, timeframe, limit))
                return _make_ohlcv(90, start=100.0, drift=0.01)

        ex = _FakeRegimeExchange(ohlcv_1d=[])  # candle_cache 사용 시 호출되지 않아야 함
        result = update_regime(ex, candle_cache=_FakeCandleCache())
        assert calls and calls[0][1] == '1d'
        assert result.btc_price > 0

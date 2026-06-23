"""
ATLAS — 지표 모듈 단위 테스트
================================
atlas_indicators.py의 공유 유틸리티 함수를 검증합니다.

실행:
  pytest tests/ -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from atlas_indicators import (
    _calc_rsi, _calc_atr, _ohlcv_to_df,
    calc_adx, calc_dynamic_rr_ma,
)


# ══════════════════════════════════════════════════════════════
#  헬퍼: 합성 OHLCV 생성
# ══════════════════════════════════════════════════════════════

def _make_ohlcv(n: int, start: float = 100.0,
                drift: float = 0.001, volatility: float = 0.015,
                seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.normal(0, volatility)))
    closes = [max(c, 0.01) for c in closes]

    result = []
    interval_ms = 4 * 3600 * 1000
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        noise_h = abs(rng.normal(0, volatility * 0.5))
        noise_l = abs(rng.normal(0, volatility * 0.5))
        h = max(o, c) * (1 + noise_h)
        l = min(o, c) * (1 - noise_l)
        v = float(rng.uniform(500, 5000))
        result.append([i * interval_ms, float(o), float(h), float(l), float(c), v])
    return result


def _make_ohlcv_1d(n: int, start: float = 100.0,
                   drift: float = 0.001, seed: int = 99) -> list:
    data = _make_ohlcv(n, start=start, drift=drift, volatility=0.02, seed=seed)
    day_ms = 24 * 3600 * 1000
    for i, bar in enumerate(data):
        bar[0] = i * day_ms
    return data


# ══════════════════════════════════════════════════════════════
#  _ohlcv_to_df
# ══════════════════════════════════════════════════════════════

class TestOhlcvToDf:
    def test_columns(self):
        ohlcv = _make_ohlcv(10)
        df = _ohlcv_to_df(ohlcv)
        assert set(df.columns) == {'ts', 'open', 'high', 'low', 'close', 'volume'}

    def test_length(self):
        ohlcv = _make_ohlcv(50)
        df = _ohlcv_to_df(ohlcv)
        assert len(df) == 50

    def test_dtypes_float(self):
        ohlcv = _make_ohlcv(10)
        df = _ohlcv_to_df(ohlcv)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert df[col].dtype == float

    def test_high_ge_low(self):
        ohlcv = _make_ohlcv(100)
        df = _ohlcv_to_df(ohlcv)
        assert (df['high'] >= df['low']).all()


# ══════════════════════════════════════════════════════════════
#  _calc_rsi
# ══════════════════════════════════════════════════════════════

class TestCalcRsi:
    def test_range_0_100(self):
        """RSI는 항상 0~100 범위."""
        close = pd.Series([float(x) for x in range(1, 50)])
        rsi = _calc_rsi(close, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rising_series_high_rsi(self):
        """상승 기조 → RSI > 60."""
        closes = []
        price = 100.0
        for i in range(60):
            if i % 5 == 4:
                price -= 0.5
            else:
                price += 2.0
            closes.append(price)
        rsi = _calc_rsi(pd.Series(closes), 14)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert float(valid.iloc[-1]) > 60

    def test_falling_series_low_rsi(self):
        """꾸준히 내리는 시리즈 → RSI < 30."""
        close = pd.Series([100.0 - i * 2 for i in range(50)])
        rsi = _calc_rsi(close, 14)
        assert float(rsi.iloc[-1]) < 30

    def test_flat_series(self):
        """변화 없는 시리즈 → RSI NaN or 50."""
        close = pd.Series([50.0] * 30)
        rsi = _calc_rsi(close, 14)
        assert pd.isna(rsi.iloc[-1]) or abs(float(rsi.iloc[-1]) - 50) < 1

    def test_warmup_nan(self):
        """워밍업 기간 동안 NaN."""
        close = pd.Series([float(x) for x in range(1, 30)])
        rsi = _calc_rsi(close, 14)
        assert rsi.iloc[:14].isna().all()

    def test_pure_uptrend_no_nan_after_warmup(self):
        """구간 내 하락일이 전혀 없는 순수 상승장 → RSI=100 (NaN 아님).
        loss=0일 때 gain/0=NaN이 되어 RSI 전체가 비어버리는 회귀 방지."""
        close = pd.Series([100.0 + i for i in range(40)])  # 매일 상승, 하락 없음
        rsi = _calc_rsi(close, 14)
        valid = rsi.iloc[14:]
        assert not valid.isna().any()
        assert (valid == 100.0).all()


# ══════════════════════════════════════════════════════════════
#  _calc_atr
# ══════════════════════════════════════════════════════════════

class TestCalcAtr:
    def test_positive(self):
        """ATR은 항상 양수."""
        ohlcv = _make_ohlcv(50)
        df    = _ohlcv_to_df(ohlcv)
        atr   = _calc_atr(df, 14)
        assert (atr.dropna() > 0).all()

    def test_high_volatility_bigger_atr(self):
        """변동성 높은 데이터 → ATR 더 큼."""
        lo_vol = _make_ohlcv(60, volatility=0.005)
        hi_vol = _make_ohlcv(60, volatility=0.05)
        df_lo = _ohlcv_to_df(lo_vol)
        df_hi = _ohlcv_to_df(hi_vol)
        atr_lo = _calc_atr(df_lo, 14).iloc[-1]
        atr_hi = _calc_atr(df_hi, 14).iloc[-1]
        assert atr_hi > atr_lo

    def test_warmup_nan(self):
        """rolling(14).mean() → 인덱스 0~12는 NaN."""
        ohlcv = _make_ohlcv(30)
        df    = _ohlcv_to_df(ohlcv)
        atr   = _calc_atr(df, 14)
        assert atr.iloc[:13].isna().all()
        assert not pd.isna(atr.iloc[13])


# ══════════════════════════════════════════════════════════════
#  calc_adx
# ══════════════════════════════════════════════════════════════

class TestCalcAdx:
    def test_returns_float(self):
        ohlcv = _make_ohlcv(100)
        adx   = calc_adx(ohlcv, 14)
        assert isinstance(adx, float)

    def test_range_0_100(self):
        ohlcv = _make_ohlcv(100)
        adx   = calc_adx(ohlcv, 14)
        assert 0.0 <= adx <= 100.0

    def test_insufficient_data_returns_zero(self):
        """데이터 부족(< period*3) → 0.0."""
        ohlcv = _make_ohlcv(10)
        adx   = calc_adx(ohlcv, 14)
        assert adx == 0.0

    def test_consistent_result(self):
        """동일 입력 → 동일 출력 (결정론적)."""
        ohlcv = _make_ohlcv(100)
        assert calc_adx(ohlcv, 14) == calc_adx(ohlcv, 14)


# ══════════════════════════════════════════════════════════════
#  calc_dynamic_rr_ma
# ══════════════════════════════════════════════════════════════

class TestCalcDynamicRrMa:
    def test_range(self):
        """항상 1.5~3.0 범위."""
        for adx in [0, 10, 25, 45, 60]:
            for gap in [0.0, 1.0, 2.0, 5.0]:
                rr = calc_dynamic_rr_ma(adx, gap)
                assert 1.5 <= rr <= 3.0

    def test_strong_trend_higher_rr(self):
        """강한 추세(높은 ADX + 큰 EMA 갭) → 더 높은 RR."""
        rr_weak   = calc_dynamic_rr_ma(15, 0.3)
        rr_strong = calc_dynamic_rr_ma(45, 3.0)
        assert rr_strong > rr_weak

    def test_boundary_min(self):
        """ADX=0, gap=0 → 최솟값 1.5."""
        assert calc_dynamic_rr_ma(0, 0) == 1.5

    def test_boundary_max(self):
        """ADX=99, gap=99 → 최댓값 3.0."""
        assert calc_dynamic_rr_ma(99, 99) == 3.0

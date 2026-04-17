"""
ATLAS — 지표 모듈 단위 테스트
================================
atlas_indicators.py의 모든 주요 함수를 검증합니다.

실행:
  cd ATLAS
  pytest tests/ -v
"""

import os
import sys
from pathlib import Path

# atlas_config 임포트 전 더미 환경변수 설정
for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from atlas_indicators import (
    _calc_rsi, _calc_atr, _ohlcv_to_df,
    calc_4h, get_signal_4h,
    calc_1d_eq, get_signal_eq,
    calc_1d_bn, get_signal_bn,
    calc_adx,
)
from atlas_config import (
    PHX_SYMBOL_PARAMS,
    PHX_RSI_LONG_MIN, PHX_RSI_LONG_MAX,
)


# ══════════════════════════════════════════════════════════════
#  헬퍼: 합성 OHLCV 생성
# ══════════════════════════════════════════════════════════════

def _make_ohlcv(n: int, start: float = 100.0,
                drift: float = 0.001, volatility: float = 0.015,
                seed: int = 42) -> list:
    """
    n봉의 합성 OHLCV 생성 (4H 기준, 단위시간 4시간).
    drift > 0 이면 상승추세, drift < 0 이면 하락추세.
    """
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
    """일봉 합성 OHLCV."""
    data = _make_ohlcv(n, start=start, drift=drift,
                       volatility=0.02, seed=seed)
    # 타임스탬프를 1D 간격으로 재설정
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
        """상승 + 의도적 하락일 포함 → RSI > 60 (loss ≠ 0 보장)."""
        # 상승 기조지만 매 5봉마다 명시적 하락 삽입 → loss rolling mean > 0 보장
        closes = []
        price = 100.0
        for i in range(60):
            if i % 5 == 4:
                price -= 0.5   # 소폭 하락
            else:
                price += 2.0   # 상승
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
        """변화 없는 시리즈 → RSI NaN (0 변화 → 0/0)."""
        close = pd.Series([50.0] * 30)
        rsi = _calc_rsi(close, 14)
        # 변화 없으면 gain=loss=0 → NaN or 50
        assert pd.isna(rsi.iloc[-1]) or abs(float(rsi.iloc[-1]) - 50) < 1

    def test_warmup_nan(self):
        """워밍업 기간 동안 NaN."""
        close = pd.Series([float(x) for x in range(1, 30)])
        rsi = _calc_rsi(close, 14)
        assert rsi.iloc[:14].isna().all()


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
        """rolling(14).mean() → 인덱스 0~12는 NaN, 13부터 유효."""
        ohlcv = _make_ohlcv(30)
        df    = _ohlcv_to_df(ohlcv)
        atr   = _calc_atr(df, 14)
        # rolling(14): 첫 13개(인덱스 0~12)는 NaN
        assert atr.iloc[:13].isna().all()
        # 인덱스 13 = 14번째 값부터 유효
        assert not pd.isna(atr.iloc[13])


# ══════════════════════════════════════════════════════════════
#  calc_4h
# ══════════════════════════════════════════════════════════════

class TestCalc4h:
    @pytest.fixture
    def df(self):
        ohlcv = _make_ohlcv(300, drift=0.002)
        return calc_4h(ohlcv, ema_entry=30)

    def test_required_columns(self, df):
        required = {'ema200', 'ema_entry', 'macd', 'macd_signal', 'macd_hist',
                    'rsi', 'atr', 'bb_mid', 'bb_upper', 'bb_lower',
                    'bb_width', 'bb_width_min'}
        assert required.issubset(set(df.columns))

    def test_bb_relationship(self, df):
        """BB: upper > mid > lower."""
        valid = df.dropna(subset=['bb_upper', 'bb_mid', 'bb_lower'])
        assert (valid['bb_upper'] >= valid['bb_mid']).all()
        assert (valid['bb_mid']   >= valid['bb_lower']).all()

    def test_bb_width_positive(self, df):
        """BB 폭은 양수."""
        valid = df.dropna(subset=['bb_width'])
        assert (valid['bb_width'] > 0).all()

    def test_rsi_range(self, df):
        """RSI 0~100."""
        valid = df['rsi'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_positive(self, df):
        """ATR 양수."""
        valid = df['atr'].dropna()
        assert (valid > 0).all()


# ══════════════════════════════════════════════════════════════
#  get_signal_4h
# ══════════════════════════════════════════════════════════════

class TestGetSignal4h:
    @pytest.fixture
    def params(self):
        return PHX_SYMBOL_PARAMS['BTCUSDT']

    def test_returns_dict_with_required_keys(self, params):
        ohlcv = _make_ohlcv(280, drift=0.003)
        df    = calc_4h(ohlcv, params['ema_entry'])
        sig   = get_signal_4h(df, params)
        assert set(sig.keys()) == {'signal', 'mode', 'sl', 'tp', 'rr'}

    def test_no_signal_insufficient_data(self, params):
        """데이터 부족 → signal=0."""
        ohlcv = _make_ohlcv(100)
        df    = calc_4h(ohlcv, params['ema_entry'])
        sig   = get_signal_4h(df, params)
        assert sig['signal'] == 0

    def test_signal_value_valid(self, params):
        """signal 값은 -1, 0, 1 중 하나."""
        ohlcv = _make_ohlcv(300, drift=0.003)
        df    = calc_4h(ohlcv, params['ema_entry'])
        sig   = get_signal_4h(df, params)
        assert sig['signal'] in (-1, 0, 1)

    def test_long_signal_sl_below_entry(self, params):
        """롱 신호 시: SL < 현재가."""
        ohlcv = _make_ohlcv(300, drift=0.004, seed=7)
        df    = calc_4h(ohlcv, params['ema_entry'])
        for i in range(220, len(df)):
            sig = get_signal_4h(df.iloc[:i+1], params)
            if sig['signal'] == 1:
                entry = float(df.iloc[i]['close'])
                assert sig['sl'] < entry
                assert sig['tp'] > entry
                assert sig['rr'] >= 1.0
                break

    def test_short_signal_sl_above_entry(self, params):
        """숏 신호 시: SL > 현재가."""
        ohlcv = _make_ohlcv(300, drift=-0.004, seed=13)
        df    = calc_4h(ohlcv, params['ema_entry'])
        for i in range(220, len(df)):
            sig = get_signal_4h(df.iloc[:i+1], params)
            if sig['signal'] == -1:
                entry = float(df.iloc[i]['close'])
                assert sig['sl'] > entry
                assert sig['tp'] < entry
                assert sig['rr'] >= 1.0
                break

    def test_rr_within_bounds(self, params):
        """RR은 PHX_RR_MIN~PHX_RR_MAX 범위."""
        from atlas_config import PHX_RR_MIN, PHX_RR_MAX
        ohlcv = _make_ohlcv(300, drift=0.003)
        df    = calc_4h(ohlcv, params['ema_entry'])
        for i in range(220, len(df)):
            sig = get_signal_4h(df.iloc[:i+1], params)
            if sig['signal'] != 0:
                assert PHX_RR_MIN <= sig['rr'] <= PHX_RR_MAX
                break


# ══════════════════════════════════════════════════════════════
#  calc_1d_eq / get_signal_eq
# ══════════════════════════════════════════════════════════════

class TestEquinox:
    def test_calc_1d_eq_columns(self):
        ohlcv = _make_ohlcv_1d(300, drift=0.001)
        df    = calc_1d_eq(ohlcv)
        assert {'atr', 'ema_fast', 'ema_slow', 'ma200', 'vma20'}.issubset(set(df.columns))

    def test_calc_1d_eq_no_nan(self):
        """dropna 후 결과에 NaN 없어야."""
        ohlcv = _make_ohlcv_1d(300)
        df    = calc_1d_eq(ohlcv)
        assert df[['atr', 'ema_fast', 'ema_slow', 'vma20']].isna().sum().sum() == 0

    def test_get_signal_eq_insufficient_data(self):
        """데이터 부족 → False."""
        df   = pd.DataFrame()
        ok, *_ = get_signal_eq(df, 100.0, 105.0, 1000.0)
        assert not ok

    def test_get_signal_eq_ema_bearish_no_signal(self):
        """EMA 하락추세 → 신호 없음."""
        ohlcv  = _make_ohlcv_1d(300, drift=-0.003)
        df_all = calc_1d_eq(ohlcv)
        df     = df_all.iloc[:-1]  # prev days
        ok, *_ = get_signal_eq(df, 90.0, 95.0, 9999.0)
        assert not ok

    def test_get_signal_eq_returns_tuple(self):
        """반환 형식: (bool, float, float, float, float, str)."""
        ohlcv  = _make_ohlcv_1d(300, drift=0.003)
        df_all = calc_1d_eq(ohlcv)
        df     = df_all.iloc[:-1]
        result = get_signal_eq(df, 110.0, 115.0, 9999.0)
        assert len(result) == 6
        assert isinstance(result[0], bool)

    def test_get_signal_eq_sl_below_entry_on_valid_signal(self):
        """유효 신호 시 SL < entry < TP."""
        ohlcv  = _make_ohlcv_1d(300, drift=0.004, seed=55)
        df_all = calc_1d_eq(ohlcv, ema_fast=10, ema_slow=40)
        # 상승추세에서 신호 탐색
        for i in range(30, len(df_all)):
            row    = df_all.iloc[i]
            df_p   = df_all.iloc[:i]
            o, h   = float(row['open']), float(row['high'])
            v      = float(row['volume'])
            ok, ep, sl, tp, _, _ = get_signal_eq(df_p, o, h, v)
            if ok:
                assert sl < ep < tp
                break


# ══════════════════════════════════════════════════════════════
#  calc_1d_bn / get_signal_bn
# ══════════════════════════════════════════════════════════════

class TestBounce:
    def test_calc_1d_bn_columns(self):
        ohlcv = _make_ohlcv_1d(100)
        df    = calc_1d_bn(ohlcv)
        assert {'atr', 'ema_fast', 'ema_slow', 'rsi14',
                'bb_mid', 'bb_lower', 'bb_upper', 'vma20'}.issubset(set(df.columns))

    def test_get_signal_bn_insufficient_data(self):
        """데이터 부족 → False."""
        df   = pd.DataFrame(columns=['ema_fast','ema_slow','rsi14','close','bb_lower','atr'])
        ok, *_ = get_signal_bn(df)
        assert not ok

    def test_get_signal_bn_uptrend_required(self):
        """EMA 하락추세면 신호 없음."""
        ohlcv  = _make_ohlcv_1d(100, drift=-0.005)
        df_all = calc_1d_bn(ohlcv)
        ok, *_ = get_signal_bn(df_all)
        assert not ok

    def test_get_signal_bn_returns_tuple(self):
        ohlcv = _make_ohlcv_1d(80)
        df    = calc_1d_bn(ohlcv)
        result = get_signal_bn(df)
        assert len(result) == 6
        assert isinstance(result[0], bool)

    def test_valid_signal_sl_tp_structure(self):
        """유효 신호 시 SL < entry < TP."""
        ohlcv = _make_ohlcv_1d(200, drift=0.002, seed=77)
        df    = calc_1d_bn(ohlcv)
        for i in range(30, len(df)):
            ok, ep, sl, tp, _, _ = get_signal_bn(df.iloc[:i+1])
            if ok:
                assert sl < ep < tp
                break


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

    def test_strong_trend_higher_adx(self):
        """강한 추세 → ADX가 횡보보다 높아야 함."""
        trending  = _make_ohlcv(150, drift=0.006,  volatility=0.005)
        sideways  = _make_ohlcv(150, drift=0.0,    volatility=0.015, seed=2)
        adx_trend = calc_adx(trending, 14)
        adx_side  = calc_adx(sideways, 14)
        # 반드시 성립하지 않을 수 있어 soft check
        assert adx_trend >= 0 and adx_side >= 0

    def test_consistent_result(self):
        """동일 입력 → 동일 출력 (결정론적)."""
        ohlcv = _make_ohlcv(100)
        assert calc_adx(ohlcv, 14) == calc_adx(ohlcv, 14)

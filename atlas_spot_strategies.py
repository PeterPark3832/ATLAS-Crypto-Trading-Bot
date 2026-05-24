"""
ATLAS Spot — 전략 구현 모듈
============================
7개 역사적 전략을 바-바이-바(bar-by-bar) 시뮬레이션에 적합한 형태로 구현.
모든 함수는 선행 편향이 없도록 현재 봉 이전 데이터만 사용.

전략 목록:
  S1: Buy & Hold (벤치마크)
  S2: SMA Golden Cross (1D)
  S3: EMA Trend Follow (4H)   ← v2 Module A 롱 전용 변환
  S4: RSI Mean Reversion (1D) ← v2 Module B 롱 전용 변환
  S5: Bollinger Band Bounce (1D)
  S6: Donchian Breakout (1D)
  S7: MACD Momentum (4H)

반환 형식:
  각 calc_sN() → pd.DataFrame (지표 컬럼 추가)
  각 get_signal_sN(df, i) → dict:
    {'signal': 1/0, 'sl': float, 'tp': float, 'rr': float,
     'exit_type': str, 'max_hold': int}
  signal=1: 매수, 0: 신호없음
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'STRATEGY')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_indicators import _ohlcv_to_df, _calc_atr, _calc_rsi
from atlas_spot_config import (
    S2_SMA_FAST, S2_SMA_SLOW, S2_ATR_PERIOD, S2_ATR_SL,
    S3_EMA_FAST, S3_EMA_SLOW, S3_EMA_TREND, S3_ADX_PERIOD, S3_ADX_MIN,
    S3_ATR_PERIOD, S3_ATR_SL, S3_RR, S3_COOLDOWN,
    S4_RSI_PERIOD, S4_RSI_ENTRY, S4_BB_PERIOD, S4_BB_SIGMA,
    S4_ATR_PERIOD, S4_ATR_SL, S4_MAX_HOLD,
    S5_BB_PERIOD, S5_BB_SIGMA, S5_RSI_CONFIRM, S5_ATR_PERIOD, S5_ATR_SL,
    S6_ENTRY_PERIOD, S6_EXIT_PERIOD, S6_VOL_MA, S6_VOL_MULT,
    S6_ATR_PERIOD, S6_ATR_SL, S6_RR,
    S7_MACD_FAST, S7_MACD_SLOW, S7_MACD_SIG, S7_EMA_TREND,
    S7_ATR_PERIOD, S7_ATR_SL, S7_RR,
)

# 데이터 부족 시 반환하는 신호 없음 dict
_NO_SIGNAL = {'signal': 0, 'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
              'exit_type': 'sl_tp', 'max_hold': 0}


def _no_sig() -> dict:
    return dict(_NO_SIGNAL)


# ══════════════════════════════════════════════════════════════
#  공통 헬퍼
# ══════════════════════════════════════════════════════════════

def _calc_bb(close: pd.Series, period: int, sigma: float):
    """BB_mid / BB_upper / BB_lower 반환."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + sigma * std, mid - sigma * std


def _calc_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ADX를 Series로 계산 (bar-by-bar 백테스트용).
    Returns: pd.Series (길이 = len(df), 초기값 NaN)
    """
    h = df['high']
    l = df['low']
    c = df['close']

    up   = h.diff()
    down = -l.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    tr_s   = tr.ewm(alpha=1/period, adjust=False).mean()
    dmp_s  = dm_p.ewm(alpha=1/period, adjust=False).mean()
    dmm_s  = dm_m.ewm(alpha=1/period, adjust=False).mean()

    di_p = 100 * dmp_s / tr_s.replace(0, np.nan)
    di_m = 100 * dmm_s / tr_s.replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx  = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ══════════════════════════════════════════════════════════════
#  S1: Buy & Hold (벤치마크)
# ══════════════════════════════════════════════════════════════

def calc_s1(ohlcv: list) -> pd.DataFrame:
    """Buy & Hold 벤치마크 — 지표 불필요, 기본 OHLCV DataFrame만 반환."""
    return _ohlcv_to_df(ohlcv)


def get_signal_s1(df: pd.DataFrame, i: int) -> dict:
    """
    첫 번째 봉에서 매수하고 끝까지 보유.
    백테스트 엔진에서 특별 처리 (SL/TP 없음).
    """
    if i == 0:
        return {'signal': 1, 'sl': 0.0, 'tp': 0.0, 'rr': 0.0,
                'exit_type': 'hold', 'max_hold': 0}
    return _no_sig()


# ══════════════════════════════════════════════════════════════
#  S2: SMA Golden Cross (1D)
# ══════════════════════════════════════════════════════════════

def calc_s2(ohlcv: list) -> pd.DataFrame:
    """SMA Golden Cross 지표 계산."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['sma_fast'] = c.rolling(S2_SMA_FAST).mean()
    df['sma_slow'] = c.rolling(S2_SMA_SLOW).mean()
    df['atr']      = _calc_atr(df, S2_ATR_PERIOD)
    return df


def get_signal_s2(df: pd.DataFrame, i: int) -> dict:
    """
    SMA Golden Cross 매수 신호.
    진입: sma_fast가 sma_slow를 상향돌파 (이전봉 ≤ → 현재봉 >)
    청산: Death Cross (sma_fast < sma_slow) 또는 SL
    """
    if i < S2_SMA_SLOW + 5:
        return _no_sig()
    prev = df.iloc[i - 1]
    cur  = df.iloc[i]
    if any(pd.isna([prev['sma_fast'], prev['sma_slow'],
                    cur['sma_fast'], cur['sma_slow'], cur['atr']])):
        return _no_sig()

    golden_cross = (prev['sma_fast'] <= prev['sma_slow'] and
                    cur['sma_fast']  >  cur['sma_slow'])
    if not golden_cross:
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    sl    = entry - atr * S2_ATR_SL
    if sl >= entry or sl <= 0:
        return _no_sig()

    return {
        'signal': 1, 'sl': sl, 'tp': 0.0,
        'rr': 0.0, 'exit_type': 'death_cross', 'max_hold': 0,
    }


def check_exit_s2(df: pd.DataFrame, i: int) -> bool:
    """Death Cross 발생 시 True 반환 (청산 트리거)."""
    if i < 1 or i >= len(df):
        return False
    cur = df.iloc[i]
    if any(pd.isna([cur['sma_fast'], cur['sma_slow']])):
        return False
    return float(cur['sma_fast']) < float(cur['sma_slow'])


# ══════════════════════════════════════════════════════════════
#  S3: EMA Trend Follow (4H) — v2 Module A 롱 전용
# ══════════════════════════════════════════════════════════════

def calc_s3(ohlcv: list) -> pd.DataFrame:
    """EMA Trend Follow 지표 계산 (4H)."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['ema_fast']  = c.ewm(span=S3_EMA_FAST,  adjust=False).mean()
    df['ema_slow']  = c.ewm(span=S3_EMA_SLOW,  adjust=False).mean()
    df['ema_trend'] = c.ewm(span=S3_EMA_TREND, adjust=False).mean()
    df['atr']       = _calc_atr(df, S3_ATR_PERIOD)
    df['adx']       = _calc_adx_series(df, S3_ADX_PERIOD)
    return df


def get_signal_s3(df: pd.DataFrame, i: int) -> dict:
    """
    EMA Golden Cross + ADX 필터 + EMA200 상단 확인.
    진입: ema_fast가 ema_slow를 상향돌파 + price > ema_trend + ADX ≥ 20
    청산: ema_fast < ema_slow (데스크로스) 또는 SL/TP
    """
    warmup = S3_EMA_TREND + 10
    if i < warmup:
        return _no_sig()
    prev = df.iloc[i - 1]
    cur  = df.iloc[i]
    cols = ['ema_fast', 'ema_slow', 'ema_trend', 'atr', 'adx']
    if any(pd.isna(cur[c]) for c in cols):
        return _no_sig()

    golden_cross = (prev['ema_fast'] <= prev['ema_slow'] and
                    cur['ema_fast']  >  cur['ema_slow'])
    above_trend  = cur['close'] > cur['ema_trend']
    strong_trend = float(cur['adx']) >= S3_ADX_MIN

    if not (golden_cross and above_trend and strong_trend):
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    sl    = entry - atr * S3_ATR_SL
    if sl >= entry or sl <= 0:
        return _no_sig()
    tp = entry + (entry - sl) * S3_RR

    return {
        'signal': 1, 'sl': sl, 'tp': tp,
        'rr': S3_RR, 'exit_type': 'sl_tp', 'max_hold': 0,
    }


def check_exit_s3(df: pd.DataFrame, i: int) -> bool:
    """EMA Death Cross 시 추가 청산 트리거."""
    if i < 1 or i >= len(df):
        return False
    cur = df.iloc[i]
    if pd.isna(cur['ema_fast']) or pd.isna(cur['ema_slow']):
        return False
    return float(cur['ema_fast']) < float(cur['ema_slow'])


# ══════════════════════════════════════════════════════════════
#  S4: RSI Mean Reversion (1D) — v2 Module B 롱 전용
# ══════════════════════════════════════════════════════════════

def calc_s4(ohlcv: list) -> pd.DataFrame:
    """RSI Mean Reversion 지표 계산 (1D)."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['rsi']      = _calc_rsi(c, S4_RSI_PERIOD)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = _calc_bb(c, S4_BB_PERIOD, S4_BB_SIGMA)
    df['atr']      = _calc_atr(df, S4_ATR_PERIOD)
    return df


def get_signal_s4(df: pd.DataFrame, i: int) -> dict:
    """
    RSI 과매도 반등 + BB 하단 확인.
    진입: RSI가 entry_level을 상향돌파 + 종가 ≤ BB_lower × 1.03
    청산: BB_mid 도달 또는 SL 또는 최대 보유일 초과
    """
    warmup = S4_BB_PERIOD + S4_RSI_PERIOD + 5
    if i < warmup:
        return _no_sig()
    prev = df.iloc[i - 1]
    cur  = df.iloc[i]
    cols = ['rsi', 'bb_mid', 'bb_lower', 'atr']
    if any(pd.isna(cur[c]) for c in cols):
        return _no_sig()

    rsi_cross   = (float(prev['rsi']) < S4_RSI_ENTRY and
                   float(cur['rsi'])  >= S4_RSI_ENTRY)
    near_bb_low = float(cur['close']) <= float(cur['bb_lower']) * 1.03

    if not (rsi_cross and near_bb_low):
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    sl    = entry - atr * S4_ATR_SL
    if sl >= entry or sl <= 0:
        return _no_sig()
    # ATR 기반 TP (1:1 RR) — BB_mid는 저항 없는 임의 목표
    tp = entry + atr * S4_ATR_SL
    rr = 1.0
    return {
        'signal': 1, 'sl': sl, 'tp': tp,
        'rr': rr, 'exit_type': 'sl_tp', 'max_hold': S4_MAX_HOLD,
    }


# ══════════════════════════════════════════════════════════════
#  S5: Bollinger Band Bounce (1D)
# ══════════════════════════════════════════════════════════════

def calc_s5(ohlcv: list) -> pd.DataFrame:
    """Bollinger Band Bounce 지표 계산 (1D)."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['rsi']      = _calc_rsi(c, 14)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = _calc_bb(c, S5_BB_PERIOD, S5_BB_SIGMA)
    df['atr']      = _calc_atr(df, S5_ATR_PERIOD)
    return df


def get_signal_s5(df: pd.DataFrame, i: int) -> dict:
    """
    BB 하단 이탈 + RSI 과매도 확인.
    진입: 종가 < BB_lower + RSI < 40
    청산: BB_upper 도달 또는 SL
    """
    warmup = S5_BB_PERIOD + 15
    if i < warmup:
        return _no_sig()
    cur = df.iloc[i]
    cols = ['rsi', 'bb_upper', 'bb_lower', 'atr']
    if any(pd.isna(cur[c]) for c in cols):
        return _no_sig()

    bb_breach  = float(cur['close']) < float(cur['bb_lower'])
    rsi_confirm = float(cur['rsi']) < S5_RSI_CONFIRM

    if not (bb_breach and rsi_confirm):
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    sl    = entry - atr * S5_ATR_SL
    if sl >= entry or sl <= 0:
        return _no_sig()
    tp  = float(cur['bb_upper'])
    if tp <= entry:
        return _no_sig()

    rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0.0
    return {
        'signal': 1, 'sl': sl, 'tp': tp,
        'rr': round(rr, 2), 'exit_type': 'sl_tp', 'max_hold': 0,
    }


# ══════════════════════════════════════════════════════════════
#  S6: Donchian Breakout (1D) — 터틀 트레이딩
# ══════════════════════════════════════════════════════════════

def calc_s6(ohlcv: list) -> pd.DataFrame:
    """Donchian Breakout 지표 계산 (1D)."""
    df = _ohlcv_to_df(ohlcv)
    df['don_high'] = df['high'].rolling(S6_ENTRY_PERIOD).max()
    df['don_low']  = df['low'].rolling(S6_EXIT_PERIOD).min()
    df['vol_ma']   = df['volume'].rolling(S6_VOL_MA).mean()
    df['atr']      = _calc_atr(df, S6_ATR_PERIOD)
    return df


def get_signal_s6(df: pd.DataFrame, i: int) -> dict:
    """
    20일 신고가 돌파 + 거래량 스파이크.
    진입: 종가 > don_high(이전봉 기준) + 거래량 > vol_ma × 1.3
    청산: 10일 신저가 이탈 (don_low) 또는 SL/TP
    """
    warmup = S6_ENTRY_PERIOD + S6_VOL_MA + 5
    if i < warmup:
        return _no_sig()
    prev = df.iloc[i - 1]   # 이전봉의 don_high (선행편향 방지)
    cur  = df.iloc[i]
    cols = ['don_high', 'don_low', 'vol_ma', 'atr']
    if any(pd.isna(prev[c]) for c in cols) or any(pd.isna(cur[c]) for c in ['atr']):
        return _no_sig()

    breakout   = float(cur['close']) > float(prev['don_high'])
    vol_spike  = float(cur['volume']) > float(prev['vol_ma']) * S6_VOL_MULT

    if not (breakout and vol_spike):
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    # 진입봉 이전 don_low를 SL 기준으로 사용 (진입 후 SL 이동 방지)
    sl    = max(float(prev['don_low']), entry - atr * S6_ATR_SL)
    if sl >= entry or sl <= 0:
        return _no_sig()
    tp = entry + atr * S6_ATR_SL * S6_RR

    rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0.0
    return {
        'signal': 1, 'sl': sl, 'tp': tp,
        'rr': round(rr, 2), 'exit_type': 'sl_tp_or_don_low', 'max_hold': 0,
    }


def check_exit_s6(df: pd.DataFrame, i: int, entry_price: float) -> bool:
    """10일 신저가 이탈 시 추가 청산."""
    if i < 1 or i >= len(df):
        return False
    cur = df.iloc[i]
    if pd.isna(cur['don_low']):
        return False
    return float(cur['close']) < float(cur['don_low'])


# ══════════════════════════════════════════════════════════════
#  S7: MACD Momentum (4H)
# ══════════════════════════════════════════════════════════════

def calc_s7(ohlcv: list) -> pd.DataFrame:
    """MACD Momentum 지표 계산 (4H)."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    ema_fast         = c.ewm(span=S7_MACD_FAST, adjust=False).mean()
    ema_slow         = c.ewm(span=S7_MACD_SLOW, adjust=False).mean()
    df['macd']       = ema_fast - ema_slow
    df['macd_sig']   = df['macd'].ewm(span=S7_MACD_SIG, adjust=False).mean()
    df['macd_hist']  = df['macd'] - df['macd_sig']
    df['ema_trend']  = c.ewm(span=S7_EMA_TREND, adjust=False).mean()
    df['atr']        = _calc_atr(df, S7_ATR_PERIOD)
    return df


def get_signal_s7(df: pd.DataFrame, i: int) -> dict:
    """
    MACD 히스토그램 0선 상향돌파 + EMA200 상단.
    진입: macd_hist가 음→양 전환 + 종가 > ema_trend
    청산: macd_hist < 0 또는 SL/TP
    """
    warmup = S7_MACD_SLOW + S7_EMA_TREND + 5
    if i < warmup:
        return _no_sig()
    prev = df.iloc[i - 1]
    cur  = df.iloc[i]
    cols = ['macd_hist', 'ema_trend', 'atr']
    if any(pd.isna(cur[c]) for c in cols) or pd.isna(prev['macd_hist']):
        return _no_sig()

    hist_cross   = (float(prev['macd_hist']) <= 0 and
                    float(cur['macd_hist'])  > 0)
    above_trend  = float(cur['close']) > float(cur['ema_trend'])

    if not (hist_cross and above_trend):
        return _no_sig()

    entry = float(cur['close'])
    atr   = float(cur['atr'])
    sl    = entry - atr * S7_ATR_SL
    if sl >= entry or sl <= 0:
        return _no_sig()
    tp = entry + (entry - sl) * S7_RR

    return {
        'signal': 1, 'sl': sl, 'tp': tp,
        'rr': S7_RR, 'exit_type': 'sl_tp', 'max_hold': 0,
    }


def check_exit_s7(df: pd.DataFrame, i: int) -> bool:
    """MACD 히스토그램이 음수 전환 시 청산."""
    if i < 1 or i >= len(df):
        return False
    cur = df.iloc[i]
    if pd.isna(cur['macd_hist']):
        return False
    return float(cur['macd_hist']) < 0


# ══════════════════════════════════════════════════════════════
#  전략 레지스트리
# ══════════════════════════════════════════════════════════════

CALC_FUNCS = {
    'S1': calc_s1,
    'S2': calc_s2,
    'S3': calc_s3,
    'S4': calc_s4,
    'S5': calc_s5,
    'S6': calc_s6,
    'S7': calc_s7,
}

SIGNAL_FUNCS = {
    'S1': get_signal_s1,
    'S2': get_signal_s2,
    'S3': get_signal_s3,
    'S4': get_signal_s4,
    'S5': get_signal_s5,
    'S6': get_signal_s6,
    'S7': get_signal_s7,
}

EXIT_CHECK_FUNCS = {
    'S2': check_exit_s2,
    'S3': check_exit_s3,
    'S6': check_exit_s6,
    'S7': check_exit_s7,
}

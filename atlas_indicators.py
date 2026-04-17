"""
ATLAS — 지표 계산 모듈
=======================
4H Phoenix 지표 + 1D Equinox/Bounce 지표 통합
"""

import numpy as np
import pandas as pd
from atlas_config import (
    # Phoenix 4H
    PHX_EMA_TREND, PHX_MACD_FAST, PHX_MACD_SLOW, PHX_MACD_SIGNAL,
    PHX_RSI_PERIOD, PHX_ATR_PERIOD, PHX_BB_PERIOD, PHX_BB_STD,
    PHX_SQUEEZE_LB, PHX_VOL_LB,
    PHX_RSI_LONG_MIN, PHX_RSI_LONG_MAX, PHX_RSI_SHORT_MIN, PHX_RSI_SHORT_MAX,
    PHX_SQUEEZE_MULT, PHX_VOL_MULT, PHX_ATR_MIN_PCT,
    PHX_RR_MIN, PHX_RR_MAX,
    # Equinox 1D
    EQ_EMA_FAST, EQ_EMA_SLOW, EQ_SL_ATR, EQ_TP_ATR,
    # Bounce 1D
    BN_EMA_FAST, BN_EMA_SLOW, BN_RSI_ENTRY, BN_BB_PERIOD, BN_BB_SIGMA,
    BN_SL_ATR, BN_TP_ATR,
)


# ══════════════════════════════════════════════════════════════
#  공통 헬퍼
# ══════════════════════════════════════════════════════════════

def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df


def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    pc = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


# ══════════════════════════════════════════════════════════════
#  Phoenix 4H 지표 계산
# ══════════════════════════════════════════════════════════════

def calc_4h(ohlcv: list, ema_entry: int) -> pd.DataFrame:
    """
    Phoenix 4H 전체 지표 계산.
    ohlcv: [[ts_ms, open, high, low, close, volume], ...]
    ema_entry: 심볼별 진입 EMA 기간 (PHX_SYMBOL_PARAMS에서 가져옴)
    Returns: DataFrame with all indicator columns
    """
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    h  = df['high']
    l  = df['low']

    # 추세 EMA
    df['ema200']      = c.ewm(span=PHX_EMA_TREND, adjust=False).mean()
    df['ema_entry']   = c.ewm(span=ema_entry,     adjust=False).mean()

    # MACD
    ema_fast           = c.ewm(span=PHX_MACD_FAST,   adjust=False).mean()
    ema_slow           = c.ewm(span=PHX_MACD_SLOW,   adjust=False).mean()
    df['macd']         = ema_fast - ema_slow
    df['macd_signal']  = df['macd'].ewm(span=PHX_MACD_SIGNAL, adjust=False).mean()
    df['macd_hist']    = df['macd'] - df['macd_signal']

    # RSI
    df['rsi'] = _calc_rsi(c, PHX_RSI_PERIOD)

    # ATR
    df['atr'] = _calc_atr(df, PHX_ATR_PERIOD)

    # Bollinger Bands
    df['bb_mid']       = c.rolling(PHX_BB_PERIOD).mean()
    bb_std             = c.rolling(PHX_BB_PERIOD).std()
    df['bb_upper']     = df['bb_mid'] + PHX_BB_STD * bb_std
    df['bb_lower']     = df['bb_mid'] - PHX_BB_STD * bb_std
    df['bb_width']     = df['bb_upper'] - df['bb_lower']
    df['bb_width_min'] = df['bb_width'].rolling(PHX_SQUEEZE_LB).min()

    return df


def _calc_dynamic_rr_4h(cur: pd.Series, df: pd.DataFrame,
                         mode: str, direction: int) -> float:
    """
    동적 손익비 계산
    Mode1(눌림목): 1.5 ~ 3.0  |  Mode2(BB스퀴즈): 2.0 ~ 3.5
    """
    if mode == 'M1':
        rsi = float(cur['rsi'])
        if direction == 1:
            rsi_room = (rsi - PHX_RSI_LONG_MIN) / max(50 - PHX_RSI_LONG_MIN, 1)
        else:
            rsi_room = (PHX_RSI_SHORT_MAX - rsi) / max(PHX_RSI_SHORT_MAX - 50, 1)
        rsi_room = max(0.0, min(1.0, rsi_room))

        hist_window = df['macd_hist'].iloc[-21:-1].abs()
        hist_abs    = abs(float(cur['macd_hist']))
        if len(hist_window) > 0 and hist_window.max() > 0:
            hist_strength = float((hist_window < hist_abs).sum()) / len(hist_window)
        else:
            hist_strength = 0.5

        rr = 1.5 + rsi_room * 0.75 + hist_strength * 0.75   # 1.5 ~ 3.0

    else:  # M2
        squeeze_bars = 0
        for i in range(len(df) - 2, max(0, len(df) - 52), -1):
            row = df.iloc[i]
            if row['bb_width'] <= row['bb_width_min'] * PHX_SQUEEZE_MULT:
                squeeze_bars += 1
            else:
                break
        rr = 2.0 + min(squeeze_bars / 15.0, 1.0) * 1.5   # 2.0 ~ 3.5

    return round(max(PHX_RR_MIN, min(rr, PHX_RR_MAX)), 2)


def get_signal_4h(df: pd.DataFrame, params: dict) -> dict:
    """
    Phoenix 4H 신호 반환.
    params: PHX_SYMBOL_PARAMS[sym] dict (atr_sl_mult 사용)
    Returns: {'signal': 1/-1/0, 'mode': 'M1'/'M2'/'',
              'sl': float, 'tp': float, 'rr': float}
    """
    if len(df) < PHX_EMA_TREND + 10:
        return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}

    cur  = df.iloc[-1]
    prev = df.iloc[-2]

    atr_sl = params['atr_sl_mult']
    atr    = cur['atr']
    rsi    = cur['rsi']

    if pd.isna(atr) or pd.isna(cur['bb_width_min']):
        return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}

    # ATR 최솟값 필터 (저변동 구간 차단)
    if atr < float(cur['close']) * PHX_ATR_MIN_PCT:
        return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}

    # ── Mode1: 눌림목 ───────────────────────────────────────
    long_m1 = (
        cur['close']  > cur['ema200'] and
        prev['close'] <= prev['ema_entry'] and cur['close'] > cur['ema_entry'] and
        cur['macd_hist'] > 0 and
        PHX_RSI_LONG_MIN < rsi < PHX_RSI_LONG_MAX
    )
    short_m1 = (
        cur['close']  < cur['ema200'] and
        prev['close'] >= prev['ema_entry'] and cur['close'] < cur['ema_entry'] and
        cur['macd_hist'] < 0 and
        PHX_RSI_SHORT_MIN < rsi < PHX_RSI_SHORT_MAX
    )

    # ── Mode2: BB 스퀴즈 ────────────────────────────────────
    is_sq    = cur['bb_width']  <= cur['bb_width_min']  * PHX_SQUEEZE_MULT
    was_sq   = prev['bb_width'] <= prev['bb_width_min'] * PHX_SQUEEZE_MULT
    vol_avg  = df['volume'].iloc[-PHX_VOL_LB - 1:-1].mean()
    vol_spike = cur['volume'] > vol_avg * PHX_VOL_MULT

    long_m2  = not is_sq and was_sq and cur['close'] > cur['bb_upper'] and vol_spike
    short_m2 = not is_sq and was_sq and cur['close'] < cur['bb_lower'] and vol_spike

    # 신호 우선순위: M1 → M2
    if long_m1:
        rr = _calc_dynamic_rr_4h(cur, df, 'M1', 1)
        sl = cur['close'] - atr * atr_sl
        tp = cur['close'] + (cur['close'] - sl) * rr
        return {'signal': 1, 'mode': 'M1', 'sl': sl, 'tp': tp, 'rr': rr}

    if short_m1:
        rr = _calc_dynamic_rr_4h(cur, df, 'M1', -1)
        sl = cur['close'] + atr * atr_sl
        tp = cur['close'] - (sl - cur['close']) * rr
        return {'signal': -1, 'mode': 'M1', 'sl': sl, 'tp': tp, 'rr': rr}

    if long_m2:
        sl = cur['close'] - atr * atr_sl
        if sl >= cur['close']:
            return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}
        rr = _calc_dynamic_rr_4h(cur, df, 'M2', 1)
        tp = cur['close'] + (cur['close'] - sl) * rr
        return {'signal': 1, 'mode': 'M2', 'sl': sl, 'tp': tp, 'rr': rr}

    if short_m2:
        sl = cur['close'] + atr * atr_sl
        if sl <= cur['close']:
            return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}
        rr = _calc_dynamic_rr_4h(cur, df, 'M2', -1)
        tp = cur['close'] - (sl - cur['close']) * rr
        return {'signal': -1, 'mode': 'M2', 'sl': sl, 'tp': tp, 'rr': rr}

    return {'signal': 0, 'mode': '', 'sl': 0, 'tp': 0, 'rr': 0}


# ══════════════════════════════════════════════════════════════
#  Equinox 1D 지표 + 신호
# ══════════════════════════════════════════════════════════════

def calc_1d_eq(ohlcv: list, ema_fast: int = EQ_EMA_FAST,
               ema_slow: int = EQ_EMA_SLOW) -> pd.DataFrame:
    """Equinox 1D 지표 계산. ema_fast/slow는 Walk-Forward로 동적 변경 가능."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    pc = c.shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs(),
    ], axis=1).max(axis=1)
    df['atr']      = tr.ewm(span=14, adjust=False).mean()
    df['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=ema_slow, adjust=False).mean()
    df['ma200']    = c.rolling(200).mean()
    df['vma20']    = df['volume'].rolling(20).median()
    return df.dropna(subset=['atr', 'ema_fast', 'ema_slow', 'vma20']).reset_index(drop=True)


def get_signal_eq(df: pd.DataFrame, today_open: float,
                  today_high: float, today_vol: float) -> tuple:
    """
    Equinox 1D LONG 신호.
    Returns: (ok, entry_price, sl, tp, atr, reason)
    조건: 전일 EMA_FAST>EMA_SLOW + Close>MA200 + 거래량 + 상승 모멘텀
    """
    if len(df) < 2:
        return False, 0, 0, 0, 0, '데이터 부족'
    prev = df.iloc[-1]
    atr  = float(prev['atr'])

    if float(prev['ema_fast']) <= float(prev['ema_slow']):
        return False, 0, 0, 0, atr, f'EMA 하락추세 ({prev["ema_fast"]:,.0f} ≤ {prev["ema_slow"]:,.0f})'

    if float(prev['ma200']) > 0 and float(prev['close']) < float(prev['ma200']):
        return False, 0, 0, 0, atr, f'MA200 아래 ({prev["close"]:,.2f} < {prev["ma200"]:,.2f})'

    if today_vol < float(prev['vma20']) * 1.0:
        return False, 0, 0, 0, atr, f'거래량 부족 ({today_vol / prev["vma20"]:.2f}x)'

    entry_cand = today_open + atr * 0.3
    if today_high < entry_cand or today_high <= today_open:
        return False, 0, 0, 0, atr, f'모멘텀 없음 (고가 {today_high:,.2f} < 목표 {entry_cand:,.2f})'

    ep = entry_cand
    sl = ep - atr * EQ_SL_ATR
    tp = ep + atr * EQ_TP_ATR
    return True, ep, sl, tp, atr, ''


# ══════════════════════════════════════════════════════════════
#  Bounce 1D 지표 + 신호
# ══════════════════════════════════════════════════════════════

def calc_1d_bn(ohlcv: list) -> pd.DataFrame:
    """Bounce 1D 지표 계산."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    pc = c.shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs(),
    ], axis=1).max(axis=1)
    df['atr']      = tr.ewm(span=14, adjust=False).mean()
    df['ema_fast'] = c.ewm(span=BN_EMA_FAST, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=BN_EMA_SLOW, adjust=False).mean()
    df['rsi14']    = _calc_rsi(c, 14)
    bb_ma          = c.rolling(BN_BB_PERIOD).mean()
    bb_std         = c.rolling(BN_BB_PERIOD).std()
    df['bb_mid']   = bb_ma
    df['bb_lower'] = bb_ma - bb_std * BN_BB_SIGMA
    df['bb_upper'] = bb_ma + bb_std * BN_BB_SIGMA
    df['vma20']    = df['volume'].rolling(20).median()
    return df.dropna().reset_index(drop=True)


def get_signal_bn(df: pd.DataFrame) -> tuple:
    """
    Bounce 1D LONG 신호.
    Returns: (ok, entry_price, sl, tp, atr, reason)
    조건: EMA_FAST>EMA_SLOW(대추세 상승) + RSI<35(과매도) + BB하단 이탈
    """
    if len(df) < BN_BB_PERIOD + 5:
        return False, 0, 0, 0, 0, '데이터 부족'

    prev  = df.iloc[-1]
    atr   = float(prev['atr'])
    ef    = float(prev['ema_fast'])
    es    = float(prev['ema_slow'])
    rsi   = float(prev['rsi14'])
    close = float(prev['close'])
    bb_lo = float(prev['bb_lower'])

    if ef <= es:
        return False, 0, 0, 0, atr, f'대추세 하락 ({ef:,.0f} ≤ {es:,.0f})'
    if rsi >= BN_RSI_ENTRY:
        return False, 0, 0, 0, atr, f'RSI 정상 ({rsi:.1f} ≥ {BN_RSI_ENTRY})'
    if close >= bb_lo:
        return False, 0, 0, 0, atr, f'BB 이탈 미발생 ({close:,.2f} ≥ {bb_lo:,.2f})'

    ep = close
    sl = ep - atr * BN_SL_ATR
    tp = ep + atr * BN_TP_ATR
    return True, ep, sl, tp, atr, ''


# ══════════════════════════════════════════════════════════════
#  ADX 계산 (장세 분류용)
# ══════════════════════════════════════════════════════════════

def calc_adx(ohlcv: list, period: int = 14) -> float:
    """
    Wilder 방식 ADX 계산.
    Returns: ADX 값 (float), 데이터 부족 시 0.0
    """
    if len(ohlcv) < period * 3:
        return 0.0

    df = _ohlcv_to_df(ohlcv)
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    dm_plus  = []
    dm_minus = []
    trs      = []

    for i in range(1, len(c)):
        up   = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        dm_plus.append(up   if up > down and up > 0 else 0.0)
        dm_minus.append(down if down > up and down > 0 else 0.0)
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        trs.append(tr)

    def _wilder(lst, p):
        result = []
        s = sum(lst[:p])
        result.append(s)
        for v in lst[p:]:
            s = s - s / p + v
            result.append(s)
        return result

    tr_w   = _wilder(trs,      period)
    dmp_w  = _wilder(dm_plus,  period)
    dmm_w  = _wilder(dm_minus, period)

    di_plus  = [100 * d / t if t else 0 for d, t in zip(dmp_w, tr_w)]
    di_minus = [100 * d / t if t else 0 for d, t in zip(dmm_w, tr_w)]
    dx_list  = [abs(p - m) / (p + m) * 100 if (p + m) else 0
                for p, m in zip(di_plus, di_minus)]

    if len(dx_list) < period:
        return 0.0

    # ADX smoothing: Wilder 원본 공식
    # 초기값 = mean(DX[:period]), 이후 = prev - prev/period + DX/period
    # (_wilder는 TR/DM용 합계 초기화 방식이라 ADX에는 별도 계산)
    adx = sum(dx_list[:period]) / period
    for dx_val in dx_list[period:]:
        adx = adx - adx / period + dx_val / period
    return round(adx, 2)

"""
ATLAS — 지표 계산 모듈
=======================
현물 봇 공유 기술 지표 라이브러리.
"""

import numpy as np
import pandas as pd


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
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    # loss=0 구간(구간 내 하락일 없음) → RS=무한대이므로 RSI=100(gain>0) 또는 50(완전 무변동)
    # 분모 0 나눗셈으로 RS가 NaN이 되어 강한 상승장에서 RSI가 통째로 비는 문제 방지
    zero_loss = loss == 0
    rsi = rsi.where(~zero_loss, np.where(gain > 0, 100.0, 50.0))
    return rsi


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

    adx = sum(dx_list[:period]) / period
    for dx_val in dx_list[period:]:
        adx = adx - adx / period + dx_val / period
    return round(adx, 2)


# ══════════════════════════════════════════════════════════════
#  Alpha 지표
# ══════════════════════════════════════════════════════════════

def calc_oi_change_pct(oi_series: list) -> float:
    """
    4H OI 연속 2개 스냅샷의 변화율(%).
    oi_series: [{'openInterestValue': float, ...}, ...] (ccxt fetch_open_interest_history 형식)
    +값: 신규 자금 유입 (진짜 추세), -값: 청산/숏스퀴즈 의심
    """
    if len(oi_series) < 2:
        return 0.0
    try:
        prev_oi = float(oi_series[-2].get('openInterestValue', 0))
        curr_oi = float(oi_series[-1].get('openInterestValue', 0))
        if prev_oi <= 0:
            return 0.0
        return (curr_oi - prev_oi) / prev_oi * 100
    except Exception:
        return 0.0


def calc_dynamic_rr_ma(adx: float, ema_gap_pct: float) -> float:
    """
    ADX 강도 + EMA 갭 기반 동적 RR (1.5~3.0).
    추세 강도가 높을수록 TP를 멀리, 약할수록 조기 이탈.
    """
    adx_norm = max(0.0, min(1.0, (adx - 15.0) / (45.0 - 15.0)))
    gap_norm = max(0.0, min(1.0, (ema_gap_pct - 0.5) / (3.0 - 0.5)))
    rr = 1.5 + adx_norm * 1.0 + gap_norm * 0.5  # 1.5 ~ 3.0
    return round(max(1.5, min(3.0, rr)), 2)

"""
ATLAS — 지표 계산 모듈
=======================
현물 봇 공유 기술 지표 라이브러리.
"""

import math
from typing import Optional

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

ADX_MIN_BARS_MULT = 3      # 필요 봉 수 = period × 이 값


def adx_min_bars(period: int = 14) -> int:
    """calc_adx가 값을 내기 위해 필요한 최소 봉 수.

    호출측이 이 값을 확인하지 않으면, 데이터가 모자랄 때 반환되는 0.0을
    '추세가 없다'로 오해하게 된다 — 레짐 분류에서는 그게 곧 RANGING이다.
    """
    return period * ADX_MIN_BARS_MULT


def calc_adx(ohlcv: list, period: int = 14) -> float:
    """
    Wilder 방식 ADX 계산.

    Returns: ADX 값 (float). **데이터 부족 시 0.0** —
    호출측은 `adx_min_bars(period)`로 봉 수를 먼저 확인할 것.
    0.0은 '추세 없음'과 구분되지 않으므로, 레짐 분류에 그대로 넘기면
    데이터 문제가 RANGING으로 위장된다.
    """
    if len(ohlcv) < adx_min_bars(period):
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

    di_plus  = [100 * d / t if t else 0 for d, t in zip(dmp_w, tr_w, strict=True)]
    di_minus = [100 * d / t if t else 0 for d, t in zip(dmm_w, tr_w, strict=True)]
    dx_list  = [abs(p - m) / (p + m) * 100 if (p + m) else 0
                for p, m in zip(di_plus, di_minus, strict=True)]

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


# 동적 RR 기여 비중 — ADX가 2/3, EMA 갭이 1/3.
# (원래 1.5 + adx*1.0 + gap*0.5 로 하드코딩돼 있던 가중치를 이름으로 뽑았다)
RR_ADX_WEIGHT = 2.0 / 3.0
RR_GAP_WEIGHT = 1.0 / 3.0


def calc_dynamic_rr_ma(adx: float, ema_gap_pct: float,
                       rr_min: float = 1.5, rr_max: float = 3.0,
                       rr_fallback: Optional[float] = None) -> float:
    """ADX 강도 + EMA 갭 기반 동적 RR.

    추세 강도가 높을수록 TP를 멀리, 약할수록 조기 이탈.

    ⚠️ 범위를 **인자로 받는다.** 예전에는 1.5~3.0이 이 함수 안에 하드코딩돼
    있었고, config의 S3/S6/S7_RR_MIN·MAX 6개는 그 값과 우연히 같아서
    "동작하는 것처럼" 보였다. 실제로는 S6_RR_MAX를 5.0으로 바꿔도 TP가
    전혀 변하지 않았다 — 튜닝 가능하다고 문서화된 값이 사실은 상수였고,
    WFO 그리드에 넣었다면 모든 후보가 동점이 되는 그 실패 그대로다.

    호출자는 자기 전략의 상수를 넘겨야 한다. 기본값은 기존 동작과 동일하다.

    ⚠️ **결측(NaN) 처리가 이 함수의 핵심 안전장치다.** 예전에는 NaN이
    그대로 흘러들어와 `max(0, min(1, nan))`이 1.0을 내놓았다 — 파이썬의
    min/max는 NaN 비교가 전부 False라 첫 인자를 돌려주기 때문이다.
    그 결과 **지표 결측이 '최강 추세'로 둔갑해 가장 공격적인 TP**가
    잡혔다. 데이터가 없을 때 가장 공격적으로 베팅하는 것은 정확히
    반대로 가는 실패다. 이제 결측이면 `rr_fallback`(전략의 정적 RR)을
    쓰고, 그것도 없으면 범위 중앙값으로 물러선다.
    """
    if rr_max < rr_min:
        rr_min, rr_max = rr_max, rr_min      # 뒤집힌 설정에도 발산하지 않게

    def _bad(v) -> bool:
        try:
            return not math.isfinite(float(v))
        except (TypeError, ValueError):
            return True

    if _bad(adx) or _bad(ema_gap_pct):
        fb = rr_fallback if rr_fallback is not None else (rr_min + rr_max) / 2.0
        return round(max(rr_min, min(rr_max, float(fb))), 2)

    adx_norm = max(0.0, min(1.0, (adx - 15.0) / (45.0 - 15.0)))
    gap_norm = max(0.0, min(1.0, (ema_gap_pct - 0.5) / (3.0 - 0.5)))
    span = rr_max - rr_min
    rr = rr_min + span * (adx_norm * RR_ADX_WEIGHT + gap_norm * RR_GAP_WEIGHT)
    return round(max(rr_min, min(rr_max, rr)), 2)

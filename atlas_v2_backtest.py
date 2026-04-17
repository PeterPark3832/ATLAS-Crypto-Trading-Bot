"""
ATLAS v2 백테스트
==================
Module A: Trend Follow 4H  — BTC/ETH/SOL/BNB  (롱+숏)
Module B: Mean Reversion 1D — ETH/BNB          (롱+숏)

[Module A]
  신호  : EMA(20) 교차 EMA(50)  ← 이전봉 vs 현재봉 크로스오버 감지
  방향  : TRENDING_UP → LONG only
          TRENDING_DOWN → SHORT only
          RANGING → 스킵 (Module B 구간)
          WEAK_TREND → 양방향 허용 (per-심볼 EMA200 필터 적용)
          CRISIS → 전면 차단
  SL    : ATR × V2_MA_ATR_SL (진입봉 ATR 고정)
  BEP   : 1R 이익 도달 시 SL → 진입가
  Trail : BEP 이후 peak/trough 갱신 → SL = peak - ATR_현재 × V2_MA_ATR_SL
  EMA200: 심볼별 자체 EMA200 vs close 비교 (BTC레짐과 독립)

[Module B]
  신호  : RSI < 30 + 종가 ≤ BB_lower → LONG
          RSI > 70 + 종가 ≥ BB_upper → SHORT
  레짐  : RANGING 전용 (Module A 비활성 구간 활용)
  SL    : ATR × V2_MR_ATR_SL (고정)
  TP    : BB_mid (SMA20) — 평균회귀 목표
  시간청산: V2_MR_MAX_BARS봉 초과 시 당일 종가로 청산

[비용]
  수수료: BT_FEE_TAKER × 2 (진입+청산)
  슬리피지: BT_SLIPPAGE (시장가 진입 시 불리한 방향)

사용법:
  python atlas_v2_backtest.py                        # 전체 (2022-01-01~현재)
  python atlas_v2_backtest.py --start 2022-01-01     # 시작일 지정
  python atlas_v2_backtest.py --module a             # Module A만
  python atlas_v2_backtest.py --module b             # Module B만
  python atlas_v2_backtest.py --wf                   # Walk-Forward 검증
  python atlas_v2_backtest.py --data-dir data/       # 로컬 CSV 사용
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# atlas_config 임포트 전 더미 환경변수 설정
for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'BACKTEST')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_config import (
    BT_FEE_TAKER, BT_SLIPPAGE,
    V2_MA_SYMBOLS, V2_MR_SYMBOLS,
    V2_MA_EMA_FAST, V2_MA_EMA_SLOW, V2_MA_ATR_PERIOD,
    V2_MA_ATR_SL, V2_MA_TRAIL_R, V2_MA_RISK_PCT,
    V2_MA_LEVERAGE, V2_MA_COOLDOWN, V2_MA_ADX_MIN,
    V2_MR_RSI_PERIOD, V2_MR_RSI_LONG, V2_MR_RSI_SHORT,
    V2_MR_BB_PERIOD, V2_MR_BB_SIGMA, V2_MR_ATR_PERIOD,
    V2_MR_ATR_SL, V2_MR_MAX_BARS, V2_MR_RISK_PCT,
    V2_MR_LEVERAGE,
)
# 기존 유틸리티 재활용 (데이터 로드 / 레짐맵 / 성과지표)
from atlas_backtest import (
    fetch_ohlcv_public, load_ohlcv_csv,
    build_regime_map, calc_metrics, TradeResult,
)
from atlas_indicators import _ohlcv_to_df, _calc_atr, _calc_rsi


# ══════════════════════════════════════════════════════════════
#  지표 계산
# ══════════════════════════════════════════════════════════════

def _calc_bb(close: pd.Series, period: int, sigma: float):
    """BB_mid / BB_upper / BB_lower 반환."""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    return mid, mid + sigma * std, mid - sigma * std


def calc_v2_ma_4h(ohlcv: list) -> pd.DataFrame:
    """
    Module A 지표 계산 (4H).
    EMA20, EMA50, EMA200, ATR14, ADX14 (rolling window)
    """
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['ema20']  = c.ewm(span=V2_MA_EMA_FAST,  adjust=False).mean()
    df['ema50']  = c.ewm(span=V2_MA_EMA_SLOW,  adjust=False).mean()
    df['ema200'] = c.ewm(span=200,              adjust=False).mean()
    df['atr']    = _calc_atr(df, V2_MA_ATR_PERIOD)

    # ADX 벡터화 계산 (Wilder smoothing)
    # TR/DM: sum-based (S = S - S/p + x)
    # ADX:   mean-based (A = A - A/p + dx/p) → 값이 0~100 범위 유지
    period = 14
    h = df['high']
    lo = df['low']
    pc = c.shift(1)
    up   = h - h.shift(1)
    down = lo.shift(1) - lo
    dm_plus  = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)

    def _wilder_sum(arr, p):
        """TR/DM용: 합계 초기화, S = S - S/p + x"""
        result = np.full(len(arr), np.nan)
        if len(arr) < p:
            return result
        result[p - 1] = float(np.nansum(arr[:p]))
        for i in range(p, len(arr)):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    def _wilder_mean(arr, p):
        """ADX용: 평균 초기화, A = A - A/p + x/p (0~100 범위 유지)"""
        result = np.full(len(arr), np.nan)
        if len(arr) < p:
            return result
        result[p - 1] = float(np.nanmean(arr[:p]))
        for i in range(p, len(arr)):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i] / p
        return result

    tr_w   = _wilder_sum(tr.values,  period)
    dmp_w  = _wilder_sum(dm_plus,    period)
    dmm_w  = _wilder_sum(dm_minus,   period)
    di_p   = np.where(tr_w > 0, 100 * dmp_w / tr_w, 0.0)
    di_m   = np.where(tr_w > 0, 100 * dmm_w / tr_w, 0.0)
    dip_m  = di_p + di_m
    dx     = np.where(dip_m > 0, np.abs(di_p - di_m) / dip_m * 100, 0.0)
    adx    = _wilder_mean(dx, period)
    df['adx'] = adx
    return df


def calc_v2_mr_1d(ohlcv: list) -> pd.DataFrame:
    """
    Module B 지표 계산 (1D).
    RSI14, BB(20, 2), ATR14
    """
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['rsi']                          = _calc_rsi(c, V2_MR_RSI_PERIOD)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = _calc_bb(c, V2_MR_BB_PERIOD, V2_MR_BB_SIGMA)
    df['atr']                          = _calc_atr(df, V2_MR_ATR_PERIOD)
    return df


# ══════════════════════════════════════════════════════════════
#  Module A: Trend Follow 4H
# ══════════════════════════════════════════════════════════════

def backtest_ma(symbol: str,
                ohlcv_4h: list,
                regime_map: dict,
                start_date: Optional[str] = None,
                end_date:   Optional[str] = None) -> tuple:
    """
    Module A 바-바이-바 시뮬레이션.
    Returns: (trades, diagnostics_dict)
    """
    MIN_WARMUP = max(V2_MA_EMA_SLOW + 30, 55)  # EMA50 + ADX 워밍업 충분히
    if len(ohlcv_4h) < MIN_WARMUP + 10:
        print(f'    [스킵] ModuleA/{symbol}: 데이터 부족 ({len(ohlcv_4h)}봉)')
        return [], {}

    df = calc_v2_ma_4h(ohlcv_4h)

    start_ts = (int(datetime.strptime(start_date, '%Y-%m-%d')
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
                if start_date else 0)
    end_ts   = (int(datetime.strptime(end_date, '%Y-%m-%d')
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
                if end_date else None)

    diag = {
        'total_bars': 0, 'entries': 0, 'exit_sl': 0, 'exit_tp': 0,
        'rejected': defaultdict(int),
        'regime_entries':    defaultdict(int),
        'direction_entries': defaultdict(int),
        'pnl_r_buckets': {
            '< -1.5R': 0, '-1.5~-1R': 0, '-1~-0.5R': 0, '-0.5~0R': 0,
            '0~0.5R':  0, '0.5~1R':   0, '1~1.5R':   0, '1.5~2R':  0, '> 2R': 0,
        },
        'durations': [],
    }

    trades:   list = []
    position: Optional[dict] = None
    cooldown: int  = 0

    for i in range(MIN_WARMUP, len(df) - 1):
        bar_ts = int(ohlcv_4h[i][0])
        if bar_ts < start_ts:
            continue
        if end_ts and bar_ts >= end_ts:
            break

        diag['total_bars'] += 1

        bar_h  = float(ohlcv_4h[i][2])
        bar_l  = float(ohlcv_4h[i][3])
        bar_c  = float(ohlcv_4h[i][4])
        ts_dt  = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
        dk     = str(ts_dt.date())
        row    = df.iloc[i]

        atr_now = float(row['atr']) if not pd.isna(row['atr']) else 0.0

        if cooldown > 0:
            cooldown -= 1

        # ── 포지션 관리 ────────────────────────────────────────────
        if position is not None:
            d       = position['direction']
            sl      = position['sl']
            tp      = position['tp']
            ent     = position['entry']
            sl_dist = position['sl_dist']
            fee_r   = position['fee_r']
            duration = i - position['entry_bar']
            exited = False

            if d == 'LONG':
                # SL 먼저 체크 (불리한 방향)
                if bar_l <= sl:
                    pnl_r = (sl - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MA_LONG', direction='LONG', mode='MA',
                        entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=V2_MA_RISK_PCT,
                        reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_sl'] += 1
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    diag['durations'].append(duration)
                    position = None; cooldown = V2_MA_COOLDOWN; exited = True
                elif bar_h >= tp:
                    pnl_r = (tp - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MA_LONG', direction='LONG', mode='MA',
                        entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=V2_MA_RISK_PCT,
                        reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_tp'] += 1
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    diag['durations'].append(duration)
                    position = None; cooldown = V2_MA_COOLDOWN; exited = True
            else:  # SHORT
                if bar_h >= sl:
                    pnl_r = (ent - sl) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MA_SHORT', direction='SHORT', mode='MA',
                        entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=V2_MA_RISK_PCT,
                        reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_sl'] += 1
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    diag['durations'].append(duration)
                    position = None; cooldown = V2_MA_COOLDOWN; exited = True
                elif bar_l <= tp:
                    pnl_r = (ent - tp) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MA_SHORT', direction='SHORT', mode='MA',
                        entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=V2_MA_RISK_PCT,
                        reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_tp'] += 1
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    diag['durations'].append(duration)
                    position = None; cooldown = V2_MA_COOLDOWN; exited = True

            if exited:
                continue

        # ── 신호 체크 ────────────────────────────────────────────
        if position is not None:
            diag['rejected']['position_open'] += 1
            continue
        if cooldown > 0:
            diag['rejected']['cooldown_active'] += 1
            continue

        regime = regime_map.get(dk, 'UNKNOWN')

        # CRISIS / RANGING / UNKNOWN → 스킵
        if regime in ('CRISIS', 'RANGING', 'UNKNOWN'):
            diag['rejected']['regime_skip'] += 1
            continue

        # EMA 교차 감지 (이전봉 vs 현재봉)
        r_prev = df.iloc[i - 1]
        ema20_prev  = float(r_prev['ema20'])
        ema50_prev  = float(r_prev['ema50'])
        ema20_curr  = float(row['ema20'])
        ema50_curr  = float(row['ema50'])
        ema200_curr = float(row['ema200'])

        adx_curr = float(row['adx']) if not pd.isna(row['adx']) else 0.0

        if any(pd.isna(v) for v in (ema20_prev, ema50_prev, ema20_curr, ema50_curr, ema200_curr)):
            diag['rejected']['nan_indicator'] += 1
            continue

        # ADX 확인: 방향성 모멘텀 부족 → 크로스오버 신호 무시
        if adx_curr < V2_MA_ADX_MIN:
            diag['rejected']['adx_too_low'] += 1
            continue

        long_cross  = (ema20_prev <= ema50_prev) and (ema20_curr > ema50_curr)
        short_cross = (ema20_prev >= ema50_prev) and (ema20_curr < ema50_curr)

        if not long_cross and not short_cross:
            diag['rejected']['no_signal'] += 1
            continue

        direction = 'LONG' if long_cross else 'SHORT'

        # 레짐-방향 필터
        if direction == 'LONG'  and regime == 'TRENDING_DOWN':
            diag['rejected']['regime_direction'] += 1
            continue
        if direction == 'SHORT' and regime == 'TRENDING_UP':
            diag['rejected']['regime_direction'] += 1
            continue

        # EMA200 필터 (심볼별 자체 EMA200, 모든 레짐에 적용)
        if direction == 'LONG'  and bar_c < ema200_curr:
            diag['rejected']['ema200_filter'] += 1
            continue
        if direction == 'SHORT' and bar_c > ema200_curr:
            diag['rejected']['ema200_filter'] += 1
            continue

        # ATR 체크
        if atr_now <= 0:
            diag['rejected']['zero_atr'] += 1
            continue

        # 다음 봉 시가에 진입
        entry_px = float(ohlcv_4h[i + 1][1])
        if entry_px <= 0:
            diag['rejected']['zero_entry'] += 1
            continue

        # 슬리피지 반영
        if direction == 'LONG':
            entry_eff = entry_px * (1 + BT_SLIPPAGE)
            sl_init   = entry_eff - atr_now * V2_MA_ATR_SL
            tp_init   = entry_eff + atr_now * V2_MA_ATR_SL * 2.0  # 2R 고정 TP
        else:
            entry_eff = entry_px * (1 - BT_SLIPPAGE)
            sl_init   = entry_eff + atr_now * V2_MA_ATR_SL
            tp_init   = entry_eff - atr_now * V2_MA_ATR_SL * 2.0  # 2R 고정 TP

        sl_dist = abs(entry_eff - sl_init)
        if sl_dist < 1e-9:
            diag['rejected']['zero_sl_dist'] += 1
            continue

        fee_r = entry_eff * 2 * BT_FEE_TAKER / sl_dist

        diag['entries'] += 1
        diag['regime_entries'][regime] += 1
        diag['direction_entries'][direction] += 1

        position = {
            'direction': direction,
            'entry':     entry_eff,
            'sl':        sl_init,
            'tp':        tp_init,
            'sl_dist':   sl_dist,
            'fee_r':     fee_r,
            'entry_bar': i + 1,
            'regime':    regime,
        }

    # 평균 보유 봉수
    diag['avg_duration'] = (sum(diag['durations']) / len(diag['durations'])
                            if diag['durations'] else 0.0)
    return trades, diag


def _bucket_pnl_r(buckets: dict, pnl_r: float) -> None:
    if   pnl_r < -1.5: buckets['< -1.5R']  += 1
    elif pnl_r < -1.0: buckets['-1.5~-1R'] += 1
    elif pnl_r < -0.5: buckets['-1~-0.5R'] += 1
    elif pnl_r <  0.0: buckets['-0.5~0R']  += 1
    elif pnl_r <  0.5: buckets['0~0.5R']   += 1
    elif pnl_r <  1.0: buckets['0.5~1R']   += 1
    elif pnl_r <  1.5: buckets['1~1.5R']   += 1
    elif pnl_r <  2.0: buckets['1.5~2R']   += 1
    else:              buckets['> 2R']      += 1


def print_ma_diagnostics(sym: str, diag: dict) -> None:
    w = 56
    print(f'\n  {"─" * w}')
    print(f'  Module A 진단: {sym}')
    print(f'  {"─" * w}')
    tb = diag['total_bars']
    print(f'  총 처리 봉수: {tb}')

    print(f'\n  [필터별 거부 현황]')
    filters = [
        ('position_open',    '포지션 보유 중'),
        ('cooldown_active',  '쿨다운 중'),
        ('regime_skip',      'CRISIS/RANGING/UNKNOWN 스킵'),
        ('adx_too_low',      'ADX 부족 (방향성 없음)'),
        ('no_signal',        'EMA 교차 없음'),
        ('regime_direction', '레짐-방향 불일치'),
        ('ema200_filter',    'EMA200 필터'),
        ('zero_atr',         'ATR=0'),
        ('zero_entry',       '진입가=0'),
        ('nan_indicator',    '지표 NaN'),
    ]
    for key, label in filters:
        cnt = diag['rejected'].get(key, 0)
        pct = cnt / tb * 100 if tb else 0
        print(f'    {label:<28} : {cnt:>5}봉  ({pct:5.1f}%)')
    ent = diag['entries']
    print(f'    {"진입 시도":<28} : {ent:>5}봉  '
          f'({ent / tb * 100:.1f}% 진입율)' if tb else '')

    print(f'\n  [레짐별 진입]')
    for r, cnt in sorted(diag['regime_entries'].items(), key=lambda x: -x[1]):
        print(f'    {r:<22} : {cnt:>4}건')

    print(f'\n  [방향별 진입]')
    for d, cnt in diag['direction_entries'].items():
        print(f'    {d:<10} : {cnt}건')

    print(f'\n  [R-배수 분포] (진입 {ent}건)')
    for label, cnt in diag['pnl_r_buckets'].items():
        bar = '#' * min(cnt, 40)
        print(f'    {label:<14} : {cnt:>4}건  {bar}')

    sl_cnt = diag['exit_sl']
    tp_cnt = diag['exit_tp']
    tot = sl_cnt + tp_cnt
    if tot:
        print(f'\n  [청산 통계]')
        print(f'    SL 청산 : {sl_cnt}건  ({sl_cnt/tot*100:.1f}%)')
        print(f'    TP 청산 : {tp_cnt}건  ({tp_cnt/tot*100:.1f}%)')
        if diag['avg_duration']:
            print(f'    평균 보유 : {diag["avg_duration"]:.1f}봉')
    print(f'  {"─" * w}')


# ══════════════════════════════════════════════════════════════
#  Module B: Mean Reversion 1D
# ══════════════════════════════════════════════════════════════

def backtest_mr(symbol: str,
                ohlcv_1d: list,
                regime_map: dict,
                start_date: Optional[str] = None,
                end_date:   Optional[str] = None) -> list:
    """
    Module B 바-바이-바 시뮬레이션 (1D).
    RANGING 레짐 전용 RSI+BB 평균회귀.
    Returns: trades list
    """
    MIN_WARMUP = max(V2_MR_RSI_PERIOD, V2_MR_BB_PERIOD) + 10
    if len(ohlcv_1d) < MIN_WARMUP + 10:
        print(f'    [스킵] ModuleB/{symbol}: 데이터 부족 ({len(ohlcv_1d)}봉)')
        return []

    df = calc_v2_mr_1d(ohlcv_1d)

    start_dt = (pd.Timestamp(start_date, tz='UTC') if start_date else None)
    end_dt   = (pd.Timestamp(end_date,   tz='UTC') if end_date   else None)

    trades:   list = []
    position: Optional[dict] = None

    for i in range(MIN_WARMUP, len(df)):
        row   = df.iloc[i]
        ts_dt = row['ts']
        if start_dt and ts_dt < start_dt:
            continue
        if end_dt and ts_dt >= end_dt:
            break

        dk    = str(ts_dt.date())
        bar_h = float(row['high'])
        bar_l = float(row['low'])
        bar_c = float(row['close'])
        regime = regime_map.get(dk, 'UNKNOWN')

        # ── 포지션 관리 ──────────────────────────────────────────
        if position is not None:
            d       = position['direction']
            ent     = position['entry']
            sl      = position['sl']
            tp      = position['tp']
            sl_dist = position['sl_dist']
            fee_r   = position['fee_r']
            duration = i - position['entry_bar']

            if d == 'LONG':
                # SL 먼저 체크 (불리한 방향)
                if bar_l <= sl:
                    pnl_r = (sl - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MR_LONG', direction='LONG', mode='MR',
                        entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=V2_MR_RISK_PCT,
                        reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    position = None; continue
                # TP 체크
                if bar_h >= tp:
                    pnl_r = (tp - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MR_LONG', direction='LONG', mode='MR',
                        entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=V2_MR_RISK_PCT,
                        reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    position = None; continue
            else:  # SHORT
                if bar_h >= sl:
                    pnl_r = (ent - sl) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MR_SHORT', direction='SHORT', mode='MR',
                        entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=V2_MR_RISK_PCT,
                        reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    position = None; continue
                if bar_l <= tp:
                    pnl_r = (ent - tp) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy='MR_SHORT', direction='SHORT', mode='MR',
                        entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=V2_MR_RISK_PCT,
                        reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    position = None; continue

            # 시간 청산 (최대 보유 봉수 초과)
            if duration >= V2_MR_MAX_BARS:
                if d == 'LONG':
                    pnl_r = (bar_c - ent) / sl_dist - fee_r
                else:
                    pnl_r = (ent - bar_c) / sl_dist - fee_r
                trades.append(TradeResult(
                    symbol=symbol, strategy=f'MR_{d}', direction=d, mode='MR',
                    entry=ent, exit_px=bar_c, pnl_r=pnl_r, risk_pct=V2_MR_RISK_PCT,
                    reason='TIME', entry_bar=position['entry_bar'], exit_bar=i,
                    regime=position['regime']))
                position = None; continue

        # ── 신호 체크 ──────────────────────────────────────────
        if position is not None:
            continue

        # RANGING 전용
        if regime != 'RANGING':
            continue

        # RSI 크로스오버 신호: 과매도/과매수 구간 진입 후 회복 시작 확인
        # 이전봉이 RSI<35였고 현재봉이 35 이상으로 올라서면 LONG
        # (= 최악의 셀오프가 끝나고 회복이 시작됨)
        if i < 1:
            continue
        row_prev = df.iloc[i - 1]
        rsi_prev = float(row_prev['rsi']) if not pd.isna(row_prev['rsi']) else 50.0
        rsi_curr = float(row['rsi'])      if not pd.isna(row['rsi'])      else 50.0
        atr      = float(row['atr'])      if not pd.isna(row['atr'])      else 0.0

        if atr <= 0:
            continue

        direction = None
        # LONG: RSI가 과매도(35 미만)에서 회복 시작 (35 이상으로 교차)
        if rsi_prev < V2_MR_RSI_LONG and rsi_curr >= V2_MR_RSI_LONG:
            direction = 'LONG'
        # SHORT: RSI가 과매수(65 초과)에서 하락 시작 (65 이하로 교차)
        elif rsi_prev > V2_MR_RSI_SHORT and rsi_curr <= V2_MR_RSI_SHORT:
            direction = 'SHORT'

        if direction is None:
            continue

        # 다음 봉 시가에 진입
        if i + 1 >= len(df):
            continue
        entry_px = float(ohlcv_1d[i + 1][1])
        if entry_px <= 0:
            continue

        # 슬리피지 반영 / TP 고정 2R (SL ATR×1.5 → TP ATR×3.0)
        if direction == 'LONG':
            entry_eff = entry_px * (1 + BT_SLIPPAGE)
            sl_val    = entry_eff - atr * V2_MR_ATR_SL
            tp_val    = entry_eff + atr * V2_MR_ATR_SL * 2.0  # 2R 고정 TP
        else:
            entry_eff = entry_px * (1 - BT_SLIPPAGE)
            sl_val    = entry_eff + atr * V2_MR_ATR_SL
            tp_val    = entry_eff - atr * V2_MR_ATR_SL * 2.0  # 2R 고정 TP

        sl_dist = abs(entry_eff - sl_val)
        if sl_dist < 1e-9:
            continue

        fee_r = entry_eff * 2 * BT_FEE_TAKER / sl_dist

        position = {
            'direction': direction,
            'entry':     entry_eff,
            'sl':        sl_val,
            'sl_dist':   sl_dist,
            'tp':        tp_val,
            'fee_r':     fee_r,
            'entry_bar': i + 1,
            'regime':    regime,
        }

    return trades


# ══════════════════════════════════════════════════════════════
#  메인 실행 함수
# ══════════════════════════════════════════════════════════════

def run_v2_backtest(start_date: str = '2022-01-01',
                    module: str = 'all',
                    data_dir: Optional[str] = None,
                    output_dir: str = '.',
                    end_date: Optional[str] = None,
                    label: str = '') -> dict:
    """ATLAS v2 전체 백테스트 실행."""
    period_str = f'{start_date} ~ {end_date}' if end_date else f'{start_date} ~ 현재'
    tag = f'  [{label}]' if label else ''
    print(f'\n{"=" * 64}')
    print(f'  ATLAS v2 백테스트{tag}  |  기간: {period_str}  |  모듈: {module}')
    print(f'{"=" * 64}')

    # ── since_ms (EMA200·ADX 워밍업 250일 포함) ───────────────
    WARMUP_DAYS = 250
    start_dt     = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    fetch_from   = start_dt - timedelta(days=WARMUP_DAYS)
    since_ms     = int(fetch_from.timestamp() * 1000)
    print(f'  데이터 로드: {fetch_from.date()} ~ 현재 (워밍업 {WARMUP_DAYS}일 포함)')

    # ── 필요 심볼 결정 ─────────────────────────────────────────
    needed_4h: set = set()
    needed_1d: set = {'BTCUSDT'}   # 레짐맵용 BTC 1D 항상 필요
    if module in ('all', 'a'):
        needed_4h.update(V2_MA_SYMBOLS)
        needed_1d.update(V2_MA_SYMBOLS)   # EMA200 (백업용)
    if module in ('all', 'b'):
        needed_1d.update(V2_MR_SYMBOLS)

    # ── 데이터 로드 ────────────────────────────────────────────
    data: dict = {}
    dd = Path(data_dir) if data_dir else None

    for sym in sorted(needed_4h | needed_1d):
        print(f'\n  [{sym}] 로드...')

        if sym in needed_4h:
            key = f'{sym}_4h'
            if dd and (dd / f'{sym}_4h.csv').exists():
                data[key] = load_ohlcv_csv(str(dd / f'{sym}_4h.csv'))
                src = 'CSV'
            else:
                data[key] = fetch_ohlcv_public(sym, '4h', since_ms=since_ms)
                src = 'API'
            bars = data[key]
            if bars:
                s = datetime.fromtimestamp(bars[0][0]  / 1000, tz=timezone.utc).date()
                e = datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc).date()
                print(f'    4H ({src}): {len(bars)}봉  [{s} ~ {e}]')
            else:
                print(f'    4H: 로드 실패')

        if sym in needed_1d:
            key = f'{sym}_1d'
            if dd and (dd / f'{sym}_1d.csv').exists():
                data[key] = load_ohlcv_csv(str(dd / f'{sym}_1d.csv'))
                src = 'CSV'
            else:
                data[key] = fetch_ohlcv_public(sym, '1d', since_ms=since_ms)
                src = 'API'
            bars = data[key]
            if bars:
                s = datetime.fromtimestamp(bars[0][0]  / 1000, tz=timezone.utc).date()
                e = datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc).date()
                print(f'    1D ({src}): {len(bars)}봉  [{s} ~ {e}]')
            else:
                print(f'    1D: 로드 실패')

    # ── 레짐 맵 (BTC 1D 기준) ──────────────────────────────────
    print('\n  레짐 분류 맵 생성...')
    regime_map = build_regime_map(data.get('BTCUSDT_1d', []))
    print(f'    → {len(regime_map)}일 분류 완료')

    # 레짐 분포 출력
    from collections import Counter
    rc = Counter(regime_map.values())
    for r, cnt in sorted(rc.items(), key=lambda x: -x[1]):
        print(f'    {r:<22}: {cnt}일 ({cnt / len(regime_map) * 100:.1f}%)')

    # ── 전략별 시뮬레이션 ──────────────────────────────────────
    all_trades: list = []
    results:    dict = {}

    if module in ('all', 'a'):
        ma_all: list = []
        print('\n  [Module A - Trend Follow 4H]')
        for sym in V2_MA_SYMBOLS:
            k4h = f'{sym}_4h'
            if not data.get(k4h):
                print(f'    {sym}: 데이터 없음')
                continue
            print(f'    {sym}...')
            t, diag = backtest_ma(sym, data[k4h], regime_map,
                                  start_date, end_date)
            print(f'    → {len(t)}건')
            if diag:
                print_ma_diagnostics(sym, diag)
            ma_all.extend(t)
            results[f'ModuleA/{sym}'] = calc_metrics(t)
        if ma_all:
            results['ModuleA (합산)'] = calc_metrics(ma_all)
        all_trades.extend(ma_all)

    if module in ('all', 'b'):
        mr_all: list = []
        print('\n  [Module B - Mean Reversion 1D]')
        for sym in V2_MR_SYMBOLS:
            k1d = f'{sym}_1d'
            if not data.get(k1d):
                print(f'    {sym}: 데이터 없음')
                continue
            print(f'    {sym}...')
            t = backtest_mr(sym, data[k1d], regime_map,
                            start_date, end_date)
            print(f'    → {len(t)}건')
            mr_all.extend(t)
            results[f'ModuleB/{sym}'] = calc_metrics(t)
        if mr_all:
            results['ModuleB (합산)'] = calc_metrics(mr_all)
        all_trades.extend(mr_all)

    results['포트폴리오 (전체)'] = calc_metrics(all_trades)

    # ── 결과 출력 ──────────────────────────────────────────────
    print(f'\n{"=" * 64}')
    print(f'  ATLAS v2 결과 요약{tag}')
    print(f'{"=" * 64}')
    for name, m in results.items():
        if not m.get('total_trades'):
            print(f'\n  [{name}]  거래없음')
            continue
        print(f'\n  [{name}]')
        print(f'    거래수:{m["total_trades"]}  '
              f'승률:{m["win_rate"]}%  '
              f'PF:{m["profit_factor"]}  '
              f'총R:{m["total_r"]:+.2f}')
        print(f'    MDD:{m["max_dd_pct"]}%  '
              f'Sharpe:{m["sharpe"]}  '
              f'최종손익:{m["total_pnl_pct"]:+.2f}%')
        sl_c = m.get('sl_count', 0)
        tp_c = m.get('tp_count', 0)
        print(f'    TP:{tp_c}건  SL:{sl_c}건  최종자본:${m["final_equity"]:,.0f}')

    # ── JSON 저장 ───────────────────────────────────────────────
    ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_tag = f'_{label}' if label else ''
    out_path = Path(output_dir) / f'v2_backtest_{ts_str}{file_tag}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_at':     datetime.now().isoformat(),
        'label':      label,
        'start_date': start_date,
        'end_date':   end_date,
        'module':     module,
        'summary':    results,
        'trades':     [asdict(t) for t in all_trades],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n  결과 저장: {out_path}\n')
    return payload


def run_v2_walk_forward(module: str = 'all',
                        data_dir: Optional[str] = None,
                        output_dir: str = '.') -> None:
    """
    Walk-Forward 검증.
    IS : 2022-01-01 ~ 2024-06-30  (파라미터 적합 구간)
    OOS: 2024-07-01 ~ 현재          (미래 일반화 검증)
    """
    IS_START  = '2022-01-01'
    IS_END    = '2024-07-01'   # exclusive
    OOS_START = '2024-07-01'

    print('\n' + '=' * 64)
    print('  ATLAS v2 Walk-Forward 검증')
    print(f'  IS  : {IS_START} ~ 2024-06-30')
    print(f'  OOS : {OOS_START} ~ 현재')
    print('=' * 64)

    is_res  = run_v2_backtest(IS_START,  module, data_dir, output_dir,
                              end_date=IS_END,  label='IS')
    oos_res = run_v2_backtest(OOS_START, module, data_dir, output_dir,
                              label='OOS')

    # ── IS vs OOS 비교표 ──────────────────────────────────────
    print('\n' + '=' * 64)
    print('  IS vs OOS 핵심 비교')
    print('=' * 64)
    key_metrics = ['total_trades', 'win_rate', 'profit_factor',
                   'total_r', 'max_dd_pct', 'sharpe', 'total_pnl_pct']
    for key in is_res['summary']:
        is_m  = is_res['summary'].get(key,  {})
        oos_m = oos_res['summary'].get(key, {})
        if not is_m.get('total_trades') and not oos_m.get('total_trades'):
            continue
        print(f'\n  [{key}]')
        print(f'  {"지표":<20} {"IS":>10} {"OOS":>10}  {"판정":>6}')
        print(f'  {"-" * 50}')
        for k in key_metrics:
            iv = is_m.get(k,  'N/A')
            ov = oos_m.get(k, 'N/A')
            verdict = ''
            if k in ('profit_factor', 'win_rate', 'sharpe', 'total_pnl_pct', 'total_r'):
                try:
                    ratio = float(ov) / float(iv) if float(iv) > 0 else 0
                    verdict = 'PASS' if ratio >= 0.70 else 'WARN'
                except Exception:
                    verdict = ''
            iv_s = f'{iv:>10.2f}' if isinstance(iv, (int, float)) else f'{iv:>10}'
            ov_s = f'{ov:>10.2f}' if isinstance(ov, (int, float)) else f'{ov:>10}'
            print(f'  {k:<20} {iv_s} {ov_s}  {verdict:>6}')
    print()


# ══════════════════════════════════════════════════════════════
#  CLI 진입점
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ATLAS v2 백테스팅')
    parser.add_argument('--start',    default='2022-01-01',
                        help='시작일 YYYY-MM-DD (기본: 2022-01-01)')
    parser.add_argument('--end',      default=None,
                        help='종료일 YYYY-MM-DD (기본: 현재)')
    parser.add_argument('--module',   default='all',
                        choices=['all', 'a', 'b'],
                        help='실행 모듈 (all/a/b, 기본: all)')
    parser.add_argument('--data-dir', default=None,
                        help='로컬 CSV 디렉토리')
    parser.add_argument('--output',   default='.',
                        help='결과 저장 경로 (기본: 현재 디렉토리)')
    parser.add_argument('--wf',       action='store_true',
                        help='Walk-Forward 검증 실행')
    args = parser.parse_args()

    if args.wf:
        run_v2_walk_forward(args.module, args.data_dir, args.output)
    else:
        run_v2_backtest(args.start, args.module, args.data_dir, args.output,
                        end_date=args.end)


if __name__ == '__main__':
    main()

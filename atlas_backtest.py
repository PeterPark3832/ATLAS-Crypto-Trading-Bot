"""
ATLAS — 백테스팅 프레임워크
============================
Phoenix 4H (BTC/SOL/BNB) / Equinox 1D (ETH) / Bounce 1D (ETH) 전략을
과거 데이터로 바-바이-바(bar-by-bar) 시뮬레이션합니다.

특징:
  - 선행 편향(look-ahead bias) 없음: 지표를 전체 데이터로 1회 사전 계산 후 슬라이스 활용
  - 레짐 필터 적용: BTC 1D ADX/EMA200 기반 장세 분류
  - BEP·Trailing SL 시뮬레이션
  - 성과 지표: 승률, PF, Sharpe, MDD, 총 PnL
  - 결과를 JSON으로 저장

사용법:
  python atlas_backtest.py                         # Binance에서 데이터 로드 (퍼블릭 API)
  python atlas_backtest.py --start 2023-01-01      # 시작일 지정
  python atlas_backtest.py --strategy phoenix      # 전략 선택 (all/phoenix/equinox/bounce)
  python atlas_backtest.py --data-dir data/        # CSV 사용 (BTCUSDT_4h.csv 형식)
  python atlas_backtest.py --output results/       # 결과 저장 경로
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# atlas_config 임포트 전 더미 환경변수 설정
# (퍼블릭 OHLCV API는 인증 불필요; _req() 오류 방지)
for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'BACKTEST')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_config import (
    PHX_SYMBOLS, EQ_SYMBOLS, BN_SYMBOLS,
    PHX_SYMBOL_PARAMS,
    PHX_COOLDOWN_CANDLES, PHX_BEP_ENABLED, PHX_BEP_TRIGGER_RR,
    PHX_BLOCK_WEEKEND, PHX_DAILY_TREND_FILTER, PHX_DAILY_TREND_BUF,
    EQ_EMA_FAST, EQ_EMA_SLOW, EQ_RISK_PCT,
    EQ_ATR_FILTER_PCT, EQ_EMA_SLOPE_BARS, EQ_EMA200_SLOPE_BARS, EQ_MOMENTUM_BARS,
    BN_RISK_PCT,
    BT_FEE_TAKER, BT_SLIPPAGE,
)
from atlas_indicators import (
    calc_4h, get_signal_4h,
    calc_1d_eq, get_signal_eq,
    calc_1d_bn, get_signal_bn,
    calc_adx, _ohlcv_to_df,
)
from atlas_regime import classify_regime


# ══════════════════════════════════════════════════════════════
#  데이터 로드
# ══════════════════════════════════════════════════════════════

def fetch_ohlcv_public(symbol: str, timeframe: str,
                        since_ms: Optional[int] = None,
                        max_bars: int = 20000) -> list:
    """
    Binance 퍼블릭 API로 OHLCV 전체 페이지네이션 (인증 불필요).

    since_ms: 이 타임스탬프(ms) 이후 모든 봉을 가져옴.
              None이면 최근 1000봉만 반환.
    max_bars: 안전장치 (무한루프 방지)
    """
    try:
        import ccxt
        ex = ccxt.binance({'options': {'defaultType': 'future'}})
        ccxt_sym  = symbol.replace('USDT', '/USDT')
        all_data: list = []
        since = since_ms

        while len(all_data) < max_bars:
            batch = ex.fetch_ohlcv(ccxt_sym, timeframe, since=since, limit=1000)
            if not batch:
                break
            # 중복 제거 (이전 배치의 마지막 봉과 겹치는 경우)
            if all_data:
                batch = [b for b in batch if b[0] > all_data[-1][0]]
            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < 1000:
                break   # 더 이상 데이터 없음 (현재 시점 도달)
            since = batch[-1][0] + 1   # 다음 배치 시작점

        return all_data
    except Exception as e:
        print(f'  [오류] {symbol} {timeframe} 데이터 로드 실패: {e}')
        return []


def load_ohlcv_csv(path: str) -> list:
    """
    CSV에서 OHLCV 로드.
    컬럼: timestamp(ms 또는 datetime 문자열), open, high, low, close, volume
    """
    df = pd.read_csv(path)
    ts_col = next((c for c in ('timestamp', 'open_time', 'ts') if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f'타임스탬프 컬럼 없음: {path}')
    ts_ms = pd.to_datetime(df[ts_col])
    if ts_ms.dt.tz is None:
        ts_ms = ts_ms.dt.tz_localize('UTC')
    df['_ts'] = ts_ms.astype(np.int64) // 1_000_000
    return df[['_ts', 'open', 'high', 'low', 'close', 'volume']].values.tolist()


# ══════════════════════════════════════════════════════════════
#  레짐 맵
# ══════════════════════════════════════════════════════════════

def build_regime_map(btc_1d_ohlcv: list) -> dict:
    """
    날짜 → 레짐 문자열 딕셔너리 반환.
    key: 'YYYY-MM-DD'  value: 'TRENDING_UP' / 'TRENDING_DOWN' / ...
    """
    if len(btc_1d_ohlcv) < 60:
        return {}

    df    = _ohlcv_to_df(btc_1d_ohlcv)
    close = df['close']
    ema200 = close.ewm(span=200, adjust=False).mean()

    pc  = close.shift(1)
    tr  = pd.concat([
        df['high'] - df['low'],
        (df['high'] - pc).abs(),
        (df['low']  - pc).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    regime_map: dict = {}
    for i in range(50, len(df)):
        ts_val   = df['ts'].iloc[i]
        date_key = str(ts_val.date())
        window   = btc_1d_ohlcv[max(0, i - 50): i + 1]
        adx      = calc_adx(window, 14)
        btc_px   = float(close.iloc[i])
        ema_val  = float(ema200.iloc[i])
        atr_val  = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else 0.0
        atr_pct  = atr_val / btc_px if btc_px else 0.0
        regime_map[date_key] = classify_regime(adx, btc_px, ema_val, atr_pct)

    return regime_map


# ══════════════════════════════════════════════════════════════
#  성과 지표
# ══════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    symbol:    str
    strategy:  str
    direction: str
    mode:      str
    entry:     float
    exit_px:   float
    pnl_r:     float      # R-배수  (1.0 = 리스크 금액과 동일한 수익)
    risk_pct:  float      # 해당 거래의 리스크 비율
    reason:    str        # 'SL' or 'TP'
    entry_bar: int
    exit_bar:  int
    regime:    str = ''


def calc_metrics(trades: list, initial_equity: float = 10_000.0) -> dict:
    """거래 목록으로 종합 성과 지표 계산."""
    if not trades:
        return {'total_trades': 0}

    pnl_r_list = [t.pnl_r for t in trades]
    wins   = [r for r in pnl_r_list if r > 0]
    losses = [r for r in pnl_r_list if r <= 0]

    win_rate = len(wins) / len(pnl_r_list)
    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    pf_denom = abs(sum(losses))
    pf       = abs(sum(wins)) / pf_denom if pf_denom > 0 else 999.0
    total_r  = float(sum(pnl_r_list))

    # 자본 곡선 (실제 risk_pct × pnl_r 반영)
    equity = initial_equity
    equity_curve = [equity]
    peak_eq = equity
    max_dd  = 0.0
    for t in trades:
        equity += equity * t.risk_pct * t.pnl_r
        equity_curve.append(equity)
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq if peak_eq > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Sharpe (거래 단위 수익률 기반)
    rets = np.array([(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                     for i in range(1, len(equity_curve))])
    sharpe = 0.0
    if len(rets) > 1:
        std = rets.std()
        if std > 0:
            sharpe = float(rets.mean() / std * math.sqrt(len(rets)))

    return {
        'total_trades':  len(trades),
        'win_rate':      round(win_rate * 100, 1),
        'avg_win_r':     round(avg_win, 3),
        'avg_loss_r':    round(avg_loss, 3),
        'profit_factor': round(min(pf, 999.0), 2),
        'total_r':       round(total_r, 2),
        'max_dd_pct':    round(max_dd * 100, 2),
        'sharpe':        round(sharpe, 3),
        'final_equity':  round(equity, 2),
        'total_pnl_pct': round((equity - initial_equity) / initial_equity * 100, 2),
        'sl_count':      sum(1 for t in trades if t.reason == 'SL'),
        'tp_count':      sum(1 for t in trades if t.reason == 'TP'),
    }


# ══════════════════════════════════════════════════════════════
#  Phoenix 4H 백테스트
# ══════════════════════════════════════════════════════════════

from collections import defaultdict


def _bucket_pnl_r(buckets: dict, pnl_r: float) -> None:
    """R-배수를 버킷에 분류."""
    if   pnl_r < -1.5:  buckets['< -1.5R']  += 1
    elif pnl_r < -1.0:  buckets['-1.5~-1R'] += 1
    elif pnl_r < -0.5:  buckets['-1~-0.5R'] += 1
    elif pnl_r <  0.0:  buckets['-0.5~0R']  += 1
    elif pnl_r <  0.5:  buckets['0~0.5R']   += 1
    elif pnl_r <  1.0:  buckets['0.5~1R']   += 1
    elif pnl_r <  1.5:  buckets['1~1.5R']   += 1
    elif pnl_r <  2.0:  buckets['1.5~2R']   += 1
    else:               buckets['> 2R']      += 1


def print_phoenix_diagnostics(sym: str, diag: dict) -> None:
    """Phoenix 진단 결과 출력."""
    w = 56
    print(f'\n  {"─" * w}')
    print(f'  Phoenix 진단: {sym}')
    print(f'  {"─" * w}')

    total_bars = diag['total_bars']
    print(f'  총 처리 봉수: {total_bars}')

    print(f'\n  [필터별 거부 현황]')
    filters = [
        ('cooldown_active',    '쿨다운 중'),
        ('position_open',      '포지션 보유'),
        ('weekend_block',      '주말 차단'),
        ('crisis_regime',      'CRISIS 레짐'),
        ('no_signal',          '신호 없음 (get_signal_4h)'),
        ('regime_direction',   '레짐-방향 불일치'),
        ('ranging_short',      'RANGING+SHORT 차단'),
        ('ema200_filter',      'EMA200 일간 필터'),
        ('zero_entry',         '진입가 0 이하'),
    ]
    passed = total_bars
    for key, label in filters:
        cnt = diag['rejected'].get(key, 0)
        passed -= cnt
        pct = cnt / total_bars * 100 if total_bars else 0
        print(f'    {label:<26} : {cnt:>5}봉  ({pct:5.1f}%)')
    print(f'    {"진입 시도":<26} : {diag["entries"]:>5}봉  '
          f'({diag["entries"] / total_bars * 100:.1f}% 진입율)')

    print(f'\n  [레짐별 진입 분포]')
    regime_entries = diag['regime_entries']
    for r, cnt in sorted(regime_entries.items(), key=lambda x: -x[1]):
        print(f'    {r:<22} : {cnt:>4}건')

    print(f'\n  [방향별 진입 분포]')
    for d, cnt in diag['direction_entries'].items():
        print(f'    {d:<10}: {cnt}건')

    print(f'\n  [모드별 진입 분포]')
    for m, cnt in diag['mode_entries'].items():
        print(f'    {m:<10}: {cnt}건')

    print(f'\n  [신호 후 결과 R-배수 분포] (진입 {diag["entries"]}건)')
    buckets = diag['pnl_r_buckets']
    for label, cnt in buckets.items():
        bar = '#' * min(cnt, 40)
        print(f'    {label:<14} : {cnt:>4}건  {bar}')

    sl_cnt = diag['exit_sl']
    tp_cnt = diag['exit_tp']
    total_exits = sl_cnt + tp_cnt
    if total_exits:
        print(f'\n  [청산 통계]')
        print(f'    SL 청산: {sl_cnt}건  ({sl_cnt/total_exits*100:.1f}%)')
        print(f'    TP 청산: {tp_cnt}건  ({tp_cnt/total_exits*100:.1f}%)')
        if diag['avg_sl_duration']:
            print(f'    SL 평균 보유 봉: {diag["avg_sl_duration"]:.1f}')
        if diag['avg_tp_duration']:
            print(f'    TP 평균 보유 봉: {diag["avg_tp_duration"]:.1f}')
    print(f'  {"─" * w}')


def backtest_phoenix(symbol: str, ohlcv_4h: list, sym_1d_ohlcv: list,
                     regime_map: dict,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> tuple:
    """
    Phoenix 4H M1(눌림목) + M2(BB스퀴즈) 시뮬레이션.
    신호 감지 봉 다음 봉 시가에 진입, 봉의 고가/저가로 SL/TP 체크.
    sym_1d_ohlcv: 해당 심볼 자신의 1D OHLCV (EMA200 추세 필터용 — BTC가 아닌 심볼별 EMA200)
    Returns: (trades, diagnostics_dict)
    """
    if len(ohlcv_4h) < 230:
        print(f'    [스킵] Phoenix/{symbol}: 데이터 부족 ({len(ohlcv_4h)}봉)')
        return [], {}

    params    = PHX_SYMBOL_PARAMS[symbol]
    ema_entry = params['ema_entry']
    risk_pct  = params['risk_pct']

    # ── 진단 카운터 초기화 ─────────────────────────────────────
    diag = {
        'total_bars':       0,
        'entries':          0,
        'exit_sl':          0,
        'exit_tp':          0,
        'rejected':         defaultdict(int),
        'regime_entries':   defaultdict(int),
        'direction_entries': defaultdict(int),
        'mode_entries':     defaultdict(int),
        'pnl_r_buckets':    {
            '< -1.5R':  0,
            '-1.5~-1R': 0,
            '-1~-0.5R': 0,
            '-0.5~0R':  0,
            '0~0.5R':   0,
            '0.5~1R':   0,
            '1~1.5R':   0,
            '1.5~2R':   0,
            '> 2R':     0,
        },
        'sl_durations': [],
        'tp_durations': [],
        'avg_sl_duration': 0.0,
        'avg_tp_duration': 0.0,
    }

    # 전체 지표 사전 계산 (선행 편향 없음: EWM/rolling은 후행)
    df_full = calc_4h(ohlcv_4h, ema_entry)

    # 1D EMA200 사전 계산 (Phoenix 일간 추세 필터용)
    # 버그 수정: 기존에는 BTC 1D ohlcv를 모든 심볼에 사용 →
    #   SOL($100) vs BTC EMA200($40,000) 비교로 LONG 항상 차단됨.
    # 수정: 각 심볼 자신의 1D EMA200과 비교
    ema200_1d_map: dict = {}
    if PHX_DAILY_TREND_FILTER and sym_1d_ohlcv:
        _df1d   = _ohlcv_to_df(sym_1d_ohlcv)
        _ema200 = _df1d['close'].ewm(span=200, adjust=False).mean()
        for _i, _ts in enumerate(_df1d['ts']):
            ema200_1d_map[str(_ts.date())] = (
                float(_df1d['close'].iloc[_i]),
                float(_ema200.iloc[_i]),
            )

    start_ts = (int(datetime.strptime(start_date, '%Y-%m-%d')
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
                if start_date else 0)
    end_ts = (int(datetime.strptime(end_date, '%Y-%m-%d')
                  .replace(tzinfo=timezone.utc).timestamp() * 1000)
              if end_date else None)

    trades: list = []
    position: Optional[dict] = None
    cooldown  = 0

    for i in range(220, len(df_full) - 1):
        bar_ts  = int(ohlcv_4h[i][0])
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

        if cooldown > 0:
            cooldown -= 1

        # ── 포지션 관리 ────────────────────────────────────────
        if position is not None:
            d   = position['direction']
            sl  = position['sl']
            tp  = position['tp']
            ent = position['entry']
            rr  = position['rr']

            # BEP (손익분기 SL 이동)
            if PHX_BEP_ENABLED:
                if d == 'LONG':
                    bep_tgt = ent + (tp - ent) * PHX_BEP_TRIGGER_RR / max(rr, 0.01)
                    if bar_h >= bep_tgt and sl < ent:
                        position['sl'] = ent
                        sl = ent
                else:
                    bep_tgt = ent - (ent - tp) * PHX_BEP_TRIGGER_RR / max(rr, 0.01)
                    if bar_l <= bep_tgt and sl > ent:
                        position['sl'] = ent
                        sl = ent

            sl0 = position['sl_original']
            sl_dist = abs(ent - sl0) if abs(ent - sl0) > 0 else 1e-9
            fee_r   = position.get('fee_r', 0.0)
            duration = i - position['entry_bar']

            if d == 'LONG':
                if bar_l <= sl:
                    pnl_r = (sl - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy=f"PHX_{position['mode']}",
                        direction='LONG', mode=position['mode'],
                        entry=ent, exit_px=sl, pnl_r=pnl_r,
                        risk_pct=risk_pct, reason='SL',
                        entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_sl'] += 1
                    diag['sl_durations'].append(duration)
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    position = None; cooldown = PHX_COOLDOWN_CANDLES; continue
                if bar_h >= tp:
                    pnl_r = (tp - ent) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy=f"PHX_{position['mode']}",
                        direction='LONG', mode=position['mode'],
                        entry=ent, exit_px=tp, pnl_r=pnl_r,
                        risk_pct=risk_pct, reason='TP',
                        entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_tp'] += 1
                    diag['tp_durations'].append(duration)
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    position = None; cooldown = PHX_COOLDOWN_CANDLES; continue
            else:  # SHORT
                if bar_h >= sl:
                    pnl_r = (ent - sl) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy=f"PHX_{position['mode']}",
                        direction='SHORT', mode=position['mode'],
                        entry=ent, exit_px=sl, pnl_r=pnl_r,
                        risk_pct=risk_pct, reason='SL',
                        entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_sl'] += 1
                    diag['sl_durations'].append(duration)
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    position = None; cooldown = PHX_COOLDOWN_CANDLES; continue
                if bar_l <= tp:
                    pnl_r = (ent - tp) / sl_dist - fee_r
                    trades.append(TradeResult(
                        symbol=symbol, strategy=f"PHX_{position['mode']}",
                        direction='SHORT', mode=position['mode'],
                        entry=ent, exit_px=tp, pnl_r=pnl_r,
                        risk_pct=risk_pct, reason='TP',
                        entry_bar=position['entry_bar'], exit_bar=i,
                        regime=position['regime']))
                    diag['exit_tp'] += 1
                    diag['tp_durations'].append(duration)
                    _bucket_pnl_r(diag['pnl_r_buckets'], pnl_r)
                    position = None; cooldown = PHX_COOLDOWN_CANDLES; continue

        # ── 신호 체크 ──────────────────────────────────────────
        if position is not None:
            diag['rejected']['position_open'] += 1
            continue
        if cooldown > 0:
            diag['rejected']['cooldown_active'] += 1
            continue

        if PHX_BLOCK_WEEKEND and ts_dt.weekday() >= 5:
            diag['rejected']['weekend_block'] += 1
            continue

        regime = regime_map.get(dk, 'UNKNOWN')
        if regime == 'CRISIS':
            diag['rejected']['crisis_regime'] += 1
            continue

        sig = get_signal_4h(df_full.iloc[:i + 1], params)
        if sig['signal'] == 0:
            diag['rejected']['no_signal'] += 1
            continue

        direction = 'LONG' if sig['signal'] == 1 else 'SHORT'
        if direction == 'LONG'  and regime == 'TRENDING_DOWN':
            diag['rejected']['regime_direction'] += 1
            continue
        if direction == 'SHORT' and regime == 'TRENDING_UP':
            diag['rejected']['regime_direction'] += 1
            continue
        # P2: 횡보장 공매도 차단 — RANGING은 방향성 없어 SHORT 노이즈 거래 방지
        if direction == 'SHORT' and regime == 'RANGING':
            diag['rejected']['ranging_short'] += 1
            continue

        # 1D EMA200 필터
        if PHX_DAILY_TREND_FILTER and dk in ema200_1d_map:
            px_1d, e200 = ema200_1d_map[dk]
            buf = e200 * PHX_DAILY_TREND_BUF
            if direction == 'LONG'  and bar_c < e200 - buf:
                diag['rejected']['ema200_filter'] += 1
                continue
            if direction == 'SHORT' and bar_c > e200 + buf:
                diag['rejected']['ema200_filter'] += 1
                continue

        # 다음 봉 시가에 진입
        entry_px = float(ohlcv_4h[i + 1][1])
        if entry_px <= 0:
            diag['rejected']['zero_entry'] += 1
            continue

        # 슬리피지 반영 (시장가 진입): LONG은 불리한 방향(위)으로
        if direction == 'LONG':
            entry_px_eff = entry_px * (1 + BT_SLIPPAGE)
        else:
            entry_px_eff = entry_px * (1 - BT_SLIPPAGE)

        sl_dist = abs(bar_c - sig['sl'])
        if sl_dist < 1e-9:
            continue
        if direction == 'LONG':
            sl_f = entry_px_eff - sl_dist
            tp_f = entry_px_eff + sl_dist * sig['rr']
        else:
            sl_f = entry_px_eff + sl_dist
            tp_f = entry_px_eff - sl_dist * sig['rr']

        # 수수료 비용: 진입·청산 각 1회 테이커 → pnl_r에서 차감
        # fee_r = entry × 2×fee / sl_dist (R 단위)
        fee_r = entry_px_eff * 2 * BT_FEE_TAKER / sl_dist

        diag['entries'] += 1
        diag['regime_entries'][regime] += 1
        diag['direction_entries'][direction] += 1
        diag['mode_entries'][sig['mode']] += 1

        position = {
            'direction':   direction,
            'entry':       entry_px_eff,
            'sl':          sl_f,
            'sl_original': sl_f,
            'tp':          tp_f,
            'mode':        sig['mode'],
            'rr':          sig['rr'],
            'fee_r':       fee_r,
            'entry_bar':   i + 1,
            'regime':      regime,
        }

    # 평균 보유 봉수
    if diag['sl_durations']:
        diag['avg_sl_duration'] = sum(diag['sl_durations']) / len(diag['sl_durations'])
    if diag['tp_durations']:
        diag['avg_tp_duration'] = sum(diag['tp_durations']) / len(diag['tp_durations'])

    return trades, diag


# ══════════════════════════════════════════════════════════════
#  Equinox 1D 백테스트
# ══════════════════════════════════════════════════════════════

def backtest_equinox(symbol: str, ohlcv_1d: list, regime_map: dict,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> list:
    """
    Equinox 1D LONG 시뮬레이션.
    Trailing SL: 진입 이후 고가 갱신 시 동일 거리로 추적.
    """
    if len(ohlcv_1d) < 250:
        print(f'    [스킵] Equinox/{symbol}: 데이터 부족')
        return []

    # calc_1d_eq는 dropna 후 index reset → df의 ts/ohlcv 사용
    df_full = calc_1d_eq(ohlcv_1d, EQ_EMA_FAST, EQ_EMA_SLOW)

    # B. 주간/월간 추세 필터용 EMA 사전 계산 (1D 기준)
    #    EMA(50)  ≈ 10주 추세 (주간 EMA 프록시)
    #    EMA(200) ≈ 월간 장기 추세 프록시
    _close        = df_full['close']
    ema50_series  = _close.ewm(span=50,  adjust=False).mean()
    ema200_series = _close.ewm(span=200, adjust=False).mean()

    start_dt = (pd.Timestamp(start_date, tz='UTC') if start_date else None)
    end_dt   = (pd.Timestamp(end_date,   tz='UTC') if end_date   else None)

    trades: list = []
    position: Optional[dict] = None

    for i in range(20, len(df_full)):
        row    = df_full.iloc[i]
        ts_dt  = row['ts']
        if start_dt and ts_dt < start_dt:
            continue
        if end_dt and ts_dt >= end_dt:
            break

        dk    = str(ts_dt.date())
        bar_h = float(row['high'])
        bar_l = float(row['low'])
        bar_o = float(row['open'])
        bar_v = float(row['volume'])
        regime = regime_map.get(dk, 'UNKNOWN')

        # ── 포지션 관리 ────────────────────────────────────────
        if position is not None:
            ent     = position['entry']
            sl      = position['sl']
            sl_dist = position['sl_dist']
            tp      = position['tp']
            peak    = position['peak']
            fee_r   = position.get('fee_r', 0.0)

            # Trailing SL
            if bar_h > peak:
                position['peak'] = bar_h
                new_sl = bar_h - sl_dist
                if new_sl > sl:
                    position['sl'] = new_sl
                    sl = new_sl

            if bar_l <= sl:
                pnl_r = (sl - ent) / sl_dist - fee_r if sl_dist > 0 else -1.0
                trades.append(TradeResult(
                    symbol=symbol, strategy='EQ_LONG', direction='LONG', mode='EQ',
                    entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=EQ_RISK_PCT,
                    reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                    regime=position['regime']))
                position = None; continue

            if bar_h >= tp:
                pnl_r = (tp - ent) / sl_dist - fee_r if sl_dist > 0 else 0.0
                trades.append(TradeResult(
                    symbol=symbol, strategy='EQ_LONG', direction='LONG', mode='EQ',
                    entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=EQ_RISK_PCT,
                    reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                    regime=position['regime']))
                position = None; continue

        # ── 신호 체크 ──────────────────────────────────────────
        if position is None:
            # A. WEAK_TREND 진입 차단 — TRENDING_UP만 허용
            if regime in ('CRISIS', 'TRENDING_DOWN', 'RANGING', 'WEAK_TREND'):
                continue

            # B-1. 주간 추세 확인: 종가 > EMA(50)
            bar_c      = float(row['close'])
            ema50_val  = float(ema50_series.iloc[i])
            ema200_val = float(ema200_series.iloc[i])
            if bar_c < ema50_val:
                continue
            if bar_c < ema200_val:
                continue

            # P1-A. 고변동성 진입 차단: ATR/종가 > EQ_ATR_FILTER_PCT
            atr_val = float(row['atr']) if 'atr' in row.index else 0.0
            if atr_val > 0 and (atr_val / bar_c) > EQ_ATR_FILTER_PCT:
                continue

            # P1-B. EMA fast 기울기 확인: N봉 전보다 낮으면 모멘텀 감소 → 차단
            if i >= EQ_EMA_SLOPE_BARS and 'ema_fast' in df_full.columns:
                ema_fast_now  = float(row['ema_fast'])
                ema_fast_past = float(df_full.iloc[i - EQ_EMA_SLOPE_BARS]['ema_fast'])
                if ema_fast_now > 0 and ema_fast_now <= ema_fast_past:
                    continue

            # P1-C. ETH EMA200 기울기: 장기 추세 방향 확인 (BTC 레짐과 독립)
            if i >= EQ_EMA200_SLOPE_BARS:
                ema200_now  = float(ema200_series.iloc[i])
                ema200_past = float(ema200_series.iloc[i - EQ_EMA200_SLOPE_BARS])
                if ema200_now > 0 and ema200_now <= ema200_past:
                    continue

            # P1-D. ETH 단기 모멘텀: 최근 N봉 대비 종가 확인
            #   BTC 랠리에 ETH가 따라가지 못하는 구간(고점 추격) 차단
            if i >= EQ_MOMENTUM_BARS:
                close_now  = float(row['close'])
                close_past = float(df_full.iloc[i - EQ_MOMENTUM_BARS]['close'])
                if close_now <= close_past:
                    continue

            df_prev = df_full.iloc[:i]
            if len(df_prev) < 5:
                continue

            ok, ep, sl, tp, atr, _reason = get_signal_eq(df_prev, bar_o, bar_h, bar_v)
            if not ok:
                continue

            ep_eff  = ep * (1 + BT_SLIPPAGE)   # 시장가 진입 슬리피지
            sl_dist = ep_eff - sl
            fee_r   = ep_eff * 2 * BT_FEE_TAKER / max(sl_dist, 1e-9)
            position = {
                'entry':     ep_eff,
                'sl':        sl,
                'sl_dist':   max(sl_dist, 1e-9),
                'tp':        tp,
                'peak':      ep_eff,
                'fee_r':     fee_r,
                'entry_bar': i,
                'regime':    regime,
            }

    return trades


# ══════════════════════════════════════════════════════════════
#  Bounce 1D 백테스트
# ══════════════════════════════════════════════════════════════

def backtest_bounce(symbol: str, ohlcv_1d: list, regime_map: dict,
                    eq_trades: list,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> list:
    """
    Bounce 1D LONG 시뮬레이션.
    Equinox 포지션 활성 구간은 진입 불가 (동일 심볼 중복 방지).
    """
    if len(ohlcv_1d) < 50:
        print(f'    [스킵] Bounce/{symbol}: 데이터 부족')
        return []

    # Equinox 포지션이 열려있는 바 인덱스 집합
    eq_open: set = set()
    for t in eq_trades:
        eq_open.update(range(t.entry_bar, t.exit_bar + 1))

    df_full  = calc_1d_bn(ohlcv_1d)
    start_dt = (pd.Timestamp(start_date, tz='UTC') if start_date else None)
    end_dt   = (pd.Timestamp(end_date,   tz='UTC') if end_date   else None)

    trades: list = []
    position: Optional[dict] = None

    for i in range(20, len(df_full)):
        row   = df_full.iloc[i]
        ts_dt = row['ts']
        if start_dt and ts_dt < start_dt:
            continue
        if end_dt and ts_dt >= end_dt:
            break

        dk    = str(ts_dt.date())
        bar_h = float(row['high'])
        bar_l = float(row['low'])
        regime = regime_map.get(dk, 'UNKNOWN')

        # ── 포지션 관리 ────────────────────────────────────────
        if position is not None:
            ent     = position['entry']
            sl      = position['sl']
            sl_dist = abs(ent - sl)
            tp      = position['tp']
            fee_r   = position.get('fee_r', 0.0)

            if bar_l <= sl:
                pnl_r = (sl - ent) / sl_dist - fee_r if sl_dist > 0 else -1.0
                trades.append(TradeResult(
                    symbol=symbol, strategy='BN_LONG', direction='LONG', mode='BN',
                    entry=ent, exit_px=sl, pnl_r=pnl_r, risk_pct=BN_RISK_PCT,
                    reason='SL', entry_bar=position['entry_bar'], exit_bar=i,
                    regime=position['regime']))
                position = None; continue

            if bar_h >= tp:
                pnl_r = (tp - ent) / sl_dist - fee_r if sl_dist > 0 else 0.0
                trades.append(TradeResult(
                    symbol=symbol, strategy='BN_LONG', direction='LONG', mode='BN',
                    entry=ent, exit_px=tp, pnl_r=pnl_r, risk_pct=BN_RISK_PCT,
                    reason='TP', entry_bar=position['entry_bar'], exit_bar=i,
                    regime=position['regime']))
                position = None; continue

        # ── 신호 체크 ──────────────────────────────────────────
        if position is None:
            if regime in ('CRISIS', 'TRENDING_DOWN'):
                continue
            if i in eq_open:
                continue

            df_prev = df_full.iloc[:i]
            if len(df_prev) < 5:
                continue

            ok, ep, sl, tp, _atr, _reason = get_signal_bn(df_prev)
            if not ok:
                continue

            ep_eff  = ep * (1 + BT_SLIPPAGE)
            sl_dist = abs(ep_eff - sl)
            fee_r   = ep_eff * 2 * BT_FEE_TAKER / max(sl_dist, 1e-9)
            position = {
                'entry':     ep_eff,
                'sl':        sl,
                'tp':        tp,
                'fee_r':     fee_r,
                'entry_bar': i,
                'regime':    regime,
            }

    return trades


# ══════════════════════════════════════════════════════════════
#  메인 실행 함수
# ══════════════════════════════════════════════════════════════

def run_backtest(start_date: str = '2022-01-01',
                 strategy: str = 'all',
                 data_dir: Optional[str] = None,
                 output_dir: str = '.',
                 end_date: Optional[str] = None,
                 label: str = '') -> dict:
    """전체 백테스트 실행 후 결과 딕셔너리 반환."""
    period_str = f'{start_date} ~ {end_date}' if end_date else f'{start_date} ~ 현재'
    tag = f'  [{label}]' if label else ''
    print(f'\n{"=" * 60}')
    print(f'  ATLAS 백테스트{tag}  |  기간: {period_str}  |  전략: {strategy}')
    print(f'{"=" * 60}')

    # ── since_ms 계산 (워밍업 포함) ───────────────────────────
    # EMA200·ADX 워밍업 + 여유분 250일을 start_date 앞에 추가로 로드
    WARMUP_DAYS = 250
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    fetch_from_dt = start_dt - timedelta(days=WARMUP_DAYS)
    since_ms = int(fetch_from_dt.timestamp() * 1000)

    print(f'  데이터 로드 범위: {fetch_from_dt.date()} ~ 현재 (워밍업 {WARMUP_DAYS}일 포함)')

    # ── 데이터 로드 ────────────────────────────────────────────
    data: dict = {}
    needed: set = {'BTCUSDT'}  # 레짐용 BTC 1D 항상 필요
    if strategy in ('all', 'phoenix'):
        needed.update(PHX_SYMBOLS)
    if strategy in ('all', 'equinox', 'bounce'):
        needed.update(EQ_SYMBOLS)

    for sym in sorted(needed):
        print(f'\n  [{sym}] 데이터 로드...')
        dd = Path(data_dir) if data_dir else None

        if sym in PHX_SYMBOLS and strategy in ('all', 'phoenix'):
            key4h = f'{sym}_4h'
            if dd and (dd / f'{sym}_4h.csv').exists():
                data[key4h] = load_ohlcv_csv(str(dd / f'{sym}_4h.csv'))
                src = 'CSV'
            else:
                data[key4h] = fetch_ohlcv_public(sym, '4h', since_ms=since_ms)
                src = 'API'
            bars = data[key4h]
            if bars:
                actual_start = datetime.fromtimestamp(bars[0][0] / 1000, tz=timezone.utc).date()
                actual_end   = datetime.fromtimestamp(bars[-1][0] / 1000, tz=timezone.utc).date()
                print(f'    4H ({src}): {len(bars)}봉  [{actual_start} ~ {actual_end}]')
                if actual_start > start_dt.date():
                    print(f'    ⚠️  실제 시작일({actual_start})이 요청({start_date})보다 늦음 — 데이터 없음')
            else:
                print(f'    4H: 로드 실패')

        key1d = f'{sym}_1d'
        if dd and (dd / f'{sym}_1d.csv').exists():
            data[key1d] = load_ohlcv_csv(str(dd / f'{sym}_1d.csv'))
            src = 'CSV'
        else:
            data[key1d] = fetch_ohlcv_public(sym, '1d', since_ms=since_ms)
            src = 'API'
        bars1d = data[key1d]
        if bars1d:
            actual_start = datetime.fromtimestamp(bars1d[0][0] / 1000, tz=timezone.utc).date()
            actual_end   = datetime.fromtimestamp(bars1d[-1][0] / 1000, tz=timezone.utc).date()
            print(f'    1D ({src}): {len(bars1d)}봉  [{actual_start} ~ {actual_end}]')
        else:
            print(f'    1D: 로드 실패')

    # ── 레짐 맵 ────────────────────────────────────────────────
    print('\n  레짐 분류 맵 생성...')
    regime_map = build_regime_map(data.get('BTCUSDT_1d', []))
    print(f'    → {len(regime_map)}일 분류 완료')

    # ── 전략별 시뮬레이션 ──────────────────────────────────────
    all_trades: list = []
    results:    dict = {}

    if strategy in ('all', 'phoenix'):
        phx_all: list = []
        print('\n  [Phoenix 4H]')
        for sym in PHX_SYMBOLS:
            k4h = f'{sym}_4h'
            if not data.get(k4h):
                continue
            print(f'    {sym}...')
            t, diag = backtest_phoenix(sym, data[k4h],
                                       data.get(f'{sym}_1d', []),
                                       regime_map, start_date, end_date)
            print(f'    → {len(t)}건')
            if diag:
                print_phoenix_diagnostics(sym, diag)
            phx_all.extend(t)
            results[f'Phoenix/{sym}'] = calc_metrics(t)
        if phx_all:
            results['Phoenix (합산)'] = calc_metrics(phx_all)
        all_trades.extend(phx_all)

    eq_trades_for_bn: list = []
    if strategy in ('all', 'equinox'):
        print('\n  [Equinox 1D]')
        for sym in EQ_SYMBOLS:
            k1d = f'{sym}_1d'
            if not data.get(k1d):
                continue
            print(f'    {sym}...')
            t = backtest_equinox(sym, data[k1d], regime_map, start_date, end_date)
            print(f'    → {len(t)}건')
            eq_trades_for_bn.extend(t)
            results[f'Equinox/{sym}'] = calc_metrics(t)
            all_trades.extend(t)

    if strategy in ('all', 'bounce'):
        print('\n  [Bounce 1D]')
        for sym in BN_SYMBOLS:
            k1d = f'{sym}_1d'
            if not data.get(k1d):
                continue
            sym_eq = [t for t in eq_trades_for_bn if t.symbol == sym]
            print(f'    {sym}...')
            t = backtest_bounce(sym, data[k1d], regime_map, sym_eq, start_date, end_date)
            print(f'    → {len(t)}건')
            results[f'Bounce/{sym}'] = calc_metrics(t)
            all_trades.extend(t)

    results['포트폴리오 (전체)'] = calc_metrics(all_trades)

    # ── 결과 출력 ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('  백테스트 결과 요약')
    print(f'{"=" * 60}')
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
        print(f'    TP:{m["tp_count"]}건  SL:{m["sl_count"]}건  '
              f'최종자본:${m["final_equity"]:,.0f}')

    # ── JSON 저장 ───────────────────────────────────────────────
    ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_tag = f'_{label}' if label else ''
    out_path = Path(output_dir) / f'backtest_{ts_str}{file_tag}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_at':     datetime.now().isoformat(),
        'label':      label,
        'start_date': start_date,
        'end_date':   end_date,
        'strategy':   strategy,
        'summary':    results,
        'trades':     [asdict(t) for t in all_trades],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n  결과 저장: {out_path}\n')

    return payload


def run_walk_forward(strategy: str = 'all',
                     data_dir: Optional[str] = None,
                     output_dir: str = '.') -> None:
    """
    Walk-Forward 검증: IS(2022-01-01~2024-06-30) / OOS(2024-07-01~현재) 분리 실행.
    IS에서 파라미터 유효성 확인 → OOS에서 일반화 성능 검증.
    """
    IS_START  = '2022-01-01'
    IS_END    = '2024-07-01'   # exclusive
    OOS_START = '2024-07-01'

    print('\n' + '=' * 60)
    print('  Walk-Forward 검증')
    print(f'  IS  : {IS_START} ~ 2024-06-30  (파라미터 적용 구간)')
    print(f'  OOS : {OOS_START} ~ 현재         (미래 검증 구간)')
    print('=' * 60)

    is_result  = run_backtest(IS_START,  strategy, data_dir, output_dir,
                              end_date=IS_END,  label='IS')
    oos_result = run_backtest(OOS_START, strategy, data_dir, output_dir,
                              label='OOS')

    # ── IS vs OOS 비교 요약 ────────────────────────────────────
    print('\n' + '=' * 60)
    print('  IS vs OOS 핵심 비교')
    print('=' * 60)
    key_metrics = ['total_trades', 'win_rate', 'profit_factor',
                   'total_r', 'max_dd_pct', 'sharpe', 'total_pnl_pct']
    for strat_key in is_result['summary']:
        is_m  = is_result['summary'].get(strat_key,  {})
        oos_m = oos_result['summary'].get(strat_key, {})
        if not is_m.get('total_trades') and not oos_m.get('total_trades'):
            continue
        print(f'\n  [{strat_key}]')
        print(f'  {"지표":<18} {"IS":>10} {"OOS":>10}  {"판정":>6}')
        print(f'  {"-"*46}')
        for k in key_metrics:
            iv = is_m.get(k,  'N/A')
            ov = oos_m.get(k, 'N/A')
            # 판정: PF·승률·Sharpe·손익은 OOS ≥ IS×0.7 이면 통과
            verdict = ''
            if k in ('profit_factor', 'win_rate', 'sharpe', 'total_pnl_pct', 'total_r'):
                try:
                    ratio = float(ov) / float(iv) if float(iv) > 0 else 0
                    verdict = 'PASS' if ratio >= 0.70 else 'WARN'
                except Exception:
                    verdict = ''
            iv_str = f'{iv:>10.2f}' if isinstance(iv, (int, float)) else f'{iv:>10}'
            ov_str = f'{ov:>10.2f}' if isinstance(ov, (int, float)) else f'{ov:>10}'
            print(f'  {k:<18} {iv_str} {ov_str}  {verdict:>6}')
    print()


def main():
    parser = argparse.ArgumentParser(description='ATLAS 백테스팅 프레임워크')
    parser.add_argument('--start',    default='2022-01-01',
                        help='백테스트 시작일 YYYY-MM-DD (기본: 2022-01-01)')
    parser.add_argument('--end',      default=None,
                        help='백테스트 종료일 YYYY-MM-DD (기본: 현재)')
    parser.add_argument('--strategy', default='all',
                        choices=['all', 'phoenix', 'equinox', 'bounce'],
                        help='실행 전략 (기본: all)')
    parser.add_argument('--data-dir', default=None,
                        help='로컬 CSV 디렉토리 (없으면 Binance API 사용)')
    parser.add_argument('--output',   default='.',
                        help='결과 JSON 저장 경로 (기본: 현재 디렉토리)')
    parser.add_argument('--wf',       action='store_true',
                        help='Walk-Forward 검증 실행 (IS/OOS 분리)')
    args = parser.parse_args()

    if args.wf:
        run_walk_forward(args.strategy, args.data_dir, args.output)
    else:
        run_backtest(args.start, args.strategy, args.data_dir, args.output,
                     end_date=args.end)


if __name__ == '__main__':
    main()

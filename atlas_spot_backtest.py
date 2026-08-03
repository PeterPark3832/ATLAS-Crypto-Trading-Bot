"""
ATLAS Spot — 백테스트 + Walk-Forward 프레임워크
================================================
7개 전략을 전체 심볼 유니버스에 대해 비교 백테스트합니다.

특징:
  - 선행 편향 없음: 지표 사전 계산 후 bar-by-bar 슬라이스
  - 레짐 필터: BTC 1D ADX/EMA200 기반 장세별 전략 활성화
  - Walk-Forward: IS(2021~2023) / OOS(2024~현재)
  - 성과 지표: CAGR, Sharpe, Sortino, MDD, WR, PF
  - 출력: 콘솔 비교표 + JSON + CSV

사용법:
  python atlas_spot_backtest.py                        # 전체 실행
  python atlas_spot_backtest.py --strategy s3          # 단일 전략
  python atlas_spot_backtest.py --symbol BTCUSDT       # 단일 심볼
  python atlas_spot_backtest.py --wf                   # Walk-Forward
  python atlas_spot_backtest.py --start 2021-01-01
  python atlas_spot_backtest.py --top 10               # 상위 10개 심볼
  python atlas_spot_backtest.py --data-dir data/       # 로컬 CSV 사용
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'BACKTEST')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_spot_config import (
    BT_SPOT_FEE, BT_SPOT_SLIPPAGE, BT_INITIAL_EQ,
    BT_SLIPPAGE_BY_TIER, BT_TIER1_SYMBOLS, BT_TIER2_SYMBOLS,
    SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_KELLY_WR_THRESH, SPOT_KELLY_PF_THRESH, SPOT_KELLY_FRACTION,
    SPOT_MAX_SL_PCT, SPOT_MAX_COST_PER_R,
    SPOT_TRAIL_ENABLED,
    SPOT_HEALTH_MIN_TRADES, SPOT_HEALTH_PF_SOFT, SPOT_HEALTH_PF_HARD,
    SPOT_HEALTH_SOFT_SCALE,
    WF_IS_START, WF_IS_END, WF_OOS_START,
    WF_OOS_MIN_SHARPE, WF_OOS_MIN_PF,
    STRATEGY_NAMES, STRATEGY_TIMEFRAMES,
    REGIME_STRATEGY_MAP, WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    SPOT_DATA_DIR, SPOT_RESULTS_DIR,
    S3_COOLDOWN,
)
from atlas_spot_strategies import (
    CALC_FUNCS, SIGNAL_FUNCS, EXIT_CHECK_FUNCS,
)
from atlas_spot_main import trailing_sl   # 추적 손절 규칙 단일 출처(라이브와 공유)
from atlas_spot_universe import get_backtest_universe
from atlas_indicators import _ohlcv_to_df
from atlas_regime import classify_regime, REGIME_ADX_TREND, REGIME_ADX_WEAK, REGIME_CRISIS_ATR


# ── 인라인 유틸리티 (구 atlas_backtest 공유 코드) ─────────────────

def _fetch_futures(symbol: str, timeframe: str,
                   since_ms: Optional[int] = None,
                   max_bars: int = 20000) -> list:
    """Binance 스팟 퍼블릭 API로 OHLCV 전체 페이지네이션."""
    try:
        import ccxt
        ex = ccxt.binance({'options': {'defaultType': 'spot'}})
        ccxt_sym = symbol.replace('USDT', '/USDT')
        all_data: list = []
        since = since_ms
        while len(all_data) < max_bars:
            batch = ex.fetch_ohlcv(ccxt_sym, timeframe, since=since, limit=1000)
            if not batch:
                break
            if all_data:
                batch = [b for b in batch if b[0] > all_data[-1][0]]
            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1][0] + 1
        return all_data
    except Exception as e:
        print(f'  [오류] {symbol} {timeframe} 데이터 로드 실패: {e}')
        return []


def load_ohlcv_csv(path: str) -> list:
    """CSV에서 OHLCV 로드. 컬럼: timestamp(ms/datetime), open, high, low, close, volume"""
    df = pd.read_csv(path)
    ts_col = next((c for c in ('timestamp', 'open_time', 'ts') if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f'타임스탬프 컬럼 없음: {path}')
    if pd.api.types.is_integer_dtype(df[ts_col]) or pd.api.types.is_float_dtype(df[ts_col]):
        ts_ms = pd.to_datetime(df[ts_col].astype(int), unit='ms', utc=True)
    else:
        ts_ms = pd.to_datetime(df[ts_col], utc=True)
    if ts_ms.dt.tz is None:
        ts_ms = ts_ms.dt.tz_localize('UTC')
    # pandas datetime64 내부 정밀도(ns/us/ms)가 버전·생성경로에 따라 달라지므로
    # astype(int)에 의존하지 않고 epoch 차분으로 ms를 직접 계산한다.
    _epoch = pd.Timestamp('1970-01-01', tz='UTC')
    df['_ts'] = (ts_ms - _epoch) // pd.Timedelta('1ms')
    return df[['_ts', 'open', 'high', 'low', 'close', 'volume']].values.tolist()


def build_regime_map(btc_1d_ohlcv: list) -> dict:
    """날짜 → 레짐 문자열 딕셔너리 반환. key: 'YYYY-MM-DD'"""
    from atlas_indicators import calc_adx
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
        date_key = str(df['ts'].iloc[i].date())
        window   = btc_1d_ohlcv[max(0, i - 50): i + 1]
        adx      = calc_adx(window, 14)
        btc_px   = float(close.iloc[i])
        ema_val  = float(ema200.iloc[i])
        atr_val  = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else 0.0
        atr_pct  = atr_val / btc_px if btc_px else 0.0
        regime_map[date_key] = classify_regime(adx, btc_px, ema_val, atr_pct)
    return regime_map


@dataclass
class TradeResult:
    symbol:    str
    strategy:  str
    direction: str
    mode:      str
    entry:     float
    exit_px:   float
    pnl_r:     float
    risk_pct:  float
    reason:    str
    entry_bar: int
    exit_bar:  int
    regime:    str = ''


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


def _get_slippage(symbol: str) -> float:
    """심볼별 슬리피지 티어 반환 (백테스트 비용 현실화)."""
    if symbol in BT_TIER1_SYMBOLS:
        return BT_SLIPPAGE_BY_TIER['tier1']
    if symbol in BT_TIER2_SYMBOLS:
        return BT_SLIPPAGE_BY_TIER['tier2']
    return BT_SLIPPAGE_BY_TIER['tier3']


# ══════════════════════════════════════════════════════════════
#  데이터 로드 (스팟 전용)
# ══════════════════════════════════════════════════════════════

def fetch_ohlcv_spot(symbol: str, timeframe: str,
                     since_ms: Optional[int] = None,
                     max_bars: int = 30000) -> list:
    """Binance 현물(Spot) 퍼블릭 API로 OHLCV 가져오기."""
    try:
        import ccxt
        ex = ccxt.binance({'options': {'defaultType': 'spot'}})
        ccxt_sym = symbol.replace('USDT', '/USDT')
        all_data: list = []
        since = since_ms

        while len(all_data) < max_bars:
            batch = ex.fetch_ohlcv(ccxt_sym, timeframe, since=since, limit=1000)
            if not batch:
                break
            if all_data:
                batch = [b for b in batch if b[0] > all_data[-1][0]]
            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1][0] + 1
            time.sleep(0.1)   # API 부하 방지

        return all_data
    except Exception as e:
        print(f'  [오류] {symbol} {timeframe} 스팟 데이터 로드 실패: {e}')
        return []


def _load_or_fetch(symbol: str, timeframe: str,
                   since_ms: int, data_dir: Optional[Path]) -> list:
    """CSV 캐시 → API 순으로 데이터 로드."""
    if data_dir:
        tf_label = timeframe.replace('h', 'H').replace('d', 'D')
        csv_path = data_dir / f'{symbol}_{tf_label}.csv'
        if csv_path.exists():
            print(f'    [CSV] {csv_path.name} 로드')
            return load_ohlcv_csv(str(csv_path))

    print(f'    [API] {symbol} {timeframe} 다운로드 중...')
    data = fetch_ohlcv_spot(symbol, timeframe, since_ms=since_ms)
    if data and data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        tf_label = timeframe.replace('h', 'H').replace('d', 'D')
        csv_path = data_dir / f'{symbol}_{tf_label}.csv'
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(csv_path, index=False)
        print(f'    [CSV] {csv_path.name} 저장 완료')
    return data


def _since_ms(date_str: str) -> int:
    """YYYY-MM-DD → Unix 밀리초."""
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ══════════════════════════════════════════════════════════════
#  성과 지표
# ══════════════════════════════════════════════════════════════

def calc_spot_metrics(trades: list, initial_equity: float = BT_INITIAL_EQ,
                      start_date: str = '', end_date: str = '') -> dict:
    """스팟 거래 목록으로 종합 성과 지표 계산."""
    if not trades:
        return {'total_trades': 0}

    pnl_r_list = [t.pnl_r for t in trades]
    wins   = [r for r in pnl_r_list if r > 0]
    losses = [r for r in pnl_r_list if r <= 0]

    win_rate = len(wins) / len(pnl_r_list) if pnl_r_list else 0
    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    pf_denom = abs(sum(losses))
    pf       = abs(sum(wins)) / pf_denom if pf_denom > 0 else 999.0

    # 자본 곡선
    equity     = initial_equity
    eq_curve   = [equity]
    peak_eq    = equity
    max_dd     = 0.0
    drawdown_durations = []
    dd_start   = None

    for t in trades:
        equity += equity * t.risk_pct * t.pnl_r
        equity = max(equity, 0.01)
        eq_curve.append(equity)
        if equity > peak_eq:
            if dd_start is not None:
                drawdown_durations.append(len(eq_curve) - dd_start)
            dd_start = None
            peak_eq = equity
        else:
            if dd_start is None:
                dd_start = len(eq_curve)
            dd = (peak_eq - equity) / peak_eq
            max_dd = max(max_dd, dd)

    total_pnl_pct = (equity - initial_equity) / initial_equity * 100

    # CAGR 계산
    cagr = 0.0
    if start_date and end_date:
        try:
            dt_s = datetime.strptime(start_date, '%Y-%m-%d')
            dt_e = datetime.strptime(end_date, '%Y-%m-%d')
            years = max((dt_e - dt_s).days / 365.25, 0.1)
            if equity > 0:
                cagr = ((equity / initial_equity) ** (1 / years) - 1) * 100
        except Exception:
            pass

    # Sharpe (연율화)
    rets = np.array([
        (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]
        for i in range(1, len(eq_curve))
    ])
    sharpe = 0.0
    if len(rets) > 1:
        std = rets.std()
        if std > 0:
            sharpe = float(rets.mean() / std * math.sqrt(252))

    # Sortino (하방 변동성만)
    down_rets = rets[rets < 0]
    sortino = 0.0
    if len(down_rets) > 1:
        down_std = down_rets.std()
        if down_std > 0:
            sortino = float(rets.mean() / down_std * math.sqrt(252))

    # Calmar
    calmar = cagr / (max_dd * 100) if max_dd > 0 else 0.0

    hold_hours = [getattr(t, 'hold_hours', 0) for t in trades if hasattr(t, 'hold_hours')]

    return {
        'total_trades':    len(trades),
        'win_rate':        round(win_rate * 100, 1),
        'avg_win_r':       round(avg_win, 3),
        'avg_loss_r':      round(avg_loss, 3),
        'profit_factor':   round(min(pf, 999.0), 2),
        'max_dd_pct':      round(max_dd * 100, 2),
        'sharpe':          round(sharpe, 3),
        'sortino':         round(sortino, 3),
        'calmar':          round(calmar, 3),
        'cagr_pct':        round(cagr, 2),
        'total_pnl_pct':   round(total_pnl_pct, 2),
        'final_equity':    round(equity, 2),
        'sl_count':        sum(1 for t in trades if t.reason == 'SL'),
        'tp_count':        sum(1 for t in trades if t.reason in ('TP', 'BB_MID', 'BB_UPPER', 'DON_LOW')),
        'time_count':      sum(1 for t in trades if t.reason == 'TIME'),
        'avg_hold_hours':  round(float(np.mean(hold_hours)), 1) if hold_hours else 0.0,
    }


# ══════════════════════════════════════════════════════════════
#  바-바이-바 시뮬레이터
# ══════════════════════════════════════════════════════════════

@dataclass
class SpotTrade(TradeResult):
    hold_hours: float = 0.0
    cost_usdt:  float = 0.0


def backtest_strategy(
    strategy_id:  str,
    symbol:       str,
    ohlcv:        list,
    regime_map:   dict,
    start_date:   str,
    end_date:     str,
    risk_pct:     float = SPOT_BASE_RISK_PCT,
    initial_equity: float = BT_INITIAL_EQ,
) -> tuple[list, dict]:
    """
    단일 전략 × 단일 심볼 bar-by-bar 시뮬레이션.

    Returns:
        (trades: list[SpotTrade], diagnostics: dict)
    """
    if len(ohlcv) < 50:
        return [], {}

    # 시작/종료 타임스탬프 필터
    ts_start = _since_ms(start_date)
    ts_end   = _since_ms(end_date) if end_date else int(datetime.now(timezone.utc).timestamp() * 1000)

    # 지표 계산 (전체 데이터, 선행편향 없음 — 슬라이스 활용)
    calc_fn   = CALC_FUNCS[strategy_id]
    signal_fn = SIGNAL_FUNCS[strategy_id]
    exit_fn   = EXIT_CHECK_FUNCS.get(strategy_id)
    tf        = STRATEGY_TIMEFRAMES[strategy_id]
    hours_per_bar = 4 if tf == '4h' else 24

    df = calc_fn(ohlcv)

    # S1 Buy & Hold 특별 처리
    if strategy_id == 'S1':
        return _backtest_buyhold(symbol, df, ts_start, ts_end, risk_pct, initial_equity)

    trades   = []
    equity   = initial_equity
    position = None      # 현재 포지션 dict or None
    cooldown = 0         # 쿨다운 카운터 (S3용)
    diag     = defaultdict(int)
    diag['total_bars'] = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts  = int(row['ts'].timestamp() * 1000) if hasattr(row['ts'], 'timestamp') else int(row['ts'])

        if ts < ts_start:
            continue
        if ts > ts_end:
            break

        diag['total_bars'] += 1
        cur_close = float(row['close'])
        cur_high  = float(row['high'])
        cur_low   = float(row['low'])
        date_key  = str(row['ts'].date()) if hasattr(row['ts'], 'date') else ''

        # ── 포지션 관리 ────────────────────────────────────────
        if position is not None:
            bars_held = i - position['entry_bar']
            max_hold  = position.get('max_hold', 0)
            exit_type = position.get('exit_type', 'sl_tp')

            reason    = None

            slip = _get_slippage(symbol)

            # 추적 손절 (라이브 `trailing_sl`과 동일 규칙 — 패리티)
            # 순서 주의: **이번 봉의 SL 판정 전에** 갱신하면 같은 봉의 고가로
            # 올린 손절이 같은 봉 저가에 걸리는 선행편향이 생긴다.
            # 따라서 갱신은 이 봉의 청산 판정이 끝난 뒤(아래)에 수행한다.
            # SL 체크 (bar 저가로 SL 이탈)
            if cur_low <= position['sl']:
                reason    = 'SL'
                exit_price = min(position['sl'], cur_close) * (1 - slip)

            # TP 체크 (bar 고가로 TP 도달)
            elif position['tp'] > 0 and cur_high >= position['tp']:
                reason    = 'TP'
                exit_price = position['tp'] * (1 - slip)

            # 추가 청산 조건 체크 (크로스 등)
            elif exit_fn is not None:
                try:
                    if strategy_id == 'S6':
                        should_exit = exit_fn(df, i, position['entry_price'])
                    else:
                        should_exit = exit_fn(df, i)
                    if should_exit:
                        reason = 'CROSS'
                        exit_price = cur_close * (1 - slip)
                except Exception:
                    pass

            # 시간 기반 청산
            if reason is None and max_hold > 0 and bars_held >= max_hold:
                reason    = 'TIME'
                exit_price = cur_close * (1 - slip)

            if reason is not None:
                entry_price = position['entry_price']
                # R배수는 **진입 시점의 위험**으로 재야 한다. 추적 손절로
                # SL이 올라간 뒤 그 값을 쓰면 분모가 줄어 R이 부풀고,
                # 그 pnl_r이 Kelly·건강도·avg_r을 오염시킨다.
                sl_dist     = abs(entry_price - position.get('orig_sl', position['sl']))
                pnl_r       = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
                pnl_pct     = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                fee_cost    = exit_price * position['qty'] * BT_SPOT_FEE
                cost_usdt   = position['cost_usdt']

                equity      += equity * position['risk_pct'] * pnl_r - fee_cost
                equity       = max(equity, 0.01)

                t = SpotTrade(
                    symbol     = symbol,
                    strategy   = strategy_id,
                    direction  = 'LONG',
                    mode       = '',
                    entry      = entry_price,
                    exit_px    = exit_price,
                    pnl_r      = round(pnl_r, 4),
                    risk_pct   = position['risk_pct'],
                    reason     = reason,
                    entry_bar  = position['entry_bar'],
                    exit_bar   = i,
                    regime     = position.get('regime', ''),
                    hold_hours = round(bars_held * hours_per_bar, 1),
                    cost_usdt  = round(cost_usdt, 2),
                )
                trades.append(t)
                position = None
                if strategy_id == 'S3':
                    cooldown = S3_COOLDOWN
                diag['exits'] += 1
                continue

            # 이 봉에서 청산되지 않았다 → 고가로 peak 갱신 후 손절 추적.
            # (판정 이후에 갱신해야 같은 봉 고가로 올린 손절이 같은 봉
            #  저가에 걸리는 선행편향이 생기지 않는다)
            if SPOT_TRAIL_ENABLED:
                position['peak'] = max(position.get('peak', position['entry_price']),
                                       cur_high)
                position['sl'] = trailing_sl(
                    position['entry_price'], position['sl'], position['peak'],
                    abs(position['entry_price'] - position.get('orig_sl', position['sl'])))

        # ── 진입 조건 ────────────────────────────────────────
        if position is not None:
            continue   # 이미 포지션 있음
        if cooldown > 0:
            cooldown -= 1
            diag['cooldown_skip'] += 1
            continue

        # 레짐 확인
        regime = regime_map.get(date_key, 'WEAK_TREND')
        allowed = REGIME_STRATEGY_MAP.get(regime, [])
        if strategy_id not in allowed:
            diag['regime_block'] += 1
            continue

        if regime == 'WEAK_TREND':
            regime_scale = WEAK_TREND_RISK_SCALE
        elif regime == 'TRENDING_DOWN':
            regime_scale = TRENDING_DOWN_RISK_SCALE
        else:
            regime_scale = 1.0

        # 신호 확인 (이전봉까지의 데이터로만 판단, 현재봉 종가는 알 수 없음)
        try:
            sig = signal_fn(df, i - 1)
        except Exception:
            continue
        if sig['signal'] != 1:
            continue

        slip        = _get_slippage(symbol)
        entry_price = float(row['open']) * (1 + slip)  # 현재봉 시가 = 신호봉 다음봉 오픈
        sl          = sig['sl']
        tp          = sig['tp']
        sl_dist     = abs(entry_price - sl)

        if sl_dist <= 0 or sl >= entry_price:
            diag['invalid_sl'] += 1
            continue

        # ── 라이브 진입 필터 패리티 ─────────────────────────────
        # 라이브에만 있고 백테스트에 없으면 성과가 구조적으로 과대평가된다.
        #
        # ※ 남아있는 한계 — **포트폴리오 제약은 모델링하지 못한다.**
        #   이 함수는 (전략 × 심볼) 단위로 독립 실행되므로 동시 포지션 수
        #   상한(SPOT_MAX_POSITIONS), 슬롯당 최소 자본(SPOT_EQUITY_PER_SLOT),
        #   USDT 예비금(SPOT_RESERVE_PCT), 전략 간 자본 경합을 반영할 수 없다.
        #   즉 백테스트는 여전히 라이브보다 **낙관적**이며, 특히 신호가 몰리는
        #   구간에서 라이브는 일부 진입을 포기한다. 결과를 상한선으로 읽을 것.
        sl_pct = sl_dist / entry_price
        if sl_pct > SPOT_MAX_SL_PCT:
            diag['sl_too_wide'] = diag.get('sl_too_wide', 0) + 1
            continue
        # 비용 대비 엣지: 왕복비용이 1R을 얼마나 잠식하는지
        # (라이브 `_cost_edge_ok`와 동일한 부등식. 스프레드는 백테스트에
        #  호가 데이터가 없어 슬리피지 모델로 근사한다)
        cost_rate  = BT_SPOT_FEE * 2 + slip * 2
        cost_per_r = cost_rate / sl_pct
        if cost_per_r > SPOT_MAX_COST_PER_R:
            diag['cost_exceeds_edge'] = diag.get('cost_exceeds_edge', 0) + 1
            continue

        # 포지션 크기 계산 (스팟: 레버리지 없음)
        # Kelly 스케일링 (SPOT_KELLY_MIN_TRADES 건 이후부터 적용)
        kelly_scale = 1.0
        if len(trades) >= SPOT_KELLY_MIN_TRADES:
            recent = trades[-200:]
            wins_r   = [t.pnl_r for t in recent if t.pnl_r > 0]
            losses_r = [t.pnl_r for t in recent if t.pnl_r <= 0]
            if not wins_r:
                # 전패 구간 — 라이브와 동일하게 최소 스케일로 (기존에는 1.0을
                # 유지해 연패 중에도 정상 사이즈로 베팅하는 낙관 편향이 있었다)
                kelly_scale = SPOT_KELLY_SCALE_MIN
            if wins_r and losses_r:
                wr = len(wins_r) / len(recent)
                b  = abs(float(np.mean(wins_r))) / abs(float(np.mean(losses_r)))
                # half-Kelly (라이브 _get_kelly_scale과 패리티 유지)
                kelly_raw = (wr - (1 - wr) / b) * SPOT_KELLY_FRACTION if b > 0 else 0.0
                # 조건부 상한: WR>55% AND PF>1.5 시 KELLY_SCALE_MAX(2.0), 아니면 1.50
                gross_w = sum(abs(r) for r in wins_r)
                gross_l = sum(abs(r) for r in losses_r)
                pf = gross_w / gross_l if gross_l > 0 else 0.0
                k_max = SPOT_KELLY_SCALE_MAX if (wr >= SPOT_KELLY_WR_THRESH and
                                                  pf >= SPOT_KELLY_PF_THRESH) else 1.50
                kelly_scale = max(SPOT_KELLY_SCALE_MIN, min(k_max, kelly_raw))
        # 전략 건강도 자기교정 (라이브 `_get_strategy_health_scale` 패리티)
        # 최근 실적의 net PF가 나쁘면 감봉/차단된다. 이걸 빼면 백테스트는
        # 라이브가 실제로는 걸러냈을 구간까지 정상 사이즈로 거래한 셈이 된다.
        health_scale = 1.0
        if len(trades) >= SPOT_HEALTH_MIN_TRADES:
            _recent_h = trades[-200:]
            # 손익 기여도 = R배수 × 그 거래의 리스크 비중 (자본 대비 비율).
            # 라이브는 USD net PF를 쓰지만 백테스트에는 거래별 USD 손익이
            # 없으므로 비율로 근사한다 — 판정 자체의 패리티가 목적이다.
            _nets = [t.pnl_r * t.risk_pct for t in _recent_h]
            _gw = sum(n for n in _nets if n > 0)
            _gl = abs(sum(n for n in _nets if n < 0))
            _pf = _gw / _gl if _gl > 0 else float('inf')
            if _pf < SPOT_HEALTH_PF_HARD:
                diag['health_blocked'] = diag.get('health_blocked', 0) + 1
                continue
            if _pf < SPOT_HEALTH_PF_SOFT:
                health_scale = SPOT_HEALTH_SOFT_SCALE

        adj_risk_pct = risk_pct * regime_scale * kelly_scale * health_scale
        risk_usd     = equity * adj_risk_pct
        qty          = risk_usd / sl_dist
        cost_usdt    = qty * entry_price

        # 배분 한도 체크
        if cost_usdt > equity * SPOT_MAX_ALLOC_PCT:
            cost_usdt = equity * SPOT_MAX_ALLOC_PCT
            qty       = cost_usdt / entry_price
            adj_risk_pct = (qty * sl_dist) / equity

        fee_cost = cost_usdt * BT_SPOT_FEE
        equity  -= fee_cost  # 매수 수수료

        position = {
            'entry_price': entry_price,
            'sl':          sl,
            'orig_sl':     sl,      # R배수 분모 — 추적으로 sl이 올라가도 불변
            'peak':        entry_price,
            'tp':          tp,
            'qty':         qty,
            'cost_usdt':   cost_usdt,
            'risk_pct':    adj_risk_pct,
            'entry_bar':   i,
            'exit_type':   sig['exit_type'],
            'max_hold':    sig['max_hold'],
            'regime':      regime,
        }


        diag['entries'] += 1

    # 미청산 포지션은 마지막 종가로 강제 청산
    if position is not None and len(df) > 0:
        last_close  = float(df.iloc[-1]['close'])
        sl_dist     = abs(position['entry_price'] - position['sl'])
        pnl_r       = (last_close - position['entry_price']) / sl_dist if sl_dist > 0 else 0.0
        bars_held   = len(df) - 1 - position['entry_bar']
        t = SpotTrade(
            symbol     = symbol, strategy = strategy_id,
            direction  = 'LONG', mode = '',
            entry      = position['entry_price'], exit_px = last_close,
            pnl_r      = round(pnl_r, 4), risk_pct = position['risk_pct'],
            reason     = 'END', entry_bar = position['entry_bar'],
            exit_bar   = len(df) - 1, regime = position.get('regime', ''),
            hold_hours = round(bars_held * hours_per_bar, 1),
            cost_usdt  = round(position['cost_usdt'], 2),
        )
        trades.append(t)

    return trades, dict(diag)


def _backtest_buyhold(symbol, df, ts_start, ts_end,
                      risk_pct, initial_equity) -> tuple:
    """S1 Buy & Hold 특별 시뮬레이션."""
    trades = []
    df_filt = df[(df['ts'].apply(lambda x: int(x.timestamp() * 1000)) >= ts_start) &
                 (df['ts'].apply(lambda x: int(x.timestamp() * 1000)) <= ts_end)]
    if len(df_filt) < 2:
        return trades, {}

    entry_price = float(df_filt.iloc[0]['close'])
    exit_price  = float(df_filt.iloc[-1]['close'])
    pnl_r       = (exit_price - entry_price) / (entry_price * 0.1) if entry_price > 0 else 0
    hold_days   = (df_filt.index[-1] - df_filt.index[0])

    t = SpotTrade(
        symbol     = symbol, strategy = 'S1',
        direction  = 'LONG', mode = '',
        entry      = entry_price, exit_px = exit_price,
        pnl_r      = round((exit_price - entry_price) / entry_price, 4),
        risk_pct   = 1.0,   # 전액 투자 = pnl_r이 직접 수익률
        reason     = 'END', entry_bar = 0,
        exit_bar   = len(df_filt) - 1, regime = '',
        hold_hours = float(len(df_filt) * 24),
        cost_usdt  = initial_equity,
    )
    # Buy & Hold 단순 수익률로 재계산
    t.pnl_r = (exit_price - entry_price) / entry_price
    trades.append(t)
    return trades, {'total_bars': len(df_filt), 'entries': 1}


# ══════════════════════════════════════════════════════════════
#  전체 전략 × 심볼 백테스트 실행
# ══════════════════════════════════════════════════════════════

def run_spot_backtest(
    strategies:  list[str],
    symbols:     list[str],
    start_date:  str,
    end_date:    str,
    data_dir:    Optional[Path] = None,
    risk_pct:    float = SPOT_BASE_RISK_PCT,
) -> dict:
    """
    전략 × 심볼 조합 전체 백테스트.
    Returns: {strategy_id: {symbol: {'trades': [...], 'metrics': {...}}}}
    """
    since_ms = _since_ms(start_date)
    # BTC 1D (레짐 맵용, 스팟 데이터 사용)
    print(f'\n[레짐맵] BTC 1D 데이터 로드...')
    btc_1d = _load_or_fetch('BTCUSDT', '1d', since_ms, data_dir)
    regime_map = build_regime_map(btc_1d) if btc_1d else {}
    print(f'  레짐맵 생성: {len(regime_map)}일')

    # 필요한 타임프레임별 심볼 목록
    need_4h = [s for s in strategies if STRATEGY_TIMEFRAMES.get(s) == '4h']
    need_1d = [s for s in strategies if STRATEGY_TIMEFRAMES.get(s) == '1d']

    # 데이터 사전 로드
    ohlcv_4h: dict = {}
    ohlcv_1d: dict = {}

    if need_4h:
        print(f'\n[데이터] 4H 심볼 로드 ({len(symbols)}개)...')
        for sym in symbols:
            data = _load_or_fetch(sym, '4h', since_ms, data_dir)
            if data:
                ohlcv_4h[sym] = data

    if need_1d:
        print(f'\n[데이터] 1D 심볼 로드 ({len(symbols)}개)...')
        for sym in symbols:
            if sym in ohlcv_1d:
                continue
            data = _load_or_fetch(sym, '1d', since_ms, data_dir)
            if data:
                ohlcv_1d[sym] = data

    # 백테스트 실행
    results = {}
    for strategy_id in strategies:
        tf      = STRATEGY_TIMEFRAMES.get(strategy_id, '1d')
        ohlcv_d = ohlcv_4h if tf == '4h' else ohlcv_1d
        results[strategy_id] = {}
        all_trades = []

        print(f'\n[{strategy_id}] {STRATEGY_NAMES[strategy_id]} 백테스트 ({tf})...')
        for sym in symbols:
            ohlcv = ohlcv_d.get(sym, [])
            if not ohlcv:
                continue
            trades, diag = backtest_strategy(
                strategy_id, sym, ohlcv, regime_map,
                start_date, end_date, risk_pct
            )
            metrics = calc_spot_metrics(trades, BT_INITIAL_EQ, start_date, end_date)
            results[strategy_id][sym] = {
                'trades':  [asdict(t) for t in trades],
                'metrics': metrics,
                'diag':    diag,
            }
            all_trades.extend(trades)
            n = len(trades)
            pnl = metrics.get('total_pnl_pct', 0)
            wr  = metrics.get('win_rate', 0)
            print(f'    {sym:<14} {n:>4}건  PnL {pnl:+6.1f}%  WR {wr:.0f}%')

        # 전략 합산 지표
        combined = calc_spot_metrics(all_trades, BT_INITIAL_EQ, start_date, end_date)
        results[strategy_id]['_combined'] = combined
        print(f'  → 합계: {len(all_trades)}건  '
              f'PnL {combined.get("total_pnl_pct", 0):+.1f}%  '
              f'Sharpe {combined.get("sharpe", 0):.2f}  '
              f'MDD {combined.get("max_dd_pct", 0):.1f}%')

    return results


# ══════════════════════════════════════════════════════════════
#  Walk-Forward 검증
# ══════════════════════════════════════════════════════════════

def run_walk_forward(
    strategies:  list[str],
    symbols:     list[str],
    data_dir:    Optional[Path] = None,
    rolling:     bool = False,
) -> dict:
    """
    IS(2021~2023) / OOS(2024~현재) Walk-Forward 검증.
    rolling=True: 롤링 윈도우 4단계 추가.
    """
    wf_results = {}

    splits = [
        ('IS',  WF_IS_START,  WF_IS_END),
        ('OOS', WF_OOS_START, datetime.now(timezone.utc).strftime('%Y-%m-%d')),
    ]

    if rolling:
        splits += [
            ('R1_IS',  '2021-01-01', '2022-06-30'),
            ('R1_OOS', '2022-07-01', '2022-12-31'),
            ('R2_IS',  '2021-01-01', '2022-12-31'),
            ('R2_OOS', '2023-01-01', '2023-06-30'),
            ('R3_IS',  '2021-01-01', '2023-06-30'),
            ('R3_OOS', '2023-07-01', '2023-12-31'),
            ('R4_IS',  '2021-01-01', '2023-12-31'),
            ('R4_OOS', '2024-01-01', datetime.now(timezone.utc).strftime('%Y-%m-%d')),
        ]

    for label, start, end in splits:
        print(f'\n[Walk-Forward] {label}: {start} ~ {end}')
        res = run_spot_backtest(strategies, symbols, start, end, data_dir)
        wf_results[label] = {
            sid: res[sid].get('_combined', {}) for sid in strategies
        }

    return wf_results


# ══════════════════════════════════════════════════════════════
#  결과 출력
# ══════════════════════════════════════════════════════════════

def print_comparison_report(results: dict, start_date: str, end_date: str,
                             symbols: list[str]) -> None:
    """전략 비교표를 콘솔에 출력."""
    w = 100
    print(f'\n{"═" * w}')
    print(f'  ATLAS Spot Backtest Report')
    print(f'  기간: {start_date} ~ {end_date}  |  심볼: {len(symbols)}개')
    print(f'{"═" * w}')

    headers = ['전략', '심볼수', '거래수', 'WR%', 'PF', 'MDD%', 'Sharpe', 'Sortino', 'CAGR%', '총PnL%']
    col_w   = [22, 6, 6, 6, 5, 6, 7, 7, 7, 7]
    header  = '  ' + '  '.join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header)
    print(f'  {"─" * (sum(col_w) + len(col_w) * 2)}')

    for sid, sym_results in results.items():
        combined = sym_results.get('_combined', {})
        if not combined or combined.get('total_trades', 0) == 0:
            continue
        n_syms = sum(1 for k, v in sym_results.items()
                     if k != '_combined' and v.get('metrics', {}).get('total_trades', 0) > 0)
        row = [
            STRATEGY_NAMES.get(sid, sid),
            str(n_syms),
            str(combined.get('total_trades', 0)),
            f"{combined.get('win_rate', 0):.1f}",
            f"{combined.get('profit_factor', 0):.2f}",
            f"{combined.get('max_dd_pct', 0):.1f}",
            f"{combined.get('sharpe', 0):.2f}",
            f"{combined.get('sortino', 0):.2f}",
            f"{combined.get('cagr_pct', 0):+.1f}",
            f"{combined.get('total_pnl_pct', 0):+.1f}",
        ]
        line = '  ' + '  '.join(v.ljust(w) for v, w in zip(row, col_w))
        print(line)

    print(f'{"═" * w}\n')


def print_wf_report(wf_results: dict, strategies: list[str]) -> None:
    """Walk-Forward 검증 결과 출력."""
    w = 80
    print(f'\n{"═" * w}')
    print(f'  Walk-Forward 검증 결과 (IS: {WF_IS_START}~{WF_IS_END} / OOS: {WF_OOS_START}~현재)')
    print(f'{"═" * w}')

    is_data  = wf_results.get('IS', {})
    oos_data = wf_results.get('OOS', {})

    for sid in strategies:
        is_m  = is_data.get(sid, {})
        oos_m = oos_data.get(sid, {})
        if not is_m or not oos_m:
            continue

        is_sharpe = is_m.get('sharpe', 0)
        oos_sharpe = oos_m.get('sharpe', 0)
        is_pf      = is_m.get('profit_factor', 0)
        oos_pf     = oos_m.get('profit_factor', 0)

        ratio_sharpe = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
        ratio_pf     = oos_pf / is_pf if is_pf > 0 else 0

        pass_sharpe = oos_sharpe >= WF_OOS_MIN_SHARPE
        pass_pf     = oos_pf >= WF_OOS_MIN_PF
        pass_pnl    = oos_m.get('total_pnl_pct', 0) >= 0
        verdict     = '✅ PASS' if (pass_sharpe and pass_pf and pass_pnl) else '❌ FAIL'

        print(f'\n  [{sid}] {STRATEGY_NAMES.get(sid, sid)}  {verdict}')
        print(f'  {"지표":<20} {"IS":>10} {"OOS":>10} {"비율":>8}  {"판정":>6}')
        print(f'  {"─" * 60}')
        metrics_rows = [
            ('거래수',       is_m.get('total_trades', 0), oos_m.get('total_trades', 0), None, None),
            ('승률%',        is_m.get('win_rate', 0),      oos_m.get('win_rate', 0),      None, None),
            ('Profit Factor', is_pf,                        oos_pf,                        ratio_pf, pass_pf),
            ('Sharpe',        is_sharpe,                    oos_sharpe,                    ratio_sharpe, pass_sharpe),
            ('CAGR%',         is_m.get('cagr_pct', 0),     oos_m.get('cagr_pct', 0),     None, pass_pnl),
            ('MDD%',          is_m.get('max_dd_pct', 0),   oos_m.get('max_dd_pct', 0),   None, None),
        ]
        for name, is_val, oos_val, ratio, passed in metrics_rows:
            r_str = f'{ratio:.2f}x' if ratio is not None else '  —  '
            p_str = '✅' if passed else ('❌' if passed is not None else ' ')
            print(f'  {name:<20} {is_val:>10.2f} {oos_val:>10.2f} {r_str:>8}  {p_str:>6}')

    print(f'\n{"═" * w}\n')


# ══════════════════════════════════════════════════════════════
#  저장
# ══════════════════════════════════════════════════════════════

def save_results(results: dict, wf_results: dict,
                 strategies: list[str], symbols: list[str],
                 start_date: str, end_date: str) -> None:
    """결과를 JSON + CSV로 저장."""
    SPOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    # JSON
    summary = {
        'run_at':     datetime.now(timezone.utc).isoformat(),
        'start_date': start_date,
        'end_date':   end_date,
        'universe':   symbols,
        'strategies': {},
    }
    for sid in strategies:
        combined = results.get(sid, {}).get('_combined', {})
        summary['strategies'][sid] = {
            'name':    STRATEGY_NAMES.get(sid, sid),
            'metrics': combined,
            'wf_is':   wf_results.get('IS', {}).get(sid, {}),
            'wf_oos':  wf_results.get('OOS', {}).get(sid, {}),
        }

    json_path = SPOT_RESULTS_DIR / f'spot_backtest_{ts}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  [저장] {json_path}')

    # CSV (전체 거래 내역)
    all_trades_rows = []
    for sid, sym_results in results.items():
        for sym, data in sym_results.items():
            if sym == '_combined':
                continue
            for t in data.get('trades', []):
                row = dict(t)
                row['strategy_name'] = STRATEGY_NAMES.get(sid, sid)
                all_trades_rows.append(row)

    if all_trades_rows:
        csv_path = SPOT_RESULTS_DIR / f'spot_trades_{ts}.csv'
        pd.DataFrame(all_trades_rows).to_csv(csv_path, index=False)
        print(f'  [저장] {csv_path}  ({len(all_trades_rows)}건)')


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ATLAS Spot 백테스트')
    parser.add_argument('--start',    default='2021-01-01', help='백테스트 시작일')
    parser.add_argument('--end',      default=datetime.now().strftime('%Y-%m-%d'), help='종료일')
    parser.add_argument('--strategy', default='all',
                        help='전략 ID (s1~s7, all). 쉼표 구분 가능')
    parser.add_argument('--symbol',   default='', help='단일 심볼 (예: BTCUSDT)')
    parser.add_argument('--top',      type=int, default=0, help='상위 N개 심볼')
    parser.add_argument('--extended', action='store_true', help='확장 유니버스 사용')
    parser.add_argument('--wf',       action='store_true', help='Walk-Forward 검증')
    parser.add_argument('--rolling',  action='store_true', help='롤링 Walk-Forward')
    parser.add_argument('--data-dir', default='', help='로컬 CSV 디렉토리')
    parser.add_argument('--save',     action='store_true', default=True, help='결과 저장')
    parser.add_argument('--risk',     type=float, default=SPOT_BASE_RISK_PCT, help='거래당 리스크 %%')
    args = parser.parse_args()

    # 전략 선택
    all_strategies = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    if args.strategy == 'all':
        strategies = all_strategies
    else:
        strategies = [s.upper() for s in args.strategy.split(',')]

    # 심볼 선택
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = get_backtest_universe(extended=args.extended)
        if args.top > 0:
            symbols = symbols[:args.top]

    # 데이터 디렉토리
    data_dir = Path(args.data_dir) if args.data_dir else SPOT_DATA_DIR

    print(f'\n{"═" * 60}')
    print(f'  ATLAS Spot Backtest')
    print(f'  기간: {args.start} ~ {args.end}')
    print(f'  전략: {strategies}')
    print(f'  심볼: {len(symbols)}개')
    print(f'  리스크: {args.risk * 100:.1f}%/거래')
    print(f'{"═" * 60}')

    # 전체 백테스트
    results = run_spot_backtest(
        strategies, symbols, args.start, args.end, data_dir, args.risk
    )
    print_comparison_report(results, args.start, args.end, symbols)

    # Walk-Forward
    wf_results = {}
    if args.wf:
        wf_results = run_walk_forward(
            strategies, symbols, data_dir, rolling=args.rolling
        )
        print_wf_report(wf_results, strategies)

    # 저장
    if args.save:
        save_results(results, wf_results, strategies, symbols, args.start, args.end)

    return results, wf_results


if __name__ == '__main__':
    main()

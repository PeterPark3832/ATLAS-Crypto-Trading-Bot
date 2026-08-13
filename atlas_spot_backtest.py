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
import bisect
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'BACKTEST')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_spot_config import (
    BT_SPOT_FEE, BT_INITIAL_EQ,
    BT_SLIPPAGE_BY_TIER, BT_TIER1_SYMBOLS, BT_TIER2_SYMBOLS,
    SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT, SPOT_MIN_ORDER_USDT,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_KELLY_WR_THRESH, SPOT_KELLY_PF_THRESH, SPOT_KELLY_FRACTION,
    SPOT_MAX_SL_PCT, SPOT_MAX_COST_PER_R,
    UNIVERSE_MOMENTUM_DAYS,
    MOMENTUM_RS_GATE_STRATS, MOMENTUM_RS_GATE_PCT,
    MOMENTUM_TOP_TIER_PCT, MOMENTUM_TOP_RISK_MULT,
    SPOT_TRAIL_ENABLED,
    SPOT_HEALTH_MIN_TRADES, SPOT_HEALTH_PF_SOFT, SPOT_HEALTH_PF_HARD,
    SPOT_HEALTH_SOFT_SCALE,
    SPOT_LEARN_ENABLED, SPOT_LEARN_HALF_LIFE_DAYS, SPOT_LEARN_MIN_INFO,
    SPOT_LEARN_UNPROVEN_SCALE, SPOT_LEARN_FLOOR, SPOT_LEARN_MAX_SCALE,
    SPOT_LEARN_GAIN, SPOT_LEARN_COST_PER_R, SPOT_LEARN_REFRESH_MIN,
    WF_IS_START, WF_IS_END, WF_OOS_START,
    WF_OOS_MIN_SHARPE, WF_OOS_MIN_PF,
    STRATEGY_NAMES, STRATEGY_TIMEFRAMES,
    REGIME_STRATEGY_MAP, WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    SPOT_DATA_DIR, SPOT_RESULTS_DIR,
    S3_COOLDOWN, S5_SL_COOLDOWN_BARS,
    FUNDING_LONG_BLOCK, FUNDING_SHORT_BOOST, FUNDING_APPLY_STRATS,
)
from atlas_spot_strategies import (
    CALC_FUNCS, SIGNAL_FUNCS, EXIT_CHECK_FUNCS,
)
# 추적 손절 규칙 단일 출처(라이브와 공유). atlas_rules 는 leaf 라서
# 이 import 는 부수효과가 없다 — 예전처럼 atlas_spot_main 을 거치면
# 백테스트를 import만 해도 라이브 봇의 로그 핸들러·mkdir가 실행됐다.
from atlas_rules import trailing_sl
from atlas_spot_universe import get_backtest_universe
from atlas_indicators import _ohlcv_to_df
from atlas_regime import classify_regime, REGIME_BTC_LOOKBACK


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


def build_regime_map(btc_1d_ohlcv: list, btc_4h_ohlcv: list | None = None) -> dict:
    """날짜 → 레짐 문자열 딕셔너리 반환. key: 'YYYY-MM-DD'

    **라이브(`atlas_regime.update_regime`)와 같은 입력을 넘겨야 한다.**
    기존에는 adx_4h·adx_slope를 빼고 호출해서 백테스트에서는
    MICRO_RANGING과 'ADX 하락 중 → WEAK_TREND 강등'이 아예 발생하지
    않았다. 같은 날짜를 라이브는 MICRO_RANGING(전 전략 차단) 또는
    WEAK_TREND(리스크 0.5배)로 보는데 백테스트는 TRENDING_UP으로 봤다는
    뜻이라, 검증한 레짐 경로를 실계좌가 따라가지 않았다.

    btc_4h_ohlcv를 주면 4H ADX까지 라이브와 동일하게 반영한다.
    """
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

    # 날짜별 4H ADX (해당 일자까지의 4H 봉만 사용 — 선행편향 방지)
    adx4h_by_date: dict = {}
    if btc_4h_ohlcv:
        df4 = _ohlcv_to_df(btc_4h_ohlcv)
        dates4 = df4['ts'].dt.date.astype(str).values
        for j in range(50, len(df4)):
            # 윈도우 길이는 라이브와 **같은 상수**를 써야 한다 — 아래 1D 주석 참조
            _w4 = btc_4h_ohlcv[max(0, j - REGIME_BTC_LOOKBACK + 1): j + 1]
            adx4h_by_date[dates4[j]] = calc_adx(_w4, 14)

    regime_map: dict = {}
    adx_hist: list = []
    for i in range(50, len(df)):
        date_key = str(df['ts'].iloc[i].date())
        # ⚠️ 윈도우 **길이**가 라이브와 같아야 한다.
        # Wilder ADX는 재귀 평활이라 워밍업 길이가 값을 바꾼다. 예전에는
        # 여기가 51봉(i-50:i+1), 라이브가 50봉(`ohlcv[-LOOKBACK:]`)이라
        # 같은 날짜의 ADX가 달랐다(36.15 vs 35.72). 한 봉 차이가 레짐
        # 경계에서 분류를 뒤집고, 그러면 어떤 전략이 거래할지가 달라진다.
        window   = btc_1d_ohlcv[max(0, i - REGIME_BTC_LOOKBACK + 1): i + 1]
        adx      = calc_adx(window, 14)
        btc_px   = float(close.iloc[i])
        ema_val  = float(ema200.iloc[i])
        atr_val  = float(atr14.iloc[i]) if not pd.isna(atr14.iloc[i]) else 0.0
        atr_pct  = atr_val / btc_px if btc_px else 0.0
        # ADX 3봉 기울기 (라이브와 동일 — 추세 소멸 조기 감지)
        adx_hist.append(adx)
        adx_slope = (adx_hist[-1] - adx_hist[-3]) if len(adx_hist) >= 3 else 0.0
        adx_4h    = adx4h_by_date.get(date_key, 0.0)
        regime_map[date_key] = classify_regime(adx, btc_px, ema_val, atr_pct,
                                               adx_4h, adx_slope)
    return regime_map


def _bt_exit_decision(position: dict, strategy_id: str, df, i: int,
                      exit_fn, slip: float,
                      cur_high: float, cur_low: float, cur_close: float,
                      bars_held: int, max_hold: int) -> tuple:
    """이 봉에서 청산되는가. Returns: (사유 or None, 청산가).

    **판정 순서가 곧 우선순위다** — SL이 TP보다 먼저 온다. 한 봉 안에서
    고가와 저가를 모두 건드렸을 때 어느 쪽이 먼저 체결됐는지는 일봉·4시간봉
    데이터로 알 수 없으므로, 불리한 쪽(SL)을 택해 성과를 과대평가하지 않는다.

    추적 손절 갱신은 **여기서 하지 않는다.** 이 봉의 판정이 끝난 뒤에 해야
    같은 봉 고가로 올린 손절이 같은 봉 저가에 걸리는 선행편향이 없다.
    """
    # SL — bar 저가가 손절선을 이탈
    if cur_low <= position['sl']:
        return 'SL', min(position['sl'], cur_close) * (1 - slip)

    # TP — bar 고가가 목표가 도달
    if position['tp'] > 0 and cur_high >= position['tp']:
        return 'TP', position['tp'] * (1 - slip)

    # 전략별 추가 청산 조건(크로스 등)
    if exit_fn is not None:
        try:
            should_exit = (exit_fn(df, i, position['entry_price'])
                           if strategy_id == 'S6' else exit_fn(df, i))
            if should_exit:
                return 'CROSS', cur_close * (1 - slip)
        except Exception:
            # 지표 결측 등으로 청산 함수가 실패해도 봉 루프를 멈추면 안 된다.
            # SL/TP는 이미 위에서 확인했으므로 보호는 유지된다.
            pass

    # 시간 기반 청산
    if max_hold > 0 and bars_held >= max_hold:
        return 'TIME', cur_close * (1 - slip)

    return None, 0.0


def _bt_rs_gate(rs_applies: bool, rank_map, rank_keys: list,
                symbol: str, date_key: str) -> tuple:
    """RS Gate 판정. Returns: (리스크 배수 or **None(차단)**, 진단키 or None).

    차단을 `None`으로 돌려주는 이유: 배수 0.0으로 표현하면 호출부에서
    곱셈에 흘러들어가 '진입은 했는데 수량이 0'인 주문이 된다. 차단과
    감봉은 다른 사건이므로 타입으로 구분한다.

    라이브 `_get_momentum_rank_pct`와 같은 기본값(0.5)을 쓴다 — 순위를
    모를 때 하위권으로 취급하면 신규 심볼이 영구 차단된다.
    """
    if not rs_applies:
        return 1.0, None
    rank_pct = _rank_pct_asof(rank_map, rank_keys, symbol, date_key)
    if rank_pct is None:
        rank_pct = 0.5
    if rank_pct > MOMENTUM_RS_GATE_PCT:
        return None, 'rs_gate_block'
    if rank_pct <= MOMENTUM_TOP_TIER_PCT:
        return MOMENTUM_TOP_RISK_MULT, 'rs_top_tier'
    return 1.0, None


def _bt_funding_scale(applies: bool, funding_map: Optional[dict],
                      ts_ms: int) -> tuple:
    """펀딩비 판정. Returns: (리스크 배수 or **None(차단)**, 진단키 or None).

    라이브 `_funding_scale` 패리티. 롱 과밀(펀딩 ≥ FUNDING_LONG_BLOCK)이면
    진입을 막고, 숏 쏠림(≤ FUNDING_SHORT_BOOST)이면 리스크를 키운다.

    데이터가 없으면 **1.0(통과)** 이다 — 라이브의 `_get_spot_funding` 도
    조회 실패 시 0.0을 돌려 통과시키므로 같은 동작이다. 퍼프가 없는 현물
    전용 심볼(MATIC·EOS 등)이 여기 해당한다. 하위권 취급으로 영구 차단하면
    라이브와 어긋난다.
    """
    if not applies or not funding_map:
        return 1.0, None
    rate = _funding_asof(funding_map, ts_ms)
    if rate is None:
        return 1.0, None
    if rate >= FUNDING_LONG_BLOCK:
        return None, 'funding_block'
    if rate <= FUNDING_SHORT_BOOST:
        return 1.20, 'funding_boost'
    return 1.0, None


def _funding_asof(funding_map: dict, ts_ms: int) -> Optional[float]:
    """그 시점에 **이미 확정된** 가장 최근 펀딩비.

    펀딩은 8시간마다 확정되고 봉은 4H/1D다. 봉 시각 이후의 펀딩을 쓰면
    미래를 보는 것이므로, ts 이하의 마지막 값만 쓴다(선행편향 차단).
    """
    keys = funding_map.get('_keys')
    if not keys:
        return None
    i = bisect.bisect_right(keys, ts_ms) - 1
    return funding_map['rates'][keys[i]] if i >= 0 else None


def build_funding_map(symbol: str, since_ms: int,
                      data_dir: Optional[Path] = None) -> dict:
    """심볼의 과거 펀딩비를 {ts_ms: rate} 로. 실패·미지원이면 빈 dict.

    OHLCV와 같은 방식으로 CSV 캐시한다 — 5년치가 심볼당 약 5,500건이라
    매 실행마다 받으면 백테스트가 느려지고 레이트리밋도 낭비된다.
    """
    csv_path = (data_dir / f'{symbol}_FUNDING.csv') if data_dir else None
    rows: list = []
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            rows = [(int(t), float(r)) for t, r in zip(df['ts'], df['rate'], strict=True)]
        except Exception as e:
            print(f'    [펀딩] {symbol} 캐시 읽기 실패 — 재수집: {e}')
            rows = []
    if not rows:
        rows = _fetch_funding_history(symbol, since_ms)
        if rows and csv_path:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=['ts', 'rate']).to_csv(csv_path, index=False)
    if not rows:
        return {}
    rows.sort()
    return {'_keys': [t for t, _ in rows], 'rates': dict(rows)}


def _fetch_funding_history(symbol: str, since_ms: int) -> list:
    """선물 펀딩비 이력 전체 페이지네이션. 퍼프 미지원이면 빈 리스트."""
    try:
        import ccxt
        ex = ccxt.binance({'options': {'defaultType': 'future'},
                           'enableRateLimit': True})
        ccxt_sym = symbol.replace('USDT', '/USDT:USDT')
        out: list = []
        since = since_ms
        while True:
            batch = ex.fetch_funding_rate_history(ccxt_sym, since=since, limit=1000)
            if not batch:
                break
            new = [(int(b['timestamp']), float(b['fundingRate']))
                   for b in batch if b.get('fundingRate') is not None]
            if out:
                new = [x for x in new if x[0] > out[-1][0]]
            if not new:
                break
            out.extend(new)
            if len(batch) < 1000:
                break
            since = new[-1][0] + 1
        print(f'    [펀딩] {symbol} {len(out)}건 수집')
        return out
    except Exception as e:
        # 퍼프가 없는 현물 전용 심볼이 여기 온다 — 라이브도 통과시키므로
        # 실패가 아니라 '해당 없음'이다.
        print(f'    [펀딩] {symbol} 미지원/실패 — 필터 미적용: {e}')
        return []


def _bt_entry_filters(sl_dist: float, entry_price: float,
                      slip: float) -> Optional[str]:
    """라이브 진입 필터 패리티. 차단되면 진단키를, 통과하면 None을 낸다.

    라이브에만 있고 백테스트에 없으면 성과가 구조적으로 과대평가된다.

    ※ 남아있는 한계 — **포트폴리오 제약은 모델링하지 못한다.**
      `backtest_strategy`는 (전략 × 심볼) 단위로 독립 실행되므로 동시
      포지션 수 상한(SPOT_MAX_POSITIONS), 슬롯당 최소 자본
      (SPOT_EQUITY_PER_SLOT), USDT 예비금(SPOT_RESERVE_PCT), 전략 간
      자본 경합을 반영할 수 없다. 즉 백테스트는 여전히 라이브보다
      **낙관적**이며, 특히 신호가 몰리는 구간에서 라이브는 일부 진입을
      포기한다. 결과를 상한선으로 읽을 것.
    """
    sl_pct = sl_dist / entry_price
    if sl_pct > SPOT_MAX_SL_PCT:
        return 'sl_too_wide'
    # 비용 대비 엣지: 왕복비용이 1R을 얼마나 잠식하는지
    # (라이브 `_cost_edge_ok`와 동일한 부등식. 스프레드는 백테스트에
    #  호가 데이터가 없어 슬리피지 모델로 근사한다)
    cost_rate = BT_SPOT_FEE * 2 + slip * 2
    if cost_rate / sl_pct > SPOT_MAX_COST_PER_R:
        return 'cost_exceeds_edge'
    return None


def _bt_kelly_scale(trades: list) -> float:
    """Kelly 스케일 — 라이브 `_get_kelly_scale`과 같은 규칙.

    `backtest_strategy`에서 뽑아낸 순수함수다. 인라인이던 시절에는 이 로직만
    따로 시험할 수 없어서, 라이브와 어긋나도 알아채기 어려웠다.
    """
    if len(trades) < SPOT_KELLY_MIN_TRADES:
        return 1.0
    recent   = trades[-200:]
    wins_r   = [t.pnl_r for t in recent if t.pnl_r > 0]
    losses_r = [t.pnl_r for t in recent if t.pnl_r <= 0]
    if not wins_r:
        # 전패 구간 — 라이브와 동일하게 최소 스케일로. (1.0을 유지하면
        # 연패 중에도 정상 사이즈로 베팅하는 낙관 편향이 생긴다)
        return SPOT_KELLY_SCALE_MIN
    if not losses_r:
        return 1.0
    wr = len(wins_r) / len(recent)
    b  = abs(float(np.mean(wins_r))) / abs(float(np.mean(losses_r)))
    # half-Kelly (라이브와 패리티 유지)
    kelly_raw = (wr - (1 - wr) / b) * SPOT_KELLY_FRACTION if b > 0 else 0.0
    # 조건부 상한: WR·PF가 모두 좋을 때만 KELLY_SCALE_MAX까지 허용
    gross_w = sum(abs(r) for r in wins_r)
    gross_l = sum(abs(r) for r in losses_r)
    pf = gross_w / gross_l if gross_l > 0 else 0.0
    k_max = (SPOT_KELLY_SCALE_MAX
             if (wr >= SPOT_KELLY_WR_THRESH and pf >= SPOT_KELLY_PF_THRESH)
             else 1.50)
    return max(SPOT_KELLY_SCALE_MIN, min(k_max, kelly_raw))


def _bt_health_scale(trades: list) -> Optional[float]:
    """전략 건강도 스케일 — 라이브 `_get_strategy_health_scale` 패리티.

    Returns:
        스케일(float), 또는 **차단이면 None**. 차단과 '스케일 0.0'을 같은
        값으로 돌려주면 호출부에서 곱셈으로 흘러들어가 조용히 0 사이즈
        주문이 된다 — 명시적으로 구분한다.
    """
    if len(trades) < SPOT_HEALTH_MIN_TRADES:
        return 1.0
    recent = trades[-200:]
    # 손익 기여도 = R배수 × 그 거래의 리스크 비중(자본 대비).
    # 라이브는 USD net PF를 쓰지만 백테스트에는 거래별 USD 손익이 없으므로
    # 비율로 근사한다 — 판정 자체의 패리티가 목적이다.
    nets = [t.pnl_r * t.risk_pct for t in recent]
    gw = sum(n for n in nets if n > 0)
    gl = abs(sum(n for n in nets if n < 0))
    pf = gw / gl if gl > 0 else float('inf')
    if pf < SPOT_HEALTH_PF_HARD:
        return None
    return SPOT_HEALTH_SOFT_SCALE if pf < SPOT_HEALTH_PF_SOFT else 1.0


def _bt_learn_config():
    """config 상수 → LearnConfig (라이브 `_learn_config`와 동일 값)."""
    import atlas_learning as _L
    return _L.LearnConfig(
        half_life_days=SPOT_LEARN_HALF_LIFE_DAYS,
        min_info=SPOT_LEARN_MIN_INFO,
        unproven_scale=SPOT_LEARN_UNPROVEN_SCALE,
        explore_floor=SPOT_LEARN_FLOOR,
        max_scale=SPOT_LEARN_MAX_SCALE,
        gain=SPOT_LEARN_GAIN,
        cost_per_r=SPOT_LEARN_COST_PER_R,
    )


def build_momentum_rank_map(ohlcv_by_symbol: dict,
                            momentum_days: int | None = None,
                            vol_days: int = 30) -> dict:
    """날짜 → {심볼: 모멘텀 순위 백분위(0=최상위)} 맵.

    라이브 `rank_by_momentum`과 **같은 산식**을 시계열로 재현한다:
        점수 = (N일 수익률) / (30일 수익률 표준편차)
    각 날짜에서 그 시점까지의 데이터만 쓰므로 선행편향이 없다.

    이게 없으면 RS Gate와 주도주 리스크 부스트를 백테스트가 모델링할 수
    없고, 그 파라미터들은 영영 검증되지 않은 채로 남는다.
    """
    if momentum_days is None:
        momentum_days = UNIVERSE_MOMENTUM_DAYS
    closes = {}
    for sym, ohlcv in (ohlcv_by_symbol or {}).items():
        if not ohlcv or len(ohlcv) < momentum_days + vol_days:
            continue
        df = _ohlcv_to_df(ohlcv)
        s = pd.Series(df['close'].values,
                      index=df['ts'].dt.date.astype(str).values)
        closes[sym] = s[~s.index.duplicated(keep='last')]
    if len(closes) < 2:
        return {}

    px = pd.DataFrame(closes).sort_index()
    mom = px / px.shift(momentum_days) - 1.0
    # ffill 후 fill_method=None — 지금 동작(pad 후 계산)을 그대로 보존하면서
    # FutureWarning을 없앤다. pct_change의 fill_method 인자 자체가 제거 예정이라,
    # 그대로 두면 pandas 업그레이드 시 **기본 동작이 바뀌어** 변동성·RS 순위가
    # 조용히 달라진다(requirements는 pandas<4.0.0까지 허용한다).
    vol = px.ffill().pct_change(fill_method=None).rolling(vol_days).std()
    score = mom / vol.replace(0, np.nan)
    # 높을수록 상위 → 내림차순 순위(0-based)를 심볼 수로 나눠 백분위화
    ranks = score.rank(axis=1, ascending=False, method='first', na_option='bottom')
    pct = (ranks - 1).div(score.shape[1])

    out: dict = {}
    for date, row in pct.iterrows():
        vals = {s: float(v) for s, v in row.items() if v == v}
        if vals:
            out[date] = vals
    return out


def _rank_pct_asof(rank_map: dict, rank_keys: list, symbol: str,
                   date_key: str) -> Optional[float]:
    """`date_key` **직전**에 확정된 순위 백분위. 없으면 None.

    직전을 쓰는 이유: 순위는 그날 종가로 계산되므로, 같은 날 시가에
    진입하는 봉에서 그 값을 쓰면 미래를 보는 것이 된다(선행편향).
    4H 전략도 같은 날짜의 일봉 종가는 아직 모른다 — 그래서 '<'로 자른다.
    """
    if not rank_keys or not date_key:
        return None
    i = bisect.bisect_left(rank_keys, date_key) - 1
    if i < 0:
        return None
    return rank_map.get(rank_keys[i], {}).get(symbol)


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

def _equity_curve(trades: list, initial_equity: float) -> tuple:
    """거래 순서대로 자본을 굴려 (자본곡선, 최종자본, 최대낙폭)을 낸다.

    복리다 — 각 거래는 **그 시점 자본**의 risk_pct만큼 건다. 그래서 같은
    거래 집합이라도 순서가 다르면 최종 자본이 달라진다. 이 함수가 순서를
    보존하는 이유이고, 지표를 거래 목록 평균으로 대신 계산하면 안 되는
    이유이기도 하다.

    자본에 하한(0.01)을 두는 것은 음수 자본에서 수익률이 발산해 Sharpe가
    무의미해지는 것을 막기 위해서다.
    """
    equity = initial_equity
    eq_curve = [equity]
    peak_eq = equity
    max_dd = 0.0
    for t in trades:
        equity += equity * t.risk_pct * t.pnl_r
        equity = max(equity, 0.01)
        eq_curve.append(equity)
        if equity > peak_eq:
            peak_eq = equity
        else:
            max_dd = max(max_dd, (peak_eq - equity) / peak_eq)
    return eq_curve, equity, max_dd


def _cagr_pct(equity: float, initial_equity: float,
              start_date: str, end_date: str) -> float:
    """연평균 성장률(%). 날짜가 없거나 파싱 불가면 0.0.

    기간 하한 0.1년: 짧은 구간에서 지수가 폭발해 CAGR이 수천 %로 나오는
    것을 막는다. 그런 값은 Calmar까지 오염시킨다.
    """
    if not start_date or not end_date or equity <= 0:
        return 0.0
    try:
        dt_s = datetime.strptime(start_date, '%Y-%m-%d')
        dt_e = datetime.strptime(end_date, '%Y-%m-%d')
    except (ValueError, TypeError):
        return 0.0
    years = max((dt_e - dt_s).days / 365.25, 0.1)
    return ((equity / initial_equity) ** (1 / years) - 1) * 100


def _risk_ratios(eq_curve: list) -> tuple:
    """(Sharpe, Sortino) — 자본곡선 수익률 기준, 252일 연율화.

    표본이 2개 미만이거나 표준편차가 0이면 0.0을 낸다. 0으로 나눠 inf가
    나오면 WFO 합격 판정이 통과해 버린다 — '변동이 없어서' 나온 무한대를
    '위험 없는 수익'으로 읽는 셈이다.
    """
    if len(eq_curve) < 2:
        return 0.0, 0.0
    rets = np.array([
        (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]
        for i in range(1, len(eq_curve))
    ])
    sharpe = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    sortino = 0.0
    down = rets[rets < 0]
    if len(down) > 1 and down.std() > 0:
        sortino = float(rets.mean() / down.std() * math.sqrt(252))
    return sharpe, sortino


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

    eq_curve, equity, max_dd = _equity_curve(trades, initial_equity)
    total_pnl_pct = (equity - initial_equity) / initial_equity * 100
    cagr = _cagr_pct(equity, initial_equity, start_date, end_date)
    sharpe, sortino = _risk_ratios(eq_curve)
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
    rank_map:     Optional[dict] = None,
    funding_map:  Optional[dict] = None,
) -> tuple[list, dict]:
    """
    단일 전략 × 단일 심볼 bar-by-bar 시뮬레이션.

    Args:
        rank_map: `build_momentum_rank_map()` 결과. 넘기면 RS Gate와
                  주도주 리스크 부스트가 라이브와 동일하게 적용된다.
                  None이면 두 규칙 모두 비활성 — 그 경우 백테스트는
                  라이브보다 낙관적이다(막혔을 진입까지 체결로 센다).
        funding_map: `build_funding_map()` 결과. 넘기면 롱 과밀 구간의
                  진입 차단·숏 쏠림 부스트가 라이브와 동일하게 적용된다.
                  None이면 비활성 — 역시 낙관 편향이 남는다.

    Returns:
        (trades: list[SpotTrade], diagnostics: dict)
    """
    if len(ohlcv) < 50:
        return [], {}

    rank_keys = sorted(rank_map) if rank_map else []
    rs_applies = bool(rank_keys) and strategy_id in MOMENTUM_RS_GATE_STRATS
    fund_applies = bool(funding_map) and strategy_id in FUNDING_APPLY_STRATS

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
    # 자기주도 학습 상태 — 라이브가 학습 배분을 쓰면 백테스트도 같은 규칙을
    # 써야 한다. 안 그러면 검증한 사이징과 실계좌 사이징이 갈라진다.
    learn_obs:   list = []      # 청산된 거래 → 관측
    learn_res:   dict = {}
    learn_at             = None   # 마지막 학습 시각(시뮬레이션 시계)
    learn_cfg = _bt_learn_config() if SPOT_LEARN_ENABLED else None
    position = None      # 현재 포지션 dict or None
    cooldown = 0         # 쿨다운 카운터 (S3 청산 후 / S5 손절 후)
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
            # exit_type은 기록용 메타데이터 — 청산 분기는 EXIT_CHECK_FUNCS가 한다

            slip = _get_slippage(symbol)
            reason, exit_price = _bt_exit_decision(
                position, strategy_id, df, i, exit_fn, slip,
                cur_high, cur_low, cur_close, bars_held, max_hold)

            if reason is not None:
                entry_price = position['entry_price']
                # R배수는 **진입 시점의 위험**으로 재야 한다. 추적 손절로
                # SL이 올라간 뒤 그 값을 쓰면 분모가 줄어 R이 부풀고,
                # 그 pnl_r이 Kelly·건강도·avg_r을 오염시킨다.
                sl_dist     = abs(entry_price - position.get('orig_sl', position['sl']))
                pnl_r       = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
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
                if learn_cfg is not None:
                    import atlas_learning as _L
                    learn_obs.append(_L.Observation(
                        strategy_id, position.get('regime', '') or 'UNKNOWN',
                        float(pnl_r), row['ts'].to_pydatetime()
                        if hasattr(row['ts'], 'to_pydatetime') else datetime.now(timezone.utc)))
                position = None
                if strategy_id == 'S3':
                    cooldown = S3_COOLDOWN
                elif strategy_id == 'S5' and reason == 'SL':
                    # 라이브 패리티: _s5_safety_block 의 SL 쿨다운.
                    # 평균회귀는 '더 싸졌으니 또 산다'가 되기 쉬워, 손절 직후
                    # 같은 종목에 재진입하면 같은 하락에 연속으로 맞는다.
                    # 라이브는 이 재진입을 막는데 백테스트가 세면, **라이브가
                    # 절대 하지 않는 거래**로 성과를 평가하게 된다.
                    # (막는 이유가 '나쁜 거래'이므로 편향은 비관 쪽이다 —
                    #  S5는 지금 OOS 기준 미달로 재최적화 대상이라 특히 중요)
                    cooldown = S5_SL_COOLDOWN_BARS
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

        # ── RS Gate / 주도주 부스트 (라이브 진입 경로 패리티) ──────────
        # 라이브는 모멘텀 하위권 심볼의 돌파 진입을 막고 상위 티어는 리스크를
        # 올린다. 백테스트가 이걸 모르면 MOMENTUM_RS_GATE_PCT는 어떤 값을
        # 넣어도 결과가 같아, WFO가 검증할 수 없는(=검증된 척만 하는) 축이 된다.
        rs_scale, rs_flag = _bt_rs_gate(rs_applies, rank_map, rank_keys,
                                        symbol, date_key)
        if rs_flag:
            diag[rs_flag] = diag.get(rs_flag, 0) + 1
        if rs_scale is None:
            continue

        # ── 펀딩비 필터 (라이브 진입 경로 패리티) ──────────────────────
        # 라이브는 롱이 과밀할 때(펀딩 ≥ 0.05%/8h) 추세추종 진입을 막는다.
        # 백테스트가 이걸 모르면 라이브가 실제로는 걸렀을 구간까지 거래한
        # 것으로 계산해 **낙관 편향**이 된다. 특히 S6는 현재 OOS PF가 가장
        # 높은 전략이라 그 수치가 부풀려져 있으면 판단이 통째로 흔들린다.
        fund_scale, fund_flag = _bt_funding_scale(fund_applies, funding_map, ts)
        if fund_flag:
            diag[fund_flag] = diag.get(fund_flag, 0) + 1
        if fund_scale is None:
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
        block = _bt_entry_filters(sl_dist, entry_price, slip)
        if block:
            diag[block] = diag.get(block, 0) + 1
            continue

        # 포지션 크기 계산 (스팟: 레버리지 없음)
        kelly_scale  = _bt_kelly_scale(trades)
        health_scale = _bt_health_scale(trades)
        if health_scale is None:          # net PF가 하드 임계 미만 → 진입 차단
            diag['health_blocked'] = diag.get('health_blocked', 0) + 1
            continue

        # 학습기 ON이면 Kelly·건강도 자리를 학습 배분이 대신한다(라이브와 동일).
        if learn_cfg is not None:
            import atlas_learning as _L
            _now_dt = (row['ts'].to_pydatetime() if hasattr(row['ts'], 'to_pydatetime')
                       else None)
            if _now_dt is not None:
                if _now_dt.tzinfo is None:
                    _now_dt = _now_dt.replace(tzinfo=timezone.utc)
                # 라이브와 같은 주기로만 재계산한다 — 매 진입마다 전체를 다시
                # 학습하면 O(n²)이 된다.
                # ※ 현 전략들은 4H·1D봉이라 진입 간격이 항상 30분을 넘는다.
                #   즉 이 스로틀은 **지금은 한 번도 걸리지 않는다**(성능 방어용).
                #   분봉 전략을 추가하면 그때부터 의미가 생긴다.
                if (learn_at is None or
                        (_now_dt - learn_at).total_seconds() >= SPOT_LEARN_REFRESH_MIN * 60):
                    learn_res = _L.learn(learn_obs, now=_now_dt, cfg=learn_cfg)
                    learn_at = _now_dt
                learn_scale = _L.scale_for(learn_res, strategy_id, regime, learn_cfg)
            else:
                learn_scale = learn_cfg.unproven_scale
            kelly_scale, health_scale = learn_scale, 1.0
            if learn_scale <= 0:
                diag['learn_blocked'] = diag.get('learn_blocked', 0) + 1
                continue

        adj_risk_pct = (risk_pct * regime_scale * kelly_scale * health_scale
                        * rs_scale * fund_scale)
        risk_usd     = equity * adj_risk_pct
        qty          = risk_usd / sl_dist
        cost_usdt    = qty * entry_price

        # 배분 한도 체크
        if cost_usdt > equity * SPOT_MAX_ALLOC_PCT:
            cost_usdt = equity * SPOT_MAX_ALLOC_PCT
            qty       = cost_usdt / entry_price
            adj_risk_pct = (qty * sl_dist) / equity

        # 거래소 최소 주문금액(NOTIONAL) — 라이브 `_spot_buy` 패리티.
        # 한도 축소 **이후**에 본다(라이브와 같은 순서). 미달이면 주문 자체가
        # 거부되므로 진입이 아니라 미체결이다.
        #
        # 기본 자본($10,000)에서는 주문이 수백 달러라 이 바닥이 걸리지 않아
        # 오랫동안 드러나지 않았다. 그러나 소액 계좌에서는 상시로 걸린다 —
        # 실계좌 $565 기준 사이징 진단이 레짐별 주문금액을 $3.26~$10.81로
        # 보고했고, 하락장 담당 S4는 $3.26으로 아예 진입 불가였다.
        # 모델링하지 않으면 WFO는 **그 계좌가 실행할 수 없는 조합**을 검증한다.
        if cost_usdt < SPOT_MIN_ORDER_USDT:
            diag['notional_block'] = diag.get('notional_block', 0) + 1
            continue

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
    initial_equity: float = BT_INITIAL_EQ,
) -> dict:
    """
    전략 × 심볼 조합 전체 백테스트.
    Returns: {strategy_id: {symbol: {'trades': [...], 'metrics': {...}}}}
    """
    since_ms = _since_ms(start_date)
    # BTC 1D (레짐 맵용, 스팟 데이터 사용)
    print(f'\n[레짐맵] BTC 1D 데이터 로드...')
    btc_1d = _load_or_fetch('BTCUSDT', '1d', since_ms, data_dir)
    # 4H도 함께 로드 — 라이브가 adx_4h로 MICRO_RANGING을 판정하므로,
    # 넘기지 않으면 백테스트만 다른 레짐 경로를 걷는다.
    btc_4h = _load_or_fetch('BTCUSDT', '4h', since_ms, data_dir)
    regime_map = build_regime_map(btc_1d, btc_4h) if btc_1d else {}
    print(f'  레짐맵 생성: {len(regime_map)}일'
          + ('' if btc_4h else ' (4H 없음 — MICRO_RANGING 미반영)'))

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

    # 모멘텀 순위는 **일봉**으로 계산한다(라이브와 동일). RS Gate 전략이
    # 4H여도 일봉이 필요하므로, 게이트 대상이 하나라도 있으면 1D를 받는다.
    need_rank = any(s in MOMENTUM_RS_GATE_STRATS for s in strategies)

    if need_1d or need_rank:
        print(f'\n[데이터] 1D 심볼 로드 ({len(symbols)}개)...')
        for sym in symbols:
            if sym in ohlcv_1d:
                continue
            data = _load_or_fetch(sym, '1d', since_ms, data_dir)
            if data:
                ohlcv_1d[sym] = data

    rank_map: dict = {}
    if need_rank:
        rank_map = build_momentum_rank_map(ohlcv_1d)
        print(f'  모멘텀 순위맵: {len(rank_map)}일 × {len(ohlcv_1d)}심볼'
              + ('' if rank_map else ' — 데이터 부족, RS Gate 미적용'))

    # 펀딩비는 추세추종 전략에만 쓰인다. 대상이 없으면 받지 않는다
    # (심볼당 5년치 약 5,500건 — 불필요하게 받으면 첫 실행이 크게 느려진다).
    funding_maps: dict = {}
    if any(s in FUNDING_APPLY_STRATS for s in strategies):
        print(f'\n[데이터] 펀딩비 로드 ({len(symbols)}개)...')
        for sym in symbols:
            fm = build_funding_map(sym, since_ms, data_dir)
            if fm:
                funding_maps[sym] = fm
        print(f'  펀딩맵: {len(funding_maps)}/{len(symbols)}심볼'
              + ('' if funding_maps else ' — 데이터 없음, 펀딩 필터 미적용'))

    # 백테스트 실행
    results = {}
    for strategy_id in strategies:
        tf      = STRATEGY_TIMEFRAMES.get(strategy_id, '1d')
        ohlcv_d = ohlcv_4h if tf == '4h' else ohlcv_1d
        results[strategy_id] = {}
        all_trades = []
        rs_stat = {'block': 0, 'top': 0}

        print(f'\n[{strategy_id}] {STRATEGY_NAMES[strategy_id]} 백테스트 ({tf})...')
        for sym in symbols:
            ohlcv = ohlcv_d.get(sym, [])
            if not ohlcv:
                continue
            trades, diag = backtest_strategy(
                strategy_id, sym, ohlcv, regime_map,
                start_date, end_date, risk_pct, initial_equity,
                rank_map=rank_map or None,
                funding_map=funding_maps.get(sym) or None,
            )
            metrics = calc_spot_metrics(trades, initial_equity, start_date, end_date)
            results[strategy_id][sym] = {
                'trades':  [asdict(t) for t in trades],
                'metrics': metrics,
                'diag':    diag,
            }
            all_trades.extend(trades)
            rs_stat['block'] += diag.get('rs_gate_block', 0)
            rs_stat['top']   += diag.get('rs_top_tier', 0)
            n = len(trades)
            pnl = metrics.get('total_pnl_pct', 0)
            wr  = metrics.get('win_rate', 0)
            print(f'    {sym:<14} {n:>4}건  PnL {pnl:+6.1f}%  WR {wr:.0f}%')

        # 전략 합산 지표
        combined = calc_spot_metrics(all_trades, initial_equity, start_date, end_date)
        results[strategy_id]['_combined'] = combined
        print(f'  → 합계: {len(all_trades)}건  '
              f'PnL {combined.get("total_pnl_pct", 0):+.1f}%  '
              f'Sharpe {combined.get("sharpe", 0):.2f}  '
              f'MDD {combined.get("max_dd_pct", 0):.1f}%')
        # 게이트가 몇 건을 실제로 막았는지 보이지 않으면, 아무것도 막지 않는
        # 임계값(현재 기본 0.99)이 '동작 중인 필터'로 오해된다.
        if strategy_id in MOMENTUM_RS_GATE_STRATS and rank_map:
            print(f'     RS Gate(≤{MOMENTUM_RS_GATE_PCT:.2f}): '
                  f'{rs_stat["block"]}건 차단, 주도주 부스트 {rs_stat["top"]}건'
                  + ('  ← 차단 0건: 임계값이 사실상 무효' if not rs_stat['block'] else ''))

    return results


# ══════════════════════════════════════════════════════════════
#  Walk-Forward 검증
# ══════════════════════════════════════════════════════════════

def run_walk_forward(
    strategies:  list[str],
    symbols:     list[str],
    data_dir:    Optional[Path] = None,
    rolling:     bool = False,
    initial_equity: float = BT_INITIAL_EQ,
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
        res = run_spot_backtest(strategies, symbols, start, end, data_dir,
                                initial_equity=initial_equity)
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
    header  = '  ' + '  '.join(h.ljust(w) for h, w in zip(headers, col_w, strict=False))
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
        line = '  ' + '  '.join(v.ljust(w) for v, w in zip(row, col_w, strict=False))
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

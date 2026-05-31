"""
ATLAS Spot — 심볼 유니버스 모듈
=================================
Binance 현물 마켓에서 유동성 기준으로 거래 대상 심볼을 동적으로 선별합니다.

주요 기능:
  - 24h 거래량 기준 상위 N개 USDT 페어 선택
  - 스테이블코인 / 레버리지토큰 자동 제외
  - 모멘텀 기반 랭킹 (90일 수익률 / 30일 변동성)
  - 24시간 주기 갱신 루프

사용법:
  python atlas_spot_universe.py          # 현재 유니버스 출력
  python atlas_spot_universe.py --top 20 # 상위 20개만 출력
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'UNIVERSE')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atlas_spot_config import (
    UNIVERSE_MIN_VOLUME_USD, UNIVERSE_MAX_SYMBOLS,
    UNIVERSE_STABLECOIN_BASE, UNIVERSE_LEVERAGED_KEYWORDS,
    UNIVERSE_MOMENTUM_DAYS, UNIVERSE_REFRESH_HOURS,
)


# ══════════════════════════════════════════════════════════════
#  핵심 필터
# ══════════════════════════════════════════════════════════════

def _is_excluded(base: str) -> bool:
    """스테이블코인 또는 레버리지 토큰 여부 확인."""
    if base in UNIVERSE_STABLECOIN_BASE:
        return True
    for kw in UNIVERSE_LEVERAGED_KEYWORDS:
        if kw in base:
            return True
    return False


def discover_universe(ex, min_volume: float = None,
                      max_symbols: int = None) -> list[str]:
    """
    Binance 현물 마켓에서 거래 대상 심볼 목록을 반환합니다.

    Args:
        ex: ccxt.binance 인스턴스 (spot type)
        min_volume: 24h 최소 거래량 USD (기본값: config)
        max_symbols: 최대 심볼 수 (기본값: config)

    Returns:
        ['BTCUSDT', 'ETHUSDT', ...] 형식의 심볼 리스트 (거래량 내림차순)
    """
    if min_volume is None:
        min_volume = UNIVERSE_MIN_VOLUME_USD
    if max_symbols is None:
        max_symbols = UNIVERSE_MAX_SYMBOLS

    print(f'  [Universe] Binance Spot 마켓 스캔 중...')
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        print(f'  [Universe] fetch_tickers 실패: {e}')
        return []

    candidates = []
    for symbol, data in tickers.items():
        # USDT 페어만
        if not symbol.endswith('/USDT'):
            continue
        base = symbol.replace('/USDT', '')

        # 제외 목록 확인
        if _is_excluded(base):
            continue

        # 거래량 필터
        quote_vol = data.get('quoteVolume') or 0
        if quote_vol < min_volume:
            continue

        candidates.append({
            'symbol': base + 'USDT',
            'base':   base,
            'volume': float(quote_vol),
        })

    # 거래량 내림차순 정렬
    candidates.sort(key=lambda x: x['volume'], reverse=True)
    result = [c['symbol'] for c in candidates[:max_symbols]]

    print(f'  [Universe] 발견: {len(candidates)}개 후보 → 상위 {len(result)}개 선택')
    return result


def filter_universe(symbols: list[str], exclude: list[str] = None) -> list[str]:
    """수동 제외 목록 적용."""
    if not exclude:
        return symbols
    excl_set = set(exclude)
    return [s for s in symbols if s not in excl_set]


def filter_tradeable(ex, symbols: list[str]) -> list[str]:
    """
    실제 거래 가능 여부 확인 (활성 마켓 + 최소 주문 금액 충족).
    라이브 트레이딩 전 최종 필터.
    """
    try:
        markets = ex.load_markets()
    except Exception:
        return symbols

    tradeable = []
    for sym in symbols:
        ccxt_sym = sym.replace('USDT', '/USDT')
        mkt = markets.get(ccxt_sym)
        if not mkt:
            continue
        if not mkt.get('active', False):
            continue
        if not mkt.get('spot', False):
            continue
        tradeable.append(sym)

    return tradeable


# ══════════════════════════════════════════════════════════════
#  모멘텀 랭킹
# ══════════════════════════════════════════════════════════════

def rank_by_momentum(symbols: list[str], ohlcv_cache: dict,
                     momentum_days: int = None) -> list[str]:
    """
    리스크 조정 모멘텀으로 심볼 재랭킹합니다.
    ohlcv_cache: {symbol: ohlcv_list} 형식 (1D 데이터)

    리스크 조정 모멘텀 = (N일 수익률) / (30일 변동성)
    데이터 없는 심볼은 뒤로 정렬.
    """
    if momentum_days is None:
        momentum_days = UNIVERSE_MOMENTUM_DAYS

    vol_days = 30
    scores = {}

    for sym in symbols:
        ohlcv = ohlcv_cache.get(sym, [])
        if len(ohlcv) < momentum_days + vol_days:
            scores[sym] = -999.0
            continue

        closes = [bar[4] for bar in ohlcv]
        closes = closes[-momentum_days - vol_days:]

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        if len(returns) < momentum_days:
            scores[sym] = -999.0
            continue

        mom_return = (closes[-1] - closes[-momentum_days]) / closes[-momentum_days]
        vol_30d = float(np.std(returns[-vol_days:])) if len(returns) >= vol_days else 0.01
        scores[sym] = mom_return / vol_30d if vol_30d > 0 else -999.0

    return sorted(symbols, key=lambda s: scores.get(s, -999.0), reverse=True)


# ══════════════════════════════════════════════════════════════
#  백테스트용 정적 유니버스 (타임스탬프 기반, 선행편향 방지)
# ══════════════════════════════════════════════════════════════

# 2021-01-01 기준으로 충분한 역사를 가진 상위 심볼들
# (선행편향 방지: 2021년 시점에 존재하던 심볼만 포함)
BACKTEST_UNIVERSE_FIXED = [
    'BTCUSDT',  'ETHUSDT',  'BNBUSDT',  'XRPUSDT',  'ADAUSDT',
    'SOLUSDT',  'DOTUSDT',  'DOGEUSDT', 'LTCUSDT',   'LINKUSDT',
    'UNIUSDT',  'AVAXUSDT', 'MATICUSDT','ATOMUSDT',  'ETCUSDT',
    'VETUSDT',  'FILUSDT',  'TRXUSDT',  'XLMUSDT',   'EOSUSDT',
    'AAVEUSDT', 'XTZUSDT',  'ALGOUSDT', 'ICPUSDT',   'THETAUSDT',
]

# 2022년 이후 충분한 데이터가 있는 심볼 (선택적 확장)
BACKTEST_UNIVERSE_EXTENDED = BACKTEST_UNIVERSE_FIXED + [
    'NEARUSDT', 'FTMUSDT',  'SANDUSDT', 'MANAUSDT',  'AXSUSDT',
    'RUNEUSDT', 'GRTUSDT',  'CRVUSDT',  'MKRUSDT',   'SNXUSDT',
]


def get_backtest_universe(extended: bool = False) -> list[str]:
    """백테스트용 고정 유니버스 반환."""
    return BACKTEST_UNIVERSE_EXTENDED if extended else BACKTEST_UNIVERSE_FIXED


# ══════════════════════════════════════════════════════════════
#  라이브 갱신 루프
# ══════════════════════════════════════════════════════════════

def universe_refresh_loop(ex, shared_state: dict, stop_event=None,
                          state_lock=None) -> None:
    """
    24시간마다 유니버스를 갱신하는 백그라운드 루프.
    shared_state['universe'] 를 갱신합니다.
    갱신 실패 시 기존 유니버스를 유지합니다 (HF-3).
    """
    import threading
    if stop_event is None:
        stop_event = threading.Event()
    if state_lock is None:
        state_lock = threading.Lock()

    while not stop_event.is_set():
        try:
            new_universe = discover_universe(ex)
            if not new_universe:
                print('  [Universe] 갱신 결과 없음 — 기존 유니버스 유지')
                stop_event.wait(UNIVERSE_REFRESH_HOURS * 3600)
                continue

            old_universe = shared_state.get('universe', [])
            new_set = set(new_universe)
            old_set = set(old_universe)

            added   = new_set - old_set
            removed = old_set - new_set

            # 모멘텀 랭킹: ohlcv_cache가 없으면 거래량순 그대로 유지
            ohlcv_cache = shared_state.get('ohlcv_1d_cache', {})
            ranked_universe = rank_by_momentum(new_universe, ohlcv_cache) if ohlcv_cache else new_universe

            with state_lock:
                shared_state['universe'] = new_universe
                shared_state['universe_ranked'] = ranked_universe   # RS Gate용 모멘텀 정렬 순서
                shared_state['universe_updated'] = datetime.now(timezone.utc).isoformat()

            if added or removed:
                print(f'  [Universe] 갱신: +{list(added)} -{list(removed)}')

        except Exception as e:
            print(f'  [Universe] 갱신 오류 — 기존 유니버스 유지: {e}')

        stop_event.wait(UNIVERSE_REFRESH_HOURS * 3600)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ATLAS Spot 유니버스 탐색')
    parser.add_argument('--top',  type=int, default=50, help='상위 N개 심볼')
    parser.add_argument('--min-vol', type=float, default=UNIVERSE_MIN_VOLUME_USD,
                        help='24h 최소 거래량 USD')
    parser.add_argument('--extended', action='store_true',
                        help='백테스트 확장 유니버스 출력')
    args = parser.parse_args()

    if args.extended:
        uni = get_backtest_universe(extended=True)
        print(f'\n[백테스트 확장 유니버스] {len(uni)}개')
        for i, s in enumerate(uni, 1):
            print(f'  {i:2d}. {s}')
        return

    import ccxt
    ex = ccxt.binance({'options': {'defaultType': 'spot'}})
    universe = discover_universe(ex, min_volume=args.min_vol, max_symbols=args.top)

    print(f'\n[현재 유니버스] {len(universe)}개  ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    print(f'  {"순위":<5} {"심볼":<15}')
    print(f'  {"─" * 20}')
    for i, sym in enumerate(universe, 1):
        print(f'  {i:<5} {sym:<15}')


if __name__ == '__main__':
    main()

"""
ATLAS Spot — 라이브 트레이딩 엔진
====================================
현물(Spot) 전용 멀티 전략 자동 매매 봇.

[아키텍처]
  ┌─ balance_poller ──────── 60초마다 총자산 갱신 (USDT + 보유코인 평가)
  ├─ regime_loop ─────────── 1시간마다 BTC 레짐 갱신
  ├─ universe_refresh_loop ─ 24시간마다 유니버스 갱신
  ├─ daily_reset_loop ────── 00:00 UTC 일일 리셋 + 브리핑
  ├─ position_reconcile_loop 10분마다 DB↔거래소 포지션 검증
  ├─ tg_cmd_loop ─────────── Telegram 명령 수신
  └─ strategy_loop × N ───── 4H/1D 타임프레임별 통합 루프

[전략별 레짐 라우팅] (REGIME_STRATEGY_MAP 기준)
  TRENDING_UP  : S6(Donchian)
  RANGING      : S4(RSI), S5(BB Bounce)
  WEAK_TREND   : S3, S5, S6 (50% 리스크)
  TRENDING_DOWN: S4(RSI 과매도 반등만, 30% 리스크) — S5 하락장 0승 이력으로 차단
  CRISIS       : 전면 차단

[스팟 특화]
  - 레버리지 없음 (leverage=1)
  - 롱 전용 (매수/매도만)
  - SL: 소프트웨어 기반 (1분 폴링)
  - 잔고: USDT + 보유 코인 현재가 합산
  - 수수료: 0.1% (BNB 보유 시 0.075%)

실행:
  python atlas_spot_main.py            # 라이브 트레이딩
  python atlas_spot_main.py --dry-run  # 가상 실행 (주문 없음)
  python atlas_spot_main.py --strategies s3,s4  # 특정 전략만
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
import queue
import shutil
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import ccxt
except ImportError:
    sys.exit('[ATLAS Spot] ccxt 미설치. pip install ccxt')

from atlas_spot_config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, TG_TOKEN, TG_CHAT_ID,
    SPOT_DB_FILE, SPOT_LOG_DIR, SPOT_DATA_DIR, SPOT_KILL_SWITCH,
    SPOT_LOG_MAX_BYTES, SPOT_LOG_BACKUPS, SPOT_CAPITAL_FLOW_PCT,
    SPOT_MAX_POSITIONS, SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT,
    SPOT_RESERVE_PCT, SPOT_DAILY_LOSS_LIMIT, SPOT_MIN_ORDER_USDT, SPOT_MAX_SL_PCT,
    SPOT_EXCHANGE_STOP, SPOT_STOP_LIMIT_GAP, SPOT_EXCHANGE_OCO,
    SPOT_KELLY_FRACTION, SPOT_EQUITY_PER_SLOT,
    SPOT_HEALTH_MIN_TRADES, SPOT_HEALTH_PF_SOFT, SPOT_HEALTH_SOFT_SCALE, SPOT_HEALTH_PF_HARD,
    SPOT_HEALTH_WINDOW_DAYS,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_KELLY_WR_THRESH, SPOT_KELLY_PF_THRESH,
    SPOT_RATCHET_DD_THRESH, SPOT_RATCHET_DD_HARD, SPOT_RATCHET_RECOVER,
    SPOT_CANDLE_4H, SPOT_CANDLE_1D, SPOT_CANDLE_CACHE_TTL, SPOT_PRICE_POLL_SEC,
    STRATEGY_TIMEFRAMES, STRATEGY_NAMES, REGIME_STRATEGY_MAP, DEFAULT_ACTIVE_STRATEGIES,
    WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    BT_SPOT_FEE,
    MOMENTUM_TOP_TIER_PCT, MOMENTUM_TOP_RISK_MULT, MOMENTUM_RS_GATE_STRATS,
    FUNDING_LONG_BLOCK, FUNDING_SHORT_BOOST, FUNDING_APPLY_STRATS,
    S3_COOLDOWN, S3_COOLDOWN_WEAK,
    S5_SL_COOLDOWN_BARS, S5_BTC_CORR_SYMBOLS, S5_CORR_MAX_POS,
)
from atlas_spot_universe import discover_universe, filter_tradeable, universe_refresh_loop
from atlas_spot_strategies import CALC_FUNCS, SIGNAL_FUNCS, EXIT_CHECK_FUNCS
from atlas_regime import (
    get_cached_regime, update_regime, regime_loop,
    REGIME_CRISIS, REGIME_RANGING, REGIME_WEAK_TREND,
    REGIME_TRENDING_UP, REGIME_TRENDING_DOWN,
)


# ══════════════════════════════════════════════════════════════
#  로깅 설정
# ══════════════════════════════════════════════════════════════

SPOT_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = SPOT_LOG_DIR / f'atlas_spot_{datetime.now().strftime("%Y%m%d")}.log'

# 로테이션 필수: 24시간 상시 구동이라 무제한 파일은 수백 MB까지 자란다.
# 디스크뿐 아니라 대시보드 응답시간에도 직결된다(로그를 파싱해 레짐/지표를 뽑음).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s | %(message)s',
    handlers=[
        RotatingFileHandler(_log_file, maxBytes=SPOT_LOG_MAX_BYTES,
                            backupCount=SPOT_LOG_BACKUPS, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('atlas_spot')


# ══════════════════════════════════════════════════════════════
#  CCXT 인스턴스 (스레드-로컬, Nonce 충돌 방지)
# ══════════════════════════════════════════════════════════════

_thread_local = threading.local()


def _get_ex() -> ccxt.binance:
    """스레드별 고유 CCXT 현물 인스턴스 반환."""
    if not hasattr(_thread_local, 'ex'):
        _thread_local.ex = ccxt.binance({
            'apiKey':  BINANCE_API_KEY,
            'secret':  BINANCE_API_SECRET,
            'options': {'defaultType': 'spot'},
        })
    return _thread_local.ex


# 펀딩비 조회용 선물 CCXT 싱글턴 (매 호출마다 인스턴스 생성 방지)
_ex_futures: ccxt.binance | None = None
_ex_futures_lock = threading.Lock()

def _get_ex_futures() -> ccxt.binance:
    global _ex_futures
    if _ex_futures is None:
        with _ex_futures_lock:
            if _ex_futures is None:
                _ex_futures = ccxt.binance({
                    'apiKey':  BINANCE_API_KEY,
                    'secret':  BINANCE_API_SECRET,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True,
                })
    return _ex_futures


# 마지막 성공 가격 캐시 (네트워크 단절 시 SL 작동 보장)
_last_known_price: dict = {}   # {ccxt_sym: (price, timestamp)}
_PRICE_CACHE_TTL = 120         # 캐시 신뢰 최대 2분


# 폴링 패스 내 시세 공유 캐시.
# 기존에는 (심볼 × 전략)마다 fetch_ticker를 따로 쳐서 50심볼·4전략이면
# 분당 200회였다. 레이트리밋보다 문제는 **지연**이었다: 순차 HTTP 왕복이
# 패스당 수십 초라 SL 체크 실주기가 60초가 아니라 85초까지 늘어났다.
_price_cache: dict = {}          # {ccxt_sym: (price, ts)}
_PRICE_FRESH_TTL = 25            # 한 패스 안에서 재사용할 신선도(초)


def _prefetch_prices(ccxt_syms: list) -> int:
    """전 심볼 시세를 배치 1회로 채운다. 반환: 채운 심볼 수.
    (/api/v3/ticker/price 다중조회 = weight 4 — 개별 조회 대비 100배 절약)"""
    syms = sorted(set(s for s in ccxt_syms if s))
    if not syms:
        return 0
    now = time.time()
    try:
        prices = _get_ex().fetch_last_prices(syms)
        n = 0
        for sym, row in (prices or {}).items():
            px = float((row or {}).get('price') or 0)
            if px > 0:
                _price_cache[sym] = (px, now)
                _last_known_price[sym] = (px, now)
                n += 1
        return n
    except Exception as e:
        log.warning(f'배치 시세 조회 실패(개별 조회로 폴백): {e}')
        return 0


def _get_price(ccxt_sym: str) -> float:
    """현재가 조회. 배치 프리페치 캐시 → 개별 조회 → 최대 2분 캐시 폴백."""
    hit = _price_cache.get(ccxt_sym)
    if hit and (time.time() - hit[1]) < _PRICE_FRESH_TTL:
        return hit[0]
    for attempt in range(3):
        try:
            ticker = _get_ex().fetch_ticker(ccxt_sym)
            price = float(ticker['last'] or ticker['close'] or 0)
            if price > 0:
                _last_known_price[ccxt_sym] = (price, time.time())
                _price_cache[ccxt_sym] = (price, time.time())
                return price
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                log.warning(f'[{ccxt_sym}] 현재가 조회 3회 실패: {e}')
    # 캐시 폴백
    cached = _last_known_price.get(ccxt_sym)
    if cached and (time.time() - cached[1]) < _PRICE_CACHE_TTL:
        log.warning(f'[{ccxt_sym}] 캐시 가격 사용: {cached[0]:.4f} (age {time.time()-cached[1]:.0f}s)')
        return cached[0]
    return 0.0


# ══════════════════════════════════════════════════════════════
#  캔들 캐시
# ══════════════════════════════════════════════════════════════

class CandleCache:
    """TTL 기반 캔들 캐시 (API 부하 절감).

    24시간 상시 구동이라 정리가 없으면 계속 자란다: 유니버스가
    UNIVERSE_REFRESH_HOURS마다 갈리는데 빠진 심볼의 캔들(1D 600봉 ≈ 155KB,
    4H 300봉 ≈ 77KB)이 그대로 남기 때문. 만료 엔트리를 주기적으로 버려
    현재 유니버스 크기에서 고정시킨다.
    """

    def __init__(self, ttl: int = SPOT_CANDLE_CACHE_TTL):
        self._cache: dict = {}
        self._locks: dict = {}
        self._meta_lock = threading.Lock()
        self._ttl = ttl
        self._last_sweep = time.time()
        self._sweep_every = max(ttl * 2, 600)

    def _sweep(self) -> int:
        """TTL이 한참 지난 엔트리 제거. 반환: 제거 개수."""
        now = time.time()
        if now - self._last_sweep < self._sweep_every:
            return 0
        stale_after = self._ttl * 3          # 여유를 두고 확실히 죽은 것만
        with self._meta_lock:
            self._last_sweep = now
            dead = [k for k, (_, ts) in self._cache.items()
                    if now - ts > stale_after]
            for k in dead:
                self._cache.pop(k, None)
                # 락 객체는 남긴다. 다른 스레드가 그 키의 임계구역 안에 있는데
                # 여기서 버리면, 뒤이어 들어온 스레드가 **새 락**을 만들어
                # 상호배제가 깨진다(중복 API 호출 + 캐시 동시 쓰기).
                # Lock 하나는 수십 바이트라 남겨도 무해하다.
        if dead:
            log.info(f'[캐시] 만료 캔들 {len(dead)}건 정리 (잔여 {len(self._cache)}건)')
        return len(dead)

    def _get_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, ex, ccxt_sym: str, timeframe: str, limit: int) -> list:
        self._sweep()
        key = f'{ccxt_sym}_{timeframe}'
        lock = self._get_lock(key)
        with lock:
            cached, ts = self._cache.get(key, (None, 0))
            if cached and (time.time() - ts) < self._ttl:
                return cached
            for attempt in range(3):
                try:
                    data = ex.fetch_ohlcv(ccxt_sym, timeframe, limit=limit)
                    if data:
                        self._cache[key] = (data, time.time())
                        return data
                except Exception as e:
                    if attempt == 2:
                        log.warning(f'[캐시] {ccxt_sym} {timeframe} 로드 실패: {e}')
                    time.sleep(1)
            return cached or []


_candle_cache = CandleCache()


# ══════════════════════════════════════════════════════════════
#  데이터베이스
# ══════════════════════════════════════════════════════════════

SPOT_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
_db_lock = threading.Lock()


@contextmanager
def _db_conn():
    conn = sqlite3.connect(str(SPOT_DB_FILE), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_spot_db():
    """DB 스키마 초기화."""
    with _db_lock, _db_conn() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS spot_positions (
            strategy     TEXT NOT NULL,
            symbol       TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            sl           REAL NOT NULL,
            tp           REAL DEFAULT 0,
            qty_tokens   REAL NOT NULL,
            cost_usdt    REAL NOT NULL,
            risk_pct     REAL NOT NULL,
            exit_type    TEXT DEFAULT 'sl_tp',
            max_hold_bars INTEGER DEFAULT 0,
            bars_held    INTEGER DEFAULT 0,
            peak_price   REAL DEFAULT 0,
            entry_ts     TEXT NOT NULL,
            regime       TEXT DEFAULT '',
            sl_order_id  TEXT DEFAULT '',
            tp_order_id  TEXT DEFAULT '',
            PRIMARY KEY (strategy, symbol)
        );

        CREATE TABLE IF NOT EXISTS spot_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy    TEXT,
            symbol      TEXT,
            entry_price REAL,
            exit_price  REAL,
            qty_tokens  REAL,
            cost_usdt   REAL,
            pnl_usdt    REAL,
            pnl_pct     REAL,
            pnl_r       REAL,
            hold_hours  REAL,
            reason      TEXT,
            entry_ts    TEXT,
            exit_ts     TEXT,
            regime      TEXT,
            fee_usdt    REAL DEFAULT 0,
            dry_run     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS spot_config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        -- 인덱스: 거래가 쌓일수록(5만 건 기준 3~4ms) 느려지는 쿼리용.
        -- 대시보드 기간 필터, S5 SL 쿨다운, 동봉 재진입 중복체크가 대상.
        -- (Kelly/건강도는 rowid 역방향 + LIMIT이라 인덱스가 불필요)
        CREATE INDEX IF NOT EXISTS ix_trades_exit
            ON spot_trades(exit_ts);
        CREATE INDEX IF NOT EXISTS ix_trades_strat_sym
            ON spot_trades(strategy, symbol, entry_ts);
        CREATE INDEX IF NOT EXISTS ix_trades_reason
            ON spot_trades(strategy, symbol, reason, exit_ts);
        """)
        # 마이그레이션: 기존 DB에 주문 ID 컬럼이 없으면 추가
        cols = [r[1] for r in conn.execute('PRAGMA table_info(spot_positions)').fetchall()]
        if 'sl_order_id' not in cols:
            conn.execute("ALTER TABLE spot_positions ADD COLUMN sl_order_id TEXT DEFAULT ''")
            log.info('[DB] spot_positions.sl_order_id 컬럼 추가 (마이그레이션)')
        if 'tp_order_id' not in cols:
            conn.execute("ALTER TABLE spot_positions ADD COLUMN tp_order_id TEXT DEFAULT ''")
            log.info('[DB] spot_positions.tp_order_id 컬럼 추가 (마이그레이션)')
        tcols = [r[1] for r in conn.execute('PRAGMA table_info(spot_trades)').fetchall()]
        if 'dry_run' not in tcols:
            conn.execute('ALTER TABLE spot_trades ADD COLUMN dry_run INTEGER DEFAULT 0')
            log.info('[DB] spot_trades.dry_run 컬럼 추가 (마이그레이션)')
    log.info('[DB] atlas_spot.db 초기화 완료')


def _cfg_get(key: str, default: str = '') -> str:
    with _db_lock, _db_conn() as conn:
        row = conn.execute('SELECT value FROM spot_config WHERE key=?', (key,)).fetchone()
        return row['value'] if row else default


def _cfg_set(key: str, value: str):
    with _db_lock, _db_conn() as conn:
        conn.execute('INSERT OR REPLACE INTO spot_config(key,value) VALUES(?,?)', (key, value))


def _save_position(strategy: str, symbol: str, entry_price: float, sl: float,
                   tp: float, qty: float, cost: float, risk_pct: float,
                   exit_type: str, max_hold: int, regime: str):
    with _db_lock, _db_conn() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO spot_positions
        (strategy, symbol, entry_price, sl, tp, qty_tokens, cost_usdt,
         risk_pct, exit_type, max_hold_bars, bars_held, peak_price, entry_ts, regime)
        VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?)
        """, (strategy, symbol, entry_price, sl, tp, qty, cost,
              risk_pct, exit_type, max_hold, entry_price,
              datetime.now(timezone.utc).isoformat(), regime))


def _load_position(strategy: str, symbol: str) -> Optional[dict]:
    with _db_lock, _db_conn() as conn:
        row = conn.execute(
            'SELECT * FROM spot_positions WHERE strategy=? AND symbol=?',
            (strategy, symbol)
        ).fetchone()
        return dict(row) if row else None


def _load_all_positions() -> list[dict]:
    with _db_lock, _db_conn() as conn:
        rows = conn.execute('SELECT * FROM spot_positions').fetchall()
        return [dict(r) for r in rows]


def _update_position_sl(strategy: str, symbol: str, new_sl: float, peak: float):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            'UPDATE spot_positions SET sl=?, peak_price=? WHERE strategy=? AND symbol=?',
            (new_sl, peak, strategy, symbol)
        )


def _update_position_tp(strategy: str, symbol: str, new_tp: float):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            'UPDATE spot_positions SET tp=? WHERE strategy=? AND symbol=?',
            (new_tp, strategy, symbol)
        )


def _delete_position(strategy: str, symbol: str):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            'DELETE FROM spot_positions WHERE strategy=? AND symbol=?',
            (strategy, symbol)
        )


def _update_position_order_id(strategy: str, symbol: str, order_id: str,
                              tp_order_id: str = ''):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            'UPDATE spot_positions SET sl_order_id=?, tp_order_id=? WHERE strategy=? AND symbol=?',
            (order_id, tp_order_id, strategy, symbol)
        )


# ══════════════════════════════════════════════════════════════
#  거래소 측 스탑 주문 (봇 다운 중에도 SL 집행 — 소프트웨어 SL은 백업)
# ══════════════════════════════════════════════════════════════

def _net_filled_qty(order: dict, ccxt_sym: str, requested: float) -> float:
    """매수 주문의 **실수령 수량** — 기초자산으로 차감된 수수료를 뺀 값.

    바이낸스 현물 매수는 수수료를 기초자산에서 뗀다(BNB 결제 시 제외).
    즉 executedQty(=filled)를 그대로 매도하려 하면 보유량을 초과해
    -2010 insufficient balance로 거절된다 — 거래소 측 SL/TP가 등록되지
    않아 봇 다운 시 무방비가 되는 원인이므로 반드시 순수량을 써야 한다.
    """
    filled = float(order.get('filled') or requested or 0.0)
    if filled <= 0:
        return 0.0
    base = ccxt_sym.split('/')[0].upper()
    fee_base = 0.0
    fees = list(order.get('fees') or [])
    if not fees and order.get('fee'):
        fees = [order['fee']]
    for f in fees:
        if f and str(f.get('currency') or '').upper() == base:
            fee_base += abs(float(f.get('cost') or 0.0))
    if fee_base <= 0:
        # ccxt 파싱이 비었을 때를 대비해 원시 응답의 fills를 직접 확인
        for fill in ((order.get('info') or {}).get('fills') or []):
            if str(fill.get('commissionAsset') or '').upper() == base:
                fee_base += abs(float(fill.get('commission') or 0.0))
    return max(filled - fee_base, 0.0)


def _sellable_qty(ccxt_sym: str, qty: float) -> float:
    """거래소 수량 정밀도로 **내림** 적용 — 보유량 초과 주문 방지."""
    try:
        return float(_get_ex().amount_to_precision(ccxt_sym, qty))
    except Exception:
        return qty


def _cancel_orphan_sell_orders(strategy: str, symbol: str, ccxt_sym: str,
                               base_asset: str, current_free: float) -> float:
    """해당 심볼의 미체결 매도 주문을 모두 취소하고 갱신된 free 잔고를 반환.

    DB에 ID가 남지 않은 '고아' 보호주문(OCO 등록 후 저장 직전 프로세스 종료,
    응답 파싱 실패 등)이 수량을 잠그면 free=0이 된다. 이를 수동매도로 오판해
    살아있는 포지션을 삭제하는 것을 막기 위해, 판정 전에 잠금을 해제한다.
    실패해도 호출측 판정을 바꾸지 않도록 원래 값을 그대로 돌려준다.
    """
    try:
        ex = _get_ex()
        # 같은 심볼을 다른 전략이 동시 보유할 수 있다(예: S3 BTCUSDT + S6 BTCUSDT).
        # 그 전략의 보호주문까지 취소하면 남의 포지션을 무방비로 만든다.
        # DB가 알고 있는 '남의 주문 ID'는 건드리지 않는다.
        others = set()
        same_symbol_others = 0
        for p in _load_all_positions():
            if p['symbol'] == symbol and p['strategy'] == strategy:
                continue
            if p['symbol'] == symbol:
                same_symbol_others += 1
            for k in ('sl_order_id', 'tp_order_id'):
                oid = str(p.get(k) or '')
                if oid:
                    others.add(oid)
        if same_symbol_others:
            # 같은 심볼을 다른 전략도 보유 중이면, DB에 없는 주문이
            # 누구 것인지 판별할 수 없다(그 전략의 OCO도 ID 파싱에 실패하면
            # 똑같이 DB에 안 남는다). 잘못 취소하면 남의 포지션을 무방비로
            # 만들고 그 코인까지 팔아버리므로, 아예 손대지 않는다.
            log.warning(f'[{strategy}] {symbol} 고아주문 정리 생략 — 같은 심볼을 '
                        f'다른 전략 {same_symbol_others}건이 보유 중이라 주문 귀속 불가')
            return current_free
        open_orders = ex.fetch_open_orders(ccxt_sym) or []
        sells = [o for o in open_orders
                 if str(o.get('side', '')).lower() == 'sell'
                 and str(o.get('id', '') or '') not in others]
        if not sells:
            return current_free
        log.warning(f'[{strategy}] {symbol} free=0이지만 미체결 매도주문 '
                    f'{len(sells)}건 발견 — 고아 주문으로 보고 취소 후 재확인'
                    + (f' (타 전략 주문 {len(others)}건은 보존)' if others else ''))
        for o in sells:
            _cancel_stop_order(strategy, symbol, ccxt_sym, str(o.get('id', '') or ''))
        refreshed = float((ex.fetch_balance()['free'] or {}).get(base_asset, 0) or 0)
        if refreshed > 0:
            _tg(f'ℹ️ [{strategy}] {symbol} 고아 보호주문 {len(sells)}건 취소 '
                f'→ 잔고 {refreshed:.6f} 회수 (허위 수동매도 처리 방지)')
        return refreshed
    except Exception as e:
        log.warning(f'[{strategy}] {symbol} 고아주문 확인 실패(무시): {e}')
        return current_free


def _place_stop_loss_order(strategy: str, symbol: str, ccxt_sym: str,
                           qty: float, sl_price: float) -> str:
    """거래소에 STOP_LOSS_LIMIT 매도 주문 등록. 실패 시 '' 반환(소프트웨어 SL 폴백)."""
    if not SPOT_EXCHANGE_STOP:
        return ''
    qty = _sellable_qty(ccxt_sym, qty)
    limit_price = sl_price * (1 - SPOT_STOP_LIMIT_GAP)
    if qty <= 0 or qty * limit_price < SPOT_MIN_ORDER_USDT:
        log.info(f'[{strategy}] {symbol} 스탑주문 생략: NOTIONAL 미달 '
                 f'(${qty * limit_price:.2f} < ${SPOT_MIN_ORDER_USDT})')
        return ''
    try:
        order = _get_ex().create_order(
            ccxt_sym, 'limit', 'sell', qty, limit_price,
            params={'stopPrice': sl_price, 'timeInForce': 'GTC'})
        order_id = str(order.get('id', '') or '')
        log.info(f'[{strategy}] {symbol} 거래소 스탑주문 등록: id={order_id} '
                 f'trigger={sl_price:,.4f} limit={limit_price:,.4f}')
        return order_id
    except Exception as e:
        log.warning(f'[{strategy}] {symbol} 스탑주문 등록 실패(소프트웨어 SL로 폴백): {e}')
        _tg(f'⚠️ [{strategy}] {symbol} 거래소 스탑주문 실패 — 소프트웨어 SL만 작동: {e}')
        return ''


def _place_protective_orders(strategy: str, symbol: str, ccxt_sym: str,
                             qty: float, sl_price: float,
                             tp_price: float = 0.0) -> tuple:
    """거래소 보호 주문 등록: (sl_order_id, tp_order_id) 반환.
    tp_price > 0이면 OCO(SL+TP 연동, 한쪽 체결 시 반대쪽 자동취소) 시도,
    실패·미해당 시 STOP_LOSS_LIMIT 단독으로 폴백. 최종 실패는 ('','') —
    소프트웨어 SL/TP가 커버한다."""
    if not SPOT_EXCHANGE_STOP:
        return '', ''
    qty = _sellable_qty(ccxt_sym, qty)
    limit_price = sl_price * (1 - SPOT_STOP_LIMIT_GAP)
    if qty <= 0 or qty * limit_price < SPOT_MIN_ORDER_USDT:
        log.info(f'[{strategy}] {symbol} 보호주문 생략: NOTIONAL 미달 '
                 f'(${qty * limit_price:.2f} < ${SPOT_MIN_ORDER_USDT})')
        return '', ''
    if SPOT_EXCHANGE_OCO and tp_price and tp_price > 0:
        try:
            ex = _get_ex()

            def _p(v):
                try:
                    return ex.price_to_precision(ccxt_sym, v)
                except Exception:
                    return f'{v:.8f}'.rstrip('0').rstrip('.')

            def _a(v):
                try:
                    return ex.amount_to_precision(ccxt_sym, v)
                except Exception:
                    return f'{v:.8f}'.rstrip('0').rstrip('.')

            # Binance Spot OCO (POST /api/v3/orderList/oco) — SELL:
            # above = TP 지정가(LIMIT_MAKER), below = SL(STOP_LOSS_LIMIT)
            res = ex.privatePostOrderListOco({
                'symbol': ccxt_sym.replace('/', ''),
                'side': 'SELL',
                'quantity': _a(qty),
                'aboveType': 'LIMIT_MAKER',
                'abovePrice': _p(tp_price),
                'belowType': 'STOP_LOSS_LIMIT',
                'belowStopPrice': _p(sl_price),
                'belowPrice': _p(limit_price),
                'belowTimeInForce': 'GTC',
            })
            sl_id = tp_id = ''
            for o in (res.get('orderReports') or res.get('orders') or []):
                otype = str(o.get('type', ''))
                oid = str(o.get('orderId', '') or '')
                if 'STOP' in otype:
                    sl_id = oid
                elif otype in ('LIMIT_MAKER', 'LIMIT'):
                    tp_id = oid
            if sl_id:
                log.info(f'[{strategy}] {symbol} OCO 등록: SL#{sl_id}@{sl_price:,.4f} '
                         f'/ TP#{tp_id}@{tp_price:,.4f}')
                return sl_id, tp_id
            # ID를 못 읽었는데 주문은 살아있다 → 추적 불가능한 고아가 된다.
            # (수량을 잠가 이후 매도를 막고, 다른 포지션의 청산 판정까지 흔든다)
            # 폴백으로 스탑을 또 걸기 전에 반드시 이 OCO를 없앤다.
            log.warning(f'[{strategy}] {symbol} OCO 응답에서 주문ID 파싱 실패 '
                        f'— 고아 방지를 위해 해당 OCO 취소 후 스탑 단독 폴백')
            try:
                list_id = str(res.get('orderListId', '') or '')
                if list_id and list_id != '-1':
                    ex.privateDeleteOrderList({
                        'symbol': ccxt_sym.replace('/', ''), 'orderListId': list_id})
                else:
                    for o in (res.get('orderReports') or res.get('orders') or []):
                        _cancel_stop_order(strategy, symbol, ccxt_sym,
                                           str(o.get('orderId', '') or ''))
            except Exception as e_cancel:
                log.error(f'[{strategy}] {symbol} 미확인 OCO 취소 실패 — 고아 주문 '
                          f'가능성: {e_cancel}')
                _tg(f'⚠️ [{strategy}] {symbol} 추적 불가 OCO가 남았을 수 있습니다. '
                    f'거래소 미체결 주문을 확인해 주세요.')
        except Exception as e:
            log.warning(f'[{strategy}] {symbol} OCO 등록 실패(스탑 단독 폴백): {e}')
    return _place_stop_loss_order(strategy, symbol, ccxt_sym, qty, sl_price), ''


def _cancel_stop_order(strategy: str, symbol: str, ccxt_sym: str, order_id: str) -> None:
    """스탑 주문 취소. 이미 체결/취소된 주문이면 조용히 무시."""
    if not order_id:
        return
    try:
        _get_ex().cancel_order(order_id, ccxt_sym)
        log.info(f'[{strategy}] {symbol} 스탑주문 취소: id={order_id}')
    except Exception as e:
        # 이미 체결/취소된 경우 등 — 매도 진행에 지장 없으므로 경고만
        log.warning(f'[{strategy}] {symbol} 스탑주문 취소 실패(무시): {e}')


def _fetch_stop_order(ccxt_sym: str, order_id: str) -> Optional[dict]:
    """스탑 주문 상태 조회. 실패 시 None (소프트웨어 SL이 계속 커버)."""
    try:
        return _get_ex().fetch_order(order_id, ccxt_sym)
    except Exception as e:
        log.warning(f'{ccxt_sym} 스탑주문 조회 실패(무시): {e}')
        return None


def _log_trade(strategy: str, symbol: str, entry_price: float, exit_price: float,
               qty: float, cost: float, pnl_usdt: float, pnl_pct: float,
               pnl_r: float, hold_hours: float, reason: str, regime: str,
               entry_ts: str) -> float:
    """거래 기록 저장. 반환값: 수수료 차감 net PnL (day_pnl 등 리스크 로직용).
    pnl_usdt/pnl_r 컬럼은 백테스트와의 비교 일관성을 위해 gross로 유지하고,
    수수료는 fee_usdt(매수+매도 왕복)에 별도 기록한다."""
    fee = (entry_price + exit_price) * qty * BT_SPOT_FEE
    # dry-run 거래도 시뮬레이션 검토용으로 기록하되 **플래그를 남긴다**.
    # 표시는 하되 리스크 계산(Kelly·전략 건강도·자본변동 판정)에서는
    # 제외해야, 가상 손실이 실계좌 전략을 차단하는 일이 없다.
    is_dry = 1 if _state.get('dry_run') else 0
    with _db_lock, _db_conn() as conn:
        conn.execute("""
        INSERT INTO spot_trades
        (strategy, symbol, entry_price, exit_price, qty_tokens, cost_usdt,
         pnl_usdt, pnl_pct, pnl_r, hold_hours, reason, entry_ts, exit_ts, regime,
         fee_usdt, dry_run)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (strategy, symbol, entry_price, exit_price, qty, cost,
              round(pnl_usdt, 4), round(pnl_pct, 4), round(pnl_r, 4),
              round(hold_hours, 2), reason,
              entry_ts, datetime.now(timezone.utc).isoformat(),
              regime, round(fee, 4), is_dry))
    return pnl_usdt - fee


# ══════════════════════════════════════════════════════════════
#  전역 상태
# ══════════════════════════════════════════════════════════════

_state = {
    'equity':           0.0,
    'usdt_balance':     0.0,
    'peak_equity':      0.0,
    'day_pnl':          0.0,
    'day_start_eq':     0.0,
    'paused':           False,
    'universe':         [],
    'universe_ranked':  [],   # 모멘텀 정렬 순서 (RS Gate용)
    'ratchet_scale':    1.0,
    'dry_run':          False,
    'active_strategies': list(DEFAULT_ACTIVE_STRATEGIES),
    'ratchet_alert_tier': 0,     # 0=정상, 1=소프트DD, 2=하드DD — 알림 중복 방지용
    'daily_loss_alerted': False, # 일간 손실 한도 알림 중복 방지 (자정 리셋)
}
_state_lock = threading.Lock()
# 매수 경로 직렬화: 4H/1D 두 전략 루프가 동시에 진입하면 포지션 수 한도
# 체크→저장 사이(TOCTOU)에 SPOT_MAX_POSITIONS를 초과할 수 있다.
_entry_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════
#  Telegram
# ══════════════════════════════════════════════════════════════

# 텔레그램 전송 큐 — 전략 루프가 동기 HTTP(타임아웃)로 막히지 않도록
# 실제 전송은 백그라운드 워커(_tg_worker)가 처리한다.
_tg_queue: "queue.Queue[str]" = queue.Queue(maxsize=200)


def _tg(msg: str):
    """텔레그램 메시지를 큐에 넣고 즉시 반환 (논블로킹)."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        _tg_queue.put_nowait(msg)
    except queue.Full:
        # 텔레그램 장기 장애로 큐가 가득 차면 가장 오래된 메시지를 버리고 최신 우선
        try:
            _tg_queue.get_nowait()
            _tg_queue.put_nowait(msg)
        except queue.Empty:
            pass


def _tg_send_now(msg: str) -> None:
    """실제 HTTP 전송 (워커/flush 전용)."""
    try:
        url = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=5)
    except Exception:
        pass


def _tg_worker(stop_event: threading.Event) -> None:
    """큐에 쌓인 텔레그램 메시지를 백그라운드에서 순차 전송."""
    while not stop_event.is_set():
        try:
            msg = _tg_queue.get(timeout=1)
        except queue.Empty:
            continue
        _tg_send_now(msg)
    # 종료 시 남은 메시지 flush (루프 실행 중 쌓인 것)
    _tg_flush()


def _tg_flush() -> None:
    """큐에 남은 메시지를 동기적으로 모두 전송 (종료 시 유실 방지)."""
    while True:
        try:
            msg = _tg_queue.get_nowait()
        except queue.Empty:
            break
        _tg_send_now(msg)


# ══════════════════════════════════════════════════════════════
#  리스크 관리
# ══════════════════════════════════════════════════════════════

def _persist_risk_state() -> None:
    """드로다운 래칫·일간 손실 한도의 기준점을 DB에 보존.

    메모리에만 두면 프로세스 재시작(systemd Restart=on-failure 포함)마다
    peak_equity가 현재 자산으로 리셋돼 낙폭이 0으로 보이고, 래칫이
    0.40에서 1.0으로 풀린다 — 하락장에서 재시작이 반복될수록 리스크 축소
    장치가 매번 무효화되는, 정확히 최악의 방향으로 틀리는 오류였다.
    """
    # dry-run은 가상 매매라 그 손익을 실거래 기준점으로 남기면 안 된다.
    # (같은 DB를 쓰므로 다음 실거래 기동이 가상 결과를 이어받게 된다)
    if _state.get('dry_run'):
        return
    try:
        with _state_lock:
            peak = _state['peak_equity']
            day_pnl = _state['day_pnl']
            day_eq = _state['day_start_eq']
            alerted = _state['daily_loss_alerted']
        _cfg_set('peak_equity', f'{peak:.8f}')
        _cfg_set('day_pnl', f'{day_pnl:.8f}')
        _cfg_set('day_start_eq', f'{day_eq:.8f}')
        _cfg_set('day_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        _cfg_set('daily_loss_alerted', '1' if alerted else '0')
    except Exception as e:
        log.warning(f'[상태보존] 실패(무시): {e}')


def _restore_risk_state(total: float) -> None:
    """기동 시 보존된 리스크 기준점 복원. 실패해도 현재 자산 기준으로 진행."""
    try:
        saved_peak = float(_cfg_get('peak_equity', '0') or 0)
    except (TypeError, ValueError):
        saved_peak = 0.0
    with _state_lock:
        # 피크는 '역대 최고'라야 낙폭이 의미를 갖는다 — 큰 쪽을 취한다.
        _state['peak_equity'] = max(saved_peak, total)
    if saved_peak > total:
        dd = (saved_peak - total) / saved_peak * 100
        log.info(f'[복원] 피크 자산 ${saved_peak:,.2f} 복원 — 현재 낙폭 {dd:.1f}% '
                 f'(리셋했다면 래칫이 잘못 풀렸을 상황)')

    # 일간 손실 한도: 같은 UTC 날짜에 재시작한 경우에만 이어받는다.
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if _cfg_get('day_date', '') == today:
        try:
            with _state_lock:
                _state['day_pnl'] = float(_cfg_get('day_pnl', '0') or 0)
                _state['day_start_eq'] = float(_cfg_get('day_start_eq', '0') or 0) or total
                _state['daily_loss_alerted'] = _cfg_get('daily_loss_alerted', '0') == '1'
            log.info(f'[복원] 당일 PnL ${_state["day_pnl"]:+.2f} '
                     f'(기준자산 ${_state["day_start_eq"]:,.2f}) 이어받음')
        except (TypeError, ValueError) as e:
            log.warning(f'[복원] 일간 상태 파싱 실패 — 새로 시작: {e}')
            with _state_lock:
                _state['day_start_eq'] = total
    else:
        with _state_lock:
            _state['day_start_eq'] = total
    _persist_risk_state()


def _rebase_peak_on_capital_flow(total: float) -> None:
    """입출금으로 자산이 변하면 피크를 같은 비율로 재조정.

    피크를 영구 보존하면 **출금**도 낙폭으로 오인해 래칫이 영영 풀리지
    않는다($10,000 중 $2,000 출금 → -20% 낙폭으로 보여 리스크 0.4배 고정).
    '열린 포지션이 없을 때'만 판정하므로 미실현 손익과 혼동되지 않고,
    실현 손익(day_pnl)으로 설명되는 변동은 제외한다.
    """
    try:
        if _load_all_positions():
            return                     # 미실현 변동과 구분 불가 → 판정하지 않음
        with _state_lock:
            prev = _state.get('last_flat_equity') or 0.0
            prev_id = _state.get('last_flat_trade_id') or 0
            peak = _state['peak_equity']
        # 실현손익은 **누적 거래기록**에서 뽑는다. day_pnl은 자정마다 0으로
        # 리셋되므로, 두 판정 시점 사이에 날짜가 바뀌면 그동안의 실현손실이
        # 통째로 '설명되지 않는 변동'이 되어 출금으로 오인된다 —
        # 며칠에 걸친 손실이 래칫을 스스로 풀어버리는 최악의 오분류.
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                'SELECT COALESCE(MAX(id), ?) AS last_id, '
                'COALESCE(SUM(pnl_usdt - COALESCE(fee_usdt, 0)), 0) AS realized '
                'FROM spot_trades WHERE id > ? AND COALESCE(dry_run,0)=0', (prev_id, prev_id)
            ).fetchone()
        last_id  = int(row['last_id'] or prev_id)
        realized = float(row['realized'] or 0.0)
        if prev > 0:
            unexplained = (total - prev) - realized
            if abs(unexplained) / prev >= SPOT_CAPITAL_FLOW_PCT:
                ratio = total / prev if prev > 0 else 1.0
                new_peak = max(total, peak * ratio)
                with _state_lock:
                    _state['peak_equity'] = new_peak
                    _state.pop('ratchet_floor', None)
                kind = '입금' if unexplained > 0 else '출금'
                log.info(f'[자본변동] {kind} ${abs(unexplained):,.2f} 감지 — '
                         f'피크 ${peak:,.2f} → ${new_peak:,.2f} 재조정')
                _tg(f'ℹ️ {kind} ${abs(unexplained):,.2f} 감지 — 드로다운 기준(피크)을 '
                    f'${new_peak:,.2f}로 재조정했습니다. (출금을 손실로 오인해 '
                    f'리스크가 잠기는 것을 방지)')
        with _state_lock:
            _state['last_flat_equity'] = total
            _state['last_flat_trade_id'] = last_id
    except Exception as e:
        log.warning(f'[자본변동] 판정 실패(무시): {e}')


def _get_spot_equity() -> tuple[float, float]:
    """총자산 = USDT + 보유 코인 현재가. Returns: (total_equity, usdt_balance)."""
    try:
        ex = _get_ex()
        bal = ex.fetch_balance({'type': 'spot'})
        usdt = float(bal.get('USDT', {}).get('total', 0) or 0)
        total = usdt
        for currency, data in bal.items():
            if currency in ('USDT', 'info', 'free', 'used', 'total'):
                continue
            if not isinstance(data, dict):
                continue
            qty = float(data.get('total', 0) or 0)
            if qty < 1e-8:
                continue
            try:
                sym = f'{currency}/USDT'
                ticker = ex.fetch_ticker(sym)
                price = float(ticker['last'] or 0)
                total += qty * price
            except Exception:
                pass
        return total, usdt
    except Exception as e:
        log.warning(f'[잔고] 조회 실패: {e}')
        return _state['equity'], _state['usdt_balance']


def _get_kelly_scale(strategy: str) -> float:
    """Kelly 스케일 계산 (최근 거래 기반, half-Kelly, 조건부 상한 2.00).
    raw Kelly는 승률 추정오차에 과베팅하므로 SPOT_KELLY_FRACTION(0.5)을 곱한다."""
    with _db_lock, _db_conn() as conn:
        rows = conn.execute(
            'SELECT pnl_r FROM spot_trades WHERE strategy=? AND COALESCE(dry_run,0)=0 '
            'ORDER BY id DESC LIMIT 200',
            (strategy,)
        ).fetchall()
    if len(rows) < SPOT_KELLY_MIN_TRADES:
        return SPOT_KELLY_SCALE_MIN
    pnl_r  = [r['pnl_r'] for r in rows]
    wins   = [r for r in pnl_r if r > 0]
    losses = [r for r in pnl_r if r <= 0]
    if not wins:
        # 전패 구간 — Kelly는 정의되지 않는다. 이때 1.0(정상 사이즈)을 주면
        # 표본 부족(0.30)보다 오히려 크게 베팅하는 역전이 생긴다.
        # 최악의 증거이므로 최소 스케일로 내린다.
        return SPOT_KELLY_SCALE_MIN
    if not losses:
        # 전승 — b를 계산할 수 없다. 표본이 편향됐을 뿐이므로 상향하지 않고 중립 유지.
        return 1.0
    wr    = len(wins) / len(pnl_r)
    avg_w = float(np.mean(wins))
    avg_l = abs(float(np.mean(losses)))
    b     = avg_w / avg_l if avg_l > 0 else 1.0
    kelly = (wr - (1 - wr) / b) * SPOT_KELLY_FRACTION if b > 0 else 0.0
    # Profit Factor 계산 (조건부 Kelly 상한 활성화)
    gross_w = sum(abs(r) for r in wins)
    gross_l = sum(abs(r) for r in losses)
    pf      = gross_w / gross_l if gross_l > 0 else 0.0
    k_max   = SPOT_KELLY_SCALE_MAX if (wr >= SPOT_KELLY_WR_THRESH and
                                        pf >= SPOT_KELLY_PF_THRESH) else 1.50
    return float(max(SPOT_KELLY_SCALE_MIN, min(k_max, kelly)))


def _get_ratchet_scale() -> float:
    """Drawdown Ratchet 스케일.
    회복 조건: DD 바닥 대비 +SPOT_RATCHET_RECOVER(15%) 상승 시 스케일 복원.
    읽기-수정-쓰기(alert_tier/floor)가 잔고폴러·양쪽 전략 루프와 경합하지
    않도록 _state_lock으로 원자화한다. 알림 전송은 락 밖에서 수행.
    """
    msg = None
    with _state_lock:
        equity = _state['equity']
        peak   = _state['peak_equity']
        if peak <= 0 or equity <= 0:
            return 1.0
        dd = (peak - equity) / peak
        if dd >= SPOT_RATCHET_DD_HARD:
            if _state['ratchet_alert_tier'] < 2:
                _state['ratchet_alert_tier'] = 2
                msg = (f'🔴 [위험] 하드 드로다운 진입: 피크 ${peak:,.2f} → 현재 ${equity:,.2f} '
                       f'(-{dd*100:.1f}%) — 리스크 스케일 40%로 축소')
            # 바닥(floor) 추적: ratchet_floor에 최저점 기록
            # 최초 진입 시에도 floor를 _state에 저장해야 함 — 그렇지 않으면 다음 호출에서도
            # get()의 기본값이 항상 "현재" equity가 되어 recover_pct가 영원히 0이 되는 버그 발생
            if 'ratchet_floor' not in _state:
                _state['ratchet_floor'] = equity
            floor = _state['ratchet_floor']
            if equity < floor:
                _state['ratchet_floor'] = equity
                floor = equity
            # 바닥 대비 회복률 확인
            recover_pct = (equity - floor) / floor if floor > 0 else 0.0
            if recover_pct >= SPOT_RATCHET_RECOVER:
                _state['ratchet_floor'] = equity  # 회복 기준점 리셋
                _state['ratchet_alert_tier'] = 1
                scale = 0.70   # Hard DD 회복 → 중간 단계로 복원
            else:
                scale = 0.40
        elif dd >= SPOT_RATCHET_DD_THRESH:
            if _state['ratchet_alert_tier'] < 1:
                _state['ratchet_alert_tier'] = 1
                msg = (f'⚠️ [경고] 드로다운 진입: 피크 ${peak:,.2f} → 현재 ${equity:,.2f} '
                       f'(-{dd*100:.1f}%) — 리스크 스케일 70%로 축소')
            _state.pop('ratchet_floor', None)  # 소프트 DD 구간 — floor 초기화
            scale = 0.70
        else:
            if _state['ratchet_alert_tier'] > 0:
                _state['ratchet_alert_tier'] = 0
                msg = (f'✅ [회복] 드로다운 해소: 현재 ${equity:,.2f} '
                       f'(피크 대비 -{dd*100:.1f}%) — 리스크 스케일 정상 복원')
            _state.pop('ratchet_floor', None)
            scale = 1.0
    if msg:
        _tg(msg)
    return scale


def _get_strategy_health_scale(strategy: str) -> float:
    """실계좌 net 성과 기반 자기교정 스케일.
    최근 SPOT_HEALTH_WINDOW_DAYS일 내 해당 전략 표본이
    SPOT_HEALTH_MIN_TRADES 이상일 때:
      net PF < SPOT_HEALTH_PF_HARD(0.7) → 0.0 (신규 진입 차단)
      net PF < SPOT_HEALTH_PF_SOFT(1.0) → SPOT_HEALTH_SOFT_SCALE(0.5) 감봉
    표본 부족 시 개입하지 않는다(1.0).

    **시간 창이 필수인 이유**: 차단된 전략은 신규 거래를 만들지 못해
    표본이 그대로 굳는다. 전체 이력을 보면 PF가 영원히 갱신되지 않아
    '성과 회복 시 자동 해제'가 원리적으로 불가능해진다(영구 차단).
    창을 두면 불량 구간이 지나가면서 표본이 최소치 아래로 떨어져
    자동으로 재진입 기회를 얻는다.
    """
    since = (datetime.now(timezone.utc)
             - timedelta(days=SPOT_HEALTH_WINDOW_DAYS)).isoformat()
    with _db_lock, _db_conn() as conn:
        rows = conn.execute(
            'SELECT pnl_usdt - COALESCE(fee_usdt, 0) AS net FROM spot_trades '
            'WHERE strategy=? AND exit_ts >= ? AND COALESCE(dry_run,0)=0 '
            'ORDER BY id DESC LIMIT 200',
            (strategy, since)
        ).fetchall()
    if len(rows) < SPOT_HEALTH_MIN_TRADES:
        return 1.0
    nets = [r['net'] for r in rows]
    gross_w = sum(n for n in nets if n > 0)
    gross_l = abs(sum(n for n in nets if n < 0))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')
    if pf < SPOT_HEALTH_PF_HARD:
        return 0.0
    if pf < SPOT_HEALTH_PF_SOFT:
        return SPOT_HEALTH_SOFT_SCALE
    return 1.0


def _get_spot_funding(sym: str) -> float:
    """Binance 선물 펀딩비 조회 (현물 추세추종 진입 필터용). 실패 시 0.0 반환."""
    try:
        ccxt_sym = sym.replace('USDT', '/USDT:USDT')
        info = _get_ex_futures().fetch_funding_rate(ccxt_sym)
        return float(info.get('fundingRate', 0))
    except Exception:
        return 0.0


def _get_momentum_rank_pct(sym: str) -> float:
    """심볼의 모멘텀 랭킹 상위 비율 (0.0=최상위, 1.0=최하위). 데이터 없으면 0.5."""
    ranked = _state.get('universe_ranked', [])
    universe = _state.get('universe', [])
    ref = ranked if ranked else universe
    if not ref or sym not in ref:
        return 0.5
    return ref.index(sym) / len(ref)


def _check_buying_power(cost_usdt: float) -> tuple[bool, str]:
    """USDT 잔고 충분 여부 확인."""
    equity = _state['equity']
    usdt   = _state['usdt_balance']
    reserve = equity * SPOT_RESERVE_PCT
    available = usdt - reserve
    if cost_usdt < SPOT_MIN_ORDER_USDT:                          # Binance NOTIONAL 필터
        return False, f'주문금액 ${cost_usdt:.2f} < 최소 ${SPOT_MIN_ORDER_USDT:.0f} (NOTIONAL)'
    if available < SPOT_MIN_ORDER_USDT:
        return False, f'USDT 부족 (가용 ${available:.0f})'
    if cost_usdt > available:
        return False, f'매수 금액 ${cost_usdt:.0f} > 가용 ${available:.0f}'
    return True, ''


# ══════════════════════════════════════════════════════════════
#  포지션 진입/청산
# ══════════════════════════════════════════════════════════════

def _spot_buy(strategy: str, symbol: str, ccxt_sym: str,
              sig: dict, price: float, regime: str) -> bool:
    """현물 매수 실행. 4H/1D 두 전략 루프가 동시에 진입해도 포지션 수
    한도 체크→저장이 원자적이 되도록 _entry_lock으로 직렬화한다."""
    with _entry_lock:
        return _spot_buy_locked(strategy, symbol, ccxt_sym, sig, price, regime)


def _spot_buy_locked(strategy: str, symbol: str, ccxt_sym: str,
                     sig: dict, price: float, regime: str) -> bool:
    if _state['paused']:
        log.info(f'[{strategy}] {symbol} 매수 차단 (일시정지)')
        return False

    if SPOT_KILL_SWITCH.exists():
        return False

    equity = _state['equity']

    # 포지션 수 제한 — 자본 연동: 슬롯당 최소 $SPOT_EQUITY_PER_SLOT
    # (소액 계좌에서 과분할 → NOTIONAL 턱걸이 + 수수료 드래그 방지)
    max_pos = min(SPOT_MAX_POSITIONS,
                  max(1, int(equity // SPOT_EQUITY_PER_SLOT)))
    all_pos = _load_all_positions()
    if len(all_pos) >= max_pos:
        log.info(f'[{strategy}] {symbol} 매수 차단 (최대 포지션 {max_pos}개 — 자본 연동)')
        return False

    # 중복 포지션 방지
    if _load_position(strategy, symbol):
        return False

    # 전략 건강도 자기교정: 실계좌 net PF 기준 감봉/차단
    health = _get_strategy_health_scale(strategy)
    if health <= 0:
        log.warning(f'[{strategy}] {symbol} 매수 차단: 실계좌 net PF < '
                    f'{SPOT_HEALTH_PF_HARD} (표본 {SPOT_HEALTH_MIN_TRADES}건+)')
        if not _state.get(f'health_blocked_{strategy}'):
            _state[f'health_blocked_{strategy}'] = True
            _tg(f'⛔ [{strategy}] 실계좌 net PF < {SPOT_HEALTH_PF_HARD} — '
                f'신규 진입 자동 차단 (성과 회복 시 자동 해제)')
        return False
    _state.pop(f'health_blocked_{strategy}', None)

    kelly    = _get_kelly_scale(strategy)
    ratchet  = _get_ratchet_scale()
    if regime == REGIME_WEAK_TREND:
        r_scale = WEAK_TREND_RISK_SCALE
    elif regime == REGIME_TRENDING_DOWN:
        r_scale = TRENDING_DOWN_RISK_SCALE
    else:
        r_scale = 1.0
    # 펀딩비/모멘텀 스케일 (strategy_timeframe_loop에서 sig에 주입)
    funding_scale = sig.pop('_funding_scale', 1.0)
    rs_scale      = sig.pop('_rs_scale', 1.0)

    entry_price = price
    sl          = sig['sl']
    tp          = sig['tp']
    sl_dist     = abs(entry_price - sl)
    if sl_dist <= 0:
        return False

    # SL 거리 상한 필터: 넓은 SL은 소형 계좌에서 최소 주문 미달 원인
    _sl_pct = sl_dist / entry_price
    if _sl_pct > SPOT_MAX_SL_PCT:
        log.info(f'[{strategy}] {symbol} 매수 차단: SL거리 {_sl_pct*100:.1f}% > 상한 {SPOT_MAX_SL_PCT*100:.0f}%')
        return False

    adj_risk   = SPOT_BASE_RISK_PCT * kelly * ratchet * r_scale * funding_scale * rs_scale * health
    risk_usd   = equity * adj_risk
    qty        = risk_usd / sl_dist
    cost_usdt  = qty * entry_price

    # 배분 한도
    if cost_usdt > equity * SPOT_MAX_ALLOC_PCT:
        cost_usdt = equity * SPOT_MAX_ALLOC_PCT
        qty       = cost_usdt / entry_price
        adj_risk  = (qty * sl_dist) / equity

    # 잔고 확인
    ok, reason = _check_buying_power(cost_usdt)
    if not ok:
        log.info(f'[{strategy}] {symbol} 매수 차단: {reason}')
        return False

    # 일간 손실 한도
    day_loss_pct = _state['day_pnl'] / max(_state['day_start_eq'], 1)
    if day_loss_pct <= SPOT_DAILY_LOSS_LIMIT:
        log.warning(f'[{strategy}] 일간 손실 한도 초과 — 진입 차단')
        if not _state['daily_loss_alerted']:
            _state['daily_loss_alerted'] = True
            _tg(f'🔴 [위험] 일간 손실 한도 초과 ({day_loss_pct*100:.1f}% ≤ '
                f'{SPOT_DAILY_LOSS_LIMIT*100:.0f}%) — 오늘 신규 진입 차단')
        return False

    log.info(f'[{strategy}] {symbol} 매수 시도 | {qty:.6f}개 @ {entry_price:,.4f} | '
             f'비용 ${cost_usdt:.2f} | 리스크 {adj_risk*100:.2f}%')

    if _state['dry_run']:
        log.info(f'[DRY-RUN] {symbol} 매수 시뮬레이션 (실제 주문 없음)')
        fill_price = entry_price
    else:
        try:
            order = _get_ex().create_market_buy_order(ccxt_sym, qty)
            # 지출분을 즉시 반영. 잔고폴러는 60초 주기라, 한 패스에서 연속
            # 매수하면 모두 '매수 전' 잔고를 보고 통과해 USDT 예비금(10%)이
            # 무너진다. 폴러가 다음 주기에 실제 값으로 덮어쓴다.
            with _state_lock:
                _state['usdt_balance'] = max(0.0, _state['usdt_balance'] - cost_usdt)
            fill_price = float(order.get('average') or order.get('price') or entry_price)
            # 실수령 수량 = 체결량 - 기초자산 수수료.
            # gross(체결량)로 매도 주문을 걸면 보유량 초과로 -2010이 나
            # 거래소 SL/TP 등록이 통째로 실패한다(→ 봇 다운 시 무방비).
            net_qty = _net_filled_qty(order, ccxt_sym, qty)
            if 0 < net_qty < qty:
                log.info(f'[{strategy}] {symbol} qty조정: {qty:.6f} -> {net_qty:.6f} '
                         f'(실수령량 = 체결량 - 기초자산 수수료)')
                qty = net_qty
        except Exception as e:
            log.error(f'[{strategy}] {symbol} 매수 주문 실패: {e}')
            _tg(f'⚠️ [{strategy}] {symbol} 매수 실패: {e}')
            return False

    # SL 재계산 (실제 체결가 기준): 원래 sl_dist(신호 기준 거리)를 체결가에 그대로 적용.
    # fill_price와 sl의 거리를 그대로 다시 빼면 항상 원래 sl과 같아져 슬리피지가
    # 무시되는 버그가 있었음 — 거리(sl_dist)는 유지하고 기준점만 체결가로 이동해야 함.
    sl_final = fill_price - sl_dist
    # TP 재계산: RR 기반 전략은 fill_price 기준으로 보정 (S5 bb_upper는 절대가격이라 제외)
    rr = sig.get('rr', 0)
    if strategy != 'S5' and rr > 0:
        tp_final = fill_price + sl_dist * rr
    else:
        tp_final = tp

    # 체결은 끝났으므로 여기서 실패하면 '거래소엔 코인이 있는데 봇은 모르는'
    # 완전 무보호 포지션이 된다(소프트웨어 SL·거래소 SL·reconcile 모두 DB
    # 포지션을 기준으로 동작). 1회 재시도 후에도 실패하면 거래소 스탑이라도
    # 걸고 운영자에게 즉시 알린다 — 조용히 삼키면 안 되는 상황.
    try:
        _save_position(
            strategy, symbol, fill_price, sl_final, tp_final,
            qty, cost_usdt, adj_risk,
            sig.get('exit_type', 'sl_tp'), sig.get('max_hold', 0), regime
        )
    except Exception as e_save:
        log.error(f'[{strategy}] {symbol} 포지션 저장 실패 — 재시도: {e_save}')
        try:
            _save_position(
                strategy, symbol, fill_price, sl_final, tp_final,
                qty, cost_usdt, adj_risk,
                sig.get('exit_type', 'sl_tp'), sig.get('max_hold', 0), regime
            )
        except Exception as e_save2:
            log.critical(f'[{strategy}] {symbol} 포지션 저장 2회 실패 — 무보호 포지션 발생: {e_save2}')
            emergency_sl = ''
            if not _state['dry_run']:
                try:
                    emergency_sl = _place_stop_loss_order(strategy, symbol, ccxt_sym,
                                                          qty, sl_final)
                except Exception:
                    pass
            _tg(f'🚨 [{strategy}] {symbol} **포지션 DB 저장 실패** — 봇이 추적하지 못하는 '
                f'보유분이 생겼습니다. 수동 확인 필요.\n'
                f'   수량 {qty:.6f} @ {fill_price:,.4f} / SL {sl_final:,.4f}\n'
                f'   거래소 스탑: {emergency_sl or "등록 실패 — 보호 없음"}\n'
                f'   오류: {e_save2}')
            return False

    # 거래소 측 보호 주문 등록 (봇 다운 중에도 SL/TP 집행 — 실패 시 소프트웨어 폴백)
    # S5는 TP(BB 상단)가 매 봉 갱신되는 동적 목표라 거래소 TP 제외 (스탑 단독)
    if not _state['dry_run']:
        tp_for_oco = 0.0 if strategy == 'S5' else float(tp_final or 0)
        sl_id, tp_id = _place_protective_orders(strategy, symbol, ccxt_sym,
                                                qty, sl_final, tp_for_oco)
        if sl_id or tp_id:
            _update_position_order_id(strategy, symbol, sl_id, tp_id)

    msg = (f'✅ [{strategy}] {symbol} 매수\n'
           f'   체결가: {fill_price:,.4f} | SL: {sl_final:,.4f}\n'
           f'   수량: {qty:.6f} | 비용: ${cost_usdt:.2f}\n'
           f'   리스크: {adj_risk*100:.2f}% | 레짐: {regime}')
    log.info(msg)
    _tg(msg)
    return True


def _spot_sell(strategy: str, symbol: str, ccxt_sym: str,
               pos: dict, price: float, reason: str):
    """현물 매도 실행."""
    entry_price = float(pos['entry_price'])
    qty         = float(pos['qty_tokens'])
    cost_usdt   = float(pos['cost_usdt'])
    risk_pct    = float(pos['risk_pct'])
    sl          = float(pos['sl'])
    entry_ts    = pos['entry_ts']
    regime      = pos.get('regime', '')

    exit_price = price

    if not _state['dry_run']:
        # 거래소 보호 주문(SL/TP) 먼저 취소 — 미취소 시 해당 수량이 잠겨(free=0)
        # 시장가 매도가 실패하고, 아래 사전잔고 확인이 수동매도로 오판한다.
        _cancel_stop_order(strategy, symbol, ccxt_sym, pos.get('sl_order_id') or '')
        _cancel_stop_order(strategy, symbol, ccxt_sym, pos.get('tp_order_id') or '')
        # 매도 전 실제 잔고 확인 (수수료 차감 등으로 DB qty > 실잔고 가능 → SL 실패 원인)
        try:
            _base_asset = ccxt_sym.split('/')[0]
            _pre_bal = _get_ex().fetch_balance()
            _actual_free = float(_pre_bal['free'].get(_base_asset, 0))
            if _actual_free <= 0.0:
                # free=0의 원인이 "수동매도"가 아니라 **미체결 매도주문이 수량을
                # 잠근 것**일 수 있다(DB에 ID가 없는 고아 OCO/스탑 등). 이를
                # 구분하지 않으면 살아있는 포지션을 허위 MANUAL_SOLD로 지운다.
                _actual_free = _cancel_orphan_sell_orders(strategy, symbol, ccxt_sym,
                                                          _base_asset, _actual_free)
            if _actual_free <= 0.0:
                _hold_h = (datetime.now(timezone.utc) - datetime.fromisoformat(entry_ts)).total_seconds() / 3600
                log.warning(f'[{strategy}] {symbol} 실잔고 0 → 수동매도로 자동처리 ({reason})')
                _tg(f'ℹ️ [{strategy}] {symbol} 잔고 없음 → 수동매도 DB정리')
                _pnl_u = (price - entry_price) * qty
                _pnl_p = (price - entry_price) / entry_price if entry_price > 0 else 0
                _sl_d = abs(entry_price - sl)
                _pnl_r = _pnl_u / (_sl_d * qty) if _sl_d > 0 else 0
                _delete_position(strategy, symbol)
                _net = _log_trade(strategy, symbol, entry_price, price, qty, cost_usdt,
                                  _pnl_u, _pnl_p, _pnl_r, round(_hold_h, 2),
                                  'MANUAL_SOLD', regime, entry_ts)
                with _state_lock:
                    _state['day_pnl'] += _net
                return
            elif _actual_free < qty:
                log.warning(f'[{strategy}] {symbol} qty조정: {qty:.6f} -> {_actual_free:.6f} (수수료 공제 등 잔고 부족)')
                qty = _actual_free
        except Exception as _be:
            log.warning(f'[{strategy}] {symbol} 사전잔고확인 실패(무시): {_be}')
        sold_ok = False
        try:
            order = _get_ex().create_market_sell_order(ccxt_sym, qty)
            exit_price = float(order.get('average') or order.get('price') or price)
            sold_ok = True
        except Exception as e:
            err_str = str(e).lower()
            log.error(f'[{strategy}] {symbol} 매도 실패: {e}')
            _tg(f'⚠️ [{strategy}] {symbol} 매도 실패: {e}')
            # insufficient balance: 실제 잔고 확인 후 0에 가까우면 수동매도로 자동 처리
            if 'insufficient balance' in err_str or 'insufficient funds' in err_str:
                try:
                    _base = ccxt_sym.split('/')[0]
                    _bal = _get_ex().fetch_balance()
                    _actual_free = float(_bal['free'].get(_base, 0))
                    if _actual_free < qty * 0.05:
                        _hold_h = (datetime.now(timezone.utc) - datetime.fromisoformat(entry_ts)).total_seconds() / 3600
                        log.warning(f'[{strategy}] {symbol} 수동매도 감지(잔고={_actual_free:.4f}) → DB자동정리')
                        _tg(f'ℹ️ [{strategy}] {symbol} 수동매도 감지 → DB 정리 완료')
                        _delete_position(strategy, symbol)
                        _pnl_u = (price - entry_price) * qty
                        _pnl_p = (price - entry_price) / entry_price if entry_price > 0 else 0
                        _sl_d = abs(entry_price - sl)
                        _pnl_r = _pnl_u / (_sl_d * qty) if _sl_d > 0 else 0
                        _net = _log_trade(strategy, symbol, entry_price, price, qty, cost_usdt,
                                          _pnl_u, _pnl_p, _pnl_r, round(_hold_h, 2),
                                          'MANUAL_SOLD', regime, entry_ts)
                        with _state_lock:
                            _state['day_pnl'] += _net
                        return
                    elif _actual_free > 0:
                        # 잔고가 DB qty보다 약간 부족(수수료 공제 등) → 실잔고로 재시도
                        # 계좌 전체 free이므로 다른 전략 보유분이 섞여 있을 수 있다.
                        # 이 포지션의 DB 수량을 넘겨 팔면 남의 코인을 처분하고
                        # 부풀린 손익을 확정 기록하게 된다 → 반드시 상한을 둔다.
                        _retry_qty = min(_actual_free, qty)
                        log.warning(f'[{strategy}] {symbol} 잔고재시도: {_retry_qty:.6f} '
                                    f'(가용 {_actual_free:.6f} / DB {qty:.6f})')
                        try:
                            _retry_order = _get_ex().create_market_sell_order(ccxt_sym, _retry_qty)
                            exit_price = float(_retry_order.get('average') or _retry_order.get('price') or price)
                            qty = _retry_qty
                            sold_ok = True   # 실제 체결됨 → 아래 공통 정산으로 내려가야 한다
                            log.info(f'[{strategy}] {symbol} 잔고재시도 성공: {_actual_free:.6f}개 @ {exit_price:.4f}')
                        except Exception as _e3:
                            log.error(f'[{strategy}] {symbol} 잔고재시도도 실패: {_e3}')
                            return
                except Exception as _e2:
                    log.error(f'[{strategy}] {symbol} 수동매도 자동처리 실패: {_e2}')
            # NOTIONAL 미달: 가격 하락으로 포지션 가치 < $5 → Binance 거부
            # _manage_position에서 사전 차단하지만, 혹시 도달하면 DB 유지 후 반환
            # (가격 회복 시 _manage_position이 자동으로 SL/TP 재시도)
            elif 'notional' in err_str or '-1013' in err_str:
                log.warning(f'[{strategy}] {symbol} NOTIONAL 미달(${qty*price:.2f}<$5) — 포지션 유지, 가격 회복 대기')
                return  # DB 유지 (삭제하지 않음)
            # 잔고 재시도로 체결된 경우에는 반환하지 않고 아래 공통 정산을 수행한다.
            # (여기서 무조건 return하면 코인은 팔렸는데 DB 포지션·거래기록이
            #  남지 않아, 다음 폴링에서 허위 MANUAL_SOLD로 잘못 기록됐다.)
            if not sold_ok:
                return

    sl_dist   = abs(entry_price - sl)
    pnl_usdt  = (exit_price - entry_price) * qty
    pnl_pct   = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    pnl_r     = pnl_usdt / (sl_dist * qty) if sl_dist > 0 else 0

    entry_dt   = datetime.fromisoformat(entry_ts)
    hold_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

    _delete_position(strategy, symbol)
    net_pnl = _log_trade(strategy, symbol, entry_price, exit_price, qty, cost_usdt,
                         pnl_usdt, pnl_pct, pnl_r, hold_hours, reason, regime, entry_ts)

    with _state_lock:
        _state['day_pnl'] += net_pnl

    emoji = '✅' if pnl_usdt > 0 else '❌'
    msg = (f'{emoji} [{strategy}] {symbol} 청산 ({reason})\n'
           f'   진입: {entry_price:,.4f} → 청산: {exit_price:,.4f}\n'
           f'   PnL: ${pnl_usdt:+.2f} ({pnl_pct*100:+.2f}%)\n'
           f'   보유: {hold_hours:.1f}시간')
    log.info(msg)
    _tg(msg)


# ══════════════════════════════════════════════════════════════
#  포지션 관리
# ══════════════════════════════════════════════════════════════

def _handle_stop_order_state(strategy: str, symbol: str, ccxt_sym: str,
                             pos: dict) -> bool:
    """거래소 보호 주문(SL/TP 레그) 상태 처리.
    어느 레그든 체결(closed) → 거래 기록 + 포지션 삭제 후 True (관리 종료).
    취소/거부/만료 감지(OCO는 한 레그 취소 시 전체 사멸) → 잔여 레그 정리 후
    보호 주문 재등록 시도. 조회 실패/미체결 → False (평소 관리 계속).
    """
    legs = [(pos.get('sl_order_id') or '', 'SL'),
            (pos.get('tp_order_id') or '', 'TP')]
    fetched = []
    unverified = []          # 조회 실패 — 거래소에 살아있는지 알 수 없는 레그
    for oid, reason in legs:
        if not oid:
            continue
        order = _fetch_stop_order(ccxt_sym, oid)
        if order is None:
            unverified.append(oid)
        if order is not None:
            fetched.append((oid, reason, order))
    if not fetched:
        return False

    # 1) 체결 레그 우선 처리
    for oid, reason, order in fetched:
        if str(order.get('status', '') or '').lower() != 'closed':
            continue
        entry_price = float(pos['entry_price'])
        qty         = float(order.get('filled') or pos['qty_tokens'])
        fallback_px = float(pos['sl']) if reason == 'SL' else float(pos.get('tp') or pos['sl'])
        exit_price  = float(order.get('average') or order.get('price') or fallback_px)
        sl_dist     = abs(entry_price - float(pos['sl']))
        pnl_usdt    = (exit_price - entry_price) * qty
        pnl_pct     = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        pnl_r       = pnl_usdt / (sl_dist * qty) if sl_dist > 0 and qty > 0 else 0
        entry_ts    = pos['entry_ts']
        hold_hours  = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(entry_ts)).total_seconds() / 3600
        _delete_position(strategy, symbol)
        net_pnl = _log_trade(strategy, symbol, entry_price, exit_price, qty,
                             float(pos['cost_usdt']), pnl_usdt, pnl_pct, pnl_r,
                             hold_hours, reason, pos.get('regime', ''), entry_ts)
        with _state_lock:
            _state['day_pnl'] += net_pnl
        # 반대 레그 정리 (OCO는 자동취소되지만 스탑 단독+소프트웨어 병행 대비)
        for other_id, _r in legs:
            if other_id and other_id != oid:
                _cancel_stop_order(strategy, symbol, ccxt_sym, other_id)
        emoji = '✅' if pnl_usdt > 0 else '❌'
        msg = (f'{emoji} [{strategy}] {symbol} 청산 ({reason} — 거래소 주문 체결)\n'
               f'   진입: {entry_price:,.4f} → 청산: {exit_price:,.4f}\n'
               f'   PnL: ${pnl_usdt:+.2f} ({pnl_pct*100:+.2f}%)')
        log.info(msg)
        _tg(msg)
        return True

    # 2) 취소/거부/만료 감지 → 잔여 레그 정리 후 재무장 (실패 시 소프트웨어 폴백)
    dead = [r for oid, r, order in fetched
            if str(order.get('status', '') or '').lower()
            in ('canceled', 'cancelled', 'rejected', 'expired')]
    if dead:
        if unverified:
            # 조회 실패한 레그는 거래소에 살아있을 수 있다. 이 상태에서 재무장하면
            # ① 살아있는 레그가 수량을 잠가 새 주문이 -2010으로 실패하고
            # ② DB ID를 덮어써 그 레그를 영영 추적할 수 없게 된다(고아 주문).
            # 다음 사이클에 다시 시도한다 — 그 사이는 소프트웨어 SL이 커버.
            log.warning(f'[{strategy}] {symbol} 보호주문 {dead} 취소 감지했으나 '
                        f'조회 실패 레그({unverified}) 존재 — 재무장 보류')
            return False
        log.warning(f'[{strategy}] {symbol} 보호주문 {dead} 취소/만료 감지 — 재등록 시도')
        for oid, _r, _order in fetched:
            _cancel_stop_order(strategy, symbol, ccxt_sym, oid)
        tp_for_oco = 0.0 if strategy == 'S5' else float(pos.get('tp') or 0)
        new_sl, new_tp = _place_protective_orders(strategy, symbol, ccxt_sym,
                                                  float(pos['qty_tokens']),
                                                  float(pos['sl']), tp_for_oco)
        _update_position_order_id(strategy, symbol, new_sl, new_tp)
    return False


_rearm_attempts: dict = {}          # (strategy, symbol) → (마지막 시도 시각, 횟수)
_REARM_INTERVAL = 300               # 자가복구 시도 간격(초)
_REARM_ALERT_AFTER = 3              # 이 횟수를 넘기면 운영자에게 알린다


def _rearm_missing_protection(strategy: str, symbol: str, ccxt_sym: str,
                              pos: dict) -> None:
    """보호주문이 없는 포지션에 대해 주기적으로 재등록을 시도한다.

    등록 실패가 곧 '영구 포기'가 되지 않도록 하는 자가복구 경로.
    반복 실패는 소프트웨어 SL만 남았다는 뜻이므로 운영자에게 알린다.
    """
    key = (strategy, symbol)
    now = time.time()
    last, cnt = _rearm_attempts.get(key, (0.0, 0))
    if now - last < _REARM_INTERVAL:
        return
    tp_for_oco = 0.0 if strategy == 'S5' else float(pos.get('tp') or 0)
    sl_id, tp_id = _place_protective_orders(strategy, symbol, ccxt_sym,
                                            float(pos['qty_tokens']),
                                            float(pos['sl']), tp_for_oco)
    if sl_id or tp_id:
        _update_position_order_id(strategy, symbol, sl_id, tp_id)
        _rearm_attempts.pop(key, None)
        log.info(f'[{strategy}] {symbol} 보호주문 자가복구 성공 (SL#{sl_id})')
        return
    cnt += 1
    _rearm_attempts[key] = (now, cnt)
    if cnt == _REARM_ALERT_AFTER:
        _tg(f'⚠️ [{strategy}] {symbol} 거래소 보호주문 등록이 {cnt}회 연속 실패했습니다.\n'
            f'   현재 소프트웨어 SL만 작동 중 — 봇이 멈추면 손절되지 않습니다.\n'
            f'   거래소 잔고/미체결 주문을 확인해 주세요.')


def _manage_position(strategy: str, symbol: str, ccxt_sym: str, df, i: int) -> None:
    """현재 포지션 SL/TP/청산 체크."""
    pos = _load_position(strategy, symbol)
    if pos is None:
        return

    # 거래소 보호 주문(SL/TP) 상태 확인 (체결됐으면 여기서 거래 기록 후 종료)
    if not _state['dry_run'] and (pos.get('sl_order_id') or pos.get('tp_order_id')):
        if _handle_stop_order_state(strategy, symbol, ccxt_sym, pos):
            return
        pos = _load_position(strategy, symbol)   # 재무장 시 sl_order_id 갱신 반영
        if pos is None:
            return
    elif (not _state['dry_run'] and SPOT_EXCHANGE_STOP
          and not (pos.get('sl_order_id') or pos.get('tp_order_id'))):
        # 보호주문 ID가 비어 있는 상태 — 진입 시 등록 실패했거나 재무장이
        # 실패한 뒤다. 예전에는 위 가드가 영원히 False라 **다시는 시도되지
        # 않아** 거래소 SL 없이 방치됐다. 주기적으로 자가복구를 시도한다.
        _rearm_missing_protection(strategy, symbol, ccxt_sym, pos)
        pos = _load_position(strategy, symbol) or pos

    price     = _get_price(ccxt_sym)
    entry     = float(pos['entry_price'])

    if price <= 0:
        # 가격 조회 불가 2분 초과 → 긴급 청산 (SL 미작동 방지)
        cached = _last_known_price.get(ccxt_sym)
        if cached and (time.time() - cached[1]) >= _PRICE_CACHE_TTL:
            log.error(f'[{strategy}/{symbol}] 가격 조회 불가 {_PRICE_CACHE_TTL}s 초과 — 긴급 청산')
            _tg(f'🚨 [{strategy}/{symbol}] 가격 조회 불가 — 긴급 청산')
            _spot_sell(strategy, symbol, ccxt_sym, pos, entry * 0.99, 'EMERGENCY')
        return

    sl        = float(pos['sl'])
    tp        = float(pos['tp'])
    exit_type = pos.get('exit_type', 'sl_tp')
    max_hold  = int(pos.get('max_hold_bars', 0))
    # bars_held: 실제 보유시간 기반으로 계산 후 DB에 라이트백
    _tf = STRATEGY_TIMEFRAMES.get(strategy, '1d')
    _hours_per_bar = 4 if _tf == '4h' else 24
    try:
        _entry_dt = datetime.fromisoformat(pos['entry_ts'])
        _hold_h   = (datetime.now(timezone.utc) - _entry_dt).total_seconds() / 3600
        bars_held = int(_hold_h / _hours_per_bar)
        if bars_held != int(pos.get('bars_held', 0)):
            with _db_lock, _db_conn() as conn:
                conn.execute('UPDATE spot_positions SET bars_held=? WHERE strategy=? AND symbol=?',
                             (bars_held, strategy, symbol))
    except Exception:
        bars_held = int(pos.get('bars_held', 0))
    peak      = float(pos.get('peak_price', entry))

    # S5: BB_upper를 실시간 TP로 업데이트 (DB도 갱신하여 대시보드 정확도 향상)
    if strategy == 'S5' and df is not None and 'bb_upper' in df.columns:
        live_bb_upper = float(df.iloc[i]['bb_upper']) if not pd.isna(df.iloc[i]['bb_upper']) else 0
        if live_bb_upper > 0:
            if abs(live_bb_upper - float(pos.get('tp', 0))) > 1e-8:
                _update_position_tp(strategy, symbol, live_bb_upper)
            tp = live_bb_upper

    # SL 체크
    if price <= sl:
        # NOTIONAL 사전 검사: 포지션 가치 < $5이면 Binance가 주문 거부
        # → 매도 시도하지 않고 DB 유지, 가격 회복 시 자동 매도
        _pos_val = price * float(pos.get('qty_tokens', 0))
        if _pos_val < SPOT_MIN_ORDER_USDT:
            log.info(f'[{strategy}] {symbol} SL 감지 → NOTIONAL 미달(${_pos_val:.2f}) — 가격 회복 대기')
            return  # 포지션 DB 유지, 다음 폴링에서 재검사
        _spot_sell(strategy, symbol, ccxt_sym, pos, price, 'SL')
        return

    # TP 체크
    if tp > 0 and price >= tp:
        _spot_sell(strategy, symbol, ccxt_sym, pos, price, 'TP')
        return

    # 추가 청산 조건 (크로스 등)
    exit_fn = EXIT_CHECK_FUNCS.get(strategy)
    if exit_fn is not None and df is not None:
        try:
            if strategy == 'S6':
                should_exit = exit_fn(df, i, entry)
            else:
                should_exit = exit_fn(df, i)
            if should_exit:
                _spot_sell(strategy, symbol, ccxt_sym, pos, price, 'CROSS')
                return
        except Exception:
            pass

    # 시간 기반 청산
    if max_hold > 0 and bars_held >= max_hold:
        _spot_sell(strategy, symbol, ccxt_sym, pos, price, 'TIME')
        return

    # S4: BB_mid 도달 청산
    if strategy == 'S4' and df is not None and 'bb_mid' in df.columns:
        bb_mid = float(df.iloc[i]['bb_mid']) if not pd.isna(df.iloc[i]['bb_mid']) else 0
        if bb_mid > 0 and price >= bb_mid:
            _spot_sell(strategy, symbol, ccxt_sym, pos, price, 'BB_MID')
            return

    # Peak 업데이트
    if price > peak:
        _update_position_sl(strategy, symbol, sl, price)


# ══════════════════════════════════════════════════════════════
#  전략 루프 (타임프레임별 통합)
# ══════════════════════════════════════════════════════════════

def _strategy_timeframe_loop(timeframe: str, strategies: list[str],
                              stop_event: threading.Event) -> None:
    """
    단일 타임프레임(4H 또는 1D)의 모든 심볼 × 전략을 순환합니다.
    """
    tf_label  = '4H' if timeframe == '4h' else '1D'
    limit     = SPOT_CANDLE_4H if timeframe == '4h' else SPOT_CANDLE_1D
    cooldowns: dict = {}   # (strategy, symbol) → 남은 쿨다운 카운터
    last_bar:  dict = {}   # (strategy, symbol) → 마지막 처리 봉 ts

    log.info(f'[{tf_label}루프] 시작 — 전략: {strategies}')

    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        universe = _state.get('universe', [])
        if not universe:
            time.sleep(10)
            continue

        # 패스 시작 시 전 심볼 시세를 배치 1회로 확보.
        # 보유 포지션 심볼은 유니버스에서 빠졌더라도 SL/TP 판정에 필요하다.
        _pass_syms = [s.replace('USDT', '/USDT') for s in universe]
        try:
            _pass_syms += [p['symbol'].replace('USDT', '/USDT')
                           for p in _load_all_positions()]
        except Exception:
            pass
        _prefetch_prices(_pass_syms)

        for symbol in universe:
            ccxt_sym = symbol.replace('USDT', '/USDT')
            ex = _get_ex()

            try:
                ohlcv = _candle_cache.get(ex, ccxt_sym, timeframe, limit)
            except Exception as e:
                log.warning(f'[{tf_label}루프] {symbol} 캔들 로드 실패: {e}')
                continue

            if not ohlcv or len(ohlcv) < 50:
                continue

            for strategy_id in strategies:
                if strategy_id not in _state.get('active_strategies', []):
                    continue

                key     = (strategy_id, symbol)
                try:
                    calc_fn   = CALC_FUNCS[strategy_id]
                    signal_fn = SIGNAL_FUNCS[strategy_id]
                    df        = calc_fn(ohlcv)
                    i         = len(df) - 1
                    price     = _get_price(ccxt_sym)

                    # 포지션 관리 (매 폴링마다)
                    _manage_position(strategy_id, symbol, ccxt_sym, df, i)

                    # 신규 봉 확인 (새 봉이 닫혔을 때만 신호 체크)
                    cur_ts = int(df.iloc[i]['ts'].timestamp())
                    if last_bar.get(key) == cur_ts:
                        continue
                    last_bar[key] = cur_ts  # 중복 방지 먼저 등록

                    # 재시작 보호: last_bar는 메모리라 재시작 시 초기화됨
                    # DB에서 이번 봉 이후 진입 기록 확인 → 동봉 재진입 방지
                    # 인덱스 사용을 위한 sargable 프리필터.
                    # datetime(entry_ts)로 컬럼을 감싸면 인덱스 범위 탐색이
                    # 불가능해 (strategy,symbol) 그룹 전체를 훑는다. 저장 형식이
                    # 다른 레거시 행이 있어도 잘라내지 않도록 2일 여유를 둔 문자열
                    # 하한을 함께 주고, 정확한 판정은 기존 datetime() 조건이 한다.
                    _cut_iso = datetime.fromtimestamp(
                        cur_ts - 172800, tz=timezone.utc).isoformat()
                    with _db_lock, _db_conn() as _rc:
                        _dupe = _rc.execute(
                            "SELECT 1 FROM spot_positions "
                            "WHERE strategy=? AND symbol=? AND entry_ts >= ? "
                            "AND datetime(entry_ts) >= datetime(?, 'unixepoch') "
                            # UNION ALL: 존재 여부만 보므로 중복 제거가 불필요하다.
                            # UNION은 임시 b-tree를 만들어 LIMIT 1 조기 종료를 막는다.
                            "UNION ALL SELECT 1 FROM spot_trades "
                            "WHERE strategy=? AND symbol=? AND entry_ts >= ? "
                            "AND datetime(entry_ts) >= datetime(?, 'unixepoch') LIMIT 1",
                            (strategy_id, symbol, _cut_iso, cur_ts,
                             strategy_id, symbol, _cut_iso, cur_ts)
                        ).fetchone()
                    if _dupe:
                        log.debug(f'[{strategy_id}] {symbol} 재시작 보호: 이번 봉 이미 처리됨')
                        continue

                    # 쿨다운 체크 (S3 전용)
                    if cooldowns.get(key, 0) > 0:
                        cooldowns[key] -= 1
                        continue

                    # 기존 포지션 있으면 신규 진입 스킵
                    if _load_position(strategy_id, symbol):
                        continue

                    # 레짐 확인
                    regime_state = get_cached_regime()
                    regime = regime_state.regime if regime_state else REGIME_WEAK_TREND
                    if regime == REGIME_CRISIS:
                        continue
                    allowed = REGIME_STRATEGY_MAP.get(regime, [])
                    if strategy_id not in allowed:
                        continue

                    # 신호 체크 (이전봉 기준)
                    sig = signal_fn(df, i - 1)
                    if sig['signal'] != 1:
                        continue

                    # ── S5 전용 안전 필터 (전문가 회의 결론 2026-06-04) ──────
                    if strategy_id == 'S5':
                        # [1] SL 쿨다운: 같은 종목 SL 후 N 바 재진입 금지
                        with _db_lock, _db_conn() as _c:
                            _sl_row = _c.execute(
                                "SELECT exit_ts FROM spot_trades "
                                "WHERE strategy='S5' AND symbol=? AND reason='SL' "
                                "ORDER BY exit_ts DESC LIMIT 1", (symbol,)
                            ).fetchone()
                        if _sl_row:
                            _sl_h = (datetime.now(timezone.utc) -
                                     datetime.fromisoformat(_sl_row[0])).total_seconds() / 3600
                            _limit_h = S5_SL_COOLDOWN_BARS * 24
                            if _sl_h < _limit_h:
                                log.info(f'[S5] {symbol} SL쿨다운: {_sl_h:.0f}h 전 SL ({_limit_h:.0f}h 대기)')
                                continue

                        # [2] BTC 상관 그룹 동시 포지션 한도
                        if symbol in S5_BTC_CORR_SYMBOLS:
                            _syms = tuple(S5_BTC_CORR_SYMBOLS)
                            with _db_lock, _db_conn() as _c:
                                _corr_n = _c.execute(
                                    "SELECT COUNT(*) FROM spot_positions WHERE strategy='S5' "
                                    f"AND symbol IN ({','.join('?'*len(_syms))})", _syms
                                ).fetchone()[0]
                            if _corr_n >= S5_CORR_MAX_POS:
                                log.info(f'[S5] {symbol} BTC상관 한도: {_corr_n}/{S5_CORR_MAX_POS}개 진입중')
                                continue
                    # ────────────────────────────────────────────────────────

                    # RS Gate: S6/S7은 모멘텀 하위 67% 심볼 진입 차단
                    if strategy_id in MOMENTUM_RS_GATE_STRATS:
                        rank_pct = _get_momentum_rank_pct(symbol)
                        if rank_pct > MOMENTUM_TOP_TIER_PCT * 3:
                            log.debug(f'[RS Gate] {symbol} 모멘텀 하위권({rank_pct:.0%}) — {strategy_id} 차단')
                            continue

                    # 펀딩비 필터: 추세추종 전략 롱 과밀 구간 차단
                    funding_scale = 1.0
                    if strategy_id in FUNDING_APPLY_STRATS:
                        funding = _get_spot_funding(symbol)
                        if funding >= FUNDING_LONG_BLOCK:
                            log.info(f'[펀딩 차단] {symbol} funding={funding*100:.3f}%/8h — 롱 과밀')
                            continue
                        if funding <= FUNDING_SHORT_BOOST:
                            funding_scale = 1.20   # 숏 스퀴즈 기대 구간 +20%

                    # 모멘텀 주도주 티어 리스크 부스트
                    rs_scale = 1.0
                    if strategy_id in MOMENTUM_RS_GATE_STRATS:
                        if _get_momentum_rank_pct(symbol) <= MOMENTUM_TOP_TIER_PCT:
                            rs_scale = MOMENTUM_TOP_RISK_MULT

                    # sig에 스케일 반영 (risk는 _spot_buy에서 SPOT_BASE_RISK_PCT 기반이므로 플래그 전달)
                    sig['_funding_scale'] = funding_scale
                    sig['_rs_scale']      = rs_scale

                    # 매수 실행
                    ok = _spot_buy(strategy_id, symbol, ccxt_sym, sig, price, regime)
                    if ok and strategy_id == 'S3':
                        cd = S3_COOLDOWN_WEAK if regime == 'WEAK_TREND' else S3_COOLDOWN
                        cooldowns[key] = cd

                except Exception as e:
                    log.error(f'[{tf_label}루프] {strategy_id}/{symbol} 오류: {e}')

        stop_event.wait(SPOT_PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  백그라운드 루프
# ══════════════════════════════════════════════════════════════

def _balance_poller(stop_event: threading.Event) -> None:
    """60초마다 총자산 갱신."""
    log.info('[잔고폴러] 시작')
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        try:
            total, usdt = _get_spot_equity()
            with _state_lock:
                _state['equity']       = total
                _state['usdt_balance'] = usdt
            # 입출금이면 피크를 재조정 (출금을 낙폭으로 오인하지 않도록)
            _rebase_peak_on_capital_flow(total)
            with _state_lock:
                if total > _state['peak_equity']:
                    _state['peak_equity'] = total
            # 60초마다 보존 — 피크와 당일 PnL 모두 최대 60초 이내 최신 상태로
            # 남으므로, 급작스러운 종료에도 리스크 기준점을 거의 잃지 않는다.
            _persist_risk_state()
            log.debug(f'[잔고] 총자산 ${total:,.2f} (USDT ${usdt:,.2f})')
        except Exception as e:
            log.warning(f'[잔고폴러] 오류: {e}')
        stop_event.wait(60)


def _daily_reset_loop(stop_event: threading.Event) -> None:
    """자정(UTC)마다 일간 통계 리셋."""
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0)
        wait_sec = (next_midnight - now).total_seconds()
        stop_event.wait(min(wait_sec, 3600))

        if stop_event.is_set():
            break
        if datetime.now(timezone.utc).hour != 0:
            continue

        with _state_lock:
            day_pnl = _state['day_pnl']
            equity  = _state['equity']
            _state['day_pnl']      = 0.0
            _state['day_start_eq'] = equity
            _state['daily_loss_alerted'] = False
        _persist_risk_state()   # 리셋 직후 재시작해도 새 기준점이 유지되도록

        # 일간 브리핑
        with _db_lock, _db_conn() as conn:
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
            rows = conn.execute(
                "SELECT COUNT(*) as n, SUM(pnl_usdt) as pnl FROM spot_trades WHERE exit_ts LIKE ?",
                (f'{yesterday}%',)
            ).fetchone()
        n_trades = rows['n'] or 0
        total_pnl = rows['pnl'] or 0.0
        msg = (f'[일간브리핑] {yesterday}\n'
               f'  거래: {n_trades}건  PnL: ${total_pnl:+.2f}\n'
               f'  총자산: ${equity:,.2f}')
        log.info(msg)
        _tg(msg)


def _position_reconcile_loop(stop_event: threading.Event) -> None:
    """10분마다 DB ↔ 거래소 보유 코인 검증 및 자동 수정."""
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        stop_event.wait(600)
        if stop_event.is_set():
            break
        # dry-run은 실제 주문을 내지 않으므로 거래소 잔고가 항상 0이다.
        # 가드가 없으면 모든 가상 포지션이 10분 내 '수동매도'로 삭제되고,
        # 허위 MANUAL_SOLD 기록이 같은 DB에 쌓여 실거래 Kelly/건강도까지 오염된다.
        if _state.get('dry_run'):
            continue
        try:
            all_pos = _load_all_positions()
            if not all_pos:
                continue
            ex  = _get_ex()
            bal = ex.fetch_balance({'type': 'spot'})
            for pos in all_pos:
                sym      = pos['symbol']
                strategy = pos['strategy']
                base     = sym.replace('USDT', '')
                actual   = float(bal.get(base, {}).get('total', 0) or 0)
                db_qty   = float(pos['qty_tokens'])

                if actual < 1e-8:
                    # 완전 소진 — 수동 매도로 간주, DB 포지션 삭제
                    log.warning(f'[검증] {sym} 잔고 없음 — DB 포지션 삭제')
                    _tg(f'⚠️ [{strategy}/{sym}] 잔고 0 감지 — 수동매도로 DB 정리')
                    # 고아 보호 주문 방지: 남아있으면 취소 (이미 체결/취소면 무시됨)
                    _cancel_stop_order(strategy, sym, sym.replace('USDT', '/USDT'),
                                       pos.get('sl_order_id') or '')
                    _cancel_stop_order(strategy, sym, sym.replace('USDT', '/USDT'),
                                       pos.get('tp_order_id') or '')
                    _delete_position(strategy, sym)
                    # 거래 기록 — _spot_sell의 수동매도 경로와 동일하게 통계 보존
                    # (체결가 불명이므로 현재가로 추정, 조회 불가 시 진입가)
                    try:
                        est_price = _get_price(sym.replace('USDT', '/USDT'))
                        entry_price = float(pos['entry_price'])
                        if est_price <= 0:
                            est_price = entry_price
                        sl_d   = abs(entry_price - float(pos['sl']))
                        pnl_u  = (est_price - entry_price) * db_qty
                        pnl_p  = (est_price - entry_price) / entry_price if entry_price > 0 else 0
                        pnl_r  = pnl_u / (sl_d * db_qty) if sl_d > 0 and db_qty > 0 else 0
                        hold_h = (datetime.now(timezone.utc) -
                                  datetime.fromisoformat(pos['entry_ts'])).total_seconds() / 3600
                        net = _log_trade(strategy, sym, entry_price, est_price, db_qty,
                                         float(pos['cost_usdt']), pnl_u, pnl_p, pnl_r,
                                         round(hold_h, 2), 'MANUAL_SOLD',
                                         pos.get('regime', ''), pos['entry_ts'])
                        with _state_lock:
                            _state['day_pnl'] += net
                    except Exception as _le:
                        log.warning(f'[검증] {sym} 수동매도 거래기록 실패(무시): {_le}')
                elif actual < db_qty * 0.90:
                    # 10% 이상 괴리 — DB 수량 실제값으로 업데이트
                    log.warning(f'[검증] {sym} DB {db_qty:.6f} vs 실제 {actual:.6f} '
                                f'({(1-actual/db_qty)*100:.1f}% 괴리) — DB 수정')
                    _tg(f'⚠️ [{strategy}/{sym}] 수량 불일치: DB {db_qty:.6f} → 실제 {actual:.6f}')
                    with _db_lock, _db_conn() as conn:
                        conn.execute('UPDATE spot_positions SET qty_tokens=? WHERE strategy=? AND symbol=?',
                                     (actual, strategy, sym))
        except Exception as e:
            log.warning(f'[검증루프] 오류: {e}')


def _db_backup_loop(stop_event: threading.Event) -> None:
    """6시간마다 DB 스냅샷 백업 (state/backups/), 최근 28개(7일치)만 보관."""
    backup_dir = SPOT_DB_FILE.parent / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        try:
            if SPOT_DB_FILE.exists():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                dest = backup_dir / f'atlas_spot_{stamp}.db'
                with _db_lock:
                    shutil.copy2(SPOT_DB_FILE, dest)
                backups = sorted(backup_dir.glob('atlas_spot_*.db'))
                for old in backups[:-28]:
                    old.unlink(missing_ok=True)
                log.info(f'[DB백업] {dest.name} 저장 완료 (보관 {min(len(backups), 28)}개)')
        except Exception as e:
            log.error(f'[DB백업] 실패: {e}')
            _tg(f'⚠️ DB 백업 실패: {e}')
        stop_event.wait(21600)  # 6시간


def _tg_cmd_loop(stop_event: threading.Event) -> None:
    """Telegram 명령 처리."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    offset = 0
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        try:
            url = f'https://api.telegram.org/bot{TG_TOKEN}/getUpdates'
            resp = requests.get(url, params={'offset': offset, 'timeout': 20}, timeout=25)
            updates = resp.json().get('result', [])
            for upd in updates:
                offset = upd['update_id'] + 1
                msg = upd.get('message', {}).get('text', '').strip().lower()
                chat_id = str(upd.get('message', {}).get('chat', {}).get('id', ''))
                if chat_id != str(TG_CHAT_ID):
                    continue
                _handle_tg_cmd(msg)
        except Exception:
            pass
        stop_event.wait(3)


def _handle_tg_cmd(cmd: str) -> None:
    if '/status' in cmd:
        all_pos = _load_all_positions()
        if not all_pos:
            _tg('[Spot] 열린 포지션 없음')
            return
        lines = ['[Spot] 현재 포지션:']
        for p in all_pos:
            price = _get_price(p['symbol'].replace('USDT', '/USDT'))
            pnl_pct = (price - p['entry_price']) / p['entry_price'] * 100 if price > 0 else 0
            lines.append(f"  {p['strategy']}/{p['symbol']}: {p['entry_price']:.4f} → "
                         f"{price:.4f} ({pnl_pct:+.1f}%)")
        _tg('\n'.join(lines))

    elif '/equity' in cmd:
        _tg(f'[Spot] 총자산: ${_state["equity"]:,.2f} (USDT: ${_state["usdt_balance"]:,.2f})')

    elif '/pause' in cmd:
        with _state_lock:
            _state['paused'] = True
        _tg('[Spot] 신규 진입 일시정지')

    elif '/resume' in cmd:
        with _state_lock:
            _state['paused'] = False
        _tg('[Spot] 신규 진입 재개')

    elif '/stop' in cmd:
        SPOT_KILL_SWITCH.touch()
        _tg('[Spot] 봇 종료 요청 접수')

    elif '/regime' in cmd:
        rs = get_cached_regime()
        if rs:
            _tg(f'[Spot] 레짐: {rs.regime} | ADX: {rs.adx:.1f}')
        else:
            _tg('[Spot] 레짐 정보 없음')


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ATLAS Spot 트레이딩 봇')
    parser.add_argument('--dry-run', action='store_true', help='가상 실행 (주문 없음)')
    parser.add_argument('--strategies', default=','.join(DEFAULT_ACTIVE_STRATEGIES),
                        help='활성 전략 (쉼표 구분)')
    args = parser.parse_args()

    # 킬스위치 기동 가드: systemd Restart=on-failure와 함께 사용 —
    # 킬스위치가 있으면 정상 종료(exit 0)해 재시작 루프를 막는다.
    # 재가동하려면 킬스위치 파일을 삭제할 것 (대시보드 start는 자동 삭제).
    if SPOT_KILL_SWITCH.exists():
        log.warning(f'[메인] 킬스위치 감지({SPOT_KILL_SWITCH}) — 기동 중단. '
                    f'재가동하려면 파일을 삭제하세요.')
        return

    _state['dry_run'] = args.dry_run
    _state['active_strategies'] = [s.strip().upper() for s in args.strategies.split(',')]

    log.info(f'{"=" * 60}')
    log.info('  ATLAS Spot Trading Bot 시작')
    log.info(f'  활성 전략: {_state["active_strategies"]}')
    log.info(f'  DRY-RUN: {args.dry_run}')
    log.info(f'{"=" * 60}')

    # DB 초기화
    init_spot_db()

    # 초기 잔고
    total, usdt = _get_spot_equity()
    with _state_lock:
        _state['equity']       = total
        _state['usdt_balance'] = usdt
    # 피크 자산·일간 손실 기준점은 재시작에도 보존돼야 한다 (아래 함수 주석 참조)
    _restore_risk_state(total)
    log.info(f'[초기] 총자산 ${total:,.2f} (USDT ${usdt:,.2f}) '
             f'| 피크 ${_state["peak_equity"]:,.2f}')

    # 유니버스 초기 로드
    ex = _get_ex()
    try:
        ex.load_markets()
        universe = discover_universe(ex)
        _state['universe'] = filter_tradeable(ex, universe)
        log.info(f'[유니버스] {len(_state["universe"])}개 심볼')
    except Exception as e:
        log.warning(f'[유니버스] 초기 로드 실패: {e} — 재시도 대기')
        _state['universe'] = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

    stop_event = threading.Event()
    threads = []

    def _t(target, name, *a):
        t = threading.Thread(target=target, args=a, name=name, daemon=True)
        t.start()
        threads.append(t)

    # 백그라운드 루프 시작
    _t(_tg_worker,               'TgWorker',        stop_event)   # 텔레그램 비동기 전송
    _t(_balance_poller,          'BalancePoll',     stop_event)
    _t(regime_loop,              'RegimeLoop',       stop_event)
    _t(universe_refresh_loop,    'UniverseRefresh', ex, _state, stop_event, _state_lock)
    _t(_daily_reset_loop,        'DailyReset',      stop_event)
    _t(_position_reconcile_loop, 'Reconcile',       stop_event)
    _t(_db_backup_loop,          'DbBackup',        stop_event)
    _t(_tg_cmd_loop,             'TgCmd',           stop_event)

    # 전략 루프 (타임프레임별 통합)
    strategies_4h = [s for s in _state['active_strategies']
                     if STRATEGY_TIMEFRAMES.get(s) == '4h']
    strategies_1d = [s for s in _state['active_strategies']
                     if STRATEGY_TIMEFRAMES.get(s) == '1d']

    if strategies_4h:
        _t(_strategy_timeframe_loop, 'Loop4H', '4h', strategies_4h, stop_event)
    if strategies_1d:
        _t(_strategy_timeframe_loop, 'Loop1D', '1d', strategies_1d, stop_event)

    _tg(f'✅ ATLAS Spot 봇 시작\n'
        f'  전략: {_state["active_strategies"]}\n'
        f'  총자산: ${total:,.2f}\n'
        f'  DRY-RUN: {args.dry_run}')

    try:
        while not SPOT_KILL_SWITCH.exists():
            time.sleep(5)
    except KeyboardInterrupt:
        log.info('[메인] 키보드 인터럽트')
    finally:
        log.info('[메인] 봇 종료 중...')
        stop_event.set()
        for t in threads:
            t.join(timeout=5)
        # 워커 종료 후 enqueue되므로 종료 알림은 동기 flush로 확실히 전송
        _tg('🔴 ATLAS Spot 봇 종료')
        _tg_flush()
        log.info('[메인] 종료 완료')


if __name__ == '__main__':
    main()

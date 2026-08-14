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
    SPOT_DB_FILE, SPOT_LOG_DIR, SPOT_KILL_SWITCH,
    SPOT_LOG_MAX_BYTES, SPOT_LOG_BACKUPS, SPOT_CAPITAL_FLOW_PCT,
    SPOT_MAX_COST_PER_R, SPOT_ASSUMED_SLIP_PCT, SPOT_DEFAULT_SPREAD_PCT,
    SPOT_EDGE_MIN_TRADES,
    SPOT_TRAIL_REARM_FRAC,
    SPOT_MAX_POSITIONS, SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT,
    SPOT_RESERVE_PCT, SPOT_DAILY_LOSS_LIMIT, SPOT_MIN_ORDER_USDT, SPOT_MAX_SL_PCT,
    SPOT_EXCHANGE_STOP, SPOT_STOP_LIMIT_GAP, SPOT_EXCHANGE_OCO,
    SPOT_KELLY_FRACTION, SPOT_EQUITY_PER_SLOT,
    SPOT_HEALTH_MIN_TRADES, SPOT_HEALTH_PF_SOFT, SPOT_HEALTH_SOFT_SCALE, SPOT_HEALTH_PF_HARD,
    SPOT_LEARN_ENABLED, SPOT_LEARN_HALF_LIFE_DAYS, SPOT_LEARN_MIN_INFO,
    SPOT_LEARN_UNPROVEN_SCALE, SPOT_LEARN_FLOOR, SPOT_LEARN_MAX_SCALE,
    SPOT_LEARN_GAIN, SPOT_LEARN_COST_PER_R, SPOT_LEARN_REFRESH_MIN,
    SPOT_BNB_MIN_USD, SPOT_BNB_ALERT_HOURS, SPOT_BNB_MIN_DAYS,
    SPOT_FEE_RECHECK_SEC,
    SPOT_BNB_PRICE_TTL_SEC, SPOT_BNB_MAX_BUY_PCT,
    SPOT_BNB_AUTO_REFILL, SPOT_BNB_TARGET_MONTHS, SPOT_BNB_MAX_BUY_USD,
    SPOT_BNB_MIN_BUY_USD, SPOT_BNB_REFILL_COOLDOWN_H,
    SPOT_HEALTH_WINDOW_DAYS,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_KELLY_WR_THRESH, SPOT_KELLY_PF_THRESH,
    SPOT_RATCHET_DD_THRESH, SPOT_RATCHET_DD_HARD, SPOT_RATCHET_RECOVER,
    SPOT_CANDLE_4H, SPOT_CANDLE_1D, SPOT_CANDLE_CACHE_TTL, SPOT_PRICE_POLL_SEC,
    STRATEGY_TIMEFRAMES, REGIME_STRATEGY_MAP, DEFAULT_ACTIVE_STRATEGIES,
    LIVE_STRATEGIES,
    WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    BT_SPOT_FEE,
    MOMENTUM_TOP_TIER_PCT, MOMENTUM_TOP_RISK_MULT, MOMENTUM_RS_GATE_STRATS,
    MOMENTUM_RS_GATE_PCT,
    FUNDING_LONG_BLOCK, FUNDING_SHORT_BOOST, FUNDING_APPLY_STRATS,
    S3_COOLDOWN, S3_COOLDOWN_WEAK,
    S5_SL_COOLDOWN_BARS, S5_BTC_CORR_SYMBOLS, S5_CORR_MAX_POS,
)
from atlas_rules import base_of, to_ccxt, trailing_sl   # 라이브·백테스트 공유 규칙 (leaf) — 재수출 겸용
from atlas_spot_universe import discover_universe, filter_tradeable, universe_refresh_loop
from atlas_spot_strategies import CALC_FUNCS, SIGNAL_FUNCS, EXIT_CHECK_FUNCS
from atlas_regime import (
    get_cached_regime, regime_loop,
    REGIME_CRISIS, REGIME_WEAK_TREND,
    REGIME_TRENDING_DOWN,
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
            entry_slip_pct REAL DEFAULT 0,
            orig_sl      REAL DEFAULT 0,
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
            dry_run     INTEGER DEFAULT 0,
            slip_pct    REAL DEFAULT 0
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
        if 'slip_pct' not in tcols:
            conn.execute('ALTER TABLE spot_trades ADD COLUMN slip_pct REAL DEFAULT 0')
            log.info('[DB] spot_trades.slip_pct 컬럼 추가 (마이그레이션)')
        pcols = [r[1] for r in conn.execute('PRAGMA table_info(spot_positions)').fetchall()]
        if 'entry_slip_pct' not in pcols:
            conn.execute('ALTER TABLE spot_positions ADD COLUMN entry_slip_pct REAL DEFAULT 0')
            log.info('[DB] spot_positions.entry_slip_pct 컬럼 추가 (마이그레이션)')
        if 'orig_sl' not in pcols:
            conn.execute('ALTER TABLE spot_positions ADD COLUMN orig_sl REAL DEFAULT 0')
            conn.execute('UPDATE spot_positions SET orig_sl=sl WHERE orig_sl=0')
            log.info('[DB] spot_positions.orig_sl 컬럼 추가 (마이그레이션)')
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
                   exit_type: str, max_hold: int, regime: str,
                   entry_slip_pct: float = 0.0):
    with _db_lock, _db_conn() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO spot_positions
        (strategy, symbol, entry_price, sl, tp, qty_tokens, cost_usdt,
         risk_pct, exit_type, max_hold_bars, bars_held, peak_price, entry_ts, regime,
         entry_slip_pct, orig_sl)
        VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)
        """, (strategy, symbol, entry_price, sl, tp, qty, cost,
              risk_pct, exit_type, max_hold, entry_price,
              datetime.now(timezone.utc).isoformat(), regime, entry_slip_pct, sl))


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
    # 자가복구 재시도 기록도 함께 버린다 — 남겨두면 종료된 포지션의 항목이
    # 계속 쌓이고, 같은 심볼에 재진입했을 때 옛 실패 횟수를 물려받는다.
    _rearm_attempts.pop((strategy, symbol), None)
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
    base = base_of(ccxt_sym).upper()
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
    """실제 free 잔고로 제한한 뒤 거래소 수량 정밀도로 **내림**.

    정밀도만 맞추고 보유량을 보지 않으면, 기록된 수량이 실제 보유량보다
    조금이라도 많을 때 보호주문이 -2010(insufficient balance)으로 **영원히**
    실패한다. `_protection_impossible`은 NOTIONAL 미달만 구조적 불가로 보므로
    이 경우는 '일시적 실패'로 분류돼 5분마다 무한 재시도하고, 재시작마다
    알림이 반복된다.

    기초자산으로 수수료가 빠지면(BNB 미사용) 체결량보다 보유량이 적어 이
    상태가 쉽게 만들어진다 — 실측: ADA 기록 44.9 / 보유 44.8551 (0.1% taker).
    청산 경로(`_spot_sell`)는 이미 같은 클램프를 하고 있었고 여기만 빠져 있었다.

    free 조회 실패나 free=0(다른 주문이 물량을 잠근 상태)일 때는 클램프하지
    않는다 — 후자는 사람이 미체결 주문을 확인해야 하는 상황이라, 조용히 0으로
    깎아 '구조적 불가'로 넘기는 대신 기존의 재시도·알림 경로를 그대로 둔다.
    """
    try:
        ex   = _get_ex()
        free = float((ex.fetch_balance()['free'] or {}).get(
            base_of(ccxt_sym), 0) or 0)
        if 0 < free < qty:
            qty = free
        qty = float(ex.amount_to_precision(ccxt_sym, qty))
    except Exception as e:
        # 보정 실패는 치명적이지 않다 — 주문이 거부되면 소프트웨어 SL이 받는다.
        # 다만 조용히 넘기면 원인 추적이 불가능하므로 흔적은 남긴다.
        # qty는 여기까지 진행된 보정(클램프)을 그대로 유지한다.
        log.debug(f'[{ccxt_sym}] 판매수량 보정 실패 — 요청 수량 사용: {e}')
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


_ppbs_mult: dict = {}                 # ccxt_sym → askMultiplierDown (심볼 필터는 거의 안 바뀐다)
_stop_alert_at: dict = {}             # (strategy, symbol) → 마지막 실패 알림 시각
_STOP_ALERT_INTERVAL = 6 * 3600       # 같은 포지션의 실패 알림 재발송 간격(초)


def _min_sell_price(ccxt_sym: str) -> float:
    """PERCENT_PRICE_BY_SIDE 가 허용하는 **최저 매도 주문가**. 0이면 제한 없음/불명.

    바이낸스는 현재가에서 너무 먼 주문을 거부한다(매도는 askMultiplierDown,
    보통 0.9 = 평균가의 -10%까지). 변동성 큰 소형 코인에서 ATR 기반 손절이
    이 범위를 넘으면 주문이 **영원히** 거부된다.

    가격 의존 조건이라 NOTIONAL 미달 같은 영구 불가와는 다르다 — 가격이
    손절선 쪽으로 내려오면 다시 등록 가능해지므로, 포기하지 않고 매 주기
    싸게 다시 판정할 수 있도록 캐시된 값만 쓴다(추가 API 호출 없음).
    """
    mult = _ppbs_mult.get(ccxt_sym)
    if mult is None:
        mult = 0.0
        try:
            for f in _get_ex().market(ccxt_sym)['info']['filters']:
                if f.get('filterType') == 'PERCENT_PRICE_BY_SIDE':
                    mult = float(f.get('askMultiplierDown') or 0)
                    break
        except Exception as e:
            log.debug(f'[{ccxt_sym}] PERCENT_PRICE_BY_SIDE 조회 실패: {e}')
        _ppbs_mult[ccxt_sym] = mult
    if mult <= 0:
        return 0.0
    px = _get_price(ccxt_sym)
    return mult * px if px > 0 else 0.0


def _stop_alert_due(strategy: str, symbol: str, kind: str = 'stop') -> bool:
    """이 포지션의 **kind 종류** 실패를 지금 알릴 차례인가.

    실패는 재시도 주기(5분)마다 반복되므로 매번 보내면 하루 288건이 된다.
    실제로 ONEUSDT 한 종목이 하루 119건을 보냈고, 그 탓에 정작 중요한
    알림이 묻혔다.

    kind로 종류를 나누는 이유: 하나의 키를 공유하면 보호주문 실패 알림이
    매도 실패 알림을 가린다(또는 그 반대). 성격이 다른 사건은 서로를
    억제하면 안 된다.
    """
    key  = (strategy, symbol, kind)
    now  = time.time()
    if now - _stop_alert_at.get(key, 0.0) < _STOP_ALERT_INTERVAL:
        return False
    _stop_alert_at[key] = now
    return True


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
    # 거부가 확실한 주문은 보내지 않는다 — API·레이트리밋 낭비이고,
    # 실패 경고가 5분마다 쌓여 로그와 텔레그램을 뒤덮는다.
    floor = _min_sell_price(ccxt_sym)
    if floor > 0 and limit_price < floor:
        log.info(f'[{strategy}] {symbol} 스탑주문 생략: 거래소 가격범위 밖 '
                 f'(지정가 {limit_price:.8g} < 허용 하한 {floor:.8g}) — 소프트웨어 SL 작동')
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
        if _stop_alert_due(strategy, symbol):
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
    # 손절 지정가가 거래소 허용 범위(PERCENT_PRICE_BY_SIDE) 밖이면 OCO도
    # 스탑 단독도 모두 거부된다. 두 번 실패하며 경고를 두 줄 남기는 대신
    # 여기서 한 번에 걸러 낸다.
    floor = _min_sell_price(ccxt_sym)
    if floor > 0 and limit_price < floor:
        log.info(f'[{strategy}] {symbol} 보호주문 생략: 거래소 가격범위 밖 '
                 f'(지정가 {limit_price:.8g} < 허용 하한 {floor:.8g}) — 소프트웨어 SL 작동')
        return '', ''
    if SPOT_EXCHANGE_OCO and tp_price and tp_price > 0:
        try:
            ex = _get_ex()

            # 정밀도 포맷 폴백. 거래소 메타데이터를 못 읽으면 수동 포맷으로
            # 물러난다 — 기능이 사라지는 건 아니지만, 반복되면 주문이 거부될
            # 수 있으므로 흔적은 남긴다(디버그 레벨: 주문마다 호출된다).
            def _p(v):
                try:
                    return ex.price_to_precision(ccxt_sym, v)
                except Exception as e:
                    log.debug(f'[{strategy}] {symbol} 가격 정밀도 조회 실패, '
                              f'수동 포맷 사용: {e}')
                    return f'{v:.8f}'.rstrip('0').rstrip('.')

            def _a(v):
                try:
                    return ex.amount_to_precision(ccxt_sym, v)
                except Exception as e:
                    log.debug(f'[{strategy}] {symbol} 수량 정밀도 조회 실패, '
                              f'수동 포맷 사용: {e}')
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
               entry_ts: str, slip_pct: float = 0.0) -> float:
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
         fee_usdt, dry_run, slip_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (strategy, symbol, entry_price, exit_price, qty, cost,
              round(pnl_usdt, 4), round(pnl_pct, 4), round(pnl_r, 4),
              round(hold_hours, 2), reason,
              entry_ts, datetime.now(timezone.utc).isoformat(),
              regime, round(fee, 4), is_dry, round(slip_pct, 6)))
    return pnl_usdt - fee


def _settle_closed_position(strategy: str, symbol: str, *, entry_price: float,
                            exit_price: float, qty: float, cost_usdt: float,
                            reason: str, regime: str, entry_ts: str,
                            sl_for_r: float, slip_pct: float = 0.0,
                            round_hold: bool = False,
                            ) -> tuple[float, float, float, float]:
    """포지션 정산 공통부: PnL 계산 → 포지션 삭제 → 거래 기록 → day_pnl 반영.

    같은 삼중주가 다섯 곳(_spot_sell 3경로·_handle_stop_order_state·검증
    루프)에 복사돼 있었고, 사본마다 미세하게 달랐다. 차이는 전부 파라미터로
    보존한다 — 이 함수 도입은 행동 변화가 아니다:
      · sl_for_r  : R배수 분모의 기준 SL. 정상 경로는 orig_sl(진입 시점
                    위험 — 추적으로 sl이 올라가도 불변이어야 R이 안 부푼다),
                    검증 루프 사본은 역사적으로 pos['sl']을 썼다. 어느 쪽을
                    쓸지는 **호출측이 명시**한다 — 발산을 침묵 속에 통일하면
                    그게 곧 행동 변경이다(별도 커밋에서 다룬다).
      · round_hold: 수동매도 계열 사본들은 보유시간을 round(,2) 했다.
      · slip_pct  : 실체결 경로만 왕복 슬리피지를 기록한다.
    pnl_r 가드는 사본들의 합집합(sl_dist>0 and qty>0)이다 — qty=0인
    퇴화 상태에서 일부 사본은 ZeroDivisionError로 죽었는데, 죽는 것보다
    0으로 기록하고 지나가는 쪽이 나머지 사본들의 기존 동작이다.

    Returns: (net_pnl, pnl_usdt, pnl_pct, hold_h) — 호출측 알림 본문용.
    보유시간을 같이 돌려주는 이유: 알림 문구가 기록과 **같은 값**을 보여야
    하는데, 호출측이 다시 계산하면 시계가 두 번 읽혀 미세하게 어긋난다.
    """
    sl_dist  = abs(entry_price - sl_for_r)
    pnl_usdt = (exit_price - entry_price) * qty
    pnl_pct  = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    pnl_r    = pnl_usdt / (sl_dist * qty) if sl_dist > 0 and qty > 0 else 0
    hold_h   = (datetime.now(timezone.utc) -
                datetime.fromisoformat(entry_ts)).total_seconds() / 3600
    if round_hold:
        hold_h = round(hold_h, 2)
    _delete_position(strategy, symbol)
    net_pnl = _log_trade(strategy, symbol, entry_price, exit_price, qty, cost_usdt,
                         pnl_usdt, pnl_pct, pnl_r, hold_h, reason, regime, entry_ts,
                         slip_pct)
    with _state_lock:
        _state['day_pnl'] += net_pnl
    return net_pnl, pnl_usdt, pnl_pct, hold_h


def _try_settle_via_stop_fill(strategy: str, symbol: str, ccxt_sym: str,
                              pos: dict, log_prefix: str) -> bool:
    """잔고 0의 가장 흔한 원인 — 거래소 보호주문 선체결 — 을 먼저 판정한다.

    체결이 확인되면 _handle_stop_order_state 가 실제 체결가·정확한 사유로
    기록·정리까지 마치고 True. 확인 불가(조회 실패)면 False 로 돌아가
    호출측이 수동매도 경로를 계속 탄다 — 포지션이 DB에 떠돌면 안 된다.
    같은 가드가 세 곳(_spot_sell 2경로·검증 루프)에 복사돼 있었다.
    """
    if not (pos.get('sl_order_id') or pos.get('tp_order_id')):
        return False
    try:
        return _handle_stop_order_state(strategy, symbol, ccxt_sym, pos)
    except Exception as e:
        log.warning(f'{log_prefix} 보호주문 체결 확인 실패 '
                    f'— 수동매도 경로로 진행: {e}')
        return False


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


_bnb_alert = {'at': 0.0}
_bnb_refill = {'at': 0.0}
_bnb_price = {'usd': 0.0, 'at': 0.0}
_bnb_capacity_alert = {'at': 0.0}


def _bnb_price_usd() -> float:
    """BNB 시세(캐시). 잔고폴러가 60초마다 도는데 매번 티커를 치면
    하루 1,440회를 낭비한다. 수수료 잔고 판정에 초 단위 정확도는 필요 없다."""
    now = time.time()
    if _bnb_price['usd'] > 0 and now - _bnb_price['at'] < SPOT_BNB_PRICE_TTL_SEC:
        return _bnb_price['usd']
    px = float(_get_ex().fetch_ticker('BNB/USDT')['last'] or 0)
    if px > 0:
        _bnb_price.update({'usd': px, 'at': now})
    return px


def _bnb_alert_threshold(monthly_usd: float) -> float:
    """경고 임계값(USD) — **소모일수 기준**.

    고정 달러로 잡으면 자본이 커질수록 무의미해진다. 자본 $10,000에서
    $5는 하루치라, 경고를 받아도 반응할 시간이 없다. 자본이 10배면
    명목가도 10배고 수수료도 10배이므로 기준도 함께 커져야 한다.

    소액 계좌에서는 소모량이 작아 임계값이 몇 센트가 되므로,
    `SPOT_BNB_MIN_USD`를 하한으로 둔다.
    """
    daily = max(monthly_usd, 0.0) / 30.0
    return max(SPOT_BNB_MIN_USD, daily * SPOT_BNB_MIN_DAYS)


def _estimate_monthly_bnb_usd() -> float:
    """최근 30일 거래에서 월 수수료 소모액(USD)을 실측 추정한다.

        소모액 = Σ(명목가) × 수수료율 × 2      (진입 + 청산)

    이력이 없으면 0.0을 돌려준다 — 호출부는 그때 최소 금액만 산다.
    추정을 부풀리면 매매에 쓸 USDT를 BNB로 묶어 두게 되므로, 모르면
    적게 사고 다음에 또 사는 편이 낫다.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                'SELECT COALESCE(SUM(cost_usdt), 0) FROM spot_trades '
                'WHERE COALESCE(dry_run,0)=0 AND exit_ts >= ?', (since,)
            ).fetchone()
        notional = float(row[0] or 0)
    except Exception as e:
        log.warning(f'[BNB] 소모량 추정 실패: {e}')
        return 0.0
    return notional * _detect_fee_rate() * 2


def _bnb_refill_amount(current_usd: float, usdt_free: float,
                       equity: float) -> tuple[float, str]:
    """지금 얼마를 살 것인가. Returns: (매수액 USD, 사유).

    0.0이면 사지 않는다. **모든 상한을 동시에 만족해야만** 산다 —
    봇이 스스로 돈을 쓰는 유일한 경로라 한 겹이라도 뚫리면 안 된다.

    부수효과가 없는 순수 판정이므로 각 상한을 따로 시험할 수 있다.
    """
    if not SPOT_BNB_AUTO_REFILL:
        return 0.0, '자동 충전 꺼짐'
    if _state.get('dry_run'):
        return 0.0, '드라이런 — 실주문 금지'
    if current_usd >= SPOT_BNB_MIN_USD:
        return 0.0, '잔고 충분'

    now = time.time()
    left_h = (SPOT_BNB_REFILL_COOLDOWN_H * 3600 - (now - _bnb_refill['at'])) / 3600
    if left_h > 0:
        return 0.0, f'쿨다운 {left_h:.0f}h 남음'

    # 필요량: 목표 개월치에서 현재 잔고를 뺀 만큼
    monthly = _estimate_monthly_bnb_usd()
    if monthly <= 0:
        # 이력이 없으면 최소 금액만. 과대 추정으로 USDT를 묶는 것보다 낫다.
        need = SPOT_BNB_MIN_BUY_USD
    else:
        need = monthly * SPOT_BNB_TARGET_MONTHS - current_usd
    # ① 1회 매수 상한 — 소모량에 맞춰 커지되 자산 비율로 최종 제한.
    #    고정 $30이면 대형 계좌에서 보충이 소모를 못 따라간다(아래 용량 점검).
    cap = max(SPOT_BNB_MAX_BUY_USD, monthly * SPOT_BNB_TARGET_MONTHS)
    cap = min(cap, equity * SPOT_BNB_MAX_BUY_PCT) if equity > 0 else SPOT_BNB_MAX_BUY_USD
    cap = max(cap, SPOT_BNB_MAX_BUY_USD)   # 소액 계좌에서 하한 아래로 내려가지 않게
    need = min(need, cap)
    if need < SPOT_BNB_MIN_BUY_USD:
        return 0.0, f'필요액 ${need:.2f} < 최소 ${SPOT_BNB_MIN_BUY_USD:.0f}'

    # ③ USDT 예비금 침범 금지 — 매매 자금을 잠식하면 본말전도다
    reserve = equity * SPOT_RESERVE_PCT
    spendable = usdt_free - reserve
    if spendable < need:
        return 0.0, (f'예비금 보호: 가용 ${max(spendable, 0.0):.2f} < '
                     f'필요 ${need:.2f}')
    return float(need), f'월 소모 추정 ${monthly:.2f} × {SPOT_BNB_TARGET_MONTHS:.0f}개월'


def _bnb_refill_capacity(monthly_usd: float, equity: float) -> tuple:
    """(일일 보충 한도, 일일 소모, 따라갈 수 있는가).

    자동 충전에는 **속도 상한**이 있다 — 1회 상한 ÷ 쿨다운. 자본이 커지면
    소모 속도가 이걸 넘어서고, 그러면 아무리 충전해도 잔고가 계속 줄어
    결국 바닥난다. 그 지점을 조용히 지나치면 "자동 충전을 켜 뒀는데도"
    보호주문이 깨진다 — 켜 뒀다는 사실이 오히려 방심을 만든다.

    한계에 도달하면 쿨다운을 줄이거나 상한 비율을 올려야 한다.
    """
    daily_use = max(monthly_usd, 0.0) / 30.0
    cap = max(SPOT_BNB_MAX_BUY_USD, monthly_usd * SPOT_BNB_TARGET_MONTHS)
    if equity > 0:
        cap = max(min(cap, equity * SPOT_BNB_MAX_BUY_PCT), SPOT_BNB_MAX_BUY_USD)
    cooldown_days = max(SPOT_BNB_REFILL_COOLDOWN_H, 1) / 24.0
    daily_cap = cap / cooldown_days
    return daily_cap, daily_use, daily_cap >= daily_use


def _warn_if_refill_cannot_keep_up(monthly_usd: float, equity: float) -> None:
    """보충 속도가 소모 속도를 못 따라가면 알린다(24시간에 한 번)."""
    if not SPOT_BNB_AUTO_REFILL or monthly_usd <= 0:
        return
    daily_cap, daily_use, ok = _bnb_refill_capacity(monthly_usd, equity)
    if ok:
        return
    now = time.time()
    if now - _bnb_capacity_alert['at'] < SPOT_BNB_ALERT_HOURS * 3600:
        return
    _bnb_capacity_alert['at'] = now
    log.warning(f'[BNB] 자동 충전 용량 부족: 보충 ${daily_cap:.2f}/일 < '
                f'소모 ${daily_use:.2f}/일')
    _tg(f'⚠️ BNB 자동 충전이 소모를 못 따라갑니다\n'
        f'   보충 한도 ${daily_cap:.2f}/일  <  소모 ${daily_use:.2f}/일\n'
        f'   이대로면 잔고가 계속 줄어 결국 손절 주문이 거부됩니다.\n'
        f'   SPOT_BNB_REFILL_COOLDOWN_H를 줄이거나 '
        f'SPOT_BNB_MAX_BUY_PCT를 올리세요.')


def _refill_bnb(current_usd: float) -> None:
    """BNB를 시장가로 매수한다. 실패해도 봇은 계속 돈다(경고 경로 유지)."""
    with _state_lock:
        usdt_free = float(_state.get('usdt_balance', 0) or 0)
        equity    = float(_state.get('equity', 0) or 0)
    amount, why = _bnb_refill_amount(current_usd, usdt_free, equity)
    if amount <= 0:
        log.debug(f'[BNB] 자동 충전 생략: {why}')
        return
    try:
        order = _get_ex().create_order(
            'BNB/USDT', 'market', 'buy', None, None,
            {'quoteOrderQty': round(amount, 2)})
        # 쿨다운은 **성공했을 때만** 갱신한다. 실패로 갱신하면 다음
        # 기회까지 72시간을 그냥 버린다.
        _bnb_refill['at'] = time.time()
        filled = float(order.get('filled') or 0)
        log.info(f'[BNB] 자동 충전 ${amount:.2f} 완료 ({filled:.6f} BNB) — {why}')
        _tg(f'🔄 BNB 수수료 잔고 자동 충전\n'
            f'   매수 ${amount:.2f} ({filled:.6f} BNB)\n'
            f'   근거: {why}\n'
            f'   충전 전 잔고 ≈${current_usd:.2f}')
    except Exception as e:
        log.error(f'[BNB] 자동 충전 실패(경고 경로는 유지): {e}')


def _check_bnb_fee_balance(bal: dict) -> None:
    """BNB 수수료 잔고가 바닥나기 전에 알린다.

    바이낸스의 "BNB로 수수료 지불"은 **잔고가 있을 때만** 동작한다.
    비면 토글이 켜져 있어도 조용히 기초자산 차감으로 되돌아가고, 그 순간
    두 가지가 동시에 일어난다:

      ① 25% 할인 상실 — 왕복 0.15% → 0.2%. 1R 잠식이 7% → 8%로 늘어난다.
      ② **보호주문이 깨진다** — 매수 체결량에서 수수료만큼 기초자산이
         빠져나가 '기록 수량 > 실제 보유량'이 된다. 손절 주문이 -2010
         (insufficient balance)으로 거부되고, 데이터를 고치기 전까지
         해소되지 않는다. 실제로 ADA·RIF·FET에서 발생했다(3fa6074).

    ②가 훨씬 비싸다 — 할인 몇 %가 아니라 **손절이 안 걸린 포지션**이
    생기기 때문이다. 그래서 잔고가 0이 되기 전에 미리 경고한다.
    """
    try:
        qty = float((bal.get('BNB') or {}).get('total', 0) or 0)
        usd = qty * _bnb_price_usd() if qty > 0 else 0.0

        # 임계값은 **소모 속도**를 따라간다. 자본이 커지면 같은 $5도
        # 하루치가 되므로, 고정 달러로는 반응할 시간을 벌 수 없다.
        monthly = _estimate_monthly_bnb_usd()
        with _state_lock:
            equity = float(_state.get('equity', 0) or 0)
        _warn_if_refill_cannot_keep_up(monthly, equity)

        threshold = _bnb_alert_threshold(monthly)
        if usd >= threshold:
            return
        days_left = usd / (monthly / 30.0) if monthly > 0 else float('inf')

        # 자동 충전이 켜져 있으면 먼저 시도한다. 성공하면 다음 폴링에서
        # 잔고가 회복돼 경고가 자연히 멎는다. 꺼져 있거나 실패하면
        # 아래 경고 경로가 그대로 동작한다.
        _refill_bnb(usd)
        now = time.time()
        if now - _bnb_alert['at'] < SPOT_BNB_ALERT_HOURS * 3600:
            return
        _bnb_alert['at'] = now
        _days = '무기한' if days_left == float('inf') else f'{days_left:.1f}일'
        log.warning(f'[BNB] 수수료 잔고 부족: {qty:.6f} BNB (≈${usd:.2f}, '
                    f'잔여 {_days}, 임계 ${threshold:.2f})')
        _tg(f'⚠️ BNB 수수료 잔고 부족 — 약 ${usd:.2f} (잔여 {_days})\n'
            f'   비면 수수료가 **기초자산에서** 빠져 손절 주문이 거부됩니다.\n'
            f'   (실제로 ADA·RIF·FET에서 발생했습니다)\n'
            f'   월 소모 추정 ${monthly:.2f} → '
            f'{SPOT_BNB_TARGET_MONTHS:.0f}개월치 ≈ '
            f'${max(monthly * SPOT_BNB_TARGET_MONTHS, SPOT_BNB_MIN_BUY_USD):.0f} '
            f'충전 권장')
    except Exception as e:
        log.warning(f'[BNB] 잔고 확인 실패(무시): {e}')


_valuation_warned: dict = {}          # currency → 마지막 경고 시각
_VALUATION_WARN_INTERVAL = 6 * 3600   # 통화별 경고 재발송 간격(초)


def _valuation_warn_due(currency: str) -> bool:
    """이 통화의 평가 실패를 지금 경고할 차례인가.

    사실 자체는 계속 유효하므로 완전히 숨기지 않는다. 다만 60초 폴링마다
    같은 줄을 남기면 로그가 그 한 줄로 뒤덮여 정작 중요한 경고가 묻힌다.
    """
    now  = time.time()
    last = _valuation_warned.get(currency, 0.0)
    if now - last < _VALUATION_WARN_INTERVAL:
        return False
    _valuation_warned[currency] = now
    return True


_thread_alerted: set = set()

# 스레드 이름 → 그 스레드가 멈추면 무엇이 멈추는가.
# 알림에 "무슨 일이 벌어지는지"를 같이 적어야 운영자가 우선순위를 정할 수 있다.
_THREAD_ROLE = {
    'Loop4H':     'SL/TP 판정·청산·보호주문 자가복구(4H 전략)',
    'Loop1D':     'SL/TP 판정·청산·보호주문 자가복구(1D 전략)',
    'BalancePoll': '총자산 갱신 — 사이징·낙폭 래칫이 옛 값으로 굳는다',
    'RegimeLoop':  '레짐 판정 — 전략 라우팅이 옛 레짐에 고정된다',
    'Reconcile':   '거래소·DB 대조(수동매도 감지)',
    'DbBackup':    'DB 스냅샷 백업',
    'DailyReset':  '일일 손익 기준 초기화',
    'TgWorker':    '텔레그램 전송 — 이후 모든 알림이 사라진다',
    'TgCmd':       '텔레그램 명령 수신',
    'UniverseRefresh': '유니버스 갱신',
}


def find_dead_threads(threads: list) -> list:
    """죽은 백그라운드 스레드 이름 목록. 같은 스레드는 **1회만** 보고한다.

    감시견(_bot_alive)은 pgrep 기반이라 **프로세스 생존만** 본다. 봇은 10개
    데몬 스레드로 도는데 그중 하나가 예외로 죽어도 프로세스는 살아 있으므로
    감시견은 계속 녹색을 보고한다. 하필 Loop1D/Loop4H가 죽으면 SL/TP 판정과
    청산이 통째로 멈추는데 겉으로는 아무 이상이 없어 보인다 —
    무인 운영에서 가장 위험한 실패 형태다.
    """
    dead = []
    for t in threads:
        try:
            if t.is_alive() or t.name in _thread_alerted:
                continue
        except Exception:      # noqa: BLE001 — 감시가 봇을 멈추면 안 된다
            continue
        _thread_alerted.add(t.name)
        dead.append(t.name)
    return dead


def _report_dead_threads(threads: list) -> None:
    """죽은 스레드를 로그·텔레그램으로 알린다(스레드당 1회)."""
    for name in find_dead_threads(threads):
        role = _THREAD_ROLE.get(name, '해당 기능')
        log.critical(f'[감시] 백그라운드 스레드 사망: {name} — {role} 중단')
        _tg(f'🚨 [감시] 스레드 "{name}" 중단\n'
            f'   멈춘 기능: {role}\n'
            f'   프로세스는 살아 있어 외부 감시로는 정상으로 보입니다.\n'
            f'   복구: systemctl restart atlas-spot')


def _get_spot_equity() -> tuple[float, float]:
    """총자산 = USDT + 보유 코인 현재가. Returns: (total_equity, usdt_balance)."""
    try:
        ex = _get_ex()
        bal = ex.fetch_balance({'type': 'spot'})
        _check_bnb_fee_balance(bal)
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
            except Exception as e:
                # 이 자산이 총자산에서 **누락**된다. 총자산은 사이징·드로다운
                # 래칫·일일 손실한도의 기준이므로, 과소평가되면 그만큼
                # 작게 베팅하고 래칫이 잘못 발동한다. 조용히 넘기면 안 된다.
                #
                # 다만 원인이 '현물 마켓이 없는 자산'(Simple Earn 잔고 LD*,
                # 상장폐지 토큰, 스테이킹 등)이면 **영원히** 실패한다.
                # 잔고 폴러가 60초마다 도므로 억제하지 않으면 같은 줄이
                # 하루 1,440회 쌓인다(실측: LDUSDT 한 건으로 7일간 1,470회).
                # 반복되는 경고는 곧 무시되는 경고다 — 통화별로 간격을 둔다.
                if _valuation_warn_due(currency):
                    log.warning(f'[자산평가] {currency} 시세 조회 실패 — '
                                f'총자산에서 제외됨(과소평가): {e}')
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


_learn_cache: dict = {'at': 0.0, 'result': {}, 'cfg': None}


def _learn_config():
    """config 상수 → LearnConfig. 매 호출 검증되므로 오설정은 여기서 걸린다."""
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


def _get_learn_result(force: bool = False) -> dict:
    """학습 결과(캐시). 진입마다 DB 전체를 훑으면 전략 루프가 느려진다.

    실패 시 **빈 dict**를 돌려준다 — 호출자는 그걸 '미검증'으로 해석해
    중립 배분을 쓴다. 학습기가 죽었다고 봇이 멈추면 안 된다.
    """
    now = time.time()
    if (not force and _learn_cache['result']
            and now - _learn_cache['at'] < SPOT_LEARN_REFRESH_MIN * 60):
        return _learn_cache['result']
    try:
        import atlas_learning as _L
        obs = _L.load_observations(SPOT_DB_FILE)
        res = _L.learn(obs, cfg=_learn_config())
    except Exception as e:
        log.warning(f'[학습] 계산 실패(중립 배분으로 진행): {e}')
        res = {}
    _learn_cache['at'] = now
    _learn_cache['result'] = res
    return res


def _get_learn_scale(strategy: str, regime: str) -> float:
    """(전략 × 레짐) 배분 배수. 학습기가 꺼져 있으면 1.0(무개입)."""
    if not SPOT_LEARN_ENABLED:
        return 1.0
    try:
        import atlas_learning as _L
        return _L.scale_for(_get_learn_result(), strategy, regime, _learn_config())
    except Exception as e:
        log.warning(f'[학습] 배분 조회 실패(중립): {e}')
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


_fee_rate: dict = {'taker': BT_SPOT_FEE, 'checked': False, 'at': 0.0}


BNB_FEE_DISCOUNT = 0.75      # BNB로 수수료 결제 시 25% 할인
_bnb_off_alert = {'at': 0.0}


def _bnb_discount_active() -> bool:
    """최근 체결의 수수료가 **BNB로** 결제됐는가.

    바이낸스 API(`tradeFee`, `commissionRates`)는 VIP 등급 기본 요율만
    돌려주고 **BNB 할인을 반영하지 않는다.** 할인은 체결 시점에 적용되며,
    그 흔적은 개별 체결의 `commissionAsset`에만 남는다.

    그래서 요율 응답만 보고 '할인이 꺼져 있다'고 판단하면 **항상 오탐**이다
    — 실제로 운영자가 토글을 켜 둔 상태에서 경고가 반복 발송됐다.

    최근 거래한 심볼의 체결 내역을 보고 판정한다. 거래 이력이 없으면
    판단 근거가 없으므로 False(모름)를 돌려주되, 호출부의 경고에는
    자체 발송 간격이 걸려 있어 소음이 되지 않는다.
    """
    try:
        with _db_lock, _db_conn() as conn:
            row = conn.execute(
                'SELECT symbol FROM spot_trades WHERE COALESCE(dry_run,0)=0 '
                'ORDER BY id DESC LIMIT 1').fetchone()
        if not row or not row[0]:
            return False
        ccxt_sym = to_ccxt(str(row[0]))
        fills = _get_ex().fetch_my_trades(ccxt_sym, limit=10) or []
    except Exception as e:
        log.debug(f'[수수료] BNB 결제 여부 확인 실패: {e}')
        return False
    for f in reversed(fills):
        asset = ((f.get('fee') or {}).get('currency')
                 or (f.get('info') or {}).get('commissionAsset') or '')
        if asset:
            return str(asset).upper() == 'BNB'
    return False


def _warn_bnb_discount_off() -> None:
    """BNB 할인이 실제로 꺼져 있을 때만 알린다(발송 간격 제한).

    예전에는 이 경고에 간격 제한이 없어서, 수수료율 재확인 주기(6시간)마다
    반복 발송됐다. 반복되는 알림은 곧 무시되는 알림이다.
    """
    now = time.time()
    if now - _bnb_off_alert['at'] < SPOT_BNB_ALERT_HOURS * 3600:
        return
    _bnb_off_alert['at'] = now
    _tg('💡 수수료 절감 여지: 최근 체결에서 수수료가 BNB로 결제되지 않았습니다.\n'
        '   "Use BNB to pay fees"가 켜져 있어도 **BNB 잔고가 비면** '
        '기초자산에서 차감됩니다.\n'
        '   왕복 수수료 25% 절감 + 보호주문 거부 방지 — BNB 잔고를 확인하세요.')


def _detect_fee_rate() -> float:
    """계정의 실제 taker 수수료율을 조회해 비용 계산에 반영.

    BNB 수수료 결제를 켜면 25% 할인(0.1% → 0.075%)이다. 왕복 0.2%가
    0.15%로 줄면 SL 5% 기준 1R의 4%→3%로 잠식이 줄어든다 — 코드 변경
    없이 얻는 유일한 확정 이득이라 켜져 있는지 확인하고 알린다.

    ⚠️ **주기적으로 다시 확인한다.** 예전에는 프로세스당 1회만 조회하고
    영원히 캐시했는데, 수수료율은 실제로 바뀐다:
      · BNB 잔고가 비면 할인이 사라진다(→ 0.075%가 다시 0.1%)
      · BNB를 채우면 할인이 살아난다
      · VIP 등급이 바뀐다
    캐시가 굳으면 봇의 비용 모델이 현실과 조용히 벌어지고, 그 값으로
    진입 가드(`_cost_edge_ok`)가 판정한다 — 실제보다 비싸다고 믿으면
    멀쩡한 신호를 막고, 싸다고 믿으면 마이너스 기대값 거래를 통과시킨다.
    """
    now = time.time()
    # .get: 캐시 dict가 옛 형태(타임스탬프 없음)여도 죽지 않는다 —
    # 여기서 예외가 나면 진입 가드 전체가 멈춘다.
    if _fee_rate['checked'] and now - _fee_rate.get('at', 0.0) < SPOT_FEE_RECHECK_SEC:
        return _fee_rate['taker']
    _fee_rate['checked'] = True
    _fee_rate['at'] = now
    try:
        ex = _get_ex()
        taker = None
        try:
            f = ex.fetch_trading_fee('BTC/USDT')
            taker = float(f.get('taker') or 0) or None
        except Exception:
            pass
        if taker is None:
            # 폴백: 계정 정보의 commissionRates
            info = ex.fetch_balance().get('info', {}) or {}
            rates = info.get('commissionRates') or {}
            if rates.get('taker') is not None:
                taker = float(rates['taker'])
        if taker and 0 < taker < 0.01:
            _fee_rate['taker'] = taker
            saved = (BT_SPOT_FEE - taker) / BT_SPOT_FEE * 100
            log.info(f'[수수료] 실제 taker {taker*100:.4f}% '
                     f'(기준 {BT_SPOT_FEE*100:.3f}% 대비 {saved:+.0f}%)')
            # ⚠️ 위 API 응답은 **VIP 등급 기본 요율**이다. BNB 할인은 체결
            # 시점에 적용되므로 여기에 **절대 반영되지 않는다** — 토글을 켜도
            # 영원히 0.1%로 답한다. 그래서 이 값만 보고 "꺼져 있다"고 알리면
            # 항상 오탐이다(실제로 운영자가 켜 둔 상태에서 반복 발송됐다).
            # 판정은 실제 체결의 수수료 자산으로 한다.
            if _bnb_discount_active():
                _fee_rate['taker'] = taker * BNB_FEE_DISCOUNT
                log.info(f'[수수료] BNB 할인 확인 — 실효 taker '
                         f'{_fee_rate["taker"]*100:.4f}%')
            elif taker >= BT_SPOT_FEE * 0.99:
                _warn_bnb_discount_off()
    except Exception as e:
        log.info(f'[수수료] 실제 요율 조회 실패 — 기본값 {BT_SPOT_FEE*100:.3f}% 사용: {e}')
    return _fee_rate['taker']


# trailing_sl 은 atlas_rules 로 이사했다(라이브·백테스트 공유 규칙의 leaf
# 거처 — 백테스트가 이 모듈을 import하며 로그 핸들러·mkdir 부수효과까지
# 물려받던 결합을 끊기 위함). 여기 재바인딩은 함수 객체 동일성을 유지한다:
# bt.trailing_sl is sm.trailing_sl (tests/test_trailing_stop.py 가 고정).
# ⚠️ SPOT_TRAIL_ENABLED 등 추적 상수의 호출 시점 조회처도 atlas_rules 로
#    옮겨갔다 — 오버라이드/패치는 atlas_rules 모듈에 해야 효과가 있다
#    (reoptimize._param_targets 튜플에 반영됨).

_NO_TRADE_ALERT_HOURS = 6      # 이 시간 넘게 전 전략이 막히면 알린다


def check_no_trade_regime(regime: str, now: float) -> None:
    """전 전략이 차단되는 레짐이 길어지면 운영자에게 알린다.

    CRISIS·MICRO_RANGING·UNKNOWN은 REGIME_STRATEGY_MAP이 빈 목록이라 그
    구간에는 어떤 신호도 진입으로 이어지지 않는다. 로그에는 `regime_block`
    카운터만 조용히 쌓여서, 봇이 멀쩡히 도는데 며칠째 거래가 없어도
    운영자는 "장이 없나 보다"라고 넘기게 된다. 지속 시간을 재서 알린다.
    """
    blocked = not REGIME_STRATEGY_MAP.get(regime, [])
    with _state_lock:
        since = _state.get('no_trade_since')
        alerted = _state.get('no_trade_alerted', False)
        if not blocked:
            _state['no_trade_since'] = None
            _state['no_trade_alerted'] = False
        elif since is None:
            _state['no_trade_since'] = now
            return
    if not blocked:
        if since is not None and alerted:
            _tg(f'✅ 레짐이 {regime}(으)로 회복 — 거래 재개')
        return
    hours = (now - since) / 3600
    if hours >= _NO_TRADE_ALERT_HOURS and not alerted:
        with _state_lock:
            _state['no_trade_alerted'] = True
        _tg(f'⏸️ {hours:.0f}시간째 거래 정지 상태입니다.\n'
            f'   레짐: {regime} — 이 레짐에 배정된 전략이 없습니다.\n'
            f'   봇은 정상 동작 중이며, 레짐이 바뀌면 자동으로 재개합니다.\n'
            f'   (의도한 정지가 아니라면 REGIME_STRATEGY_MAP을 확인하세요)')


def validate_active_strategies(strategies: list[str]) -> tuple[list, list]:
    """활성 전략 목록을 검증한다. 반환: (실행 가능한 목록, 문제 설명 목록).

    `--strategies`는 아무 문자열이나 받는데 검증이 없어서, 오타 하나로
    봇이 **아무것도 거래하지 않으면서 "정상 기동"을 보고**하는 상태가 됐다.
    죽는 방식이 세 가지라 각각 구분해서 알린다:
      · STRATEGY_TIMEFRAMES에 없음  → 어느 루프에도 배정되지 않아 완전 침묵
      · 전략 함수 없음               → 매 심볼 예외 후 조용히 무시
      · REGIME_STRATEGY_MAP에 없음   → 모든 봉에서 레짐 차단, 신호 0
    """
    ok, problems = [], []
    for s in strategies:
        if s not in STRATEGY_TIMEFRAMES:
            problems.append(f'{s}: 타임프레임 미정의 — 어느 루프에도 배정되지 '
                            f'않아 완전히 동작하지 않음 (대소문자 확인)')
        elif s not in CALC_FUNCS or s not in SIGNAL_FUNCS:
            problems.append(f'{s}: 전략 함수 없음 — 신호 계산 불가')
        elif not any(s in ss for ss in REGIME_STRATEGY_MAP.values()):
            problems.append(f'{s}: 어느 레짐에도 배정되지 않음 — 모든 봉에서 '
                            f'차단되어 진입이 0건')
        else:
            ok.append(s)
    return ok, problems


def _typical_sl_pct() -> float:
    """실제 거래에서 관측된 SL 거리 중앙값. 표본이 없으면 보수적 기본값.

    사이징 진단은 '전형적인 신호'를 가정해야 의미가 있으므로 상수보다
    실측값을 쓴다.
    """
    try:
        with _db_lock, _db_conn() as conn:
            rows = conn.execute(
                'SELECT entry_price, pnl_r, pnl_usdt, qty_tokens FROM spot_trades '
                'WHERE COALESCE(dry_run,0)=0 AND pnl_r != 0 AND entry_price > 0 '
                'AND qty_tokens > 0 ORDER BY id DESC LIMIT 200'
            ).fetchall()
        vals = []
        for r in rows:
            # pnl_r = pnl_usdt / (sl_dist * qty)  →  sl_dist = pnl_usdt / (pnl_r * qty)
            sl_dist = abs(float(r['pnl_usdt']) / (float(r['pnl_r']) * float(r['qty_tokens'])))
            pct = sl_dist / float(r['entry_price'])
            if 0.002 < pct < 0.5:
                vals.append(pct)
        if len(vals) >= 10:
            return float(np.median(vals))
    except Exception as e:
        log.debug(f'[진단] SL 중앙값 계산 실패: {e}')
    return 0.05          # 관측치가 없을 때의 대표값 (ATR 기반 전략들의 통상 범위)


def _diagnose_sizing_capability(equity: float) -> list[dict]:
    """전략 × 레짐 조합별로 **실제로 진입 가능한지** 점검한다.

    사이징은 여러 스케일의 곱이라(기본 × Kelly × 래칫 × 레짐 × 건강도)
    작은 값들이 겹치면 주문금액이 거래소 최소치($5) 아래로 내려간다.
    그러면 그 조합은 **신호가 나와도 영원히 체결되지 않는데**, 로그에만
    한 줄 남아 운영자는 '전략이 돌고 있다'고 믿게 된다. 소액 계좌에서
    하락장 커버리지가 통째로 죽는 것이 대표적인 경우다.
    기동 시 한 번 계산해 죽은 조합을 명시적으로 알린다.
    """
    sl_pct = _typical_sl_pct()
    out = []
    for regime, strats in REGIME_STRATEGY_MAP.items():
        if regime == 'WEAK_TREND':
            r_scale = WEAK_TREND_RISK_SCALE
        elif regime == 'TRENDING_DOWN':
            r_scale = TRENDING_DOWN_RISK_SCALE
        else:
            r_scale = 1.0
        for sid in strats:
            kelly  = _get_kelly_scale(sid)
            health = _get_strategy_health_scale(sid)
            risk   = SPOT_BASE_RISK_PCT * kelly * r_scale * health
            cost   = (equity * risk / sl_pct) if sl_pct > 0 else 0.0
            cost   = min(cost, equity * SPOT_MAX_ALLOC_PCT)
            out.append({
                'strategy': sid, 'regime': regime,
                'risk_pct': risk, 'cost_usdt': cost,
                'tradable': cost >= SPOT_MIN_ORDER_USDT,
                'kelly': kelly, 'health': health, 'regime_scale': r_scale,
            })
    return out


def _report_sizing_capability(equity: float) -> None:
    """진단 결과를 로그·텔레그램으로 알린다."""
    try:
        rows = _diagnose_sizing_capability(equity)
    except Exception as e:
        log.warning(f'[진단] 사이징 점검 실패(무시): {e}')
        return
    sl_pct = _typical_sl_pct()
    dead = [r for r in rows if not r['tradable']]
    log.info(f'[진단] 사이징 점검 (자산 ${equity:,.2f}, 전형 SL {sl_pct*100:.1f}%)')
    for r in rows:
        mark = '  ' if r['tradable'] else '✗ '
        log.info(f'  {mark}{r["strategy"]}/{r["regime"]:<14} '
                 f'리스크 {r["risk_pct"]*100:5.3f}% '
                 f'(설정 {SPOT_BASE_RISK_PCT*100:.1f}% 대비 '
                 f'{r["risk_pct"]/SPOT_BASE_RISK_PCT*100:3.0f}%) '
                 f'→ 주문 ${r["cost_usdt"]:6.2f}')
    if dead:
        # 금액만 알리면 "왜 죽었는지"를 알 수 없어 어느 레버를 당길지 못 정한다.
        # 배수를 분해해 보여준다 — 대개 Kelly가 하한까지 내려간 것이 주원인이고,
        # 그건 그 전략의 실적이 나쁘다는 뜻이라 '고칠 문제'가 아닐 수도 있다
        # (성적 나쁜 전략을 하락장에서 최소로 줄이는 건 설계대로 동작한 것이다).
        lines = '\n'.join(
            f'   • {r["strategy"]} / {r["regime"]} — 주문 ${r["cost_usdt"]:.2f} '
            f'({SPOT_MIN_ORDER_USDT / r["cost_usdt"]:.2f}x 부족)\n'
            f'     kelly {r["kelly"]:.2f} × 건강도 {r["health"]:.2f} × '
            f'레짐 {r["regime_scale"]:.2f} → 리스크 {r["risk_pct"] * 100:.3f}%'
            for r in dead if r['cost_usdt'] > 0)
        need_eq = max((equity * SPOT_MIN_ORDER_USDT / r['cost_usdt']
                       for r in dead if r['cost_usdt'] > 0), default=0.0)
        _tg(f'⚠️ 진입 불가 조합 {len(dead)}건 (주문금액이 거래소 최소 '
            f'${SPOT_MIN_ORDER_USDT:.0f} 미달)\n{lines}\n'
            f'   신호가 나와도 체결되지 않습니다.\n'
            f'   해소 방법: 자산 ${need_eq:,.0f} 이상으로 늘리거나, 위 배수 중 '
            f'하나를 조정합니다. kelly가 하한이면 그 전략의 실적이 나쁘다는 '
            f'뜻이므로 배수를 올리기 전에 전략 자체를 먼저 검토하세요.')


_regime_idle_alerted: set = set()
_REGIME_IDLE_CHECK_SEC = 600     # 재점검 간격(초)
_regime_idle_last = 0.0


def tradable_strategies(regime: str, equity: float) -> list:
    """현 레짐에서 **실제로 진입 가능한** 전략 목록."""
    return [r['strategy'] for r in _diagnose_sizing_capability(equity)
            if r['regime'] == regime and r['tradable']]


def check_regime_idle(regime: str, equity: float) -> str:
    """지금 레짐에서 아무 전략도 진입할 수 없으면 알림 문구를, 아니면 ''.

    기동 시 진단(_report_sizing_capability)은 죽은 **조합**을 나열하지만,
    "지금 이 레짐에서는 하나도 못 산다"는 상태 자체는 말해주지 않는다.
    소액 계좌에서 하락장에 들어가면 담당 전략이 통째로 최소주문액 아래로
    떨어져 봇이 **조용히 논다** — 로그는 정상이고 프로세스도 살아 있어
    운영자는 계속 매매 중이라 믿는다. 자산이 줄면 더 많은 조합이 죽는데
    기동 진단은 그때 이미 지나간 뒤다.

    CRISIS는 설계상 전면 차단이므로 알리지 않는다(정상 동작).
    레짐당 1회만 알리고, 진입 가능해지면 해제해 다음 발생 시 다시 알린다.
    """
    # 담당 전략이 **애초에 배정되지 않은** 레짐은 이 경보의 대상이 아니다.
    # 이 경보가 말하려는 건 "자본이 모자라 신호가 나와도 못 산다"이지
    # "설계상 쉬는 중"이 아니다. 맵에는 빈 레짐이 셋 있다 —
    #   CRISIS(변동성 폭발 시 전면 정지) · MICRO_RANGING(기존 동작 보존) ·
    #   UNKNOWN(레짐 판별 실패 시 안전 정지)
    # 특히 UNKNOWN은 기동 직후 RegimeLoop가 첫 분류를 내기까지 5초 남짓
    # 반드시 지나가는 상태라, 거르지 않으면 **재시작할 때마다** 허위 경보가
    # 나간다(실측: 11:14:56 '레짐(UNKNOWN)에서 진입 가능한 전략이 없습니다').
    if not REGIME_STRATEGY_MAP.get(regime):
        return ''
    if tradable_strategies(regime, equity):
        _regime_idle_alerted.discard(regime)
        return ''
    if regime in _regime_idle_alerted:
        return ''
    _regime_idle_alerted.add(regime)
    assigned = REGIME_STRATEGY_MAP.get(regime, [])
    return (f'⚠️ 현재 레짐({regime})에서 진입 가능한 전략이 없습니다\n'
            f'   담당 전략: {", ".join(assigned) or "없음"}\n'
            f'   전부 주문금액이 거래소 최소 ${SPOT_MIN_ORDER_USDT:.0f} 미달입니다.\n'
            f'   신호가 나와도 체결되지 않습니다 — 봇은 돌지만 실질적으로 대기 상태입니다.\n'
            f'   자산 ${equity:,.2f} · 해소하려면 자본을 늘리거나 해당 레짐의 '
            f'리스크 스케일을 조정해야 합니다.')


def _report_regime_idle() -> None:
    """주기적으로 '현 레짐에서 아무것도 못 사는' 상태를 점검·알린다."""
    global _regime_idle_last
    now = time.time()
    if now - _regime_idle_last < _REGIME_IDLE_CHECK_SEC:
        return
    _regime_idle_last = now
    try:
        rs = get_cached_regime()
        msg = check_regime_idle(rs.regime if rs else '', _state['equity'])
    except Exception as e:
        log.debug(f'[진단] 레짐 진입가능 점검 실패(무시): {e}')
        return
    if msg:
        log.warning(msg.replace('\n', ' '))
        _tg(msg)


def _estimate_round_trip_cost(ccxt_sym: str) -> tuple[float, float]:
    """(왕복 비용률, 스프레드율) 추정.

    시장가 왕복은 수수료 2회 + 스프레드 1회분(매수는 ask, 매도는 bid)
    + 호가 이탈 슬리피지 2회를 지불한다. 호가 조회 실패 시 보수적 기본값.
    """
    spread = SPOT_DEFAULT_SPREAD_PCT
    try:
        t = _get_ex().fetch_ticker(ccxt_sym)
        bid = float(t.get('bid') or 0)
        ask = float(t.get('ask') or 0)
        if bid > 0 and ask > bid:
            spread = (ask - bid) / ((ask + bid) / 2)
    except Exception as e:
        log.debug(f'[{ccxt_sym}] 호가 조회 실패 — 기본 스프레드 사용: {e}')
    cost = _detect_fee_rate() * 2 + spread + SPOT_ASSUMED_SLIP_PCT * 2
    return cost, spread


def _get_realized_avg_r(strategy: str) -> tuple[float, int, float]:
    """(실현 avg_r, 표본수, 표준오차).
    pnl_r은 수수료 차감 전(gross)이라 비용률과 직접 비교할 수 있다."""
    with _db_lock, _db_conn() as conn:
        rows = conn.execute(
            'SELECT pnl_r FROM spot_trades WHERE strategy=? AND COALESCE(dry_run,0)=0 '
            'ORDER BY id DESC LIMIT 200', (strategy,)
        ).fetchall()
    if not rows:
        return 0.0, 0, 0.0
    vals = [float(r['pnl_r'] or 0) for r in rows]
    n = len(vals)
    stderr = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(np.mean(vals)), n, stderr


def _cost_edge_ok(strategy: str, symbol: str, ccxt_sym: str,
                  sl_dist: float, entry_price: float) -> tuple[bool, str]:
    """비용이 1R을 얼마나 잠식하는지 확인.

    포지션 명목가 = 리스크금액 / SL거리% 이므로 왕복비용은
    리스크금액 × (비용률 / SL거리%)가 된다. 이 값이 전략의 실현 avg_r을
    넘으면 그 거래는 구조적으로 마이너스 기대값이다 — 승률과 무관하게.
    """
    if entry_price <= 0 or sl_dist <= 0:
        return True, ''
    sl_dist_pct = sl_dist / entry_price
    cost_rate, spread = _estimate_round_trip_cost(ccxt_sym)
    cost_per_r = cost_rate / sl_dist_pct
    if cost_per_r > SPOT_MAX_COST_PER_R:
        return False, (f'비용이 1R의 {cost_per_r*100:.0f}% 잠식 '
                       f'(SL {sl_dist_pct*100:.2f}%, 스프레드 {spread*100:.3f}%) '
                       f'> 한도 {SPOT_MAX_COST_PER_R*100:.0f}%')
    avg_r, n, stderr = _get_realized_avg_r(strategy)
    # 표본 노이즈로 멀쩡한 전략을 막지 않도록, **낙관적 상단(avg_r+1SE)조차**
    # 비용을 못 넘을 때만 차단한다. 20건 남짓의 avg_r 추정은 분산이 크다.
    if n >= SPOT_EDGE_MIN_TRADES and (avg_r + stderr) <= cost_per_r:
        return False, (f'실현 avg_r {avg_r:+.3f}±{stderr:.3f}(n={n})의 낙관치도 '
                       f'비용 {cost_per_r:.3f}R을 못 넘음 — 마이너스 기대값')
    return True, ''


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
    # 학습기를 켜면 이 경로는 **비활성**이다 — 같은 일(성과 기반 감봉)을
    # 레짐까지 나눠서 하므로, 둘 다 걸면 같은 근거로 두 번 깎는 셈이다.
    health = 1.0 if SPOT_LEARN_ENABLED else _get_strategy_health_scale(strategy)
    if health <= 0:
        log.warning(f'[{strategy}] {symbol} 매수 차단: 실계좌 net PF < '
                    f'{SPOT_HEALTH_PF_HARD} (표본 {SPOT_HEALTH_MIN_TRADES}건+)')
        if not _state.get(f'health_blocked_{strategy}'):
            _state[f'health_blocked_{strategy}'] = True
            _tg(f'⛔ [{strategy}] 실계좌 net PF < {SPOT_HEALTH_PF_HARD} — '
                f'신규 진입 자동 차단 (성과 회복 시 자동 해제)')
        return False
    _state.pop(f'health_blocked_{strategy}', None)

    # 학습기 ON이면 Kelly 자리를 학습 배분이 대신한다(곱하지 않는다).
    # 스케일을 겹겹이 곱하면 주문이 거래소 최소액 아래로 내려가는 문제가
    # 있었고(실효 리스크가 설정값의 9%까지 내려갔다), 학습기는 Kelly가
    # 하려던 일을 레짐까지 나눠서 더 정교하게 한다.
    if SPOT_LEARN_ENABLED:
        kelly = _get_learn_scale(strategy, regime)
    else:
        kelly = _get_kelly_scale(strategy)
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

    # 비용 대비 엣지: SL이 좁을수록 명목가가 커져 왕복비용이 R을 잠식한다.
    # 승률이 아무리 좋아도 비용이 avg_r을 넘으면 그 거래는 마이너스 기대값이다.
    _cost_ok, _cost_why = _cost_edge_ok(strategy, symbol, ccxt_sym, sl_dist, entry_price)
    if not _cost_ok:
        log.info(f'[{strategy}] {symbol} 매수 차단(비용): {_cost_why}')
        return False

    adj_risk, qty, cost_usdt = _size_position(
        equity, sl_dist, entry_price,
        kelly=kelly, ratchet=ratchet, regime=r_scale,
        funding=funding_scale, rs=rs_scale, health=health)

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
    # 체결 슬리피지 계측: 신호가 대비 실제 체결가. 매수는 비싸게 사면 손해(양수).
    # 계측이 없으면 실행 비용이 엣지를 갉아먹는지 알 방법이 없다.
    entry_slip = (fill_price - entry_price) / entry_price if entry_price > 0 else 0.0
    if abs(entry_slip) > 0.002:
        log.warning(f'[{strategy}] {symbol} 진입 슬리피지 {entry_slip*100:+.3f}% '
                    f'(신호 {entry_price:,.4f} → 체결 {fill_price:,.4f})')
    try:
        _save_position(
            strategy, symbol, fill_price, sl_final, tp_final,
            qty, cost_usdt, adj_risk,
            sig.get('exit_type', 'sl_tp'), sig.get('max_hold', 0), regime,
            entry_slip
        )
    except Exception as e_save:
        log.error(f'[{strategy}] {symbol} 포지션 저장 실패 — 재시도: {e_save}')
        try:
            _save_position(
                strategy, symbol, fill_price, sl_final, tp_final,
                qty, cost_usdt, adj_risk,
                sig.get('exit_type', 'sl_tp'), sig.get('max_hold', 0), regime,
                entry_slip
            )
        except Exception as e_save2:
            log.critical(f'[{strategy}] {symbol} 포지션 저장 2회 실패 — 무보호 포지션 발생: {e_save2}')
            emergency_sl = ''
            if not _state['dry_run']:
                try:
                    emergency_sl = _place_stop_loss_order(strategy, symbol, ccxt_sym,
                                                          qty, sl_final)
                except Exception as e:
                    # DB 저장이 이미 실패한 상태에서 비상 손절까지 실패하면
                    # **아무 보호도 없는 포지션**이 남는다. 아래 알림은 DB
                    # 실패만 알리므로, 이걸 삼키면 운영자는 손절이 걸린 줄 안다.
                    log.error(f'[{strategy}] {symbol} 비상 손절 주문 실패 — '
                              f'무보호 포지션: {e}')
                    emergency_sl = ''
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
    sl          = float(pos['sl'])
    # R배수는 **진입 시점의 위험**으로 재야 한다. 추적 손절로 sl이 올라간
    # 뒤 그 값을 분모로 쓰면 R이 부풀고, 그 pnl_r이 Kelly·건강도·avg_r과
    # 비용 가드까지 연쇄 오염시킨다.
    orig_sl     = float(pos.get('orig_sl') or 0) or sl
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
            _base_asset = base_of(ccxt_sym)
            _pre_bal = _get_ex().fetch_balance()
            _actual_free = float(_pre_bal['free'].get(_base_asset, 0))
            if _actual_free <= 0.0:
                # free=0의 원인이 "수동매도"가 아니라 **미체결 매도주문이 수량을
                # 잠근 것**일 수 있다(DB에 ID가 없는 고아 OCO/스탑 등). 이를
                # 구분하지 않으면 살아있는 포지션을 허위 MANUAL_SOLD로 지운다.
                _actual_free = _cancel_orphan_sell_orders(strategy, symbol, ccxt_sym,
                                                          _base_asset, _actual_free)
            if _actual_free <= 0.0:
                # 잔고가 0인 **가장 흔한 이유는 거래소 보호주문이 먼저 체결된
                # 것**이다(매도 전 스탑 취소는 이미 체결된 주문에선 조용히
                # 무시된다). 체결이면 실제 체결가·정확한 사유로 기록된다.
                if _try_settle_via_stop_fill(strategy, symbol, ccxt_sym, pos,
                                             f'[{strategy}] {symbol}'):
                    return
                log.warning(f'[{strategy}] {symbol} 실잔고 0 → 수동매도로 자동처리 ({reason})')
                _tg(f'ℹ️ [{strategy}] {symbol} 잔고 없음 → 수동매도 DB정리')
                _settle_closed_position(
                    strategy, symbol, entry_price=entry_price, exit_price=price,
                    qty=qty, cost_usdt=cost_usdt, reason='MANUAL_SOLD',
                    regime=regime, entry_ts=entry_ts, sl_for_r=orig_sl,
                    round_hold=True)
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
            # 잔고 부족이 아닌 오류(거래정지·상장폐지·레이트리밋 등)면 포지션이
            # 남고, 다음 관리 주기(5분)마다 같은 청산 판정 → 같은 실패 →
            # 같은 알림이 반복된다. 로그는 매번 남기되 알림만 간격을 둔다.
            if _stop_alert_due(strategy, symbol, 'sell_fail'):
                _tg(f'⚠️ [{strategy}] {symbol} 매도 실패: {e}')
            # insufficient balance: 실제 잔고 확인 후 0에 가까우면 수동매도로 자동 처리
            if 'insufficient balance' in err_str or 'insufficient funds' in err_str:
                # 잔고가 없는 **가장 흔한 이유는 거래소 보호주문이 먼저
                # 체결된 것**이다. 체결이면 실제 체결가·정확한 사유로 기록된다.
                if _try_settle_via_stop_fill(strategy, symbol, ccxt_sym, pos,
                                             f'[{strategy}] {symbol}'):
                    return
                try:
                    _base = base_of(ccxt_sym)
                    _bal = _get_ex().fetch_balance()
                    _actual_free = float(_bal['free'].get(_base, 0))
                    if _actual_free < qty * 0.05:
                        log.warning(f'[{strategy}] {symbol} 수동매도 감지(잔고={_actual_free:.4f}) → DB자동정리')
                        _tg(f'ℹ️ [{strategy}] {symbol} 수동매도 감지 → DB 정리 완료')
                        _settle_closed_position(
                            strategy, symbol, entry_price=entry_price,
                            exit_price=price, qty=qty, cost_usdt=cost_usdt,
                            reason='MANUAL_SOLD', regime=regime,
                            entry_ts=entry_ts, sl_for_r=orig_sl, round_hold=True)
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

    # 왕복 슬리피지 = 진입(비싸게 삼) + 청산(싸게 팜). 둘 다 양수면 손실.
    _exit_slip = (price - exit_price) / price if price > 0 else 0.0
    _slip_total = float(pos.get('entry_slip_pct') or 0.0) + _exit_slip
    net_pnl, pnl_usdt, pnl_pct, hold_hours = _settle_closed_position(
        strategy, symbol, entry_price=entry_price, exit_price=exit_price,
        qty=qty, cost_usdt=cost_usdt, reason=reason, regime=regime,
        entry_ts=entry_ts, sl_for_r=orig_sl, slip_pct=_slip_total)

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
        _net, pnl_usdt, pnl_pct, _hold = _settle_closed_position(
            strategy, symbol, entry_price=entry_price, exit_price=exit_price,
            qty=qty, cost_usdt=float(pos['cost_usdt']), reason=reason,
            regime=pos.get('regime', ''), entry_ts=pos['entry_ts'],
            # R배수 분모는 진입 시점의 위험 (추적 손절로 sl이 올라가도 불변)
            sl_for_r=(float(pos.get('orig_sl') or 0) or float(pos['sl'])))
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
_REARM_GIVEUP = 86400               # 구조적 불가 시 재시도 보류 기간(초)


def _rearm_trailing_stop(strategy: str, symbol: str, ccxt_sym: str,
                         pos: dict, new_sl: float) -> None:
    """추적 손절로 SL이 올라갔을 때 거래소 보호주문도 새 가격으로 재등록.

    갱신하지 않으면 거래소에는 원래(더 낮은) 손절만 남아, 봇이 멈춘 사이
    추적으로 확보한 이익이 보호되지 않는다. 다만 주문 취소·재등록은
    API 비용이 있으므로 **의미 있게 움직였을 때만** 수행한다.
    """
    if _state.get('dry_run') or not SPOT_EXCHANGE_STOP:
        return
    old_sl = float(pos.get('sl') or 0)
    sl_dist = abs(float(pos['entry_price']) - old_sl)
    if sl_dist <= 0 or (new_sl - old_sl) < sl_dist * SPOT_TRAIL_REARM_FRAC:
        return                                  # 미세 이동 — 주문 churn 방지
    try:
        _cancel_stop_order(strategy, symbol, ccxt_sym, pos.get('sl_order_id') or '')
        _cancel_stop_order(strategy, symbol, ccxt_sym, pos.get('tp_order_id') or '')
        tp_for_oco = 0.0 if strategy == 'S5' else float(pos.get('tp') or 0)
        sl_id, tp_id = _place_protective_orders(
            strategy, symbol, ccxt_sym, float(pos['qty_tokens']), new_sl, tp_for_oco)
        _update_position_order_id(strategy, symbol, sl_id, tp_id)
        if not sl_id:
            log.warning(f'[{strategy}] {symbol} 추적 손절 재등록 실패 — '
                        f'소프트웨어 SL만 작동 (자가복구가 재시도한다)')
    except Exception as e:
        log.warning(f'[{strategy}] {symbol} 추적 손절 재등록 오류(무시): {e}')


def _protection_impossible(ccxt_sym: str, qty: float, sl_price: float) -> bool:
    """거래소 보호주문이 **구조적으로** 불가능한가.

    주문 금액이 거래소 최소치에 못 미치면 재시도해도 영원히 실패한다.
    (소액 계좌에서 흔하다 — `_diagnose_sizing_capability` 참조)
    일시적 실패와 구분하지 않으면 5분마다 무한 재시도하며 API만 낭비한다.
    """
    q = _sellable_qty(ccxt_sym, qty)
    return q <= 0 or q * sl_price * (1 - SPOT_STOP_LIMIT_GAP) < SPOT_MIN_ORDER_USDT


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
    # 구조적 불가(주문금액 < 거래소 최소치)면 재시도해도 영원히 실패한다.
    # 소프트웨어 SL이 유일한 보호라는 사실만 1회 알리고 재시도를 멈춘다.
    if _protection_impossible(ccxt_sym, float(pos['qty_tokens']), float(pos['sl'])):
        if cnt == 0:
            _tg(f'ℹ️ [{strategy}] {symbol} 거래소 보호주문 불가 — 주문금액이 '
                f'최소치(${SPOT_MIN_ORDER_USDT:.0f}) 미만입니다.\n'
                f'   소프트웨어 SL만 작동합니다(봇이 멈추면 손절되지 않음).')
        _rearm_attempts[key] = (now + _REARM_GIVEUP, max(cnt, 1))
        return
    # 손절가가 거래소 허용 가격범위(PERCENT_PRICE_BY_SIDE) 밖인 경우.
    # NOTIONAL 미달과 달리 **가격에 따라 변하는** 조건이라 영구 포기하면 안 된다 —
    # 가격이 손절선 쪽으로 내려오면 등록이 가능해지고, 그때가 정확히 보호가
    # 필요한 순간이다. 판정은 캐시된 값만 쓰므로 매 주기 다시 봐도 싸다.
    # 다만 사유는 사람이 알아야 하므로 간격을 두고 한 번씩만 알린다.
    floor = _min_sell_price(ccxt_sym)
    sl_lim = float(pos['sl']) * (1 - SPOT_STOP_LIMIT_GAP)
    if floor > 0 and sl_lim < floor:
        # 알림과 같은 간격으로 로그도 남긴다. 흔적이 전혀 없으면 운영자가
        # 로그를 뒤져도 '왜 보호주문이 없는지' 알 수 없다(조용한 보류 금지).
        # 매 주기(5분) 남기면 하루 288줄이라 그 자체가 스팸이 된다.
        if _stop_alert_due(strategy, symbol):
            gap = (floor - sl_lim) / floor * 100
            log.info(f'[{strategy}] {symbol} 보호주문 보류: 손절 지정가 '
                     f'{sl_lim:.8g} < 거래소 허용 하한 {floor:.8g} '
                     f'({gap:.1f}% 초과) — 소프트웨어 SL 감시 중')
            _tg(f'ℹ️ [{strategy}] {symbol} 거래소 보호주문 보류 — 손절가가 '
                f'거래소 허용 범위 밖입니다({gap:.1f}% 초과).\n'
                f'   현재가가 손절선에 가까워지면 자동으로 등록됩니다.\n'
                f'   그때까지는 소프트웨어 SL이 감시합니다.')
        _rearm_attempts[key] = (now, cnt)
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


def _s5_safety_block(symbol: str) -> Optional[str]:
    """S5 전용 안전 필터. 차단 사유를 반환하고, 통과하면 None.

    (전문가 회의 결론 2026-06-04)
      [1] SL 쿨다운 — 같은 종목에서 손절 직후 재진입하면 같은 하락에
          연속으로 맞는다. 평균회귀 전략이라 '더 싸졌으니 또 산다'가
          되기 쉬운 구조다.
      [2] BTC 상관 그룹 한도 — 상관이 높은 종목에 동시 진입하면 분산이
          아니라 같은 베팅을 여러 번 하는 것이다.
    """
    with _db_lock, _db_conn() as c:
        row = c.execute(
            "SELECT exit_ts FROM spot_trades "
            "WHERE strategy='S5' AND symbol=? AND reason='SL' "
            "ORDER BY exit_ts DESC LIMIT 1", (symbol,)
        ).fetchone()
    if row:
        try:
            hours = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(row[0])).total_seconds() / 3600
        except (ValueError, TypeError) as e:
            log.warning(f'[S5] {symbol} SL 시각 파싱 실패 — 쿨다운 미적용: {e}')
            hours = None
        if hours is not None:
            limit_h = S5_SL_COOLDOWN_BARS * 24
            if hours < limit_h:
                log.info(f'[S5] {symbol} SL쿨다운: {hours:.0f}h 전 SL ({limit_h:.0f}h 대기)')
                return 'sl_cooldown'

    if symbol in S5_BTC_CORR_SYMBOLS:
        syms = tuple(S5_BTC_CORR_SYMBOLS)
        with _db_lock, _db_conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM spot_positions WHERE strategy='S5' "
                f"AND symbol IN ({','.join('?' * len(syms))})", syms
            ).fetchone()[0]
        if n >= S5_CORR_MAX_POS:
            log.info(f'[S5] {symbol} BTC상관 한도: {n}/{S5_CORR_MAX_POS}개 진입중')
            return 'btc_corr_limit'
    return None


def _rs_gate_scale(strategy_id: str, symbol: str) -> Optional[float]:
    """모멘텀 RS Gate + 주도주 부스트. 차단이면 **None**.

    차단(None)과 감봉(작은 배수)을 타입으로 구분한다 — 0.0으로 표현하면
    곱셈에 흘러들어가 '수량 0 주문'이 된다.

    백테스트 `_bt_rs_gate`와 같은 규칙이다. 두 함수가 갈라지면 WFO가
    검증한 임계값이 실계좌에서 다르게 동작한다.
    """
    if strategy_id not in MOMENTUM_RS_GATE_STRATS:
        return 1.0
    rank_pct = _get_momentum_rank_pct(symbol)
    if rank_pct > MOMENTUM_RS_GATE_PCT:
        log.debug(f'[RS Gate] {symbol} 모멘텀 하위권({rank_pct:.0%}) — {strategy_id} 차단')
        return None
    return MOMENTUM_TOP_RISK_MULT if rank_pct <= MOMENTUM_TOP_TIER_PCT else 1.0


def _funding_scale(strategy_id: str, symbol: str) -> Optional[float]:
    """펀딩비 기반 리스크 배수. 롱 과밀이면 **None**(진입 차단).

    백테스트도 같은 규칙을 적용한다 — `atlas_spot_backtest._bt_funding_scale`
    이 `build_funding_map()` 으로 받은 과거 펀딩 이력을 보고 판정한다.
    (예전에는 이력을 받지 않아 백테스트가 라이브보다 낙관적이었다)
    퍼프가 없는 심볼은 여기서 조회 실패로 0.0이 되어 통과하는데,
    백테스트도 데이터 없음을 통과로 처리해 동작을 맞춘다.
    """
    if strategy_id not in FUNDING_APPLY_STRATS:
        return 1.0
    funding = _get_spot_funding(symbol)
    if funding >= FUNDING_LONG_BLOCK:
        log.info(f'[펀딩 차단] {symbol} funding={funding*100:.3f}%/8h — 롱 과밀')
        return None
    if funding <= FUNDING_SHORT_BOOST:
        return 1.20      # 숏 스퀴즈 기대 구간 +20%
    return 1.0


def _size_position(equity: float, sl_dist: float, entry_price: float, *,
                   kelly: float = 1.0, ratchet: float = 1.0,
                   regime: float = 1.0, funding: float = 1.0,
                   rs: float = 1.0, health: float = 1.0) -> tuple:
    """(실효 리스크비율, 수량, 주문금액)을 계산한다. **부수효과 없음.**

    리스크는 스케일들의 **곱**이다. 이 저장소에서 가장 비싼 버그가 여기서
    나왔다 — 1보다 작은 값이 겹쳐 실효 리스크가 설정값의 9%까지 내려갔고,
    주문금액이 거래소 최소치($5) 아래로 떨어져 하락장 전략이 통째로
    체결되지 않았다(dc7fb24). 그래서 각 스케일을 **키워드 인자로** 받아
    호출부에서 무엇이 곱해지는지 한눈에 보이게 한다.

    배분 상한에 걸리면 수량을 줄이고 **실효 리스크를 역산**한다. 상한 때문에
    실제로 건 위험이 줄었는데 adj_risk를 그대로 두면, 그 값이 DB에 저장돼
    Kelly·건강도·학습기 통계를 전부 부풀린다.
    """
    if equity <= 0 or sl_dist <= 0 or entry_price <= 0:
        return 0.0, 0.0, 0.0
    adj_risk  = (SPOT_BASE_RISK_PCT * kelly * ratchet * regime
                 * funding * rs * health)
    qty       = (equity * adj_risk) / sl_dist
    cost_usdt = qty * entry_price
    cap = equity * SPOT_MAX_ALLOC_PCT
    if cost_usdt > cap:
        cost_usdt = cap
        qty       = cost_usdt / entry_price
        adj_risk  = (qty * sl_dist) / equity
    return adj_risk, qty, cost_usdt


def _live_exit_decision(strategy: str, symbol: str, df, i: int, price: float,
                        entry: float, sl: float, tp: float,
                        bars_held: int, max_hold: int) -> Optional[str]:
    """지금 청산해야 하는가. Returns: 사유 문자열 or None. **부수효과 없음.**

    백테스트의 `_bt_exit_decision`과 **대칭으로** 뽑아 둔 함수다. 두 함수를
    나란히 두면 라이브에만 있는 규칙이 눈에 보인다 — 실제로 이 분리 과정에서
    아래 두 가지 격차가 드러났다.

    ⚠️ **백테스트가 모델링하지 않는 라이브 전용 청산 2건**

      1. `BB_MID` (S4) — 라이브는 가격이 볼린저 중앙선에 닿으면 청산한다.
         백테스트의 `EXIT_CHECK_FUNCS`에는 S4가 **없어서**(S2/S3/S6/S7만)
         SL·TP·TIME으로만 청산한다. 즉 백테스트의 S4는 라이브보다 오래
         들고 간다 — 평균회귀 전략이라 되돌림을 더 먹거나 더 토해낸다.

      2. S5의 실시간 TP 갱신 — 라이브는 매 폴링마다 TP를 현재
         `bb_upper`로 옮긴다(이 함수 호출 **전에** 수행). 백테스트는
         진입 시점의 정적 TP를 끝까지 쓴다.

    둘 다 성과 차이를 만들지만, 고치려면 백테스트 청산 규칙을 바꿔야 하고
    그건 과거 검증 결과 전체를 무효화한다. 지금은 **격차를 명시**해 두고,
    WFO를 다시 돌릴 때 함께 처리하는 편이 안전하다.
    """
    if price <= sl:
        return 'SL'
    if tp > 0 and price >= tp:
        return 'TP'

    exit_fn = EXIT_CHECK_FUNCS.get(strategy)
    if exit_fn is not None and df is not None:
        try:
            should_exit = (exit_fn(df, i, entry) if strategy == 'S6'
                           else exit_fn(df, i))
            if should_exit:
                return 'CROSS'
        except Exception as e:
            # 청산 함수가 죽으면 그 봉의 청산 신호가 **사라진다**. SL/TP는
            # 별도로 살아 있으므로 즉시 위험하진 않지만, 반복되면 전략이
            # 의도한 청산 규칙 없이 도는 셈이다.
            log.warning(f'[{strategy}] {symbol} 청산 조건 평가 실패 '
                        f'(이번 봉 CROSS 청산 건너뜀): {e}')

    if max_hold > 0 and bars_held >= max_hold:
        return 'TIME'

    # S4: BB 중앙선 도달 청산 (라이브 전용 — 위 경고 참조)
    if strategy == 'S4' and df is not None and 'bb_mid' in df.columns:
        bb_mid = float(df.iloc[i]['bb_mid']) if not pd.isna(df.iloc[i]['bb_mid']) else 0
        if bb_mid > 0 and price >= bb_mid:
            return 'BB_MID'

    return None


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
            # 거래소 보호주문이 있으면 청산하지 않는다.
            #
            # 이 경로의 목적은 'SL이 작동하지 못하는 상태'를 막는 것인데,
            # 거래소 스탑이 걸려 있으면 SL은 봇의 시야와 무관하게 거래소가
            # 집행한다. 오히려 _spot_sell 은 매도 **전에 그 스탑을 취소**하므로,
            # 이어지는 시장가 매도가 실패하면(가격 조회를 막은 그 장애로
            # 실패하기 쉽다) 보호가 통째로 사라진 채 포지션만 남는다 —
            # 의도와 정반대다.
            #
            # 시세 장애는 몇 분씩 흔하고, 하필 그때의 시장가 체결은 가장
            # 불리하다. 보호가 있는 포지션까지 던질 이유가 없다.
            if pos.get('sl_order_id') or pos.get('tp_order_id'):
                if _stop_alert_due(strategy, symbol):
                    log.warning(f'[{strategy}/{symbol}] 가격 조회 불가 '
                                f'{_PRICE_CACHE_TTL}s 초과 — 거래소 보호주문이 '
                                f'있어 긴급 청산 보류(SL은 거래소가 집행)')
                    _tg(f'⚠️ [{strategy}/{symbol}] 시세 조회 불가 — 봇이 가격을 '
                        f'보지 못합니다.\n'
                        f'   거래소 손절이 걸려 있어 청산하지 않고 대기합니다.')
                return
            log.error(f'[{strategy}/{symbol}] 가격 조회 불가 {_PRICE_CACHE_TTL}s 초과 — 긴급 청산')
            _tg(f'🚨 [{strategy}/{symbol}] 가격 조회 불가 — 긴급 청산')
            _spot_sell(strategy, symbol, ccxt_sym, pos, entry * 0.99, 'EMERGENCY')
        return

    sl        = float(pos['sl'])
    tp        = float(pos['tp'])
    # exit_type은 **기록용 메타데이터**다. 실제 청산 분기는
    # EXIT_CHECK_FUNCS(전략별 청산 함수)가 담당하므로 여기서 읽지 않는다.
    # (읽고 버리면 이 값이 동작을 제어한다고 오해하게 된다)
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
    except Exception as e:
        # 보유 봉수가 어긋나면 시간 기반 청산(max_hold) 시점이 밀린다.
        bars_held = int(pos.get('bars_held', 0))
        log.debug(f'[{strategy}] {symbol} 보유봉수 계산 실패, DB값 사용: {e}')
    peak      = float(pos.get('peak_price', entry))

    # S5: BB_upper를 실시간 TP로 업데이트 (DB도 갱신하여 대시보드 정확도 향상)
    if strategy == 'S5' and df is not None and 'bb_upper' in df.columns:
        live_bb_upper = float(df.iloc[i]['bb_upper']) if not pd.isna(df.iloc[i]['bb_upper']) else 0
        if live_bb_upper > 0:
            if abs(live_bb_upper - float(pos.get('tp', 0))) > 1e-8:
                _update_position_tp(strategy, symbol, live_bb_upper)
            tp = live_bb_upper

    reason = _live_exit_decision(strategy, symbol, df, i, price,
                                 entry, sl, tp, bars_held, max_hold)
    if reason == 'SL':
        # NOTIONAL 사전 검사: 포지션 가치 < $5이면 Binance가 주문 거부
        # → 매도 시도하지 않고 DB 유지, 가격 회복 시 자동 매도
        _pos_val = price * float(pos.get('qty_tokens', 0))
        if _pos_val < SPOT_MIN_ORDER_USDT:
            log.info(f'[{strategy}] {symbol} SL 감지 → NOTIONAL 미달(${_pos_val:.2f}) — 가격 회복 대기')
            return  # 포지션 DB 유지, 다음 폴링에서 재검사
    if reason:
        _spot_sell(strategy, symbol, ccxt_sym, pos, price, reason)
        return

    # Peak 갱신 + 추적 손절
    # (기존에는 peak만 기록하고 sl은 그대로 다시 써서 효과가 0이었다)
    if price > peak:
        peak = price
        new_sl = trailing_sl(entry, sl, peak, abs(entry - float(pos['sl'])))
        _update_position_sl(strategy, symbol, new_sl, peak)
        if new_sl > sl:
            log.info(f'[{strategy}] {symbol} 추적 손절 {sl:,.4f} → {new_sl:,.4f} '
                     f'(최고 {peak:,.4f})')
            _rearm_trailing_stop(strategy, symbol, ccxt_sym, pos, new_sl)


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
        _pass_syms = [to_ccxt(s) for s in universe]
        try:
            _pass_syms += [to_ccxt(p['symbol'])
                           for p in _load_all_positions()]
        except Exception as e:
            # 보유 심볼이 프리페치에서 빠지면 그 포지션의 SL/TP 판정이
            # 시세 없이 돌아 이번 사이클을 건너뛴다.
            log.warning(f'[시세] 보유 심볼 목록 조회 실패 — 일부 포지션 '
                        f'판정이 지연될 수 있음: {e}')
        _prefetch_prices(_pass_syms)

        # 유니버스에서 빠진 심볼도 '보유 중이면' 계속 순회한다.
        # SL/TP 판정·청산·보호주문 재등록이 전부 이 루프에서만 일어나므로
        # (_manage_position 호출부는 아래 한 곳뿐), 4시간마다 도는 유니버스
        # 갱신으로 심볼이 빠지면 그 포지션이 통째로 방치된다 —
        # 소프트웨어 SL조차 돌지 않아 거래소 주문이 없으면 완전 무방비가 된다.
        # 이 심볼들은 관리 전용이며, 신규 진입은 유니버스 안에서만 한다.
        _held: list[str] = []
        try:
            _held = [p['symbol'] for p in _load_all_positions()
                     if p['strategy'] in strategies]
        except Exception as e:
            # 보유 목록이 비면 관리 대상에서 빠져 **이번 사이클 청산 판정이
            # 통째로 누락**된다. 유니버스 밖으로 나간 심볼이 특히 위험하다.
            log.error(f'[관리] 보유 포지션 목록 조회 실패 — 이번 사이클 '
                      f'청산 판정 누락 위험: {e}')
        _uni_set  = set(universe)
        scan_syms = list(universe) + [s for s in dict.fromkeys(_held)
                                      if s not in _uni_set]

        for symbol in scan_syms:
            manage_only = symbol not in _uni_set
            ccxt_sym = to_ccxt(symbol)
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

                    # 지표 판정은 **완성된 마지막 봉**으로 한다.
                    # 거래소가 돌려주는 마지막 봉(i)은 형성 중이라 BB·EMA가
                    # 폴링마다 흔들린다. 그 값으로 청산하면
                    #   ① 봉 마감 시점의 값과 달라 백테스트가 검증한 적 없는
                    #      동작이 되고(신호는 이미 i-1을 쓰므로 내부 불일치),
                    #   ② 봉 중간에 잠깐 뒤집힌 크로스에 휩쓸려 조기 청산된다.
                    # 가격 비교(SL/TP)는 여전히 실시간 price로 한다.
                    i_closed  = max(0, len(df) - 2)

                    # 포지션 관리 (매 폴링마다)
                    _manage_position(strategy_id, symbol, ccxt_sym, df, i_closed)

                    # 유니버스 밖 심볼은 관리 전용 — 신규 진입 금지.
                    # (유니버스에서 탈락했다는 건 거래량·모멘텀 기준을 더는
                    #  만족하지 않는다는 뜻이므로 새로 사지 않는다)
                    if manage_only:
                        continue

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

                    if strategy_id == 'S5' and _s5_safety_block(symbol):
                        continue

                    # 모멘텀 RS Gate + 주도주 부스트
                    rs_scale = _rs_gate_scale(strategy_id, symbol)
                    if rs_scale is None:
                        continue

                    # 펀딩비 필터: 추세추종 전략 롱 과밀 구간 차단
                    funding_scale = _funding_scale(strategy_id, symbol)
                    if funding_scale is None:
                        continue

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
            # 전 전략이 막히는 레짐이 길어지면 알린다(조용한 정지 방지)
            try:
                check_no_trade_regime(get_cached_regime().regime, time.time())
            except Exception as _e:
                log.debug(f'[레짐] 정지 감시 실패(무시): {_e}')
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
        # 리셋 전 값은 그날의 최종 손익이다 — 버리면 사후에 재구성할 수 없다.
        log.info(f'[일일 리셋] 전일 손익 ${day_pnl:+,.2f} → 새 기준 자본 ${equity:,.2f}')
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
                    # 잔고가 0인 **가장 흔한 이유는 거래소 보호주문 체결**이다.
                    # 확인하지 않고 MANUAL_SOLD로 적으면 두 가지가 함께 틀어진다:
                    #   · 사유가 SL/TP가 아니라 MANUAL_SOLD로 남는다
                    #   · 체결가를 몰라 '현재가'로 추정한다(그 사이 가격이 움직인다)
                    # 이 통계는 Kelly 사이징과 전략 건강도(net PF)에 그대로 들어가므로
                    # 오염되면 배분 판단까지 흔들린다.
                    # 전략 루프가 쓰는 판정 함수를 먼저 태운다 — 체결을 찾으면
                    # 실제 체결가와 올바른 사유로 기록하고 포지션까지 정리한다.
                    # (검증 루프가 5분 주기 전략 루프보다 먼저 도는 경합에서 발생)
                    if _try_settle_via_stop_fill(strategy, sym, to_ccxt(sym), pos,
                                                 f'[검증] {sym}'):
                        continue
                    # 완전 소진 — 수동 매도로 간주, DB 포지션 삭제
                    log.warning(f'[검증] {sym} 잔고 없음 — DB 포지션 삭제')
                    _tg(f'⚠️ [{strategy}/{sym}] 잔고 0 감지 — 수동매도로 DB 정리')
                    # 고아 보호 주문 방지: 남아있으면 취소 (이미 체결/취소면 무시됨)
                    _cancel_stop_order(strategy, sym, to_ccxt(sym),
                                       pos.get('sl_order_id') or '')
                    _cancel_stop_order(strategy, sym, to_ccxt(sym),
                                       pos.get('tp_order_id') or '')
                    # 시세 조회가 죽어도 포지션은 반드시 지워져야 하므로 아래
                    # try 밖에서 먼저 삭제한다 (헬퍼 내부 삭제는 무해한 no-op).
                    _delete_position(strategy, sym)
                    # 거래 기록 — _spot_sell의 수동매도 경로와 동일하게 통계 보존
                    # (체결가 불명이므로 현재가로 추정, 조회 불가 시 진입가)
                    try:
                        est_price = _get_price(to_ccxt(sym))
                        entry_price = float(pos['entry_price'])
                        if est_price <= 0:
                            est_price = entry_price
                        _settle_closed_position(
                            strategy, sym, entry_price=entry_price,
                            exit_price=est_price, qty=db_qty,
                            cost_usdt=float(pos['cost_usdt']),
                            reason='MANUAL_SOLD', regime=pos.get('regime', ''),
                            entry_ts=pos['entry_ts'],
                            # R배수 분모는 진입 시점의 위험 — 추적손절로 sl이
                            # 올라간 뒤 그 값을 쓰면 R이 부푼다 (다른 4개
                            # 정산 경로와 동일한 기준. 레거시 행은 sl 폴백).
                            sl_for_r=(float(pos.get('orig_sl') or 0)
                                      or float(pos['sl'])),
                            round_hold=True)
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
            price = _get_price(to_ccxt(p['symbol']))
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
    _requested = [s.strip().upper() for s in args.strategies.split(',') if s.strip()]
    _valid, _problems = validate_active_strategies(_requested)
    if _problems:
        for p in _problems:
            log.error(f'[기동] 실행 불가 전략 — {p}')
        _tg('⚠️ 실행되지 않는 전략이 지정됐습니다:\n'
            + '\n'.join(f'   • {p}' for p in _problems)
            + f'\n실제 가동: {_valid or "없음"}')
    if not _valid:
        # 전략이 하나도 없으면 봇은 살아만 있고 아무 거래도 하지 않는다.
        # "정상 기동"으로 보고하면 운영자가 몇 주를 그대로 흘려보낸다.
        log.error(f'[기동] 실행 가능한 전략이 없습니다 (요청: {_requested}). '
                  f'선택 가능: {sorted(LIVE_STRATEGIES)}')
        _tg(f'🚨 실행 가능한 전략이 없어 기동을 중단합니다.\n'
            f'   요청: {_requested}\n   선택 가능: {sorted(LIVE_STRATEGIES)}')
        _tg_flush()
        return
    _state['active_strategies'] = _valid

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
    if not _state['dry_run']:
        _detect_fee_rate()      # 실제 요율 반영 + BNB 할인 미적용 시 안내
        _report_sizing_capability(total)   # 진입 불가 조합 사전 경고
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
    threads: list = []

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
            _report_dead_threads(threads)
            _report_regime_idle()
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

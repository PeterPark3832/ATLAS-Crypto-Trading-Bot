"""
╔══════════════════════════════════════════════════════════════════╗
║   ATLAS — 통합 트레이딩 봇  (v1.1 — 코드리뷰 반영)               ║
║   Phoenix 4H (BTC/SOL/BNB) + Equinox 1D (ETH) + Bounce 1D (ETH) ║
╠══════════════════════════════════════════════════════════════════╣
║  [v1.1 수정사항]                                                  ║
║   1. CCXT 스레드 안전: _get_ex() thread-local 패턴               ║
║      → 스레드별 독립 인스턴스, 마켓 데이터 공유로 load_markets 1회 ║
║   2. atlas_regime.py __import__('pandas') → import pandas as pd  ║
║   3. 수량 정밀도: round() → ex.amount_to_precision()             ║
║   4. 지정가 폴백: cancel 후 fetch_order 재확인 → 이중체결 방지    ║
║   5. 쿨다운: (strategy,sym) → sym 레벨 통합, M1/M2 모두 감소     ║
║   6. PHX 진입 전 M1+M2 전체 포지션 체크                          ║
║   7. manage_eq_position 이중 DB 쓰기 제거                        ║
║   8. 텔레그램 ReadTimeout/ConnectionError 명시적 처리            ║
║   9. WF 최적화 데이터 300→500봉 (통계적 유의성 확보)              ║
║  10. check_phx_signal_and_enter 인라인 pandas import 제거         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  1. Imports
# ══════════════════════════════════════════════════════════════
import ccxt
import json
import math
import os
import sqlite3
import sys
import threading
import time
import logging
import logging.handlers
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atlas_config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, TG_TOKEN, TG_CHAT_ID,
    BASE_DIR, DB_FILE, LOG_DIR, KILL_SWITCH_FILE,
    PHX_SYMBOLS, EQ_SYMBOLS, BN_SYMBOLS, ALL_SYMBOLS,
    MAX_OPEN_POSITIONS, PORTFOLIO_VAR_CAP, DAILY_LOSS_LIMIT,
    CORR_SCALE_FLOOR, VOL_PARITY_FLOOR, VOL_PARITY_CAP,
    PHX_SYMBOL_PARAMS, PHX_BLOCK_WEEKEND,
    PHX_FUNDING_LONG_MAX, PHX_FUNDING_SHORT_MIN,
    PHX_DAILY_TREND_FILTER, PHX_DAILY_TREND_BUF,
    PHX_BEP_ENABLED, PHX_BEP_TRIGGER_RR, PHX_COOLDOWN_CANDLES,
    EQ_EMA_FAST, EQ_EMA_SLOW, EQ_RISK_PCT, EQ_LEVERAGE,
    EQ_EMA_FAST_CANDIDATES, EQ_EMA_SLOW_CANDIDATES,
    EQ_WF_OOS_MIN_SHARPE, EQ_WF_OOS_IMPROVE, EQ_MTF_FILTER,
    BN_RISK_PCT, BN_LEVERAGE,
    KELLY_MIN_TRADES, KELLY_SCALE_MIN, KELLY_SCALE_MAX,
    RATCHET_DD_THRESH, RATCHET_DD_HARD, RATCHET_RECOVER_PCT,
    LIMIT_OFFSET_PCT, LIMIT_TIMEOUT_SEC,
    TWAP_RISK_THRESHOLD, TWAP_SPLITS, TWAP_INTERVAL_SEC,
    CANDLE_CACHE_TTL, PRICE_POLL_SEC,
    CANDLE_LIMIT_4H, CANDLE_LIMIT_1D, WF_CANDLE_LIMIT,
)
from atlas_indicators import (
    calc_4h, get_signal_4h,
    calc_1d_eq, get_signal_eq,
    calc_1d_bn, get_signal_bn,
)
from atlas_regime import (
    regime_loop, update_regime, get_cached_regime,
    is_phoenix_long_allowed, is_phoenix_short_allowed,
    is_equinox_long_allowed, is_bounce_allowed,
    is_crisis, get_risk_scale,
)
from atlas_recovery import recovery_loop


# ══════════════════════════════════════════════════════════════
#  2. 로깅
# ══════════════════════════════════════════════════════════════
LOG_DIR.mkdir(exist_ok=True)
_logger = logging.getLogger('atlas')
_logger.setLevel(logging.DEBUG)

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'atlas.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(threadName)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
_logger.addHandler(_fh)
_logger.addHandler(_ch)

def log(msg: str, level: str = 'info'):
    getattr(_logger, level, _logger.info)(msg)


# ══════════════════════════════════════════════════════════════
#  3. CCXT 스레드 안전 인스턴스 관리
#     Fix 1: 단일 공유 인스턴스 → thread-local 패턴
#     - 각 스레드가 자신만의 CCXT 인스턴스를 보유 → Nonce 충돌 제거
#     - markets 데이터는 공유하여 load_markets() API 호출을 최초 1회로 제한
# ══════════════════════════════════════════════════════════════
_tl            = threading.local()          # thread-local storage
_markets_lock  = threading.Lock()
_shared_markets: dict | None = None


def _get_ex() -> ccxt.binance:
    """
    현재 스레드 전용 CCXT 인스턴스를 반환한다.
    최초 접근 시 인스턴스를 생성하고, markets 데이터는 공유 캐시에서 주입한다.
    """
    global _shared_markets
    if not hasattr(_tl, 'ex'):
        ex = ccxt.binance({
            'apiKey':  BINANCE_API_KEY,
            'secret':  BINANCE_API_SECRET,
            'options': {'defaultType': 'future'},
        })
        with _markets_lock:
            if _shared_markets is None:
                ex.load_markets()
                _shared_markets = ex.markets
                log('[Exchange] 마켓 데이터 로드 완료 (최초)')
            else:
                ex.markets = _shared_markets   # 기존 캐시 재사용
        _tl.ex = ex
    return _tl.ex


# ══════════════════════════════════════════════════════════════
#  4. DB 초기화 (SQLite WAL)
# ══════════════════════════════════════════════════════════════
_db_lock = threading.Lock()

def init_db():
    DB_FILE.parent.mkdir(exist_ok=True)
    with _db_lock:
        con = sqlite3.connect(DB_FILE)
        con.execute('PRAGMA journal_mode=WAL')
        con.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl          REAL NOT NULL,
                tp          REAL NOT NULL,
                qty         REAL NOT NULL,
                leverage    INTEGER NOT NULL,
                risk_usd    REAL DEFAULT 0,
                peak        REAL DEFAULT 0,
                sl_order_id TEXT DEFAULT '',
                entry_ts    TEXT NOT NULL,
                mode        TEXT DEFAULT '',
                rr          REAL DEFAULT 0,
                UNIQUE(strategy, symbol)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT, symbol TEXT, direction TEXT,
                mode        TEXT, entry_price REAL, exit_price REAL,
                qty         REAL, leverage INTEGER,
                pnl_usd     REAL, pnl_r REAL, rr REAL,
                hold_hours  REAL, exit_reason TEXT,
                entry_ts    TEXT, exit_ts TEXT, regime TEXT
            );
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY, value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
        """)
        con.commit()
        con.close()
    log('[DB] 초기화 완료')


def _db_execute(sql: str, params=()):
    with _db_lock:
        con = sqlite3.connect(DB_FILE)
        con.execute('PRAGMA journal_mode=WAL')
        con.execute(sql, params)
        con.commit()
        con.close()


def _db_fetchall(sql: str, params=()) -> list:
    con = sqlite3.connect(DB_FILE)
    con.execute('PRAGMA journal_mode=WAL')
    rows = con.execute(sql, params).fetchall()
    con.close()
    return rows


def _db_fetchone(sql: str, params=()):
    con = sqlite3.connect(DB_FILE)
    con.execute('PRAGMA journal_mode=WAL')
    row = con.execute(sql, params).fetchone()
    con.close()
    return row


# ── 포지션 CRUD ───────────────────────────────────────────────
_POS_COLS = ['id','strategy','symbol','direction','entry_price','sl','tp',
             'qty','leverage','risk_usd','peak','sl_order_id','entry_ts','mode','rr']

def save_position(strategy: str, sym: str, direction: str,
                  entry: float, sl: float, tp: float, qty: float,
                  leverage: int, risk_usd: float,
                  mode: str = '', rr: float = 0.0, sl_order_id: str = ''):
    ts = datetime.now(timezone.utc).isoformat()
    _db_execute("""
        INSERT OR REPLACE INTO positions
          (strategy,symbol,direction,entry_price,sl,tp,qty,leverage,
           risk_usd,peak,sl_order_id,entry_ts,mode,rr)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (strategy, sym, direction, entry, sl, tp, qty, leverage,
          risk_usd, entry, sl_order_id, ts, mode, rr))


def load_position(strategy: str, sym: str) -> dict | None:
    row = _db_fetchone('SELECT * FROM positions WHERE strategy=? AND symbol=?', (strategy, sym))
    return dict(zip(_POS_COLS, row)) if row else None


def load_all_positions() -> list[dict]:
    return [dict(zip(_POS_COLS, r)) for r in _db_fetchall('SELECT * FROM positions')]


def update_position_peak(strategy: str, sym: str, peak: float):
    _db_execute('UPDATE positions SET peak=? WHERE strategy=? AND symbol=?', (peak, strategy, sym))


def update_position_sl(strategy: str, sym: str, sl: float, sl_order_id: str = ''):
    _db_execute('UPDATE positions SET sl=?,sl_order_id=? WHERE strategy=? AND symbol=?',
                (sl, sl_order_id, strategy, sym))


def delete_position(strategy: str, sym: str):
    _db_execute('DELETE FROM positions WHERE strategy=? AND symbol=?', (strategy, sym))


def log_trade(strategy: str, sym: str, direction: str, mode: str,
              entry: float, exit_price: float, qty: float, leverage: int,
              pnl_usd: float, pnl_r: float, rr: float,
              hold_hours: float, exit_reason: str, entry_ts: str, regime: str):
    _db_execute("""
        INSERT INTO trades
          (strategy,symbol,direction,mode,entry_price,exit_price,qty,leverage,
           pnl_usd,pnl_r,rr,hold_hours,exit_reason,entry_ts,exit_ts,regime)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (strategy, sym, direction, mode, entry, exit_price, qty, leverage,
          pnl_usd, pnl_r, rr, hold_hours, exit_reason, entry_ts,
          datetime.now(timezone.utc).isoformat(), regime))


def get_config(key: str, default: str = '') -> str:
    row = _db_fetchone('SELECT value FROM config WHERE key=?', (key,))
    return row[0] if row else default


def set_config(key: str, value: str):
    _db_execute('INSERT OR REPLACE INTO config(key,value) VALUES(?,?)', (key, value))


# ══════════════════════════════════════════════════════════════
#  5. CandleCache — Fix 1 반영: ex 파라미터 제거, _get_ex() 사용
# ══════════════════════════════════════════════════════════════
class CandleCache:
    def __init__(self, ttl: int = CANDLE_CACHE_TTL):
        self._ttl   = ttl
        self._cache = {}
        self._locks = {}
        self._meta  = threading.Lock()

    def _key_lock(self, key: tuple) -> threading.Lock:
        with self._meta:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, ex, ccxt_sym: str, tf: str, limit: int) -> list:
        """
        ex: 호출 스레드의 전용 CCXT 인스턴스 (_get_ex() 결과).
            atlas_regime.py 등 외부 모듈도 동일 시그니처로 호출 가능.
        key_lock으로 동일 키는 한 번만 API 호출 (thundering herd 방지).
        """
        key = (ccxt_sym, tf)
        with self._key_lock(key):
            cached, ts = self._cache.get(key, (None, 0))
            if cached and time.time() - ts < self._ttl:
                return cached
            for attempt in range(3):
                try:
                    data = ex.fetch_ohlcv(ccxt_sym, tf, limit=limit)
                    self._cache[key] = (data, time.time())
                    return data
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
                    log(f'[Cache] {ccxt_sym} {tf} 재시도 {attempt+1}: {e}', 'warning')
            return cached or []


# ══════════════════════════════════════════════════════════════
#  6. Shared 상태
# ══════════════════════════════════════════════════════════════
_shared: dict = {
    'equity':       10000.0,
    'day_eq_start': 10000.0,
    'paused':       False,
    # Fix 5: 쿨다운 키를 (strategy,sym) → sym 레벨로 통합
    # 형식: {'BTCUSDT': 캔들수, ...}  (Phoenix 심볼 전용)
    'phx_cooldowns': {},
    'eq_params':    {'fast': EQ_EMA_FAST, 'slow': EQ_EMA_SLOW},
    'corr_matrix':  None,
    'vol_cache':    {},
    'corr_updated': 0.0,
    'ratchet':      {'peak_equity': 10000.0},
}
_shared_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════
#  7. 잔고 폴러
# ══════════════════════════════════════════════════════════════
def balance_poller(interval: int = 60):
    while True:
        try:
            ex  = _get_ex()
            bal = ex.fetch_balance()
            eq  = float(bal.get('USDT', {}).get('total', 0) or
                        bal.get('total', {}).get('USDT', 0))
            if eq > 0:
                with _shared_lock:
                    _shared['equity'] = eq
                    pnl_pct = (eq - _shared['day_eq_start']) / max(_shared['day_eq_start'], 1)
                    if pnl_pct <= DAILY_LOSS_LIMIT and not _shared['paused']:
                        _shared['paused'] = True
                        tg(f'🚨 [ATLAS] 일일 손실한도 도달 ({pnl_pct*100:.1f}%). 봇 일시정지.')
                        log(f'[Risk] 일일 손실한도 {pnl_pct*100:.1f}%', 'warning')
        except Exception as e:
            log(f'[Balance] 갱신 실패: {e}', 'warning')
        time.sleep(interval)


# ══════════════════════════════════════════════════════════════
#  8. Telegram
#     Fix 7: ReadTimeout / ConnectionError 명시적 처리
# ══════════════════════════════════════════════════════════════
_tg_session = requests.Session()

def tg(msg: str):
    # 💡 마크다운 오류 방지: 언더바(_)를 하이픈(-)으로 치환
    safe_msg = msg.replace('_', '-')
    
    try:
        _tg_session.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': safe_msg,
                  'parse_mode': 'Markdown', 'disable_web_page_preview': True},
            timeout=10)
    except (requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError) as e:
        log(f'[TG] 전송 타임아웃/연결오류: {e}', 'warning')
    except Exception as e:
        log(f'[TG] 전송 실패: {e}', 'warning')


# ══════════════════════════════════════════════════════════════
#  9. 리스크 헬퍼
# ══════════════════════════════════════════════════════════════

def get_kelly_scale(strategy: str, base_risk: float) -> tuple[float, str]:
    rows = _db_fetchall(
        'SELECT pnl_r FROM trades WHERE strategy=? ORDER BY id DESC LIMIT 200', (strategy,))
    if len(rows) < KELLY_MIN_TRADES:
        return base_risk, f'Kelly: 데이터 부족 ({len(rows)}건)'
    returns  = [r[0] for r in rows]
    wins     = [r for r in returns if r > 0]
    losses   = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns)
    avg_win  = sum(wins)  / len(wins)   if wins   else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1
    b = avg_win / avg_loss if avg_loss else 1
    kelly = max(KELLY_SCALE_MIN, min(KELLY_SCALE_MAX,
                (win_rate - (1 - win_rate) / b) / 2))
    return base_risk * kelly, f'Kelly×{kelly:.2f} (WR={win_rate*100:.0f}% b={b:.2f})'


def get_ratchet_scale() -> float:
    with _shared_lock:
        eq   = _shared['equity']
        peak = _shared['ratchet'].get('peak_equity', eq)
    if eq > peak:
        with _shared_lock:
            _shared['ratchet']['peak_equity'] = eq
        peak = eq
    dd = (peak - eq) / peak if peak else 0
    if dd >= RATCHET_DD_HARD:
        return 0.40
    if dd >= RATCHET_DD_THRESH:
        trough   = peak * (1 - dd)
        recovery = (eq - trough) / max(peak - trough, 1e-9)
        steps    = int(recovery / RATCHET_RECOVER_PCT)
        return min(1.0, 0.70 + steps * 0.10)
    return 1.0


def check_portfolio_var() -> tuple[bool, str]:
    with _shared_lock:
        eq = _shared['equity']
    total_risk = sum(p.get('risk_usd', 0) for p in load_all_positions())
    var_pct = total_risk / eq if eq else 0
    if var_pct >= PORTFOLIO_VAR_CAP:
        return False, f'VaR {var_pct*100:.1f}% ≥ {PORTFOLIO_VAR_CAP*100:.0f}%'
    return True, ''


def count_open_positions() -> int:
    row = _db_fetchone('SELECT COUNT(*) FROM positions')
    return row[0] if row else 0


def get_corr_scale(sym: str, direction: str) -> float:
    with _shared_lock:
        matrix = _shared.get('corr_matrix')
    positions = load_all_positions()
    if matrix is None:
        same_dir = sum(1 for p in positions if p['direction'] == direction)
        return {1: 1.00, 2: 0.80, 3: 0.65, 4: 0.55}.get(same_dir + 1, 0.50)
    sym_base      = sym.replace('USDT', '')
    same_dir_syms = [p['symbol'].replace('USDT','') for p in positions
                     if p['direction'] == direction]
    max_corr = max((matrix.get(tuple(sorted([sym_base, s])), 0.0)
                    for s in same_dir_syms), default=0.0)
    return max(CORR_SCALE_FLOOR, 1.0 - max_corr * (1 - CORR_SCALE_FLOOR))


def get_vol_parity_scale(sym: str) -> float:
    with _shared_lock:
        vol_cache = _shared.get('vol_cache', {})
    btc_vol = vol_cache.get('BTCUSDT', 0)
    sym_vol  = vol_cache.get(sym, 0)
    if btc_vol <= 0 or sym_vol <= 0:
        return 1.0
    return max(VOL_PARITY_FLOOR, min(VOL_PARITY_CAP, btc_vol / sym_vol))


# ══════════════════════════════════════════════════════════════
#  10. 거래소 헬퍼 (ex 파라미터 제거 → _get_ex() 사용)
# ══════════════════════════════════════════════════════════════
_funding_cache: dict = {}
_FUNDING_TTL = 900

def get_funding_rate(ccxt_sym: str) -> float:
    now    = time.time()
    cached = _funding_cache.get(ccxt_sym)
    if cached and now - cached[1] < _FUNDING_TTL:
        return cached[0]
    try:
        data = _get_ex().fetch_funding_rate(ccxt_sym)
        rate = float(data.get('fundingRate', 0) or 0)
        _funding_cache[ccxt_sym] = (rate, now)
        return rate
    except Exception:
        return 0.0


def get_price(ccxt_sym: str) -> float:
    try:
        return float(_get_ex().fetch_ticker(ccxt_sym).get('last', 0) or 0)
    except Exception:
        return 0.0


def get_qty_precision(ccxt_sym: str) -> tuple[int, float]:
    try:
        ex     = _get_ex()
        market = ex.market(ccxt_sym)
        prec   = market.get('precision', {})
        limits = market.get('limits', {})
        qty_prec = int(prec.get('amount', 3))
        min_qty  = float((limits.get('amount') or {}).get('min', 0.001) or 0.001)
        return qty_prec, min_qty
    except Exception:
        return 3, 0.001


def set_leverage(ccxt_sym: str, leverage: int):
    try:
        _get_ex().set_leverage(leverage, ccxt_sym)
    except Exception as e:
        log(f'[Leverage] {ccxt_sym} {leverage}x 설정 실패: {e}', 'warning')


# ══════════════════════════════════════════════════════════════
#  11. 주문 실행
#      Fix 3: amount_to_precision 적용
#      Fix 4: 지정가 폴백 이중체결 방지
# ══════════════════════════════════════════════════════════════

def _wait_fill(ccxt_sym: str, side: str, timeout: int = 20) -> float:
    expected = 'long' if side == 'buy' else 'short'
    ex = _get_ex()
    for _ in range(timeout // 2):
        time.sleep(2)
        try:
            pos = ex.fetch_position(ccxt_sym)
            sz  = abs(float(pos.get('contracts', 0) or 0))
            s   = (pos.get('side') or '').lower()
            if sz > 0 and s == expected:
                return float(pos.get('entryPrice', 0) or 0)
        except Exception:
            pass
    return 0.0


def market_order(ccxt_sym: str, side: str, qty: float) -> tuple[bool, float, str]:
    try:
        _get_ex().create_market_order(ccxt_sym, side, qty)
        fill = _wait_fill(ccxt_sym, side)
        return True, fill, ''
    except Exception as e:
        return False, 0, str(e)


def limit_or_market(ccxt_sym: str, side: str, qty: float,
                    signal_price: float) -> tuple[bool, float, str]:
    """
    지정가 우선 → 미체결 시 취소 후 시장가 폴백.
    Fix 4: cancel 후 fetch_order로 실제 filled 재확인 → 이중체결 방지
    """
    ex = _get_ex()
    limit_px = round(signal_price * (1 - LIMIT_OFFSET_PCT if side == 'buy'
                                     else 1 + LIMIT_OFFSET_PCT), 2)
    try:
        order = ex.create_limit_order(ccxt_sym, side, qty, limit_px)
        oid   = order.get('id', '')
    except Exception as e:
        log(f'[Exec] 지정가 생성 실패 → 시장가: {e}', 'warning')
        return market_order(ccxt_sym, side, qty)

    deadline = time.time() + LIMIT_TIMEOUT_SEC
    while time.time() < deadline:
        time.sleep(5)
        try:
            o  = ex.fetch_order(oid, ccxt_sym)
            st = o.get('status', '')
            if st == 'closed':
                return True, float(o.get('average', 0) or limit_px), ''
            if st == 'canceled':
                break
        except Exception:
            pass

    # 취소 요청
    try:
        ex.cancel_order(oid, ccxt_sym)
    except Exception:
        pass

    # Fix 4: 취소 후 최종 상태 재확인 (완전 체결 엣지케이스 방어)
    try:
        o       = ex.fetch_order(oid, ccxt_sym)
        filled  = float(o.get('filled', 0) or 0)
        status  = o.get('status', '')
        # 취소 직전에 완전 체결된 경우 → 시장가 주문 불필요
        if status == 'closed' and filled >= qty * 0.99:
            return True, float(o.get('average', 0) or signal_price), '지정가 완전체결(지연감지)'
        # 부분 체결 → 나머지만 시장가
        if filled > 0:
            avg_px = float(o.get('average', 0) or limit_px)
            rem_raw = qty - filled
            rem     = float(_get_ex().amount_to_precision(ccxt_sym, rem_raw)) if rem_raw > 0 else 0
            if rem > 0:
                ok, mkt_px, _ = market_order(ccxt_sym, side, rem)
                if ok and mkt_px > 0:
                    # 가중평균 체결가
                    blended = (avg_px * filled + mkt_px * rem) / qty
                    return True, blended, f'부분체결({filled:.4f})+시장가({rem:.4f})'
            return True, avg_px, f'부분체결({filled:.4f})만 처리됨'
    except Exception as e:
        log(f'[Exec] 취소 후 상태 확인 실패: {e}', 'warning')

    return market_order(ccxt_sym, side, qty)


def twap_order(ccxt_sym: str, side: str, total_qty: float,
               qty_prec: int) -> tuple[bool, float, str]:
    ex       = _get_ex()
    split    = float(ex.amount_to_precision(ccxt_sym, total_qty / TWAP_SPLITS))
    fills    = []
    for i in range(TWAP_SPLITS):
        ok, px, _ = market_order(ccxt_sym, side, split)
        if ok and px > 0:
            fills.append(px)
        if i < TWAP_SPLITS - 1:
            time.sleep(TWAP_INTERVAL_SEC)
    if not fills:
        return False, 0, 'TWAP 전체 실패'
    return True, sum(fills) / len(fills), f'TWAP {len(fills)}/{TWAP_SPLITS}'


def place_exchange_sl(ccxt_sym: str, side: str, qty: float, sl_price: float) -> str:
    close_side = 'sell' if side == 'long' else 'buy'
    try:
        order = _get_ex().create_order(
            ccxt_sym, 'stop_market', close_side, qty,
            params={'stopPrice': sl_price, 'closePosition': True, 'reduceOnly': True})
        return order.get('id', '')
    except Exception as e:
        log(f'[SL Order] {ccxt_sym} 등록 실패: {e}', 'warning')
        return ''


def cancel_exchange_sl(ccxt_sym: str, sl_order_id: str):
    if not sl_order_id:
        return
    try:
        _get_ex().cancel_order(sl_order_id, ccxt_sym)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  12. 진입 / 청산 공통
#      Fix 3: amount_to_precision 적용 (round 제거)
# ══════════════════════════════════════════════════════════════

def calc_raw_qty(equity: float, entry: float, sl: float,
                 risk_pct: float) -> tuple[float, float]:
    """SL 거리 기반 원시 수량(미정제) 반환. 호출자가 amount_to_precision 적용."""
    sl_dist  = abs(entry - sl)
    if sl_dist <= 0 or entry <= 0:
        return 0.0, 0.0
    risk_usd = equity * risk_pct
    return risk_usd / sl_dist, risk_usd


def enter_position(ccxt_sym: str, strategy: str, sym: str,
                   direction: str, mode: str, rr: float,
                   entry_sig: float, sl: float, tp: float,
                   leverage: int, risk_pct: float) -> bool:
    """포지션 진입 공통 함수."""
    ex = _get_ex()

    with _shared_lock:
        equity = _shared['equity']
        paused = _shared['paused']
    if paused:
        return False
    if is_crisis():
        return False
    if count_open_positions() >= MAX_OPEN_POSITIONS:
        log(f'[진입차단] {sym}: 최대 포지션 수 도달')
        return False

    var_ok, var_reason = check_portfolio_var()
    if not var_ok:
        log(f'[진입차단] {sym}: {var_reason}')
        return False

    # 주말 차단 (Phoenix)
    if strategy.startswith('PHX') and PHX_BLOCK_WEEKEND:
        if datetime.now(timezone.utc).weekday() >= 5:
            return False

    # 펀딩비 필터
    fr = get_funding_rate(ccxt_sym)
    if direction == 'LONG'  and fr >  PHX_FUNDING_LONG_MAX:
        log(f'[진입차단] {sym}: 펀딩비 {fr:.4%} LONG 차단')
        return False
    if direction == 'SHORT' and fr <  PHX_FUNDING_SHORT_MIN:
        log(f'[진입차단] {sym}: 펀딩비 {fr:.4%} SHORT 차단')
        return False

    # Fix 5: 쿨다운 — sym 레벨 통합
    if strategy.startswith('PHX'):
        with _shared_lock:
            cd = _shared['phx_cooldowns'].get(sym, 0)
        if cd > 0:
            log(f'[진입차단] {sym}: 쿨다운 {cd}캔들 남음')
            return False

    # 리스크 스케일링
    regime_scale  = get_risk_scale()
    kelly_risk, kelly_note = get_kelly_scale(strategy, risk_pct)
    ratchet_scale = get_ratchet_scale()
    corr_scale    = get_corr_scale(sym, direction)
    vol_scale     = get_vol_parity_scale(sym)
    final_risk    = kelly_risk * regime_scale * ratchet_scale * corr_scale * vol_scale
    final_risk    = max(0.001, min(0.05, final_risk))

    # Fix 3: amount_to_precision으로 수량 정밀도 처리 (LOT_SIZE 에러 방지)
    qty_raw, risk_usd = calc_raw_qty(equity, entry_sig, sl, final_risk)
    if qty_raw <= 0:
        return False
    try:
        qty = float(ex.amount_to_precision(ccxt_sym, qty_raw))
    except Exception:
        qty_prec, _ = get_qty_precision(ccxt_sym)
        qty = round(qty_raw, qty_prec)

    _, min_qty = get_qty_precision(ccxt_sym)
    if qty < min_qty:
        log(f'[진입차단] {sym}: 수량 {qty} < 최소 {min_qty}')
        return False

    set_leverage(ccxt_sym, leverage)

    side = 'buy' if direction == 'LONG' else 'sell'
    if final_risk >= TWAP_RISK_THRESHOLD:
        qty_prec, _ = get_qty_precision(ccxt_sym)
        ok, fill, note = twap_order(ccxt_sym, side, qty, qty_prec)
    else:
        ok, fill, note = limit_or_market(ccxt_sym, side, qty, entry_sig)

    if not ok or fill <= 0:
        log(f'[진입실패] {sym} {strategy}: {note}', 'error')
        return False

    # fill 기준 SL/TP 재계산 (슬리피지 흡수)
    if direction == 'LONG':
        sl_dist  = abs(entry_sig - sl)
        sl_final = fill - sl_dist
        tp_final = fill + sl_dist * rr
    else:
        sl_dist  = abs(sl - entry_sig)
        sl_final = fill + sl_dist
        tp_final = fill - sl_dist * rr

    pos_side    = 'long' if direction == 'LONG' else 'short'
    sl_order_id = place_exchange_sl(ccxt_sym, pos_side, qty, sl_final)
    save_position(strategy, sym, direction, fill, sl_final, tp_final,
                  qty, leverage, risk_usd, mode, rr, sl_order_id)

    regime_name = get_cached_regime().regime
    tg(f"✅ [{strategy}] {sym} {direction} 진입\n"
       f"  모드:{mode} RR:{rr:.2f} {leverage}x\n"
       f"  진입:{fill:,.4f} SL:{sl_final:,.4f} TP:{tp_final:,.4f}\n"
       f"  리스크:${risk_usd:.1f}({final_risk*100:.2f}%) {regime_name}\n"
       f"  {kelly_note}")
    return True


def exit_position(ccxt_sym: str, pos: dict, exit_price: float, reason: str) -> bool:
    strategy  = pos['strategy']
    sym       = pos['symbol']
    direction = pos['direction']
    qty       = pos['qty']
    entry     = pos['entry_price']
    entry_ts  = pos['entry_ts']
    leverage  = pos['leverage']

    cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))

    side = 'sell' if direction == 'LONG' else 'buy'
    try:
        _get_ex().create_market_order(ccxt_sym, side, qty, params={'reduceOnly': True})
    except Exception as e:
        log(f'[청산실패] {sym} {strategy}: {e}', 'error')
        return False

    # 부분체결 검증: 포지션이 실제로 0이 되었는지 확인하고 잔여분 재청산
    time.sleep(2)
    try:
        ex_pos    = _get_ex().fetch_position(ccxt_sym)
        remaining = abs(float(ex_pos.get('contracts', 0) or 0))
        ex_side   = (ex_pos.get('side') or '').lower()
        expected  = 'long' if direction == 'LONG' else 'short'
        if remaining > 0 and ex_side == expected:
            log(f'[청산검증] {sym} {strategy} 잔여 {remaining} contracts → 재시도', 'warning')
            try:
                rem_prec = float(_get_ex().amount_to_precision(ccxt_sym, remaining))
                if rem_prec > 0:
                    _get_ex().create_market_order(
                        ccxt_sym, side, rem_prec, params={'reduceOnly': True})
                    log(f'[청산재시도] {sym} {strategy} {rem_prec} 청산 완료')
            except Exception as retry_e:
                log(f'[청산재시도] {sym} {strategy} 실패: {retry_e}', 'error')
                tg(f'⚠️ [{strategy}] {sym} 부분청산 잔여 {remaining} 처리 실패!\n수동 확인 필요.')
    except Exception as e:
        log(f'[청산검증] {sym} 포지션 확인 실패: {e}', 'warning')

    pnl_pct  = (exit_price - entry) / entry if direction == 'LONG' else (entry - exit_price) / entry
    pnl_usd  = pnl_pct * qty * entry
    sl_dist  = abs(entry - pos['sl'])
    pnl_r    = pnl_usd / (qty * sl_dist) if sl_dist > 0 else 0
    try:
        entry_dt = datetime.fromisoformat(entry_ts.replace('Z', '+00:00'))
        hold_h   = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
    except Exception:
        hold_h = 0

    regime_name = get_cached_regime().regime
    log_trade(strategy, sym, direction, pos.get('mode',''), entry, exit_price,
              qty, leverage, pnl_usd, pnl_r, pos.get('rr',0), hold_h, reason,
              entry_ts, regime_name)
    delete_position(strategy, sym)

    # Fix 5: 쿨다운 — sym 레벨, PHX 전 전략 적용
    if strategy.startswith('PHX'):
        with _shared_lock:
            _shared['phx_cooldowns'][sym] = PHX_COOLDOWN_CANDLES

    emoji = '✅' if pnl_usd >= 0 else '❌'
    tg(f"{emoji} [{strategy}] {sym} {direction} 청산({reason})\n"
       f"  {entry:,.4f} → {exit_price:,.4f} | ${pnl_usd:+.1f} ({pnl_pct*100:+.2f}%) {hold_h:.1f}H")
    return True


# ══════════════════════════════════════════════════════════════
#  13. Phoenix 4H 포지션 관리 + 신호
#      Fix 5,6: PHX 쿨다운 sym 레벨 + 전체 모드 포지션 체크
#      Fix 9: 인라인 pandas import 제거 (모듈 상단 pd 사용)
# ══════════════════════════════════════════════════════════════

def manage_phx_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    direction = pos['direction']
    entry     = pos['entry_price']
    sl        = pos['sl']
    tp        = pos['tp']
    peak      = pos.get('peak', entry)
    rr        = pos.get('rr', 2.0)
    strategy  = pos['strategy']

    if direction == 'LONG':
        if price <= sl:
            exit_position(ccxt_sym, pos, price, 'SL'); return
        if price >= tp:
            exit_position(ccxt_sym, pos, price, 'TP'); return
        if PHX_BEP_ENABLED:
            bep_target = entry + (tp - entry) * PHX_BEP_TRIGGER_RR / max(rr, 0.01)
            if price >= bep_target and sl < entry:
                update_position_sl(strategy, sym, entry)
                log(f'[BEP] {sym} {strategy} SL → {entry:,.4f}')
        if price > peak:
            update_position_peak(strategy, sym, price)
    else:
        if price >= sl:
            exit_position(ccxt_sym, pos, price, 'SL'); return
        if price <= tp:
            exit_position(ccxt_sym, pos, price, 'TP'); return
        if PHX_BEP_ENABLED:
            bep_target = entry - (entry - tp) * PHX_BEP_TRIGGER_RR / max(rr, 0.01)
            if price <= bep_target and sl > entry:
                update_position_sl(strategy, sym, entry)
                log(f'[BEP] {sym} {strategy} SL → {entry:,.4f}')
        if price < peak:
            update_position_peak(strategy, sym, price)


def check_phx_signal_and_enter(sym: str, df_4h: pd.DataFrame, cache: CandleCache):
    """
    Fix 6: 진입 전 M1+M2 전체 확인 → 이미 하나라도 있으면 스킵
    Fix 9: 인라인 pandas import 제거 (모듈 상단 pd 사용)
    """
    # Fix 6: 어떤 PHX 모드든 이미 포지션 있으면 스킵
    if any(load_position(m, sym) is not None for m in ('PHX_M1', 'PHX_M2')):
        return

    params = PHX_SYMBOL_PARAMS[sym]
    signal = get_signal_4h(df_4h, params)
    if signal['signal'] == 0:
        return

    direction = 'LONG' if signal['signal'] == 1 else 'SHORT'
    strategy  = f"PHX_{signal['mode']}"
    ccxt_sym  = sym.replace('USDT', '/USDT')

    if direction == 'LONG'  and not is_phoenix_long_allowed(sym):
        return
    if direction == 'SHORT' and not is_phoenix_short_allowed(sym):
        return

    # 1D EMA200 추세 필터 — Fix 9: pd 사용
    if PHX_DAILY_TREND_FILTER:
        try:
            ohlcv_1d = cache.get(_get_ex(), ccxt_sym, '1d', 210)
            if ohlcv_1d:
                c1d     = pd.DataFrame(ohlcv_1d, columns=['ts','o','h','l','c','v'])['c'].astype(float)
                ema200  = float(c1d.ewm(span=200, adjust=False).mean().iloc[-1])
                price   = float(df_4h.iloc[-1]['close'])
                buf     = ema200 * PHX_DAILY_TREND_BUF
                if direction == 'LONG'  and price < ema200 - buf:
                    log(f'[필터] {sym} PHX LONG: 1D EMA200 하회')
                    return
                if direction == 'SHORT' and price > ema200 + buf:
                    log(f'[필터] {sym} PHX SHORT: 1D EMA200 상회')
                    return
        except Exception as e:
            log(f'[필터] {sym} 1D EMA200: {e}', 'warning')

    entry_price = float(df_4h.iloc[-1]['close'])
    enter_position(ccxt_sym, strategy, sym, direction,
                   signal['mode'], signal['rr'],
                   entry_price, signal['sl'], signal['tp'],
                   params['leverage'], params['risk_pct'])


def phx_symbol_loop(sym: str, cache: CandleCache):
    """
    Phoenix 4H 심볼별 메인 루프.
    Fix 5: 쿨다운 sym 레벨로 통합, 매 루프마다 M1+M2 모두 감소
    """
    ccxt_sym  = sym.replace('USDT', '/USDT')
    ema_entry = PHX_SYMBOL_PARAMS[sym]['ema_entry']
    log(f'[Phoenix] {sym} 루프 시작')
    last_4h_ts = None

    while True:
        try:
            # Fix 5: 쿨다운 1씩 감소 (심볼 레벨 → M1/M2 공통)
            with _shared_lock:
                cd = _shared['phx_cooldowns'].get(sym, 0)
                if cd > 0:
                    _shared['phx_cooldowns'][sym] = cd - 1

            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(PRICE_POLL_SEC); continue

            ohlcv_4h  = cache.get(_get_ex(), ccxt_sym, '4h', CANDLE_LIMIT_4H)
            if not ohlcv_4h:
                time.sleep(PRICE_POLL_SEC); continue

            df_4h      = calc_4h(ohlcv_4h, ema_entry)
            cur_4h_ts  = ohlcv_4h[-1][0]

            # 포지션 감시
            for mode in ('PHX_M1', 'PHX_M2'):
                pos = load_position(mode, sym)
                if pos:
                    manage_phx_position(ccxt_sym, sym, pos, price)

            # 새 4H 캔들 → 신호 체크
            if cur_4h_ts != last_4h_ts and last_4h_ts is not None:
                check_phx_signal_and_enter(sym, df_4h, cache)
            last_4h_ts = cur_4h_ts

        except Exception as e:
            log(f'[Phoenix] {sym} 루프 오류: {e}', 'error')
        time.sleep(PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  14. Equinox + Bounce 포지션 관리 + 신호
#      Fix 7: manage_eq_position 이중 DB 쓰기 제거
# ══════════════════════════════════════════════════════════════

def manage_eq_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    """
    Fix 7: trailing SL 갱신 시 DB 쓰기 1회로 통합
    (기존: update → cancel → place → update 4단계 → cancel → place → update 3단계)
    """
    strategy  = pos['strategy']
    direction = pos['direction']
    entry     = pos['entry_price']
    sl        = pos['sl']
    tp        = pos['tp']
    peak      = pos.get('peak', entry)

    if direction != 'LONG':
        return   # Equinox/Bounce는 LONG만

    if price <= sl:
        exit_position(ccxt_sym, pos, price, 'SL'); return
    if price >= tp:
        exit_position(ccxt_sym, pos, price, 'TP'); return

    # Trailing SL
    if price > peak:
        update_position_peak(strategy, sym, price)
        sl_dist = entry - sl
        new_sl  = price - sl_dist
        if new_sl > sl:
            # Fix 7: 기존 SL 취소 → 새 SL 등록 → DB 1회 업데이트
            cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))
            new_oid = place_exchange_sl(ccxt_sym, 'long', pos['qty'], new_sl)
            update_position_sl(strategy, sym, new_sl, new_oid)
            log(f'[Trail] {sym} {strategy} SL {sl:,.4f} → {new_sl:,.4f}')


def check_eq_signal_and_enter(sym: str, ohlcv_1d: list, cache: CandleCache):
    strategy = 'EQ_LONG'
    ccxt_sym = sym.replace('USDT', '/USDT')
    if load_position(strategy, sym) is not None:
        return
    if not is_equinox_long_allowed(sym):
        return

    with _shared_lock:
        eq_params = _shared['eq_params']
    df = calc_1d_eq(ohlcv_1d, eq_params['fast'], eq_params['slow'])
    if len(df) < 5:
        return

    try:
        today      = _get_ex().fetch_ohlcv(ccxt_sym, '1d', limit=2)[-1]
        today_open = float(today[1])
        today_high = float(today[2])
        today_vol  = float(today[5])
    except Exception:
        return

    ok, ep, sl, tp, atr, reason = get_signal_eq(df, today_open, today_high, today_vol)
    if not ok:
        log(f'[EQ] {sym} 신호없음: {reason}')
        return

    # 4H MTF 필터
    if EQ_MTF_FILTER:
        try:
            ohlcv_4h = cache.get(_get_ex(), ccxt_sym, '4h', 80)
            c4  = pd.DataFrame(ohlcv_4h, columns=['ts','o','h','l','c','v'])['c'].astype(float)
            f4h = float(c4.ewm(span=20, adjust=False).mean().iloc[-1])
            s4h = float(c4.ewm(span=60, adjust=False).mean().iloc[-1])
            if f4h < s4h:
                log(f'[EQ] {sym} 4H EMA 하락추세 → 스킵')
                return
        except Exception as e:
            log(f'[EQ] {sym} 4H 필터 오류: {e}', 'warning')

    rr = (tp - ep) / max(ep - sl, 1e-9)
    enter_position(ccxt_sym, strategy, sym, 'LONG', 'EQ', rr,
                   ep, sl, tp, EQ_LEVERAGE, EQ_RISK_PCT)


def check_bn_signal_and_enter(sym: str, ohlcv_1d: list):
    strategy = 'BN_LONG'
    ccxt_sym = sym.replace('USDT', '/USDT')
    # Equinox 포지션 있으면 Bounce 중복 금지
    if load_position('EQ_LONG', sym) is not None:
        return
    if load_position(strategy, sym) is not None:
        return
    if not is_bounce_allowed(sym):
        return

    df = calc_1d_bn(ohlcv_1d)
    if len(df) < 5:
        return

    ok, ep, sl, tp, atr, reason = get_signal_bn(df)
    if not ok:
        log(f'[BN] {sym} 신호없음: {reason}')
        return

    rr = (tp - ep) / max(ep - sl, 1e-9)
    enter_position(ccxt_sym, strategy, sym, 'LONG', 'BN', rr,
                   ep, sl, tp, BN_LEVERAGE, BN_RISK_PCT)


def eq_symbol_loop(sym: str, cache: CandleCache):
    ccxt_sym   = sym.replace('USDT', '/USDT')
    log(f'[Equinox/Bounce] {sym} 루프 시작')
    last_1d_ts = None

    while True:
        try:
            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(PRICE_POLL_SEC); continue

            ohlcv_1d  = cache.get(_get_ex(), ccxt_sym, '1d', CANDLE_LIMIT_1D)
            if not ohlcv_1d:
                time.sleep(PRICE_POLL_SEC); continue

            cur_1d_ts = ohlcv_1d[-1][0]

            for strategy in ('EQ_LONG', 'BN_LONG'):
                pos = load_position(strategy, sym)
                if pos:
                    manage_eq_position(ccxt_sym, sym, pos, price)

            if cur_1d_ts != last_1d_ts and last_1d_ts is not None:
                check_eq_signal_and_enter(sym, ohlcv_1d, cache)
                check_bn_signal_and_enter(sym, ohlcv_1d)
            last_1d_ts = cur_1d_ts

        except Exception as e:
            log(f'[Equinox/Bounce] {sym} 루프 오류: {e}', 'error')
        time.sleep(PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  15. Walk-Forward EMA 최적화
#      Fix 8: WF_CANDLE_LIMIT(500) 사용 → 통계 유의성 확보
# ══════════════════════════════════════════════════════════════

def _ema_sharpe(closes: pd.Series, fast: int, slow: int) -> float:
    signal    = (closes.ewm(span=fast, adjust=False).mean() >
                 closes.ewm(span=slow, adjust=False).mean()).astype(int).shift(1).fillna(0)
    strat_ret = signal * closes.pct_change().fillna(0)
    std       = strat_ret.std()
    return float(strat_ret.mean() / std * (252 ** 0.5)) if std > 0 else 0.0


def run_wf_optimization(cache: CandleCache):
    """Fix 8: WF_CANDLE_LIMIT(500봉) 사용 — IS 75% ≈ 375일, OOS 25% ≈ 125일"""
    try:
        ohlcv = cache.get(_get_ex(), 'BTC/USDT', '1d', WF_CANDLE_LIMIT)
        if len(ohlcv) < 200:
            log('[WF] 데이터 부족 → 스킵', 'warning')
            return
        closes = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])['c'].astype(float)
        split  = int(len(closes) * 0.75)
        is_    = closes.iloc[:split]
        oos_   = closes.iloc[split:]

        best_is, best_params = -999, (EQ_EMA_FAST, EQ_EMA_SLOW)
        for fast in EQ_EMA_FAST_CANDIDATES:
            for slow in EQ_EMA_SLOW_CANDIDATES:
                if fast >= slow:
                    continue
                s = _ema_sharpe(is_, fast, slow)
                if s > best_is:
                    best_is, best_params = s, (fast, slow)

        oos_sharpe = _ema_sharpe(oos_, *best_params)
        with _shared_lock:
            cur = _shared['eq_params']
        cur_oos = _ema_sharpe(oos_, cur['fast'], cur['slow'])

        log(f'[WF] IS={best_is:.2f} OOS={oos_sharpe:.2f} 현행OOS={cur_oos:.2f} 최적={best_params}')
        if oos_sharpe >= EQ_WF_OOS_MIN_SHARPE and oos_sharpe >= cur_oos + EQ_WF_OOS_IMPROVE:
            with _shared_lock:
                _shared['eq_params'] = {'fast': best_params[0], 'slow': best_params[1]}
            set_config('eq_ema_fast', str(best_params[0]))
            set_config('eq_ema_slow', str(best_params[1]))
            tg(f'📐 [WF] EMA 갱신: {cur["fast"]}/{cur["slow"]} → {best_params[0]}/{best_params[1]}\n'
               f'  OOS Sharpe: {cur_oos:.2f} → {oos_sharpe:.2f}')
        else:
            log('[WF] 기존 파라미터 유지 (OOS 기준 미달)')
    except Exception as e:
        log(f'[WF] 오류: {e}', 'error')


def wf_optimizer_loop(cache: CandleCache):
    while True:
        now = datetime.now(timezone.utc)
        if now.weekday() == 6 and now.hour == 0 and 29 <= now.minute <= 35:
            run_wf_optimization(cache)
            time.sleep(3600)
        else:
            time.sleep(300)


# ══════════════════════════════════════════════════════════════
#  16. 상관계수 + 변동성 갱신 (1시간)
# ══════════════════════════════════════════════════════════════

def update_corr_vol(cache: CandleCache):
    try:
        closes_dict = {}
        for sym in ALL_SYMBOLS:
            ccxt_sym = sym.replace('USDT', '/USDT')
            ohlcv    = cache.get(_get_ex(), ccxt_sym, '1d', 35)
            if len(ohlcv) < 30:
                continue
            c = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])['c'].astype(float)
            closes_dict[sym] = c.pct_change().dropna().iloc[-30:]

        vol_cache = {s: float(r.std()) for s, r in closes_dict.items()}
        corr_matrix = {}
        syms = list(closes_dict.keys())
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s1, s2 = syms[i], syms[j]
                if len(closes_dict[s1]) == len(closes_dict[s2]):
                    c   = float(closes_dict[s1].corr(closes_dict[s2]))
                    key = tuple(sorted([s1.replace('USDT',''), s2.replace('USDT','')]))
                    corr_matrix[key] = max(0, c)

        with _shared_lock:
            _shared['corr_matrix']  = corr_matrix
            _shared['vol_cache']    = vol_cache
            _shared['corr_updated'] = time.time()
        log(f'[Corr/Vol] 갱신: {len(corr_matrix)}쌍')
    except Exception as e:
        log(f'[Corr/Vol] 오류: {e}', 'error')


def corr_vol_loop(cache: CandleCache):
    while True:
        update_corr_vol(cache)
        time.sleep(3600)


# ══════════════════════════════════════════════════════════════
#  17. 일일 리셋 + 브리핑
# ══════════════════════════════════════════════════════════════

def daily_reset_loop():
    last_day = None
    while True:
        now = datetime.now(timezone.utc)
        if now.date() != last_day and now.hour == 0 and now.minute < 5:
            with _shared_lock:
                eq = _shared['equity']
                _shared['day_eq_start'] = eq
                if _shared['paused']:
                    _shared['paused'] = False
                    tg('🌅 [ATLAS] 일일 손실한도 자동 해제')
            rows = _db_fetchall("""
                SELECT strategy, COUNT(*), SUM(pnl_usd),
                       SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END)
                FROM trades
                WHERE DATE(exit_ts) = DATE('now', '-1 day')
                GROUP BY strategy
            """)
            if rows:
                total = sum(r[2] or 0 for r in rows)
                lines = ['📊 *[ATLAS 일일 성과]*']
                for r in rows:
                    strat, cnt, pnl, wins = r
                    wr = (wins or 0) / cnt * 100 if cnt else 0
                    lines.append(f"  {strat}: {cnt}건 WR{wr:.0f}% ${pnl:+.1f}")
                lines.append(f"  합계:${total:+.1f}  잔고:${eq:,.0f}  {get_cached_regime().regime}")
                tg('\n'.join(lines))
            last_day = now.date()
        time.sleep(60)


# ══════════════════════════════════════════════════════════════
#  18. Telegram 명령
#      Fix 7: ReadTimeout/ConnectionError 명시적 처리
# ══════════════════════════════════════════════════════════════

def _handle_tg_cmd(text: str) -> str:
    cmd = text.strip().lower().split()[0]
    if cmd == '/status':
        positions = load_all_positions()
        if not positions:
            return '📭 현재 열린 포지션 없음'
        lines = [f'📊 *[ATLAS] 포지션 현황* ({len(positions)}개)']
        for p in positions:
            arrow = '▲' if p['direction'] == 'LONG' else '▼'
            lines.append(
                f"  {p['strategy']} {p['symbol']} {arrow}\n"
                f"    진입:{p['entry_price']:,.4f} SL:{p['sl']:,.4f} TP:{p['tp']:,.4f}")
        return '\n'.join(lines)
    if cmd == '/stats':
        rows = _db_fetchall("""
            SELECT strategy, COUNT(*), SUM(pnl_usd),
                   SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END), AVG(pnl_r)
            FROM trades GROUP BY strategy
        """)
        if not rows:
            return '📭 거래 이력 없음'
        lines = ['📈 *[ATLAS] 전략별 통계]*']
        for r in rows:
            strat, cnt, total, wins, avg_r = r
            wr = (wins or 0) / cnt * 100 if cnt else 0
            lines.append(f"  {strat}: {cnt}건 WR{wr:.0f}% ${total:+.0f} avgR={avg_r:.2f}")
        return '\n'.join(lines)
    if cmd == '/regime':
        return get_cached_regime().summary()
    if cmd == '/pause':
        with _shared_lock: _shared['paused'] = True
        return '⏸ 신규 진입 일시정지'
    if cmd == '/resume':
        with _shared_lock: _shared['paused'] = False
        return '▶️ 신규 진입 재개'
    if cmd == '/stop':
        KILL_SWITCH_FILE.touch()
        return '🛑 Kill Switch 활성화. 봇 종료 중...'
    if cmd == '/equity':
        with _shared_lock: eq = _shared['equity']
        return f'💰 잔고: ${eq:,.2f}'
    if cmd == '/help':
        return ('📖 *[ATLAS] 명령어*\n'
                '  /status /stats /regime /equity\n'
                '  /pause /resume /stop /help')
    return f'❓ 알 수 없는 명령: {cmd}'


def tg_cmd_loop():
    """
    Fix 7: ReadTimeout, ConnectionError 명시적 처리 → 스레드 생존성 확보
    """
    offset = 0
    while True:
        if KILL_SWITCH_FILE.exists():
            break
        try:
            resp = _tg_session.get(
                f'https://api.telegram.org/bot{TG_TOKEN}/getUpdates',
                params={'offset': offset, 'timeout': 30},
                timeout=35)
            for u in resp.json().get('result', []):
                offset = u['update_id'] + 1
                msg    = u.get('message', {})
                text   = msg.get('text', '')
                chat   = str(msg.get('chat', {}).get('id', ''))
                if text.startswith('/') and chat == TG_CHAT_ID:
                    tg(_handle_tg_cmd(text))
        except requests.exceptions.ReadTimeout:
            pass   # 롱폴링 타임아웃 정상 케이스 — 루프 계속
        except requests.exceptions.ConnectionError as e:
            log(f'[TG Cmd] 연결 오류: {e}', 'warning')
            time.sleep(10)
        except Exception as e:
            log(f'[TG Cmd] 오류: {e}', 'warning')
            time.sleep(10)


# ══════════════════════════════════════════════════════════════
#  19. Main
# ══════════════════════════════════════════════════════════════

def main():
    log('╔══════════════════════════════════════╗')
    log('║   ATLAS v1.1 통합 봇 시작             ║')
    log('╚══════════════════════════════════════╝')

    if KILL_SWITCH_FILE.exists():
        KILL_SWITCH_FILE.unlink()

    init_db()

    # EMA 파라미터 복원
    fast_s = get_config('eq_ema_fast')
    slow_s = get_config('eq_ema_slow')
    if fast_s and slow_s:
        with _shared_lock:
            _shared['eq_params'] = {'fast': int(fast_s), 'slow': int(slow_s)}
        log(f'[복원] EMA {fast_s}/{slow_s}')

    # 첫 번째 거래소 인스턴스 (메인 스레드 = 메인 스레드의 thread-local)
    ex = _get_ex()
    log('[Exchange] Binance Futures 연결 완료')

    # 초기 잔고
    try:
        bal = ex.fetch_balance()
        eq  = float(bal.get('USDT', {}).get('total', 0) or
                    bal.get('total', {}).get('USDT', 0))
        with _shared_lock:
            _shared['equity']                = eq
            _shared['day_eq_start']          = eq
            _shared['ratchet']['peak_equity']= eq
        log(f'[잔고] ${eq:,.2f}')
    except Exception as e:
        log(f'[초기화] 잔고 조회 실패: {e}', 'warning')

    cache = CandleCache()

    # 초기 레짐 (메인 스레드 ex 사용)
    update_regime(ex, cache)
    log(f'[초기 장세] {get_cached_regime().summary()}')
    tg(f'🚀 *ATLAS v1.1 봇 시작*\n{get_cached_regime().summary()}')

    threads = []
    def _t(target, name, *args):
        t = threading.Thread(target=target, args=args, name=name, daemon=True)
        t.start()
        threads.append(t)

    _t(balance_poller,   'BalancePoller')
    _t(regime_loop,      'RegimeLoop',    ex, cache)   # regime_loop은 ex 필요
    _t(corr_vol_loop,    'CorrVolLoop',   cache)
    _t(wf_optimizer_loop,'WFOptimizer',   cache)
    _t(daily_reset_loop, 'DailyReset')
    _t(tg_cmd_loop,      'TGCmdLoop')
    _t(recovery_loop,    'RecoveryLoop')

    for sym in PHX_SYMBOLS:
        _t(phx_symbol_loop, f'PHX_{sym}', sym, cache)
    for sym in EQ_SYMBOLS:
        _t(eq_symbol_loop, f'EQ_{sym}',   sym, cache)

    log(f'[ATLAS] 스레드 {len(threads)}개 기동')
    log(f'  Phoenix 4H : {PHX_SYMBOLS}')
    log(f'  Equinox/BN : {EQ_SYMBOLS}')

    try:
        while True:
            if KILL_SWITCH_FILE.exists():
                log('[ATLAS] Kill Switch → 종료')
                tg('🛑 *ATLAS 봇 종료*')
                break
            time.sleep(10)
    except KeyboardInterrupt:
        tg('🛑 *ATLAS 봇 종료* (수동)')


if __name__ == '__main__':
    main()

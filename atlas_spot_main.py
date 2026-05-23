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

[전략별 레짐 라우팅]
  TRENDING_UP  : S2(SMA Cross), S3(EMA Trend), S6(Donchian), S7(MACD)
  RANGING      : S4(RSI MR), S5(BB Bounce)
  WEAK_TREND   : 전체 (70% 리스크)
  TRENDING_DOWN: S4, S5만 허용 (하락추세 반등)
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
    SPOT_MAX_POSITIONS, SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT,
    SPOT_RESERVE_PCT, SPOT_DAILY_LOSS_LIMIT, SPOT_MIN_ORDER_USDT,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_RATCHET_DD_THRESH, SPOT_RATCHET_DD_HARD, SPOT_RATCHET_RECOVER,
    SPOT_CANDLE_4H, SPOT_CANDLE_1D, SPOT_CANDLE_CACHE_TTL, SPOT_PRICE_POLL_SEC,
    STRATEGY_TIMEFRAMES, STRATEGY_NAMES, REGIME_STRATEGY_MAP, WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    BT_SPOT_FEE,
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s | %(message)s',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
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


def _get_price(ccxt_sym: str) -> float:
    """현재가 조회."""
    try:
        ticker = _get_ex().fetch_ticker(ccxt_sym)
        return float(ticker['last'] or ticker['close'] or 0)
    except Exception as e:
        log.warning(f'[{ccxt_sym}] 현재가 조회 실패: {e}')
        return 0.0


# ══════════════════════════════════════════════════════════════
#  캔들 캐시
# ══════════════════════════════════════════════════════════════

class CandleCache:
    """TTL 기반 캔들 캐시 (API 부하 절감)."""

    def __init__(self, ttl: int = SPOT_CANDLE_CACHE_TTL):
        self._cache: dict = {}
        self._locks: dict = {}
        self._meta_lock = threading.Lock()
        self._ttl = ttl

    def _get_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def get(self, ex, ccxt_sym: str, timeframe: str, limit: int) -> list:
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
            fee_usdt    REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS spot_config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """)
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


def _delete_position(strategy: str, symbol: str):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            'DELETE FROM spot_positions WHERE strategy=? AND symbol=?',
            (strategy, symbol)
        )


def _log_trade(strategy: str, symbol: str, entry_price: float, exit_price: float,
               qty: float, cost: float, pnl_usdt: float, pnl_pct: float,
               pnl_r: float, hold_hours: float, reason: str, regime: str,
               entry_ts: str):
    fee = exit_price * qty * BT_SPOT_FEE
    with _db_lock, _db_conn() as conn:
        conn.execute("""
        INSERT INTO spot_trades
        (strategy, symbol, entry_price, exit_price, qty_tokens, cost_usdt,
         pnl_usdt, pnl_pct, pnl_r, hold_hours, reason, entry_ts, exit_ts, regime, fee_usdt)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (strategy, symbol, entry_price, exit_price, qty, cost,
              round(pnl_usdt, 4), round(pnl_pct, 4), round(pnl_r, 4),
              round(hold_hours, 2), reason,
              entry_ts, datetime.now(timezone.utc).isoformat(),
              regime, round(fee, 4)))


# ══════════════════════════════════════════════════════════════
#  전역 상태
# ══════════════════════════════════════════════════════════════

_state = {
    'equity':        0.0,
    'usdt_balance':  0.0,
    'peak_equity':   0.0,
    'day_pnl':       0.0,
    'day_start_eq':  0.0,
    'paused':        False,
    'universe':      [],
    'ratchet_scale': 1.0,
    'dry_run':       False,
    'active_strategies': ['S3', 'S5', 'S6', 'S7V4'],
}
_state_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════
#  Telegram
# ══════════════════════════════════════════════════════════════

def _tg(msg: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=5)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  리스크 관리
# ══════════════════════════════════════════════════════════════

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
    """Kelly 스케일 계산 (최근 거래 기반)."""
    with _db_lock, _db_conn() as conn:
        rows = conn.execute(
            'SELECT pnl_r FROM spot_trades WHERE strategy=? ORDER BY id DESC LIMIT 200',
            (strategy,)
        ).fetchall()
    if len(rows) < SPOT_KELLY_MIN_TRADES:
        return 1.0
    pnl_r = [r['pnl_r'] for r in rows]
    wins   = [r for r in pnl_r if r > 0]
    losses = [r for r in pnl_r if r <= 0]
    if not wins or not losses:
        return 1.0
    wr  = len(wins) / len(pnl_r)
    avg_w = np.mean(wins)
    avg_l = abs(np.mean(losses))
    b = avg_w / avg_l if avg_l > 0 else 1.0
    kelly = (wr - (1 - wr) / b) if b > 0 else 0.0
    return float(max(SPOT_KELLY_SCALE_MIN, min(SPOT_KELLY_SCALE_MAX, kelly)))


def _get_ratchet_scale() -> float:
    """Drawdown Ratchet 스케일."""
    equity = _state['equity']
    peak   = _state['peak_equity']
    if peak <= 0 or equity <= 0:
        return 1.0
    dd = (peak - equity) / peak
    if dd >= SPOT_RATCHET_DD_HARD:
        return 0.40
    if dd >= SPOT_RATCHET_DD_THRESH:
        return 0.70
    return 1.0


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
    """현물 매수 실행."""
    if _state['paused']:
        log.info(f'[{strategy}] {symbol} 매수 차단 (일시정지)')
        return False

    if SPOT_KILL_SWITCH.exists():
        return False

    # 포지션 수 제한
    all_pos = _load_all_positions()
    if len(all_pos) >= SPOT_MAX_POSITIONS:
        log.info(f'[{strategy}] {symbol} 매수 차단 (최대 포지션 {SPOT_MAX_POSITIONS}개)')
        return False

    # 중복 포지션 방지
    if _load_position(strategy, symbol):
        return False

    equity   = _state['equity']
    kelly    = _get_kelly_scale(strategy)
    ratchet  = _get_ratchet_scale()
    if regime == REGIME_WEAK_TREND:
        r_scale = WEAK_TREND_RISK_SCALE
    elif regime == REGIME_TRENDING_DOWN:
        r_scale = TRENDING_DOWN_RISK_SCALE
    else:
        r_scale = 1.0

    entry_price = price
    sl          = sig['sl']
    tp          = sig['tp']
    sl_dist     = abs(entry_price - sl)
    if sl_dist <= 0:
        return False

    adj_risk   = SPOT_BASE_RISK_PCT * kelly * ratchet * r_scale
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
    if _state['day_pnl'] / max(_state['day_start_eq'], 1) <= SPOT_DAILY_LOSS_LIMIT:
        log.warning(f'[{strategy}] 일간 손실 한도 초과 — 진입 차단')
        return False

    log.info(f'[{strategy}] {symbol} 매수 시도 | {qty:.6f}개 @ {entry_price:,.4f} | '
             f'비용 ${cost_usdt:.2f} | 리스크 {adj_risk*100:.2f}%')

    if _state['dry_run']:
        log.info(f'[DRY-RUN] {symbol} 매수 시뮬레이션 (실제 주문 없음)')
        fill_price = entry_price
    else:
        try:
            order = _get_ex().create_market_buy_order(ccxt_sym, qty)
            fill_price = float(order.get('average') or order.get('price') or entry_price)
            # 실제 체결량 사용 (수수료가 base asset에서 차감될 경우 요청량보다 적을 수 있음)
            filled_qty = float(order.get('filled') or qty)
            if 0 < filled_qty < qty * 0.999:
                log.info(f'[{strategy}] {symbol} qty조정: {qty:.6f} -> {filled_qty:.6f} (실체결량 기준)')
                qty = filled_qty
        except Exception as e:
            log.error(f'[{strategy}] {symbol} 매수 주문 실패: {e}')
            _tg(f'⚠️ [{strategy}] {symbol} 매수 실패: {e}')
            return False

    # SL 재계산 (실제 체결가 기준)
    sl_dist_actual = abs(fill_price - sl)
    sl_final = fill_price - sl_dist_actual if sl_dist_actual > 0 else sl
    tp_final = tp

    _save_position(
        strategy, symbol, fill_price, sl_final, tp_final,
        qty, cost_usdt, adj_risk,
        sig.get('exit_type', 'sl_tp'), sig.get('max_hold', 0), regime
    )

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
        # 매도 전 실제 잔고 확인 (수수료 차감 등으로 DB qty > 실잔고 가능 → SL 실패 원인)
        try:
            _base_asset = ccxt_sym.split('/')[0]
            _pre_bal = _get_ex().fetch_balance()
            _actual_free = float(_pre_bal['free'].get(_base_asset, 0))
            if _actual_free <= 0.0:
                _hold_h = (datetime.now(timezone.utc) - datetime.fromisoformat(entry_ts)).total_seconds() / 3600
                log.warning(f'[{strategy}] {symbol} 실잔고 0 → 수동매도로 자동처리 ({reason})')
                _tg(f'ℹ️ [{strategy}] {symbol} 잔고 없음 → 수동매도 DB정리')
                _pnl_u = (price - entry_price) * qty
                _pnl_p = (price - entry_price) / entry_price if entry_price > 0 else 0
                _sl_d = abs(entry_price - sl)
                _pnl_r = _pnl_u / (_sl_d * qty) if _sl_d > 0 else 0
                _delete_position(strategy, symbol)
                _log_trade(strategy, symbol, entry_price, price, qty, cost_usdt,
                           _pnl_u, _pnl_p, _pnl_r, round(_hold_h, 2),
                           'MANUAL_SOLD', regime, entry_ts)
                with _state_lock:
                    _state['day_pnl'] += _pnl_u
                return
            elif _actual_free < qty * 0.98:
                log.warning(f'[{strategy}] {symbol} qty조정: {qty:.6f} -> {_actual_free:.6f} (실잔고 기준)')
                qty = _actual_free
        except Exception as _be:
            log.warning(f'[{strategy}] {symbol} 사전잔고확인 실패(무시): {_be}')
        try:
            order = _get_ex().create_market_sell_order(ccxt_sym, qty)
            exit_price = float(order.get('average') or order.get('price') or price)
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
                        _log_trade(strategy, symbol, entry_price, price, qty, cost_usdt,
                                   _pnl_u, _pnl_p, _pnl_r, round(_hold_h, 2),
                                   'MANUAL_SOLD', regime, entry_ts)
                        with _state_lock:
                            _state['day_pnl'] += _pnl_u
                        return
                except Exception as _e2:
                    log.error(f'[{strategy}] {symbol} 수동매도 자동처리 실패: {_e2}')
            return

    sl_dist   = abs(entry_price - sl)
    pnl_usdt  = (exit_price - entry_price) * qty
    pnl_pct   = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    pnl_r     = pnl_usdt / (sl_dist * qty) if sl_dist > 0 else 0

    entry_dt   = datetime.fromisoformat(entry_ts)
    hold_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

    _delete_position(strategy, symbol)
    _log_trade(strategy, symbol, entry_price, exit_price, qty, cost_usdt,
               pnl_usdt, pnl_pct, pnl_r, hold_hours, reason, regime, entry_ts)

    with _state_lock:
        _state['day_pnl'] += pnl_usdt

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

def _manage_position(strategy: str, symbol: str, ccxt_sym: str, df, i: int) -> None:
    """현재 포지션 SL/TP/청산 체크."""
    pos = _load_position(strategy, symbol)
    if pos is None:
        return

    price     = _get_price(ccxt_sym)
    if price <= 0:
        return

    entry     = float(pos['entry_price'])
    sl        = float(pos['sl'])
    tp        = float(pos['tp'])
    exit_type = pos.get('exit_type', 'sl_tp')
    max_hold  = int(pos.get('max_hold_bars', 0))
    # bars_held: DB 값 대신 실제 보유시간 기반으로 계산 (DB 업데이트 없는 버그 우회)
    _tf = STRATEGY_TIMEFRAMES.get(strategy, '1d')
    _hours_per_bar = 4 if _tf == '4h' else 24
    try:
        _entry_dt = datetime.fromisoformat(pos['entry_ts'])
        _hold_h   = (datetime.now(timezone.utc) - _entry_dt).total_seconds() / 3600
        bars_held = int(_hold_h / _hours_per_bar)
    except Exception:
        bars_held = int(pos.get('bars_held', 0))
    peak      = float(pos.get('peak_price', entry))

    # S5: BB_upper를 실시간 TP로 업데이트
    if strategy == 'S5' and df is not None and 'bb_upper' in df.columns:
        live_bb_upper = float(df.iloc[i]['bb_upper']) if not pd.isna(df.iloc[i]['bb_upper']) else 0
        if live_bb_upper > 0:
            tp = live_bb_upper

    # SL 체크
    if price <= sl:
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
                    last_bar[key] = cur_ts

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

                    # 매수 실행
                    ok = _spot_buy(strategy_id, symbol, ccxt_sym, sig, price, regime)
                    if ok and strategy_id == 'S3':
                        from atlas_spot_config import S3_COOLDOWN
                        cooldowns[key] = S3_COOLDOWN

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
                if total > _state['peak_equity']:
                    _state['peak_equity'] = total
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
    """10분마다 DB ↔ 거래소 보유 코인 검증."""
    while not stop_event.is_set() and not SPOT_KILL_SWITCH.exists():
        stop_event.wait(600)
        if stop_event.is_set():
            break
        try:
            all_pos = _load_all_positions()
            if not all_pos:
                continue
            ex    = _get_ex()
            bal   = ex.fetch_balance({'type': 'spot'})
            for pos in all_pos:
                sym    = pos['symbol']
                base   = sym.replace('USDT', '')
                actual = float(bal.get(base, {}).get('total', 0) or 0)
                db_qty = float(pos['qty_tokens'])
                if actual < db_qty * 0.5:
                    log.warning(f'[검증] {sym} DB수량 {db_qty:.6f} vs 실제 {actual:.6f} — 불일치')
                    _tg(f'⚠️ [{sym}] 포지션 불일치 감지: DB {db_qty:.6f} vs 거래소 {actual:.6f}')
        except Exception as e:
            log.warning(f'[검증루프] 오류: {e}')


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
        _state['paused'] = True
        _tg('[Spot] 신규 진입 일시정지')

    elif '/resume' in cmd:
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
    parser.add_argument('--strategies', default='S3,S5,S6,S7V4',
                        help='활성 전략 (쉼표 구분)')
    args = parser.parse_args()

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
        _state['peak_equity']  = total
        _state['day_start_eq'] = total
    log.info(f'[초기] 총자산 ${total:,.2f} (USDT ${usdt:,.2f})')

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
    _t(_balance_poller,          'BalancePoll',     stop_event)
    _t(regime_loop,              'RegimeLoop',       stop_event)
    _t(universe_refresh_loop,    'UniverseRefresh', ex, _state, stop_event)
    _t(_daily_reset_loop,        'DailyReset',      stop_event)
    _t(_position_reconcile_loop, 'Reconcile',       stop_event)
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
        _tg('🔴 ATLAS Spot 봇 종료')
        log.info('[메인] 종료 완료')


if __name__ == '__main__':
    main()

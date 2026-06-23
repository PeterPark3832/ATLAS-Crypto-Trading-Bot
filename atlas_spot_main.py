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
  RANGING      : S5(BB Bounce)
  WEAK_TREND   : S3, S5, S6 (50% 리스크)
  TRENDING_DOWN: S7V4(MACD) — S5 하락장 0승 이력으로 차단, 30% 리스크
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
    SPOT_MAX_POSITIONS, SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT,
    SPOT_RESERVE_PCT, SPOT_DAILY_LOSS_LIMIT, SPOT_MIN_ORDER_USDT, SPOT_MAX_SL_PCT,
    SPOT_KELLY_MIN_TRADES, SPOT_KELLY_SCALE_MIN, SPOT_KELLY_SCALE_MAX,
    SPOT_KELLY_WR_THRESH, SPOT_KELLY_PF_THRESH,
    SPOT_RATCHET_DD_THRESH, SPOT_RATCHET_DD_HARD, SPOT_RATCHET_RECOVER,
    SPOT_CANDLE_4H, SPOT_CANDLE_1D, SPOT_CANDLE_CACHE_TTL, SPOT_PRICE_POLL_SEC,
    STRATEGY_TIMEFRAMES, STRATEGY_NAMES, REGIME_STRATEGY_MAP, WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
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


def _get_price(ccxt_sym: str) -> float:
    """현재가 조회. 실패 시 최대 2분간 캐시 사용, 이후 0.0 반환."""
    for attempt in range(3):
        try:
            ticker = _get_ex().fetch_ticker(ccxt_sym)
            price = float(ticker['last'] or ticker['close'] or 0)
            if price > 0:
                _last_known_price[ccxt_sym] = (price, time.time())
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
    'active_strategies': ['S2', 'S3', 'S4', 'S5', 'S6', 'S7'],
    'ratchet_alert_tier': 0,     # 0=정상, 1=소프트DD, 2=하드DD — 알림 중복 방지용
    'daily_loss_alerted': False, # 일간 손실 한도 알림 중복 방지 (자정 리셋)
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
    """Kelly 스케일 계산 (최근 거래 기반, 조건부 상한 2.00)."""
    with _db_lock, _db_conn() as conn:
        rows = conn.execute(
            'SELECT pnl_r FROM spot_trades WHERE strategy=? ORDER BY id DESC LIMIT 200',
            (strategy,)
        ).fetchall()
    if len(rows) < SPOT_KELLY_MIN_TRADES:
        return SPOT_KELLY_SCALE_MIN
    pnl_r  = [r['pnl_r'] for r in rows]
    wins   = [r for r in pnl_r if r > 0]
    losses = [r for r in pnl_r if r <= 0]
    if not wins or not losses:
        return 1.0
    wr    = len(wins) / len(pnl_r)
    avg_w = float(np.mean(wins))
    avg_l = abs(float(np.mean(losses)))
    b     = avg_w / avg_l if avg_l > 0 else 1.0
    kelly = (wr - (1 - wr) / b) if b > 0 else 0.0
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
    """
    equity = _state['equity']
    peak   = _state['peak_equity']
    if peak <= 0 or equity <= 0:
        return 1.0
    dd = (peak - equity) / peak
    if dd >= SPOT_RATCHET_DD_HARD:
        if _state['ratchet_alert_tier'] < 2:
            _state['ratchet_alert_tier'] = 2
            _tg(f'🔴 [위험] 하드 드로다운 진입: 피크 ${peak:,.2f} → 현재 ${equity:,.2f} '
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
            return 0.70   # Hard DD 회복 → 중간 단계로 복원
        return 0.40
    if dd >= SPOT_RATCHET_DD_THRESH:
        if _state['ratchet_alert_tier'] < 1:
            _state['ratchet_alert_tier'] = 1
            _tg(f'⚠️ [경고] 드로다운 진입: 피크 ${peak:,.2f} → 현재 ${equity:,.2f} '
                f'(-{dd*100:.1f}%) — 리스크 스케일 70%로 축소')
        _state.pop('ratchet_floor', None)  # 소프트 DD 구간 — floor 초기화
        return 0.70
    if _state['ratchet_alert_tier'] > 0:
        _state['ratchet_alert_tier'] = 0
        _tg(f'✅ [회복] 드로다운 해소: 현재 ${equity:,.2f} (피크 대비 -{dd*100:.1f}%) — 리스크 스케일 정상 복원')
    _state.pop('ratchet_floor', None)
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

    adj_risk   = SPOT_BASE_RISK_PCT * kelly * ratchet * r_scale * funding_scale * rs_scale
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
            elif _actual_free < qty:
                log.warning(f'[{strategy}] {symbol} qty조정: {qty:.6f} -> {_actual_free:.6f} (수수료 공제 등 잔고 부족)')
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
                    elif _actual_free > 0:
                        # 잔고가 DB qty보다 약간 부족(수수료 공제 등) → 실잔고로 재시도
                        log.warning(f'[{strategy}] {symbol} 잔고재시도: {_actual_free:.6f} (DB={qty:.6f})')
                        try:
                            _retry_order = _get_ex().create_market_sell_order(ccxt_sym, _actual_free)
                            exit_price = float(_retry_order.get('average') or _retry_order.get('price') or price)
                            qty = _actual_free
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
                    with _db_lock, _db_conn() as _rc:
                        _dupe = _rc.execute(
                            "SELECT 1 FROM spot_positions "
                            "WHERE strategy=? AND symbol=? "
                            "AND datetime(entry_ts) >= datetime(?, 'unixepoch') "
                            "UNION SELECT 1 FROM spot_trades "
                            "WHERE strategy=? AND symbol=? "
                            "AND datetime(entry_ts) >= datetime(?, 'unixepoch') LIMIT 1",
                            (strategy_id, symbol, cur_ts,
                             strategy_id, symbol, cur_ts)
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
            _state['daily_loss_alerted'] = False

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
                    _delete_position(strategy, sym)
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
        _tg('🔴 ATLAS Spot 봇 종료')
        log.info('[메인] 종료 완료')


if __name__ == '__main__':
    main()

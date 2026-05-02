"""
ATLAS v2 — 라이브 트레이딩 엔진
================================
Module A: Trend Follow 4H   BTC/ETH/SOL/BNB  롱+숏
Module B: Mean Reversion 1D ETH/BNB          롱+숏

[아키텍처]
  ┌─ balance_poller ─────────── 30초마다 잔고 갱신
  ├─ regime_loop    ─────────── 1시간마다 BTC 레짐 갱신
  ├─ corr_vol_loop  ─────────── 1시간마다 상관/변동성 갱신
  ├─ daily_reset_loop ──────── 00:00 UTC 일일 리셋+브리핑
  ├─ tg_cmd_loop    ─────────── Telegram 명령 수신
  ├─ ma_symbol_loop × 4 ──── Module A 4H 루프 (BTC/ETH/SOL/BNB)
  └─ mr_symbol_loop × 2 ──── Module B 1D 루프 (ETH/BNB)

[Module A 신호]
  EMA(20) 교차 EMA(50) + ADX ≥ 20 + EMA200 방향 필터
  레짐: TRENDING_UP→LONG, TRENDING_DOWN→SHORT,
       RANGING→스킵, WEAK_TREND→양방향(EMA200 적용), CRISIS→차단
  SL: ATR×2.5 / TP: ATR×5 (고정 2R) / BEP: 1R 도달 시 SL→진입가

[Module B 신호]
  RSI 크로스오버: prev<35,curr≥35→LONG / prev>65,curr≤65→SHORT
  레짐: RANGING 전용
  SL: ATR×1.5 / TP: ATR×3 (고정 2R) / 시간청산: 15일 초과

[리스크 관리]
  Kelly 스케일링 + Drawdown Ratchet + 레짐 배율
  상관계수 보정 + 변동성 패리티 + 포트폴리오 VAR 한도
  일일 손실 -3% 자동 정지 / 최대 동시 포지션 5개
"""

import logging
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── CCXT (lazy import guard) ──────────────────────────────────
try:
    import ccxt
except ImportError:
    sys.exit('[ATLAS v2] ccxt 미설치. pip install ccxt')

from atlas_config import (
    # API
    BINANCE_API_KEY, BINANCE_API_SECRET, TG_TOKEN, TG_CHAT_ID,
    # 경로
    BASE_DIR, LOG_DIR, KILL_SWITCH_FILE,
    # 포트폴리오 리스크
    MAX_OPEN_POSITIONS, PORTFOLIO_VAR_CAP, DAILY_LOSS_LIMIT,
    CORR_SCALE_FLOOR, VOL_PARITY_FLOOR, VOL_PARITY_CAP,
    # Kelly + Ratchet
    KELLY_MIN_TRADES, KELLY_SCALE_MIN, KELLY_SCALE_MAX,
    RATCHET_DD_THRESH, RATCHET_DD_HARD, RATCHET_RECOVER_PCT,
    # 실행 엔진
    LIMIT_OFFSET_PCT, LIMIT_TIMEOUT_SEC,
    TWAP_RISK_THRESHOLD, TWAP_SPLITS, TWAP_INTERVAL_SEC,
    # 루프/캐시
    CANDLE_CACHE_TTL, PRICE_POLL_SEC,
    CANDLE_LIMIT_4H, CANDLE_LIMIT_1D,
    # 레짐
    REGIME_BTC_LOOKBACK,
    # v2 심볼
    V2_MA_SYMBOLS, V2_MR_SYMBOLS, V2_MC_SYMBOLS,
    # Module A
    V2_MA_EMA_FAST, V2_MA_EMA_SLOW, V2_MA_ATR_PERIOD,
    V2_MA_ATR_SL, V2_MA_TRAIL_R, V2_MA_RISK_PCT,
    V2_MA_LEVERAGE, V2_MA_COOLDOWN, V2_MA_ADX_MIN,
    # Module B
    V2_MR_RSI_PERIOD, V2_MR_RSI_LONG, V2_MR_RSI_SHORT,
    V2_MR_BB_PERIOD, V2_MR_BB_SIGMA, V2_MR_ATR_PERIOD,
    V2_MR_ATR_SL, V2_MR_MAX_BARS,
    V2_MR_RISK_PCT, V2_MR_RISK_PCT_BNB, V2_MR_LEVERAGE,
    # Module C
    V2_MC_ATR_PERIOD, V2_MC_ATR_SL, V2_MC_RR,
    V2_MC_VOL_LB, V2_MC_VOL_MULT, V2_MC_HTF_EMA,
    V2_MC_CHASE_LIMIT, V2_MC_COOLDOWN, V2_MC_LIMIT_SEC,
    V2_MC_LEVERAGE, V2_MC_RISK_PCT,
    CANDLE_LIMIT_15M,
    # Module D
    V2_MD_SYMBOLS, V2_MD_FUNDING_ENTER, V2_MD_FUNDING_EXIT,
    V2_MD_LEVERAGE, V2_MD_RISK_PCT, V2_MD_ATR_SL, V2_MD_ATR_PERIOD,
    V2_MD_MAX_HOURS, V2_MD_LIMIT_SEC, V2_MD_CHECK_SEC,
)
from atlas_regime import (
    get_cached_regime, update_regime, regime_loop,
    REGIME_CRISIS, REGIME_RANGING, REGIME_WEAK_TREND,
    REGIME_TRENDING_UP, REGIME_TRENDING_DOWN,
    get_risk_scale,
)
from atlas_indicators import _ohlcv_to_df, _calc_atr, _calc_rsi


# ══════════════════════════════════════════════════════════════
#  0. v2 전용 상수
# ══════════════════════════════════════════════════════════════

V2_DB_FILE       = BASE_DIR / 'state' / 'atlas_v2.db'
V2_KILL_SWITCH   = Path('/tmp/ATLAS_V2_STOP')

# Module A BEP (1R 도달 시 SL→진입가) — 백테스트와 미세 괴리 있지만 안전장치로 활성화
V2_MA_BEP_ENABLED   = True
V2_MA_BEP_TRIGGER_R = V2_MA_TRAIL_R    # 1.0R 이익 시 BEP

# Module B 펀딩비 필터 (평균회귀 방향성 비용 억제)
V2_MR_FUNDING_LONG_MAX  =  0.0008   # +0.08% 초과 → LONG 차단
V2_MR_FUNDING_SHORT_MIN = -0.0005   # -0.05% 미만 → SHORT 차단

# Module A 펀딩비 필터
V2_MA_FUNDING_LONG_MAX  =  0.0010
V2_MA_FUNDING_SHORT_MIN = -0.0006

ALL_V2_SYMBOLS = list(dict.fromkeys(V2_MA_SYMBOLS + V2_MR_SYMBOLS + V2_MC_SYMBOLS + V2_MD_SYMBOLS))


# ══════════════════════════════════════════════════════════════
#  1. 로그 초기화
# ══════════════════════════════════════════════════════════════

LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = LOG_DIR / f'atlas_v2_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
_logger = logging.getLogger('atlas_v2')


def log(msg: str, level: str = 'info'):
    getattr(_logger, level)(msg)


# ══════════════════════════════════════════════════════════════
#  2. 전역 공유 상태 (스레드 안전)
# ══════════════════════════════════════════════════════════════

_shared_lock = threading.Lock()
_shared: dict = {
    'equity':       0.0,
    'day_eq_start': 0.0,
    'paused':       False,
    'ratchet': {
        'peak_equity':   0.0,
        'scale':         1.0,
        'recover_anchor':0.0,
    },
    'kelly_stats':    {},       # strategy → {'wins':0,'total':0}
    'corr_matrix':    {},       # (sym_a, sym_b) → float
    'vol_cache':      {},       # sym → daily_vol_std
    'corr_updated':   0.0,
    # Module A 쿨다운: sym → 잔여 봉수
    'ma_cooldowns':   {},
    # Module C 쿨다운: sym → 잔여 봉수
    'mc_cooldowns':   {},
    # Module D 쿨다운: sym → 재진입 금지 해제 timestamp
    'md_cooldowns':   {},
}


# ══════════════════════════════════════════════════════════════
#  3. Thread-local CCXT 인스턴스 (Nonce 충돌 방지)
# ══════════════════════════════════════════════════════════════

_tls = threading.local()
_global_markets = None  # OOM 방지를 위한 전역 마켓 캐시

def _get_ex() -> ccxt.binance:
    global _global_markets
    """현재 스레드 전용 CCXT 인스턴스 반환 (메모리 최적화)."""
    if not hasattr(_tls, 'ex'):
        ex = ccxt.binance({
            'apiKey':  BINANCE_API_KEY,
            'secret':  BINANCE_API_SECRET,
            'options': {'defaultType': 'future'},
        })
        
        # OOM 방지: 첫 인스턴스에서만 마켓을 로드하고 나머지는 캐시된 데이터를 안전하게 주입
        if _global_markets is None:
            ex.load_markets()
            _global_markets = ex.markets
        else:
            ex.set_markets(_global_markets)
            
        _tls.ex = ex
    return _tls.ex


# ══════════════════════════════════════════════════════════════
#  4. DB 계층 (SQLite WAL)
# ══════════════════════════════════════════════════════════════

V2_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
_db_lock = threading.Lock()


@contextmanager
def _db_conn():
    with _db_lock:
        con = sqlite3.connect(V2_DB_FILE, timeout=15)
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA synchronous=NORMAL')
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()


def _db_exec(sql: str, params=()):
    with _db_conn() as con:
        con.execute(sql, params)


def _db_fetchall(sql: str, params=()):
    with _db_conn() as con:
        return con.execute(sql, params).fetchall()


def _db_fetchone(sql: str, params=()):
    with _db_conn() as con:
        return con.execute(sql, params).fetchone()


def init_db():
    with _db_conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            strategy    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            direction   TEXT NOT NULL,
            entry_price REAL NOT NULL,
            sl          REAL NOT NULL,
            tp          REAL NOT NULL,
            qty         REAL NOT NULL,
            leverage    INTEGER NOT NULL,
            risk_usd    REAL NOT NULL,
            mode        TEXT DEFAULT '',
            rr          REAL DEFAULT 2.0,
            peak        REAL DEFAULT 0.0,
            bep_done    INTEGER DEFAULT 0,
            sl_order_id TEXT DEFAULT '',
            entry_ts    TEXT NOT NULL,
            PRIMARY KEY (strategy, symbol)
        );
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy    TEXT,
            symbol      TEXT,
            direction   TEXT,
            mode        TEXT,
            entry_price REAL,
            exit_price  REAL,
            qty         REAL,
            leverage    INTEGER,
            pnl_usd     REAL,
            pnl_r       REAL,
            rr          REAL,
            hold_hours  REAL,
            reason      TEXT,
            entry_ts    TEXT,
            exit_ts     TEXT,
            regime      TEXT
        );
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """)
    log('[DB] atlas_v2.db 초기화 완료')


# ── CRUD ──────────────────────────────────────────────────────

def save_position(strategy, symbol, direction, entry, sl, tp,
                  qty, leverage, risk_usd, mode, rr, sl_order_id=''):
    _db_exec("""
        INSERT OR REPLACE INTO positions
        (strategy,symbol,direction,entry_price,sl,tp,qty,leverage,
         risk_usd,mode,rr,peak,bep_done,sl_order_id,entry_ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)
    """, (strategy, symbol, direction, entry, sl, tp, qty, leverage,
          risk_usd, mode, rr, entry, sl_order_id,
          datetime.now(timezone.utc).isoformat()))


def load_position(strategy: str, symbol: str) -> Optional[dict]:
    row = _db_fetchone(
        'SELECT * FROM positions WHERE strategy=? AND symbol=?',
        (strategy, symbol))
    return dict(row) if row else None


def load_all_positions() -> list:
    rows = _db_fetchall('SELECT * FROM positions')
    return [dict(r) for r in rows]


def count_open_positions() -> int:
    row = _db_fetchone('SELECT COUNT(*) as cnt FROM positions')
    return row['cnt'] if row else 0


def delete_position(strategy: str, symbol: str):
    _db_exec('DELETE FROM positions WHERE strategy=? AND symbol=?',
             (strategy, symbol))


def update_position_sl(strategy, symbol, new_sl, sl_order_id=''):
    _db_exec(
        'UPDATE positions SET sl=?, sl_order_id=? WHERE strategy=? AND symbol=?',
        (new_sl, sl_order_id, strategy, symbol))


def update_position_bep(strategy, symbol):
    """BEP 완료 플래그 설정."""
    _db_exec(
        'UPDATE positions SET bep_done=1 WHERE strategy=? AND symbol=?',
        (strategy, symbol))


def log_trade(strategy, symbol, direction, mode, entry, exit_px,
              qty, leverage, pnl_usd, pnl_r, rr, hold_h, reason,
              entry_ts, regime_name):
    _db_exec("""
        INSERT INTO trades
        (strategy,symbol,direction,mode,entry_price,exit_price,qty,leverage,
         pnl_usd,pnl_r,rr,hold_hours,reason,entry_ts,exit_ts,regime)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (strategy, symbol, direction, mode, entry, exit_px, qty, leverage,
          pnl_usd, pnl_r, rr, hold_h, reason, entry_ts,
          datetime.now(timezone.utc).isoformat(), regime_name))

    # Kelly 통계 갱신
    with _shared_lock:
        ks = _shared['kelly_stats'].setdefault(strategy, {'wins': 0, 'total': 0})
        ks['total'] += 1
        if pnl_usd > 0:
            ks['wins'] += 1


def get_config(key: str) -> Optional[str]:
    row = _db_fetchone('SELECT value FROM config WHERE key=?', (key,))
    return row['value'] if row else None


def set_config(key: str, value: str):
    _db_exec('INSERT OR REPLACE INTO config (key,value) VALUES (?,?)', (key, value))


# ══════════════════════════════════════════════════════════════
#  5. CandleCache (API 부하 절감)
# ══════════════════════════════════════════════════════════════

class CandleCache:
    def __init__(self):
        self._cache: dict = {}
        self._lock = threading.Lock()

    def get(self, ex, ccxt_sym: str, tf: str, limit: int) -> list:
        key = (ccxt_sym, tf)
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry['ts']) < CANDLE_CACHE_TTL:
                return entry['data']
        try:
            data = ex.fetch_ohlcv(ccxt_sym, tf, limit=limit)
        except Exception as e:
            log(f'[Cache] {ccxt_sym} {tf} 로드 실패: {e}', 'warning')
            with self._lock:
                return self._cache.get(key, {}).get('data', [])
        with self._lock:
            self._cache[key] = {'data': data, 'ts': time.time()}
        return data


# ══════════════════════════════════════════════════════════════
#  6. Telegram
# ══════════════════════════════════════════════════════════════

_tg_session = requests.Session()


def tg(msg: str):
    # 전략명(V2_MA_LONG 등)의 언더바가 Markdown 이탤릭 파싱 오류를 유발하므로 치환
    safe_msg = msg.replace('_', '-')
    try:
        _tg_session.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': safe_msg,
                  'parse_mode': 'Markdown'},
            timeout=10)
    except Exception as e:
        log(f'[TG] 전송 실패: {e}', 'warning')


# ══════════════════════════════════════════════════════════════
#  7. 잔고 폴러
# ══════════════════════════════════════════════════════════════

def balance_poller():
    while True:
        try:
            bal = _get_ex().fetch_balance()
            eq  = float(bal.get('USDT', {}).get('total', 0) or
                        bal.get('total', {}).get('USDT', 0))
            with _shared_lock:
                prev_eq = _shared['equity']
                _shared['equity'] = eq

                # 일일 손실 한도 체크
                day_start = _shared['day_eq_start']
                if day_start > 0 and eq > 0:
                    day_pnl_pct = (eq - day_start) / day_start
                    if day_pnl_pct <= DAILY_LOSS_LIMIT and not _shared['paused']:
                        _shared['paused'] = True
                        tg(f'⛔ [ATLAS v2] 일일 손실 한도 도달 ({day_pnl_pct*100:.2f}%)\n'
                           f'  신규 진입 정지. 내일 00:00 UTC 자동 해제.')

                # Ratchet peak 갱신
                rat = _shared['ratchet']
                if eq > rat['peak_equity']:
                    rat['peak_equity'] = eq
                    rat['recover_anchor'] = eq

                # Ratchet 스케일 갱신
                peak = rat['peak_equity']
                if peak > 0:
                    dd = (peak - eq) / peak
                    if dd >= RATCHET_DD_HARD:
                        rat['scale'] = 0.40
                    elif dd >= RATCHET_DD_THRESH:
                        rat['scale'] = 0.70
                    else:
                        # 회복 구간: 저점 대비 20%마다 +10% 복원
                        if eq > rat['recover_anchor'] * (1 + RATCHET_RECOVER_PCT):
                            rat['scale'] = min(1.0, rat['scale'] + 0.10)
                            rat['recover_anchor'] = eq

        except Exception as e:
            log(f'[Balance] 조회 실패: {e}', 'warning')
        time.sleep(30)


# ══════════════════════════════════════════════════════════════
#  8. 리스크 스케일링 헬퍼
# ══════════════════════════════════════════════════════════════

def get_kelly_scale(strategy: str, base_risk: float) -> tuple[float, str]:
    with _shared_lock:
        ks = _shared['kelly_stats'].get(strategy, {})
    total = ks.get('total', 0)
    wins  = ks.get('wins',  0)
    if total < KELLY_MIN_TRADES:
        return base_risk, f'Kelly:기본(n={total})'
    wr   = wins / total
    rr   = 2.0  # 고정 2R 전략
    kelly = max(0, (wr * (rr + 1) - 1) / rr)
    scale = max(KELLY_SCALE_MIN, min(KELLY_SCALE_MAX, kelly))
    return base_risk * scale, f'Kelly:{scale:.2f}x(WR={wr:.0%})'


def get_ratchet_scale() -> float:
    with _shared_lock:
        return _shared['ratchet']['scale']


def check_portfolio_var() -> tuple[bool, str]:
    """현재 열린 포지션의 총 리스크 합계 vs VAR 한도."""
    positions = load_all_positions()
    if not positions:
        return True, ''
    with _shared_lock:
        eq = _shared['equity']
    if eq <= 0:
        return True, ''
    total_risk = sum(p.get('risk_usd', 0) for p in positions)
    var_pct = total_risk / eq
    if var_pct >= PORTFOLIO_VAR_CAP:
        return False, f'포트폴리오 VAR {var_pct*100:.1f}% >= {PORTFOLIO_VAR_CAP*100:.0f}%'
    return True, ''


def get_corr_scale(sym: str, direction: str) -> float:
    """동일 방향 포지션과의 상관계수가 높으면 리스크 축소."""
    with _shared_lock:
        corr_matrix = _shared['corr_matrix']
        if not corr_matrix:
            return 1.0
    positions = load_all_positions()
    max_corr = 0.0
    sym_clean = sym.replace('USDT', '')
    for p in positions:
        if p['direction'] != direction:
            continue
        p_clean = p['symbol'].replace('USDT', '')
        key     = tuple(sorted([sym_clean, p_clean]))
        corr    = corr_matrix.get(key, 0.0)
        max_corr = max(max_corr, corr)
    if max_corr <= 0:
        return 1.0
    # 상관 1.0 → CORR_SCALE_FLOOR, 상관 0.0 → 1.0
    scale = 1.0 - max_corr * (1.0 - CORR_SCALE_FLOOR)
    return max(CORR_SCALE_FLOOR, scale)


def get_vol_parity_scale(sym: str) -> float:
    """변동성이 평균 대비 높으면 리스크 축소, 낮으면 확대 (패리티)."""
    with _shared_lock:
        vol_cache = _shared['vol_cache']
    if len(vol_cache) < 2:
        return 1.0
    vols = list(vol_cache.values())
    avg_vol = sum(vols) / len(vols)
    sym_vol = vol_cache.get(sym, avg_vol)
    if sym_vol <= 0 or avg_vol <= 0:
        return 1.0
    scale = avg_vol / sym_vol
    return max(VOL_PARITY_FLOOR, min(VOL_PARITY_CAP, scale))


# ══════════════════════════════════════════════════════════════
#  9. 거래소 헬퍼
# ══════════════════════════════════════════════════════════════

def get_price(ccxt_sym: str) -> float:
    try:
        t = _get_ex().fetch_ticker(ccxt_sym)
        return float(t.get('last', 0) or 0)
    except Exception as e:
        log(f'[Price] {ccxt_sym} 조회 실패: {e}', 'warning')
        return 0.0


def get_funding_rate(ccxt_sym: str) -> float:
    try:
        info = _get_ex().fetch_funding_rate(ccxt_sym)
        return float(info.get('fundingRate', 0) or 0)
    except Exception:
        return 0.0


def get_qty_precision(ccxt_sym: str) -> tuple[int, float]:
    try:
        ex  = _get_ex()
        if not ex.markets:
            ex.load_markets()
        mkt = ex.market(ccxt_sym)
        prec     = int(mkt.get('precision', {}).get('amount', 3) or 3)
        min_qty  = float(mkt.get('limits', {}).get('amount', {}).get('min', 0.001) or 0.001)
        return prec, min_qty
    except Exception:
        return 3, 0.001


def set_leverage(ccxt_sym: str, leverage: int):
    try:
        _get_ex().set_leverage(leverage, ccxt_sym)
    except Exception as e:
        log(f'[Leverage] {ccxt_sym} {leverage}x 설정 실패: {e}', 'warning')


def place_exchange_sl(ccxt_sym: str, pos_side: str, qty: float, sl_price: float) -> str:
    close_side = 'sell' if pos_side == 'long' else 'buy'
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


def _wait_fill(ccxt_sym: str, side: str, timeout: int = 20) -> float:
    expected = 'long' if side == 'buy' else 'short'
    ex = _get_ex()
    for _ in range(timeout // 2):
        time.sleep(2)
        try:
            # 💡 패치: 옵션 마켓 버그 우회 (fetch_position -> fetch_positions)
            positions = ex.fetch_positions([ccxt_sym])
            pos = positions[0] if positions else {}
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
    """지정가 우선 → 미체결 시 취소 후 시장가 폴백 (이중체결 방지)."""
    ex = _get_ex()
    limit_px = round(signal_price * (1 - LIMIT_OFFSET_PCT if side == 'buy'
                                     else 1 + LIMIT_OFFSET_PCT), 2)
    try:
        order = ex.create_limit_order(ccxt_sym, side, qty, limit_px)
        oid   = order.get('id', '')
    except Exception as e:
        log(f'[Exec] 지정가 생성 실패 -> 시장가: {e}', 'warning')
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

    try:
        ex.cancel_order(oid, ccxt_sym)
    except Exception:
        pass

    # 취소 후 최종 상태 확인 (이중체결 방지)
    try:
        o      = ex.fetch_order(oid, ccxt_sym)
        filled = float(o.get('filled', 0) or 0)
        status = o.get('status', '')
        if status == 'closed' and filled >= qty * 0.99:
            return True, float(o.get('average', 0) or signal_price), '지정가 완전체결(지연감지)'
        if filled > 0:
            avg_px  = float(o.get('average', 0) or limit_px)
            rem_raw = qty - filled
            try:
                rem = float(ex.amount_to_precision(ccxt_sym, rem_raw)) if rem_raw > 0 else 0
            except Exception:
                rem = rem_raw
            if rem > 0:
                ok, mkt_px, _ = market_order(ccxt_sym, side, rem)
                if ok and mkt_px > 0:
                    blended = (avg_px * filled + mkt_px * rem) / qty
                    return True, blended, f'부분체결+시장가'
            return True, avg_px, '부분체결'
    except Exception as e:
        log(f'[Exec] 취소 후 상태 확인 실패: {e}', 'warning')

    return market_order(ccxt_sym, side, qty)


def twap_order(ccxt_sym: str, side: str, total_qty: float) -> tuple[bool, float, str]:
    ex = _get_ex()
    try:
        split = float(ex.amount_to_precision(ccxt_sym, total_qty / TWAP_SPLITS))
    except Exception:
        split = total_qty / TWAP_SPLITS
    fills = []
    for i in range(TWAP_SPLITS):
        ok, px, _ = market_order(ccxt_sym, side, split)
        if ok and px > 0:
            fills.append(px)
        if i < TWAP_SPLITS - 1:
            time.sleep(TWAP_INTERVAL_SEC)
    if not fills:
        return False, 0, 'TWAP 전체 실패'
    return True, sum(fills) / len(fills), f'TWAP {len(fills)}/{TWAP_SPLITS}'


# ══════════════════════════════════════════════════════════════
#  10. 진입 / 청산 공통
# ══════════════════════════════════════════════════════════════

def calc_raw_qty(equity: float, entry: float, sl: float,
                 risk_pct: float) -> tuple[float, float]:
    sl_dist  = abs(entry - sl)
    if sl_dist <= 0 or entry <= 0:
        return 0.0, 0.0
    risk_usd = equity * risk_pct
    return risk_usd / sl_dist, risk_usd


def enter_position(ccxt_sym: str, strategy: str, sym: str,
                   direction: str, mode: str, rr: float,
                   entry_sig: float, sl: float, tp: float,
                   leverage: int, risk_pct: float) -> bool:
    ex = _get_ex()

    with _shared_lock:
        equity = _shared['equity']
        paused = _shared['paused']
    if paused:
        return False

    regime = get_cached_regime()
    if regime.regime == REGIME_CRISIS:
        return False

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        log(f'[진입차단] {sym}: 최대 포지션 수 도달')
        return False

    var_ok, var_reason = check_portfolio_var()
    if not var_ok:
        log(f'[진입차단] {sym}: {var_reason}')
        return False

    # 리스크 스케일링
    regime_scale  = get_risk_scale()
    kelly_risk, kelly_note = get_kelly_scale(strategy, risk_pct)
    ratchet_scale = get_ratchet_scale()
    corr_scale    = get_corr_scale(sym, direction)
    vol_scale     = get_vol_parity_scale(sym)
    final_risk    = kelly_risk * regime_scale * ratchet_scale * corr_scale * vol_scale
    final_risk    = max(0.001, min(0.05, final_risk))

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
        ok, fill, note = twap_order(ccxt_sym, side, qty)
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

    regime_name = regime.regime
    tg(f"[{strategy}] {sym} {direction} 진입\n"
       f"  모드:{mode} RR:{rr:.2f} {leverage}x\n"
       f"  진입:{fill:,.4f} SL:{sl_final:,.4f} TP:{tp_final:,.4f}\n"
       f"  리스크:${risk_usd:.1f}({final_risk*100:.2f}%) {regime_name}\n"
       f"  {kelly_note}")
    log(f'[진입] {sym} {strategy} {direction} fill={fill:.4f} SL={sl_final:.4f} TP={tp_final:.4f}')
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
        err = str(e).lower()
        # 거래소에 포지션이 이미 없는 경우 (수동 청산 등) → DB만 정리하고 정상 처리
        if any(k in err for k in ('no open', 'position not found', '-4021', '-2022',
                                   'does not exist', 'reduceonly')):
            log(f'[수동청산 감지] {sym} {strategy}: 포지션이 거래소에 없음 — DB 정리')
            tg(f'⚠️ [{strategy}] {sym} {direction} 수동청산 감지\nDB 자동 정리 완료')
            # 아래 log_trade + delete_position 으로 진행 (return False 하지 않음)
        else:
            log(f'[청산실패] {sym} {strategy}: {e}', 'error')
            return False

    # 잔여 포지션 재청산
    time.sleep(2)
    try:
        # 💡 패치: 옵션 마켓 버그 우회
        positions = _get_ex().fetch_positions([ccxt_sym])
        ex_pos = positions[0] if positions else {}
        remaining = abs(float(ex_pos.get('contracts', 0) or 0))
        ex_side   = (ex_pos.get('side') or '').lower()
        expected  = 'long' if direction == 'LONG' else 'short'
        if remaining > 0 and ex_side == expected:
            log(f'[청산검증] {sym} 잔여 {remaining} -> 재시도', 'warning')
            try:
                rem_prec = float(_get_ex().amount_to_precision(ccxt_sym, remaining))
                if rem_prec > 0:
                    _get_ex().create_market_order(
                        ccxt_sym, side, rem_prec, params={'reduceOnly': True})
            except Exception as retry_e:
                log(f'[청산재시도] {sym} 실패: {retry_e}', 'error')
                tg(f'[{strategy}] {sym} 부분청산 잔여 {remaining} 처리 실패!\n수동 확인 필요.')
    except Exception as e:
        log(f'[청산검증] {sym} 확인 실패: {e}', 'warning')

    pnl_pct = (exit_price - entry) / entry if direction == 'LONG' else (entry - exit_price) / entry
    pnl_usd = pnl_pct * qty * entry
    sl_dist = abs(entry - pos['sl'])
    pnl_r   = pnl_usd / (qty * sl_dist) if sl_dist > 0 else 0
    try:
        entry_dt = datetime.fromisoformat(entry_ts.replace('Z', '+00:00'))
        hold_h   = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
    except Exception:
        hold_h = 0

    regime_name = get_cached_regime().regime
    log_trade(strategy, sym, direction, pos.get('mode', ''), entry, exit_price,
              qty, leverage, pnl_usd, pnl_r, pos.get('rr', 2.0), hold_h, reason,
              entry_ts, regime_name)
    delete_position(strategy, sym)

    # Module A / C 쿨다운 갱신
    if strategy.startswith('V2_MA'):
        with _shared_lock:
            _shared['ma_cooldowns'][sym] = V2_MA_COOLDOWN
    elif strategy.startswith('V2_MC'):
        with _shared_lock:
            _shared['mc_cooldowns'][sym] = V2_MC_COOLDOWN

    emoji = '✅' if pnl_usd >= 0 else '❌'
    tg(f"{emoji} [{strategy}] {sym} {direction} 청산({reason})\n"
       f"  {entry:,.4f} -> {exit_price:,.4f} | ${pnl_usd:+.1f} ({pnl_pct*100:+.2f}%) {hold_h:.1f}H")
    log(f'[청산] {sym} {strategy} {direction} {reason} PnL=${pnl_usd:+.1f} ({pnl_r:+.2f}R)')
    return True


# ══════════════════════════════════════════════════════════════
#  11. Module A 지표 계산
# ══════════════════════════════════════════════════════════════

def _calc_adx_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ADX. TR/DM=sum-based, ADX=mean-based."""
    h  = df['high'].values
    lo = df['low'].values
    c  = df['close'].values
    pc = np.concatenate([[np.nan], c[:-1]])
    up   = h - np.concatenate([[np.nan], h[:-1]])
    down = np.concatenate([[np.nan], lo[:-1]]) - lo
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)

    high_s = pd.Series(h)
    low_s  = pd.Series(lo)
    prev_c = pd.Series(pc)
    tr_arr = pd.concat([
        high_s - low_s,
        (high_s - prev_c).abs(),
        (low_s  - prev_c).abs(),
    ], axis=1).max(axis=1).values

    def _wsum(arr, p):
        res = np.full(len(arr), np.nan)
        if len(arr) < p:
            return res
        res[p - 1] = float(np.nansum(arr[:p]))
        for i in range(p, len(arr)):
            res[i] = res[i - 1] - res[i - 1] / p + arr[i]
        return res

    def _wmean(arr, p):
        res = np.full(len(arr), np.nan)
        if len(arr) < p:
            return res
        res[p - 1] = float(np.nanmean(arr[:p]))
        for i in range(p, len(arr)):
            res[i] = res[i - 1] - res[i - 1] / p + arr[i] / p
        return res

    tr_w  = _wsum(tr_arr, period)
    dmp_w = _wsum(dm_p,   period)
    dmm_w = _wsum(dm_m,   period)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_p  = np.where(tr_w > 0, 100 * dmp_w / tr_w, 0.0)
        di_m  = np.where(tr_w > 0, 100 * dmm_w / tr_w, 0.0)
        dip_m = di_p + di_m
        dx    = np.where(dip_m > 0, np.abs(di_p - di_m) / dip_m * 100, 0.0)
        
    return _wmean(dx, period)


def calc_ma_indicators(ohlcv: list) -> pd.DataFrame:
    """Module A 지표: EMA20, EMA50, EMA200, ATR14, ADX14."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['ema20']  = c.ewm(span=V2_MA_EMA_FAST, adjust=False).mean()
    df['ema50']  = c.ewm(span=V2_MA_EMA_SLOW, adjust=False).mean()
    df['ema200'] = c.ewm(span=200,             adjust=False).mean()
    df['atr']    = _calc_atr(df, V2_MA_ATR_PERIOD)
    df['adx']    = _calc_adx_series(df, 14)
    return df


def calc_mr_indicators(ohlcv: list) -> pd.DataFrame:
    """Module B 지표: RSI14, BB(20,2), ATR14."""
    df = _ohlcv_to_df(ohlcv)
    c  = df['close']
    df['rsi'] = _calc_rsi(c, V2_MR_RSI_PERIOD)
    mid       = c.rolling(V2_MR_BB_PERIOD).mean()
    std       = c.rolling(V2_MR_BB_PERIOD).std()
    df['bb_mid']   = mid
    df['bb_upper'] = mid + V2_MR_BB_SIGMA * std
    df['bb_lower'] = mid - V2_MR_BB_SIGMA * std
    df['atr']      = _calc_atr(df, V2_MR_ATR_PERIOD)
    return df


# ══════════════════════════════════════════════════════════════
#  12. Module A — 포지션 관리 + 신호 진입
# ══════════════════════════════════════════════════════════════

def manage_ma_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    """Module A 포지션 감시: SL/TP 확인 + BEP 이동."""
    strategy  = pos['strategy']
    direction = pos['direction']
    entry     = pos['entry_price']
    sl        = pos['sl']
    tp        = pos['tp']
    bep_done  = bool(pos.get('bep_done', 0))
    sl_dist   = abs(entry - sl)

    if direction == 'LONG':
        if price <= sl:
            exit_position(ccxt_sym, pos, price, 'SL')
            return
        if price >= tp:
            exit_position(ccxt_sym, pos, price, 'TP')
            return
        # BEP: 1R 이익 도달 시 SL → 진입가
        if V2_MA_BEP_ENABLED and not bep_done:
            bep_target = entry + sl_dist * V2_MA_BEP_TRIGGER_R
            if price >= bep_target:
                cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))
                new_oid = place_exchange_sl(ccxt_sym, 'long', pos['qty'], entry)
                update_position_sl(strategy, sym, entry, new_oid)
                update_position_bep(strategy, sym)
                log(f'[BEP] {sym} {strategy} SL -> 진입가 {entry:.4f}')
    else:  # SHORT
        if price >= sl:
            exit_position(ccxt_sym, pos, price, 'SL')
            return
        if price <= tp:
            exit_position(ccxt_sym, pos, price, 'TP')
            return
        if V2_MA_BEP_ENABLED and not bep_done:
            bep_target = entry - sl_dist * V2_MA_BEP_TRIGGER_R
            if price <= bep_target:
                cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))
                new_oid = place_exchange_sl(ccxt_sym, 'short', pos['qty'], entry)
                update_position_sl(strategy, sym, entry, new_oid)
                update_position_bep(strategy, sym)
                log(f'[BEP] {sym} {strategy} SL -> 진입가 {entry:.4f}')


def check_ma_signal_and_enter(sym: str, df: pd.DataFrame, cache: CandleCache):
    """Module A 신호 감지 후 진입 시도."""
    strategy_long  = 'V2_MA_LONG'
    strategy_short = 'V2_MA_SHORT'
    ccxt_sym = sym.replace('USDT', '/USDT')

    # 이미 포지션 있으면 스킵
    if (load_position(strategy_long,  sym) is not None or
            load_position(strategy_short, sym) is not None):
        return

    # 쿨다운 체크
    with _shared_lock:
        cd = _shared['ma_cooldowns'].get(sym, 0)
    if cd > 0:
        log(f'[MA] {sym} 쿨다운 {cd}봉 남음')
        return

    # 레짐 체크
    regime = get_cached_regime().regime
    if regime in (REGIME_CRISIS, REGIME_RANGING, 'UNKNOWN'):
        log(f'[MA 스킵] {sym} 레짐={regime} (진입불가)')
        return

    if len(df) < 3:
        return
    row      = df.iloc[-1]
    row_prev = df.iloc[-2]

    # 지표값 추출
    ema20_curr  = float(row['ema20'])
    ema50_curr  = float(row['ema50'])
    ema200_curr = float(row['ema200'])
    adx_curr    = float(row['adx']) if not pd.isna(row['adx']) else 0.0
    atr_curr    = float(row['atr']) if not pd.isna(row['atr']) else 0.0
    close_curr  = float(row['close'])

    ema20_prev = float(row_prev['ema20'])
    ema50_prev = float(row_prev['ema50'])

    if any(pd.isna(v) for v in (ema20_curr, ema50_curr, ema200_curr,
                                  ema20_prev, ema50_prev)):
        return
    if atr_curr <= 0:
        return

    # ADX 필터
    if adx_curr < V2_MA_ADX_MIN:
        log(f'[MA 스킵] {sym} ADX={adx_curr:.1f} < 기준{V2_MA_ADX_MIN} (추세 약함) | 레짐={regime}')
        return

    low_prev  = float(row_prev['low'])
    high_prev = float(row_prev['high'])

    # ── 신호 A: EMA 크로스오버 ──────────────────────────────────
    long_cross  = (ema20_prev <= ema50_prev) and (ema20_curr > ema50_curr)
    short_cross = (ema20_prev >= ema50_prev) and (ema20_curr < ema50_curr)

    # ── 신호 B: EMA20 눌림목 (추세 중 되돌림 후 반등/거부) ──────
    long_pb  = (ema20_curr > ema50_curr and
                low_prev   <= ema20_prev and
                close_curr  > ema20_curr)
    short_pb = (ema20_curr < ema50_curr and
                high_prev  >= ema20_prev and
                close_curr  < ema20_curr)

    # 신호 없으면 스킵
    if not (long_cross or short_cross or long_pb or short_pb):
        gap = (ema20_curr - ema50_curr) / ema50_curr * 100
        log(f'[MA 스킵] {sym} 크로스/눌림목 없음 (EMA갭={gap:+.3f}%) ADX={adx_curr:.1f} | 레짐={regime}')
        return

    # 방향 결정 (크로스오버 > 눌림목 우선순위, 신호 충돌 시 스킵)
    if long_cross:
        direction, sig_type = 'LONG',  'CROSS'
    elif short_cross:
        direction, sig_type = 'SHORT', 'CROSS'
    elif long_pb and not short_pb:
        direction, sig_type = 'LONG',  'PULLBACK'
    elif short_pb and not long_pb:
        direction, sig_type = 'SHORT', 'PULLBACK'
    else:
        return  # 신호 충돌

    # 레짐-방향 필터
    if direction == 'LONG' and regime == REGIME_TRENDING_DOWN:
        log(f'[MA 스킵] {sym} {sig_type} LONG이나 레짐=TRENDING_DOWN — 방향 불일치')
        return
    if direction == 'SHORT' and regime == REGIME_TRENDING_UP:
        log(f'[MA 스킵] {sym} {sig_type} SHORT이나 레짐=TRENDING_UP — 방향 불일치')
        return

    # EMA200 방향 필터 (심볼별 자체)
    if direction == 'LONG' and close_curr < ema200_curr:
        log(f'[MA 스킵] {sym} LONG이나 종가({close_curr:.2f}) < EMA200({ema200_curr:.2f})')
        return
    if direction == 'SHORT' and close_curr > ema200_curr:
        log(f'[MA 스킵] {sym} SHORT이나 종가({close_curr:.2f}) > EMA200({ema200_curr:.2f})')
        return

    # 펀딩비 필터
    fr = get_funding_rate(ccxt_sym)
    if direction == 'LONG'  and fr >  V2_MA_FUNDING_LONG_MAX:
        log(f'[MA] {sym} LONG 펀딩비 차단 ({fr:.4%})')
        return
    if direction == 'SHORT' and fr <  V2_MA_FUNDING_SHORT_MIN:
        log(f'[MA] {sym} SHORT 펀딩비 차단 ({fr:.4%})')
        return

    # 진입가 = 현재 시장가 (다음 캔들 시가 대신 즉시 시장가)
    entry_price = get_price(ccxt_sym)
    if entry_price <= 0:
        return

    if direction == 'LONG':
        sl_price = entry_price - atr_curr * V2_MA_ATR_SL
        tp_price = entry_price + atr_curr * V2_MA_ATR_SL * 2.0
    else:
        sl_price = entry_price + atr_curr * V2_MA_ATR_SL
        tp_price = entry_price - atr_curr * V2_MA_ATR_SL * 2.0

    strategy = strategy_long if direction == 'LONG' else strategy_short
    leverage = V2_MA_LEVERAGE.get(sym, 3)
    risk_pct = V2_MA_RISK_PCT.get(sym, 0.010)
    log(f'[MA 신호] {sym} {direction} {sig_type} | ADX={adx_curr:.1f} 레짐={regime} {leverage}x/{risk_pct*100:.1f}%')
    enter_position(ccxt_sym, strategy, sym, direction, 'MA', 2.0,
                   entry_price, sl_price, tp_price,
                   leverage, risk_pct)


def ma_symbol_loop(sym: str, cache: CandleCache):
    """Module A 4H 심볼 루프."""
    ccxt_sym   = sym.replace('USDT', '/USDT')
    last_4h_ts = None
    log(f'[MA] {sym} 루프 시작')

    while True:
        if V2_KILL_SWITCH.exists():
            break
        try:
            # 쿨다운 감소
            with _shared_lock:
                cd = _shared['ma_cooldowns'].get(sym, 0)
                if cd > 0:
                    _shared['ma_cooldowns'][sym] = cd - 1

            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(PRICE_POLL_SEC); continue

            ohlcv_4h = cache.get(_get_ex(), ccxt_sym, '4h', CANDLE_LIMIT_4H)
            if not ohlcv_4h:
                time.sleep(PRICE_POLL_SEC); continue

            df         = calc_ma_indicators(ohlcv_4h)
            cur_4h_ts  = ohlcv_4h[-1][0]

            # 포지션 감시 (매 틱)
            for strat in ('V2_MA_LONG', 'V2_MA_SHORT'):
                pos = load_position(strat, sym)
                if pos:
                    manage_ma_position(ccxt_sym, sym, pos, price)

            # 새 4H 캔들 확인 후 신호 체크
            if cur_4h_ts != last_4h_ts and last_4h_ts is not None:
                check_ma_signal_and_enter(sym, df, cache)
            last_4h_ts = cur_4h_ts

        except Exception as e:
            log(f'[MA] {sym} 루프 오류: {e}', 'error')
        time.sleep(PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  13. Module B — 포지션 관리 + 신호 진입
# ══════════════════════════════════════════════════════════════

def _mr_hold_days(pos: dict) -> int:
    """Module B 포지션 보유 일수."""
    try:
        entry_dt = datetime.fromisoformat(pos['entry_ts'].replace('Z', '+00:00'))
        return int((datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400)
    except Exception:
        return 0


def manage_mr_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    """Module B 포지션 감시: SL/TP + 시간청산."""
    direction = pos['direction']
    entry     = pos['entry_price']
    sl        = pos['sl']
    tp        = pos['tp']

    if direction == 'LONG':
        if price <= sl:
            exit_position(ccxt_sym, pos, price, 'SL')
            return
        if price >= tp:
            exit_position(ccxt_sym, pos, price, 'TP')
            return
    else:
        if price >= sl:
            exit_position(ccxt_sym, pos, price, 'SL')
            return
        if price <= tp:
            exit_position(ccxt_sym, pos, price, 'TP')
            return

    # 시간청산: V2_MR_MAX_BARS일 초과
    hold_days = _mr_hold_days(pos)
    if hold_days >= V2_MR_MAX_BARS:
        log(f'[MR] {sym} {direction} 시간청산 ({hold_days}일)')
        exit_position(ccxt_sym, pos, price, 'TIME')


def check_mr_signal_and_enter(sym: str, df: pd.DataFrame):
    """Module B 신호 감지 후 진입 시도."""
    strategy_long  = 'V2_MR_LONG'
    strategy_short = 'V2_MR_SHORT'
    ccxt_sym = sym.replace('USDT', '/USDT')

    if (load_position(strategy_long,  sym) is not None or
            load_position(strategy_short, sym) is not None):
        return

    # RANGING + WEAK_TREND 허용 (RANGING 전용 → 약추세 포함으로 확장)
    regime = get_cached_regime().regime
    if regime not in (REGIME_RANGING, REGIME_WEAK_TREND):
        log(f'[MR 스킵] {sym} 레짐={regime} (RANGING/WEAK_TREND 필요 — 대기 중)')
        return

    if len(df) < 3:
        return
    row      = df.iloc[-1]
    row_prev = df.iloc[-2]

    rsi_curr   = float(row['rsi'])      if not pd.isna(row['rsi'])      else 50.0
    rsi_prev   = float(row_prev['rsi']) if not pd.isna(row_prev['rsi']) else 50.0
    atr        = float(row['atr'])      if not pd.isna(row['atr'])      else 0.0
    close_curr = float(row['close'])
    bb_upper   = float(row['bb_upper']) if not pd.isna(row['bb_upper']) else 0.0
    bb_lower   = float(row['bb_lower']) if not pd.isna(row['bb_lower']) else 0.0

    if atr <= 0:
        return

    # RSI 크로스오버 신호
    direction = None
    if rsi_prev < V2_MR_RSI_LONG and rsi_curr >= V2_MR_RSI_LONG:
        direction = 'LONG'
    elif rsi_prev > V2_MR_RSI_SHORT and rsi_curr <= V2_MR_RSI_SHORT:
        direction = 'SHORT'

    if direction is None:
        log(f'[MR 스킵] {sym} RSI 크로스 없음 (RSI={rsi_curr:.1f}, 기준 L<{V2_MR_RSI_LONG}/S>{V2_MR_RSI_SHORT})')
        return

    # BB 위치 필터 — 단일 RSI 신호 약점 보완
    bb_tolerance = 0.02 if regime == REGIME_RANGING else 0.03
    if direction == 'LONG' and bb_lower > 0 and close_curr > bb_lower * (1 + bb_tolerance):
        log(f'[MR 스킵] {sym} LONG BB 미충족 '
            f'(종가={close_curr:.2f} > BB하단×{1+bb_tolerance:.3f}={bb_lower*(1+bb_tolerance):.2f})')
        return
    if direction == 'SHORT' and bb_upper > 0 and close_curr < bb_upper * (1 - bb_tolerance):
        log(f'[MR 스킵] {sym} SHORT BB 미충족 '
            f'(종가={close_curr:.2f} < BB상단×{1-bb_tolerance:.3f}={bb_upper*(1-bb_tolerance):.2f})')
        return

    # 펀딩비 필터
    fr = get_funding_rate(ccxt_sym)
    if direction == 'LONG'  and fr >  V2_MR_FUNDING_LONG_MAX:
        log(f'[MR] {sym} LONG 펀딩비 차단 ({fr:.4%})')
        return
    if direction == 'SHORT' and fr <  V2_MR_FUNDING_SHORT_MIN:
        log(f'[MR] {sym} SHORT 펀딩비 차단 ({fr:.4%})')
        return

    entry_price = get_price(ccxt_sym)
    if entry_price <= 0:
        return

    if direction == 'LONG':
        sl_price = entry_price - atr * V2_MR_ATR_SL
        tp_price = entry_price + atr * V2_MR_ATR_SL * 2.0
    else:
        sl_price = entry_price + atr * V2_MR_ATR_SL
        tp_price = entry_price - atr * V2_MR_ATR_SL * 2.0

    # 심볼별 리스크
    _MR_RISK = {
        'ETHUSDT': V2_MR_RISK_PCT,
        'BNBUSDT': V2_MR_RISK_PCT_BNB,
        'XRPUSDT': 0.004,
    }
    risk_pct = _MR_RISK.get(sym, 0.004)

    strategy = strategy_long if direction == 'LONG' else strategy_short
    enter_position(ccxt_sym, strategy, sym, direction, 'MR', 2.0,
                   entry_price, sl_price, tp_price,
                   V2_MR_LEVERAGE, risk_pct)


def mr_symbol_loop(sym: str, cache: CandleCache):
    """Module B 1D 심볼 루프."""
    ccxt_sym   = sym.replace('USDT', '/USDT')
    last_1d_ts = None
    log(f'[MR] {sym} 루프 시작')

    while True:
        if V2_KILL_SWITCH.exists():
            break
        try:
            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(PRICE_POLL_SEC); continue

            ohlcv_1d  = cache.get(_get_ex(), ccxt_sym, '1d', CANDLE_LIMIT_1D)
            if not ohlcv_1d:
                time.sleep(PRICE_POLL_SEC); continue

            df         = calc_mr_indicators(ohlcv_1d)
            cur_1d_ts  = ohlcv_1d[-1][0]

            # 포지션 감시 (매 틱)
            for strat in ('V2_MR_LONG', 'V2_MR_SHORT'):
                pos = load_position(strat, sym)
                if pos:
                    manage_mr_position(ccxt_sym, sym, pos, price)

            # 새 1D 캔들 확인 후 신호 체크
            if cur_1d_ts != last_1d_ts and last_1d_ts is not None:
                check_mr_signal_and_enter(sym, df)
            last_1d_ts = cur_1d_ts

        except Exception as e:
            log(f'[MR] {sym} 루프 오류: {e}', 'error')
        time.sleep(PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  14-C. Module C — 15m 브레이크아웃 모멘텀
# ══════════════════════════════════════════════════════════════

def calc_mc_indicators(ohlcv: list) -> pd.DataFrame:
    """Module C 지표: ATR14 + 거래량 이동평균(20)."""
    df = _ohlcv_to_df(ohlcv)
    df['atr']     = _calc_atr(df, V2_MC_ATR_PERIOD)
    df['vol_avg'] = df['volume'].rolling(V2_MC_VOL_LB).mean()
    return df


def _limit_only_order(ccxt_sym: str, side: str, qty: float,
                       price: float, timeout_sec: int) -> tuple:
    """
    지정가 주문 후 timeout_sec 내 미체결 시 취소 — 시장가 폴백 없음.
    Returns: (ok, avg_price, filled_qty, note)
    부분 체결 후 타임아웃 시에도 체결된 수량을 반환하여 고아 포지션 방지.
    """
    ex = _get_ex()
    try:
        order = ex.create_limit_order(ccxt_sym, side, qty, price)
        oid   = order['id']
    except Exception as e:
        return False, 0.0, 0.0, f'지정가 주문 실패: {e}'

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(3)
        try:
            o      = ex.fetch_order(oid, ccxt_sym)
            status = o.get('status', '')
            filled = float(o.get('filled', 0) or 0)
            if status == 'closed' and filled >= qty * 0.99:
                return True, float(o.get('average', price)), filled, '지정가 완전체결'
        except Exception:
            pass

    # 타임아웃 → 취소 요청
    try:
        ex.cancel_order(oid, ccxt_sym)
    except Exception:
        pass

    # 취소 후 최종 부분 체결량 확인 — 고아 포지션 방지 핵심
    try:
        o      = ex.fetch_order(oid, ccxt_sym)
        filled = float(o.get('filled', 0) or 0)
        if filled > 0:
            return True, float(o.get('average', price)), filled, f'부분체결({filled:.4f})'
    except Exception as e:
        log(f'[MC Exec] 취소 후 상태 확인 실패: {e}', 'warning')

    return False, 0.0, 0.0, '지정가 타임아웃 — 시그널 스킵'


def mc_enter_position(ccxt_sym: str, strategy: str, sym: str,
                       direction: str, entry_sig: float,
                       sl: float, tp: float,
                       leverage: int, risk_pct: float) -> bool:
    """
    Module C 전용 진입 함수.
    enter_position()과 동일한 리스크 스케일링을 적용하되,
    주문 실행은 _limit_only_order()만 사용 (시장가 폴백 없음).
    """
    ex = _get_ex()

    with _shared_lock:
        equity = _shared['equity']
        paused = _shared['paused']
    if paused or equity <= 0:
        return False

    regime = get_cached_regime()
    if regime.regime == REGIME_CRISIS:
        return False

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        log(f'[MC 진입차단] {sym}: 최대 포지션 수 도달')
        return False

    var_ok, var_reason = check_portfolio_var()
    if not var_ok:
        log(f'[MC 진입차단] {sym}: {var_reason}')
        return False

    # 리스크 스케일링 (A/B와 동일 로직)
    regime_scale          = get_risk_scale()
    kelly_risk, kelly_note = get_kelly_scale(strategy, risk_pct)
    ratchet_scale         = get_ratchet_scale()
    corr_scale            = get_corr_scale(sym, direction)
    vol_scale             = get_vol_parity_scale(sym)
    final_risk            = kelly_risk * regime_scale * ratchet_scale * corr_scale * vol_scale
    final_risk            = max(0.001, min(0.05, final_risk))

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
        log(f'[MC 진입차단] {sym}: 수량 {qty} < 최소 {min_qty}')
        return False

    # 가용 마진 사전 체크 — -2019 Margin insufficient 방지
    try:
        bal          = ex.fetch_balance({'type': 'future'})
        free_margin  = float(bal.get('USDT', {}).get('free', 0) or 0)
        notional     = qty * entry_sig
        required_margin = notional / leverage * 1.05  # 5% 버퍼
        if free_margin < required_margin:
            log(f'[MC 진입차단] {sym}: 가용 마진 부족 '
                f'(필요=${required_margin:.1f}, 가용=${free_margin:.1f})')
            return False
    except Exception as e:
        log(f'[MC] 마진 체크 실패 ({e}) — 진입 계속')

    set_leverage(ccxt_sym, leverage)

    side = 'buy' if direction == 'LONG' else 'sell'
    ok, fill, filled_qty, note = _limit_only_order(ccxt_sym, side, qty, entry_sig, V2_MC_LIMIT_SEC)

    if not ok or fill <= 0 or filled_qty <= 0:
        log(f'[MC 진입스킵] {sym} {strategy}: {note}')
        return False

    # fill 기준 SL/TP 재계산 (슬리피지 흡수)
    sl_dist = abs(entry_sig - sl)
    if direction == 'LONG':
        sl_final = fill - sl_dist
        tp_final = fill + sl_dist * V2_MC_RR
    else:
        sl_final = fill + sl_dist
        tp_final = fill - sl_dist * V2_MC_RR

    # SL 주문·DB 저장 모두 실제 체결 수량(filled_qty) 기준 — 고아 포지션 방지
    pos_side    = 'long' if direction == 'LONG' else 'short'
    sl_order_id = place_exchange_sl(ccxt_sym, pos_side, filled_qty, sl_final)
    save_position(strategy, sym, direction, fill, sl_final, tp_final,
                  filled_qty, leverage, risk_usd, 'MC', V2_MC_RR, sl_order_id)

    regime_name = regime.regime
    tg(f"[{strategy}] {sym} {direction} 진입\n"
       f"  모드:MC RR:{V2_MC_RR:.1f} {leverage}x\n"
       f"  진입:{fill:,.4f} SL:{sl_final:,.4f} TP:{tp_final:,.4f}\n"
       f"  리스크:${risk_usd:.1f}({final_risk*100:.2f}%) {regime_name}\n"
       f"  {kelly_note}")
    log(f'[MC 진입] {sym} {strategy} {direction} fill={fill:.4f} SL={sl_final:.4f} TP={tp_final:.4f}')
    return True


def manage_mc_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    """Module C 포지션 감시: SL/TP 확인 (BEP 없음 — 단기 타임프레임)."""
    direction = pos['direction']
    if direction == 'LONG':
        if price <= pos['sl']:
            exit_position(ccxt_sym, pos, price, 'SL')
        elif price >= pos['tp']:
            exit_position(ccxt_sym, pos, price, 'TP')
    else:
        if price >= pos['sl']:
            exit_position(ccxt_sym, pos, price, 'SL')
        elif price <= pos['tp']:
            exit_position(ccxt_sym, pos, price, 'TP')


def check_mc_signal_and_enter(sym: str, df_15m: pd.DataFrame, htf_ema: float):
    """
    Module C 신호 감지 후 진입 시도.
    신호: 전 캔들 고/저점 돌파 + 거래량 스파이크 + 1H EMA 방향 일치
    레짐: TRENDING_UP→LONG만, TRENDING_DOWN→SHORT만,
          WEAK_TREND→양방향(HTF EMA 필터), RANGING/CRISIS→스킵
    """
    strategy_long  = 'V2_MC_LONG'
    strategy_short = 'V2_MC_SHORT'
    ccxt_sym = sym.replace('USDT', '/USDT')

    # 이미 포지션 있으면 스킵
    if (load_position(strategy_long,  sym) is not None or
            load_position(strategy_short, sym) is not None):
        return

    # 쿨다운 체크
    with _shared_lock:
        cd = _shared['mc_cooldowns'].get(sym, 0)
    if cd > 0:
        log(f'[MC 스킵] {sym} 쿨다운 {cd}봉 남음')
        return

    # 레짐 체크
    regime = get_cached_regime().regime
    if regime in (REGIME_CRISIS, REGIME_RANGING, 'UNKNOWN'):
        log(f'[MC 스킵] {sym} 레짐={regime} (진입불가)')
        return

    if len(df_15m) < V2_MC_VOL_LB + 3:
        return

    row      = df_15m.iloc[-1]
    row_prev = df_15m.iloc[-2]

    atr       = float(row['atr'])          if not pd.isna(row['atr'])          else 0.0
    # 거래량은 완성된 직전 캔들 기준 — 형성 중인 현재 캔들은 경과 시간에 따라
    # 절대량이 낮아 스파이크 판정 불가 (Volume Trap 방지)
    prev_vol  = float(row_prev['volume'])
    vol_avg   = float(row_prev['vol_avg']) if not pd.isna(row_prev['vol_avg']) else 0.0
    prev_high = float(row_prev['high'])
    prev_low  = float(row_prev['low'])
    cur_close = float(row['close'])

    if atr <= 0 or vol_avg <= 0:
        return

    vol_ratio = prev_vol / vol_avg if vol_avg > 0 else 0.0
    htf_up    = htf_ema <= 0 or cur_close > htf_ema
    htf_down  = htf_ema <= 0 or cur_close < htf_ema
    direction      = None
    breakout_level = 0.0
    sig_type       = 'BREAKOUT'

    # ── 신호 A: 즉시 브레이크아웃 (거래량 스파이크 필요) ─────────
    if prev_vol > vol_avg * V2_MC_VOL_MULT:
        long_break  = cur_close > prev_high and htf_up
        short_break = cur_close < prev_low  and htf_down
        if regime == REGIME_TRENDING_UP:   short_break = False
        elif regime == REGIME_TRENDING_DOWN: long_break = False

        if long_break:
            direction = 'LONG';  breakout_level = prev_high
        elif short_break:
            direction = 'SHORT'; breakout_level = prev_low

    # ── 신호 B: 브레이크아웃 리테스트 (2~4봉 전 돌파 레벨 복귀) ─
    if direction is None and len(df_15m) >= 6:
        for lb in range(2, 5):
            r_break = df_15m.iloc[-lb]
            r_base  = df_15m.iloc[-lb - 1]
            bo_vol    = float(r_break['volume'])
            base_vavg = float(r_base['vol_avg']) if not pd.isna(r_base['vol_avg']) else 0.0
            if base_vavg <= 0 or bo_vol <= base_vavg * V2_MC_VOL_MULT:
                continue

            base_high = float(r_base['high'])
            base_low  = float(r_base['low'])

            if (float(r_break['close']) > base_high and htf_up and
                    regime != REGIME_TRENDING_DOWN and
                    base_high * 0.998 <= cur_close <= base_high * 1.004):
                direction = 'LONG'; breakout_level = base_high; sig_type = 'RETEST'
                break

            if (float(r_break['close']) < base_low and htf_down and
                    regime != REGIME_TRENDING_UP and
                    base_low * 0.996 <= cur_close <= base_low * 1.002):
                direction = 'SHORT'; breakout_level = base_low; sig_type = 'RETEST'
                break

    if direction is None:
        htf_str = f'HTF={htf_ema:.2f}' if htf_ema > 0 else 'HTF=없음'
        log(f'[MC 스킵] {sym} 볼륨({vol_ratio:.2f}x) 브레이크아웃/리테스트 없음 '
            f'(종={cur_close:.4f}) {htf_str} | 레짐={regime}')
        return

    # 추격 거리 필터
    if sig_type == 'BREAKOUT':
        if direction == 'LONG' and cur_close > breakout_level * (1 + V2_MC_CHASE_LIMIT):
            log(f'[MC] {sym} LONG 추격 초과 ({cur_close:.4f} > {breakout_level*(1+V2_MC_CHASE_LIMIT):.4f})')
            return
        if direction == 'SHORT' and cur_close < breakout_level * (1 - V2_MC_CHASE_LIMIT):
            log(f'[MC] {sym} SHORT 추격 초과 ({cur_close:.4f} < {breakout_level*(1-V2_MC_CHASE_LIMIT):.4f})')
            return

    # SL/TP 계산
    sl_mult = V2_MC_ATR_SL if sig_type == 'BREAKOUT' else V2_MC_ATR_SL * 0.8
    if direction == 'LONG':
        entry_sig = breakout_level * (1 + LIMIT_OFFSET_PCT)
        sl_price  = entry_sig - atr * sl_mult
        tp_price  = entry_sig + atr * sl_mult * V2_MC_RR
    else:
        entry_sig = breakout_level * (1 - LIMIT_OFFSET_PCT)
        sl_price  = entry_sig + atr * sl_mult
        tp_price  = entry_sig - atr * sl_mult * V2_MC_RR

    leverage = V2_MC_LEVERAGE.get(sym, 3)
    risk_pct = V2_MC_RISK_PCT.get(sym, 0.008)
    strategy = strategy_long if direction == 'LONG' else strategy_short
    log(f'[MC 신호] {sym} {direction} {sig_type} 레벨={breakout_level:.4f} | 레짐={regime}')

    mc_enter_position(ccxt_sym, strategy, sym, direction,
                      entry_sig, sl_price, tp_price, leverage, risk_pct)


def mc_symbol_loop(sym: str, cache: CandleCache):
    """Module C 15m 심볼 루프."""
    ccxt_sym    = sym.replace('USDT', '/USDT')
    last_15m_ts = None
    log(f'[MC] {sym} 루프 시작')

    while True:
        if V2_KILL_SWITCH.exists():
            break
        try:
            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(PRICE_POLL_SEC); continue

            ohlcv_15m = cache.get(_get_ex(), ccxt_sym, '15m', CANDLE_LIMIT_15M)
            ohlcv_1h  = cache.get(_get_ex(), ccxt_sym, '1h',  50)
            if not ohlcv_15m or not ohlcv_1h:
                time.sleep(PRICE_POLL_SEC); continue

            df_15m     = calc_mc_indicators(ohlcv_15m)
            cur_15m_ts = ohlcv_15m[-1][0]

            # 1H EMA21 계산 (HTF 방향 필터)
            df_1h   = _ohlcv_to_df(ohlcv_1h)
            htf_ema = float(
                df_1h['close'].ewm(span=V2_MC_HTF_EMA, adjust=False).mean().iloc[-1]
            ) if len(df_1h) >= V2_MC_HTF_EMA else 0.0

            # 포지션 감시 (매 틱)
            for strat in ('V2_MC_LONG', 'V2_MC_SHORT'):
                pos = load_position(strat, sym)
                if pos:
                    manage_mc_position(ccxt_sym, sym, pos, price)

            # 새 15m 캔들 확인 → 쿨다운 감소 + 신호 체크
            if cur_15m_ts != last_15m_ts and last_15m_ts is not None:
                with _shared_lock:
                    cd = _shared['mc_cooldowns'].get(sym, 0)
                    if cd > 0:
                        _shared['mc_cooldowns'][sym] = cd - 1
                check_mc_signal_and_enter(sym, df_15m, htf_ema)
            last_15m_ts = cur_15m_ts

        except Exception as e:
            log(f'[MC] {sym} 루프 오류: {e}', 'error')
        time.sleep(PRICE_POLL_SEC)


# ══════════════════════════════════════════════════════════════
#  14. 상관계수 + 변동성 갱신
# ══════════════════════════════════════════════════════════════

def update_corr_vol(cache: CandleCache):
    try:
        closes_dict = {}
        for sym in ALL_V2_SYMBOLS:
            ccxt_sym = sym.replace('USDT', '/USDT')
            ohlcv    = cache.get(_get_ex(), ccxt_sym, '1d', 35)
            if len(ohlcv) < 30:
                continue
            c = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])['c'].astype(float)
            closes_dict[sym] = c.pct_change().dropna().iloc[-30:]

        vol_cache   = {s: float(r.std()) for s, r in closes_dict.items()}
        corr_matrix = {}
        syms = list(closes_dict.keys())
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s1, s2 = syms[i], syms[j]
                if len(closes_dict[s1]) == len(closes_dict[s2]):
                    c   = float(closes_dict[s1].corr(closes_dict[s2]))
                    key = tuple(sorted([s1.replace('USDT', ''), s2.replace('USDT', '')]))
                    corr_matrix[key] = max(0, c)

        with _shared_lock:
            _shared['corr_matrix']  = corr_matrix
            _shared['vol_cache']    = vol_cache
            _shared['corr_updated'] = time.time()
        log(f'[Corr/Vol] 갱신: {len(corr_matrix)}쌍')
    except Exception as e:
        log(f'[Corr/Vol] 오류: {e}', 'error')


# ══════════════════════════════════════════════════════════════
#  14-D. Module D — 펀딩비 차익 (Funding Rate Harvest)
# ══════════════════════════════════════════════════════════════

def md_enter_position(sym: str) -> bool:
    """
    펀딩비 > V2_MD_FUNDING_ENTER 이면 SHORT 진입.
    지정가 전용(Maker 0.02%) — 수수료 최소화.
    """
    strategy = 'V2_MD_SHORT'
    ccxt_sym = sym.replace('USDT', '/USDT')
    ex = _get_ex()

    if load_position(strategy, sym) is not None:
        return False

    # 쿨다운 (직전 청산 후 1시간)
    with _shared_lock:
        cd_until = _shared['md_cooldowns'].get(sym, 0.0)
    if time.time() < cd_until:
        return False

    with _shared_lock:
        equity = _shared['equity']
        paused = _shared['paused']
    if paused or equity <= 0:
        return False

    if get_cached_regime().regime == REGIME_CRISIS:
        return False

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return False

    var_ok, var_reason = check_portfolio_var()
    if not var_ok:
        log(f'[MD 진입차단] {sym}: {var_reason}')
        return False

    # 현재 펀딩비 확인
    fr = get_funding_rate(ccxt_sym)
    if fr < V2_MD_FUNDING_ENTER:
        log(f'[MD 스킵] {sym} 펀딩비 {fr:.4%} < 임계값 {V2_MD_FUNDING_ENTER:.4%}')
        return False

    entry_price = get_price(ccxt_sym)
    if entry_price <= 0:
        return False

    # ATR 계산 (1H 캔들 기반)
    try:
        ohlcv = ex.fetch_ohlcv(ccxt_sym, '1h', limit=50)
        df_h  = _ohlcv_to_df(ohlcv)
        atr   = float(_calc_atr(df_h, V2_MD_ATR_PERIOD).iloc[-1])
    except Exception:
        atr = entry_price * 0.01  # fallback: 1%

    if atr <= 0:
        return False

    sl_price = entry_price + atr * V2_MD_ATR_SL   # SHORT → SL 위
    tp_price = entry_price - atr * V2_MD_ATR_SL * 3.0  # TP는 참고용 (실제 청산은 펀딩비 기준)

    # 리스크 스케일링
    regime_scale = get_risk_scale()
    ratchet_scale = get_ratchet_scale()
    final_risk = V2_MD_RISK_PCT * regime_scale * ratchet_scale
    final_risk = max(0.001, min(0.03, final_risk))

    qty_raw, risk_usd = calc_raw_qty(equity, entry_price, sl_price, final_risk)
    if qty_raw <= 0:
        return False

    try:
        qty = float(ex.amount_to_precision(ccxt_sym, qty_raw))
    except Exception:
        qty_prec, _ = get_qty_precision(ccxt_sym)
        qty = round(qty_raw, qty_prec)

    _, min_qty = get_qty_precision(ccxt_sym)
    if qty < min_qty:
        return False

    # 가용 마진 사전 체크
    try:
        bal = ex.fetch_balance({'type': 'future'})
        free_margin = float(bal.get('USDT', {}).get('free', 0) or 0)
        required_margin = qty * entry_price / V2_MD_LEVERAGE * 1.05
        if free_margin < required_margin:
            log(f'[MD 진입차단] {sym}: 가용 마진 부족 '
                f'(필요=${required_margin:.1f}, 가용=${free_margin:.1f})')
            return False
    except Exception as e:
        log(f'[MD] 마진 체크 실패 ({e}) — 진입 계속')

    set_leverage(ccxt_sym, V2_MD_LEVERAGE)

    ok, fill, filled_qty, note = _limit_only_order(
        ccxt_sym, 'sell', qty, entry_price, V2_MD_LIMIT_SEC)

    if not ok or fill <= 0 or filled_qty <= 0:
        log(f'[MD 진입스킵] {sym}: {note}')
        return False

    sl_final = fill + atr * V2_MD_ATR_SL
    tp_final = fill - atr * V2_MD_ATR_SL * 3.0

    sl_order_id = place_exchange_sl(ccxt_sym, 'short', filled_qty, sl_final)
    save_position(strategy, sym, 'SHORT', fill, sl_final, tp_final,
                  filled_qty, V2_MD_LEVERAGE, risk_usd, 'MD', 0.0, sl_order_id)

    tg(f'[V2_MD_SHORT] {sym} SHORT 진입 (펀딩비 수집)\n'
       f'  펀딩비: {fr:.4%}/8h  레버리지: {V2_MD_LEVERAGE}x\n'
       f'  진입: {fill:,.4f}  SL: {sl_final:,.4f}\n'
       f'  리스크: ${risk_usd:.1f}  최대보유: {V2_MD_MAX_HOURS}h')
    log(f'[MD 진입] {sym} SHORT fill={fill:.4f} SL={sl_final:.4f} FR={fr:.4%}')
    return True


def manage_md_position(ccxt_sym: str, sym: str, pos: dict, price: float):
    """
    Module D 포지션 감시.
    청산 조건: SL 히트 / 펀딩비 정상화 / 48h 초과
    """
    direction = pos['direction']  # 항상 SHORT

    # SL 히트
    if direction == 'SHORT' and price >= pos['sl']:
        exit_position(ccxt_sym, pos, price, 'SL')
        with _shared_lock:
            _shared['md_cooldowns'][sym] = time.time() + 3600  # 1시간 쿨다운
        return

    # 보유 시간 초과
    entry_ts = pos.get('entry_ts', '')
    if entry_ts:
        try:
            et = datetime.fromisoformat(entry_ts.replace('Z', '+00:00'))
            held_hours = (datetime.now(timezone.utc) - et).total_seconds() / 3600
            if held_hours >= V2_MD_MAX_HOURS:
                log(f'[MD] {sym} 최대 보유 {V2_MD_MAX_HOURS}h 초과 → 청산')
                exit_position(ccxt_sym, pos, price, 'TIME')
                with _shared_lock:
                    _shared['md_cooldowns'][sym] = time.time() + 3600
                return
        except Exception:
            pass

    # 펀딩비 정상화 체크
    fr = get_funding_rate(ccxt_sym)
    if fr < V2_MD_FUNDING_EXIT:
        log(f'[MD] {sym} 펀딩비 정상화 ({fr:.4%} < {V2_MD_FUNDING_EXIT:.4%}) → 청산')
        exit_position(ccxt_sym, pos, price, 'FUNDING_NORM')
        with _shared_lock:
            _shared['md_cooldowns'][sym] = time.time() + 3600
        return

    log(f'[MD 유지] {sym} SHORT price={price:.4f} FR={fr:.4%} SL={pos["sl"]:.4f}')


def md_symbol_loop(sym: str):
    """Module D 심볼 루프 — 5분마다 펀딩비 확인."""
    ccxt_sym = sym.replace('USDT', '/USDT')
    log(f'[MD] {sym} 루프 시작')

    while True:
        if V2_KILL_SWITCH.exists():
            break
        try:
            price = get_price(ccxt_sym)
            if price <= 0:
                time.sleep(V2_MD_CHECK_SEC)
                continue

            pos = load_position('V2_MD_SHORT', sym)
            if pos:
                manage_md_position(ccxt_sym, sym, pos, price)
            else:
                fr = get_funding_rate(ccxt_sym)
                if fr >= V2_MD_FUNDING_ENTER:
                    log(f'[MD] {sym} 펀딩비 {fr:.4%} — 진입 시도')
                    md_enter_position(sym)
                else:
                    log(f'[MD 스킵] {sym} 펀딩비 {fr:.4%} < 임계값 {V2_MD_FUNDING_ENTER:.4%}')

        except Exception as e:
            log(f'[MD] {sym} 오류: {e}', 'error')

        time.sleep(V2_MD_CHECK_SEC)


def position_reconcile_loop():
    """
    5분마다 SQLite DB 포지션 ↔ 바이낸스 실제 포지션 비교.
    수동 청산 / 강제 청산 등으로 거래소에서 사라진 포지션을 DB에서 자동 제거.
    """
    log('[정합성] 포지션 정합성 검사 루프 시작')
    while True:
        if V2_KILL_SWITCH.exists():
            break
        try:
            db_positions = load_all_positions()
            if db_positions:
                ex = _get_ex()
                for pos in db_positions:
                    sym      = pos['symbol']
                    strategy = pos['strategy']
                    direction= pos['direction']
                    ccxt_sym = sym.replace('USDT', '/USDT')
                    try:
                        # 💡 패치: 옵션 마켓 버그 우회
                        positions = ex.fetch_positions([ccxt_sym])
                        ex_pos    = positions[0] if positions else {}
                        contracts = abs(float(ex_pos.get('contracts', 0) or 0))
                        ex_side   = (ex_pos.get('side') or '').lower()
                        expected  = 'long' if direction == 'LONG' else 'short'

                        if contracts == 0 or ex_side != expected:
                            log(f'[정합성] {sym} {strategy} {direction} — 거래소 포지션 없음 → 수동청산 처리')
                            cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))
                            entry    = float(pos['entry_price'])
                            qty      = float(pos['qty'])
                            sl_dist  = abs(entry - float(pos['sl']))

                            # ── 실제 체결가 조회 (fetch_my_trades) ──────────────
                            close_price = get_price(ccxt_sym)  # fallback: 현재가
                            price_source = '추정가'
                            try:
                                entry_ts_str = str(pos['entry_ts']).replace('Z', '+00:00')
                                entry_dt     = datetime.fromisoformat(entry_ts_str)
                                entry_ms     = int(entry_dt.timestamp() * 1000)
                                close_side   = 'sell' if direction == 'LONG' else 'buy'
                                my_trades    = ex.fetch_my_trades(ccxt_sym, limit=20)
                                close_trades = [t for t in my_trades
                                                if t['timestamp'] > entry_ms
                                                and (t.get('side') or '').lower() == close_side]
                                if close_trades:
                                    total_qty  = sum(abs(float(t['amount'])) for t in close_trades)
                                    total_cost = sum(abs(float(t['amount'])) * float(t['price'])
                                                     for t in close_trades)
                                    if total_qty > 0:
                                        close_price  = total_cost / total_qty
                                        price_source = '실제체결가'
                            except Exception as te:
                                log(f'[정합성] {sym} 체결가 조회 실패 ({te}) — 현재가로 대체')

                            try:
                                entry_dt_c = datetime.fromisoformat(
                                    str(pos['entry_ts']).replace('Z', '+00:00'))
                                hold_h = (datetime.now(timezone.utc) - entry_dt_c).total_seconds() / 3600
                            except Exception:
                                hold_h = 0

                            pnl_pct = ((close_price - entry) / entry if direction == 'LONG'
                                       else (entry - close_price) / entry)
                            pnl_usd = pnl_pct * qty * entry
                            pnl_r   = pnl_usd / (qty * sl_dist) if sl_dist > 0 else 0

                            log_trade(strategy, sym, direction, pos.get('mode', ''), entry, close_price,
                                      qty, int(pos['leverage']), pnl_usd, pnl_r,
                                      float(pos.get('rr', 2.0)), hold_h, 'MANUAL',
                                      str(pos['entry_ts']), get_cached_regime().regime)
                            delete_position(strategy, sym)
                            tg(f'⚠️ [{strategy}] {sym} {direction} 수동청산 감지\n'
                               f'  청산가: {close_price:,.4f} ({price_source})\n'
                               f'  PnL: ${pnl_usd:+.2f} ({pnl_r:+.2f}R)\n'
                               f'  DB 자동 정리 완료 — 신규 진입 재개')
                    except Exception as e:
                        log(f'[정합성] {sym} 조회 실패: {e}', 'warning')
        except Exception as e:
            log(f'[정합성] 루프 오류: {e}', 'error')
        time.sleep(300)  # 5분마다 검사


def corr_vol_loop(cache: CandleCache):
    while True:
        update_corr_vol(cache)
        time.sleep(3600)


# ══════════════════════════════════════════════════════════════
#  15. 일일 리셋 + 브리핑
# ══════════════════════════════════════════════════════════════

def daily_reset_loop():
    last_day      = None
    last_heartbeat = None
    while True:
        now = datetime.now(timezone.utc)

        # ── 09:00 UTC 하트비트 (하루 1회) ──────────────────────
        if now.hour == 9 and now.minute < 5 and now.date() != last_heartbeat:
            with _shared_lock:
                eq     = _shared['equity']
                paused = _shared['paused']
            positions = load_all_positions()
            regime    = get_cached_regime().regime
            fr_lines  = []
            for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']:
                try:
                    fr = get_funding_rate(s.replace('USDT', '/USDT'))
                    fr_lines.append(f"  {s}: {fr:.4%}")
                except Exception:
                    pass
            fr_str = '\n'.join(fr_lines) if fr_lines else '  조회 실패'
            status = '⏸ 일시정지' if paused else '✅ 정상 가동'
            tg(
                f'💓 ATLAS v2 일일 하트비트\n'
                f'  상태: {status}\n'
                f'  레짐: {regime}\n'
                f'  잔고: ${eq:,.2f}\n'
                f'  열린포지션: {len(positions)}개\n'
                f'펀딩비 현황:\n{fr_str}'
            )
            last_heartbeat = now.date()

        if now.date() != last_day and now.hour == 0 and now.minute < 5:
            with _shared_lock:
                eq = _shared['equity']
                _shared['day_eq_start'] = eq
                if _shared['paused']:
                    _shared['paused'] = False
                    tg('[ATLAS v2] 일일 손실한도 자동 해제')

            rows = _db_fetchall("""
                SELECT strategy, COUNT(*), SUM(pnl_usd),
                       SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END)
                FROM trades
                WHERE DATE(exit_ts) = DATE('now', '-1 day')
                GROUP BY strategy
            """)
            if rows:
                total = sum(r[2] or 0 for r in rows)
                lines = ['[ATLAS v2 일일 성과]']
                for r in rows:
                    strat, cnt, pnl, wins = r
                    wr = (wins or 0) / cnt * 100 if cnt else 0
                    lines.append(f"  {strat}: {cnt}건 WR{wr:.0f}% ${pnl:+.1f}")
                lines.append(f"  합계:${total:+.1f}  잔고:${eq:,.0f}  {get_cached_regime().regime}")
                tg('\n'.join(lines))
            last_day = now.date()
        time.sleep(60)


# ══════════════════════════════════════════════════════════════
#  16. Telegram 명령 처리
# ══════════════════════════════════════════════════════════════

def _handle_tg_cmd(text: str) -> str:
    cmd = text.strip().lower().split()[0]

    if cmd == '/status':
        positions = load_all_positions()
        if not positions:
            return '현재 열린 포지션 없음'
        lines = [f'[ATLAS v2] 포지션 현황 ({len(positions)}개)']
        for p in positions:
            arrow = 'L' if p['direction'] == 'LONG' else 'S'
            hold  = _mr_hold_days(p) if p['strategy'].startswith('V2_MR') else 0
            lines.append(
                f"  {p['strategy']} {p['symbol']} [{arrow}]\n"
                f"    진입:{p['entry_price']:,.4f} SL:{p['sl']:,.4f} TP:{p['tp']:,.4f}"
                + (f" {hold}일" if hold else ""))
        return '\n'.join(lines)

    if cmd == '/stats':
        rows = _db_fetchall("""
            SELECT strategy, COUNT(*), SUM(pnl_usd),
                   SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END), AVG(pnl_r)
            FROM trades GROUP BY strategy
        """)
        if not rows:
            return '거래 이력 없음'
        lines = ['[ATLAS v2] 전략별 통계']
        for r in rows:
            strat, cnt, total, wins, avg_r = r
            wr = (wins or 0) / cnt * 100 if cnt else 0
            lines.append(f"  {strat}: {cnt}건 WR{wr:.0f}% ${total:+.0f} avgR={avg_r or 0:.2f}")
        return '\n'.join(lines)

    if cmd == '/regime':
        return get_cached_regime().summary()

    if cmd == '/equity':
        with _shared_lock:
            eq = _shared['equity']
        return f'잔고: ${eq:,.2f}'

    if cmd == '/pause':
        with _shared_lock:
            _shared['paused'] = True
        return '신규 진입 일시정지'

    if cmd == '/resume':
        with _shared_lock:
            _shared['paused'] = False
        return '신규 진입 재개'

    if cmd == '/stop':
        V2_KILL_SWITCH.touch()
        return 'Kill Switch 활성화. 봇 종료 중...'

    if cmd == '/ratchet':
        with _shared_lock:
            rat = _shared['ratchet']
        return (f"Ratchet 상태\n"
                f"  Peak: ${rat['peak_equity']:,.0f}\n"
                f"  Scale: {rat['scale']:.0%}")

    if cmd == '/help':
        return ('명령어\n'
                '  /status /stats /regime /equity\n'
                '  /pause /resume /stop /ratchet /help')

    return f'알 수 없는 명령: {cmd}'


def tg_cmd_loop():
    offset = 0

    # 💡 [필수 추가] 루프 시작 전, 쌓여있는 옛날 메시지들을 모두 읽음 처리(Flush)
    try:
        flush_resp = _tg_session.get(
            f'https://api.telegram.org/bot{TG_TOKEN}/getUpdates',
            params={'offset': offset, 'timeout': 5}, timeout=10)
        for u in flush_resp.json().get('result', []):
            offset = u['update_id'] + 1
    except Exception:
        pass

    start_time = time.time()  # 💡 봇 시작 시간 기록

    while True:
        if V2_KILL_SWITCH.exists():
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
                msg_date = msg.get('date', 0)
                
                # 💡 핵심 방어 로직: 봇 시작 시간 이전에 수신된 과거 명령어는 무시
                if msg_date < start_time:
                    continue
                    
                if text.startswith('/') and chat == TG_CHAT_ID:
                    tg(_handle_tg_cmd(text))
                    
        except requests.exceptions.ReadTimeout:
            pass
        except requests.exceptions.ConnectionError as e:
            log(f'[TG Cmd] 연결 오류: {e}', 'warning')
            time.sleep(10)
        except Exception as e:
            log(f'[TG Cmd] 오류: {e}', 'warning')
            time.sleep(10)


# ══════════════════════════════════════════════════════════════
#  17. 포트폴리오 복구 루프 (비정상 포지션 DB 정리)
# ══════════════════════════════════════════════════════════════

def _reconcile_one(ex, pos: dict, tag: str) -> bool:
    """
    DB 포지션 1개를 거래소와 대조. 거래소에 없으면 PnL 기록 후 DB 삭제.
    tag: 로그 접두사 ([시작복구] / [Recovery] / [정합성])
    Returns True if cleaned up.
    """
    sym      = pos['symbol']
    strategy = pos['strategy']
    direction= pos['direction']
    ccxt_sym = sym.replace('USDT', '/USDT')
    try:
        # 💡 패치: 옵션 마켓 버그 우회
        positions = ex.fetch_positions([ccxt_sym])
        ex_pos    = positions[0] if positions else {}
        contracts = abs(float(ex_pos.get('contracts', 0) or 0))
        ex_side   = (ex_pos.get('side') or '').lower()
        expected  = 'long' if direction == 'LONG' else 'short'
        if contracts > 0 and ex_side == expected:
            return False  # 포지션 유효 — 아무것도 안 함
    except Exception as e:
        log(f'[{tag}] {sym} 거래소 조회 실패: {e}', 'warning')
        return False

    # 거래소에 포지션 없음 → 청산 기록 후 DB 삭제
    log(f'[{tag}] {sym} {strategy} {direction} — 거래소 포지션 없음 → 정리', 'warning')
    cancel_exchange_sl(ccxt_sym, pos.get('sl_order_id', ''))

    cur_price    = get_price(ccxt_sym)
    close_price  = cur_price
    price_src    = '추정가'
    try:
        entry_ms  = int(datetime.fromisoformat(
            str(pos['entry_ts']).replace('Z', '+00:00')).timestamp() * 1000)
        close_side = 'sell' if direction == 'LONG' else 'buy'
        my_trades  = ex.fetch_my_trades(ccxt_sym, limit=20)
        ct = [t for t in my_trades if t['timestamp'] > entry_ms
              and (t.get('side') or '').lower() == close_side]
        if ct:
            tq = sum(abs(float(t['amount'])) for t in ct)
            tc = sum(abs(float(t['amount'])) * float(t['price']) for t in ct)
            if tq > 0:
                close_price = tc / tq
                price_src   = '실제체결가'
    except Exception as e:
        log(f'[{tag}] {sym} 체결가 조회 실패 ({e}) — 현재가 사용')

    entry   = float(pos['entry_price'])
    qty     = float(pos['qty'])
    sl_dist = abs(entry - float(pos['sl']))
    pnl_pct = (close_price - entry)/entry if direction == 'LONG' else (entry - close_price)/entry
    pnl_usd = pnl_pct * qty * entry
    pnl_r   = pnl_usd / (qty * sl_dist) if sl_dist > 0 else 0
    try:
        hold_h = (datetime.now(timezone.utc) -
                  datetime.fromisoformat(str(pos['entry_ts']).replace('Z', '+00:00'))
                  ).total_seconds() / 3600
    except Exception:
        hold_h = 0

    log_trade(strategy, sym, direction, pos.get('mode', ''), entry, close_price,
              qty, int(pos['leverage']), pnl_usd, pnl_r,
              float(pos.get('rr', 2.0)), hold_h, 'MANUAL',
              str(pos['entry_ts']), get_cached_regime().regime)
    delete_position(strategy, sym)
    tg(f'⚠️ [{tag}] {sym} {strategy} {direction} 봇 중단 중 청산 감지\n'
       f'  청산가: {close_price:,.4f} ({price_src})\n'
       f'  PnL: ${pnl_usd:+.2f} ({pnl_r:+.2f}R)\n'
       f'  DB 자동 정리 완료')
    return True


def startup_reconcile(ex) -> None:
    """
    봇 시작 직후 동기 실행 — 스레드 기동 전에 포지션 상태 즉시 확인.
    봇이 내려가 있는 동안 청산된 포지션을 감지하고 DB를 정리한다.
    """
    db_pos = load_all_positions()
    if not db_pos:
        log('[시작] 보유 포지션 없음 — 정상 시작')
        return

    log(f'[시작] DB 포지션 {len(db_pos)}개 감지 — 거래소 대조 중...')
    valid = cleaned = 0
    for pos in db_pos:
        if _reconcile_one(ex, pos, '시작복구'):
            cleaned += 1
        else:
            valid   += 1
            log(f'[시작] {pos["symbol"]} {pos["strategy"]} {pos["direction"]} — 유효, 감시 재개')
    log(f'[시작] 포지션 정합성 완료: 유효 {valid}개 / 정리 {cleaned}개')


def recovery_loop():
    """10분마다 DB↔거래소 포지션 대조 — startup_reconcile의 지속 백업."""
    time.sleep(120)  # 시작 후 2분 대기 (startup_reconcile 완료 여유)
    while True:
        try:
            ex = _get_ex()
            for pos in load_all_positions():
                _reconcile_one(ex, pos, 'Recovery')
        except Exception as e:
            log(f'[Recovery] 오류: {e}', 'error')
        time.sleep(600)  # 10분마다


# ══════════════════════════════════════════════════════════════
#  18. Main
# ══════════════════════════════════════════════════════════════

def main():
    log('=' * 50)
    log('  ATLAS v2 -- 라이브 트레이딩 엔진 시작')
    log(f'  Module A (Trend Follow 4H): {V2_MA_SYMBOLS}')
    log(f'  Module B (Mean Reversion 1D): {V2_MR_SYMBOLS}')
    log('=' * 50)

    if V2_KILL_SWITCH.exists():
        V2_KILL_SWITCH.unlink()

    init_db()

    ex = _get_ex()
    log('[Exchange] Binance Futures 연결 중...')
    try:
        ex.load_markets()
        log('[Exchange] 마켓 로드 완료')
    except Exception as e:
        log(f'[Exchange] 마켓 로드 실패: {e}', 'warning')

    # 초기 잔고
    try:
        bal = ex.fetch_balance()
        eq  = float(bal.get('USDT', {}).get('total', 0) or
                    bal.get('total', {}).get('USDT', 0))
        with _shared_lock:
            _shared['equity']                = eq
            _shared['day_eq_start']          = eq
            _shared['ratchet']['peak_equity']= eq
            _shared['ratchet']['recover_anchor'] = eq
        log(f'[잔고] ${eq:,.2f}')
    except Exception as e:
        log(f'[초기화] 잔고 조회 실패: {e}', 'warning')

    cache = CandleCache()

    # 초기 레짐 (메인 스레드 ex 사용)
    try:
        state = update_regime(ex, cache)
        log(f'[초기 장세] {state.summary()}')
    except Exception as e:
        log(f'[초기화] 레짐 조회 실패: {e}', 'warning')

    tg(f'*ATLAS v2 봇 시작*\n{get_cached_regime().summary()}\n'
       f'MA: {V2_MA_SYMBOLS}\nMR: {V2_MR_SYMBOLS}\n'
       f'MC: {V2_MC_SYMBOLS}\nMD: {V2_MD_SYMBOLS}')

    # 스레드 기동
    threads = []

    def _t(target, name, *args):
        t = threading.Thread(target=target, args=args, name=name, daemon=True)
        t.start()
        threads.append(t)

    _t(balance_poller,   'BalancePoller')
    _t(regime_loop,      'RegimeLoop',   ex, cache)
    _t(corr_vol_loop,    'CorrVolLoop',  cache)
    _t(daily_reset_loop, 'DailyReset')
    _t(tg_cmd_loop,      'TGCmdLoop')
    _t(recovery_loop,         'RecoveryLoop')
    _t(position_reconcile_loop, 'ReconcileLoop')

    for sym in V2_MA_SYMBOLS:
        _t(ma_symbol_loop, f'MA_{sym}', sym, cache)

    for sym in V2_MR_SYMBOLS:
        _t(mr_symbol_loop, f'MR_{sym}', sym, cache)

    for sym in V2_MC_SYMBOLS:
        _t(mc_symbol_loop, f'MC_{sym}', sym, cache)

    for sym in V2_MD_SYMBOLS:
        _t(md_symbol_loop, f'MD_{sym}', sym)

    log(f'[ATLAS v2] 스레드 {len(threads)}개 기동')
    log(f'  Module A 4H : {V2_MA_SYMBOLS}')
    log(f'  Module B 1D : {V2_MR_SYMBOLS}')
    log(f'  Module C 15m: {V2_MC_SYMBOLS}')
    log(f'  Module D FR : {V2_MD_SYMBOLS}')

    try:
        while True:
            if V2_KILL_SWITCH.exists():
                log('[ATLAS v2] Kill Switch -> 종료')
                tg('*ATLAS v2 봇 종료*')
                break
            time.sleep(10)
    except KeyboardInterrupt:
        tg('*ATLAS v2 봇 종료* (수동)')


if __name__ == '__main__':
    main()
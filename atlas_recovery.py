"""
ATLAS — 포지션 복구·검증 모듈
================================
거래소 실제 포지션과 DB 기록을 주기적으로 대조하여:
  1. 고아 포지션  : 거래소에는 있지만 DB에 없음 → Telegram 경고
  2. 유령 DB 기록 : DB에는 있지만 거래소에 없음 → Telegram 경고 + 선택적 DB 정리
  3. 수량 불일치  : DB qty와 거래소 contracts가 크게 다름 → 경고

통합 방법 (atlas_main.py):
  from atlas_recovery import recovery_loop
  threading.Thread(target=recovery_loop, name='RecoveryLoop', daemon=True).start()

독립 실행:
  python atlas_recovery.py
"""

import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'RECOVERY_STANDALONE')

sys.path.insert(0, str(Path(__file__).parent))
from atlas_config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    TG_TOKEN, TG_CHAT_ID,
    DB_FILE,
)

_log = logging.getLogger('atlas.recovery')

# 복구 루프 주기 (초)
RECOVERY_INTERVAL = 300   # 5분마다 점검

# DB qty와 거래소 qty의 허용 오차 비율
QTY_TOLERANCE = 0.05      # 5% 이상 차이 시 경고


# ══════════════════════════════════════════════════════════════
#  거래소 인스턴스 (recovery 전용 CCXT)
# ══════════════════════════════════════════════════════════════

def _make_exchange():
    """recovery 스레드 전용 CCXT 인스턴스."""
    import ccxt
    return ccxt.binance({
        'apiKey':  BINANCE_API_KEY,
        'secret':  BINANCE_API_SECRET,
        'options': {'defaultType': 'future'},
    })


# ══════════════════════════════════════════════════════════════
#  DB 접근 (atlas_main 의존 없이 독립 구현)
# ══════════════════════════════════════════════════════════════

_db_lock = threading.Lock()

def _db_load_positions() -> list:
    """DB에서 모든 열린 포지션 로드."""
    if not DB_FILE.exists():
        return []
    cols = ['id', 'strategy', 'symbol', 'direction', 'entry_price',
            'sl', 'tp', 'qty', 'leverage', 'risk_usd', 'peak',
            'sl_order_id', 'entry_ts', 'mode', 'rr']
    try:
        with _db_lock:
            con  = sqlite3.connect(DB_FILE)
            rows = con.execute('SELECT * FROM positions').fetchall()
            con.close()
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        _log.warning(f'[Recovery] DB 읽기 실패: {e}')
        return []


def _db_delete_position(strategy: str, symbol: str):
    """DB에서 포지션 삭제 (유령 레코드 정리용)."""
    try:
        with _db_lock:
            con = sqlite3.connect(DB_FILE)
            con.execute('DELETE FROM positions WHERE strategy=? AND symbol=?',
                        (strategy, symbol))
            con.commit()
            con.close()
        _log.info(f'[Recovery] DB 유령 레코드 삭제: {strategy}/{symbol}')
    except Exception as e:
        _log.warning(f'[Recovery] DB 삭제 실패: {e}')


# ══════════════════════════════════════════════════════════════
#  Telegram 알림 (atlas_main 의존 없이 독립 구현)
# ══════════════════════════════════════════════════════════════

_tg_session = requests.Session()

def _tg_alert(msg: str):
    safe = msg.replace('_', '-')
    try:
        _tg_session.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': safe,
                  'parse_mode': 'Markdown', 'disable_web_page_preview': True},
            timeout=10,
        )
    except Exception as e:
        _log.warning(f'[Recovery] Telegram 전송 실패: {e}')


# ══════════════════════════════════════════════════════════════
#  핵심 검증 로직
# ══════════════════════════════════════════════════════════════

@dataclass
class Discrepancy:
    kind:       str   # 'ORPHAN_EXCHANGE' | 'GHOST_DB' | 'QTY_MISMATCH'
    symbol:     str
    direction:  str
    ex_qty:     float
    db_qty:     float
    strategy:   str = ''
    detail:     str = ''


def detect_discrepancies(ex, db_positions: list) -> list:
    """
    거래소 포지션 ↔ DB 포지션 대조.
    Returns: Discrepancy 목록
    """
    try:
        raw = ex.fetch_positions()
    except Exception as e:
        _log.warning(f'[Recovery] 거래소 포지션 조회 실패: {e}')
        return []

    # 거래소 오픈 포지션: (symbol_USDT, LONG/SHORT) → contracts
    ex_open: dict = {}
    for p in raw:
        contracts = abs(float(p.get('contracts', 0) or 0))
        if contracts <= 0:
            continue
        raw_sym = (p.get('symbol') or '').replace('/', '').replace(':USDT', '')
        if not raw_sym.endswith('USDT'):
            raw_sym += 'USDT'
        side = 'LONG' if (p.get('side') or '').lower() == 'long' else 'SHORT'
        ex_open[(raw_sym, side)] = contracts

    # DB 포지션: (symbol, direction) → {qty, strategy}
    db_map: dict = {}
    for pos in db_positions:
        key = (pos['symbol'], pos['direction'])
        db_map[key] = pos

    issues: list = []

    # 1. 고아 포지션: 거래소에 있고 DB에 없음
    for (sym, side), qty in ex_open.items():
        if (sym, side) not in db_map:
            issues.append(Discrepancy(
                kind='ORPHAN_EXCHANGE', symbol=sym, direction=side,
                ex_qty=qty, db_qty=0.0,
                detail=f'거래소 {side} {sym} {qty} contracts, DB 기록 없음'))

    # 2. 유령 레코드: DB에 있고 거래소에 없음
    for (sym, side), pos in db_map.items():
        if (sym, side) not in ex_open:
            issues.append(Discrepancy(
                kind='GHOST_DB', symbol=sym, direction=side,
                ex_qty=0.0, db_qty=float(pos.get('qty', 0)),
                strategy=pos.get('strategy', ''),
                detail=f'DB {pos.get("strategy")} {side} {sym}, 거래소에 없음'))

    # 3. 수량 불일치: 양쪽 다 있지만 qty 차이가 큰 경우
    for (sym, side), pos in db_map.items():
        if (sym, side) in ex_open:
            ex_q = ex_open[(sym, side)]
            db_q = float(pos.get('qty', 0))
            if db_q > 0 and abs(ex_q - db_q) / db_q > QTY_TOLERANCE:
                issues.append(Discrepancy(
                    kind='QTY_MISMATCH', symbol=sym, direction=side,
                    ex_qty=ex_q, db_qty=db_q,
                    strategy=pos.get('strategy', ''),
                    detail=f'DB qty={db_q:.4f} vs 거래소={ex_q:.4f}'))

    return issues


def handle_discrepancy(ex, disc: Discrepancy,
                        auto_close_orphan: bool = False,
                        auto_clean_ghost:  bool = True):
    """
    불일치 항목 처리:
    - ORPHAN_EXCHANGE: Telegram 경고 + (선택) 시장가 청산
    - GHOST_DB:        Telegram 경고 + (선택) DB 삭제
    - QTY_MISMATCH:    Telegram 경고만
    """
    _log.warning(f'[Recovery] {disc.kind}: {disc.detail}')

    if disc.kind == 'ORPHAN_EXCHANGE':
        _tg_alert(
            f'⚠️ *[ATLAS Recovery] 고아 포지션 감지*\n'
            f'  {disc.symbol} {disc.direction} {disc.ex_qty} contracts\n'
            f'  DB에 기록 없음 — 수동 확인 필요\n'
            f'  {"자동 청산 시도" if auto_close_orphan else "자동 청산 비활성"}'
        )
        if auto_close_orphan:
            _try_close_orphan(ex, disc)

    elif disc.kind == 'GHOST_DB':
        _tg_alert(
            f'⚠️ *[ATLAS Recovery] 유령 DB 레코드 감지*\n'
            f'  {disc.strategy} {disc.symbol} {disc.direction}\n'
            f'  거래소에 포지션 없음 — DB 레코드 정리 중...'
        )
        if auto_clean_ghost and disc.strategy:
            _db_delete_position(disc.strategy, disc.symbol)
            _tg_alert(f'  ✅ DB 유령 레코드 삭제 완료: {disc.strategy}/{disc.symbol}')

    elif disc.kind == 'QTY_MISMATCH':
        _tg_alert(
            f'⚠️ *[ATLAS Recovery] 수량 불일치*\n'
            f'  {disc.strategy} {disc.symbol} {disc.direction}\n'
            f'  DB: {disc.db_qty:.4f}  거래소: {disc.ex_qty:.4f}'
        )


def _try_close_orphan(ex, disc: Discrepancy):
    """고아 포지션 시장가 청산 시도."""
    try:
        ccxt_sym  = disc.symbol.replace('USDT', '/USDT')
        close_side = 'sell' if disc.direction == 'LONG' else 'buy'
        ex.create_market_order(
            ccxt_sym, close_side, disc.ex_qty,
            params={'reduceOnly': True})
        _log.info(f'[Recovery] 고아 청산 완료: {disc.symbol} {disc.direction}')
        _tg_alert(
            f'✅ *[ATLAS Recovery] 고아 포지션 청산 완료*\n'
            f'  {disc.symbol} {disc.direction} {disc.ex_qty}')
    except Exception as e:
        _log.error(f'[Recovery] 고아 청산 실패: {e}')
        _tg_alert(f'❌ *[ATLAS Recovery] 고아 청산 실패*\n  {disc.symbol}: {e}')


# ══════════════════════════════════════════════════════════════
#  주기적 복구 루프
# ══════════════════════════════════════════════════════════════

def run_once(ex=None,
             auto_close_orphan: bool = False,
             auto_clean_ghost:  bool = True) -> list:
    """
    1회 점검 실행. 불일치 목록 반환.
    ex: CCXT 인스턴스 (None이면 내부 생성)
    """
    if ex is None:
        ex = _make_exchange()

    db_positions = _db_load_positions()
    issues = detect_discrepancies(ex, db_positions)

    if not issues:
        _log.debug('[Recovery] 포지션 정합성 이상 없음')
    else:
        _log.warning(f'[Recovery] {len(issues)}건 불일치 감지')
        for disc in issues:
            handle_discrepancy(ex, disc,
                               auto_close_orphan=auto_close_orphan,
                               auto_clean_ghost=auto_clean_ghost)

    return issues


def recovery_loop(interval: int = RECOVERY_INTERVAL,
                  auto_close_orphan: bool = False,
                  auto_clean_ghost:  bool = True):
    """
    백그라운드 스레드 진입점.
    atlas_main.py에서 daemon 스레드로 기동.

    예시:
        import threading
        from atlas_recovery import recovery_loop
        t = threading.Thread(target=recovery_loop, name='RecoveryLoop', daemon=True)
        t.start()
    """
    _log.info(f'[Recovery] 루프 시작 (주기: {interval}s)')
    ex = _make_exchange()

    while True:
        try:
            run_once(ex,
                     auto_close_orphan=auto_close_orphan,
                     auto_clean_ghost=auto_clean_ghost)
        except Exception as e:
            _log.warning(f'[Recovery] 루프 오류: {e}')
        time.sleep(interval)


# ══════════════════════════════════════════════════════════════
#  독립 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='ATLAS 포지션 복구 검증')
    parser.add_argument('--auto-close', action='store_true',
                        help='고아 포지션 자동 청산')
    parser.add_argument('--no-clean-ghost', action='store_true',
                        help='유령 DB 레코드 자동 삭제 비활성')
    parser.add_argument('--watch', action='store_true',
                        help='반복 실행 모드 (5분 간격)')
    args = parser.parse_args()

    if args.watch:
        recovery_loop(
            auto_close_orphan=args.auto_close,
            auto_clean_ghost=not args.no_clean_ghost,
        )
    else:
        issues = run_once(
            auto_close_orphan=args.auto_close,
            auto_clean_ghost=not args.no_clean_ghost,
        )
        if not issues:
            print('✅ 포지션 정합성 이상 없음')
        else:
            print(f'⚠️ {len(issues)}건 불일치:')
            for d in issues:
                print(f'  [{d.kind}] {d.symbol} {d.direction}: {d.detail}')

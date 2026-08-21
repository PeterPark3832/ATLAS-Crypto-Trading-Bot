"""
ATLAS Spot — 보호주문 점검 · 수량 교정
=====================================
열려 있는 포지션이 **실제로 거래소 손절 주문에 의해 보호되고 있는지**
확인한다. DB의 `sl_order_id`가 채워져 있다는 것만으로는 부족하다 —
그 주문이 거래소에 살아 있는지, 수량이 실제 보유량과 맞는지 봐야 한다.

이 점검이 필요한 이유:
  기초자산으로 수수료가 빠지면(BNB 잔고가 비었을 때) 체결량보다 보유량이
  적어져 '기록 수량 > 실제 보유량'이 된다. 그러면 손절 주문이
  -2010(insufficient balance)으로 거부되고, **DB에는 포지션이 있는데
  거래소에는 보호가 없는** 상태가 된다. 봇이 멈추면 손절되지 않는다.
  실제로 ADA·RIF·FET에서 발생했다.

읽기 전용이 기본이다. `--fix`를 줘야 DB 수량을 실측치로 교정한다.
주문을 새로 내지는 않는다 — 그건 봇의 자가복구가 할 일이다.

사용법:
  python check_protection.py              # 점검만 (읽기 전용)
  python check_protection.py --fix        # DB 수량을 실측 보유량으로 교정
"""

import argparse
import sqlite3
import sys
from pathlib import Path


# ⚠️ 여기에 환경변수 스텁(os.environ.setdefault)으로 자격증명 더미를 넣지 말 것.
#    이 도구는 잔고·미체결 주문을 보므로 **진짜 키가 필요하다**(프라이빗
#    엔드포인트). atlas_spot_config는 load_dotenv를 override 없이 부르므로,
#    미리 넣어둔 더미가 .env의 실제 키를 가려버려 도구 전체가
#    -2008 Invalid Api-Key로 죽는다. 백테스트·전략 모듈의 관용구(공개
#    엔드포인트만 쓰므로 무해)를 그대로 가져오면 안 되는 이유다.
#    키는 모두 _opt라 스텁 없이도 import된다 — 없으면 아래 가드가 잡는다.
sys.path.insert(0, str(Path(__file__).parent))
from atlas_spot_config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, SPOT_DB_FILE,
    SPOT_MIN_ORDER_USDT, SPOT_STOP_LIMIT_GAP,
)

# 수량 불일치 판정 기준. 수수료(0.1%)로 생기는 차이를 잡되, 부동소수
# 오차나 거래소 정밀도 반올림은 걸리지 않을 만큼의 여유를 둔다.
MISMATCH_PCT = 0.0001      # 0.01%


def credential_error() -> str:
    """자격증명 문제를 사람이 읽을 수 있는 한 줄로. 정상이면 ''.

    ccxt의 -2008 스택트레이스는 원인을 알려주지 않으므로 미리 잡는다.
    """
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return ('API 키가 없습니다 — .env의 BINANCE_API_KEY / '
                'BINANCE_API_SECRET를 확인하세요. '
                '(잔고·미체결 주문 조회는 프라이빗 엔드포인트입니다)')
    if BINANCE_API_KEY in ('CHECK', 'TEST', 'BACKTEST'):
        return (f'API 키가 테스트용 더미({BINANCE_API_KEY})입니다 — '
                '환경변수 스텁이 .env의 실제 키를 가리고 있습니다.')
    return ''


def _exchange():
    import ccxt
    return ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })


def _positions() -> list[dict]:
    if not Path(SPOT_DB_FILE).exists():
        print(f'DB 없음: {SPOT_DB_FILE}')
        return []
    conn = sqlite3.connect(str(SPOT_DB_FILE))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT strategy, symbol, qty_tokens, sl, tp, '
            '       sl_order_id, tp_order_id, entry_price '
            'FROM spot_positions ORDER BY strategy, symbol'
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _fix_qty(strategy: str, symbol: str, qty: float) -> None:
    conn = sqlite3.connect(str(SPOT_DB_FILE))
    try:
        conn.execute('UPDATE spot_positions SET qty_tokens=? '
                     'WHERE strategy=? AND symbol=?', (qty, strategy, symbol))
        conn.commit()
    finally:
        conn.close()


def main(argv: list | None = None) -> int:
    """Returns: 0 = 이상 없음, 1 = 확인이 필요한 포지션 있음.

    argv를 인자로 받는다 — `sys.argv`를 직접 읽으면 테스트에서 pytest의
    인자를 파싱해 버려 시험할 수 없다.
    """
    ap = argparse.ArgumentParser(description='보호주문 점검')
    ap.add_argument('--fix', action='store_true',
                    help='DB 수량을 실제 보유량으로 교정(주문은 내지 않는다)')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    err = credential_error()
    if err:
        print(f'⚠️ {err}')
        return 2

    pos = _positions()
    if not pos:
        print('열린 포지션 없음')
        return 0

    ex  = _exchange()
    bal = ex.fetch_balance()
    # ⚠️ free가 아니라 total(= free + locked)로 비교해야 한다.
    #    포지션의 물량은 **자기 손절 주문이 잠그고 있는 것**이 정상이다.
    #    free로 비교하면 제대로 보호된 포지션일수록 free가 0에 가까워
    #    "수량 초과 99.9%"로 판정되고, --fix가 DB 수량을 먼지 값으로
    #    덮어써 포지션 기록이 파괴된다(청산 시 그만큼만 팔고 나머지는 방치).
    held = bal.get('total', {}) or {}

    # 심볼을 지정하지 않은 fetch_open_orders는 ccxt가 레이트리밋 경고로
    # 막아 조회 자체가 실패한다(= 주문 존재 여부를 영영 확인 못 함).
    # 포지션 수만큼만 심볼별로 조회한다.
    live_by_sym: dict[str, set | None] = {}
    for _p in pos:
        _sym = str(_p['symbol'])
        try:
            _orders = ex.fetch_open_orders(_sym.replace('USDT', '/USDT'))
            live_by_sym[_sym] = {str(o.get('id')) for o in _orders}
        except Exception as e:
            print(f'⚠️ {_sym} 미체결 주문 조회 실패 — '
                  f'주문 존재 여부를 확인할 수 없습니다: {e}')
            live_by_sym[_sym] = None

    w = 96
    print('═' * w)
    print('  ATLAS 보호주문 점검')
    print('═' * w)
    print(f'  {"전략/심볼":<18}{"DB 수량":>14}{"실보유":>14}{"차이":>9}'
          f'{"SL주문":>10}{"거래소":>9}  판정')
    print(f'  {"─" * (w - 4)}')

    problems, fixes = [], []
    for p in pos:
        sym  = str(p['symbol'])
        base = sym.replace('USDT', '')
        db_q = float(p['qty_tokens'] or 0)
        act  = float(held.get(base, 0) or 0)
        diff = (db_q - act) / db_q if db_q > 0 else 0.0
        sl_id = str(p['sl_order_id'] or '')
        live_ids = live_by_sym.get(sym)

        # 거래소에 그 주문이 실제로 살아 있는가
        if live_ids is None:
            alive = '?'
        elif not sl_id:
            alive = '—'
        else:
            alive = 'O' if sl_id in live_ids else 'X'

        verdict, tag = [], '  ✅ 보호됨'
        if diff > MISMATCH_PCT:
            verdict.append(f'수량 초과 {diff * 100:.3f}%')
        if not sl_id:
            verdict.append('SL주문 ID 없음')
        elif alive == 'X':
            verdict.append('거래소에 주문 없음')

        # 최소 주문금액 미달이면 애초에 등록이 불가능하다(구조적)
        sl_px = float(p['sl'] or 0)
        if act > 0 and sl_px > 0:
            notional = act * sl_px * (1 - SPOT_STOP_LIMIT_GAP)
            if notional < SPOT_MIN_ORDER_USDT:
                verdict.append(f'NOTIONAL ${notional:.2f} 미달(구조적 불가)')

        if verdict:
            tag = '  ⚠️ ' + ' / '.join(verdict)
            problems.append((p, act, diff, verdict))
            if diff > MISMATCH_PCT and act > 0:
                fixes.append((p, act))

        print(f'  {p["strategy"]}/{sym:<13}{db_q:>14.8f}{act:>14.8f}'
              f'{diff * 100:>8.3f}%{(sl_id or "—")[:9]:>10}{alive:>9}{tag}')

    print(f'\n  포지션 {len(pos)}개 중 문제 {len(problems)}개')

    if fixes:
        print(f'\n  수량 교정 대상 {len(fixes)}개')
        for p, act in fixes:
            print(f'    {p["strategy"]}/{p["symbol"]}: '
                  f'{float(p["qty_tokens"]):.8f} → {act:.8f}')
        if args.fix:
            for p, act in fixes:
                _fix_qty(str(p['strategy']), str(p['symbol']), act)
            print('  ✅ DB 수량을 실측치로 교정했습니다.')
            print('     봇이 다음 사이클(최대 5분)에 보호주문을 자가복구합니다.')
        else:
            print('  → 교정하려면: python check_protection.py --fix')

    if problems and not fixes:
        print('\n  수량은 맞지만 보호주문이 없습니다. 봇이 돌고 있다면')
        print('  5분 주기 자가복구가 시도합니다. 계속 실패하면 로그를 확인하세요:')
        print('    grep -E "보호주문|자가복구|-2010" logs/spot_main.log | tail -30')

    print('═' * w)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())

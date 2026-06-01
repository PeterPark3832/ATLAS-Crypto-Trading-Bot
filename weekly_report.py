"""
ATLAS Spot 주간 성과 요약 — 매주 월요일 00:00 UTC crontab 실행
  0 0 * * 1 python3 /root/atlas_bot/weekly_report.py
"""

import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

TG_TOKEN        = os.getenv('TG_TOKEN', '')
TG_CHAT_ID      = os.getenv('TG_CHAT_ID', '')
DB_FILE         = Path(os.getenv('DB_FILE',
                    str(Path(__file__).parent / 'state' / 'atlas_spot.db')))
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))


def tg(msg: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(msg)
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            data={'chat_id': TG_CHAT_ID, 'text': msg},
            timeout=10,
        )
    except Exception as e:
        print(f'TG 전송 실패: {e}')


def load_trades(days: int) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, timeout=10)
        df  = pd.read_sql_query(f"""
            SELECT strategy, symbol, direction, pnl_usd, pnl_r, reason, exit_ts
            FROM trades
            WHERE exit_ts >= datetime('now', '-{days} days')
            ORDER BY exit_ts ASC
        """, con)
        con.close()
        df['exit_ts'] = pd.to_datetime(df['exit_ts'], utc=True, errors='coerce')
        return df
    except Exception as e:
        print(f'DB 오류: {e}')
        return pd.DataFrame()


def calc(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total  = len(df)
    wins   = int((df['pnl_usd'] > 0).sum())
    pnl    = float(df['pnl_usd'].sum())
    gw     = float(df[df['pnl_usd'] > 0]['pnl_usd'].sum())
    gl     = abs(float(df[df['pnl_usd'] < 0]['pnl_usd'].sum()))
    pf     = round(gw / gl, 2) if gl > 0 else float('inf')
    avg_r  = float(df['pnl_r'].mean())
    tp_cnt = int((df['reason'] == 'TP').sum())
    sl_cnt = int((df['reason'] == 'SL').sum())

    cum  = df['pnl_usd'].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum
    mdd  = float(dd.max()) / INITIAL_CAPITAL * 100

    return dict(total=total, wins=wins, wr=wins/total*100,
                pnl=pnl, pf=pf, avg_r=avg_r,
                tp=tp_cnt, sl=sl_cnt, mdd=mdd)


def strategy_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return '  거래 없음'
    lines = []
    for strat in sorted(df['strategy'].unique()):
        g     = df[df['strategy'] == strat]
        wins  = int((g['pnl_usd'] > 0).sum())
        lines.append(
            f"  {strat}: {len(g)}건 WR{wins/len(g)*100:.0f}% "
            f"${g['pnl_usd'].sum():+.1f}"
        )
    return '\n'.join(lines)


def main():
    now   = datetime.now(timezone.utc)
    week  = load_trades(7)
    month = load_trades(30)
    total = load_trades(9999)

    w = calc(week)
    m = calc(month)
    t = calc(total)

    if not t:
        tg('📊 ATLAS Spot 주간 리포트\n데이터 없음')
        return

    week_str = (
        f"이번주 ({w['total']}건):\n"
        f"  PnL: ${w['pnl']:+.2f}  WR: {w['wr']:.0f}%  PF: {w['pf']:.2f}\n"
        f"  TP:{w['tp']} SL:{w['sl']}  avgR: {w['avg_r']:+.3f}\n"
        f"  주간 MDD: {w['mdd']:.1f}%"
    ) if w else "이번주: 거래 없음"

    msg = (
        f"📊 ATLAS Spot 주간 리포트 ({now.strftime('%Y-%m-%d')})\n"
        f"{'─'*30}\n"
        f"{week_str}\n\n"
        f"전략별 (7일):\n{strategy_summary(week)}\n\n"
        f"누적 전체 ({t['total']}건):\n"
        f"  PnL: ${t['pnl']:+.2f}  WR: {t['wr']:.0f}%  PF: {t['pf']:.2f}\n"
        f"  MDD: {t['mdd']:.1f}%  avgR: {t['avg_r']:+.3f}\n"
        f"{'─'*30}\n"
        f"{'✅ 정상' if t['mdd'] < 10 and t['pf'] >= 1.0 else '⚠️ 점검 필요'}"
    )
    tg(msg)
    print(msg)


if __name__ == '__main__':
    main()

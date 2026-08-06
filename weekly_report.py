"""
ATLAS Spot 주간 성과 요약 — 매주 월요일 00:00 UTC 실행

설치(systemd 타이머):
  sudo cp deploy/atlas-weekly-report.service \
          deploy/atlas-weekly-report.timer /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now atlas-weekly-report.timer

수동 실행:
  python3 weekly_report.py
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

from atlas_spot_config import SPOT_DB_FILE   # noqa: E402  (.env 로드 후여야 한다)

TG_TOKEN        = os.getenv('TG_TOKEN', '')
TG_CHAT_ID      = os.getenv('TG_CHAT_ID', '')


def _resolve_db() -> Path:
    """봇이 쓰는 DB를 본다. DB_FILE 환경변수는 **존재할 때만** 우선한다.

    예전에는 DB_FILE만 보고 기본값으로 폴백했다. 그런데 서버 .env에는 옛
    선물봇 경로(/root/atlas_bot/state/atlas_v2.db)가 남아 있었고, 그 파일이
    없으면 load_trades가 빈 DataFrame을 돌려주므로 리포트는 **매번
    '데이터 없음'** 을 냈다. 거래 40건이 쌓여 있는데도 조용히 그랬다.

    이제 봇과 같은 단일 출처(SPOT_DB_FILE)를 기본으로 쓰고, 오버라이드가
    실재하지 않으면 그 사실을 알리고 정식 경로로 되돌린다.
    """
    override = os.getenv('DB_FILE', '').strip()
    if override:
        p = Path(override)
        if p.exists():
            return p
        print(f'⚠️ DB_FILE={p} 이(가) 없습니다 — 봇 DB로 대체합니다: {SPOT_DB_FILE}')
    return Path(SPOT_DB_FILE)


DB_FILE         = _resolve_db()
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
            SELECT strategy, symbol,
                   pnl_usdt - COALESCE(fee_usdt, 0) AS pnl_usdt,
                   pnl_r, reason, exit_ts
            FROM spot_trades
            WHERE exit_ts >= datetime('now', '-{int(days)} days')
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
    wins   = int((df['pnl_usdt'] > 0).sum())
    pnl    = float(df['pnl_usdt'].sum())
    gw     = float(df[df['pnl_usdt'] > 0]['pnl_usdt'].sum())
    gl     = abs(float(df[df['pnl_usdt'] < 0]['pnl_usdt'].sum()))
    pf     = round(gw / gl, 2) if gl > 0 else float('inf')
    avg_r  = float(df['pnl_r'].mean())
    tp_cnt = int((df['reason'] == 'TP').sum())
    sl_cnt = int((df['reason'] == 'SL').sum())

    # MDD: 자산 곡선의 피크 대비 % (고정 초기자본 대비는 자본 성장 시 왜곡)
    eq   = np.concatenate([[INITIAL_CAPITAL],
                           INITIAL_CAPITAL + df['pnl_usdt'].cumsum().values])
    peak = np.maximum.accumulate(eq)
    mdd  = float(((peak - eq) / np.where(peak > 0, peak, 1)).max()) * 100

    return dict(total=total, wins=wins, wr=wins/total*100,
                pnl=pnl, pf=pf, avg_r=avg_r,
                tp=tp_cnt, sl=sl_cnt, mdd=mdd)


def strategy_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return '  거래 없음'
    lines = []
    for strat in sorted(df['strategy'].unique()):
        g     = df[df['strategy'] == strat]
        wins  = int((g['pnl_usdt'] > 0).sum())
        lines.append(
            f"  {strat}: {len(g)}건 WR{wins/len(g)*100:.0f}% "
            f"${g['pnl_usdt'].sum():+.1f}"
        )
    return '\n'.join(lines)


def main():
    now   = datetime.now(timezone.utc)
    week  = load_trades(7)
    month = load_trades(30)
    total = load_trades(9999)

    w = calc(week)
    m = calc(month)      # 30일 — 주간의 잡음과 누적의 둔감함 사이를 메운다
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

    # 30일 구간 — 예전에는 계산해 놓고 리포트에 넣지 않아 매주 버려졌다.
    # 주간은 표본이 적어 잡음이 크고 누적은 최근 변화에 둔감하므로,
    # 그 사이를 메우는 이 구간이 실제 판단에 가장 쓸모 있다.
    month_str = (
        f"최근 30일 ({m['total']}건):\n"
        f"  PnL: ${m['pnl']:+.2f}  WR: {m['wr']:.0f}%  PF: {m['pf']:.2f}\n"
        f"  MDD: {m['mdd']:.1f}%  avgR: {m['avg_r']:+.3f}"
    ) if m else "최근 30일: 거래 없음"

    msg = (
        f"📊 ATLAS Spot 주간 리포트 ({now.strftime('%Y-%m-%d')})\n"
        f"{'─'*30}\n"
        f"{week_str}\n\n"
        f"{month_str}\n\n"
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

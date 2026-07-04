"""
ATLAS MCP 서버
==============
Claude Code에서 ATLAS Spot 봇 상태를 직접 조회/분석할 수 있는 도구 모음.
Vultr 서버에 SSH로 접속해 DB와 로그를 읽어옵니다.

[제공 도구]
  get_trade_history(days)   — 최근 N일 거래 내역
  get_pnl_by_strategy()     — 전략별 성과 비교 (S3–S7)
  check_alert_thresholds()  — 경고/즉시중단 임계값 체크
  get_error_logs(n)         — 최근 ERROR/WARNING 로그

[설치]
  pip install fastmcp paramiko python-dotenv

[등록]
  claude mcp add --transport stdio atlas -- python atlas_mcp_server.py

[설정]
  .env 파일에 VULTR_HOST, VULTR_USER, VULTR_KEY_PATH 설정
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import paramiko
from dotenv import load_dotenv
from fastmcp import FastMCP

# ── 환경 설정 ─────────────────────────────────────────────────
_env_path = Path(__file__).parent / '.env'
load_dotenv(_env_path if _env_path.exists() else None)

SSH_HOST     = os.getenv('VULTR_HOST', '')
SSH_PORT     = int(os.getenv('VULTR_PORT', '22'))
SSH_USER     = os.getenv('VULTR_USER', 'root')
SSH_KEY_PATH = os.getenv('VULTR_KEY_PATH', '')
SSH_PASSWORD = os.getenv('VULTR_PASSWORD', '')

REMOTE_DB_PATH  = os.getenv('REMOTE_DB_PATH',  '/root/ATLAS/state/atlas_spot.db')
REMOTE_LOG_DIR  = os.getenv('REMOTE_LOG_DIR',  '/root/ATLAS/logs')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))

mcp = FastMCP("ATLAS Spot")


# ── 내부 헬퍼 ─────────────────────────────────────────────────

def _ssh_connect() -> paramiko.SSHClient:
    """SSH 클라이언트 연결 반환."""
    if not SSH_HOST:
        raise RuntimeError(".env에 VULTR_HOST가 설정되지 않았습니다.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = dict(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER)
    if SSH_KEY_PATH:
        kwargs['key_filename'] = SSH_KEY_PATH
    elif SSH_PASSWORD:
        kwargs['password'] = SSH_PASSWORD
    else:
        raise RuntimeError("VULTR_KEY_PATH 또는 VULTR_PASSWORD 중 하나를 .env에 설정하세요.")
    client.connect(**kwargs)
    return client


def _query_db(sql: str, params: list | None = None) -> list[dict]:
    """원격 DB를 임시 파일로 다운로드 후 쿼리 실행."""
    client = _ssh_connect()
    tmp_path = None
    try:
        sftp = client.open_sftp()
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            tmp_path = f.name
        sftp.get(REMOTE_DB_PATH, tmp_path)
        sftp.close()

        conn = sqlite3.connect(tmp_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params or [])
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    finally:
        client.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_log_lines(n_tail: int = 3000) -> list[str]:
    """최신 로그 파일에서 마지막 N줄 반환."""
    client = _ssh_connect()
    try:
        stdin, stdout, stderr = client.exec_command(
            f'ls -t {REMOTE_LOG_DIR}/atlas_spot_*.log 2>/dev/null | head -1'
        )
        latest = stdout.read().decode().strip()
        if not latest:
            return []
        stdin, stdout, stderr = client.exec_command(f'tail -n {n_tail} "{latest}"')
        return stdout.read().decode().splitlines()
    finally:
        client.close()


# ── MCP 도구 ──────────────────────────────────────────────────

@mcp.tool()
def get_trade_history(days: int = 30) -> str:
    """
    최근 N일간 청산된 거래 내역을 조회합니다.
    각 거래의 전략, 심볼, 방향, 손익($), R-multiple, 청산 사유를 보여줍니다.
    최대 25건 표시.
    """
    rows = _query_db(f"""
        SELECT strategy, symbol, 'LONG' AS direction,
               ROUND(pnl_usdt - COALESCE(fee_usdt, 0), 2) AS pnl_usd,
               ROUND(pnl_r,    3) AS pnl_r,
               reason, regime,
               entry_ts, exit_ts
        FROM spot_trades
        WHERE exit_ts >= datetime('now', '-{int(days)} days')
        ORDER BY exit_ts DESC
    """)

    if not rows:
        return f"최근 {days}일간 청산된 거래 없음"

    total_pnl = sum(r['pnl_usd'] for r in rows)
    wins      = sum(1 for r in rows if r['pnl_usd'] > 0)
    losses    = len(rows) - wins
    wr        = wins / len(rows) * 100

    lines = [
        f"=== 최근 {days}일 거래 내역 ({len(rows)}건) ===",
        f"총 PnL : ${total_pnl:+.2f}",
        f"승/패  : {wins}W / {losses}L  (승률 {wr:.1f}%)",
        "",
    ]
    for r in rows[:25]:
        sign = "+" if r['pnl_usd'] >= 0 else ""
        ts   = (r['exit_ts'] or "")[:16]
        lines.append(
            f"[{ts}] {r['strategy']:<6} {r['symbol']:<10} {r['direction']:<5} "
            f"{sign}{r['pnl_usd']:>7.2f}$ ({sign}{r['pnl_r']:.2f}R)  "
            f"사유:{r['reason']}  레짐:{r['regime']}"
        )
    if len(rows) > 25:
        lines.append(f"... 외 {len(rows)-25}건")
    return "\n".join(lines)


@mcp.tool()
def get_pnl_by_strategy() -> str:
    """
    전략별 누적 성과를 비교합니다.
    S3(EMA추세), S4(RSI평균회귀), S5(BB반등), S6(Donchian돌파), S7(MACD)의
    거래수, 승률, PF, 평균R을 심볼별로 반환합니다.
    """
    rows = _query_db("""
        WITH t AS (
            SELECT strategy, symbol, pnl_r,
                   pnl_usdt - COALESCE(fee_usdt, 0) AS net_pnl
            FROM spot_trades
        )
        SELECT
            strategy,
            symbol,
            COUNT(*)                                                              AS total,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)                          AS wins,
            ROUND(SUM(net_pnl), 2)                                                AS total_pnl,
            ROUND(AVG(pnl_r),   3)                                                AS avg_r,
            ROUND(
                SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0.0 END) /
                NULLIF(ABS(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0.0 END)), 0),
            2)                                                                    AS pf,
            ROUND(MIN(net_pnl), 2)                                                AS worst,
            ROUND(MAX(net_pnl), 2)                                                AS best
        FROM t
        GROUP BY strategy, symbol
        ORDER BY strategy, symbol
    """)

    if not rows:
        return "거래 데이터 없음 — 아직 청산 거래가 없습니다."

    from collections import defaultdict
    by_strategy: dict = defaultdict(list)
    for r in rows:
        by_strategy[r['strategy']].append(r)

    lines = ["=== 전략별 누적 성과 ==="]
    for strategy, items in sorted(by_strategy.items()):
        lines.append(f"\n[{strategy}]")
        for r in items:
            wr    = r['wins'] / r['total'] * 100 if r['total'] else 0
            trust = "⚠️ 통계 불충분" if r['total'] < 20 else "✅"
            lines.append(
                f"  {r['symbol']:<10}  거래:{r['total']:>3}건 {trust}  "
                f"승률:{wr:>5.1f}%  PF:{str(r['pf']):>5}  "
                f"총PnL:${r['total_pnl']:>8.2f}  avgR:{r['avg_r']:>6.3f}  "
                f"최악:${r['worst']:>7.2f}  최선:${r['best']:>7.2f}"
            )

    all_pnl  = sum(r['total_pnl'] for r in rows)
    all_tot  = sum(r['total']     for r in rows)
    all_wins = sum(r['wins']      for r in rows)
    lines.append(f"\n[전체 합계]  거래:{all_tot}건  승률:{all_wins/all_tot*100:.1f}%  총PnL:${all_pnl:.2f}" if all_tot else "")
    return "\n".join(lines)


@mcp.tool()
def check_alert_thresholds() -> str:
    """
    봇 개입이 필요한지 판단합니다. 경고/즉시중단 기준과 현재 지표를 비교합니다.

    경고 기준: 실전 승률 < 35%, PF < 1.0, MDD > 15%
    즉시중단:  MDD > 20%

    거래 수가 10건 미만이면 통계 신뢰도 경고를 함께 표시합니다.
    """
    trades = _query_db("""
        SELECT pnl_usdt - COALESCE(fee_usdt, 0) AS pnl_usdt, pnl_r, exit_ts
        FROM spot_trades ORDER BY exit_ts
    """)

    if not trades:
        return "거래 데이터 없음 — 아직 청산 거래가 없습니다."

    total      = len(trades)
    wins       = sum(1 for t in trades if t['pnl_usdt'] > 0)
    wr         = wins / total
    gross_win  = sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] > 0)
    gross_loss = abs(sum(t['pnl_usdt'] for t in trades if t['pnl_usdt'] < 0))
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

    cum = 0.0
    peak = 0.0
    max_dd_usd = 0.0
    for t in trades:
        cum += t['pnl_usdt']
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd_usd:
            max_dd_usd = dd

    mdd_pct   = max_dd_usd / INITIAL_CAPITAL * 100
    total_pnl = sum(t['pnl_usdt'] for t in trades)

    alerts = []
    stops  = []

    if total >= 10:
        if wr < 0.35:
            alerts.append(f"⚠️  승률 {wr*100:.1f}% < 35% 경고선")
        if pf < 1.0:
            alerts.append(f"⚠️  PF {pf:.2f} < 1.0 — 손실 구간 진입")
    if mdd_pct > 20:
        stops.append(f"🚨 MDD {mdd_pct:.1f}% > 20% — 즉시 중단 검토 필요")
    elif mdd_pct > 15:
        alerts.append(f"⚠️  MDD {mdd_pct:.1f}% > 15% — 허용치 초과")

    lines = [
        "=== 봇 개입 판단 체크 ===",
        "",
        f"[현재 지표]",
        f"  거래수   : {total}건",
        f"  실전승률 : {wr*100:.1f}%",
        f"  PF       : {pf:.2f}  (기준 > 1.0)",
        f"  MDD(추정): {mdd_pct:.1f}%  = ${max_dd_usd:.2f}  (경고 15% / 중단 20%)",
        f"  누적PnL  : ${total_pnl:+.2f}",
        "",
        "[판정]",
    ]

    if stops:
        lines += ["  " + s for s in stops]
        lines.append("")
        lines.append("→ 즉시 중단 또는 포지션 정리를 검토하세요.")
    elif alerts:
        lines += ["  " + a for a in alerts]
        lines.append("")
        lines.append("→ 경고 상태: 모니터링 강화 필요. /pause 고려.")
    else:
        lines.append("  ✅ 모든 지표 정상 범위")
        lines.append("")
        lines.append("→ 정상 운용 중. 개입 불필요.")

    if total < 10:
        lines.append(f"\n📌 주의: 거래 {total}건으로 통계 신뢰도 낮음 (10건 이상 권장).")
    return "\n".join(lines)


@mcp.tool()
def get_error_logs(n: int = 50) -> str:
    """
    최근 로그 파일에서 WARNING/ERROR/CRITICAL 줄을 추출합니다.
    봇 오류, API 실패, 예외 상황을 빠르게 확인할 때 사용합니다.
    n: 반환할 최대 줄 수 (기본 50)
    """
    all_lines = _read_log_lines(n_tail=5000)

    if not all_lines:
        return "로그 파일을 찾을 수 없거나 비어 있습니다."

    error_lines = [
        l for l in all_lines
        if any(tag in l for tag in ('[WARNING]', '[ERROR]', '[CRITICAL]'))
    ]

    if not error_lines:
        return "✅ 최근 로그에 WARNING/ERROR/CRITICAL 없음 — 정상 운용 중입니다."

    result = [f"=== 최근 오류 로그 ({len(error_lines)}건, 최신 {n}건 표시) ===", ""]
    result += error_lines[-n:]
    return "\n".join(result)


# ── 진입점 ────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()

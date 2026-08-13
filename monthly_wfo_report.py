"""
ATLAS Spot — 월간 Walk-Forward OOS 자동 리포트
================================================
매월 1회 IS/OOS Walk-Forward 검증을 재실행하고 결과를 텔레그램으로 전송한다.
weekly_report.py 와 동일한 배달·환경변수 규약을 따른다 (tg / .env).

동작:
  1. LIVE_STRATEGIES × 고정 백테스트 유니버스로 run_walk_forward 실행
     (data-dir 캐시 우선 — 최초 1회만 바이낸스에서 5년치 fetch, 이후 증분)
  2. 전략별 IS vs OOS 지표(PF·Sharpe·CAGR·WR)와 PASS/FAIL 판정을 요약
  3. 텔레그램 전송 + 전체 결과를 results/ 에 JSON 저장
     (results/wfo_latest.json — reoptimize.py 가 읽어 재최적화 대상 파악)
  4. OOS PF < WF_OOS_MIN_PF(1.10) 전략이 있고 WFO_AUTO_REOPT=1 이면
     재최적화 제안 잡(atlas-wfo-reopt.service)을 순차 트리거한다.

주의: 이 스크립트는 조회·검증 전용이다. 라이브 config·봇을 절대 변경하지 않는다.

사용법:
  python monthly_wfo_report.py                # 월간 리포트 (텔레그램 + 저장)
  python monthly_wfo_report.py --rolling      # 롤링 윈도우 4단계 추가
  python monthly_wfo_report.py --no-tg        # 텔레그램 없이 콘솔만
  python monthly_wfo_report.py --strategies S4,S5

메모리: run_spot_backtest 가 유니버스 전 심볼의 OHLCV를 동시에 적재하므로
        systemd 유닛에서 MemoryMax 로 격리해 실행할 것 (deploy/atlas-wfo-report.service).
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / '.env')

from atlas_spot_backtest import run_walk_forward
from atlas_spot_config import (
    LIVE_STRATEGIES, STRATEGY_NAMES,
    WF_IS_START, WF_IS_END, WF_OOS_START,
    WF_OOS_MIN_PF, WF_OOS_MIN_SHARPE,
    SPOT_DATA_DIR, SPOT_RESULTS_DIR,
    SPOT_MAX_POSITIONS, SPOT_EQUITY_PER_SLOT, SPOT_DB_FILE,
)
from atlas_spot_universe import get_backtest_universe

TG_TOKEN   = os.getenv('TG_TOKEN', '')
TG_CHAT_ID = os.getenv('TG_CHAT_ID', '')

LATEST_PATH = SPOT_RESULTS_DIR / 'wfo_latest.json'


def tg(msg: str) -> None:
    """텔레그램 전송 (자격증명 없으면 print). weekly_report.tg 와 동일 규약."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print(msg)
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            data={'chat_id': TG_CHAT_ID, 'text': msg},
            timeout=15,
        )
    except Exception as e:
        print(f'TG 전송 실패: {e}')
        print(msg)


def evaluate(wf_results: dict, strategies: list[str]) -> list[dict]:
    """
    WF 결과에서 전략별 IS/OOS 지표와 PASS/FAIL 판정을 뽑는다.
    print_wf_report(atlas_spot_backtest) 의 판정 로직과 동일한 기준을 사용한다.
    Returns: [{sid, name, is, oos, ratio_pf, pass_pf, pass_sharpe, pass_pnl, verdict, has_data}]
    """
    is_data  = wf_results.get('IS', {})
    oos_data = wf_results.get('OOS', {})
    rows = []
    for sid in strategies:
        is_m  = is_data.get(sid, {}) or {}
        oos_m = oos_data.get(sid, {}) or {}
        has_data = bool(is_m) and bool(oos_m) and oos_m.get('total_trades', 0) > 0
        is_pf      = is_m.get('profit_factor', 0.0)
        oos_pf     = oos_m.get('profit_factor', 0.0)
        oos_sharpe = oos_m.get('sharpe', 0.0)
        oos_pnl    = oos_m.get('total_pnl_pct', 0.0)

        pass_pf     = oos_pf >= WF_OOS_MIN_PF
        pass_sharpe = oos_sharpe >= WF_OOS_MIN_SHARPE
        pass_pnl    = oos_pnl >= 0
        verdict = 'PASS' if (has_data and pass_pf and pass_sharpe and pass_pnl) else 'FAIL'
        rows.append({
            'sid': sid,
            'name': STRATEGY_NAMES.get(sid, sid),
            'is': is_m,
            'oos': oos_m,
            'ratio_pf': round(oos_pf / is_pf, 2) if is_pf > 0 else 0.0,
            'pass_pf': pass_pf,
            'pass_sharpe': pass_sharpe,
            'pass_pnl': pass_pnl,
            'verdict': verdict,
            'has_data': has_data,
        })
    return rows


def live_equity() -> float | None:
    """슬롯 수 계산에 쓸 자산. 조회 실패는 치명적이지 않다(단서만 일반화).

    봇이 기록해 둔 값을 읽는다 — 리포트는 격리된 oneshot 잡이라 거래소를
    직접 부르지 않는 편이 가볍고 자격증명도 필요 없다.

    현재 자산(`equity`)은 DB에 저장되지 않는다(메모리 상태로만 산다).
    남는 것은 그날 시작 자산과 피크뿐이므로 day_start_eq 를 쓴다 — 슬롯 수는
    $20 단위의 계단 함수라 하루 등락으로는 거의 바뀌지 않아 이 근사로 충분하다.
    """
    try:
        con = sqlite3.connect(f'file:{SPOT_DB_FILE}?mode=ro', uri=True, timeout=10)
        try:
            row = con.execute(
                "SELECT value FROM spot_config WHERE key='day_start_eq'").fetchone()
        finally:
            con.close()
        val = float(row[0]) if row and row[0] else 0.0
        return val if val > 0 else None
    except Exception as e:
        print(f'[자산 조회 실패 — 단서를 일반 문구로 대체] {e}')
        return None


def format_message(rows: list[dict], now: datetime, oos_end: str,
                   equity: float | None = None) -> str:
    """텔레그램용 요약 메시지 생성."""
    n_pass = sum(1 for r in rows if r['verdict'] == 'PASS')
    n_eval = sum(1 for r in rows if r['has_data'])
    head = (
        f"🧪 ATLAS WFO 월간 리포트 ({now.strftime('%Y-%m-%d')})\n"
        f"IS {WF_IS_START}~{WF_IS_END} / OOS {WF_OOS_START}~{oos_end}\n"
        f"기준: OOS PF≥{WF_OOS_MIN_PF} · Sharpe≥{WF_OOS_MIN_SHARPE} · PnL≥0\n"
        f"{'─' * 30}\n"
    )
    body = []
    for r in rows:
        if not r['has_data']:
            body.append(f"[{r['sid']}] {r['name']}\n  ⚠️ 데이터/거래 없음 — 판정 불가")
            continue
        mark = '✅' if r['verdict'] == 'PASS' else '❌'
        is_m, oos_m = r['is'], r['oos']
        body.append(
            f"{mark} [{r['sid']}] {r['name']}\n"
            f"  PF   IS {is_m.get('profit_factor', 0):.2f} → OOS {oos_m.get('profit_factor', 0):.2f}"
            f" ({r['ratio_pf']:.2f}x)\n"
            f"  Shrp IS {is_m.get('sharpe', 0):.2f} → OOS {oos_m.get('sharpe', 0):.2f}\n"
            f"  CAGR OOS {oos_m.get('cagr_pct', 0):+.1f}%  WR {oos_m.get('win_rate', 0):.0f}%"
            f"  거래 {oos_m.get('total_trades', 0)}"
        )
    failing = [r['sid'] for r in rows if r['has_data'] and r['verdict'] == 'FAIL']
    tail = f"\n{'─' * 30}\n종합: {n_pass}/{n_eval} PASS"
    if failing:
        tail += f"\n⚠️ 재최적화 후보: {', '.join(failing)} (OOS 기준 미달)"
    else:
        tail += "\n✅ 전 전략 OOS 기준 통과"
    tail += '\n' + portfolio_caveat(equity)
    return head + '\n\n'.join(body) + tail


def portfolio_caveat(equity: float | None) -> str:
    """이 수치를 '상한선'으로 읽어야 하는 이유를 함께 알린다.

    backtest_strategy 는 (전략 × 심볼) 단위로 독립 실행되므로 동시 포지션
    수 상한·슬롯당 최소 자본·USDT 예비금을 반영하지 못한다(코드에 한계로
    명시돼 있다). 즉 신호가 몰리는 구간에서 라이브는 일부 진입을 포기하는데
    백테스트는 전부 잡는다 — 결과가 구조적으로 낙관적이다.

    그런데 정작 판정을 전달하는 리포트에는 이 단서가 없었다. PASS/FAIL로
    전략 존폐를 결정하는 사람이 수치를 액면 그대로 믿게 된다.
    현재 자본으로 계산한 **실제 슬롯 수**를 함께 보여 낙관 정도를 가늠하게 한다.
    """
    base = ('※ 백테스트는 포트폴리오 제약(동시 포지션 한도·슬롯당 최소자본·'
            'USDT 예비금)을 모델링하지 않아 실제보다 낙관적입니다.')
    if not equity or equity <= 0:
        return base + f'\n   라이브 한도: 최대 {SPOT_MAX_POSITIONS}슬롯'
    slots = min(SPOT_MAX_POSITIONS, int(equity // SPOT_EQUITY_PER_SLOT))
    return (base + f'\n   현재 라이브 한도: {slots}슬롯 '
            f'(자산 ${equity:,.0f} ÷ 슬롯당 ${SPOT_EQUITY_PER_SLOT:.0f}, '
            f'상한 {SPOT_MAX_POSITIONS})')


def save_latest(rows: list[dict], wf_results: dict, meta: dict) -> None:
    """reoptimize.py 가 소비할 최신 WF 결과를 저장 + 타임스탬프 백업."""
    SPOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_at': meta['run_at'],
        'is_window': [WF_IS_START, WF_IS_END],
        'oos_window': [WF_OOS_START, meta['oos_end']],
        'thresholds': {'pf': WF_OOS_MIN_PF, 'sharpe': WF_OOS_MIN_SHARPE},
        'universe_size': meta['universe_size'],
        'summary': [
            {
                'sid': r['sid'], 'name': r['name'], 'verdict': r['verdict'],
                'has_data': r['has_data'],
                'is': r['is'], 'oos': r['oos'], 'ratio_pf': r['ratio_pf'],
            } for r in rows
        ],
        'wf_raw': wf_results,
    }
    LATEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    (SPOT_RESULTS_DIR / f'wfo_{ts}.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[저장] {LATEST_PATH}')


def maybe_trigger_reopt(failing: list[str]) -> None:
    """재최적화 제안 잡을 순차 트리거한다.

    트리거 조건 (WFO_AUTO_REOPT=1 전제):
      · OOS 미달 전략이 있을 때  — 원래 목적: 망가진 전략 구제
      · WFO_REOPT_ALWAYS=1 일 때 — 전 전략이 통과 중이어도 실행

    ⚠️ 두 번째 조건이 필요한 이유:
       재최적화기는 '실패 구제' 도구로 만들어졌지만, 지금은 **신규 기능의
       검증 통로**이기도 하다(추적 손절 ON/OFF 등이 그리드에 있다).
       실패했을 때만 돌면 전 전략이 통과 중인 정상 상태에서는 그 검증이
       **영원히 일어나지 않는다** — 기능을 넣어두고 데이터가 판단하게
       하겠다는 계획이 그대로 멈춘다.
       서버가 RAM 951MB / vCPU 1개라 매달 추가 실행이 부담일 수 있어
       기본값은 끄고, 운영자가 켜도록 했다(권장: 분기 1회 수동 실행).

    리포트 프로세스 종료 후 별도 cgroup(oneshot)에서 실행되도록 --no-block 사용.
    로컬/수동 실행(WFO_AUTO_REOPT 미설정)에서는 아무것도 하지 않는다."""
    if os.getenv('WFO_AUTO_REOPT') != '1':
        return
    always = os.getenv('WFO_REOPT_ALWAYS') == '1'
    if not failing and not always:
        return
    reason = (', '.join(failing) if failing
              else '전 전략 통과 — WFO_REOPT_ALWAYS로 정기 검증')
    try:
        subprocess.run(
            ['systemctl', 'start', '--no-block', 'atlas-wfo-reopt.service'],
            check=False, timeout=15,
        )
        print(f'[트리거] atlas-wfo-reopt.service 시작 요청 (대상: {reason})')
    except Exception as e:
        print(f'[트리거 실패] {e}')


def main() -> int:
    ap = argparse.ArgumentParser(description='ATLAS Spot 월간 WFO 리포트')
    ap.add_argument('--rolling', action='store_true', help='롤링 윈도우 4단계 추가')
    ap.add_argument('--no-tg', action='store_true', help='텔레그램 없이 콘솔만')
    ap.add_argument('--strategies', default='', help='쉼표구분 전략 (기본: LIVE_STRATEGIES)')
    ap.add_argument('--data-dir', default=str(SPOT_DATA_DIR), help='OHLCV CSV 캐시 경로')
    args = ap.parse_args()

    if args.no_tg:
        global TG_TOKEN
        TG_TOKEN = ''

    strategies = ([s.strip().upper() for s in args.strategies.split(',') if s.strip()]
                  if args.strategies else list(LIVE_STRATEGIES))
    symbols  = get_backtest_universe()
    data_dir = Path(args.data_dir) if args.data_dir else None
    now      = datetime.now(timezone.utc)
    oos_end  = now.strftime('%Y-%m-%d')

    print(f'[WFO] 전략 {strategies} · 심볼 {len(symbols)}개 · data_dir={data_dir}')
    try:
        wf_results = run_walk_forward(strategies, symbols, data_dir=data_dir,
                                      rolling=args.rolling)
    except Exception as e:
        tg(f'🧪 ATLAS WFO 월간 리포트\n❌ 실행 실패: {e}')
        traceback.print_exc()
        return 1

    rows = evaluate(wf_results, strategies)
    msg  = format_message(rows, now, oos_end, live_equity())
    tg(msg)
    print(msg)

    save_latest(rows, wf_results, {
        'run_at': now.isoformat(),
        'oos_end': oos_end,
        'universe_size': len(symbols),
    })

    failing = [r['sid'] for r in rows if r['has_data'] and r['verdict'] == 'FAIL']
    maybe_trigger_reopt(failing)
    return 0


if __name__ == '__main__':
    sys.exit(main())

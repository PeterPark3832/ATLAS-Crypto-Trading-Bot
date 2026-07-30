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


def format_message(rows: list[dict], now: datetime, oos_end: str) -> str:
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
    return head + '\n\n'.join(body) + tail


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
    """OOS 미달 전략이 있고 WFO_AUTO_REOPT=1 이면 재최적화 제안 잡을 순차 트리거.

    리포트 프로세스 종료 후 별도 cgroup(oneshot)에서 실행되도록 --no-block 사용.
    로컬/수동 실행(WFO_AUTO_REOPT 미설정)에서는 아무것도 하지 않는다."""
    if not failing or os.getenv('WFO_AUTO_REOPT') != '1':
        return
    try:
        subprocess.run(
            ['systemctl', 'start', '--no-block', 'atlas-wfo-reopt.service'],
            check=False, timeout=15,
        )
        print(f'[트리거] atlas-wfo-reopt.service 시작 요청 (대상: {", ".join(failing)})')
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
    msg  = format_message(rows, now, oos_end)
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

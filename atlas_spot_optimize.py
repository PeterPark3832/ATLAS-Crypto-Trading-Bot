#!/usr/bin/env python3
"""
ATLAS Spot — 파라미터 최적화 (과최적화 방지 내장)
=================================================
백테스트 성적을 올리는 파라미터는 언제나 찾을 수 있다. 문제는 그게
**과거에만 맞춘 값**이면 실계좌에서 그대로 잃는다는 것이다.
이 도구는 그 함정을 피하도록 절차 자체를 강제한다.

원칙
----
1. **탐색은 IS(In-Sample) 구간에서만.** OOS는 최종 후보 검증에 딱 한 번.
2. **피크가 아니라 고원(plateau)을 고른다.** 이웃 파라미터도 함께 좋아야
   한다. 한 점만 튀는 설정은 거의 항상 노이즈다.
3. **최소 거래 수 게이트.** 표본이 적으면 성적이 아니라 운이다.
4. **다중검정 인플레를 명시한다.** N개를 시도하면 순수 우연으로도 최고
   성적은 부풀려진다. 몇 개를 봤는지 항상 출력한다.
5. **IS→OOS 열화율을 반드시 보고한다.** 이 값이 크면 과최적화다.

사용법
------
  # 로컬 CSV로 (권장 — 재현 가능)
  python atlas_spot_optimize.py --strategy s3 --data-dir data/

  # 탐색 범위·횟수 조절
  python atlas_spot_optimize.py --strategy s3 --trials 40 --top 5

  # 현재 설정만 IS/OOS 평가 (탐색 없이 기준선 확인)
  python atlas_spot_optimize.py --strategy s3 --baseline-only

주의
----
이 도구가 "합격"을 줘도 그것은 **가설**이지 보장이 아니다. 반영 전에
소액 실계좌 또는 dry-run으로 표본을 더 쌓을 것.
"""

import argparse
import itertools
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'OPTIMIZE')

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import atlas_spot_backtest as bt
import atlas_spot_strategies as st
from atlas_spot_config import (
    WF_IS_START, WF_IS_END, WF_OOS_START,
    WF_OOS_MIN_SHARPE, WF_OOS_MIN_PF,
    SPOT_RESULTS_DIR, STRATEGY_TIMEFRAMES,
)

# 탐색 공간 — 각 전략의 **의미 있는 축**만 둔다.
# 축을 늘릴수록 다중검정 인플레가 커지므로 의도적으로 좁게 유지한다.
PARAM_GRID: dict = {
    'S3': {
        'S3_ADX_MIN':  [20, 22, 25, 28, 30],
        'S3_ATR_SL':   [2.0, 2.5, 3.0],
        'S3_RR_MIN':   [1.2, 1.5, 1.8],
        'S3_RR_MAX':   [2.5, 3.0, 3.5],
    },
    'S4': {
        'S4_RSI_ENTRY': [25, 30, 35],
        'S4_ATR_SL':    [1.5, 2.0, 2.5],
        'S4_RR':        [1.2, 1.5, 2.0],
        'S4_MAX_HOLD':  [10, 20, 30],
    },
    'S5': {
        'S5_BB_SIGMA':   [2.0, 2.2, 2.5],
        'S5_RSI_CONFIRM': [25, 30, 35],
        'S5_ATR_SL':     [1.2, 1.5, 2.0],
        'S5_MAX_HOLD':   [8, 12, 16],
    },
    'S6': {
        'S6_ENTRY_PERIOD': [15, 20, 25],
        'S6_EXIT_PERIOD':  [8, 10, 12],
        'S6_VOL_MULT':     [1.5, 2.0, 2.5],
        'S6_ATR_SL':       [1.5, 2.0, 2.5],
    },
}

MIN_TRADES = 30          # 이보다 적으면 성적이 아니라 운
TOP_KEEP   = 8           # 고원 평가 대상 후보 수


@dataclass
class Result:
    params: dict
    metrics: dict
    score: float
    plateau: float = 0.0


def _apply(params: dict) -> dict:
    """전략 모듈의 파라미터를 임시 교체. 이전 값을 반환(복원용)."""
    old = {}
    for k, v in params.items():
        old[k] = getattr(st, k)
        setattr(st, k, v)
    return old


def _score(m: dict) -> float:
    """IS 점수 — 수익성(PF)과 위험조정(Sharpe)을 함께 보고 MDD로 감점.

    단일 지표를 극대화하면 그 지표에만 맞춰지므로 복합 지표를 쓴다.
    거래 수가 적으면 점수를 0으로 만들어 노이즈 설정을 배제한다.
    """
    if m.get('total_trades', 0) < MIN_TRADES:
        return 0.0
    pf     = min(m.get('profit_factor', 0.0), 5.0)     # 999 같은 발산값 절단
    sharpe = m.get('sharpe', 0.0)
    mdd    = max(m.get('max_dd_pct', 0.0), 1.0)
    if pf <= 1.0 or sharpe <= 0:
        return 0.0
    return (pf - 1.0) * sharpe * (20.0 / mdd)


def _run(strategy: str, data: dict, regime_map: dict,
         start: str, end: str, params: dict | None = None) -> dict:
    """주어진 파라미터로 전 심볼 백테스트 → 종합 지표."""
    old = _apply(params) if params else {}
    try:
        all_trades = []
        for symbol, ohlcv in data.items():
            try:
                trades, _ = bt.backtest_strategy(
                    strategy, symbol, ohlcv, regime_map, start, end)
                all_trades.extend(trades)
            except Exception as e:
                print(f'    [경고] {symbol} 실패: {e}')
        return bt.calc_spot_metrics(all_trades, start_date=start, end_date=end)
    finally:
        if old:
            _apply(old)


def _combos(grid: dict, trials: int, seed: int) -> list:
    """전수 탐색이 크면 무작위 표본. 시드를 고정해 재현 가능하게 한다."""
    keys = list(grid)
    full = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    if len(full) <= trials:
        return full
    rng = random.Random(seed)
    return rng.sample(full, trials)


def _neighbors(params: dict, grid: dict) -> list:
    """각 축에서 한 칸씩 움직인 이웃 조합."""
    out = []
    for k, v in params.items():
        vals = grid[k]
        i = vals.index(v)
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                n = dict(params)
                n[k] = vals[j]
                out.append(n)
    return out


def optimize(strategy: str, data: dict, regime_map: dict,
             trials: int, top: int, seed: int) -> dict:
    grid = PARAM_GRID[strategy]
    combos = _combos(grid, trials, seed)
    print(f'\n[IS 탐색] {WF_IS_START} ~ {WF_IS_END} | 후보 {len(combos)}개 '
          f'(전수 {np.prod([len(v) for v in grid.values()])}개 중)')

    results = []
    for i, p in enumerate(combos, 1):
        m = _run(strategy, data, regime_map, WF_IS_START, WF_IS_END, p)
        s = _score(m)
        results.append(Result(p, m, s))
        if i % 10 == 0 or i == len(combos):
            print(f'  {i}/{len(combos)} 완료')

    results.sort(key=lambda r: r.score, reverse=True)
    scored = [r for r in results if r.score > 0]
    if not scored:
        print('\n[결과] IS에서 조건(PF>1, Sharpe>0, 최소 거래수)을 만족하는 '
              '설정이 없습니다. 탐색 범위나 기간을 재검토하세요.')
        return {}

    # ── 고원(plateau) 평가: 이웃도 함께 좋은 설정을 우대 ──────────
    # 한 점만 튀는 설정은 거의 항상 과거 노이즈에 맞춰진 것이다.
    print(f'\n[고원 평가] 상위 {min(TOP_KEEP, len(scored))}개의 이웃 파라미터 확인')
    cache = {tuple(sorted(r.params.items())): r.score for r in results}
    for r in scored[:TOP_KEEP]:
        ns = _neighbors(r.params, grid)
        vals = []
        for n in ns:
            key = tuple(sorted(n.items()))
            if key not in cache:
                cache[key] = _score(_run(strategy, data, regime_map,
                                         WF_IS_START, WF_IS_END, n))
            vals.append(cache[key])
        # 이웃 평균이 낮으면 고립된 피크 → 감점
        r.plateau = float(np.mean(vals)) if vals else 0.0

    ranked = sorted(scored[:TOP_KEEP],
                    key=lambda r: min(r.score, r.plateau * 1.5), reverse=True)
    best = ranked[0]

    print(f'\n[IS 최고] {best.params}')
    print(f'  점수 {best.score:.3f} | 이웃평균 {best.plateau:.3f} | '
          f'PF {best.metrics["profit_factor"]} Sharpe {best.metrics["sharpe"]} '
          f'MDD {best.metrics["max_dd_pct"]}% 거래 {best.metrics["total_trades"]}건')
    if best.plateau < best.score * 0.5:
        print('  ⚠️ 이웃 성적이 크게 낮습니다 — 고립된 피크(과최적화 의심)')

    # ── OOS 검증: 여기서 딱 한 번만 본다 ────────────────────────
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f'\n[OOS 검증] {WF_OOS_START} ~ {today} (후보 1개 + 현재설정 1개)')
    oos_best = _run(strategy, data, regime_map, WF_OOS_START, today, best.params)
    oos_base = _run(strategy, data, regime_map, WF_OOS_START, today)
    is_base  = _run(strategy, data, regime_map, WF_IS_START, WF_IS_END)

    return {
        'strategy': strategy, 'trials': len(combos),
        'evaluated': len(cache),
        'best_params': best.params,
        'is_best': best.metrics, 'oos_best': oos_best,
        'is_baseline': is_base, 'oos_baseline': oos_base,
        'plateau': best.plateau, 'is_score': best.score,
    }


def report(r: dict) -> bool:
    """결과 출력. 반환값: 채택 권고 여부."""
    if not r:
        return False
    ib, ob = r['is_best'], r['oos_best']
    bb, bo = r['is_baseline'], r['oos_baseline']

    def _row(label, m):
        if not m or m.get('total_trades', 0) == 0:
            return f'  {label:14s} 거래 없음'
        return (f'  {label:14s} PF {m["profit_factor"]:5.2f} | '
                f'Sharpe {m["sharpe"]:6.3f} | MDD {m["max_dd_pct"]:5.1f}% | '
                f'거래 {m["total_trades"]:4d} | 수익 {m["total_pnl_pct"]:+7.1f}%')

    print('\n' + '=' * 72)
    print(f'  {r["strategy"]} 최적화 결과')
    print('=' * 72)
    print(f'\n제안 파라미터: {r["best_params"]}\n')
    print('[In-Sample — 탐색에 사용, 성적이 좋은 건 당연하다]')
    print(_row('현재 설정', bb))
    print(_row('제안 설정', ib))
    print('\n[Out-of-Sample — 탐색에 쓰지 않은 구간. 여기가 진짜다]')
    print(_row('현재 설정', bo))
    print(_row('제안 설정', ob))

    n_tr = ob.get('total_trades', 0)
    sharpe_ok = ob.get('sharpe', 0) >= WF_OOS_MIN_SHARPE
    pf_ok     = ob.get('profit_factor', 0) >= WF_OOS_MIN_PF
    trades_ok = n_tr >= MIN_TRADES
    beats     = ob.get('sharpe', 0) > bo.get('sharpe', -9)

    print('\n[판정]')
    print(f'  OOS Sharpe ≥ {WF_OOS_MIN_SHARPE}  : {"통과" if sharpe_ok else "미달"} '
          f'({ob.get("sharpe", 0)})')
    print(f'  OOS PF     ≥ {WF_OOS_MIN_PF}  : {"통과" if pf_ok else "미달"} '
          f'({ob.get("profit_factor", 0)})')
    print(f'  OOS 거래수  ≥ {MIN_TRADES}   : {"통과" if trades_ok else "미달"} ({n_tr})')
    print(f'  현재 설정보다 나음         : {"예" if beats else "아니오"}')

    # 과최적화 신호
    if ib.get('sharpe', 0) > 0:
        deg = (1 - ob.get('sharpe', 0) / ib['sharpe']) * 100
        print(f'\n  IS→OOS Sharpe 열화율: {deg:+.0f}% ', end='')
        print('(정상 범위)' if deg < 50 else '← 과최적화 의심')
    print(f'  다중검정: 설정 {r["evaluated"]}개를 평가했습니다. 그중 최고는 '
          f'순수 우연으로도 부풀려집니다.')

    ok = sharpe_ok and pf_ok and trades_ok and beats
    print('\n' + ('  ✅ 채택 후보 — 단, 소액 실계좌/dry-run으로 표본을 더 쌓은 뒤 반영하세요.'
                  if ok else
                  '  ❌ 채택 불가 — 현재 설정을 유지하세요. (백테스트에서 좋아 보였던 것은\n'
                  '     과거에 맞춘 결과일 가능성이 높습니다)'))
    print('=' * 72)
    return ok


def main():
    ap = argparse.ArgumentParser(description='ATLAS Spot 파라미터 최적화')
    ap.add_argument('--strategy', required=True, help='s3 / s4 / s5 / s6')
    ap.add_argument('--data-dir', default='', help='로컬 CSV 디렉토리')
    ap.add_argument('--top', type=int, default=10, help='심볼 수')
    ap.add_argument('--trials', type=int, default=40, help='IS 탐색 후보 수')
    ap.add_argument('--seed', type=int, default=42, help='표본 추출 시드(재현용)')
    ap.add_argument('--baseline-only', action='store_true',
                    help='탐색 없이 현재 설정만 IS/OOS 평가')
    a = ap.parse_args()

    strategy = a.strategy.upper()
    if strategy not in PARAM_GRID:
        print(f'지원하지 않는 전략: {strategy} (가능: {list(PARAM_GRID)})')
        return 1

    tf = STRATEGY_TIMEFRAMES.get(strategy, '4h')
    data_dir = Path(a.data_dir) if a.data_dir else None
    universe = bt.get_backtest_universe()[:a.top]
    print(f'[데이터] {strategy} ({tf}) | 심볼 {len(universe)}개 로드 중...')

    data = {}
    since = bt._since_ms(WF_IS_START)
    for sym in universe:
        ohlcv = bt._load_or_fetch(sym, tf, since, data_dir)
        if ohlcv and len(ohlcv) > 200:
            data[sym] = ohlcv
    if not data:
        print('데이터를 불러오지 못했습니다. --data-dir 로 CSV를 지정하거나 '
              '네트워크를 확인하세요.')
        return 1
    print(f'[데이터] {len(data)}개 심볼 준비 완료')

    btc = bt._load_or_fetch('BTCUSDT', '1d', since, data_dir)
    regime_map = bt.build_regime_map(btc) if btc else {}

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if a.baseline_only:
        print('\n[현재 설정 기준선]')
        print('  IS :', bt.calc_spot_metrics([]) if not data else
              _run(strategy, data, regime_map, WF_IS_START, WF_IS_END))
        print('  OOS:', _run(strategy, data, regime_map, WF_OOS_START, today))
        return 0

    res = optimize(strategy, data, regime_map, a.trials, a.top, a.seed)
    ok = report(res)

    if res:
        SPOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = SPOT_RESULTS_DIR / f'optimize_{strategy}_{today}.json'
        out.write_text(json.dumps({**res, 'accepted': ok}, indent=2,
                                  ensure_ascii=False))
        print(f'\n저장: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

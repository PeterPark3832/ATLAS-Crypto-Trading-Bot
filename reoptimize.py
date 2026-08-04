"""
ATLAS Spot — 파라미터 자동 재최적화 (제안 전용)
================================================
OOS PF < WF_OOS_MIN_PF(1.10) 로 판정된 전략에 대해 소규모 그리드 서치로
개선 후보 파라미터를 탐색하고 "제안"만 한다.

★ 안전 규약 ★
  - 이 스크립트는 atlas_spot_config.py / 라이브 봇을 절대 수정하지 않는다.
  - 결과는 텔레그램 알림 + results/reopt_proposal_*.json 으로만 남는다.
  - 사람이 검토 후 config 상수를 손으로 바꾸고 봇을 재시작해야 실제 반영된다.

★ 과최적화 방지 ★
  - 후보 선택은 IS(2021~2023) 성과로만 한다. OOS 를 직접 최대화하지 않는다.
  - 선택된 IS-최적 조합을 OOS(2024~현재)로 검증해 "그 값"을 보고한다.
  - 현재값의 OOS PF 를 넘고 OOS 기준(PF≥1.10)을 통과할 때만 제안한다.
    (즉 OOS 는 채점이 아니라 검증용 — walk-forward optimization 규율)

대상 파라미터: 전략 함수가 '호출 시점'에 조회하는 진입/리스크 상수만.
  (max_hold 처럼 모듈 로드 시 dict 에 박히는 값은 몽키패치로 안 바뀌므로 제외)

사용법:
  python reoptimize.py                  # wfo_latest.json 의 FAIL 전략 대상
  python reoptimize.py --strategies S4  # 특정 전략 강제
  python reoptimize.py --no-tg
"""

import argparse
import itertools
import json
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / '.env')

import atlas_spot_backtest as _bt_mod
import atlas_spot_main as _live_mod
import atlas_spot_strategies as strat
from atlas_spot_backtest import (
    _since_ms, _load_or_fetch, build_regime_map,
    backtest_strategy, calc_spot_metrics,
)
from atlas_spot_config import (
    STRATEGY_TIMEFRAMES, STRATEGY_NAMES, LIVE_STRATEGIES,
    WF_IS_START, WF_IS_END, WF_OOS_START, WF_OOS_MIN_PF, WF_OOS_MIN_SHARPE,
    SPOT_DATA_DIR, SPOT_RESULTS_DIR, BT_INITIAL_EQ, SPOT_BASE_RISK_PCT,
)
from atlas_spot_universe import get_backtest_universe

TG_TOKEN   = os.getenv('TG_TOKEN', '')
TG_CHAT_ID = os.getenv('TG_CHAT_ID', '')

# ── 전략별 그리드 (호출 시점 조회 상수만) ──────────────────────────
#  값은 atlas_spot_strategies 모듈 전역명 = atlas_spot_config 상수명과 동일.
#  조합 폭발 방지를 위해 파라미터 2~3개 × 소수 값으로 제한한다.
GRIDS: dict[str, dict[str, list]] = {
    'S3': {'S3_ADX_MIN': [20, 25, 30], 'S3_ATR_SL': [2.0, 2.5, 3.0],
           'SPOT_TRAIL_ENABLED': [False, True]},
    'S4': {'S4_RSI_ENTRY': [25, 30, 35], 'S4_ATR_SL': [1.5, 2.0, 2.5], 'S4_RR': [1.5, 2.0]},
    'S5': {'S5_BB_SIGMA': [2.0, 2.2, 2.5], 'S5_RSI_CONFIRM': [25, 30, 35], 'S5_ATR_SL': [1.0, 1.2, 1.5]},
    'S6': {'S6_VOL_MULT': [1.5, 2.0, 2.5], 'S6_ATR_SL': [1.5, 2.0, 2.5],
           'SPOT_TRAIL_ENABLED': [False, True]},
}
# 추적 손절은 추세추종(S3·S6)에만 그리드에 넣는다. 평균회귀(S4·S5)는
# 되돌림을 먹는 구조라 조기 청산으로 손해 볼 가능성이 크고, 축을 늘릴수록
# 다중검정 인플레만 커진다. 필요하면 위 dict에 같은 항목을 추가하면 된다.

# ⚠️ 그리드에는 **백테스트가 실제로 반영하는** 파라미터만 넣을 것.
#    백테스트가 모르는 값을 넣으면 모든 후보가 동점이 되고, 최적화기는
#    그중 첫 값을 "개선"으로 제안한다. 검증된 적 없는 변경이 검증된 것처럼
#    보고되는 셈이라 가장 위험한 실패다.
#    (실제로 MOMENTUM_RS_GATE_PCT를 넣었다가 이 함정에 걸렸다 — 백테스트에
#     RS Gate 구현이 없어 세 값이 모두 같은 결과를 냈다. 라이브에서는 S6
#     진입의 2/3를 막는 큰 변경인데도 "OOS 개선"으로 제안됐다)
#    tests/test_reoptimize_guards.py 의 TestGridEffectiveness 가 이를 막는다.

# 고원(plateau) 판정에서 제외할 축.
# 고원 개념은 '파라미터 표면이 매끄러운가'를 보는 것이라 **순서 있는 수치
# 축**에만 의미가 있다. ON/OFF 같은 구조적 토글은 이웃이 항상 반대값이라,
# 그 기능이 실제로 효과가 클수록 오히려 '고립된 피크'로 오판된다.
PLATEAU_EXCLUDE = {'SPOT_TRAIL_ENABLED'}

MIN_TRADES = 20   # IS/OOS 최소 거래 수 — 표본 부족 조합 배제 (과최적화 방지)
PLATEAU_MIN_RATIO = 0.5   # 이웃 평균 IS 점수 / 최적 점수의 하한.
                          # 이보다 낮으면 '고립된 피크'로 보고 제안하지 않는다.


def _neighbors(combo: dict, grid: dict) -> list[dict]:
    """각 축에서 한 칸씩 움직인 이웃 조합."""
    out = []
    for k, v in combo.items():
        if k in PLATEAU_EXCLUDE:
            continue
        vals = grid[k]
        try:
            i = vals.index(v)
        except ValueError:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                n = dict(combo)
                n[k] = vals[j]
                out.append(n)
    return out


def _is_score(m: dict) -> float:
    """IS 점수 (PF × Sharpe). 표본 미달이면 0."""
    if m.get('total_trades', 0) < MIN_TRADES:
        return 0.0
    pf = min(m.get('profit_factor', 0.0), 5.0)   # 999 같은 발산값 절단
    sh = m.get('sharpe', 0.0)
    return pf * sh if pf > 0 and sh > 0 else 0.0


def tg(msg: str) -> None:
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


@contextmanager
def override_params(params: dict):
    """atlas_spot_strategies 모듈 전역을 임시 치환. 종료 시 원복.
    (신호 함수가 전역을 호출 시점에 조회하므로 재할당이 즉시 반영된다.)"""
    targets = {k: _param_targets(k) for k in params}
    missing = [k for k, t in targets.items() if not t]
    if missing:
        raise KeyError(f'어느 모듈에도 없는 파라미터: {missing}')
    saved = [(m, k, getattr(m, k)) for k, ms in targets.items() for m in ms]
    try:
        for k, v in params.items():
            for m in targets[k]:
                setattr(m, k, v)
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def _param_targets(key: str) -> tuple:
    """이 파라미터를 어느 모듈에 써야 하는가.

    전략 진입 상수는 atlas_spot_strategies 전역이지만, 추적 손절 같은
    **실행 규칙**은 라이브(atlas_spot_main)와 백테스트 양쪽 전역에 있다.
    한쪽만 바꾸면 검증이 실제 동작과 어긋난다.
    """
    mods = tuple(m for m in (strat, _live_mod, _bt_mod) if hasattr(m, key))
    return mods


def current_params(sid: str) -> dict:
    """그리드에 해당하는 현재(라이브) 파라미터 값."""
    out = {}
    for k in GRIDS[sid]:
        t = _param_targets(k)
        if t:
            out[k] = getattr(t[0], k)
    return out


def load_symbol_data(sid: str, symbols: list[str], data_dir: Path):
    """전략 타임프레임 데이터 + BTC 1D 레짐맵을 1회 로드 (IS·OOS 재사용)."""
    since_ms = _since_ms(WF_IS_START)
    tf = STRATEGY_TIMEFRAMES.get(sid, '1d')
    btc_1d = _load_or_fetch('BTCUSDT', '1d', since_ms, data_dir)
    btc_4h = _load_or_fetch('BTCUSDT', '4h', since_ms, data_dir)   # 레짐 패리티
    regime_map = build_regime_map(btc_1d, btc_4h) if btc_1d else {}
    ohlcv: dict[str, list] = {}
    for sym in symbols:
        data = _load_or_fetch(sym, tf, since_ms, data_dir)
        if data:
            ohlcv[sym] = data
    return ohlcv, regime_map


def run_window(sid: str, symbols: list[str], ohlcv: dict, regime_map: dict,
               start: str, end: str) -> dict:
    """주어진 파라미터(현재 모듈 상태) + 기간으로 전략 합산 지표 계산."""
    all_trades = []
    for sym in symbols:
        data = ohlcv.get(sym)
        if not data:
            continue
        trades, _ = backtest_strategy(sid, sym, data, regime_map,
                                      start, end, SPOT_BASE_RISK_PCT)
        all_trades.extend(trades)
    return calc_spot_metrics(all_trades, BT_INITIAL_EQ, start, end)


def optimize_strategy(sid: str, symbols: list[str], data_dir: Path,
                      oos_end: str) -> dict:
    """한 전략을 그리드 서치. Returns 제안 dict (개선 없으면 accepted=False)."""
    grid = GRIDS[sid]
    combos = [dict(zip(grid.keys(), vals))
              for vals in itertools.product(*grid.values())]
    print(f'\n[{sid}] {STRATEGY_NAMES.get(sid, sid)} — {len(combos)}개 조합 그리드')

    ohlcv, regime_map = load_symbol_data(sid, symbols, data_dir)
    if not ohlcv:
        return {'sid': sid, 'accepted': False, 'reason': '데이터 없음'}

    cur = current_params(sid)
    # 현재값 기준선 (IS·OOS)
    with override_params(cur):
        base_is  = run_window(sid, symbols, ohlcv, regime_map, WF_IS_START, WF_IS_END)
        base_oos = run_window(sid, symbols, ohlcv, regime_map, WF_OOS_START, oos_end)

    # ── IS 로만 후보 선택 (OOS 미열람) ──
    best = None
    for combo in combos:
        with override_params(combo):
            m_is = run_window(sid, symbols, ohlcv, regime_map, WF_IS_START, WF_IS_END)
        if m_is.get('total_trades', 0) < MIN_TRADES:
            continue
        score = (m_is.get('profit_factor', 0), m_is.get('sharpe', 0))
        if best is None or score > best['score']:
            best = {'combo': combo, 'score': score, 'is': m_is}

    if best is None:
        return {'sid': sid, 'accepted': False,
                'reason': f'IS 최소 거래({MIN_TRADES}) 만족 조합 없음',
                'baseline': {'is': base_is, 'oos': base_oos}}

    # ── 고원(plateau) 확인: 이웃 파라미터도 함께 좋은가 ──
    # 한 점만 튀는 설정은 거의 항상 과거 노이즈에 맞춰진 것이다. IS 최고점을
    # 그대로 믿으면 그 노이즈를 실계좌에 반영하게 되므로, 이웃의 IS 점수가
    # 크게 낮으면 '고립된 피크'로 보고 제안하지 않는다. (OOS는 여전히 미열람)
    peak_score = _is_score(best['is'])
    neigh_scores = []
    for n in _neighbors(best['combo'], grid):
        with override_params(n):
            neigh_scores.append(_is_score(
                run_window(sid, symbols, ohlcv, regime_map, WF_IS_START, WF_IS_END)))
    plateau = (sum(neigh_scores) / len(neigh_scores)) if neigh_scores else 0.0
    isolated = bool(peak_score > 0 and plateau < peak_score * PLATEAU_MIN_RATIO)
    print(f'  [{sid}] IS 최고점 {peak_score:.2f} / 이웃평균 {plateau:.2f}'
          + ('  ← 고립된 피크(과최적화 의심)' if isolated else ''))

    # ── 선택된 IS-최적 조합을 OOS 로 검증 ──
    with override_params(best['combo']):
        cand_oos = run_window(sid, symbols, ohlcv, regime_map, WF_OOS_START, oos_end)

    cand_pf   = cand_oos.get('profit_factor', 0)
    base_pf   = base_oos.get('profit_factor', 0)
    improved  = cand_pf > base_pf
    pass_oos  = (cand_pf >= WF_OOS_MIN_PF
                 and cand_oos.get('sharpe', 0) >= WF_OOS_MIN_SHARPE
                 and cand_oos.get('total_trades', 0) >= MIN_TRADES)
    accepted  = bool(improved and pass_oos and best['combo'] != cur and not isolated)

    reason = ''
    if best['combo'] == cur:
        reason = '현재값이 이미 IS-최적'
    elif isolated:
        reason = (f'고립된 피크 — 이웃 IS 점수가 최고점의 '
                  f'{plateau / peak_score * 100:.0f}%뿐 (과최적화 의심)')
    elif not pass_oos:
        reason = f'IS-최적 조합이 OOS 기준 미달 (PF {cand_pf:.2f})'
    elif not improved:
        reason = f'OOS 개선 없음 (현재 {base_pf:.2f} ≥ 후보 {cand_pf:.2f})'

    # IS→OOS 열화율: 탐색 구간에서만 좋았던 것인지 판별하는 핵심 지표
    is_sh = best['is'].get('sharpe', 0)
    degrade = round((1 - cand_oos.get('sharpe', 0) / is_sh) * 100, 1) if is_sh > 0 else None

    return {
        'sid': sid, 'name': STRATEGY_NAMES.get(sid, sid),
        'accepted': accepted, 'reason': reason,
        'current': cur, 'proposed': best['combo'],
        'baseline': {'is': base_is, 'oos': base_oos},
        'candidate': {'is': best['is'], 'oos': cand_oos},
        'n_combos': len(combos),
        # 과최적화 진단 — 제안을 사람이 검토할 때 반드시 함께 봐야 하는 값들
        'evaluated': len(combos) + len(neigh_scores),
        'peak_is_score': round(peak_score, 3),
        'plateau_is_score': round(plateau, 3),
        'isolated_peak': isolated,
        'is_oos_degrade_pct': degrade,
    }


def format_proposal(props: list[dict], now: datetime) -> str:
    accepted = [p for p in props if p.get('accepted')]
    head = (
        f"🔧 ATLAS 파라미터 재최적화 제안 ({now.strftime('%Y-%m-%d')})\n"
        f"※ 제안만 — config·봇 미변경. 검토 후 수동 반영 필요.\n"
        f"{'─' * 30}\n"
    )
    if not accepted:
        lines = [f"[{p['sid']}] {p.get('reason', '개선 후보 없음')}" for p in props]
        return head + "개선 제안 없음 (현 파라미터 유지 권장)\n" + '\n'.join(lines)

    blocks = []
    for p in accepted:
        b_oos, c_oos = p['baseline']['oos'], p['candidate']['oos']
        diff = '\n'.join(
            f"    {k} = {p['current'][k]}  →  {v}"
            for k, v in p['proposed'].items() if p['current'][k] != v
        )
        # 과최적화 진단을 제안과 **같은 화면에** 둔다. 따로 두면 안 본다.
        deg = p.get('is_oos_degrade_pct')
        diag = (f"  진단: 평가 {p.get('evaluated', '?')}개 조합 중 최고 "
                f"(우연으로도 부풀려짐)\n"
                f"        이웃평균/최고점 {p.get('plateau_is_score', 0):.2f}/"
                f"{p.get('peak_is_score', 0):.2f}")
        if deg is not None:
            diag += f" · IS→OOS 열화 {deg:+.0f}%"
            if deg > 50:
                diag += ' ⚠️'
        blocks.append(
            f"✅ [{p['sid']}] {p['name']}\n"
            f"  OOS PF {b_oos.get('profit_factor', 0):.2f} → {c_oos.get('profit_factor', 0):.2f}"
            f"  · Sharpe {b_oos.get('sharpe', 0):.2f} → {c_oos.get('sharpe', 0):.2f}\n"
            f"{diag}\n"
            f"  변경 제안 (atlas_spot_config.py):\n{diff}"
        )
    tail = (
        f"\n{'─' * 30}\n"
        f"⚠️ 과최적화 위험 — 반영 전 반드시 검토.\n"
        f"반영: config 상수 수정 → git commit → systemctl restart atlas-spot"
    )
    return head + '\n\n'.join(blocks) + tail


def _json_safe(obj):
    if hasattr(obj, '__dict__') and not isinstance(obj, dict):
        return asdict(obj)
    return obj


def save_proposal(props: list[dict], now: datetime) -> Path:
    SPOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = now.strftime('%Y%m%d_%H%M')
    path = SPOT_RESULTS_DIR / f'reopt_proposal_{ts}.json'
    payload = {
        'run_at': now.isoformat(),
        'note': '제안 전용 — config/봇 미변경. 사람이 검토 후 수동 반영.',
        'is_window': [WF_IS_START, WF_IS_END],
        'oos_window': [WF_OOS_START, now.strftime('%Y-%m-%d')],
        'proposals': props,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe),
                    encoding='utf-8')
    print(f'[저장] {path}')
    return path


def pick_targets(args_strategies: str) -> list[str]:
    """대상 전략 결정: --strategies > wfo_latest.json FAIL > LIVE_STRATEGIES."""
    if args_strategies:
        return [s.strip().upper() for s in args_strategies.split(',') if s.strip()]
    latest = SPOT_RESULTS_DIR / 'wfo_latest.json'
    if latest.exists():
        try:
            data = json.loads(latest.read_text(encoding='utf-8'))
            failing = [s['sid'] for s in data.get('summary', [])
                       if s.get('has_data') and s.get('verdict') == 'FAIL']
            if failing:
                print(f'[대상] wfo_latest.json FAIL 전략: {failing}')
                return failing
            print('[대상] wfo_latest.json — FAIL 전략 없음. 종료.')
            return []
        except Exception as e:
            print(f'[경고] wfo_latest.json 파싱 실패: {e}')
    return list(LIVE_STRATEGIES)


def main() -> int:
    ap = argparse.ArgumentParser(description='ATLAS 파라미터 재최적화 (제안 전용)')
    ap.add_argument('--strategies', default='', help='쉼표구분 전략 강제 지정')
    ap.add_argument('--no-tg', action='store_true')
    ap.add_argument('--data-dir', default=str(SPOT_DATA_DIR))
    args = ap.parse_args()

    if args.no_tg:
        global TG_TOKEN
        TG_TOKEN = ''

    targets = [s for s in pick_targets(args.strategies) if s in GRIDS]
    if not targets:
        print('재최적화 대상 없음.')
        return 0

    symbols  = get_backtest_universe()
    data_dir = Path(args.data_dir) if args.data_dir else None
    now      = datetime.now(timezone.utc)
    oos_end  = now.strftime('%Y-%m-%d')

    props = []
    for sid in targets:
        try:
            props.append(optimize_strategy(sid, symbols, data_dir, oos_end))
        except Exception as e:
            print(f'[{sid}] 재최적화 실패: {e}')
            traceback.print_exc()
            props.append({'sid': sid, 'accepted': False, 'reason': f'오류: {e}'})

    msg = format_proposal(props, now)
    tg(msg)
    print(msg)
    save_proposal(props, now)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
ATLAS Spot — 수익성 설계 계산기
================================
"최대 수익화"에서 **시장 예측 없이 확정적으로 결정되는 부분**만 계산한다.

기대손익은 이렇게 갈린다:

    거래당 순기대값 = R × (avg_r − 왕복비용률 / SL거리)
    총기대값        = 거래당 순기대값 × 실제로 체결된 거래 수

오른쪽 두 항 — **비용**과 **체결 가능 여부** — 는 시세와 무관하게 지금
결정돼 있다. 반면 avg_r은 시장이 준다. 그래서 이 도구는 시장을 예측하지
않고, 이미 확정된 쪽에서 새는 돈만 정확히 짚는다:

  ① 죽은 조합   — 신호가 나와도 주문금액이 거래소 최소치 미달이라
                  영원히 체결되지 않는 (전략 × 레짐) 칸. 그 레짐 기간의
                  기대값은 0이 아니라 **정의되지 않는다**(거래가 없다).
  ② 슬롯 상한   — 자본이 SPOT_MAX_POSITIONS를 지탱하지 못하면, 설정된
                  분산은 서류상으로만 존재한다.
  ③ 비용 잠식   — 왕복비용이 1R의 몇 %를 먹는가. BNB 수수료 할인은
                  코드 변경 없이 얻는 유일한 확정 이득이다.
  ④ 손익분기선  — 위 비용을 넘으려면 승률/손익비가 얼마여야 하는가.
                  "무조건 버는 전략"은 없지만, **져도 되는 한계선**은
                  정확히 계산된다.

사용법:
  python capital_plan.py --equity 300
  python capital_plan.py --equity 300 --sl-pct 0.04 --bnb
  python capital_plan.py --equity 300 --json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'PLAN')

sys.path.insert(0, str(Path(__file__).parent))
from atlas_spot_config import (
    SPOT_BASE_RISK_PCT, SPOT_MAX_ALLOC_PCT, SPOT_MAX_POSITIONS,
    SPOT_MIN_ORDER_USDT, SPOT_RESERVE_PCT, SPOT_EQUITY_PER_SLOT,
    SPOT_MAX_COST_PER_R, SPOT_MAX_SL_PCT,
    SPOT_ASSUMED_SLIP_PCT, SPOT_DEFAULT_SPREAD_PCT,
    BT_SPOT_FEE, REGIME_STRATEGY_MAP,
    WEAK_TREND_RISK_SCALE, TRENDING_DOWN_RISK_SCALE,
    SPOT_KELLY_SCALE_MIN, SPOT_HEALTH_SOFT_SCALE,
    STRATEGY_NAMES,
)

BNB_FEE_DISCOUNT = 0.75      # BNB로 수수료 결제 시 25% 할인
DEFAULT_SL_PCT   = 0.05      # 실측 표본이 없을 때의 대표 SL 거리


def regime_scale(regime: str) -> float:
    if regime == 'WEAK_TREND':
        return WEAK_TREND_RISK_SCALE
    if regime == 'TRENDING_DOWN':
        return TRENDING_DOWN_RISK_SCALE
    return 1.0


# ══════════════════════════════════════════════════════════════
#  ① 죽은 조합 — 언제 살아나는가
# ══════════════════════════════════════════════════════════════

def order_usdt(equity: float, risk_pct: float, sl_pct: float) -> float:
    """이 리스크·SL 거리에서 실제로 나가는 주문 금액."""
    if equity <= 0 or sl_pct <= 0 or risk_pct <= 0:
        return 0.0
    return min(equity * risk_pct / sl_pct, equity * SPOT_MAX_ALLOC_PCT)


def activation_equity(risk_pct: float, sl_pct: float) -> float:
    """이 조합이 거래소 최소 주문액을 넘기려면 자본이 얼마여야 하는가.

    주문액은 자본에 **선형**이므로 역산이 정확하다. 배분 상한이 걸리면
    상한 쪽이 병목이 된다(리스크를 더 키워도 소용없다는 뜻).
    """
    if risk_pct <= 0 or sl_pct <= 0:
        return math.inf
    rate = min(risk_pct / sl_pct, SPOT_MAX_ALLOC_PCT)
    return SPOT_MIN_ORDER_USDT / rate if rate > 0 else math.inf


def required_scale(equity: float, base_risk: float, sl_pct: float) -> float:
    """현재 자본에서 이 조합을 살리려면 레짐 스케일이 얼마여야 하는가.

    자본을 못 늘릴 때의 대안. 다만 이건 **리스크를 키우는 것**이므로
    공짜가 아니다 — 죽은 채로 두는 것과의 트레이드오프다.
    """
    if equity <= 0 or base_risk <= 0 or sl_pct <= 0:
        return math.inf
    need_risk = SPOT_MIN_ORDER_USDT * sl_pct / equity
    if SPOT_MIN_ORDER_USDT / equity > SPOT_MAX_ALLOC_PCT:
        return math.inf          # 배분 상한이 병목 — 스케일로 못 푼다
    return need_risk / base_risk


def dead_cells(equity: float, sl_pct: float = DEFAULT_SL_PCT,
               kelly: float = 1.0, health: float = 1.0) -> list[dict]:
    """(전략 × 레짐) 칸별 체결 가능성.

    Kelly·건강도는 실적에 따라 변하므로 기본은 1.0(가장 낙관적)으로 본다.
    즉 **여기서 죽은 칸은 어떤 실적에서도 죽어 있다.**
    """
    rows = []
    for regime, strats in REGIME_STRATEGY_MAP.items():
        rs = regime_scale(regime)
        for sid in strats:
            base = SPOT_BASE_RISK_PCT * kelly * health
            risk = base * rs
            cost = order_usdt(equity, risk, sl_pct)
            rows.append({
                'strategy':      sid,
                'strategy_name': STRATEGY_NAMES.get(sid, sid),
                'regime':        regime,
                'regime_scale':  rs,
                'risk_pct':      risk,
                'order_usdt':    cost,
                'tradable':      cost >= SPOT_MIN_ORDER_USDT,
                'activation_equity': activation_equity(risk, sl_pct),
                'required_scale':    required_scale(equity, base, sl_pct),
            })
    return rows


def uncovered_regimes(equity: float, sl_pct: float = DEFAULT_SL_PCT) -> list[str]:
    """**한 전략도** 체결되지 않는 레짐. 그 기간 수익은 구조적으로 0이다.

    전략이 아예 배정되지 않은 레짐(빈 리스트)도 포함한다 — 의도된
    무거래일 수 있으나, 커버리지 관점에서는 같은 공백이다.
    """
    rows = dead_cells(equity, sl_pct)
    by_regime: dict = {}
    for r in rows:
        by_regime.setdefault(r['regime'], []).append(r['tradable'])
    out = [rg for rg, strats in REGIME_STRATEGY_MAP.items() if not strats]
    out += [rg for rg, flags in by_regime.items() if flags and not any(flags)]
    return sorted(set(out))


# ══════════════════════════════════════════════════════════════
#  ② 슬롯 — 설정된 분산이 실제로 가능한가
# ══════════════════════════════════════════════════════════════

def slot_capacity(equity: float) -> dict:
    """자본이 지탱하는 동시 포지션 수 vs 설정값."""
    usable = equity * (1 - SPOT_RESERVE_PCT)
    afford = int(usable // SPOT_EQUITY_PER_SLOT) if SPOT_EQUITY_PER_SLOT > 0 else 0
    real   = max(0, min(afford, SPOT_MAX_POSITIONS))
    return {
        'usable_usdt':   usable,
        'configured':    SPOT_MAX_POSITIONS,
        'affordable':    afford,
        'effective':     real,
        'constrained':   real < SPOT_MAX_POSITIONS,
        'equity_for_configured': (SPOT_MAX_POSITIONS * SPOT_EQUITY_PER_SLOT
                                  / (1 - SPOT_RESERVE_PCT)),
    }


# ══════════════════════════════════════════════════════════════
#  ③ 비용 — 시장과 무관하게 확정된 마이너스
# ══════════════════════════════════════════════════════════════

def cost_profile(sl_pct: float = DEFAULT_SL_PCT, bnb: bool = False,
                 spread_pct: float | None = None) -> dict:
    """왕복비용률과 그것이 1R에서 차지하는 비중.

    시장가 왕복은 수수료 2회 + 스프레드 1회분 + 슬리피지 2회를 낸다.
    이 값은 avg_r에서 **그대로 차감**되므로, 줄인 만큼이 확정 이득이다.
    """
    fee = BT_SPOT_FEE * (BNB_FEE_DISCOUNT if bnb else 1.0)
    spread = SPOT_DEFAULT_SPREAD_PCT if spread_pct is None else spread_pct
    rate = fee * 2 + spread + SPOT_ASSUMED_SLIP_PCT * 2
    return {
        'fee_rate':      fee,
        'spread_pct':    spread,
        'slip_pct':      SPOT_ASSUMED_SLIP_PCT,
        'round_trip':    rate,
        'cost_per_r':    rate / sl_pct if sl_pct > 0 else math.inf,
        'gate':          SPOT_MAX_COST_PER_R,
        'blocked':       (rate / sl_pct if sl_pct > 0 else math.inf) > SPOT_MAX_COST_PER_R,
        'min_sl_pct':    rate / SPOT_MAX_COST_PER_R if SPOT_MAX_COST_PER_R > 0 else math.inf,
    }


def bnb_saving(sl_pct: float = DEFAULT_SL_PCT) -> dict:
    """BNB 할인의 가치 — 1R 대비 몇 %p를 되찾는가."""
    a = cost_profile(sl_pct, bnb=False)
    b = cost_profile(sl_pct, bnb=True)
    return {
        'cost_per_r_taker': a['cost_per_r'],
        'cost_per_r_bnb':   b['cost_per_r'],
        'recovered_r':      a['cost_per_r'] - b['cost_per_r'],
    }


# ══════════════════════════════════════════════════════════════
#  ④ 손익분기선 — "져도 되는 한계"
# ══════════════════════════════════════════════════════════════

def breakeven(sl_pct: float = DEFAULT_SL_PCT, bnb: bool = False,
              payoffs: tuple = (1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """손익비별 손익분기 승률.

    비용을 넣은 식:  WR × b − (1 − WR) × 1 − cost_per_r = 0
                     WR = (1 + cost_per_r) / (1 + b)
    이 선을 넘겨야 비로소 플러스다. "무조건 수익"은 존재하지 않지만,
    **무조건 손실인 구간**은 이렇게 확정적으로 그을 수 있다.
    """
    c = cost_profile(sl_pct, bnb)['cost_per_r']
    out = []
    for b in payoffs:
        wr = (1 + c) / (1 + b)
        wr_free = 1 / (1 + b)
        out.append({
            'payoff_r':      b,
            'breakeven_wr':  wr,
            'costless_wr':   wr_free,
            'cost_burden':   wr - wr_free,
            'achievable':    wr < 1.0,
        })
    return out


def expected_value(avg_r: float, sl_pct: float = DEFAULT_SL_PCT,
                   bnb: bool = False, risk_pct: float | None = None) -> dict:
    """관측된 avg_r이 주어졌을 때의 거래당 순기대값(자본 대비 %)."""
    risk = SPOT_BASE_RISK_PCT if risk_pct is None else risk_pct
    c = cost_profile(sl_pct, bnb)['cost_per_r']
    net_r = avg_r - c
    return {
        'avg_r':        avg_r,
        'cost_per_r':   c,
        'net_r':        net_r,
        'ev_pct':       net_r * risk,
        'profitable':   net_r > 0,
        'required_avg_r': c,
    }


# ══════════════════════════════════════════════════════════════
#  리포트
# ══════════════════════════════════════════════════════════════

def stressed_cells(equity: float, sl_pct: float = DEFAULT_SL_PCT) -> list[dict]:
    """연패 직후 상태(Kelly 하한 × 건강도 감봉)에서의 체결 가능성.

    사이징은 스케일들의 **곱**이라, 실적이 나빠질수록 주문이 작아진다.
    즉 회복이 가장 필요한 순간에 조합들이 조용히 죽는다 — 이 도구가
    낙관 시나리오만 보여주면 그 함정을 놓친다.
    """
    return dead_cells(equity, sl_pct,
                      kelly=SPOT_KELLY_SCALE_MIN, health=SPOT_HEALTH_SOFT_SCALE)


def learn_cells(equity: float, sl_pct: float = DEFAULT_SL_PCT) -> list[dict]:
    """자기주도 학습을 **켰을 때**의 체결 가능성.

    학습기는 Kelly·건강도를 대체하며, 이력이 없는 조합은
    `SPOT_LEARN_UNPROVEN_SCALE`(0.25)에서 시작한다. 즉 **켜는 순간
    모든 조합이 1/4 크기**가 되므로, 소액 계좌에서는 주문이 거래소
    최소액 아래로 내려가 봇이 조용히 멈출 수 있다.

    탐색 하한도 같은 값이라, 최악의 경우에도 이 배분이 바닥이다.
    """
    from atlas_spot_config import SPOT_LEARN_UNPROVEN_SCALE
    return dead_cells(equity, sl_pct, kelly=SPOT_LEARN_UNPROVEN_SCALE, health=1.0)


def learn_activation_equity(sl_pct: float = DEFAULT_SL_PCT) -> float:
    """학습기를 켜도 **모든 조합이 살아 있는** 최소 자본."""
    rows = learn_cells(1000.0, sl_pct)      # 자본 무관 — activation만 본다
    return max((r['activation_equity'] for r in rows), default=math.inf)


def build_report(equity: float, sl_pct: float, bnb: bool) -> dict:
    return {
        'equity':      equity,
        'sl_pct':      sl_pct,
        'bnb':         bnb,
        'cells':       dead_cells(equity, sl_pct),
        'stressed':    stressed_cells(equity, sl_pct),
        'learn':       learn_cells(equity, sl_pct),
        'learn_activation': learn_activation_equity(sl_pct),
        'uncovered':   uncovered_regimes(equity, sl_pct),
        'slots':       slot_capacity(equity),
        'cost':        cost_profile(sl_pct, bnb),
        'bnb_saving':  bnb_saving(sl_pct),
        'breakeven':   breakeven(sl_pct, bnb),
    }


def _fmt_eq(v: float) -> str:
    return '불가' if v == math.inf else f'${v:,.0f}'


def print_report(rep: dict) -> None:
    eq, sl = rep['equity'], rep['sl_pct']
    w = 76
    print('═' * w)
    print(f'  ATLAS 수익성 설계 — 자본 ${eq:,.2f} / 전형 SL {sl * 100:.1f}%'
          + ('  [BNB 할인 적용]' if rep['bnb'] else ''))
    print('═' * w)

    # ① 죽은 조합
    dead = [r for r in rep['cells'] if not r['tradable']]
    print(f'\n① 체결 가능성  —  {len(rep["cells"]) - len(dead)}/{len(rep["cells"])}칸 생존')
    for r in sorted(rep['cells'], key=lambda x: (x['regime'], x['strategy'])):
        mark = '  ' if r['tradable'] else '✗ '
        tail = ''
        if not r['tradable']:
            tail = (f'   → 필요자본 {_fmt_eq(r["activation_equity"])}'
                    f' 또는 레짐스케일 {r["required_scale"]:.2f}'
                    if r['required_scale'] != math.inf else
                    f'   → 필요자본 {_fmt_eq(r["activation_equity"])}')
        print(f'  {mark}{r["strategy"]:<4}/{r["regime"]:<14} '
              f'리스크 {r["risk_pct"] * 100:5.3f}%  주문 ${r["order_usdt"]:7.2f}{tail}')
    if dead:
        cheapest = min(d['activation_equity'] for d in dead)
        print(f'\n  ⚠ 죽은 칸 {len(dead)}개. 가장 가까운 활성화 자본: {_fmt_eq(cheapest)}')
        print(f'    이 칸들은 신호가 나와도 체결되지 않는다 — 로그는 정상으로 보인다.')
    if rep['uncovered']:
        print(f'  ⚠ 거래가 아예 없는 레짐: {", ".join(rep["uncovered"])}')
        print(f'    해당 장세 동안 계좌는 현금으로 대기한다(손실도 수익도 없음).')

    # ①-b 연패 시나리오 — 회복이 필요한 순간에 죽는 칸
    sdead = [r for r in rep['stressed'] if not r['tradable']]
    print(f'\n①-b 연패 직후(Kelly {SPOT_KELLY_SCALE_MIN:.2f} × 건강도 '
          f'{SPOT_HEALTH_SOFT_SCALE:.2f})  —  '
          f'{len(rep["stressed"]) - len(sdead)}/{len(rep["stressed"])}칸 생존')
    if sdead:
        for r in sorted(sdead, key=lambda x: -x['activation_equity']):
            print(f'  ✗ {r["strategy"]:<4}/{r["regime"]:<14} '
                  f'주문 ${r["order_usdt"]:6.2f}   '
                  f'→ 필요자본 {_fmt_eq(r["activation_equity"])}')
        worst = max(d['activation_equity'] for d in sdead)
        print(f'  ⚠ 손실 구간에서 {len(sdead)}칸이 멈춘다. 전부 살리려면 '
              f'{_fmt_eq(worst)} 필요.')
        print(f'    회복이 가장 필요할 때 거래가 끊긴다는 뜻이다.')
    else:
        print(f'  최악의 사이징에서도 전 조합이 체결 가능하다.')

    # ①-c 학습기를 켰을 때
    from atlas_spot_config import SPOT_LEARN_ENABLED, SPOT_LEARN_UNPROVEN_SCALE
    ldead = [r for r in rep['learn'] if not r['tradable']]
    state = '켜짐' if SPOT_LEARN_ENABLED else '꺼짐'
    print(f'\n①-c 자기주도 학습 켤 때(현재 {state}, 미검증 배분 '
          f'{SPOT_LEARN_UNPROVEN_SCALE:.2f})  —  '
          f'{len(rep["learn"]) - len(ldead)}/{len(rep["learn"])}칸 생존')
    if ldead:
        for r in sorted(ldead, key=lambda x: -x['activation_equity']):
            print(f'  ✗ {r["strategy"]:<4}/{r["regime"]:<14} '
                  f'주문 ${r["order_usdt"]:6.2f}   '
                  f'→ 필요자본 {_fmt_eq(r["activation_equity"])}')
        print(f'  ⚠ 지금 켜면 {len(ldead)}칸이 즉시 멈춘다. '
              f'전 조합을 살리려면 {_fmt_eq(rep["learn_activation"])} 필요.')
        print(f'    학습기는 이력이 쌓이기 전 모든 조합을 미검증으로 보므로, '
              f'켜는 순간이 가장 작은 주문이 나가는 시점이다.')
    else:
        print(f'  지금 켜도 전 조합이 체결 가능하다.')

    # ② 슬롯
    s = rep['slots']
    print(f'\n② 동시 포지션  —  설정 {s["configured"]}개 / 실제 가능 {s["effective"]}개')
    print(f'  가용자본 ${s["usable_usdt"]:,.2f} (예비금 {SPOT_RESERVE_PCT * 100:.0f}% 제외)'
          f'  ÷ 슬롯당 ${SPOT_EQUITY_PER_SLOT:,.0f}')
    if s['constrained']:
        print(f'  ⚠ 설정한 {s["configured"]}개 분산은 ${s["equity_for_configured"]:,.0f} '
              f'이상에서만 성립한다. 지금은 {s["effective"]}개로 집중된다.')

    # ③ 비용
    c = rep['cost']
    print(f'\n③ 비용 잠식  —  왕복 {c["round_trip"] * 100:.3f}% '
          f'= 1R의 {c["cost_per_r"] * 100:.1f}%')
    print(f'  수수료 {c["fee_rate"] * 100:.3f}%×2 + 스프레드 {c["spread_pct"] * 100:.3f}%'
          f' + 슬리피지 {c["slip_pct"] * 100:.3f}%×2')
    print(f'  진입 가드 상한 {c["gate"] * 100:.0f}% → SL {c["min_sl_pct"] * 100:.2f}% '
          f'미만인 신호는 차단된다'
          + ('  ⚠ 현재 설정이 가드에 걸린다' if c['blocked'] else ''))
    b = rep['bnb_saving']
    if not rep['bnb']:
        print(f'  BNB 수수료 결제 시: 1R의 {b["cost_per_r_taker"] * 100:.1f}%'
              f' → {b["cost_per_r_bnb"] * 100:.1f}% '
              f'({b["recovered_r"] * 100:.1f}%p 회수 — 코드 변경 없는 확정 이득)')

    # ④ 손익분기
    print(f'\n④ 손익분기선  —  이 선을 넘어야 비로소 플러스')
    print(f'  {"손익비":<8}{"필요 승률":<12}{"비용 없을 때":<14}{"비용 부담"}')
    for r in rep['breakeven']:
        print(f'  {r["payoff_r"]:<8.1f}{r["breakeven_wr"] * 100:>7.1f}%     '
              f'{r["costless_wr"] * 100:>9.1f}%     '
              f'{r["cost_burden"] * 100:>+7.1f}%p'
              + ('' if r['achievable'] else '   ← 달성 불가'))
    print(f'\n  avg_r이 {rep["cost"]["cost_per_r"]:.3f}R 이하면 거래할수록 잃는다.')
    print('═' * w)


def main() -> None:
    ap = argparse.ArgumentParser(description='ATLAS 수익성 설계 계산기')
    ap.add_argument('--equity', type=float, required=True, help='계좌 자산(USDT)')
    ap.add_argument('--sl-pct', type=float, default=DEFAULT_SL_PCT,
                    help=f'전형적 SL 거리 비율 (기본 {DEFAULT_SL_PCT})')
    ap.add_argument('--bnb', action='store_true', help='BNB 수수료 할인 적용 가정')
    ap.add_argument('--json', action='store_true', help='JSON 출력')
    a = ap.parse_args()

    if a.sl_pct <= 0 or a.sl_pct > SPOT_MAX_SL_PCT:
        ap.error(f'--sl-pct 는 0 초과 {SPOT_MAX_SL_PCT} 이하여야 한다 '
                 f'(SPOT_MAX_SL_PCT 초과 신호는 어차피 차단된다)')

    rep = build_report(a.equity, a.sl_pct, a.bnb)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(rep)


if __name__ == '__main__':
    main()

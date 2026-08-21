"""
ATLAS — 수익성 설계 계산기
================================
"최대 수익화"에서 시장 예측이 필요 없는 부분만 다룬다.

    거래당 순기대값 = R × (avg_r − 왕복비용률 / SL거리)

avg_r은 시장이 주지만, **비용**과 **체결 가능 여부**는 지금 확정돼 있다.
이 테스트는 계산기가 그 확정된 쪽을 틀리지 않게 고정한다. 특히:

  - 주문액은 자본에 선형 → 활성화 자본 역산이 정확해야 한다
  - 배분 상한이 병목이면 리스크를 키워도 안 풀린다(잘못 권하면 안 된다)
  - 손익분기 승률은 비용을 포함해야 한다. 비용을 빼먹으면 실제로는
    지는 설정을 "이기고 있다"고 읽게 된다

실행:
  pytest tests/test_capital_plan.py -v
"""

import math
import sys
from pathlib import Path


import pytest

import atlas_spot_config as cfg
import capital_plan as cp


# ══════════════════════════════════════════════════════════════
#  ① 체결 가능성 / 활성화 자본
# ══════════════════════════════════════════════════════════════

class TestOrderSizing:
    def test_order_is_linear_in_equity(self):
        a = cp.order_usdt(100, 0.02, 0.05)
        b = cp.order_usdt(200, 0.02, 0.05)
        assert b == pytest.approx(2 * a)

    def test_alloc_cap_binds(self):
        """리스크/SL 비율이 배분 상한을 넘으면 상한이 주문액을 정한다."""
        eq = 1000
        big = cp.order_usdt(eq, 0.50, 0.05)      # 비율 10 → 상한에 걸림
        assert big == pytest.approx(eq * cfg.SPOT_MAX_ALLOC_PCT)

    def test_zero_inputs_are_safe(self):
        assert cp.order_usdt(0, 0.02, 0.05) == 0.0
        assert cp.order_usdt(100, 0, 0.05) == 0.0
        assert cp.order_usdt(100, 0.02, 0) == 0.0


class TestActivationEquity:
    @pytest.mark.parametrize('risk,sl', [(0.02, 0.05), (0.006, 0.05), (0.01, 0.03)])
    def test_activation_is_exact(self, risk, sl):
        """역산한 자본에서 주문액이 정확히 최소치가 돼야 한다."""
        eq = cp.activation_equity(risk, sl)
        assert cp.order_usdt(eq, risk, sl) == pytest.approx(cfg.SPOT_MIN_ORDER_USDT)

    def test_just_below_is_dead(self):
        risk, sl = 0.006, 0.05
        eq = cp.activation_equity(risk, sl)
        assert cp.order_usdt(eq * 0.999, risk, sl) < cfg.SPOT_MIN_ORDER_USDT

    def test_alloc_cap_floors_activation(self):
        """상한이 병목이면 리스크를 아무리 키워도 이 자본 아래로는 못 내려간다."""
        floor = cfg.SPOT_MIN_ORDER_USDT / cfg.SPOT_MAX_ALLOC_PCT
        assert cp.activation_equity(9.99, 0.05) == pytest.approx(floor)

    def test_degenerate_inputs(self):
        assert cp.activation_equity(0, 0.05) == math.inf
        assert cp.activation_equity(0.02, 0) == math.inf


class TestRequiredScale:
    def test_scale_makes_cell_tradable(self):
        eq, base, sl = 300.0, cfg.SPOT_BASE_RISK_PCT, 0.05
        s = cp.required_scale(eq, base, sl)
        assert cp.order_usdt(eq, base * s, sl) == pytest.approx(cfg.SPOT_MIN_ORDER_USDT)

    def test_impossible_when_alloc_cap_binds(self):
        """자본이 너무 작으면 상한 때문에 스케일로는 못 푼다 —
        '스케일만 올리면 된다'고 잘못 권하지 않아야 한다."""
        tiny = cfg.SPOT_MIN_ORDER_USDT / cfg.SPOT_MAX_ALLOC_PCT * 0.9
        assert cp.required_scale(tiny, cfg.SPOT_BASE_RISK_PCT, 0.05) == math.inf


class TestDeadCells:
    def test_all_map_cells_present(self):
        rows = cp.dead_cells(300)
        expected = sum(len(v) for v in cfg.REGIME_STRATEGY_MAP.values())
        assert len(rows) == expected

    def test_downtrend_needs_most_capital(self):
        """하락장 스케일이 가장 낮으므로 가장 늦게 살아나고 가장 먼저 죽는다."""
        rows = cp.dead_cells(300)
        down = next(r for r in rows if r['regime'] == 'TRENDING_DOWN')
        assert down['activation_equity'] == max(r['activation_equity'] for r in rows)
        # 그 문턱 바로 아래에서는 실제로 죽어 있어야 한다
        below = cp.dead_cells(down['activation_equity'] * 0.99)
        assert not next(r for r in below if r['regime'] == 'TRENDING_DOWN')['tradable']

    def test_large_account_all_alive(self):
        rows = cp.dead_cells(100_000)
        assert all(r['tradable'] for r in rows)

    def test_stressed_is_never_better_than_base(self):
        """연패 시나리오가 낙관 시나리오보다 좋게 나오면 계산이 뒤집힌 것이다."""
        base = {(r['strategy'], r['regime']): r['order_usdt']
                for r in cp.dead_cells(300)}
        for r in cp.stressed_cells(300):
            assert r['order_usdt'] <= base[(r['strategy'], r['regime'])] + 1e-9

    def test_stressed_threshold_is_reported(self):
        """연패 구간에서 전 조합을 살리는 자본이 계산돼야 한다 —
        회복이 필요한 순간에 거래가 끊기는 것이 가장 비싼 실패다."""
        rows = cp.stressed_cells(300)
        dead = [r for r in rows if not r['tradable']]
        assert dead, '$300에서는 연패 시 죽는 칸이 있어야 한다(현재 설정 기준)'
        need = max(r['activation_equity'] for r in dead)
        assert all(r['tradable'] for r in cp.stressed_cells(need * 1.001))


class TestLearnerInteraction:
    """학습기를 켜면 **켜는 순간이 가장 작은 주문**이 나가는 시점이다.

    이력이 없으면 전 조합이 미검증(0.25)이라, 소액 계좌에서는 켜자마자
    주문이 거래소 최소액 아래로 내려가 봇이 조용히 멈출 수 있다.
    운영자가 켜기 전에 문턱을 알아야 한다.
    """

    def test_learner_shrinks_orders(self):
        base = {(r['strategy'], r['regime']): r['order_usdt']
                for r in cp.dead_cells(300)}
        for r in cp.learn_cells(300):
            assert r['order_usdt'] <= base[(r['strategy'], r['regime'])] + 1e-9

    def test_uses_unproven_scale(self):
        rows = cp.learn_cells(1000)
        up = next(r for r in rows if r['regime'] == 'TRENDING_UP')
        assert up['risk_pct'] == pytest.approx(
            cfg.SPOT_BASE_RISK_PCT * cfg.SPOT_LEARN_UNPROVEN_SCALE)

    def test_activation_threshold_is_finite_and_sufficient(self):
        need = cp.learn_activation_equity()
        assert math.isfinite(need)
        assert all(r['tradable'] for r in cp.learn_cells(need * 1.001))

    def test_just_below_threshold_has_dead_cells(self):
        need = cp.learn_activation_equity()
        assert any(not r['tradable'] for r in cp.learn_cells(need * 0.99))

    def test_threshold_appears_in_report(self, capsys):
        cp.print_report(cp.build_report(300, 0.05, False))
        out = capsys.readouterr().out
        assert '자기주도 학습' in out


class TestUncoveredRegimes:
    def test_empty_map_entries_are_uncovered(self):
        out = cp.uncovered_regimes(100_000)
        for rg, strats in cfg.REGIME_STRATEGY_MAP.items():
            if not strats:
                assert rg in out, f'{rg}: 전략 미배정인데 공백으로 안 잡혔다'

    def test_tiny_account_uncovers_more(self):
        assert len(cp.uncovered_regimes(20)) >= len(cp.uncovered_regimes(100_000))


# ══════════════════════════════════════════════════════════════
#  ② 슬롯
# ══════════════════════════════════════════════════════════════

class TestSlotCapacity:
    def test_reserve_is_excluded(self):
        s = cp.slot_capacity(1000)
        assert s['usable_usdt'] == pytest.approx(1000 * (1 - cfg.SPOT_RESERVE_PCT))

    def test_effective_never_exceeds_configured(self):
        for eq in (10, 100, 1000, 1_000_000):
            s = cp.slot_capacity(eq)
            assert s['effective'] <= s['configured']

    def test_threshold_actually_unlocks_configured(self):
        s = cp.slot_capacity(cp.slot_capacity(1000)['equity_for_configured'])
        assert s['effective'] == cfg.SPOT_MAX_POSITIONS
        assert not s['constrained']

    def test_zero_equity_is_safe(self):
        assert cp.slot_capacity(0)['effective'] == 0


# ══════════════════════════════════════════════════════════════
#  ③ 비용
# ══════════════════════════════════════════════════════════════

class TestCostProfile:
    def test_round_trip_components(self):
        c = cp.cost_profile(0.05)
        assert c['round_trip'] == pytest.approx(
            cfg.BT_SPOT_FEE * 2 + cfg.SPOT_DEFAULT_SPREAD_PCT
            + cfg.SPOT_ASSUMED_SLIP_PCT * 2)

    def test_tighter_sl_costs_more_per_r(self):
        """SL이 좁을수록 같은 비용이 1R에서 차지하는 비중이 커진다."""
        assert cp.cost_profile(0.02)['cost_per_r'] > cp.cost_profile(0.10)['cost_per_r']

    def test_min_sl_matches_gate(self):
        c = cp.cost_profile(0.05)
        at_gate = cp.cost_profile(c['min_sl_pct'])
        assert at_gate['cost_per_r'] == pytest.approx(cfg.SPOT_MAX_COST_PER_R)

    def test_gate_blocks_below_min_sl(self):
        c = cp.cost_profile(0.05)
        assert cp.cost_profile(c['min_sl_pct'] * 0.99)['blocked'] is True
        assert cp.cost_profile(c['min_sl_pct'] * 1.01)['blocked'] is False

    def test_bnb_discount_is_real_gain(self):
        b = cp.bnb_saving(0.05)
        assert b['cost_per_r_bnb'] < b['cost_per_r_taker']
        assert b['recovered_r'] > 0

    def test_bnb_only_discounts_fees(self):
        """할인은 수수료에만 적용된다 — 스프레드·슬리피지는 그대로다."""
        a, b = cp.cost_profile(0.05, False), cp.cost_profile(0.05, True)
        assert a['round_trip'] - b['round_trip'] == pytest.approx(
            cfg.BT_SPOT_FEE * 2 * (1 - cp.BNB_FEE_DISCOUNT))


# ══════════════════════════════════════════════════════════════
#  ④ 손익분기
# ══════════════════════════════════════════════════════════════

class TestBreakeven:
    def test_cost_raises_required_win_rate(self):
        for r in cp.breakeven(0.05):
            assert r['breakeven_wr'] > r['costless_wr']
            assert r['cost_burden'] > 0

    def test_costless_formula(self):
        for r in cp.breakeven(0.05):
            assert r['costless_wr'] == pytest.approx(1 / (1 + r['payoff_r']))

    def test_breakeven_is_actually_breakeven(self):
        """계산된 승률에서 순기대값이 0이어야 한다."""
        c = cp.cost_profile(0.05)['cost_per_r']
        for r in cp.breakeven(0.05):
            wr, b = r['breakeven_wr'], r['payoff_r']
            net = wr * b - (1 - wr) * 1 - c
            assert net == pytest.approx(0.0, abs=1e-9)

    def test_higher_payoff_needs_lower_wr(self):
        rows = cp.breakeven(0.05)
        wrs = [r['breakeven_wr'] for r in rows]
        assert wrs == sorted(wrs, reverse=True)

    def test_extreme_cost_marks_unachievable(self):
        """비용이 1R을 넘으면 어떤 승률로도 못 이긴다 — 그렇게 표시돼야 한다."""
        rows = cp.breakeven(0.0005, payoffs=(1.0,))
        assert rows[0]['breakeven_wr'] > 1.0
        assert rows[0]['achievable'] is False


class TestExpectedValue:
    def test_below_cost_is_losing(self):
        c = cp.cost_profile(0.05)['cost_per_r']
        ev = cp.expected_value(c * 0.5, 0.05)
        assert ev['profitable'] is False and ev['ev_pct'] < 0

    def test_above_cost_is_winning(self):
        c = cp.cost_profile(0.05)['cost_per_r']
        ev = cp.expected_value(c * 2, 0.05)
        assert ev['profitable'] is True and ev['ev_pct'] > 0

    def test_exactly_at_cost_is_zero(self):
        c = cp.cost_profile(0.05)['cost_per_r']
        assert cp.expected_value(c, 0.05)['net_r'] == pytest.approx(0.0)

    def test_required_avg_r_matches_cost(self):
        ev = cp.expected_value(0.3, 0.05)
        assert ev['required_avg_r'] == pytest.approx(ev['cost_per_r'])

    def test_ev_scales_with_risk(self):
        a = cp.expected_value(0.3, 0.05, risk_pct=0.01)['ev_pct']
        b = cp.expected_value(0.3, 0.05, risk_pct=0.02)['ev_pct']
        assert b == pytest.approx(2 * a)


# ══════════════════════════════════════════════════════════════
#  리포트 배선
# ══════════════════════════════════════════════════════════════

class TestReport:
    def test_build_report_has_all_sections(self):
        rep = cp.build_report(300, 0.05, False)
        for k in ('cells', 'stressed', 'uncovered', 'slots',
                  'cost', 'bnb_saving', 'breakeven'):
            assert k in rep, k

    def test_print_runs(self, capsys):
        cp.print_report(cp.build_report(300, 0.05, False))
        out = capsys.readouterr().out
        assert '손익분기' in out and '비용 잠식' in out

    def test_print_runs_for_tiny_account(self, capsys):
        cp.print_report(cp.build_report(30, 0.05, True))
        assert '체결 가능성' in capsys.readouterr().out

    def test_cli_rejects_sl_above_gate(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / 'capital_plan.py'),
             '--equity', '300', '--sl-pct', str(cfg.SPOT_MAX_SL_PCT + 0.01)],
            capture_output=True, text=True)
        assert r.returncode != 0

    def test_cli_json(self):
        import json as _json
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / 'capital_plan.py'),
             '--equity', '300', '--json'], capture_output=True, text=True)
        assert r.returncode == 0
        assert 'breakeven' in _json.loads(r.stdout)

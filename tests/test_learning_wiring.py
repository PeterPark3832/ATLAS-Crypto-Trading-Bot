"""
ATLAS — 학습기 배선 (라이브 · 백테스트)
================================
학습기는 `SPOT_LEARN_ENABLED`가 꺼져 있으면 **죽은 코드**다. 스위치를 켠
상태에서 검증하지 않으면 "테스트 다 통과"가 아무 의미도 없다.

여기서 고정하는 것:

  ① 켜면 Kelly·건강도를 **대체**한다(곱하지 않는다).
     곱셈으로 누적하면 스케일이 겹겹이 쌓여 주문이 거래소 최소액 아래로
     내려간다 — 실효 리스크가 설정값의 9%까지 떨어진 적이 있다.
  ② 라이브와 백테스트가 **같은 설정·같은 규칙**을 쓴다.
     한쪽만 바꾸면 검증한 사이징과 실계좌 사이징이 갈라진다.
  ③ 학습기가 죽어도 봇은 죽지 않는다(중립 배분으로 계속).

실행:
  pytest tests/test_learning_wiring.py -v
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_learning as L
import atlas_spot_backtest as bt
import atlas_spot_config as cfg
import atlas_spot_main as sm


@pytest.fixture
def learn_on(monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_LEARN_ENABLED', True)
    monkeypatch.setattr(bt, 'SPOT_LEARN_ENABLED', True)
    monkeypatch.setattr(sm, '_learn_cache', {'at': 0.0, 'result': {}, 'cfg': None})
    return True


# ══════════════════════════════════════════════════════════════
#  ① 스위치
# ══════════════════════════════════════════════════════════════

class TestSwitch:
    def test_default_is_off(self):
        """실계좌 사이징을 바꾸는 기능이라 기본은 꺼져 있어야 한다."""
        assert cfg.SPOT_LEARN_ENABLED is False

    def test_off_is_neutral(self, monkeypatch):
        monkeypatch.setattr(sm, 'SPOT_LEARN_ENABLED', False)
        assert sm._get_learn_scale('S6', 'TRENDING_UP') == 1.0

    def test_on_with_no_history_is_unproven(self, learn_on, monkeypatch):
        monkeypatch.setattr(sm, '_get_learn_result', lambda force=False: {})
        assert sm._get_learn_scale('S6', 'TRENDING_UP') == cfg.SPOT_LEARN_UNPROVEN_SCALE

    def test_on_uses_learned_value(self, learn_on, monkeypatch):
        monkeypatch.setattr(sm, '_get_learn_result',
                            lambda force=False: {('S6', 'TRENDING_UP'): {'scale': 1.37}})
        assert sm._get_learn_scale('S6', 'TRENDING_UP') == pytest.approx(1.37)


# ══════════════════════════════════════════════════════════════
#  ② 설정 패리티 — 라이브와 백테스트가 같은 규칙을 써야 한다
# ══════════════════════════════════════════════════════════════

class TestConfigParity:
    def test_live_and_backtest_configs_identical(self):
        assert sm._learn_config() == bt._bt_learn_config()

    def test_config_comes_from_module_globals(self, monkeypatch):
        """재최적화기가 상수를 몽키패치할 때 즉시 반영돼야 한다."""
        monkeypatch.setattr(sm, 'SPOT_LEARN_GAIN', 0.9)
        assert sm._learn_config().gain == pytest.approx(0.9)

    def test_config_is_validated(self, monkeypatch):
        """오설정이 조용히 통과하면 안 된다."""
        monkeypatch.setattr(sm, 'SPOT_LEARN_MAX_SCALE', 0.1)
        monkeypatch.setattr(sm, 'SPOT_LEARN_FLOOR', 0.25)
        with pytest.raises(ValueError):
            sm._learn_config()

    def test_cost_excludes_slippage(self):
        """pnl_r은 실제 체결가 기반이라 슬리피지·스프레드가 이미 반영돼 있다.
        왕복비용 전체를 넣으면 두 번 세어 엣지를 과소평가한다.

        (두 값의 단위가 다르다 — cost_per_r은 R 대비 비율, 나머지는 가격
         대비 비율이다. 전형 SL 5%로 환산해 비교한다)
        """
        sl_pct = 0.05
        fee_only = cfg.BT_SPOT_FEE * 2 / sl_pct
        full = (cfg.BT_SPOT_FEE * 2 + cfg.SPOT_DEFAULT_SPREAD_PCT
                + cfg.SPOT_ASSUMED_SLIP_PCT * 2) / sl_pct
        assert pytest.approx(fee_only, rel=0.05) == cfg.SPOT_LEARN_COST_PER_R
        assert full > cfg.SPOT_LEARN_COST_PER_R


# ══════════════════════════════════════════════════════════════
#  ③ 대체이지 누적이 아니다
# ══════════════════════════════════════════════════════════════

class TestReplacesNotStacks:
    def test_health_is_bypassed_when_learning(self):
        """학습기 ON이면 건강도 경로가 비활성이어야 한다 — 같은 근거로
        두 번 깎으면 주문이 최소액 아래로 내려간다."""
        src = Path(sm.__file__).read_text()
        assert '1.0 if SPOT_LEARN_ENABLED else _get_strategy_health_scale' in src

    def test_kelly_is_replaced_not_multiplied(self):
        src = Path(sm.__file__).read_text()
        i = src.index('if SPOT_LEARN_ENABLED:\n        kelly = _get_learn_scale')
        assert i > 0, '학습 배분이 Kelly 자리를 대신해야 한다'
        # 곱셈식에 learn 스케일이 추가로 곱해지면 안 된다
        line = next(l for l in src.splitlines() if 'adj_risk   = SPOT_BASE_RISK_PCT' in l)
        assert 'learn' not in line, f'누적 곱셈이 생겼다: {line}'

    def test_backtest_replaces_too(self):
        src = Path(bt.__file__).read_text()
        assert 'kelly_scale, health_scale = learn_scale, 1.0' in src, (
            '백테스트가 대체하지 않으면 라이브와 사이징이 갈라진다')


# ══════════════════════════════════════════════════════════════
#  ④ 학습기가 죽어도 봇은 죽지 않는다
# ══════════════════════════════════════════════════════════════

class TestFailSafe:
    def test_load_failure_is_neutral(self, learn_on, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError('DB 손상')
        monkeypatch.setattr(L, 'load_observations', boom)
        monkeypatch.setattr(sm, '_learn_cache', {'at': 0.0, 'result': {}, 'cfg': None})
        assert sm._get_learn_result() == {}
        assert sm._get_learn_scale('S6', 'TRENDING_UP') == cfg.SPOT_LEARN_UNPROVEN_SCALE

    def test_scale_lookup_failure_is_neutral(self, learn_on, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError('깨짐')
        monkeypatch.setattr(sm, '_get_learn_result', boom)
        assert sm._get_learn_scale('S6', 'TRENDING_UP') == 1.0

    def test_cache_avoids_recompute(self, learn_on, monkeypatch):
        calls = []
        monkeypatch.setattr(L, 'load_observations',
                            lambda *a, **k: calls.append(1) or [])
        monkeypatch.setattr(sm, '_learn_cache',
                            {'at': 0.0, 'result': {}, 'cfg': None})
        sm._get_learn_result()
        n = len(calls)
        monkeypatch.setattr(sm, '_learn_cache',
                            {'at': __import__('time').time(),
                             'result': {('S6', 'X'): {'scale': 1.0}}, 'cfg': None})
        sm._get_learn_result()
        assert len(calls) == n, '캐시가 있는데 다시 계산했다'

    def test_force_bypasses_cache(self, learn_on, monkeypatch):
        calls = []
        monkeypatch.setattr(sm, '_learn_cache',
                            {'at': __import__('time').time(),
                             'result': {('S6', 'X'): {'scale': 1.0}}, 'cfg': None})
        import atlas_learning as _L
        monkeypatch.setattr(_L, 'load_observations',
                            lambda *a, **k: calls.append(1) or [])
        sm._get_learn_result(force=True)
        assert len(calls) == 1, 'force=True인데 캐시를 그대로 썼다'


# ══════════════════════════════════════════════════════════════
#  ⑤ 백테스트 경로 — 실제로 동작하는가
# ══════════════════════════════════════════════════════════════

def _breakout(n=600, seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    px, out, ts = 100.0, [], 1606780800000
    for i in range(n):
        o = px
        drift = 0.02 if (i // 20) % 2 == 0 else -0.005
        px *= (1 + drift + rng.normal(0, 0.02))
        px = max(px, 1.0)
        out.append([ts + i * 86400000, o, max(o, px) * 1.01,
                    min(o, px) * 0.99, px, 1e6 * (2.5 if drift > 0 else 1.0)])
    return out


class TestBacktestPath:
    def test_runs_without_error_when_enabled(self, learn_on):
        trades, diag = bt.backtest_strategy(
            'S6', 'AUSDT', _breakout(), {}, '2021-01-01', '2022-12-31')
        assert isinstance(trades, list)

    def test_changes_sizing(self, learn_on, monkeypatch):
        """학습기를 켜면 리스크 배분이 실제로 달라져야 한다.
        같으면 배선이 죽어 있다는 뜻이다."""
        args = ('S6', 'AUSDT', _breakout(), {}, '2021-01-01', '2022-12-31')
        monkeypatch.setattr(bt, 'SPOT_LEARN_ENABLED', False)
        off, _ = bt.backtest_strategy(*args)
        monkeypatch.setattr(bt, 'SPOT_LEARN_ENABLED', True)
        on, _ = bt.backtest_strategy(*args)
        if not off or not on:
            pytest.skip('합성 데이터에서 거래 없음')
        assert [t.risk_pct for t in on] != [t.risk_pct for t in off]

    def test_first_trades_use_unproven_scale(self, learn_on):
        """이력이 없는 초기 구간은 미검증 배분이어야 한다."""
        trades, _ = bt.backtest_strategy(
            'S6', 'AUSDT', _breakout(), {}, '2021-01-01', '2022-12-31')
        if not trades:
            pytest.skip('거래 없음')
        # 레짐맵이 비어 있으면 WEAK_TREND로 폴백하므로 레짐 스케일도 곱해진다
        expect = (cfg.SPOT_BASE_RISK_PCT * cfg.SPOT_LEARN_UNPROVEN_SCALE
                  * cfg.WEAK_TREND_RISK_SCALE)
        assert trades[0].risk_pct == pytest.approx(expect, rel=0.02)

    def test_observations_carry_regime(self, learn_on):
        """레짐이 관측에 실리지 않으면 (전략 × 레짐) 학습이 성립하지 않는다."""
        rmap = {}
        base = datetime(2020, 12, 1, tzinfo=timezone.utc)
        for i in range(900):
            rmap[(base + timedelta(days=i)).strftime('%Y-%m-%d')] = 'TRENDING_UP'
        trades, _ = bt.backtest_strategy(
            'S6', 'AUSDT', _breakout(), rmap, '2021-01-01', '2022-12-31')
        if trades:
            assert all(t.regime == 'TRENDING_UP' for t in trades)

    def test_no_lookahead_in_learning(self, learn_on):
        """학습은 그 시점까지의 거래만 봐야 한다.

        기간을 뒤로 늘려도 **앞 구간 거래의 사이징은 바뀌지 않아야** 한다.
        바뀐다면 미래 거래가 과거 배분에 영향을 준 것이다.
        """
        data = _breakout()
        short, _ = bt.backtest_strategy('S6', 'AUSDT', data, {},
                                        '2021-01-01', '2021-12-31')
        long_, _ = bt.backtest_strategy('S6', 'AUSDT', data, {},
                                        '2021-01-01', '2022-12-31')
        if not short:
            pytest.skip('거래 없음')
        for a, b in zip(short, long_, strict=False):
            assert a.risk_pct == pytest.approx(b.risk_pct), (
                '기간을 늘렸더니 과거 거래의 사이징이 바뀌었다 — 선행편향')


# ══════════════════════════════════════════════════════════════
#  ⑥ 알려진 한계를 문서로 고정
# ══════════════════════════════════════════════════════════════

class TestKnownLimits:
    def test_backtest_learns_within_one_strategy_only(self):
        """`backtest_strategy`는 (전략 × 심볼) 단위로 독립 실행된다.

        따라서 백테스트의 학습기는 **그 전략의 레짐들끼리만** 비교하고,
        라이브처럼 전 전략을 가로질러 비교하지 못한다. 상대 배분의 기준점이
        다르므로 수치가 정확히 일치하지 않는다 — 포트폴리오 제약을 모델링
        못하는 것과 같은 계열의 한계다. 결과를 상한선으로 읽어야 한다.
        """
        src = Path(bt.__file__).read_text()
        assert '포트폴리오 제약은 모델링하지 못한다' in src

    def test_min_notional_interaction_is_documented(self):
        """미검증 배분 0.25는 소액 계좌에서 주문을 최소액 아래로 내릴 수 있다."""
        src = Path(cfg.__file__).read_text()
        assert 'capital_plan' in src and 'SPOT_LEARN_UNPROVEN_SCALE' in src


class TestRefreshThrottle:
    """재계산 스로틀은 **현 타임프레임에서는 걸리지 않는다** — 성능 방어용.

    4H·1D봉 전략은 진입 간격이 항상 30분을 넘으므로 매 진입마다 재학습한다.
    이 사실을 테스트로 박아 두지 않으면, 나중에 분봉 전략을 넣었을 때
    '왜 배분이 갱신되지 않지?'를 처음부터 다시 조사하게 된다.
    """

    def test_throttle_does_not_bind_at_current_timeframes(self, learn_on, monkeypatch):
        calls = []
        orig = L.learn
        monkeypatch.setattr(L, 'learn',
                            lambda *a, **k: calls.append(1) or orig(*a, **k))
        trades, _ = bt.backtest_strategy(
            'S6', 'AUSDT', _breakout(), {}, '2021-01-01', '2022-12-31')
        if not trades:
            pytest.skip('거래 없음')
        assert len(calls) == len(trades) + 1 or len(calls) == len(trades), (
            f'진입 {len(trades)}건에 학습 {len(calls)}회 — 스로틀이 걸리고 있다')

    def test_throttle_exists_for_future_intraday_strategies(self):
        src = Path(bt.__file__).read_text()
        assert 'SPOT_LEARN_REFRESH_MIN' in src

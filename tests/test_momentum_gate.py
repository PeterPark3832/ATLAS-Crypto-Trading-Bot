"""
ATLAS — 모멘텀 RS Gate / 유니버스 패리티
================================
발견:
  ① RS Gate 임계값이 `MOMENTUM_TOP_TIER_PCT * 3`(= 0.99)로 인라인돼 있어
     **어떤 심볼도 차단된 적이 없었다.** rank_pct의 최댓값은 (n-1)/n 이므로
     유니버스가 100개여도 0.99를 넘지 못한다. 주석은 "하위 67% 차단"이라
     되어 있었으니 코드와 문서가 어긋난 상태였다.
  ② 백테스트는 2021년 고정 25종목을 쓰고, 라이브는 4시간마다 모멘텀
     재랭킹한 동적 유니버스를 쓴다. 즉 라이브의 종목 선정 로직은 검증된
     적이 없다. (고정 유니버스 자체는 생존편향을 줄이려는 의도된 선택)

실행:
  pytest tests/test_momentum_gate.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_config as cfg
import atlas_spot_main as sm
import atlas_spot_universe as uni


# ══════════════════════════════════════════════════════════════
#  ① RS Gate — 죽어있던 필터
# ══════════════════════════════════════════════════════════════

class TestRsGate:
    def test_threshold_is_named_constant(self):
        """인라인 `* 3` 대신 이름 있는 상수여야 의도가 드러나고 튜닝된다."""
        assert hasattr(cfg, 'MOMENTUM_RS_GATE_PCT')
        src = Path(sm.__file__).read_text()
        assert 'MOMENTUM_TOP_TIER_PCT * 3' not in src

    def test_current_threshold_blocks_nothing(self):
        """현재 값(0.99)은 어떤 유니버스 크기에서도 차단하지 않는다.
        이 사실을 테스트로 박아 두어야 '동작하는 필터'로 오해하지 않는다."""
        for n in (10, 25, 50, 100):
            worst_rank_pct = (n - 1) / n
            assert worst_rank_pct <= cfg.MOMENTUM_RS_GATE_PCT, (
                f'유니버스 {n}개에서 최하위 {worst_rank_pct}가 '
                f'{cfg.MOMENTUM_RS_GATE_PCT}를 넘어 차단된다')

    def test_documented_intent_would_block(self):
        """주석의 의도(상위 33%만 통과)로 바꾸면 실제로 차단이 일어난다."""
        intended = cfg.MOMENTUM_TOP_TIER_PCT
        assert intended < (49 / 50), '0.33이면 하위권이 실제로 차단된다'

    def test_top_tier_boost_still_works(self, monkeypatch):
        """게이트는 죽어 있었지만 상위 티어 리스크 부스트는 정상 동작한다."""
        monkeypatch.setattr(sm, '_state',
                            {'universe_ranked': [f'S{i}USDT' for i in range(30)]})
        assert sm._get_momentum_rank_pct('S0USDT') == 0.0        # 최상위
        assert sm._get_momentum_rank_pct('S29USDT') == pytest.approx(29 / 30)
        top = sm._get_momentum_rank_pct('S5USDT')
        assert top <= cfg.MOMENTUM_TOP_TIER_PCT, '상위 티어로 인식돼야 한다'

    def test_unknown_symbol_is_neutral(self, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'universe_ranked': ['AUSDT']})
        assert sm._get_momentum_rank_pct('ZUSDT') == 0.5

    def test_in_wfo_grid_now_that_backtest_models_it(self):
        """WFO 그리드에 넣었다가 뺐다가, 다시 넣었다.

        처음엔 백테스트에 RS Gate 구현이 없어서 세 값이 **모두 같은 결과**를
        냈고, 최적화기는 동점 중 첫 값(0.33)을 "OOS PF 2.22 → 3.54 개선"으로
        제안했다 — 검증된 적 없는 변경이 검증된 척한 것이다. 그래서 뺐다.
        지금은 백테스트가 순위맵으로 게이트를 재현하므로, 임계값을 감이 아니라
        데이터가 고르게 하려고 되돌려 놓았다.
        """
        import reoptimize as ro
        assert 'MOMENTUM_RS_GATE_PCT' in ro.GRIDS['S6']

    def test_grid_can_patch_it(self):
        import reoptimize as ro
        before = sm.MOMENTUM_RS_GATE_PCT
        with ro.override_params({'MOMENTUM_RS_GATE_PCT': 0.33}):
            assert sm.MOMENTUM_RS_GATE_PCT == 0.33
        assert before == sm.MOMENTUM_RS_GATE_PCT


# ══════════════════════════════════════════════════════════════
#  ② 모멘텀 랭킹 자체
# ══════════════════════════════════════════════════════════════

def _series(n, daily_ret, vol, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    px, out, ts = 100.0, [], 1609459200000
    for i in range(n):
        o = px
        px *= (1 + daily_ret + rng.normal(0, vol))
        out.append([ts + i * 86400000, o, max(o, px), min(o, px), px, 1e6])
    return out


class TestMomentumRanking:
    def test_higher_risk_adjusted_momentum_ranks_first(self):
        cache = {
            'STRONG': _series(120, 0.004, 0.01, seed=2),   # 고수익 저변동
            'WEAK':   _series(120, 0.000, 0.01, seed=3),
            'NOISY':  _series(120, 0.004, 0.05, seed=4),   # 같은 수익, 고변동
        }
        ranked = uni.rank_by_momentum(list(cache), cache)
        assert ranked.index('STRONG') < ranked.index('WEAK')
        assert ranked.index('STRONG') < ranked.index('NOISY'), (
            '변동성으로 나눈 값이므로 같은 수익이면 저변동이 앞서야 한다')

    def test_missing_data_sinks_to_bottom(self):
        cache = {'GOOD': _series(120, 0.003, 0.01), 'SHORT': _series(10, 0.01, 0.01)}
        ranked = uni.rank_by_momentum(['GOOD', 'SHORT'], cache)
        assert ranked[-1] == 'SHORT'

    def test_empty_cache_is_safe(self):
        assert uni.rank_by_momentum(['A', 'B'], {}) == ['A', 'B']


# ══════════════════════════════════════════════════════════════
#  ③ 유니버스 패리티 — 알려진 한계를 문서로 고정
# ══════════════════════════════════════════════════════════════

class TestUniverseParityGap:
    def test_backtest_universe_is_static(self):
        """백테스트는 고정 유니버스를 쓴다(생존편향 축소 목적).
        따라서 라이브의 동적 모멘텀 선정은 백테스트로 검증되지 않는다."""
        a = uni.get_backtest_universe()
        b = uni.get_backtest_universe()
        assert a == b and len(a) >= 20

    def test_gap_is_documented(self):
        """이 한계가 코드에 적혀 있어야 결과를 오해하지 않는다."""
        src = Path(uni.__file__).read_text()
        assert '선행편향' in src or '생존' in src

    def test_symbol_selection_still_unvalidated(self):
        """RS Gate는 백테스트가 재현하지만, **종목 선정** 자체는 아니다.

        백테스트는 고정 유니버스 안에서 순위를 매길 뿐이고, 라이브는 4시간
        마다 유니버스 자체를 갈아끼운다. 이 간극은 남아 있으므로 결과를
        '라이브 그대로'로 읽으면 안 된다.
        """
        import atlas_spot_backtest as bt
        src = Path(bt.__file__).read_text()
        assert 'get_backtest_universe' in src or 'BT_TIER1_SYMBOLS' in src


# ══════════════════════════════════════════════════════════════
#  ④ 백테스트 RS Gate — 이제 데이터가 임계값을 고를 수 있다
# ══════════════════════════════════════════════════════════════

class TestBacktestRankMap:
    """`build_momentum_rank_map`은 라이브 랭킹을 시계열로 재현한다.

    이게 없으면 MOMENTUM_RS_GATE_PCT는 어떤 값을 넣어도 백테스트 결과가
    같아서, WFO가 '검증했다'고 보고해도 실제로는 아무것도 검증하지 않는다.
    """

    @pytest.fixture
    def data(self):
        return {
            'STRONGUSDT': _series(200, 0.004, 0.01, seed=2),
            'WEAKUSDT':   _series(200, 0.000, 0.01, seed=3),
            'MIDUSDT':    _series(200, 0.002, 0.01, seed=5),
        }

    def test_builds_daily_map(self, data):
        import atlas_spot_backtest as bt
        m = bt.build_momentum_rank_map(data)
        assert m, '순위맵이 비어 있으면 게이트가 조용히 꺼진다'
        day = sorted(m)[-1]
        assert set(m[day]) == set(data)

    def test_percentile_matches_live_ordering(self, data):
        """같은 데이터에서 라이브 랭킹과 순서가 일치해야 패리티다."""
        import atlas_spot_backtest as bt
        m = bt.build_momentum_rank_map(data)
        day = sorted(m)[-1]
        bt_order = sorted(m[day], key=lambda s: m[day][s])
        live_order = uni.rank_by_momentum(list(data), data)
        assert bt_order == live_order

    def test_percentile_range(self, data):
        import atlas_spot_backtest as bt
        m = bt.build_momentum_rank_map(data)
        for day, row in m.items():
            for sym, pct in row.items():
                assert 0.0 <= pct < 1.0, f'{day} {sym} {pct}'

    def test_too_few_symbols_returns_empty(self):
        """1종목으로는 상대강도가 성립하지 않는다 — 조용히 1등을 주면 안 된다."""
        import atlas_spot_backtest as bt
        assert bt.build_momentum_rank_map({'AUSDT': _series(200, 0.003, 0.01)}) == {}

    def test_lookup_uses_prior_day_only(self):
        """당일 종가로 만든 순위를 당일 시가 진입에 쓰면 선행편향이다."""
        import atlas_spot_backtest as bt
        rank_map = {'2024-01-01': {'A': 0.1}, '2024-01-02': {'A': 0.9}}
        keys = sorted(rank_map)
        assert bt._rank_pct_asof(rank_map, keys, 'A', '2024-01-02') == 0.1
        assert bt._rank_pct_asof(rank_map, keys, 'A', '2024-01-03') == 0.9

    def test_lookup_before_history_is_none(self):
        import atlas_spot_backtest as bt
        rank_map = {'2024-01-05': {'A': 0.1}}
        assert bt._rank_pct_asof(rank_map, ['2024-01-05'], 'A', '2024-01-01') is None

    def test_lookup_without_map_is_none(self):
        import atlas_spot_backtest as bt
        assert bt._rank_pct_asof({}, [], 'A', '2024-01-01') is None


class TestBacktestRsGateApplied:
    """게이트가 실제로 진입을 막고 리스크를 키우는지 — 행동 검증."""

    @staticmethod
    def _run(strategy_id, rank_map, monkeypatch=None):
        import atlas_spot_backtest as bt
        ohlcv = _breakout_series()
        return bt.backtest_strategy(
            strategy_id, 'AUSDT', ohlcv,
            {}, '2021-01-01', '2022-12-31', rank_map=rank_map)

    def test_gate_blocks_bottom_ranked(self):
        import atlas_spot_backtest as bt
        gate = bt.MOMENTUM_RS_GATE_PCT
        try:
            bt.MOMENTUM_RS_GATE_PCT = 0.33
            low = {d: {'AUSDT': 0.90} for d in _date_keys()}
            _, diag = self._run('S6', low)
            assert diag.get('rs_gate_block', 0) > 0, (
                '하위권 심볼인데 한 번도 차단되지 않았다 — 게이트가 꺼져 있다')
        finally:
            bt.MOMENTUM_RS_GATE_PCT = gate

    def test_top_tier_gets_risk_boost(self):
        import atlas_spot_backtest as bt
        top = {d: {'AUSDT': 0.01} for d in _date_keys()}
        trades, diag = self._run('S6', top)
        if not trades:
            pytest.skip('합성 데이터에서 S6 진입 없음')
        assert diag.get('rs_top_tier', 0) > 0
        base, _ = self._run('S6', None)
        if base:
            assert trades[0].risk_pct > base[0].risk_pct, (
                f'주도주 부스트({bt.MOMENTUM_TOP_RISK_MULT}배)가 반영되지 않았다')

    def test_non_gate_strategy_unaffected(self):
        """RS Gate는 추세돌파 계열만 — S4(평균회귀)는 영향받지 않아야 한다."""
        import atlas_spot_backtest as bt
        assert 'S4' not in bt.MOMENTUM_RS_GATE_STRATS
        low = {d: {'AUSDT': 0.99} for d in _date_keys()}
        _, diag = self._run('S4', low)
        assert diag.get('rs_gate_block', 0) == 0

    def test_no_rank_map_disables_gate(self):
        """순위맵을 안 넘기면 게이트가 꺼진다(기존 호출부 호환)."""
        _, diag = self._run('S6', None)
        assert diag.get('rs_gate_block', 0) == 0
        assert diag.get('rs_top_tier', 0) == 0


def _date_keys():
    from datetime import datetime, timedelta
    d = datetime(2020, 12, 1)
    return [(d + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(900)]


def _breakout_series(n=600, seed=11):
    """돌파가 자주 나오는 합성 일봉 — S6 진입을 유도한다."""
    import numpy as np
    rng = np.random.default_rng(seed)
    px, out, ts = 100.0, [], 1606780800000   # 2020-12-01
    for i in range(n):
        o = px
        drift = 0.02 if (i // 20) % 2 == 0 else -0.005
        px *= (1 + drift + rng.normal(0, 0.02))
        px = max(px, 1.0)
        out.append([ts + i * 86400000, o, max(o, px) * 1.01,
                    min(o, px) * 0.99, px, 1e6 * (2.5 if drift > 0 else 1.0)])
    return out

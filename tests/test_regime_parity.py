"""
ATLAS — 레짐 엔진 패리티 / 거래 정지 가시화
================================
레짐은 깔때기 최상단이다. 여기가 어긋나면 아래 전부가 어긋난다.

발견된 두 문제:
  ① 라이브는 `classify_regime(adx, px, ema, atr, adx_4h, adx_slope)`로 부르는데
     백테스트는 뒤 두 인자를 빼고 불렀다. 그래서 백테스트에서는
     MICRO_RANGING과 'ADX 하락 중 → WEAK_TREND 강등'이 **한 번도 발생하지
     않았다**. 같은 날을 라이브는 거래정지/리스크 0.5배로 보는데 백테스트는
     TRENDING_UP으로 본 것이라, 검증한 레짐 경로를 실계좌가 따라가지 않았다.
  ② MICRO_RANGING·UNKNOWN이 REGIME_STRATEGY_MAP에 없어 `.get(기본값 [])`으로
     전 전략이 조용히 차단됐다. 로그에도 안 남아 운영자는 알 수 없다.

실행:
  pytest tests/test_regime_parity.py -v
"""

from pathlib import Path


import numpy as np
import pytest

import atlas_regime as R
import atlas_spot_backtest as bt
import atlas_spot_main as sm
from atlas_spot_config import REGIME_STRATEGY_MAP


# ══════════════════════════════════════════════════════════════
#  ① 맵 완전성 — 빠진 레짐은 조용한 거래 정지가 된다
# ══════════════════════════════════════════════════════════════

def _regime_constants() -> set:
    return {v for k, v in vars(R).items()
            if k.startswith('REGIME_') and isinstance(v, str)}


class TestRegimeMapCompleteness:
    def test_every_regime_is_declared(self):
        """classify_regime이 낼 수 있는 값은 전부 맵에 있어야 한다.
        빠지면 .get(기본값 [])으로 그 구간 전체가 조용히 거래 정지된다."""
        missing = sorted(r for r in _regime_constants()
                         if r not in REGIME_STRATEGY_MAP)
        assert not missing, (
            f'맵에 없는 레짐 {missing} — 해당 구간에 전 전략이 차단되지만 '
            f'로그에도 남지 않는다')

    def test_no_trade_regimes_are_intentional(self):
        """빈 목록은 '의도한 정지'여야 한다. 현재 무엇이 정지인지 고정한다."""
        blocked = sorted(k for k, v in REGIME_STRATEGY_MAP.items() if not v)
        assert blocked == ['CRISIS', 'MICRO_RANGING', 'UNKNOWN']

    def test_mapped_strategies_are_runnable(self):
        """맵에 적힌 전략은 전부 실제로 실행 가능해야 한다."""
        listed = {s for ss in REGIME_STRATEGY_MAP.values() for s in ss}
        ok, problems = sm.validate_active_strategies(sorted(listed))
        assert problems == [], f'맵에 있지만 실행 불가: {problems}'


# ══════════════════════════════════════════════════════════════
#  ② 라이브/백테스트 레짐 엔진 패리티
# ══════════════════════════════════════════════════════════════

class TestClassifyParity:
    def test_extra_signals_change_outcome(self):
        """adx_4h / adx_slope를 빼면 다른 레짐이 나온다 — 백테스트가 이걸
        빠뜨리면 전혀 다른 경로를 검증하게 된다."""
        args = dict(adx=28, btc_price=110, ema200=100, atr_pct=0.03)
        assert R.classify_regime(**args) == 'TRENDING_UP'
        assert R.classify_regime(**args, adx_4h=15, adx_slope=1) == 'MICRO_RANGING'
        assert R.classify_regime(**args, adx_4h=30, adx_slope=-2) == 'WEAK_TREND'

    def test_build_regime_map_accepts_4h(self):
        import inspect
        sig = inspect.signature(bt.build_regime_map)
        assert 'btc_4h_ohlcv' in sig.parameters, (
            '백테스트가 4H를 받지 못하면 MICRO_RANGING이 영원히 발생하지 않는다')

    def _series(self, n, seed, trend):
        rng = np.random.default_rng(seed)
        ts, px, rows = 1609459200000, 100.0, []
        for i in range(n):
            o = px
            px *= (1 + trend + rng.normal(0, 0.02))
            rows.append([ts + i * 86400000, o, max(o, px) * 1.01,
                         min(o, px) * 0.99, px, 1e6])
        return rows

    def test_adx_slope_applied_without_4h(self):
        """4H가 없어도 adx_slope는 1D만으로 계산되므로 반영돼야 한다."""
        rows = self._series(400, 3, 0.004)
        m = bt.build_regime_map(rows)
        assert m, '레짐맵이 비었다'
        # 강등이 동작하면 WEAK_TREND가 적어도 한 번은 나온다
        assert 'WEAK_TREND' in set(m.values())

    def test_4h_input_can_produce_micro_ranging(self):
        """4H를 주면 라이브에서만 나던 MICRO_RANGING이 백테스트에도 나온다."""
        d1 = self._series(400, 3, 0.004)
        # 4H는 잔잔하게(ADX 낮게) 만들어 MICRO_RANGING 조건을 유도
        rng = np.random.default_rng(11)
        ts, px, d4 = 1609459200000, 100.0, []
        for i in range(400 * 6):
            o = px
            px *= (1 + rng.normal(0, 0.0015))
            d4.append([ts + i * 4 * 3600000, o, max(o, px) * 1.001,
                       min(o, px) * 0.999, px, 1e6])
        m_no4h = bt.build_regime_map(d1)
        m_4h = bt.build_regime_map(d1, d4)
        assert m_no4h != m_4h, '4H를 넘겼는데 레짐 경로가 그대로다'

    def test_empty_4h_is_safe(self):
        rows = self._series(400, 3, 0.004)
        assert bt.build_regime_map(rows, []) == bt.build_regime_map(rows)


# ══════════════════════════════════════════════════════════════
#  ③ 거래 정지 가시화
# ══════════════════════════════════════════════════════════════

class TestNoTradeAlert:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        self.sent = []
        monkeypatch.setattr(sm, '_tg', lambda m: self.sent.append(m))
        monkeypatch.setattr(sm, '_state', {})

    def test_silent_before_threshold(self):
        sm.check_no_trade_regime('CRISIS', 1000.0)
        sm.check_no_trade_regime('CRISIS', 1000.0 + 3600 * 2)
        assert self.sent == []

    def test_alerts_after_threshold(self):
        sm.check_no_trade_regime('CRISIS', 1000.0)
        sm.check_no_trade_regime('CRISIS', 1000.0 + 3600 * 7)
        assert len(self.sent) == 1 and '거래 정지' in self.sent[0]

    def test_alerts_only_once(self):
        sm.check_no_trade_regime('CRISIS', 1000.0)
        for h in (7, 8, 20):
            sm.check_no_trade_regime('CRISIS', 1000.0 + 3600 * h)
        assert len(self.sent) == 1

    def test_recovery_notified(self):
        sm.check_no_trade_regime('CRISIS', 1000.0)
        sm.check_no_trade_regime('CRISIS', 1000.0 + 3600 * 7)
        sm.check_no_trade_regime('TRENDING_UP', 1000.0 + 3600 * 8)
        assert any('회복' in m for m in self.sent)

    def test_trading_regime_never_alerts(self):
        for h in range(0, 48, 4):
            sm.check_no_trade_regime('RANGING', 1000.0 + 3600 * h)
        assert self.sent == []

    def test_micro_ranging_counts_as_blocked(self):
        """맵에 명시됐지만 빈 목록이므로 여전히 거래 정지다."""
        sm.check_no_trade_regime('MICRO_RANGING', 1000.0)
        sm.check_no_trade_regime('MICRO_RANGING', 1000.0 + 3600 * 7)
        assert len(self.sent) == 1

    def test_unknown_regime_counts_as_blocked(self):
        sm.check_no_trade_regime('UNKNOWN', 1000.0)
        sm.check_no_trade_regime('UNKNOWN', 1000.0 + 3600 * 7)
        assert len(self.sent) == 1


# ══════════════════════════════════════════════════════════════
#  형성 중인 봉 · ADX 윈도우 길이 (2026-08 감사)
# ══════════════════════════════════════════════════════════════

class TestFormingBarExcluded:
    """거래소는 **아직 끝나지 않은** 현재 봉을 마지막에 붙여 준다.
    그 값은 하루 종일 변하므로 지표에 넣으면 레짐이 하루에도 몇 번 뒤집힌다 —
    백테스트가 한 번도 본 적 없는 동작이다. 합성 데이터에서 약 15%의
    레짐 판정이 달랐다."""

    def test_drops_last_bar(self):
        import atlas_regime as rg
        oh = [[i, 1, 2, 0.5, 1.5, 100] for i in range(10)]
        assert rg._drop_forming_bar(oh) == oh[:-1]

    def test_short_input_is_safe(self):
        import atlas_regime as rg
        assert rg._drop_forming_bar([]) == []
        assert rg._drop_forming_bar(None) == []
        assert len(rg._drop_forming_bar([[1, 1, 1, 1, 1, 1]])) == 1

    def test_does_not_mutate_input(self):
        import atlas_regime as rg
        oh = [[i, 1, 2, 0.5, 1.5, 100] for i in range(5)]
        rg._drop_forming_bar(oh)
        assert len(oh) == 5, '원본을 건드리면 호출부가 조용히 영향받는다'

    def test_update_regime_uses_it(self):
        """호출되지 않으면 헬퍼만 있고 동작은 그대로다."""
        src = Path(__import__('atlas_regime').__file__).read_text()
        body = src[src.index('def update_regime'):]
        assert '_drop_forming_bar(ohlcv)' in body


class TestAdxWindowParity:
    """Wilder ADX는 재귀 평활이라 **워밍업 길이가 값을 바꾼다.**
    라이브 50봉 vs 백테스트 51봉 오프바이원으로 같은 날 ADX가 달랐다
    (36.15 vs 35.72). 레짐 경계에서 이 차이가 분류를 뒤집는다."""

    @staticmethod
    def _series(n=180, seed=5):
        import numpy as np
        rng = np.random.default_rng(seed)
        px, out = 100.0, []
        for i in range(n):
            o = px
            px = max(px * (1 + 0.004 + rng.normal(0, 0.02)), 1.0)
            out.append([i * 86400000, o, max(o, px) * 1.01, min(o, px) * 0.99, px, 1e6])
        return out

    def test_window_lengths_match(self):
        from atlas_indicators import calc_adx
        from atlas_regime import REGIME_BTC_LOOKBACK as LB
        from atlas_regime import _drop_forming_bar
        o = _drop_forming_bar(self._series())
        live = calc_adx(o[-LB:], 14)
        bt_win = o[max(0, len(o) - 1 - LB + 1): len(o)]
        assert len(bt_win) == LB, f'백테스트 윈도우 {len(bt_win)}봉 ≠ 라이브 {LB}봉'
        assert calc_adx(bt_win, 14) == pytest.approx(live)

    def test_slope_endpoint_equals_level(self):
        """같은 함수가 레벨과 기울기를 한 봉 어긋난 기준으로 읽으면 안 된다."""
        from atlas_indicators import calc_adx
        from atlas_regime import REGIME_BTC_LOOKBACK as LB
        from atlas_regime import _drop_forming_bar
        o = _drop_forming_bar(self._series())
        level = calc_adx(o[-LB:], 14)
        series = [calc_adx(o[-(LB + k):(-k if k else None)], 14)
                  for k in range(2, -1, -1)]
        assert series[-1] == pytest.approx(level)

    def test_slope_matches_backtest(self):
        from atlas_indicators import calc_adx
        from atlas_regime import REGIME_BTC_LOOKBACK as LB
        from atlas_regime import _drop_forming_bar
        o = _drop_forming_bar(self._series())
        series = [calc_adx(o[-(LB + k):(-k if k else None)], 14)
                  for k in range(2, -1, -1)]
        hist = [calc_adx(o[max(0, i - LB + 1):i + 1], 14)
                for i in range(len(o) - 3, len(o))]
        assert (series[-1] - series[0]) == pytest.approx(hist[-1] - hist[-3])

    def test_backtest_uses_named_constant(self):
        """매직넘버 50이 남아 있으면 상수를 바꿔도 한쪽만 움직인다."""
        import atlas_spot_backtest as bt
        src = Path(bt.__file__).read_text()
        body = src[src.index('def build_regime_map'):src.index('def _bt_exit_decision')]
        assert 'i - 50' not in body and 'j - 50' not in body
        assert 'REGIME_BTC_LOOKBACK' in body


class TestUpdateRegimeMatchesBacktest:
    """앞의 검사들은 슬라이스 로직을 **복제**해서 확인한다 — 프로덕션 코드가
    바뀌어도 복제본은 그대로라 변형을 놓친다(실제로 기울기 범위를 되돌리는
    변형이 살아남았다). 여기서는 진짜 `update_regime`을 돌려
    `build_regime_map`이 같은 날짜에 내는 값과 직접 대조한다."""

    class _FakeEx:
        def __init__(self, d1, d4):
            self._d1, self._d4 = d1, d4

        def fetch_ohlcv(self, symbol, timeframe, limit=None):
            return list(self._d1 if timeframe == '1d' else self._d4)

    @staticmethod
    def _bars(n, seed, step_ms):
        import numpy as np
        rng = np.random.default_rng(seed)
        px, out = 100.0, []
        base = 1577836800000
        for i in range(n):
            o = px
            drift = 0.010 if (i // 30) % 2 == 0 else -0.006
            px = max(px * (1 + drift + rng.normal(0, 0.02)), 1.0)
            out.append([base + i * step_ms, o, max(o, px) * 1.012,
                        min(o, px) * 0.988, px, 1e6])
        return out

    def test_adx_and_slope_match_build_regime_map(self, monkeypatch):
        import atlas_regime as rg
        import atlas_spot_backtest as bt

        d1 = self._bars(260, 11, 86400000)
        d4 = self._bars(260, 12, 4 * 3600000)

        state = rg.update_regime(self._FakeEx(d1, d4))

        # 라이브는 형성 중인 마지막 봉을 버린다 → 백테스트에서 대응하는
        # 날짜는 '끝에서 두 번째' 일봉이다.
        rmap = bt.build_regime_map(d1[:-1], d4)
        import pandas as pd
        last_date = str(pd.to_datetime(d1[-2][0], unit='ms', utc=True).date())

        assert state.regime == rmap[last_date], (
            f'같은 날짜에 라이브 {state.regime} vs 백테스트 {rmap[last_date]}')

    def test_slope_is_not_stale_by_one_bar(self, monkeypatch):
        """기울기 끝점이 레벨보다 이르면 이 검사가 실패한다."""
        import atlas_regime as rg
        from atlas_indicators import calc_adx
        from atlas_regime import REGIME_BTC_LOOKBACK as LB

        d1 = self._bars(260, 21, 86400000)
        d4 = self._bars(260, 22, 4 * 3600000)
        state = rg.update_regime(self._FakeEx(d1, d4))

        completed = d1[:-1]
        expect_level = calc_adx(completed[-LB:], 14)
        expect_slope = expect_level - calc_adx(completed[-(LB + 2):-2], 14)

        assert state.adx == pytest.approx(expect_level), (
            'ADX 레벨이 완성봉 기준이 아니다')
        assert state.adx_slope == pytest.approx(expect_slope), (
            f'기울기 {state.adx_slope:+.3f} ≠ 기대 {expect_slope:+.3f} — '
            f'끝점이 레벨과 다른 봉을 보고 있다')

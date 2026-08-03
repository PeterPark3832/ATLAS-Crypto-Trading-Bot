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

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

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

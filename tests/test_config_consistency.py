"""
ATLAS — 전략 구성 정합성 테스트
================================
REGIME_STRATEGY_MAP / DEFAULT_ACTIVE_STRATEGIES / 전략 레지스트리 간의
배선 누락을 검증합니다.

과거 사례: S7V4가 기본 활성 전략이면서 어떤 레짐에도 배정되지 않아
영구 진입불가(죽은 전략) 상태였고, TRENDING_DOWN 담당 S4는 기본
전략에서 빠져 있어 하락장에서 아무 전략도 진입하지 못했음.

실행:
  pytest tests/test_config_consistency.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas_spot_config import (
    REGIME_STRATEGY_MAP, DEFAULT_ACTIVE_STRATEGIES,
    STRATEGY_TIMEFRAMES, STRATEGY_NAMES,
)
from atlas_spot_strategies import CALC_FUNCS, SIGNAL_FUNCS


def _mapped_strategies() -> set:
    out = set()
    for strats in REGIME_STRATEGY_MAP.values():
        out.update(strats)
    return out


class TestDefaultStrategiesRoutable:
    def test_every_default_strategy_is_in_some_regime(self):
        """기본 활성 전략이 어떤 레짐에도 배정되지 않으면 영구 진입불가 —
        지표 계산만 낭비하는 죽은 전략이 된다 (과거 S7V4 사례)."""
        mapped = _mapped_strategies()
        dead = [s for s in DEFAULT_ACTIVE_STRATEGIES if s not in mapped]
        assert dead == [], f'레짐 미배정 기본 전략(영구 진입불가): {dead}'

    def test_every_mapped_strategy_is_default_active(self):
        """레짐 맵에 배정된 전략이 기본 활성 목록에 없으면 해당 레짐이
        기본 구성에서 마비된다 (과거 TRENDING_DOWN→S4 사례)."""
        missing = [s for s in _mapped_strategies() if s not in DEFAULT_ACTIVE_STRATEGIES]
        assert missing == [], f'기본 전략에서 빠진 레짐 담당 전략: {missing}'


class TestMappedStrategiesRegistered:
    def test_mapped_strategies_have_calc_and_signal_funcs(self):
        for s in _mapped_strategies():
            assert s in CALC_FUNCS, f'{s}: CALC_FUNCS 미등록'
            assert s in SIGNAL_FUNCS, f'{s}: SIGNAL_FUNCS 미등록'

    def test_mapped_strategies_have_timeframe(self):
        for s in _mapped_strategies():
            assert s in STRATEGY_TIMEFRAMES, f'{s}: STRATEGY_TIMEFRAMES 미등록'
            assert STRATEGY_TIMEFRAMES[s] in ('4h', '1d')

    def test_mapped_strategies_have_names(self):
        for s in _mapped_strategies():
            assert s in STRATEGY_NAMES, f'{s}: STRATEGY_NAMES 미등록'


class TestMainUsesDefaultList:
    def test_argparse_default_matches_config(self):
        """main.py --strategies 기본값이 config 상수와 동기화되는지 확인."""
        import atlas_spot_main as sm
        assert sm._state['active_strategies'] == list(DEFAULT_ACTIVE_STRATEGIES)

    def test_dashboard_mirror_map_matches_config(self):
        """대시보드의 수동 미러 맵이 config와 어긋나면 브리핑이 거짓말을 한다."""
        import atlas_web_dashboard as wd
        assert wd._REGIME_STRAT_MAP == REGIME_STRATEGY_MAP

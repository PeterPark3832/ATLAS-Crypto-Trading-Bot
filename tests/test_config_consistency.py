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

from pathlib import Path


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

    def test_dashboard_imports_map_from_config(self):
        """대시보드가 레짐 맵을 config에서 직접 import하는지 — 수동 복사본이면
        drift로 브리핑이 거짓말을 하게 된다 (이제 동일 객체여야 함)."""
        import atlas_web_dashboard as wd
        assert wd._REGIME_STRAT_MAP is REGIME_STRATEGY_MAP

    def test_dashboard_metadata_covers_live_strategies(self):
        """라이브 라우팅되는 전 전략이 대시보드 표시 메타데이터에 존재해야
        브리핑에서 이름/조건이 누락되지 않는다."""
        import atlas_web_dashboard as wd
        from atlas_spot_config import LIVE_STRATEGIES
        for s in LIVE_STRATEGIES:
            assert s in wd._STRAT_FULL, f'{s}: 한글명 누락'
            assert s in wd._STRAT_CONDITION, f'{s}: 진입조건 누락'

    def test_live_strategies_derived_from_regime_map(self):
        from atlas_spot_config import LIVE_STRATEGIES
        assert set(LIVE_STRATEGIES) == _mapped_strategies()


# ══════════════════════════════════════════════════════════════
#  죽은 설정 상수 래칫
# ══════════════════════════════════════════════════════════════

# 운영 코드가 참조하지 않는 config 상수 — **의도적으로 남겨둔 것만** 여기 적는다.
# 각 항목에 이유를 붙여, 새로 죽은 상수가 생기면 눈에 띄게 한다.
KNOWN_UNREFERENCED = {
    'BASE_DIR':               'config 내부에서 경로 조립에 쓰인다(파생 상수)',
    'BT_SPOT_SLIPPAGE':       '티어 폴백용이었으나 _get_slippage의 tier3가 catch-all이라 도달하지 않는다',
    'S2_EXIT_TYPE':           'S2는 라우팅에서 빠진 연구용 전략',
    'S5_EXIT_TYPE':           'S5 청산은 _manage_position이 BB 상단을 실시간 갱신해 처리한다',
    'SPOT_MIN_ALLOC_PCT':     '최소 배분 하한 — 현재는 SPOT_MIN_ORDER_USDT가 그 역할을 한다',
    'UNIVERSE_QUOTE_CURRENCY': '유니버스 필터가 USDT를 직접 쓴다',
}


def _config_constants() -> set:
    import re
    src = (Path(__file__).parent.parent / 'atlas_spot_config.py').read_text(encoding='utf-8')
    return set(re.findall(r'^([A-Z][A-Z0-9_]{3,})\s*=', src, re.M))


def _production_blob() -> str:
    root = Path(__file__).parent.parent
    return '\n'.join(p.read_text(encoding='utf-8') for p in root.glob('*.py')
                     if p.name != 'atlas_spot_config.py')


class TestNoNewDeadConstants:
    """설정 상수가 운영 코드에서 참조되지 않으면 **그 설정은 동작하지 않는다**.

    이 저장소는 실제로 이 부류로 크게 당했다 — 죽은 상수 9개가 "값을 바꿔도
    거래가 그대로"인 상태를 만들었고(d8f8e38), ruff의 미사용 import 경고
    하나에서 실마리가 잡혔다. 상수는 import되지 않으면 경고조차 안 나므로
    별도 래칫이 필요하다.

    새 상수를 넣고 배선을 잊으면 여기서 잡힌다. 의도적으로 남기는 것이라면
    KNOWN_UNREFERENCED 에 **이유와 함께** 등록해야 한다.
    """

    def test_no_unlisted_dead_constants(self):
        import re
        blob = _production_blob()
        dead = sorted(n for n in _config_constants()
                      if not re.search(rf'\b{n}\b', blob))
        unexpected = [n for n in dead if n not in KNOWN_UNREFERENCED]
        assert not unexpected, (
            f'운영 코드가 참조하지 않는 새 설정 상수: {unexpected}\n'
            f'  배선을 잊었다면 그 설정은 아무 효과가 없다. 의도적이라면 '
            f'KNOWN_UNREFERENCED 에 이유와 함께 등록할 것.')

    def test_ratchet_is_not_stale(self):
        """이미 배선된 상수가 목록에 남아 있으면 래칫이 헐거워진다."""
        import re
        blob = _production_blob()
        stale = sorted(n for n in KNOWN_UNREFERENCED
                       if re.search(rf'\b{n}\b', blob))
        assert not stale, (
            f'이제 참조되는 상수가 예외 목록에 남아 있다: {stale} — 목록에서 뺄 것')

    def test_listed_constants_still_exist(self):
        """이름이 바뀌거나 삭제된 항목이 목록에 남으면 의미가 없다."""
        missing = sorted(set(KNOWN_UNREFERENCED) - _config_constants())
        assert not missing, f'config에 없는 상수가 목록에 있다: {missing}'

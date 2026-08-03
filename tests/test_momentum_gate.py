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
        assert (49 / 50) > intended, '0.33이면 하위권이 실제로 차단된다'

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

    def test_in_wfo_grid(self):
        """데이터가 판단하도록 WFO 그리드에 올라가 있어야 한다."""
        import reoptimize as ro
        assert 'MOMENTUM_RS_GATE_PCT' in ro.GRIDS['S6']
        assert cfg.MOMENTUM_RS_GATE_PCT in ro.GRIDS['S6']['MOMENTUM_RS_GATE_PCT']

    def test_grid_can_patch_it(self):
        import reoptimize as ro
        before = sm.MOMENTUM_RS_GATE_PCT
        with ro.override_params({'MOMENTUM_RS_GATE_PCT': 0.33}):
            assert sm.MOMENTUM_RS_GATE_PCT == 0.33
        assert sm.MOMENTUM_RS_GATE_PCT == before


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

    def test_live_ranking_not_used_by_backtest(self):
        """백테스트가 rank_by_momentum을 쓰지 않는다는 사실을 고정한다.
        (나중에 쓰게 되면 이 테스트가 실패하며 문서 갱신을 강제한다)"""
        import atlas_spot_backtest as bt
        assert 'rank_by_momentum' not in Path(bt.__file__).read_text()

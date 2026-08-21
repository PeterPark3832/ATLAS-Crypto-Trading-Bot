"""
ATLAS — 청산 판정 (라이브 ↔ 백테스트)
================================
청산 판정을 라이브·백테스트 양쪽에서 **같은 모양의 순수함수**로 뽑아 두면,
한쪽에만 있는 규칙이 눈에 보인다. 실제로 그 분리 과정에서 백테스트가
모델링하지 않는 라이브 전용 청산 2건이 드러났다.

이 파일은 두 가지를 한다:
  ① 청산 우선순위·경계 조건을 고정한다(순수함수라 직접 시험 가능)
  ② **알려진 파리티 격차를 테스트로 박아 둔다** — 나중에 백테스트가
     같은 규칙을 구현하면 이 테스트가 실패하며 문서 갱신을 강제한다

실행:
  pytest tests/test_exit_decision_parity.py -v
"""

from pathlib import Path


import pandas as pd

import atlas_spot_backtest as bt
import atlas_spot_main as sm

D = sm._live_exit_decision


def _df(**cols):
    n = 3
    base = {'close': [100.0] * n, 'bb_mid': [999.0] * n, 'bb_upper': [999.0] * n}
    base.update({k: [v] * n for k, v in cols.items()})
    return pd.DataFrame(base)


# ══════════════════════════════════════════════════════════════
#  ① 우선순위와 경계
# ══════════════════════════════════════════════════════════════

class TestPriority:
    def test_sl_wins_over_tp(self):
        """한 틱에서 둘 다 만족하면 불리한 쪽(SL)을 택해 과대평가를 막는다."""
        assert D('S4', 'X', None, 0, price=50, entry=100, sl=90, tp=40,
                 bars_held=0, max_hold=0) == 'SL'

    def test_sl_is_inclusive(self):
        assert D('S4', 'X', None, 0, 90, 100, 90, 0, 0, 0) == 'SL'
        assert D('S4', 'X', None, 0, 90.01, 100, 90, 0, 0, 0) is None

    def test_tp_is_inclusive(self):
        assert D('S4', 'X', None, 0, 110, 100, 90, 110, 0, 0) == 'TP'
        assert D('S4', 'X', None, 0, 109.99, 100, 90, 110, 0, 0) is None

    def test_tp_zero_disables(self):
        assert D('S4', 'X', None, 0, 1e9, 100, 90, 0, 0, 0) is None

    def test_time_exit(self):
        assert D('S4', 'X', None, 0, 100, 100, 90, 0, 5, 5) == 'TIME'
        assert D('S4', 'X', None, 0, 100, 100, 90, 0, 4, 5) is None

    def test_max_hold_zero_disables_time(self):
        assert D('S4', 'X', None, 0, 100, 100, 90, 0, 9999, 0) is None

    def test_no_condition_returns_none(self):
        assert D('S4', 'X', None, 0, 100, 100, 90, 200, 0, 0) is None

    def test_is_side_effect_free(self):
        """부수효과가 있으면 판정만 시험할 수 없다 — 매도가 호출되면 실패."""
        called = []
        orig = sm._spot_sell
        sm._spot_sell = lambda *a, **k: called.append(1)
        try:
            D('S4', 'X', None, 0, 10, 100, 90, 0, 0, 0)
        finally:
            sm._spot_sell = orig
        assert not called


class TestCrossExit:
    def test_cross_calls_exit_fn(self, monkeypatch):
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', lambda df, i: True)
        assert D('S3', 'X', _df(), 0, 100, 100, 90, 0, 0, 0) == 'CROSS'

    def test_s6_gets_entry_price(self, monkeypatch):
        seen = {}
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S6',
                            lambda df, i, e: seen.setdefault('entry', e) or False)
        D('S6', 'X', _df(), 0, 100, 123.0, 90, 0, 0, 0)
        assert seen['entry'] == 123.0

    def test_exit_fn_failure_does_not_exit(self, monkeypatch, caplog):
        """예외가 나면 CROSS는 건너뛰되 **로그를 남긴다**(조용한 삼킴 금지)."""
        def boom(df, i):
            raise RuntimeError('지표 결측')
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', boom)
        with caplog.at_level('WARNING'):
            assert D('S3', 'X', _df(), 0, 100, 100, 90, 0, 0, 0) is None
        assert any('청산 조건 평가 실패' in r.message for r in caplog.records)

    def test_sl_still_works_when_exit_fn_broken(self, monkeypatch):
        """청산 함수가 죽어도 SL 보호는 살아 있어야 한다."""
        def boom(df, i):
            raise RuntimeError('x')
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', boom)
        assert D('S3', 'X', _df(), 0, 10, 100, 90, 0, 0, 0) == 'SL'


# ══════════════════════════════════════════════════════════════
#  ② 알려진 파리티 격차 — 백테스트가 모델링하지 않는 라이브 전용 규칙
# ══════════════════════════════════════════════════════════════

class TestKnownParityGaps:
    """이 격차들은 성과 차이를 만든다. 고치려면 백테스트 청산 규칙을 바꿔야
    하고, 그러면 과거 WFO 검증이 전부 무효가 된다. 그래서 지금은
    **명시**만 하고, 다음 재최적화 때 함께 처리한다."""

    def test_bb_mid_is_live_only(self):
        assert D('S4', 'X', _df(bb_mid=95.0), 0, 96, 100, 90, 0, 0, 0) == 'BB_MID'
        assert 'S4' not in bt.EXIT_CHECK_FUNCS, (
            '백테스트가 S4 청산을 구현했다면 이 격차는 해소된 것이다 — '
            '_live_exit_decision docstring의 경고를 갱신할 것')

    def test_bb_mid_only_for_s4(self):
        assert D('S5', 'X', _df(bb_mid=95.0), 0, 96, 100, 90, 0, 0, 0) is None

    def test_bb_mid_needs_column(self):
        assert D('S4', 'X', pd.DataFrame({'close': [1.0]}), 0,
                 96, 100, 90, 0, 0, 0) is None

    def test_bb_mid_nan_is_ignored(self):
        assert D('S4', 'X', _df(bb_mid=float('nan')), 0,
                 96, 100, 90, 0, 0, 0) is None

    def test_s5_live_tp_update_is_live_only(self):
        """라이브는 매 폴링마다 TP를 bb_upper로 옮긴다. 백테스트는 정적 TP."""
        assert 'bb_upper' not in Path(bt.__file__).read_text(), (
            '백테스트가 실시간 TP 갱신을 구현했다면 격차가 해소된 것이다')

    def test_gap_is_documented(self):
        doc = sm._live_exit_decision.__doc__ or ''
        assert 'BB_MID' in doc and '백테스트' in doc, (
            '격차가 코드에 적혀 있지 않으면 다음 사람이 모르고 지나친다')


class TestSharedReasonVocabulary:
    """두 경로가 서로 다른 사유 문자열을 쓰면 성과 비교가 어긋난다."""

    def test_backtest_reasons_are_subset(self):
        live = {'SL', 'TP', 'CROSS', 'TIME', 'BB_MID'}
        src = Path(bt.__file__).read_text()
        bt_body = src[src.index('def _bt_exit_decision'):src.index('def _bt_kelly_scale')]
        used = {r for r in ('SL', 'TP', 'CROSS', 'TIME') if f"'{r}'" in bt_body}
        assert used <= live and used == {'SL', 'TP', 'CROSS', 'TIME'}

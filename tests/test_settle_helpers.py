"""정산 공통 헬퍼 (_settle_closed_position / _try_settle_via_stop_fill) 직접 검증.

PR4에서 5개 정산 사본을 통합하며 도입된 헬퍼들이다. 특히 sl_for_r
파라미터는 사본 간 발산(4곳 orig_sl vs 검증루프 pos['sl'])을 **보존**하기
위한 것으로, 이 파라미터화가 정확해야 다음 커밋(검증루프 분모 수정)의
회귀 테스트가 성립한다.
"""
from datetime import datetime, timedelta, timezone
import threading

import atlas_rules
import atlas_spot_main as sm


def _ts_hours_ago(h: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()


class TestSettleClosedPosition:
    def _run(self, monkeypatch, **kw):
        """헬퍼 실행 후 (반환값, _log_trade 호출 kwargs, 삭제 호출, day_pnl)."""
        logged: dict = {}
        deleted: list = []
        state = {'day_pnl': 0.0}

        def fake_log_trade(strategy, symbol, entry, exit_, qty, cost,
                           pnl_u, pnl_p, pnl_r, hold, reason, regime,
                           entry_ts, slip_pct=0.0):
            logged.update(pnl_u=pnl_u, pnl_p=pnl_p, pnl_r=pnl_r, hold=hold,
                          reason=reason, regime=regime, slip_pct=slip_pct,
                          qty=qty, cost=cost)
            return pnl_u - 0.1  # net = gross - 수수료 흉내

        monkeypatch.setattr(sm, '_log_trade', fake_log_trade)
        monkeypatch.setattr(sm, '_delete_position',
                            lambda s, y: deleted.append((s, y)))
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())

        defaults = dict(entry_price=100.0, exit_price=110.0, qty=2.0,
                        cost_usdt=200.0, reason='TP', regime='trend',
                        entry_ts=_ts_hours_ago(3.0), sl_for_r=95.0)
        defaults.update(kw)
        ret = sm._settle_closed_position('S3', 'BTCUSDT', **defaults)
        return ret, logged, deleted, state

    def test_pnl_trio_and_side_effects(self, monkeypatch):
        (net, pnl_u, pnl_p, hold), logged, deleted, state = self._run(monkeypatch)
        assert pnl_u == 20.0                       # (110-100)*2
        assert pnl_p == 0.1                        # (110-100)/100
        assert logged['pnl_r'] == 2.0              # 20 / (5*2)
        assert deleted == [('S3', 'BTCUSDT')]
        assert net == 19.9
        assert state['day_pnl'] == 19.9
        assert 2.9 < hold < 3.1
        assert logged['slip_pct'] == 0.0

    def test_sl_for_r_denominator_is_caller_controlled(self, monkeypatch):
        # 같은 손익이라도 분모 SL이 다르면 pnl_r이 달라진다 — 발산 보존의 핵심.
        # orig_sl(95, 위험 5) 기준 R=2.0 vs 추적조정 sl(99, 위험 1) 기준 R=10.0.
        _, logged_orig, _, _ = self._run(monkeypatch, sl_for_r=95.0)
        _, logged_trail, _, _ = self._run(monkeypatch, sl_for_r=99.0)
        assert logged_orig['pnl_r'] == 2.0
        assert logged_trail['pnl_r'] == 10.0       # 검증루프 사본의 과대평가 재현

    def test_zero_qty_records_zero_r_instead_of_crashing(self, monkeypatch):
        (net, pnl_u, _, _), logged, deleted, _ = self._run(monkeypatch, qty=0.0)
        assert pnl_u == 0.0
        assert logged['pnl_r'] == 0
        assert deleted  # 퇴화 상태여도 포지션 정리는 진행

    def test_zero_sl_dist_records_zero_r(self, monkeypatch):
        _, logged, _, _ = self._run(monkeypatch, sl_for_r=100.0)
        assert logged['pnl_r'] == 0

    def test_round_hold_rounds_to_2dp(self, monkeypatch):
        _, logged, _, _ = self._run(monkeypatch, round_hold=True)
        assert logged['hold'] == round(logged['hold'], 2)

    def test_slip_pct_passthrough(self, monkeypatch):
        _, logged, _, _ = self._run(monkeypatch, slip_pct=0.0042)
        assert logged['slip_pct'] == 0.0042

    def test_zero_entry_price_degenerate_guard(self, monkeypatch):
        _, logged, _, _ = self._run(monkeypatch, entry_price=0.0, sl_for_r=0.0)
        assert logged['pnl_p'] == 0


class TestSymbolHelpers:
    """PR4에서 main의 인라인 문자열 치환 ~8곳을 이 헬퍼들로 교체했다."""

    def test_to_ccxt(self):
        assert atlas_rules.to_ccxt('BTCUSDT') == 'BTC/USDT'
        assert atlas_rules.to_ccxt('1000PEPEUSDT') == '1000PEPE/USDT'

    def test_base_of(self):
        assert atlas_rules.base_of('BTC/USDT') == 'BTC'
        assert atlas_rules.base_of('SOL/USDT') == 'SOL'

    def test_roundtrip(self):
        for sym in ('BTCUSDT', 'ETHUSDT', 'FETUSDT'):
            assert atlas_rules.base_of(atlas_rules.to_ccxt(sym)) + 'USDT' == sym


class TestTrySettleViaStopFill:
    def test_no_protective_ids_returns_false_without_calling(self, monkeypatch):
        called = []
        monkeypatch.setattr(sm, '_handle_stop_order_state',
                            lambda *a: called.append(a) or True)
        ok = sm._try_settle_via_stop_fill('S3', 'BTCUSDT', 'BTC/USDT',
                                          {'sl_order_id': '', 'tp_order_id': None},
                                          '[t]')
        assert ok is False
        assert called == []

    def test_fill_detected_returns_true(self, monkeypatch):
        monkeypatch.setattr(sm, '_handle_stop_order_state',
                            lambda *a: True)
        assert sm._try_settle_via_stop_fill(
            'S3', 'BTCUSDT', 'BTC/USDT', {'sl_order_id': 'oid1'}, '[t]') is True

    def test_not_filled_returns_false(self, monkeypatch):
        monkeypatch.setattr(sm, '_handle_stop_order_state',
                            lambda *a: False)
        assert sm._try_settle_via_stop_fill(
            'S3', 'BTCUSDT', 'BTC/USDT', {'tp_order_id': 'oid2'}, '[t]') is False

    def test_lookup_failure_falls_back_to_manual_path(self, monkeypatch, caplog):
        # 조회 실패는 삼켜서 False — 포지션이 DB에 떠돌면 안 되기 때문.
        def boom(*a):
            raise RuntimeError('network down')
        monkeypatch.setattr(sm, '_handle_stop_order_state', boom)
        ok = sm._try_settle_via_stop_fill(
            'S3', 'BTCUSDT', 'BTC/USDT', {'sl_order_id': 'oid3'}, '[검증] BTCUSDT')
        assert ok is False
        assert '[검증] BTCUSDT 보호주문 체결 확인 실패' in caplog.text

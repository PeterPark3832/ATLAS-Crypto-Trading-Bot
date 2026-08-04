"""
ATLAS — 거래소 측 스탑 주문 단위 테스트
================================
_place_stop_loss_order / _cancel_stop_order / _handle_stop_order_state와
_spot_buy(스탑 등록), _spot_sell(선취소), _manage_position(체결 감지),
main() 킬스위치 기동 가드를 검증합니다.

실행:
  pytest tests/test_stop_orders.py -v
"""

import os
import sys
import threading
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  헬퍼: 스탑 주문 지원 가짜 거래소
# ══════════════════════════════════════════════════════════════

class _FakeStopExchange:
    def __init__(self):
        self.calls = []                 # 호출 순서 추적 ('create_order'|'cancel'|'buy'|'sell'|...)
        self.create_order_result = {'id': 'STOP-1'}   # dict 또는 Exception
        self.cancel_raises = None
        self.fetched_order = None       # dict / Exception / None
        self.free_balance = {}

    def create_order(self, ccxt_sym, otype, side, qty, price, params=None):
        self.calls.append(('create_order', ccxt_sym, otype, side, qty, price, params))
        if isinstance(self.create_order_result, Exception):
            raise self.create_order_result
        return self.create_order_result

    def cancel_order(self, order_id, ccxt_sym):
        self.calls.append(('cancel', order_id, ccxt_sym))
        if self.cancel_raises:
            raise self.cancel_raises

    def fetch_order(self, order_id, ccxt_sym):
        self.calls.append(('fetch_order', order_id, ccxt_sym))
        if isinstance(self.fetched_order, Exception):
            raise self.fetched_order
        return self.fetched_order

    def create_market_buy_order(self, ccxt_sym, qty):
        self.calls.append(('buy', ccxt_sym, qty))
        return {'average': None, 'price': None, 'filled': qty}

    def create_market_sell_order(self, ccxt_sym, qty):
        self.calls.append(('sell', ccxt_sym, qty))
        return {'average': None, 'price': None}

    def fetch_balance(self):
        return {'free': dict(self.free_balance)}


# ══════════════════════════════════════════════════════════════
#  공용 fixture
# ══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(sm, '_tg', lambda msg: sent.append(msg))
    return sent


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / 'test_spot.db'
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', db_file)
    sm.init_spot_db()
    yield db_file


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'NOT_PRESENT')


@pytest.fixture
def _state(monkeypatch):
    fresh = {
        'equity': 10_000.0,
        'usdt_balance': 10_000.0,
        'peak_equity': 0.0,
        'ratchet_alert_tier': 0,
        'paused': False,
        'dry_run': False,          # 스탑 주문은 라이브 모드에서만
        'day_pnl': 0.0,
        'day_start_eq': 10_000.0,
        'daily_loss_alerted': False,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


@pytest.fixture
def _fake_ex(monkeypatch):
    ex = _FakeStopExchange()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


def _save_pos(order_id=''):
    sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                       0.02, 'sl_tp', 0, 'TRENDING_UP')
    if order_id:
        sm._update_position_order_id('S3', 'BTCUSDT', order_id)


def _trades(db_file):
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM spot_trades').fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
#  _place_stop_loss_order
# ══════════════════════════════════════════════════════════════

class TestPlaceStopLossOrder:
    def test_places_stop_limit_with_gap(self, _fake_ex):
        oid = sm._place_stop_loss_order('S3', 'BTCUSDT', 'BTC/USDT', 10.0, 95.0)
        assert oid == 'STOP-1'
        call = _fake_ex.calls[0]
        assert call[0] == 'create_order'
        assert call[2] == 'limit' and call[3] == 'sell'
        assert call[4] == 10.0
        assert call[5] == pytest.approx(95.0 * (1 - sm.SPOT_STOP_LIMIT_GAP))
        assert call[6]['stopPrice'] == 95.0

    def test_skips_below_notional(self, _fake_ex):
        oid = sm._place_stop_loss_order('S3', 'XXXUSDT', 'XXX/USDT', 1.0, 3.0)
        assert oid == ''
        assert _fake_ex.calls == []

    def test_exchange_error_returns_empty_and_alerts(self, _fake_ex, _no_telegram):
        _fake_ex.create_order_result = RuntimeError('binance rejected')
        oid = sm._place_stop_loss_order('S3', 'BTCUSDT', 'BTC/USDT', 10.0, 95.0)
        assert oid == ''
        assert any('스탑주문 실패' in m for m in _no_telegram)

    def test_disabled_by_config_flag(self, _fake_ex, monkeypatch):
        monkeypatch.setattr(sm, 'SPOT_EXCHANGE_STOP', False)
        oid = sm._place_stop_loss_order('S3', 'BTCUSDT', 'BTC/USDT', 10.0, 95.0)
        assert oid == ''
        assert _fake_ex.calls == []


# ══════════════════════════════════════════════════════════════
#  _cancel_stop_order
# ══════════════════════════════════════════════════════════════

class TestCancelStopOrder:
    def test_cancels_by_id(self, _fake_ex):
        sm._cancel_stop_order('S3', 'BTCUSDT', 'BTC/USDT', 'STOP-1')
        assert _fake_ex.calls == [('cancel', 'STOP-1', 'BTC/USDT')]

    def test_empty_id_is_noop(self, _fake_ex):
        sm._cancel_stop_order('S3', 'BTCUSDT', 'BTC/USDT', '')
        assert _fake_ex.calls == []

    def test_cancel_error_swallowed(self, _fake_ex):
        _fake_ex.cancel_raises = RuntimeError('order not found')
        sm._cancel_stop_order('S3', 'BTCUSDT', 'BTC/USDT', 'STOP-1')  # 예외 전파 없어야 함


# ══════════════════════════════════════════════════════════════
#  _spot_buy — 매수 후 스탑 주문 등록
# ══════════════════════════════════════════════════════════════

class TestSpotBuyPlacesStop:
    def _sig(self):
        return {'sl': 95.0, 'tp': 110.0, 'rr': 2.0, 'exit_type': 'sl_tp', 'max_hold': 0}

    def test_live_buy_registers_stop_and_stores_id(self, _state, _fake_ex, _temp_db):
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', self._sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['sl_order_id'] == 'STOP-1'
        kinds = [c[0] for c in _fake_ex.calls]
        assert kinds.index('buy') < kinds.index('create_order')

    def test_stop_failure_keeps_position_with_empty_id(self, _state, _fake_ex, _temp_db):
        _fake_ex.create_order_result = RuntimeError('rejected')
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', self._sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['sl_order_id'] == ''

    def test_dry_run_places_no_stop(self, _state, _fake_ex, _temp_db):
        _state['dry_run'] = True
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', self._sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        assert all(c[0] != 'create_order' for c in _fake_ex.calls)
        assert sm._load_position('S3', 'BTCUSDT')['sl_order_id'] == ''


# ══════════════════════════════════════════════════════════════
#  _spot_sell — 매도 전 스탑 주문 선취소
# ══════════════════════════════════════════════════════════════

class TestSpotSellCancelsStopFirst:
    def test_cancel_happens_before_market_sell(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-1')
        _fake_ex.free_balance = {'BTC': 10.0}
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'TP')
        kinds = [c[0] for c in _fake_ex.calls]
        assert 'cancel' in kinds and 'sell' in kinds
        assert kinds.index('cancel') < kinds.index('sell')
        assert sm._load_position('S3', 'BTCUSDT') is None

    def test_no_stop_id_skips_cancel(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='')
        _fake_ex.free_balance = {'BTC': 10.0}
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'TP')
        assert all(c[0] != 'cancel' for c in _fake_ex.calls)


# ══════════════════════════════════════════════════════════════
#  _handle_stop_order_state
# ══════════════════════════════════════════════════════════════

class TestHandleStopOrderState:
    def test_filled_stop_logs_sl_trade_and_deletes(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-1')
        _fake_ex.fetched_order = {'status': 'closed', 'filled': 10.0, 'average': 94.8}
        pos = sm._load_position('S3', 'BTCUSDT')
        handled = sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos)
        assert handled is True
        assert sm._load_position('S3', 'BTCUSDT') is None
        rows = _trades(_temp_db)
        assert len(rows) == 1
        assert rows[0]['reason'] == 'SL'
        assert rows[0]['exit_price'] == pytest.approx(94.8)
        # day_pnl은 왕복 수수료 차감 net
        _fee = (100.0 + 94.8) * 10.0 * 0.001
        assert _state['day_pnl'] == pytest.approx((94.8 - 100.0) * 10.0 - _fee)

    def test_canceled_stop_rearms_new_order(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-OLD')
        _fake_ex.fetched_order = {'status': 'canceled'}
        _fake_ex.create_order_result = {'id': 'STOP-NEW'}
        pos = sm._load_position('S3', 'BTCUSDT')
        handled = sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos)
        assert handled is False
        assert sm._load_position('S3', 'BTCUSDT')['sl_order_id'] == 'STOP-NEW'

    def test_canceled_stop_rearm_failure_falls_back_to_software(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-OLD')
        _fake_ex.fetched_order = {'status': 'canceled'}
        _fake_ex.create_order_result = RuntimeError('rejected')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos)
        assert sm._load_position('S3', 'BTCUSDT')['sl_order_id'] == ''

    def test_open_stop_returns_false_keeps_position(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-1')
        _fake_ex.fetched_order = {'status': 'open'}
        pos = sm._load_position('S3', 'BTCUSDT')
        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is False
        assert sm._load_position('S3', 'BTCUSDT') is not None

    def test_fetch_failure_returns_false(self, _state, _fake_ex, _temp_db):
        _save_pos(order_id='STOP-1')
        _fake_ex.fetched_order = RuntimeError('network')
        pos = sm._load_position('S3', 'BTCUSDT')
        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is False
        assert sm._load_position('S3', 'BTCUSDT') is not None


# ══════════════════════════════════════════════════════════════
#  _manage_position 통합 — 스탑 체결 감지
# ══════════════════════════════════════════════════════════════

class TestManagePositionStopIntegration:
    def test_filled_stop_detected_in_manage(self, _state, _fake_ex, _temp_db, monkeypatch):
        _save_pos(order_id='STOP-1')
        _fake_ex.fetched_order = {'status': 'closed', 'filled': 10.0, 'average': 94.8}
        monkeypatch.setattr(sm, '_get_price', lambda s: 100.0)
        sm._manage_position('S3', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert sm._load_position('S3', 'BTCUSDT') is None
        assert _trades(_temp_db)[0]['reason'] == 'SL'

    def test_dry_run_skips_stop_check(self, _state, _fake_ex, _temp_db, monkeypatch):
        _state['dry_run'] = True
        _save_pos(order_id='STOP-1')
        monkeypatch.setattr(sm, '_get_price', lambda s: 100.0)
        sm._manage_position('S3', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert all(c[0] != 'fetch_order' for c in _fake_ex.calls)
        assert sm._load_position('S3', 'BTCUSDT') is not None


# ══════════════════════════════════════════════════════════════
#  reconcile — 잔고 0 삭제 시 고아 스탑 주문 취소
# ══════════════════════════════════════════════════════════════

class _OneShotEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self._waits = 0

    def wait(self, timeout=None):
        self._waits += 1
        if self._waits >= 2:
            self.set()
        return self.is_set()


class TestReconcileCancelsOrphanStop:
    def test_zero_balance_cancels_stop_order(self, _state, _temp_db, monkeypatch):
        _save_pos(order_id='STOP-1')

        class _BalEx(_FakeStopExchange):
            def fetch_balance(self, params=None):
                return {'BTC': {'total': 0}}

        bal_ex = _BalEx()
        # reconcile은 _get_ex()로 잔고 조회, _cancel_stop_order도 _get_ex() 사용
        monkeypatch.setattr(sm, '_get_ex', lambda: bal_ex)
        ev = _OneShotEvent()
        sm._position_reconcile_loop(ev)
        assert ('cancel', 'STOP-1', 'BTC/USDT') in bal_ex.calls
        assert sm._load_position('S3', 'BTCUSDT') is None


# ══════════════════════════════════════════════════════════════
#  main() 킬스위치 기동 가드
# ══════════════════════════════════════════════════════════════

class TestKillSwitchStartupGuard:
    def test_existing_kill_switch_aborts_startup(self, tmp_path, monkeypatch):
        ks = tmp_path / 'KILL'
        ks.touch()
        monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', ks)
        monkeypatch.setattr(sys, 'argv', ['atlas_spot_main.py'])
        called = []
        monkeypatch.setattr(sm, 'init_spot_db', lambda: called.append('init'))
        sm.main()   # 즉시 반환해야 함 (스레드/DB 초기화 없이)
        assert called == []

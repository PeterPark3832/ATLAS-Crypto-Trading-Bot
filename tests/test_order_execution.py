"""
ATLAS — 주문 실행(매수/매도) 단위 테스트
================================
atlas_spot_main.py의 _spot_buy / _spot_sell을 검증합니다.
실거래소 호출은 가짜 ccxt 인스턴스로 대체합니다.

실행:
  pytest tests/test_order_execution.py -v
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  헬퍼: 가짜 ccxt 거래소
# ══════════════════════════════════════════════════════════════

class _FakeExchange:
    def __init__(self):
        self.buy_calls = []
        self.sell_calls = []
        self.buy_order = None       # dict 또는 Exception
        self.sell_order = None      # dict 또는 Exception
        self.free_balance = {}      # {asset: qty}
        self.fetch_balance_fail_times = 0  # 처음 N회는 fetch_balance가 예외 발생

    def create_market_buy_order(self, ccxt_sym, qty):
        self.buy_calls.append((ccxt_sym, qty))
        if isinstance(self.buy_order, Exception):
            raise self.buy_order
        return self.buy_order or {'average': None, 'price': None, 'filled': qty}

    def create_market_sell_order(self, ccxt_sym, qty):
        self.sell_calls.append((ccxt_sym, qty))
        if isinstance(self.sell_order, Exception):
            raise self.sell_order
        return self.sell_order or {'average': None, 'price': None}

    def fetch_balance(self):
        if self.fetch_balance_fail_times > 0:
            self.fetch_balance_fail_times -= 1
            raise RuntimeError('temporary network error')
        return {'free': dict(self.free_balance)}


def _base_sig(entry: float = 100.0, sl_pct: float = 0.05, rr: float = 2.0) -> dict:
    sl = entry * (1 - sl_pct)
    tp = entry * (1 + sl_pct * rr)
    return {'sl': sl, 'tp': tp, 'rr': rr, 'exit_type': 'sl_tp', 'max_hold': 10}


def _make_position(entry_price=100.0, qty=10.0, sl=95.0, entry_ts=None) -> dict:
    return {
        'entry_price': entry_price,
        'qty_tokens': qty,
        'cost_usdt': entry_price * qty,
        'risk_pct': 0.02,
        'sl': sl,
        'entry_ts': (entry_ts or datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        'regime': 'TRENDING_UP',
    }


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
        'peak_equity': 0.0,         # 0 → 랫칫 비활성(1.0)
        'ratchet_alert_tier': 0,
        'paused': False,
        'dry_run': True,
        'day_pnl': 0.0,
        'day_start_eq': 10_000.0,
        'daily_loss_alerted': False,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


@pytest.fixture
def _fake_ex(monkeypatch):
    ex = _FakeExchange()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


def _trades(db_file) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM spot_trades').fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
#  _spot_buy — 진입 차단 조건
# ══════════════════════════════════════════════════════════════

class TestSpotBuyBlocking:
    def test_blocked_when_paused(self, _state):
        _state['paused'] = True
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False
        assert sm._load_position('S3', 'BTCUSDT') is None

    def test_blocked_when_kill_switch_present(self, _state, tmp_path, monkeypatch):
        kill = tmp_path / 'STOP'
        kill.touch()
        monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', kill)
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_when_max_positions_reached(self, _state):
        for i in range(sm.SPOT_MAX_POSITIONS):
            sm._save_position('S3', f'SYM{i}USDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                               0.02, 'sl_tp', 10, 'TRENDING_UP')
        ok = sm._spot_buy('S3', 'NEWUSDT', 'NEW/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_on_duplicate_position(self, _state):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_when_sl_equals_entry(self, _state):
        sig = _base_sig()
        sig['sl'] = 100.0  # entry와 동일 → sl_dist=0
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_when_sl_distance_exceeds_cap(self, _state):
        sig = _base_sig(sl_pct=0.30)  # SPOT_MAX_SL_PCT(0.20) 초과
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_when_insufficient_buying_power(self, _state):
        _state['usdt_balance'] = 0.0
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_blocked_on_daily_loss_limit(self, _state, _no_telegram):
        _state['day_pnl'] = -500.0  # -5% < SPOT_DAILY_LOSS_LIMIT(-4%)
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False
        assert _state['daily_loss_alerted'] is True
        assert len(_no_telegram) == 1

    def test_daily_loss_alert_not_duplicated(self, _state, _no_telegram):
        _state['day_pnl'] = -500.0
        sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        sm._spot_buy('S3', 'ETHUSDT', 'ETH/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert len(_no_telegram) == 1


# ══════════════════════════════════════════════════════════════
#  _spot_buy — 사이징 / 체결
# ══════════════════════════════════════════════════════════════

class TestSpotBuySizing:
    def test_alloc_cap_limits_cost(self, _state):
        """SL이 좁으면 리스크 기준 수량이 커지므로 배분 한도(15%)로 캡 됨.
        (SL 1%는 이제 비용 가드가 먼저 막으므로 3%로 검증 — 아래 별도 테스트 참조)"""
        sig = _base_sig(sl_pct=0.03)
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['cost_usdt'] <= _state['equity'] * sm.SPOT_MAX_ALLOC_PCT + 1e-6

    def test_dry_run_does_not_call_exchange(self, _state, _fake_ex):
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        assert _fake_ex.buy_calls == []
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['entry_price'] == 100.0

    def test_live_buy_uses_fill_price_and_qty(self, _state, _fake_ex):
        _state['dry_run'] = False
        _fake_ex.buy_order = {'average': 101.0, 'filled': 5.0}
        sig = _base_sig()
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')
        assert ok is True
        assert len(_fake_ex.buy_calls) == 1
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['entry_price'] == 101.0
        assert pos['qty_tokens'] == 5.0

    def test_live_buy_recomputes_sl_from_fill_price(self, _state, _fake_ex):
        """체결가가 신호 가격과 다르면 SL은 동일한 거리만큼 체결가 기준으로 재계산."""
        _state['dry_run'] = False
        _fake_ex.buy_order = {'average': 110.0, 'filled': 1.0}
        sig = _base_sig(entry=100.0, sl_pct=0.05)  # sl = 95.0, sl_dist = 5.0
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['sl'] == pytest.approx(110.0 - 5.0)

    def test_live_buy_order_exception_returns_false(self, _state, _fake_ex, _no_telegram):
        _state['dry_run'] = False
        _fake_ex.buy_order = RuntimeError('exchange unavailable')
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False
        assert sm._load_position('S3', 'BTCUSDT') is None
        assert any('실패' in m for m in _no_telegram)


# ══════════════════════════════════════════════════════════════
#  _spot_sell — dry-run
# ══════════════════════════════════════════════════════════════

class TestSpotSellDryRun:
    def test_dry_run_logs_trade_and_updates_day_pnl(self, _state):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        assert sm._load_position('S3', 'BTCUSDT') is None
        # day_pnl은 왕복 수수료 차감 net: (110-100)*10 - (100+110)*10*0.001
        assert _state['day_pnl'] == pytest.approx(100.0 - 2.1)

    def test_dry_run_records_correct_pnl_r(self, _state, _temp_db):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        trades = _trades(_temp_db)
        assert len(trades) == 1
        assert trades[0]['pnl_r'] == pytest.approx(2.0)  # 10 gain / 5 sl_dist


# ══════════════════════════════════════════════════════════════
#  _spot_sell — 실거래
# ══════════════════════════════════════════════════════════════

class TestSpotSellLive:
    def test_normal_sell_deletes_position_and_logs_trade(self, _state, _fake_ex, _temp_db):
        _state['dry_run'] = False
        _fake_ex.free_balance = {'BTC': 10.0}
        _fake_ex.sell_order = {'average': 108.0}
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        assert sm._load_position('S3', 'BTCUSDT') is None
        assert len(_fake_ex.sell_calls) == 1
        trades = _trades(_temp_db)
        assert trades[0]['exit_price'] == 108.0

    def test_zero_balance_triggers_manual_sold_cleanup(self, _state, _fake_ex, _temp_db):
        """실잔고가 0이면 매도 주문 시도 없이 DB만 정리 (수동매도로 간주)."""
        _state['dry_run'] = False
        _fake_ex.free_balance = {'BTC': 0.0}
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        assert sm._load_position('S3', 'BTCUSDT') is None
        assert _fake_ex.sell_calls == []
        trades = _trades(_temp_db)
        assert trades[0]['reason'] == 'MANUAL_SOLD'

    def test_partial_balance_adjusts_qty_before_selling(self, _state, _fake_ex):
        """DB 수량보다 실잔고가 적으면(수수료 등) 실잔고로 줄여서 매도."""
        _state['dry_run'] = False
        _fake_ex.free_balance = {'BTC': 8.0}  # DB qty=10
        _fake_ex.sell_order = {'average': 108.0}
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        assert _fake_ex.sell_calls == [('BTC/USDT', 8.0)]

    def test_insufficient_balance_error_with_near_zero_actual_triggers_manual_sold(self, _state, _fake_ex, _temp_db):
        """사전 잔고확인이 실패(네트워크 오류)해 DB수량(10) 그대로 매도를 시도했다가
        거부된 뒤, 예외 핸들러의 잔고조회는 성공해 실잔고가 DB수량의 5% 미만임을
        확인하고 수동매도로 자동 정리하는 경로."""
        _state['dry_run'] = False
        _fake_ex.fetch_balance_fail_times = 1  # 사전 잔고확인만 실패
        _fake_ex.sell_order = Exception('Insufficient balance')
        _fake_ex.free_balance = {'BTC': 0.001}  # DB qty=10 → 0.001 < 10*0.05
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')
        assert sm._load_position('S3', 'BTCUSDT') is None
        trades = _trades(_temp_db)
        assert trades[0]['reason'] == 'MANUAL_SOLD'

    def test_notional_error_keeps_position_in_db(self, _state, _fake_ex):
        """NOTIONAL 미달 오류 시 포지션을 DB에서 삭제하지 않고 유지 (가격 회복 대기)."""
        _state['dry_run'] = False
        _fake_ex.free_balance = {'BTC': 10.0}
        _fake_ex.sell_order = Exception('Filter failure: NOTIONAL')
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 0.40, 'SL')
        assert sm._load_position('S3', 'BTCUSDT') is not None

    def test_generic_sell_error_keeps_position_and_does_not_log_trade(self, _state, _fake_ex, _temp_db):
        _state['dry_run'] = False
        _fake_ex.free_balance = {'BTC': 10.0}
        _fake_ex.sell_order = Exception('some other exchange error')
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 110.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 10, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 90.0, 'SL')
        assert sm._load_position('S3', 'BTCUSDT') is not None
        assert _trades(_temp_db) == []


# ══════════════════════════════════════════════════════════════
#  진입 직렬화 (_entry_lock — 4H/1D 루프 동시 진입 TOCTOU 방지)
# ══════════════════════════════════════════════════════════════

class TestEntrySerialization:
    def test_spot_buy_holds_entry_lock(self, _state, monkeypatch):
        """포지션 수 한도 체크→저장 구간이 _entry_lock 안에서 실행되는지 검증."""
        observed = []

        def _fake_locked(*a, **k):
            observed.append(sm._entry_lock.locked())
            return True

        monkeypatch.setattr(sm, '_spot_buy_locked', _fake_locked)
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0,
                             'TRENDING_UP') is True
        assert observed == [True]
        assert not sm._entry_lock.locked()   # 종료 후 해제

"""
ATLAS — DB CRUD 및 백그라운드 루프(atlas_spot_main.py) 단위 테스트
================================
_cfg_get/_cfg_set, _load_all_positions, _update_position_tp, _delete_position,
_log_trade, _get_spot_equity, _get_spot_funding,
_balance_poller, _daily_reset_loop, _db_backup_loop, _tg_cmd_loop을 검증합니다.

실행:
  pytest tests/test_db_and_loops.py -v
"""

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  공용 fixture
# ══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, '_tg', lambda msg: calls.append(msg))
    return calls


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / 'test_spot.db'
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', db_file)
    sm.init_spot_db()
    yield db_file


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'NOT_PRESENT')


class _OneShotEvent(threading.Event):
    """wait()가 호출되면 즉시 반환하고, 두 번째 루프 진입을 막기 위해
    다음 wait() 호출 시 set() 상태로 만든다 (1회만 본문 실행)."""
    def __init__(self):
        super().__init__()
        self._waits = 0

    def wait(self, timeout=None):
        self._waits += 1
        if self._waits >= 2:
            self.set()
        return self.is_set()


class _SetOnWaitEvent(threading.Event):
    """body(); wait() 순서의 루프에서 본문을 정확히 1회만 실행시키기 위해,
    wait() 호출 시점에 즉시 set()한다."""
    def wait(self, timeout=None):
        self.set()
        return True


# ══════════════════════════════════════════════════════════════
#  _cfg_get / _cfg_set
# ══════════════════════════════════════════════════════════════

class TestCfgGetSet:
    def test_missing_key_returns_default(self):
        assert sm._cfg_get('nope', 'fallback') == 'fallback'

    def test_set_then_get_roundtrip(self):
        sm._cfg_set('mode', 'live')
        assert sm._cfg_get('mode') == 'live'

    def test_set_overwrites_existing_value(self):
        sm._cfg_set('mode', 'live')
        sm._cfg_set('mode', 'dry')
        assert sm._cfg_get('mode') == 'dry'


# ══════════════════════════════════════════════════════════════
#  _load_all_positions / _update_position_tp / _delete_position
# ══════════════════════════════════════════════════════════════

class TestPositionCrud:
    def test_load_all_positions_empty(self):
        assert sm._load_all_positions() == []

    def test_load_all_positions_returns_multiple(self):
        sm._save_position('S3', 'BTCUSDT', 100.0, 90.0, 120.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        sm._save_position('S6', 'ETHUSDT', 50.0, 45.0, 60.0, 2.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        rows = sm._load_all_positions()
        assert len(rows) == 2
        assert {r['symbol'] for r in rows} == {'BTCUSDT', 'ETHUSDT'}

    def test_update_position_tp_changes_value(self):
        sm._save_position('S3', 'BTCUSDT', 100.0, 90.0, 120.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        sm._update_position_tp('S3', 'BTCUSDT', 150.0)
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['tp'] == pytest.approx(150.0)

    def test_delete_position_removes_row(self):
        sm._save_position('S3', 'BTCUSDT', 100.0, 90.0, 120.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        sm._delete_position('S3', 'BTCUSDT')
        assert sm._load_position('S3', 'BTCUSDT') is None

    def test_delete_position_nonexistent_is_noop(self):
        sm._delete_position('S3', 'BTCUSDT')  # 예외 없이 통과해야 함
        assert sm._load_position('S3', 'BTCUSDT') is None


# ══════════════════════════════════════════════════════════════
#  _log_trade
# ══════════════════════════════════════════════════════════════

class TestLogTrade:
    def test_inserts_row_with_rounded_values_and_roundtrip_fee(self):
        net = sm._log_trade('S3', 'BTCUSDT', 100.0, 110.0, 1.0, 100.0,
                             10.0001234, 10.0001234, 1.23456, 5.555, 'TP',
                             'TRENDING_UP', '2099-01-01T00:00:00+00:00')
        with sm._db_lock, sm._db_conn() as conn:
            row = dict(conn.execute('SELECT * FROM spot_trades').fetchone())
        assert row['pnl_usdt'] == pytest.approx(10.0001)
        assert row['pnl_r'] == pytest.approx(1.2346)
        assert row['hold_hours'] == pytest.approx(5.55)
        # 왕복 수수료 = (entry + exit) * qty * BT_SPOT_FEE = (100+110) * 1.0 * 0.001
        assert row['fee_usdt'] == pytest.approx(0.21)
        assert row['reason'] == 'TP'
        assert row['regime'] == 'TRENDING_UP'
        # 반환값은 수수료 차감 net PnL (day_pnl 등 리스크 로직용)
        assert net == pytest.approx(10.0001234 - 0.21)


# ══════════════════════════════════════════════════════════════
#  _get_spot_equity
# ══════════════════════════════════════════════════════════════

class _FakeEquityExchange:
    def __init__(self, balance, tickers):
        self._balance = balance
        self._tickers = tickers

    def fetch_balance(self, params=None):
        return self._balance

    def fetch_ticker(self, sym):
        return self._tickers[sym]


class TestGetSpotEquity:
    def test_sums_usdt_and_coin_values(self, monkeypatch):
        ex = _FakeEquityExchange(
            balance={'USDT': {'total': 500.0}, 'BTC': {'total': 0.1}},
            tickers={'BTC/USDT': {'last': 50000.0}},
        )
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        total, usdt = sm._get_spot_equity()
        assert usdt == pytest.approx(500.0)
        assert total == pytest.approx(500.0 + 0.1 * 50000.0)

    def test_ignores_dust_below_threshold(self, monkeypatch):
        ex = _FakeEquityExchange(
            balance={'USDT': {'total': 500.0}, 'DUST': {'total': 1e-9}},
            tickers={},
        )
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        total, usdt = sm._get_spot_equity()
        assert total == pytest.approx(500.0)

    def test_ticker_failure_for_one_coin_does_not_abort(self, monkeypatch):
        class _Ex(_FakeEquityExchange):
            def fetch_ticker(self, sym):
                raise RuntimeError('no market')

        ex = _Ex(balance={'USDT': {'total': 500.0}, 'BTC': {'total': 0.1}}, tickers={})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        total, usdt = sm._get_spot_equity()
        assert total == pytest.approx(500.0)
        assert usdt == pytest.approx(500.0)

    def test_exchange_failure_falls_back_to_state(self, monkeypatch):
        class _RaisingExchange:
            def fetch_balance(self, params=None):
                raise RuntimeError('network down')

        monkeypatch.setattr(sm, '_get_ex', lambda: _RaisingExchange())
        monkeypatch.setattr(sm, '_state', {'equity': 777.0, 'usdt_balance': 333.0})
        total, usdt = sm._get_spot_equity()
        assert total == 777.0
        assert usdt == 333.0


# ══════════════════════════════════════════════════════════════
#  _get_spot_funding
# ══════════════════════════════════════════════════════════════

class TestGetSpotFunding:
    def test_returns_funding_rate(self, monkeypatch):
        class _FakeFuturesExchange:
            def fetch_funding_rate(self, ccxt_sym):
                assert ccxt_sym == 'BTC/USDT:USDT'
                return {'fundingRate': 0.0005}

        monkeypatch.setattr(sm, '_get_ex_futures', lambda: _FakeFuturesExchange())
        assert sm._get_spot_funding('BTCUSDT') == pytest.approx(0.0005)

    def test_exception_returns_zero(self, monkeypatch):
        class _RaisingExchange:
            def fetch_funding_rate(self, ccxt_sym):
                raise RuntimeError('no perp market')

        monkeypatch.setattr(sm, '_get_ex_futures', lambda: _RaisingExchange())
        assert sm._get_spot_funding('BTCUSDT') == 0.0


# ══════════════════════════════════════════════════════════════
#  _balance_poller (1회 실행)
# ══════════════════════════════════════════════════════════════

class TestBalancePoller:
    def test_updates_state_and_peak_equity(self, monkeypatch):
        state = {'equity': 0.0, 'usdt_balance': 0.0, 'peak_equity': 100.0}
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())
        monkeypatch.setattr(sm, '_get_spot_equity', lambda: (250.0, 200.0))
        ev = _SetOnWaitEvent()
        sm._balance_poller(ev)
        assert state['equity'] == 250.0
        assert state['usdt_balance'] == 200.0
        assert state['peak_equity'] == 250.0

    def test_peak_equity_not_lowered(self, monkeypatch):
        state = {'equity': 0.0, 'usdt_balance': 0.0, 'peak_equity': 500.0}
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())
        monkeypatch.setattr(sm, '_get_spot_equity', lambda: (250.0, 200.0))
        ev = _SetOnWaitEvent()
        sm._balance_poller(ev)
        assert state['peak_equity'] == 500.0

    def test_exception_is_swallowed(self, monkeypatch):
        state = {'equity': 0.0, 'usdt_balance': 0.0, 'peak_equity': 0.0}
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())

        def _raise():
            raise RuntimeError('boom')

        monkeypatch.setattr(sm, '_get_spot_equity', _raise)
        ev = _SetOnWaitEvent()
        sm._balance_poller(ev)  # 예외가 전파되지 않아야 함


# ══════════════════════════════════════════════════════════════
#  _daily_reset_loop
# ══════════════════════════════════════════════════════════════

class _FixedDatetime(datetime):
    """sm.datetime.now()가 항상 고정된 시각을 반환하도록 패치하기 위한 헬퍼."""
    _fixed = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed


def _make_fixed_datetime(dt):
    cls = type('_FixedDatetime', (_FixedDatetime,), {'_fixed': dt})
    return cls


class TestDailyResetLoop:
    def test_at_midnight_resets_day_stats_and_sends_briefing(self, monkeypatch, _no_telegram):
        midnight = datetime(2099, 1, 2, 0, 0, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(sm, 'datetime', _make_fixed_datetime(midnight))
        state = {'day_pnl': 42.0, 'equity': 1234.0, 'day_start_eq': 0.0,
                  'daily_loss_alerted': True}
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())
        sm._log_trade('S3', 'BTCUSDT', 100.0, 110.0, 1.0, 100.0, 10.0, 10.0,
                       1.0, 1.0, 'TP', 'TRENDING_UP', '2099-01-01T00:00:00+00:00')
        with sm._db_lock, sm._db_conn() as conn:
            conn.execute("UPDATE spot_trades SET exit_ts='2099-01-01T12:00:00+00:00'")

        ev = _OneShotEvent()
        sm._daily_reset_loop(ev)

        assert state['day_pnl'] == 0.0
        assert state['day_start_eq'] == 1234.0
        assert state['daily_loss_alerted'] is False
        assert any('일간브리핑' in m for m in _no_telegram)
        assert any('거래: 1건' in m for m in _no_telegram)

    def test_not_midnight_skips_reset(self, monkeypatch, _no_telegram):
        noon = datetime(2099, 1, 2, 12, 0, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(sm, 'datetime', _make_fixed_datetime(noon))
        state = {'day_pnl': 42.0, 'equity': 1234.0, 'day_start_eq': 0.0,
                  'daily_loss_alerted': True}
        monkeypatch.setattr(sm, '_state', state)
        monkeypatch.setattr(sm, '_state_lock', threading.Lock())

        ev = _OneShotEvent()
        sm._daily_reset_loop(ev)

        assert state['day_pnl'] == 42.0
        assert _no_telegram == []


# ══════════════════════════════════════════════════════════════
#  _db_backup_loop
# ══════════════════════════════════════════════════════════════

class TestDbBackupLoop:
    def test_creates_backup_copy(self, monkeypatch, _temp_db):
        monkeypatch.setattr(sm, '_db_lock', threading.Lock())
        ev = _SetOnWaitEvent()
        sm._db_backup_loop(ev)
        backups = list((_temp_db.parent / 'backups').glob('atlas_spot_*.db'))
        assert len(backups) == 1

    def test_prunes_old_backups_beyond_28(self, monkeypatch, _temp_db, _no_telegram):
        backup_dir = _temp_db.parent / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            (backup_dir / f'atlas_spot_old{i:02d}.db').write_bytes(b'x')
        monkeypatch.setattr(sm, '_db_lock', threading.Lock())
        ev = _SetOnWaitEvent()
        sm._db_backup_loop(ev)
        remaining = sorted(backup_dir.glob('atlas_spot_*.db'))
        assert len(remaining) == 28

    def test_missing_db_file_skips_silently(self, monkeypatch, tmp_path, _no_telegram):
        monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'nonexistent.db')
        monkeypatch.setattr(sm, '_db_lock', threading.Lock())
        ev = _SetOnWaitEvent()
        sm._db_backup_loop(ev)
        assert list((tmp_path / 'backups').glob('atlas_spot_*.db')) == []
        assert _no_telegram == []


# ══════════════════════════════════════════════════════════════
#  _tg_cmd_loop
# ══════════════════════════════════════════════════════════════

class TestTgCmdLoop:
    def test_no_token_returns_immediately(self, monkeypatch):
        monkeypatch.setattr(sm, 'TG_TOKEN', '')
        monkeypatch.setattr(sm, 'TG_CHAT_ID', '')
        ev = _SetOnWaitEvent()
        sm._tg_cmd_loop(ev)  # 즉시 반환해야 함 (wait가 호출되지 않음)
        assert not ev.is_set()

    def test_dispatches_matching_chat_id(self, monkeypatch):
        monkeypatch.setattr(sm, 'TG_TOKEN', 'tok')
        monkeypatch.setattr(sm, 'TG_CHAT_ID', '111')
        handled = []
        monkeypatch.setattr(sm, '_handle_tg_cmd', lambda cmd: handled.append(cmd))

        class _Resp:
            def json(self):
                return {'result': [
                    {'update_id': 1, 'message': {'text': '/status',
                                                   'chat': {'id': 111}}},
                ]}

        monkeypatch.setattr(sm.requests, 'get', lambda url, params, timeout: _Resp())
        ev = _SetOnWaitEvent()
        sm._tg_cmd_loop(ev)
        assert handled == ['/status']

    def test_ignores_other_chat_id(self, monkeypatch):
        monkeypatch.setattr(sm, 'TG_TOKEN', 'tok')
        monkeypatch.setattr(sm, 'TG_CHAT_ID', '111')
        handled = []
        monkeypatch.setattr(sm, '_handle_tg_cmd', lambda cmd: handled.append(cmd))

        class _Resp:
            def json(self):
                return {'result': [
                    {'update_id': 1, 'message': {'text': '/status',
                                                   'chat': {'id': 999}}},
                ]}

        monkeypatch.setattr(sm.requests, 'get', lambda url, params, timeout: _Resp())
        ev = _SetOnWaitEvent()
        sm._tg_cmd_loop(ev)
        assert handled == []

    def test_request_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(sm, 'TG_TOKEN', 'tok')
        monkeypatch.setattr(sm, 'TG_CHAT_ID', '111')

        def _raise(url, params, timeout):
            raise RuntimeError('timeout')

        monkeypatch.setattr(sm.requests, 'get', _raise)
        ev = _SetOnWaitEvent()
        sm._tg_cmd_loop(ev)  # 예외가 전파되지 않아야 함

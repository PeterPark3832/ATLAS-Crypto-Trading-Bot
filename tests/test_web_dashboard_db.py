"""
ATLAS — 웹 대시보드 DB 조회 및 인증/잔고 함수 단위 테스트
================================
_auth, _q, _trades, _positions, _spot_balance를 검증합니다.

실행:
  pytest tests/test_web_dashboard_db.py -v
"""

import json
import time


import sqlite3

import pytest
from fastapi import HTTPException

import atlas_web_dashboard as wd


@pytest.fixture(autouse=True)
def _clear_balance_cache(monkeypatch):
    monkeypatch.setattr(wd, '_bal_cache', {'val': None, 'ts': 0.0})


def _make_db(tmp_path):
    db_file = tmp_path / 'state' / 'atlas_spot.db'
    db_file.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_file))
    con.execute("""
        CREATE TABLE spot_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, entry_price REAL, exit_price REAL,
            qty_tokens REAL, cost_usdt REAL, pnl_usdt REAL, pnl_pct REAL,
            pnl_r REAL, hold_hours REAL, reason TEXT, entry_ts TEXT,
            exit_ts TEXT, regime TEXT, fee_usdt REAL DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE spot_positions (
            strategy TEXT, symbol TEXT, entry_price REAL, sl REAL, tp REAL,
            qty_tokens REAL, cost_usdt REAL, risk_pct REAL, exit_type TEXT,
            max_hold_bars INTEGER, bars_held INTEGER, peak_price REAL,
            entry_ts TEXT, regime TEXT
        )
    """)
    con.execute("""INSERT INTO spot_trades
        (strategy, symbol, entry_price, exit_price, qty_tokens, cost_usdt,
         pnl_usdt, pnl_pct, pnl_r, hold_hours, reason, entry_ts, exit_ts, regime, fee_usdt)
        VALUES ('S6','BTCUSDT',100,110,1.0,100,10.0,10.0,1.0,5.0,'TP',
                '2099-01-01T00:00:00','2099-01-01T05:00:00','TRENDING_UP',0.1)""")
    con.execute("""INSERT INTO spot_positions
        (strategy, symbol, entry_price, sl, tp, qty_tokens, cost_usdt, risk_pct,
         exit_type, max_hold_bars, bars_held, peak_price, entry_ts, regime)
        VALUES ('S3','ETHUSDT',50,45,60,2.0,100,0.02,'sl_tp',0,0,50,
                '2099-01-02T00:00:00','WEAK_TREND')""")
    con.commit()
    con.close()
    return db_file


# ══════════════════════════════════════════════════════════════
#  _auth
# ══════════════════════════════════════════════════════════════

class TestAuth:
    def test_valid_token_passes(self, monkeypatch):
        tok = wd._new_token()
        wd._auth(tok)  # 예외 없이 통과해야 함

    def test_invalid_token_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            wd._auth('garbage-token')
        assert exc.value.status_code == 401

    def test_expired_token_raises_401(self, monkeypatch):
        tok = 'expired-tok'
        monkeypatch.setitem(wd._tokens, tok, time.time() - 1)
        with pytest.raises(HTTPException):
            wd._auth(tok)


# ══════════════════════════════════════════════════════════════
#  _q / _trades / _positions
# ══════════════════════════════════════════════════════════════

class TestDbQueries:
    def test_q_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, 'DB_FILE', tmp_path / 'nope.db')
        assert wd._q('SELECT 1').empty

    def test_q_bad_sql_returns_empty(self, tmp_path, monkeypatch):
        db_file = _make_db(tmp_path)
        monkeypatch.setattr(wd, 'DB_FILE', db_file)
        assert wd._q('SELECT * FROM no_such_table').empty

    def test_trades_real_schema_with_computed_columns(self, tmp_path, monkeypatch):
        db_file = _make_db(tmp_path)
        monkeypatch.setattr(wd, 'DB_FILE', db_file)
        df = wd._trades()
        assert len(df) == 1
        row = df.iloc[0]
        assert row['pnl_usd'] == 10.0
        assert row['direction'] == 'LONG'
        assert row['net_pnl'] == pytest.approx(9.9)  # pnl_usd - fee_usd

    def test_trades_day_filter_excludes_old_rows(self, tmp_path, monkeypatch):
        db_file = _make_db(tmp_path)
        con = sqlite3.connect(str(db_file))
        con.execute("UPDATE spot_trades SET exit_ts='2000-01-01T00:00:00'")
        con.commit()
        con.close()
        monkeypatch.setattr(wd, 'DB_FILE', db_file)
        df = wd._trades(days=7)  # exit_ts가 2000년이므로 최근 7일 필터에서 제외
        assert df.empty

    def test_positions_real_schema(self, tmp_path, monkeypatch):
        db_file = _make_db(tmp_path)
        monkeypatch.setattr(wd, 'DB_FILE', db_file)
        df = wd._positions()
        assert len(df) == 1
        row = df.iloc[0]
        assert row['symbol'] == 'ETHUSDT'
        assert row['direction'] == 'LONG'


# ══════════════════════════════════════════════════════════════
#  _spot_balance
# ══════════════════════════════════════════════════════════════

class _FakeResp:
    def __init__(self, ok=True, status_code=200, json_data=None):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


class TestSpotBalance:
    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.setattr(wd, 'API_KEY', '')
        monkeypatch.setattr(wd, 'API_SECRET', '')
        assert wd._spot_balance() is None

    def test_cached_value_returned_within_ttl(self, monkeypatch):
        monkeypatch.setattr(wd, '_bal_cache', {'val': 1234.5, 'ts': time.time()})
        assert wd._spot_balance() == 1234.5

    def test_fetches_usdt_balance_and_adds_position_value(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wd, 'API_KEY', 'key')
        monkeypatch.setattr(wd, 'API_SECRET', 'secret')
        monkeypatch.setattr(wd, '_positions',
                             lambda: __import__('pandas').DataFrame(
                                 [{'symbol': 'BTCUSDT', 'qty': 0.1, 'risk_usd': 5000.0}]))

        def _fake_get(url, params=None, headers=None, timeout=None):
            if 'account' in url:
                return _FakeResp(json_data={'balances': [{'asset': 'USDT', 'free': '100', 'locked': '0'}]})
            # 배치 ticker 응답 (리스트 형식)
            return _FakeResp(json_data=[{'symbol': 'BTCUSDT', 'price': '50000'}])

        monkeypatch.setattr(wd.requests, 'get', _fake_get)
        result = wd._spot_balance()
        assert result == pytest.approx(100.0 + 0.1 * 50000.0)

    def test_account_http_failure_falls_back_to_stale_cache(self, monkeypatch):
        monkeypatch.setattr(wd, 'API_KEY', 'key')
        monkeypatch.setattr(wd, 'API_SECRET', 'secret')
        monkeypatch.setattr(wd, '_bal_cache', {'val': 999.0, 'ts': 0.0})  # TTL 만료된 stale 캐시

        def _fake_get(url, params=None, headers=None, timeout=None):
            return _FakeResp(ok=False, status_code=500)

        monkeypatch.setattr(wd.requests, 'get', _fake_get)
        assert wd._spot_balance() == 999.0

    def test_ticker_failure_falls_back_to_risk_usd(self, monkeypatch):
        monkeypatch.setattr(wd, 'API_KEY', 'key')
        monkeypatch.setattr(wd, 'API_SECRET', 'secret')
        monkeypatch.setattr(wd, '_positions',
                             lambda: __import__('pandas').DataFrame(
                                 [{'symbol': 'BTCUSDT', 'qty': 0.1, 'risk_usd': 500.0}]))

        def _fake_get(url, params=None, headers=None, timeout=None):
            if 'account' in url:
                return _FakeResp(json_data={'balances': [{'asset': 'USDT', 'free': '100', 'locked': '0'}]})
            return _FakeResp(ok=False, status_code=500)

        monkeypatch.setattr(wd.requests, 'get', _fake_get)
        result = wd._spot_balance()
        assert result == pytest.approx(100.0 + 500.0)


class TestOtherAssetsReconciliation:
    """화면 숫자와 봇 로그의 '총자산'이 서로 설명돼야 한다.

    화면은 USDT+포지션(거래 가능 자산)을, 봇은 보유 코인 전부를 기준으로
    사이징한다. 실측 $197.32(화면) vs $217.02(봇) — 차이는 대부분 수수료용
    BNB였다. 둘 다 내부적으로는 맞지만 같은 이름으로 불려 혼란을 준다.
    차액을 함께 내보내 actual_balance + other_assets = 봇 기준이 되게 한다.
    """

    def _setup(self, monkeypatch, balances, prices):
        monkeypatch.setattr(wd, 'API_KEY', 'key')
        monkeypatch.setattr(wd, 'API_SECRET', 'secret')
        monkeypatch.setattr(wd, '_bal_cache', {'val': None, 'ts': 0.0, 'other': 0.0})
        monkeypatch.setattr(wd, '_positions',
                            lambda: __import__('pandas').DataFrame(
                                [{'symbol': 'ADAUSDT', 'qty': 44.8, 'risk_usd': 8.5}]))

        def _fake_get(url, params=None, headers=None, timeout=None):
            if 'account' in url:
                return _FakeResp(json_data={'balances': balances})
            want = json.loads(params['symbols'])
            return _FakeResp(json_data=[{'symbol': s, 'price': str(prices[s])}
                                        for s in want if s in prices])

        monkeypatch.setattr(wd.requests, 'get', _fake_get)

    def test_counts_holdings_outside_positions(self, monkeypatch):
        self._setup(
            monkeypatch,
            balances=[{'asset': 'USDT', 'free': '132.28', 'locked': '0'},
                      {'asset': 'ADA',  'free': '0.05', 'locked': '44.8'},
                      {'asset': 'BNB',  'free': '0.033', 'locked': '0'}],
            prices={'ADAUSDT': 0.17, 'BNBUSDT': 596.0})
        wd._spot_balance()
        assert wd._bal_cache['other'] == pytest.approx(0.033 * 596.0, rel=1e-3), (
            '포지션 밖 보유분(수수료용 BNB)이 누락되면 두 숫자가 계속 어긋난다')

    def test_position_asset_not_double_counted(self, monkeypatch):
        """포지션 자산은 이미 시가로 더해졌으므로 other에 또 넣으면 안 된다."""
        self._setup(
            monkeypatch,
            balances=[{'asset': 'USDT', 'free': '100', 'locked': '0'},
                      {'asset': 'ADA',  'free': '0', 'locked': '44.8'}],
            prices={'ADAUSDT': 0.17})
        wd._spot_balance()
        assert wd._bal_cache['other'] == 0.0

    def test_unpriceable_asset_is_skipped(self, monkeypatch):
        """현물 마켓이 없는 자산(Simple Earn LD*, 상장폐지)은 건너뛴다."""
        self._setup(
            monkeypatch,
            balances=[{'asset': 'USDT', 'free': '100', 'locked': '0'},
                      {'asset': 'LDUSDT', 'free': '50', 'locked': '0'}],
            prices={})
        wd._spot_balance()
        assert wd._bal_cache['other'] == 0.0

"""
ATLAS — 가격조회/캔들캐시/텔레그램명령/포지션검증루프 단위 테스트
================================
atlas_spot_main.py의 _get_price, CandleCache, _get_momentum_rank_pct,
_handle_tg_cmd, _position_reconcile_loop을 검증합니다.

실행:
  pytest tests/test_main_misc.py -v
"""

import re
import threading
import time
from pathlib import Path


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


@pytest.fixture(autouse=True)
def _clear_price_cache(monkeypatch):
    monkeypatch.setattr(sm, '_last_known_price', {})
    # 패스 내 시세 공유 캐시도 테스트마다 비운다 — 비우지 않으면 앞 테스트의
    # 가격이 남아 개별 조회 경로가 아예 실행되지 않는다.
    monkeypatch.setattr(sm, '_price_cache', {})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sm.time, 'sleep', lambda s: None)


@pytest.fixture
def _state(monkeypatch):
    fresh = {
        'equity': 10_000.0,
        'usdt_balance': 10_000.0,
        'universe': [],
        'universe_ranked': [],
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


# ══════════════════════════════════════════════════════════════
#  _get_price
# ══════════════════════════════════════════════════════════════

class TestValuationWarnThrottle:
    """평가 실패 경고는 통화별로 간격을 둔다.

    현물 마켓이 없는 자산(Simple Earn LD*, 상장폐지, 스테이킹)은 **영원히**
    실패한다. 잔고 폴러가 60초마다 도므로 억제가 없으면 같은 줄이 하루
    1,440회 쌓인다 — 실측으로 LDUSDT 한 건이 7일간 1,470회를 남겼고,
    그 사이 정작 중요한 경고가 묻혔다.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(sm, '_valuation_warned', {})

    def test_first_warning_passes(self):
        assert sm._valuation_warn_due('LDUSDT') is True

    def test_repeat_is_suppressed(self):
        sm._valuation_warn_due('LDUSDT')
        assert sm._valuation_warn_due('LDUSDT') is False, (
            '60초 폴링마다 같은 줄이 쌓여 로그가 뒤덮인다')

    def test_other_currency_not_suppressed(self):
        """한 통화의 억제가 다른 통화의 경고를 가리면 안 된다."""
        sm._valuation_warn_due('LDUSDT')
        assert sm._valuation_warn_due('PENGU') is True

    def test_warns_again_after_interval(self, monkeypatch):
        """사실은 계속 유효하므로 완전히 숨기지는 않는다."""
        sm._valuation_warn_due('LDUSDT')
        aged = time.time() - sm._VALUATION_WARN_INTERVAL - 1
        monkeypatch.setitem(sm._valuation_warned, 'LDUSDT', aged)
        assert sm._valuation_warn_due('LDUSDT') is True


class _FakeTickerExchange:
    def __init__(self, prices=None, raises_times=0):
        self._prices = prices or {}
        self._raises_times = raises_times
        self.calls = 0

    def fetch_ticker(self, ccxt_sym):
        self.calls += 1
        if self._raises_times > 0:
            self._raises_times -= 1
            raise RuntimeError('network error')
        return self._prices.get(ccxt_sym, {'last': 0, 'close': 0})


class TestGetPrice:
    def test_returns_price_on_success(self, monkeypatch):
        ex = _FakeTickerExchange(prices={'BTC/USDT': {'last': 50000.0, 'close': 50000.0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 50000.0

    def test_retries_then_succeeds(self, monkeypatch):
        ex = _FakeTickerExchange(prices={'BTC/USDT': {'last': 50000.0, 'close': 50000.0}},
                                  raises_times=2)
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 50000.0
        assert ex.calls == 3

    def test_falls_back_to_close_when_last_missing(self, monkeypatch):
        ex = _FakeTickerExchange(prices={'BTC/USDT': {'last': None, 'close': 49000.0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 49000.0

    def test_falls_back_to_cache_on_total_failure(self, monkeypatch):
        sm._last_known_price['BTC/USDT'] = (48000.0, time.time())
        ex = _FakeTickerExchange(raises_times=99)
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 48000.0

    def test_returns_zero_when_cache_stale(self, monkeypatch):
        sm._last_known_price['BTC/USDT'] = (48000.0, time.time() - sm._PRICE_CACHE_TTL - 1)
        ex = _FakeTickerExchange(raises_times=99)
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 0.0

    def test_returns_zero_when_no_cache_and_total_failure(self, monkeypatch):
        ex = _FakeTickerExchange(raises_times=99)
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        assert sm._get_price('BTC/USDT') == 0.0

    def test_updates_cache_on_success(self, monkeypatch):
        ex = _FakeTickerExchange(prices={'BTC/USDT': {'last': 50000.0, 'close': 50000.0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        sm._get_price('BTC/USDT')
        cached_price, ts = sm._last_known_price['BTC/USDT']
        assert cached_price == 50000.0
        assert ts > 0


# ══════════════════════════════════════════════════════════════
#  CandleCache
# ══════════════════════════════════════════════════════════════

class _FakeCandleExchange:
    def __init__(self, data=None, raises_times=0):
        self._data = data or []
        self._raises_times = raises_times
        self.calls = 0

    def fetch_ohlcv(self, ccxt_sym, timeframe, limit):
        self.calls += 1
        if self._raises_times > 0:
            self._raises_times -= 1
            raise RuntimeError('network error')
        return self._data


class TestCandleCache:
    def test_fetches_and_caches(self):
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(data=[[0, 1, 2, 0, 1, 100]])
        result = cache.get(ex, 'BTC/USDT', '1h', 100)
        assert result == [[0, 1, 2, 0, 1, 100]]
        assert ex.calls == 1

    def test_returns_cached_value_within_ttl(self):
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(data=[[0, 1, 2, 0, 1, 100]])
        cache.get(ex, 'BTC/USDT', '1h', 100)
        cache.get(ex, 'BTC/USDT', '1h', 100)
        assert ex.calls == 1  # 두 번째 호출은 캐시에서 반환, API 미호출

    def test_refetches_after_ttl_expiry(self):
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(data=[[0, 1, 2, 0, 1, 100]])
        cache.get(ex, 'BTC/USDT', '1h', 100)
        key = 'BTC/USDT_1h'
        data, _ts = cache._cache[key]
        cache._cache[key] = (data, time.time() - 61)
        cache.get(ex, 'BTC/USDT', '1h', 100)
        assert ex.calls == 2

    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(sm.time, 'sleep', lambda s: None)
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(data=[[0, 1, 2, 0, 1, 100]], raises_times=2)
        result = cache.get(ex, 'BTC/USDT', '1h', 100)
        assert result == [[0, 1, 2, 0, 1, 100]]
        assert ex.calls == 3

    def test_falls_back_to_stale_cache_on_total_failure(self, monkeypatch):
        monkeypatch.setattr(sm.time, 'sleep', lambda s: None)
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(data=[[0, 1, 2, 0, 1, 100]])
        cache.get(ex, 'BTC/USDT', '1h', 100)
        key = 'BTC/USDT_1h'
        data, _ts = cache._cache[key]
        cache._cache[key] = (data, time.time() - 61)
        ex2 = _FakeCandleExchange(raises_times=99)
        result = cache.get(ex2, 'BTC/USDT', '1h', 100)
        assert result == [[0, 1, 2, 0, 1, 100]]

    def test_returns_empty_when_no_cache_and_total_failure(self, monkeypatch):
        monkeypatch.setattr(sm.time, 'sleep', lambda s: None)
        cache = sm.CandleCache(ttl=60)
        ex = _FakeCandleExchange(raises_times=99)
        result = cache.get(ex, 'BTC/USDT', '1h', 100)
        assert result == []


# ══════════════════════════════════════════════════════════════
#  _get_momentum_rank_pct
# ══════════════════════════════════════════════════════════════

class TestGetMomentumRankPct:
    def test_top_ranked_returns_zero(self, _state):
        _state['universe_ranked'] = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
        assert sm._get_momentum_rank_pct('BTCUSDT') == 0.0

    def test_mid_ranked_returns_fraction(self, _state):
        _state['universe_ranked'] = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT']
        assert sm._get_momentum_rank_pct('ETHUSDT') == pytest.approx(0.25)

    def test_missing_symbol_returns_half(self, _state):
        _state['universe_ranked'] = ['BTCUSDT', 'ETHUSDT']
        assert sm._get_momentum_rank_pct('NOPEUSDT') == 0.5

    def test_no_ranked_falls_back_to_universe(self, _state):
        _state['universe_ranked'] = []
        _state['universe'] = ['BTCUSDT', 'ETHUSDT']
        assert sm._get_momentum_rank_pct('ETHUSDT') == 0.5

    def test_empty_universe_returns_half(self, _state):
        _state['universe_ranked'] = []
        _state['universe'] = []
        assert sm._get_momentum_rank_pct('BTCUSDT') == 0.5

    def test_single_symbol_universe_returns_zero(self, _state):
        _state['universe_ranked'] = ['BTCUSDT']
        assert sm._get_momentum_rank_pct('BTCUSDT') == 0.0


# ══════════════════════════════════════════════════════════════
#  _handle_tg_cmd
# ══════════════════════════════════════════════════════════════

class TestHandleTgCmd:
    def test_status_with_no_positions(self, _no_telegram):
        sm._handle_tg_cmd('/status')
        assert any('포지션 없음' in m for m in _no_telegram)

    def test_status_with_open_position_reports_pnl(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        monkeypatch.setattr(sm, '_get_price', lambda ccxt_sym: 110.0)
        sm._handle_tg_cmd('/status')
        assert any('+10.0%' in m for m in _no_telegram)

    def test_equity_reports_balances(self, _state, _no_telegram):
        sm._handle_tg_cmd('/equity')
        assert any('10,000.00' in m for m in _no_telegram)

    def test_pause_sets_state_and_notifies(self, _state, _no_telegram):
        _state['paused'] = False
        sm._handle_tg_cmd('/pause')
        assert _state['paused'] is True
        assert any('일시정지' in m for m in _no_telegram)

    def test_resume_clears_state(self, _state, _no_telegram):
        _state['paused'] = True
        sm._handle_tg_cmd('/resume')
        assert _state['paused'] is False

    def test_stop_touches_kill_switch(self, _no_telegram):
        assert not sm.SPOT_KILL_SWITCH.exists()
        sm._handle_tg_cmd('/stop')
        assert sm.SPOT_KILL_SWITCH.exists()

    def test_regime_with_cached_state(self, monkeypatch, _no_telegram):
        class _RS:
            regime = 'TRENDING_UP'
            adx = 30.5
        monkeypatch.setattr(sm, 'get_cached_regime', lambda: _RS())
        sm._handle_tg_cmd('/regime')
        assert any('TRENDING_UP' in m and '30.5' in m for m in _no_telegram)

    def test_regime_with_no_cached_state(self, monkeypatch, _no_telegram):
        monkeypatch.setattr(sm, 'get_cached_regime', lambda: None)
        sm._handle_tg_cmd('/regime')
        assert any('정보 없음' in m for m in _no_telegram)

    def test_unknown_command_does_nothing(self, _no_telegram):
        sm._handle_tg_cmd('/banana')
        assert _no_telegram == []


# ══════════════════════════════════════════════════════════════
#  _position_reconcile_loop (단일 실행)
# ══════════════════════════════════════════════════════════════

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


class _FakeBalanceExchange:
    def __init__(self, balances):
        self._balances = balances

    def fetch_balance(self, params=None):
        return self._balances


class TestReconcilePrefersActualFill:
    """잔고 0의 가장 흔한 이유는 **거래소 보호주문 체결**이다.

    확인하지 않고 MANUAL_SOLD로 적으면 사유(SL/TP)와 체결가가 모두 틀어진다.
    체결가를 몰라 '현재가'로 추정하는데, 체결과 검증 사이에 가격이 움직인다.
    이 통계는 Kelly 사이징과 전략 건강도(net PF)에 그대로 들어가므로 오염되면
    배분 판단까지 흔들린다. 실제로 라이브 42건 중 6건이 MANUAL_SOLD로 남았고,
    그중 하나는 pnl_r +1.31로 익절처럼 보였다.

    검증 루프가 5분 주기 전략 루프보다 먼저 도는 경합에서 발생한다.
    """

    def _pos(self, sl_id='STOP-9'):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        sm._update_position_order_id('S4', 'BTCUSDT', sl_id)

    def test_filled_stop_recorded_as_sl_with_real_price(self, monkeypatch,
                                                        _no_telegram):
        self._pos()
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))
        # 거래소에는 체결 정보가 남아 있다 — 실제 체결가 92.5
        monkeypatch.setattr(sm, '_fetch_stop_order',
                            lambda c, o: {'status': 'closed', 'filled': 10.0,
                                          'average': 92.5})
        monkeypatch.setattr(sm, '_get_price', lambda s: 105.0)   # 그 사이 반등
        sm._position_reconcile_loop(_OneShotEvent())
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        assert len(rows) == 1
        assert rows[0]['reason'] == 'SL', (
            f"체결된 손절이 {rows[0]['reason']}로 기록됐다 — 사유 통계 오염")
        assert rows[0]['exit_price'] == pytest.approx(92.5), (
            '현재가(105)로 추정해 손익이 왜곡됐다 — 실제 체결가는 92.5')

    def test_falls_back_to_manual_when_order_not_filled(self, monkeypatch,
                                                        _no_telegram):
        """진짜 수동매도(주문 미체결)면 기존 경로를 그대로 탄다."""
        self._pos()
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))
        monkeypatch.setattr(sm, '_fetch_stop_order',
                            lambda c, o: {'status': 'open'})
        monkeypatch.setattr(sm, '_get_price', lambda s: 105.0)
        sm._position_reconcile_loop(_OneShotEvent())
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        assert rows[0]['reason'] == 'MANUAL_SOLD'

    def test_order_lookup_failure_still_cleans_up(self, monkeypatch, _no_telegram):
        """체결 확인이 불가능해도 포지션이 DB에 남아 떠돌면 안 된다."""
        self._pos()
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))

        def _boom(strategy, symbol, ccxt_sym, pos):
            raise RuntimeError('조회 불가')
        monkeypatch.setattr(sm, '_handle_stop_order_state', _boom)
        monkeypatch.setattr(sm, '_get_price', lambda s: 105.0)
        sm._position_reconcile_loop(_OneShotEvent())
        assert sm._load_position('S4', 'BTCUSDT') is None

    def test_no_order_id_skips_lookup(self, monkeypatch, _no_telegram):
        """추적 주문이 없으면 조회 없이 기존 경로로 간다(불필요한 API 호출 방지)."""
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        called = []
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))
        monkeypatch.setattr(sm, '_handle_stop_order_state',
                            lambda *a: called.append(1) or False)
        monkeypatch.setattr(sm, '_get_price', lambda s: 105.0)
        sm._position_reconcile_loop(_OneShotEvent())
        assert called == []


class TestReconcileRDenominator:
    """검증 루프 정산의 R배수 분모는 **진입 시점 위험(orig_sl)** 이어야 한다.

    다른 4개 정산 경로(_spot_sell 3경로·_handle_stop_order_state)는 모두
    orig_sl을 쓰는데, 이 사본만 역사적으로 pos['sl'](추적조정 후)을 분모로
    썼다. 추적손절이 SL을 진입가 근처까지 올린 포지션이 수동매도로 정리되면
    위험 분모가 10→1로 줄어 pnl_r이 10배 부풀고, 그 값이 Kelly 사이징·
    전략 건강도·학습기 입력에 그대로 들어갔다.
    """

    def test_pnl_r_uses_entry_time_risk_not_trailed_sl(self, monkeypatch,
                                                       _no_telegram):
        # 진입 100 / 원 SL 90 (위험 $10/개) → 추적손절이 SL을 99로 올린 상태
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        with sm._db_lock, sm._db_conn() as conn:
            conn.execute("UPDATE spot_positions SET sl=99.0 "
                         "WHERE strategy='S4' AND symbol='BTCUSDT'")
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))
        monkeypatch.setattr(sm, '_get_price', lambda s: 110.0)
        sm._position_reconcile_loop(_OneShotEvent())
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        assert len(rows) == 1 and rows[0]['reason'] == 'MANUAL_SOLD'
        # pnl_u = (110-100)×10 = $100, 진입 시점 위험 = (100-90)×10 = $100 → 1R.
        # 조정 후 sl(99)을 분모로 쓰면 (100-99)×10 = $10 → 10R로 부푼다.
        assert rows[0]['pnl_r'] == pytest.approx(1.0), (
            f"pnl_r={rows[0]['pnl_r']} — 추적조정 후 sl을 분모로 써서 "
            f"R배수가 부풀었다 (orig_sl 기준이어야 함)")

    def test_legacy_position_without_orig_sl_falls_back_to_sl(self, monkeypatch,
                                                              _no_telegram):
        """orig_sl=0(마이그레이션 전 레거시 행)이면 기존 sl로 폴백한다
        — main:2252의 `or sl` 관용구와 동일."""
        sm._save_position('S4', 'BTCUSDT', 100.0, 95.0, 120.0, 10.0, 1000.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        with sm._db_lock, sm._db_conn() as conn:
            conn.execute("UPDATE spot_positions SET orig_sl=0 "
                         "WHERE strategy='S4' AND symbol='BTCUSDT'")
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _FakeBalanceExchange({'BTC': {'total': 0}}))
        monkeypatch.setattr(sm, '_get_price', lambda s: 110.0)
        sm._position_reconcile_loop(_OneShotEvent())
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        # 위험 = (100-95)×10 = $50, pnl_u = $100 → 2R
        assert rows[0]['pnl_r'] == pytest.approx(2.0)


class TestPositionReconcileLoop:
    def test_no_positions_does_nothing(self, monkeypatch, _no_telegram):
        ev = _OneShotEvent()
        monkeypatch.setattr(sm, '_get_ex', lambda: _FakeBalanceExchange({}))
        sm._position_reconcile_loop(ev)
        assert _no_telegram == []

    def test_zero_balance_deletes_position(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()
        ex = _FakeBalanceExchange({'BTC': {'total': 0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        sm._position_reconcile_loop(ev)
        assert sm._load_position('S4', 'BTCUSDT') is None
        assert any('잔고 0' in m for m in _no_telegram)

    def test_zero_balance_logs_manual_sold_trade(self, monkeypatch, _no_telegram):
        """수동매도 정리 시 spot_trades에 기록을 남겨야 통계가 유실되지 않는다
        (과거: _delete_position만 하고 거래 기록 없음 — _spot_sell 경로와 비일관)."""
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()
        ex = _FakeBalanceExchange({'BTC': {'total': 0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        monkeypatch.setattr(sm, '_get_price', lambda s: 105.0)
        sm._position_reconcile_loop(ev)
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        assert len(rows) == 1
        assert rows[0]['reason'] == 'MANUAL_SOLD'
        assert rows[0]['exit_price'] == pytest.approx(105.0)

    def test_zero_balance_price_unavailable_logs_at_entry(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()
        ex = _FakeBalanceExchange({'BTC': {'total': 0}})
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        monkeypatch.setattr(sm, '_get_price', lambda s: 0.0)
        sm._position_reconcile_loop(ev)
        with sm._db_lock, sm._db_conn() as conn:
            rows = [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]
        assert len(rows) == 1
        assert rows[0]['exit_price'] == pytest.approx(100.0)  # 진입가 폴백
        assert rows[0]['pnl_usdt'] == pytest.approx(0.0)

    def test_large_mismatch_updates_db_qty(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()
        ex = _FakeBalanceExchange({'BTC': {'total': 8.0}})  # 20% 괴리 > 10% 임계값
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        sm._position_reconcile_loop(ev)
        pos = sm._load_position('S4', 'BTCUSDT')
        assert pos['qty_tokens'] == pytest.approx(8.0)
        assert any('수량 불일치' in m for m in _no_telegram)

    def test_small_mismatch_does_not_update(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()
        ex = _FakeBalanceExchange({'BTC': {'total': 9.5}})  # 5% 괴리 < 10% 임계값
        monkeypatch.setattr(sm, '_get_ex', lambda: ex)
        sm._position_reconcile_loop(ev)
        pos = sm._load_position('S4', 'BTCUSDT')
        assert pos['qty_tokens'] == pytest.approx(10.0)
        assert _no_telegram == []

    def test_exception_is_caught_and_logged(self, monkeypatch, _no_telegram):
        sm._save_position('S4', 'BTCUSDT', 100.0, 90.0, 120.0, 10.0, 1000.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        ev = _OneShotEvent()

        class _RaisingExchange:
            def fetch_balance(self, params=None):
                raise RuntimeError('network error')

        monkeypatch.setattr(sm, '_get_ex', lambda: _RaisingExchange())
        sm._position_reconcile_loop(ev)  # 예외가 전파되지 않아야 함


class _FakeThread:
    def __init__(self, name, alive=True):
        self.name, self._alive = name, alive

    def is_alive(self):
        return self._alive


class TestDeadThreadDetection:
    """스레드가 조용히 죽는 것을 잡는다.

    감시견(_bot_alive)은 pgrep 기반이라 프로세스 생존만 본다. 봇은 10개
    데몬 스레드로 도는데, Loop1D가 예외로 죽어도 프로세스는 살아 있으므로
    감시견은 계속 녹색을 보고한다. 그 사이 SL/TP 판정과 청산이 멈춘다 —
    무인 운영에서 가장 위험한 실패 형태다.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(sm, '_thread_alerted', set())

    def test_detects_dead_thread(self):
        ts = [_FakeThread('Loop1D', alive=False), _FakeThread('TgWorker')]
        assert sm.find_dead_threads(ts) == ['Loop1D']

    def test_all_alive_reports_nothing(self):
        assert sm.find_dead_threads([_FakeThread('Loop1D')]) == []

    def test_reports_each_thread_once(self):
        """5초마다 도는 루프에서 검사하므로 억제가 없으면 알림이 폭주한다."""
        ts = [_FakeThread('Loop1D', alive=False)]
        assert sm.find_dead_threads(ts) == ['Loop1D']
        assert sm.find_dead_threads(ts) == []

    def test_survives_broken_thread_object(self):
        """감시 자체가 봇을 멈추면 안 된다."""
        class _Broken:
            name = 'X'

            def is_alive(self):
                raise RuntimeError('boom')
        assert sm.find_dead_threads([_Broken(), _FakeThread('Loop1D', alive=False)]) \
            == ['Loop1D']

    def test_every_spawned_thread_has_a_role_description(self):
        """알림은 '무엇이 멈췄는지'를 말해야 우선순위를 정할 수 있다."""
        src = Path(sm.__file__).read_text(encoding='utf-8')
        spawned = set(re.findall(r"_t\([_a-zA-Z]+,\s*'([A-Za-z0-9]+)'", src))
        missing = spawned - set(sm._THREAD_ROLE)
        assert not missing, f'역할 설명이 없는 스레드: {missing}'

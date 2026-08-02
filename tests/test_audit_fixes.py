"""
ATLAS — 정확성 감사 수정 회귀 테스트
================================
감사에서 나온 결함들이 다시 들어오지 못하게 고정한다.

① `_spot_sell` 잔고재시도 성공 → 거래기록·포지션삭제 누락 (치명)
② 고아 매도주문이 free=0을 만들어 허위 MANUAL_SOLD로 포지션 삭제 (치명)
③ `_handle_stop_order_state` 조회실패 레그가 있는데 재무장하며 ID 덮어씀
④ `_position_reconcile_loop` dry-run 가드 부재
⑤ Kelly 전패 구간이 표본부족보다 큰 스케일을 받는 역전
⑥ 전략 건강도 하드 차단의 영구화 (시간 창 부재)
⑦ 매수 체결 후 `_save_position` 실패 시 무보호 포지션 + 무알림

실행:
  pytest tests/test_audit_fixes.py -v
"""

import os
import sys
import threading
from datetime import datetime, timedelta, timezone
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
    sent = []
    monkeypatch.setattr(sm, '_tg', lambda msg: sent.append(msg))
    return sent


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'audit.db')
    sm.init_spot_db()


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'ABSENT')


@pytest.fixture
def _state(monkeypatch):
    fresh = {
        'equity': 10_000.0, 'usdt_balance': 10_000.0, 'peak_equity': 0.0,
        'ratchet_alert_tier': 0, 'paused': False, 'dry_run': False,
        'day_pnl': 0.0, 'day_start_eq': 10_000.0, 'daily_loss_alerted': False,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


def _trades():
    with sm._db_lock, sm._db_conn() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM spot_trades').fetchall()]


def _save_pos(strategy='S3', symbol='BTCUSDT', qty=10.0, sl_id='', tp_id=''):
    sm._save_position(strategy, symbol, 100.0, 95.0, 110.0, qty, 1000.0,
                       0.02, 'sl_tp', 0, 'TRENDING_UP')
    if sl_id or tp_id:
        sm._update_position_order_id(strategy, symbol, sl_id, tp_id)
    return sm._load_position(strategy, symbol)


class _Ex:
    """필요한 부분만 흉내내는 가짜 거래소."""

    def __init__(self):
        self.calls = []
        self.free = {}
        self.open_orders = []
        self.sell_errors = []      # 순서대로 소비할 예외 (None이면 성공)
        self.sold = []

    # --- 주문 ---
    def create_market_sell_order(self, ccxt_sym, qty):
        self.calls.append(('sell', ccxt_sym, qty))
        err = self.sell_errors.pop(0) if self.sell_errors else None
        if err:
            raise err
        self.sold.append(qty)
        return {'average': 105.0, 'price': 105.0, 'filled': qty}

    def create_order(self, ccxt_sym, otype, side, qty, price, params=None):
        self.calls.append(('create_order', ccxt_sym, qty))
        return {'id': 'NEWSTOP'}

    def cancel_order(self, oid, ccxt_sym):
        self.calls.append(('cancel', str(oid)))

    def fetch_open_orders(self, ccxt_sym):
        self.calls.append(('fetch_open_orders', ccxt_sym))
        return list(self.open_orders)

    def fetch_balance(self, params=None):
        self.calls.append(('fetch_balance',))
        return {'free': dict(self.free), **{k: {'total': v} for k, v in self.free.items()}}

    def amount_to_precision(self, ccxt_sym, qty):
        return f'{float(qty):.8f}'

    def price_to_precision(self, ccxt_sym, px):
        return f'{float(px):.2f}'


@pytest.fixture
def _ex(monkeypatch):
    ex = _Ex()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


# ══════════════════════════════════════════════════════════════
#  ① 잔고재시도 성공 경로 — 팔렸으면 반드시 기록된다
# ══════════════════════════════════════════════════════════════

class TestSellRetryAccounting:
    def test_retry_success_records_trade_and_deletes_position(self, _state, _ex):
        """감사 #1: 재시도로 실제 체결됐는데 거래기록·포지션삭제가 스킵되던 결함.
        결과적으로 다음 폴링에서 허위 MANUAL_SOLD가 기록됐다."""
        pos = _save_pos(qty=10.0)
        _ex.free = {'BTC': 9.99}
        # 사전잔고 확인은 통과시키고(9.99<10 → qty=9.99), 첫 매도만 실패시킨다
        _ex.sell_errors = [Exception('Account has insufficient balance for requested action')]
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'CROSS')

        assert sm._load_position('S3', 'BTCUSDT') is None, '포지션이 삭제돼야 한다'
        rows = _trades()
        assert len(rows) == 1
        assert rows[0]['reason'] == 'CROSS', f"실제 청산 사유여야 한다 (got {rows[0]['reason']})"
        assert rows[0]['reason'] != 'MANUAL_SOLD'
        assert _state['day_pnl'] != 0.0, 'day_pnl(일간 손실 한도 입력값)에 반영돼야 한다'

    def test_retry_failure_keeps_position(self, _state, _ex):
        pos = _save_pos(qty=10.0)
        _ex.free = {'BTC': 9.99}
        _ex.sell_errors = [Exception('insufficient balance'), Exception('still failing')]
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'CROSS')
        assert sm._load_position('S3', 'BTCUSDT') is not None
        assert _trades() == []

    def test_normal_sell_still_records_once(self, _state, _ex):
        pos = _save_pos(qty=10.0)
        _ex.free = {'BTC': 10.0}
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'TP')
        rows = _trades()
        assert len(rows) == 1 and rows[0]['reason'] == 'TP'


# ══════════════════════════════════════════════════════════════
#  ② 고아 매도주문 → 허위 MANUAL_SOLD 방지
# ══════════════════════════════════════════════════════════════

class TestOrphanOrderGuard:
    def test_locked_balance_is_recovered_not_treated_as_manual_sale(self, _state, _ex, _no_telegram):
        """감사 #3: DB에 ID가 없는 고아 OCO가 수량을 잠그면 free=0.
        이를 수동매도로 오판해 살아있는 포지션을 지우던 결함."""
        pos = _save_pos(qty=10.0)          # sl_order_id 없음 = 고아 상태
        _ex.free = {'BTC': 0.0}
        _ex.open_orders = [{'id': 'ORPHAN1', 'side': 'sell'}]

        def _unlock(oid, ccxt_sym):
            _ex.calls.append(('cancel', str(oid)))
            _ex.free = {'BTC': 10.0}       # 취소되면 잠금 해제
        _ex.cancel_order = _unlock

        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'SL')

        rows = _trades()
        assert len(rows) == 1
        assert rows[0]['reason'] == 'SL', '고아주문 취소 후 정상 청산돼야 한다'
        assert all('MANUAL_SOLD' != r['reason'] for r in rows)
        assert ('cancel', 'ORPHAN1') in _ex.calls

    def test_genuine_manual_sale_still_detected(self, _state, _ex):
        """미체결 매도주문이 없으면(=진짜 수동매도) 기존 동작 유지."""
        pos = _save_pos(qty=10.0)
        _ex.free = {'BTC': 0.0}
        _ex.open_orders = []
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'SL')
        rows = _trades()
        assert len(rows) == 1 and rows[0]['reason'] == 'MANUAL_SOLD'
        assert sm._load_position('S3', 'BTCUSDT') is None

    def test_open_order_query_failure_is_non_fatal(self, _state, _ex):
        pos = _save_pos(qty=10.0)
        _ex.free = {'BTC': 0.0}

        def _boom(ccxt_sym):
            raise Exception('rate limit')
        _ex.fetch_open_orders = _boom
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'SL')
        assert _trades()[0]['reason'] == 'MANUAL_SOLD'   # 원래 판정으로 폴백


# ══════════════════════════════════════════════════════════════
#  ③ 조회 실패 레그가 있으면 재무장 보류
# ══════════════════════════════════════════════════════════════

class TestRearmSafety:
    def test_unverified_leg_blocks_rearm(self, _state, _ex, monkeypatch):
        """감사 #4: 한 레그 조회 실패 + 다른 레그 취소 감지 시,
        재무장하면 살아있을지 모르는 레그의 ID를 덮어써 추적 불가가 된다."""
        pos = _save_pos(sl_id='111', tp_id='222')

        def _fetch(ccxt_sym, oid):
            return None if oid == '111' else {'status': 'canceled'}
        monkeypatch.setattr(sm, '_fetch_stop_order', _fetch)

        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is False
        after = sm._load_position('S3', 'BTCUSDT')
        assert after['sl_order_id'] == '111', '조회 못한 레그 ID를 잃으면 안 된다'
        assert ('create_order', 'BTC/USDT', 10.0) not in _ex.calls

    def test_all_legs_verified_rearms(self, _state, _ex, monkeypatch):
        pos = _save_pos(sl_id='111', tp_id='222')
        monkeypatch.setattr(sm, '_fetch_stop_order',
                            lambda ccxt_sym, oid: {'status': 'canceled'})
        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is False
        after = sm._load_position('S3', 'BTCUSDT')
        assert after['sl_order_id'] == 'NEWSTOP'


# ══════════════════════════════════════════════════════════════
#  ④ reconcile dry-run 가드
# ══════════════════════════════════════════════════════════════

class TestReconcileDryRunGuard:
    def test_dry_run_positions_survive_reconcile(self, _state, _ex):
        """감사 #5: dry-run은 실주문이 없어 잔고가 0 → 가상 포지션이 전부
        MANUAL_SOLD로 삭제되고 허위 기록이 실거래 통계를 오염시켰다."""
        _state['dry_run'] = True
        _save_pos(qty=10.0)
        _ex.free = {}

        stop = threading.Event()
        orig_wait = stop.wait

        def _wait_once(timeout=None):
            stop.set()          # 1회전만 돌게 한다
            return orig_wait(0)
        stop.wait = _wait_once
        sm._position_reconcile_loop(stop)

        assert sm._load_position('S3', 'BTCUSDT') is not None
        assert _trades() == [], 'dry-run 거래가 실거래 DB에 기록되면 안 된다'


# ══════════════════════════════════════════════════════════════
#  ⑤ Kelly 전패 역전
# ══════════════════════════════════════════════════════════════

def _insert(pnl_r, strategy='S3', pnl_usdt=None, fee=0.0, days_ago=1):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    pnl = pnl_usdt if pnl_usdt is not None else pnl_r * 10
    with sm._db_lock, sm._db_conn() as conn:
        conn.execute("""INSERT INTO spot_trades
            (strategy, symbol, pnl_usdt, pnl_r, fee_usdt, reason, entry_ts, exit_ts)
            VALUES (?,?,?,?,?,?,?,?)""",
            (strategy, 'BTCUSDT', pnl, pnl_r, fee, 'SL', ts, ts))


class TestKellyLosingStreak:
    def test_all_losses_not_larger_than_small_sample(self):
        """감사 #8: 전패 15건이 1.0을 받아, 표본부족(0.30)보다 3.3배 크게
        베팅하던 역전."""
        for _ in range(15):
            _insert(-0.5)
        streak = sm._get_kelly_scale('S3')
        for _ in range(sm.SPOT_KELLY_MIN_TRADES - 1):
            _insert(-0.5, strategy='S9')
        small_sample = sm._get_kelly_scale('S9')
        assert streak <= small_sample


# ══════════════════════════════════════════════════════════════
#  ⑥ 건강도 차단의 자동 해제 (시간 창)
# ══════════════════════════════════════════════════════════════

class TestHealthAutoRelease:
    def test_recent_bad_streak_blocks(self):
        for _ in range(5):
            _insert(1.0, pnl_usdt=10.0, days_ago=3)
        for _ in range(15):
            _insert(-1.0, pnl_usdt=-10.0, days_ago=3)
        assert sm._get_strategy_health_scale('S3') == 0.0

    def test_block_auto_releases_once_window_passes(self):
        """감사 #7: 차단되면 신규 거래가 없어 표본이 굳는다. 시간 창이 없으면
        '성과 회복 시 자동 해제'가 원리적으로 불가능(영구 차단)."""
        old = sm.SPOT_HEALTH_WINDOW_DAYS + 10
        for _ in range(5):
            _insert(1.0, pnl_usdt=10.0, days_ago=old)
        for _ in range(15):
            _insert(-1.0, pnl_usdt=-10.0, days_ago=old)
        assert sm._get_strategy_health_scale('S3') == 1.0

    def test_window_boundary_excludes_older_trades(self):
        for _ in range(20):
            _insert(-1.0, pnl_usdt=-10.0, days_ago=sm.SPOT_HEALTH_WINDOW_DAYS + 1)
        for _ in range(3):
            _insert(-1.0, pnl_usdt=-10.0, days_ago=1)
        # 창 안 표본이 최소치 미만 → 개입 없음
        assert sm._get_strategy_health_scale('S3') == 1.0


# ══════════════════════════════════════════════════════════════
#  ⑦ 포지션 저장 실패 시 알림 + 보호 시도
# ══════════════════════════════════════════════════════════════

class TestSavePositionFailure:
    def test_alerts_and_attempts_stop_on_persistent_failure(self, _state, _ex,
                                                             _no_telegram, monkeypatch):
        """감사 #10: 저장 실패 시 조용히 무보호 포지션이 생기고 알림도 없었다."""
        monkeypatch.setattr(sm, '_get_kelly_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_get_ratchet_scale', lambda: 1.0)
        _ex.create_market_buy_order = lambda cs, q: {'average': 100.0, 'filled': q}

        def _boom(*a, **k):
            raise Exception('database is locked')
        monkeypatch.setattr(sm, '_save_position', _boom)

        sig = {'sl': 95.0, 'tp': 110.0, 'rr': 2.0, 'exit_type': 'sl_tp', 'max_hold': 10}
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP')

        assert ok is False
        alerts = [m for m in _no_telegram if '저장 실패' in m]
        assert len(alerts) == 1, '운영자에게 반드시 알려야 한다'
        assert any(c[0] == 'create_order' for c in _ex.calls), '최소한 거래소 스탑은 시도해야 한다'

    def test_transient_failure_recovers_on_retry(self, _state, _ex, monkeypatch):
        monkeypatch.setattr(sm, '_get_kelly_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_get_ratchet_scale', lambda: 1.0)
        _ex.create_market_buy_order = lambda cs, q: {'average': 100.0, 'filled': q}

        real_save = sm._save_position
        calls = {'n': 0}

        def _flaky(*a, **k):
            calls['n'] += 1
            if calls['n'] == 1:
                raise Exception('database is locked')
            return real_save(*a, **k)
        monkeypatch.setattr(sm, '_save_position', _flaky)

        sig = {'sl': 95.0, 'tp': 110.0, 'rr': 2.0, 'exit_type': 'sl_tp', 'max_hold': 10}
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', sig, 100.0, 'TRENDING_UP') is True
        assert sm._load_position('S3', 'BTCUSDT') is not None

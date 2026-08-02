"""
ATLAS — 매매 로직 기대값 개선 4종 단위 테스트
================================
① half-Kelly (SPOT_KELLY_FRACTION)
② MAX_POSITIONS 자본 연동 (SPOT_EQUITY_PER_SLOT)
③ OCO 보호 주문 (SL+TP, S5 제외, 폴백 체인)
④ 전략 건강도 자기교정 (net PF 기반 감봉/차단)

실행:
  pytest tests/test_expectancy_upgrades.py -v
"""

import os
import sys
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
        'equity': 10_000.0, 'usdt_balance': 10_000.0, 'peak_equity': 0.0,
        'ratchet_alert_tier': 0, 'paused': False, 'dry_run': True,
        'day_pnl': 0.0, 'day_start_eq': 10_000.0, 'daily_loss_alerted': False,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


def _insert_trade(pnl_r, strategy='S3', pnl_usdt=None, fee=0.0):
    pnl = pnl_usdt if pnl_usdt is not None else pnl_r * 10
    with sm._db_lock, sm._db_conn() as conn:
        conn.execute("""INSERT INTO spot_trades
            (strategy, symbol, pnl_usdt, pnl_r, fee_usdt, reason, entry_ts, exit_ts)
            VALUES (?,?,?,?,?,?,?,?)""",
            (strategy, 'BTCUSDT', pnl, pnl_r, fee, 'TP',
             '2099-01-01T00:00:00', '2099-01-01T05:00:00'))


def _base_sig(entry=100.0, sl_pct=0.05, rr=2.0):
    sl = entry * (1 - sl_pct)
    return {'sl': sl, 'tp': entry * (1 + sl_pct * rr), 'rr': rr,
            'exit_type': 'sl_tp', 'max_hold': 10}


# ══════════════════════════════════════════════════════════════
#  ① half-Kelly
# ══════════════════════════════════════════════════════════════

class TestHalfKelly:
    def test_kelly_is_halved(self):
        """raw Kelly가 명확히 계산되는 분포에서 half가 적용되는지 검증.
        wr=0.5, avg_w=2.0, avg_l=0.5 → b=4 → raw K = 0.5-0.5/4 = 0.375
        → half = 0.1875 → 하한 0.30 미만이므로 SPOT_KELLY_SCALE_MIN."""
        for _ in range(10):
            _insert_trade(2.0)
        for _ in range(10):
            _insert_trade(-0.5)
        assert sm._get_kelly_scale('S3') == pytest.approx(sm.SPOT_KELLY_SCALE_MIN)

    def test_strong_edge_still_above_floor(self):
        """wr=0.75, avg_w=2, avg_l=0.4 → b=5 → raw=0.70 → half=0.35 (>하한)."""
        for _ in range(15):
            _insert_trade(2.0)
        for _ in range(5):
            _insert_trade(-0.4)
        scale = sm._get_kelly_scale('S3')
        assert scale == pytest.approx(0.5 * (0.75 - 0.25 / 5.0), abs=1e-6)

    def test_backtest_uses_same_fraction(self):
        """백테스트 상수 패리티 — 라이브와 같은 SPOT_KELLY_FRACTION 사용."""
        import atlas_spot_backtest as bt
        assert bt.SPOT_KELLY_FRACTION == sm.SPOT_KELLY_FRACTION == 0.5


# ══════════════════════════════════════════════════════════════
#  ② MAX_POSITIONS 자본 연동
# ══════════════════════════════════════════════════════════════

class TestEquityLinkedSlots:
    def _fill_positions(self, n):
        for i in range(n):
            sm._save_position('S3', f'SYM{i}USDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                               0.02, 'sl_tp', 10, 'TRENDING_UP')

    def test_small_account_gets_fewer_slots(self, _state):
        """$134 계좌 → 슬롯 134//20 = 6개. 6개 차 있으면 차단."""
        _state['equity'] = 134.0
        _state['usdt_balance'] = 134.0
        self._fill_positions(6)
        ok = sm._spot_buy('S3', 'NEWUSDT', 'NEW/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_small_account_below_cap_allows(self, _state):
        _state['equity'] = 134.0
        _state['usdt_balance'] = 134.0
        self._fill_positions(5)
        ok = sm._spot_buy('S3', 'NEWUSDT', 'NEW/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is True

    def test_large_account_capped_at_max_positions(self, _state):
        """자본이 커도 SPOT_MAX_POSITIONS를 넘지 않음."""
        self._fill_positions(sm.SPOT_MAX_POSITIONS)
        ok = sm._spot_buy('S3', 'NEWUSDT', 'NEW/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False

    def test_zero_equity_allows_single_slot(self, _state):
        """잔고 미조회(0)여도 최소 1슬롯은 보장 (잔고 확인이 별도 차단)."""
        _state['equity'] = 0.0
        _state['usdt_balance'] = 0.0
        self._fill_positions(1)
        ok = sm._spot_buy('S3', 'NEWUSDT', 'NEW/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is False   # 1슬롯 이미 사용 → 차단


# ══════════════════════════════════════════════════════════════
#  ③ OCO 보호 주문
# ══════════════════════════════════════════════════════════════

class _FakeOcoExchange:
    STEP = 1e-05     # BTC/USDT 수량 최소단위

    def __init__(self):
        self.buy_result = None      # None이면 기본(수수료 없음) 응답
        self.calls = []
        self.oco_result = {'orderReports': [
            {'orderId': 111, 'type': 'STOP_LOSS_LIMIT'},
            {'orderId': 222, 'type': 'LIMIT_MAKER'},
        ]}
        self.fetch_results = {}    # order_id(str) → dict/Exception
        self.free_balance = {}

    def privatePostOrderListOco(self, params):
        self.calls.append(('oco', params))
        if isinstance(self.oco_result, Exception):
            raise self.oco_result
        return self.oco_result

    def create_order(self, ccxt_sym, otype, side, qty, price, params=None):
        self.calls.append(('create_order', ccxt_sym, otype, side, qty, price, params))
        return {'id': 'STOP-FALLBACK'}

    def cancel_order(self, order_id, ccxt_sym):
        self.calls.append(('cancel', str(order_id), ccxt_sym))

    def fetch_order(self, order_id, ccxt_sym):
        self.calls.append(('fetch', str(order_id), ccxt_sym))
        r = self.fetch_results.get(str(order_id))
        if isinstance(r, Exception):
            raise r
        return r or {'status': 'open'}

    def amount_to_precision(self, ccxt_sym, qty):
        """바이낸스와 동일하게 내림(TRUNCATE)."""
        import math
        return f'{math.floor(float(qty) / self.STEP) * self.STEP:.8f}'

    def price_to_precision(self, ccxt_sym, px):
        return f'{float(px):.2f}'

    def create_market_buy_order(self, ccxt_sym, qty):
        self.calls.append(('buy', ccxt_sym, qty))
        if self.buy_result is not None:
            return self.buy_result
        return {'average': None, 'price': None, 'filled': qty}

    def create_market_sell_order(self, ccxt_sym, qty):
        self.calls.append(('sell', ccxt_sym, qty))
        return {'average': None, 'price': None}

    def fetch_balance(self):
        return {'free': dict(self.free_balance)}


@pytest.fixture
def _oco_ex(monkeypatch):
    ex = _FakeOcoExchange()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


def _save_pos(strategy='S3', sl_id='', tp_id='', tp=110.0):
    sm._save_position(strategy, 'BTCUSDT', 100.0, 95.0, tp, 10.0, 1000.0,
                       0.02, 'sl_tp', 0, 'TRENDING_UP')
    if sl_id or tp_id:
        sm._update_position_order_id(strategy, 'BTCUSDT', sl_id, tp_id)


class TestPlaceProtectiveOrders:
    def test_oco_places_both_legs(self, _oco_ex):
        sl_id, tp_id = sm._place_protective_orders('S3', 'BTCUSDT', 'BTC/USDT',
                                                   10.0, 95.0, 110.0)
        assert (sl_id, tp_id) == ('111', '222')
        kind, params = _oco_ex.calls[0]
        assert kind == 'oco'
        assert params['symbol'] == 'BTCUSDT'
        assert params['side'] == 'SELL'
        assert float(params['abovePrice']) == pytest.approx(110.0)
        assert float(params['belowStopPrice']) == pytest.approx(95.0)

    def test_no_tp_uses_stop_only(self, _oco_ex):
        sl_id, tp_id = sm._place_protective_orders('S3', 'BTCUSDT', 'BTC/USDT',
                                                   10.0, 95.0, 0.0)
        assert tp_id == ''
        assert sl_id == 'STOP-FALLBACK'
        assert all(c[0] != 'oco' for c in _oco_ex.calls)

    def test_oco_failure_falls_back_to_stop(self, _oco_ex):
        _oco_ex.oco_result = RuntimeError('oco rejected')
        sl_id, tp_id = sm._place_protective_orders('S3', 'BTCUSDT', 'BTC/USDT',
                                                   10.0, 95.0, 110.0)
        assert sl_id == 'STOP-FALLBACK' and tp_id == ''

    def test_buy_excludes_tp_for_s5(self, _state, _oco_ex, monkeypatch):
        """S5는 동적 TP(BB 상단) → OCO 대신 스탑 단독."""
        _state['dry_run'] = False
        ok = sm._spot_buy('S5', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'RANGING')
        assert ok is True
        assert all(c[0] != 'oco' for c in _oco_ex.calls)
        pos = sm._load_position('S5', 'BTCUSDT')
        assert pos['sl_order_id'] == 'STOP-FALLBACK'
        assert pos['tp_order_id'] == ''

    def test_buy_stores_both_ids_for_oco_strategy(self, _state, _oco_ex):
        _state['dry_run'] = False
        ok = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['sl_order_id'] == '111'
        assert pos['tp_order_id'] == '222'


class TestTpLegFillDetection:
    """거래소 보호주문 경로는 실거래 전용(_manage_position이 dry-run을 가드)."""

    @pytest.fixture(autouse=True)
    def _live(self, _state):
        _state['dry_run'] = False

    def test_tp_fill_logs_tp_trade(self, _state, _oco_ex, _temp_db):
        _save_pos(sl_id='111', tp_id='222')
        _oco_ex.fetch_results['111'] = {'status': 'canceled'}   # OCO 반대 레그 자동취소
        _oco_ex.fetch_results['222'] = {'status': 'closed', 'filled': 10.0, 'average': 110.2}
        pos = sm._load_position('S3', 'BTCUSDT')
        handled = sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos)
        assert handled is True
        assert sm._load_position('S3', 'BTCUSDT') is None
        with sm._db_lock, sm._db_conn() as conn:
            row = dict(conn.execute('SELECT * FROM spot_trades').fetchone())
        assert row['reason'] == 'TP'
        assert row['exit_price'] == pytest.approx(110.2)

    def test_sl_fill_still_logs_sl(self, _state, _oco_ex, _temp_db):
        _save_pos(sl_id='111', tp_id='222')
        _oco_ex.fetch_results['111'] = {'status': 'closed', 'filled': 10.0, 'average': 94.9}
        _oco_ex.fetch_results['222'] = {'status': 'canceled'}
        pos = sm._load_position('S3', 'BTCUSDT')
        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is True
        with sm._db_lock, sm._db_conn() as conn:
            row = dict(conn.execute('SELECT * FROM spot_trades').fetchone())
        assert row['reason'] == 'SL'

    def test_external_cancel_rearms_protective_orders(self, _state, _oco_ex, _temp_db):
        """양 레그가 외부 취소되면 (체결 없이) 보호 주문 재무장."""
        _save_pos(sl_id='111', tp_id='222')
        _oco_ex.fetch_results['111'] = {'status': 'canceled'}
        _oco_ex.fetch_results['222'] = {'status': 'canceled'}
        _oco_ex.oco_result = {'orderReports': [
            {'orderId': 333, 'type': 'STOP_LOSS_LIMIT'},
            {'orderId': 444, 'type': 'LIMIT_MAKER'},
        ]}
        pos = sm._load_position('S3', 'BTCUSDT')
        assert sm._handle_stop_order_state('S3', 'BTCUSDT', 'BTC/USDT', pos) is False
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['sl_order_id'] == '333'
        assert pos['tp_order_id'] == '444'

    def test_sell_cancels_both_legs_first(self, _state, _oco_ex):
        _state['dry_run'] = False
        _save_pos(sl_id='111', tp_id='222')
        _oco_ex.free_balance = {'BTC': 10.0}
        pos = sm._load_position('S3', 'BTCUSDT')
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 105.0, 'CROSS')
        cancels = [c[1] for c in _oco_ex.calls if c[0] == 'cancel']
        kinds = [c[0] for c in _oco_ex.calls]
        assert '111' in cancels and '222' in cancels
        assert kinds.index('cancel') < kinds.index('sell')


# ══════════════════════════════════════════════════════════════
#  ⑤ 기초자산 수수료 차감 — 실수령 수량 (거래소 보호주문 -2010 방지)
# ══════════════════════════════════════════════════════════════

class TestNetFilledQty:
    def test_ccxt_parsed_fee_in_base_asset(self):
        order = {'filled': 0.001, 'fee': {'currency': 'BTC', 'cost': 0.000001}}
        assert sm._net_filled_qty(order, 'BTC/USDT', 0.001) == pytest.approx(0.000999)

    def test_fees_list_form(self):
        order = {'filled': 2.0, 'fees': [{'currency': 'ETH', 'cost': 0.002}]}
        assert sm._net_filled_qty(order, 'ETH/USDT', 2.0) == pytest.approx(1.998)

    def test_raw_fills_fallback(self):
        """ccxt 파싱이 비었을 때 원시 응답 fills에서 커미션 추출."""
        order = {'filled': 0.001, 'info': {'fills': [
            {'commission': '0.00000050', 'commissionAsset': 'BTC'},
            {'commission': '0.00000050', 'commissionAsset': 'BTC'},
        ]}}
        assert sm._net_filled_qty(order, 'BTC/USDT', 0.001) == pytest.approx(0.000999)

    def test_bnb_fee_leaves_qty_intact(self):
        """BNB로 수수료를 내면 기초자산 차감이 없어 체결량 그대로."""
        order = {'filled': 0.001, 'fee': {'currency': 'BNB', 'cost': 0.0004}}
        assert sm._net_filled_qty(order, 'BTC/USDT', 0.001) == pytest.approx(0.001)

    def test_missing_fee_data_falls_back_to_filled(self):
        assert sm._net_filled_qty({'filled': 0.5}, 'BTC/USDT', 0.7) == pytest.approx(0.5)

    def test_missing_filled_falls_back_to_requested(self):
        assert sm._net_filled_qty({}, 'BTC/USDT', 0.7) == pytest.approx(0.7)

    def test_never_negative(self):
        order = {'filled': 0.001, 'fee': {'currency': 'BTC', 'cost': 0.005}}
        assert sm._net_filled_qty(order, 'BTC/USDT', 0.001) == 0.0

    def test_buy_stores_net_qty_in_position(self, _state, _oco_ex):
        """매수 후 포지션 수량은 '팔 수 있는' 실수령량이어야 한다."""
        _state['dry_run'] = False
        _oco_ex.buy_result = {'average': 100.0, 'price': 100.0, 'filled': 1.0,
                              'fee': {'currency': 'BTC', 'cost': 0.001}}
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['qty_tokens'] == pytest.approx(0.999)

    def test_protective_order_qty_never_exceeds_holdings(self, _oco_ex):
        """보호주문 수량은 정밀도 내림 후에도 보유량을 넘지 않는다."""
        holdings = 0.000999
        sm._place_protective_orders('S3', 'BTCUSDT', 'BTC/USDT', holdings, 95_000.0, 110_000.0)
        kind, params = _oco_ex.calls[0]
        assert kind == 'oco'
        assert float(params['quantity']) <= holdings

    def test_stop_only_qty_truncated(self, _oco_ex):
        sm._place_stop_loss_order('S3', 'BTCUSDT', 'BTC/USDT', 0.000999, 95_000.0)
        call = [c for c in _oco_ex.calls if c[0] == 'create_order'][0]
        assert call[4] <= 0.000999      # 요청 수량 인자

    def test_zero_qty_skips_order(self, _oco_ex):
        """정밀도 내림으로 0이 되면 주문을 내지 않는다."""
        assert sm._place_protective_orders('S3', 'BTCUSDT', 'BTC/USDT',
                                           1e-9, 95_000.0, 110_000.0) == ('', '')
        assert _oco_ex.calls == []


# ══════════════════════════════════════════════════════════════
#  ④ 전략 건강도 자기교정
# ══════════════════════════════════════════════════════════════

class TestStrategyHealth:
    def test_insufficient_sample_no_intervention(self):
        for _ in range(sm.SPOT_HEALTH_MIN_TRADES - 1):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        assert sm._get_strategy_health_scale('S3') == 1.0

    def test_healthy_pf_full_scale(self):
        for _ in range(15):
            _insert_trade(1.0, pnl_usdt=15.0)
        for _ in range(10):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        assert sm._get_strategy_health_scale('S3') == 1.0   # PF 2.25

    def test_losing_pf_soft_demotion(self):
        """net PF = 90/110 ≈ 0.82 → 0.5 감봉."""
        for _ in range(9):
            _insert_trade(1.0, pnl_usdt=10.0)
        for _ in range(11):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        assert sm._get_strategy_health_scale('S3') == sm.SPOT_HEALTH_SOFT_SCALE

    def test_deep_losing_pf_blocks(self):
        """net PF = 50/150 ≈ 0.33 < 0.7 → 차단(0.0)."""
        for _ in range(5):
            _insert_trade(1.0, pnl_usdt=10.0)
        for _ in range(15):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        assert sm._get_strategy_health_scale('S3') == 0.0

    def test_pf_uses_net_of_fees(self):
        """gross PF는 1.0이지만 수수료 차감 net으로는 <1 → 감봉."""
        for _ in range(10):
            _insert_trade(1.0, pnl_usdt=10.0, fee=1.0)
        for _ in range(10):
            _insert_trade(-1.0, pnl_usdt=-10.0, fee=1.0)
        assert sm._get_strategy_health_scale('S3') == sm.SPOT_HEALTH_SOFT_SCALE

    def test_strategy_isolation(self):
        for _ in range(25):
            _insert_trade(-1.0, pnl_usdt=-10.0, strategy='S4')
        assert sm._get_strategy_health_scale('S3') == 1.0
        assert sm._get_strategy_health_scale('S4') == 0.0

    def test_blocked_strategy_cannot_buy_and_alerts_once(self, _state, _no_telegram):
        for _ in range(5):
            _insert_trade(1.0, pnl_usdt=10.0)
        for _ in range(15):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        ok1 = sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        ok2 = sm._spot_buy('S3', 'ETHUSDT', 'ETH/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok1 is False and ok2 is False
        assert len([m for m in _no_telegram if '자동 차단' in m]) == 1   # 알림 1회만

    def test_soft_demotion_halves_position_risk(self, _state):
        """감봉 상태에서 진입 시 risk_pct가 절반으로 줄어드는지."""
        for _ in range(9):
            _insert_trade(1.0, pnl_usdt=10.0)
        for _ in range(11):
            _insert_trade(-1.0, pnl_usdt=-10.0)
        ok = sm._spot_buy('S3', 'ETHUSDT', 'ETH/USDT', _base_sig(), 100.0, 'TRENDING_UP')
        assert ok is True
        pos = sm._load_position('S3', 'ETHUSDT')
        # 표본 20건 → Kelly도 활성(min 10 초과). 검증 초점은 health 반영 여부:
        # health=0.5가 빠지면 risk_pct가 2배가 되므로 상한으로 구분 가능
        kelly = sm._get_kelly_scale('S3')
        expected = sm.SPOT_BASE_RISK_PCT * kelly * 1.0 * 1.0 * sm.SPOT_HEALTH_SOFT_SCALE
        assert pos['risk_pct'] == pytest.approx(expected, rel=1e-3)

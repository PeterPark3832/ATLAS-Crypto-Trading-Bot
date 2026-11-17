"""atlas_fut_main — 가짜 거래소로 주문 흐름 전체를 검증한다 (네트워크 없음)."""

from dataclasses import replace
from decimal import Decimal

import atlas_fut_config as cfg
import atlas_fut_core as core
import atlas_fut_main as fm
from atlas_fut_core import DAY_MS
from atlas_fut_exchange import BinanceAPIError, SymbolRules

DAY0 = 1767225600000                     # 2026-01-01 00:00 UTC (테스트 기준일)
RULES = SymbolRules('X', Decimal('0.01'), Decimal('0.001'), Decimal('0.001'), Decimal('5'),
                    Decimal('0.001'), Decimal('0.001'))


class Clock:
    def __init__(self, ms):
        self.ms = ms

    def __call__(self):
        return self.ms / 1000


class FakeEx:
    base_url = 'fake://'

    def __init__(self, clock):
        self.clock = clock
        self.daily: dict[str, list[list]] = {}
        self.book: dict[str, tuple[float, float]] = {}
        self.mark: dict[str, float] = {}
        self.funding: dict[str, float] = {}
        self.equity = 10_000.0
        self.positions: dict[str, tuple[float, float]] = {}
        self.orders: dict[int, dict] = {}
        self.algos: dict[int, dict] = {}
        self.fills: list[dict] = []
        self.income_rows: list[dict] = []
        self.hedge = False
        self.reject_next_gtx = False      # 주문 직전에 가격이 움직여 메이커 주문이 거부되는 상황
        self.crash_after_send = False     # 주문은 거래소에 들어갔는데 응답을 받기 전에 프로세스가 죽는 상황
        self.next_id = 1

    # 기본
    def sync_time(self):
        return 0

    def exchange_info(self):
        return {}

    def rules(self, symbol):
        return replace(RULES, symbol=symbol)

    def klines(self, symbol, interval, limit=500, start_ms=None):
        return list(self.daily.get(symbol, [])) if interval == '1d' else []

    def premium_index(self, symbol=None):
        return [{'symbol': s, 'markPrice': str(m), 'lastFundingRate': str(self.funding.get(s, 0.0001))}
                for s, m in self.mark.items()]

    def book_ticker(self, symbol):
        b, a = self.book[symbol]
        return {'bidPrice': str(b), 'askPrice': str(a)}

    def account(self):
        return {'totalMarginBalance': str(self.equity)}

    def position_risk(self, symbol=None):
        return [{'symbol': s, 'positionAmt': str(a), 'entryPrice': str(p)} for s, (a, p) in self.positions.items() if a]

    def get_position_mode(self):
        return self.hedge

    def set_position_mode(self, hedge):
        self.hedge = hedge

    def get_multi_assets(self):
        return False

    def set_margin_type(self, symbol, margin_type):
        return {}

    def set_leverage(self, symbol, lev):
        return {}

    def income(self, income_type=None, symbol=None, start_ms=None, limit=1000):
        return [r for r in self.income_rows if income_type in (None, r.get('incomeType'))
                and (start_ms is None or int(r.get('time', 0)) >= start_ms)]

    def user_trades(self, symbol, start_ms=None, limit=500):
        return [f for f in self.fills if f['symbol'] == symbol]

    # 주문
    def _id(self):
        self.next_id += 1
        return self.next_id

    def _fill(self, o, px, qty):
        s, sign = o['symbol'], (1 if o['side'] == 'BUY' else -1)
        amt, entry = self.positions.get(s, (0.0, 0.0))
        realized = 0.0
        if amt == 0 or (amt > 0) == (sign > 0):
            new = amt + sign * qty
            entry = (abs(amt) * entry + qty * px) / abs(new)
        else:
            q = min(qty, abs(amt))
            realized = q * (px - entry) * (1 if amt > 0 else -1)
            new = amt + sign * q
        self.positions[s] = (round(new, 10), entry if new else 0.0)
        fee = px * qty * (0.0002 if o['type'] == 'LIMIT' else 0.0005)
        self.fills.append({'symbol': s, 'side': o['side'], 'price': str(px), 'qty': str(qty),
                           'realizedPnl': str(realized), 'commission': str(fee), 'time': int(self.clock.ms)})
        o.update(status='FILLED', executedQty=str(qty), avgPrice=str(px))

    def new_order(self, symbol, side, order_type, quantity, *, price=None, time_in_force=None,
                  reduce_only=False, client_id=None):
        oid = self._id()
        o = {'orderId': oid, 'symbol': symbol, 'side': side, 'type': order_type, 'origQty': str(quantity),
             'price': str(price) if price is not None else '0', 'status': 'NEW', 'executedQty': '0',
             'avgPrice': '0', 'reduceOnly': reduce_only, 'timeInForce': time_in_force,
             'clientOrderId': client_id or f'web_{oid}'}
        bid, ask = self.book[symbol]
        if order_type == 'LIMIT' and time_in_force == 'GTX' and (self.reject_next_gtx or
                                                                 (side == 'BUY' and price >= Decimal(str(ask))) or
                                                                 (side == 'SELL' and price <= Decimal(str(bid)))):
            o['status'] = 'EXPIRED'
            self.reject_next_gtx = False
        if order_type == 'MARKET':
            self._fill(o, ask if side == 'BUY' else bid, float(quantity))
        self.orders[oid] = o
        if self.crash_after_send:
            self.crash_after_send = False
            raise RuntimeError('응답을 받기 전에 프로세스가 죽음 (시뮬레이션)')
        return dict(o)

    def fill(self, oid, px=None):
        o = self.orders[oid]
        self._fill(o, px if px is not None else float(o['price']), float(o['origQty']))

    def query_order(self, symbol, order_id=None, client_id=None):
        if order_id is None:
            for o in self.orders.values():
                if o['clientOrderId'] == client_id:
                    return dict(o)
            raise BinanceAPIError(400, -2013, 'Order does not exist.', '/fapi/v1/order')
        return dict(self.orders[order_id])

    def cancel_order(self, symbol, order_id=None, client_id=None):
        o = self.orders[order_id]
        if o['status'] not in ('NEW', 'PARTIALLY_FILLED'):
            raise BinanceAPIError(400, -2011, 'Unknown order sent.', '/fapi/v1/order')
        o['status'] = 'CANCELED'
        return dict(o)

    def open_orders(self, symbol=None):
        return [dict(o) for o in self.orders.values() if o['status'] in ('NEW', 'PARTIALLY_FILLED')]

    def new_stop_market(self, symbol, side, quantity, trigger, *, working_type='MARK_PRICE', client_algo_id=None):
        aid = self._id()
        self.algos[aid] = {'algoId': aid, 'symbol': symbol, 'side': side, 'quantity': str(quantity),
                           'triggerPrice': str(trigger), 'algoStatus': 'NEW', 'workingType': working_type,
                           'clientAlgoId': client_algo_id or f'web_{aid}'}
        return dict(self.algos[aid])

    def trigger(self, aid, px):
        a = self.algos[aid]
        a['algoStatus'] = 'TRIGGERED'
        o = {'orderId': self._id(), 'symbol': a['symbol'], 'side': a['side'], 'type': 'MARKET'}
        self._fill(o, px, float(a['quantity']))

    def query_algo_order(self, algo_id=None, client_algo_id=None):
        return dict(self.algos[algo_id])

    def cancel_algo_order(self, algo_id=None, client_algo_id=None):
        a = self.algos[algo_id]
        if a['algoStatus'] != 'NEW':
            raise BinanceAPIError(400, -2011, 'Unknown algo order', '/fapi/v1/algoOrder')
        a['algoStatus'] = 'CANCELED'
        return dict(a)

    def open_algo_orders(self, symbol=None):
        return [dict(a) for a in self.algos.values() if a['algoStatus'] == 'NEW']


def _daily(closes, vols, end_ms):
    """마지막 원소가 end_ms 직전에 끝난 일봉이 되도록 + 형성 중인 오늘 봉 하나."""
    n = len(closes)
    out, prev = [], closes[0]
    for i, (c, v) in enumerate(zip(closes, vols, strict=True)):
        ts = end_ms - (n - i) * DAY_MS
        out.append([ts, prev, max(prev, c) + 0.5, min(prev, c) - 0.5, c, v])
        prev = c
    out.append([end_ms, prev, prev + 1, prev - 1, prev, 10.0])       # 형성 중 — 버려져야 한다
    return out


def _flat(n=70):
    return [100 + (0.5 if i % 2 else -0.5) for i in range(n)]


def _bot(mode='testnet', regime='TRENDING_UP', clock_ms=DAY0 + 90_000):
    clock = Clock(clock_ms)
    ex = FakeEx(clock)
    store = fm.Store(':memory:')
    msgs: list[str] = []
    bot = fm.FuturesBot(ex, store, msgs.append, mode=mode, clock=clock, observe_equity=10_000)

    def fixed_regime():
        bot.last_regime = regime
        return regime, core.RegimeInputs(64000, 76000, 33.0, 1.0, 0.04), 30.0

    bot.current_regime = fixed_regime
    return bot, ex, store, msgs, clock


def _with_breakout(ex, symbol='ETHUSDT', up=True):
    closes = _flat() + [112.0 if up else 88.0]
    vols = [1000.0] * 70 + [6000.0]
    ex.daily[symbol] = _daily(closes, vols, DAY0)
    px = closes[-1]
    ex.book[symbol] = (px - 0.01, px + 0.01)
    ex.mark[symbol] = px


def _open_long_position(bot, ex, store):
    _with_breakout(ex)
    rep = bot.on_day_close()
    oid = store.entries()['ETHUSDT']['order_id']
    ex.fill(oid)
    bot.poll_entries()
    return rep, store.positions()['ETHUSDT']


def test_day_close_places_maker_limit_entry():
    bot, ex, store, msgs, _ = _bot()
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert rep['regime'] == 'TRENDING_UP' and len(rep['entries']) == 1
    e = store.entries()['ETHUSDT']
    o = ex.orders[e['order_id']]
    assert o['type'] == 'LIMIT' and o['timeInForce'] == 'GTX' and o['status'] == 'NEW'
    assert Decimal(o['price']) <= Decimal('111.99')                   # 매수 호가 이하 → 메이커
    risk = float(e['qty']) * e['sl_dist']
    assert abs(risk - 10_000 * cfg.FUT_RISK_PCT) < 1.0                # 위험 = 자산 0.5%
    assert 'ETHUSDT' in msgs[-1] and store.get_state('last_day_processed') == str(DAY0)


def test_fill_places_stop_on_algo_api_and_take_profit():
    bot, ex, store, msgs, _ = _bot()
    _, pos = _open_long_position(bot, ex, store)
    assert 'ETHUSDT' not in store.entries()
    algo = ex.algos[pos['sl_algo_id']]
    assert algo['side'] == 'SELL' and Decimal(algo['triggerPrice']) <= Decimal(str(pos['entry']))
    tp = ex.orders[pos['tp_order_id']]
    assert tp['reduceOnly'] and tp['side'] == 'SELL' and Decimal(tp['price']) > Decimal(str(pos['entry']))
    assert any('진입' in m for m in msgs)


def test_unfilled_limit_falls_back_to_market_after_window():
    bot, ex, store, _, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    bot.poll_entries()
    pos = store.positions()['ETHUSDT']
    assert pos['maker'] is False
    assert any(o['type'] == 'MARKET' and o['status'] == 'FILLED' for o in ex.orders.values())
    assert pos['sl_algo_id'] in ex.algos


def test_fallback_cancelled_when_regime_changed():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()

    def micro():
        return 'MICRO_RANGING', core.RegimeInputs(1, 1, 1, 0, 0), 10.0

    bot.current_regime = micro
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    bot.poll_entries()
    assert store.positions() == {} and store.entries() == {}
    assert any('진입 취소' in m for m in msgs)


def test_post_only_rejection_is_repegged():
    bot, ex, store, _, _ = _bot()
    _with_breakout(ex)
    ex.reject_next_gtx = True                 # 걸자마자 가격이 닿아 메이커 주문이 거부됨
    rep = bot.on_day_close()
    first = rep['entries'][0]
    assert first['status'] == 'EXPIRED'
    bot.poll_entries()
    e = store.entries()['ETHUSDT']
    assert e['order_id'] != first['order_id'] and ex.orders[e['order_id']]['status'] == 'NEW'


def test_take_profit_fill_cancels_stop_and_records_trade():
    bot, ex, store, msgs, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    clock.ms += DAY_MS
    ex.fill(pos['tp_order_id'])
    bot.poll_positions()
    assert store.positions() == {}
    assert ex.algos[pos['sl_algo_id']]['algoStatus'] == 'CANCELED'
    t = store.trades()[0]
    assert t['reason'] == 'TP' and t['net'] > 0 and t['net_r'] > 1
    assert any('청산(TP)' in m for m in msgs)


def test_stop_trigger_cancels_take_profit():
    bot, ex, store, _, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    clock.ms += DAY_MS
    ex.trigger(pos['sl_algo_id'], pos['sl'])
    bot.poll_positions()
    assert ex.orders[pos['tp_order_id']]['status'] == 'CANCELED'
    t = store.trades()[0]
    assert t['reason'] == 'SL' and -1.2 < t['net_r'] < -0.9


def test_trend_exit_closes_at_market_on_day_close():
    bot, ex, store, _, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    clock.ms += 3 * DAY_MS
    closes = _flat() + [112.0, 111.0, 95.0]           # 10일 신저가 이탈
    ex.daily['ETHUSDT'] = _daily(closes, [1000.0] * len(closes), DAY0 + 3 * DAY_MS)
    ex.book['ETHUSDT'] = (94.99, 95.01)
    rep = bot.on_day_close()
    assert rep['exits'] == ['ETHUSDT'] and store.positions() == {}
    assert store.trades()[0]['reason'] == 'CROSS'
    assert ex.algos[pos['sl_algo_id']]['algoStatus'] == 'CANCELED'


def test_soft_stop_closes_when_exchange_stop_missing():
    bot, ex, store, msgs, _ = _bot()
    _, pos = _open_long_position(bot, ex, store)
    ex.algos[pos['sl_algo_id']]['algoStatus'] = 'CANCELED'      # 거래소 손절이 사라짐
    ex.mark['ETHUSDT'] = pos['sl'] - 1
    ex.book['ETHUSDT'] = (pos['sl'] - 1.01, pos['sl'] - 0.99)
    bot.poll_positions()                    # 1회차: 기다리며 손절 재등록
    assert store.positions()
    bot.poll_positions()                    # 2회차: 직접 청산
    assert store.positions() == {}
    assert store.trades()[0]['reason'] == 'SOFT_SL'
    assert any('시장가로 청산' in m for m in msgs)


def test_missing_protection_is_replaced():
    bot, ex, store, _, _ = _bot()
    _, pos = _open_long_position(bot, ex, store)
    ex.algos[pos['sl_algo_id']]['algoStatus'] = 'CANCELED'
    ex.orders[pos['tp_order_id']]['status'] = 'CANCELED'
    bot.ensure_protection()
    new = store.positions()['ETHUSDT']
    assert new['sl_algo_id'] != pos['sl_algo_id'] and new['tp_order_id'] != pos['tp_order_id']


def _settle(bot, eq):
    """같은 자산이 두 번 연속 보이면 판정한다 (급변 직후 한 번은 이체 반영을 기다린다)."""
    bot.update_drawdown(eq)
    return bot.update_drawdown(eq)


def _poll(bot, clock, sec=30):
    clock.ms += sec * 1000
    bot.poll()


def test_drawdown_halts_new_entries_until_resume():
    bot, ex, _, msgs, _ = _bot()
    assert _settle(bot, 10_000) == 1.0
    assert bot.update_drawdown(8_400) == 1.0           # 30초 새 16% 급변 → 이체일 수 있어 한 번 미룬다
    assert bot.update_drawdown(8_400) == 0.5           # 그대로면 16% 낙폭 → 위험 절반
    assert _settle(bot, 7_900) == 0.0                  # 21% → 정지
    assert _settle(bot, 9_900) == 0.0                  # 회복해도 수동 재개 전까지 정지
    ex.equity = 7_900
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert rep['entries'] == [] and any('낙폭' in why for _, _, why in rep['skipped'])
    assert any('신규 진입 정지' in m for m in msgs)


def test_regime_without_route_places_nothing():
    bot, ex, store, _, _ = _bot(regime='MICRO_RANGING')
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert rep['entries'] == [] and store.entries() == {} and ex.orders == {}


def test_short_signal_in_downtrend():
    bot, ex, store, _, _ = _bot(regime='TRENDING_DOWN')
    _with_breakout(ex, up=False)
    rep = bot.on_day_close()
    e = rep['entries'][0]
    assert e['strategy'] == 'SX' and e['side'] == -1
    assert ex.orders[store.entries()['ETHUSDT']['order_id']]['side'] == 'SELL'


def test_observe_mode_never_orders():
    bot, ex, store, msgs, _ = _bot(mode='observe')
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert len(rep['entries']) == 1 and ex.orders == {} and store.entries() == {}
    bot.poll()
    assert ex.orders == {}
    assert '[observe]' in msgs[-1]


def test_min_notional_skips_tiny_orders():
    bot, ex, _, _, _ = _bot()
    ex.equity = 30.0
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert rep['entries'] == [] and any('최소 주문' in why for _, _, why in rep['skipped'])


def test_reconcile_finalizes_missing_and_flags_unknown():
    bot, ex, store, msgs, _ = _bot()
    _, pos = _open_long_position(bot, ex, store)
    ex.fill(pos['tp_order_id'])                          # 봇이 꺼진 사이 익절
    ex.positions['BTCUSDT'] = (0.01, 60000.0)            # 사람이 연 포지션
    bot.reconcile()
    assert store.positions() == {} and store.trades()[0]['reason'] == 'TP'
    assert any('봇이 모르는 포지션' in m for m in msgs)


def test_startup_switches_to_one_way_mode():
    bot, ex, _, msgs, _ = _bot()
    ex.hedge = True
    bot.startup()
    assert ex.hedge is False and any('단방향' in m for m in msgs)


def test_day_due_window():
    bot, _, store, _, clock = _bot(clock_ms=DAY0 + 30_000)
    assert bot.day_due() is False                        # 마감 직후 대기
    clock.ms = DAY0 + 90_000
    assert bot.day_due() is True
    store.set_state('last_day_processed', DAY0)
    assert bot.day_due() is False
    clock.ms = DAY0 + DAY_MS + 10 * 3600_000            # 10시간 늦게 켜져도 그날 판정은 한다(추세 이탈 청산)
    assert bot.day_due() is True


def test_deposit_at_peak_is_not_counted_twice():
    """고점에서 입금하면 잔고가 먼저 오르고 이체 기록이 뒤따를 수 있다. 두 번 세면 고점이
    20,000으로 부풀어 25% 가짜 낙폭 → 신규 진입 정지가 된다 (예전 구조의 결함)."""
    for late_record in (False, True):
        bot, ex, store, msgs, clock = _bot()
        _poll(bot, clock)
        _poll(bot, clock)                                # 기준 고점 10,000
        assert float(store.get_state('peak_equity')) == 10_000
        ex.equity = 15_000
        row = {'incomeType': 'TRANSFER', 'income': '5000', 'asset': 'USDT', 'time': clock.ms + 1}
        if not late_record:
            ex.income_rows.append(row)
        _poll(bot, clock)
        if late_record:
            ex.income_rows.append(row)                   # 기록이 한 점검 늦게 보인다
        _poll(bot, clock)
        _poll(bot, clock)
        assert float(store.get_state('peak_equity')) == 15_000, late_record
        assert store.get_state('dd_halt', '0') == '0' and bot.update_drawdown(15_000) == 1.0
        assert any('입금' in m for m in msgs)


def test_withdrawal_is_not_mistaken_for_loss():
    """40% 출금을 낙폭으로 보면 곧바로 정지(20%)가 걸리고, 정지는 --resume 전까지 안 풀린다."""
    for late_record in (False, True):
        bot, ex, store, msgs, clock = _bot()
        _poll(bot, clock)
        _poll(bot, clock)
        ex.equity = 6_000
        row = {'incomeType': 'TRANSFER', 'income': '-4000', 'asset': 'USDT', 'time': clock.ms + 1}
        if not late_record:
            ex.income_rows.append(row)
        _poll(bot, clock)
        if late_record:
            ex.income_rows.append(row)
        _poll(bot, clock)
        assert float(store.get_state('peak_equity')) == 6_000, late_record
        assert store.get_state('dd_halt', '0') == '0' and store.get_state('dd_tier', '0') == '0'
        assert not any('신규 진입 정지' in m for m in msgs)


def test_transfer_lookup_failure_still_updates_drawdown():
    """이체 기록 조회가 실패해도 낙폭 감시는 멈추면 안 된다."""
    bot, ex, store, _, clock = _bot()
    _poll(bot, clock)
    _poll(bot, clock)

    def broken(*a, **k):
        raise BinanceAPIError(500, -1000, 'boom', '/fapi/v1/income')

    ex.income = broken
    ex.equity = 9_700
    _poll(bot, clock)
    assert float(store.get_state('last_equity')) == 9_700


def test_live_mode_can_share_spot_key_but_testnet_cannot():
    live_env = {'FUT_MODE': 'live', 'FUT_LIVE_CONFIRM': cfg.FUT_LIVE_CONFIRM_PHRASE,
                'BINANCE_API_KEY': 'spotkey', 'BINANCE_API_SECRET': 'spotsecret'}
    s = fm.load_settings(live_env)
    assert (s.api_key, s.api_secret) == ('spotkey', 'spotsecret') and 'BINANCE_API_KEY' in s.key_source
    own = fm.load_settings({**live_env, 'FUT_API_KEY': 'fk', 'FUT_API_SECRET': 'fs'})
    assert own.api_key == 'fk' and own.key_source == 'FUT_API_KEY'
    for env in ({**live_env, 'FUT_API_KEY': 'fk'},                       # 반쪽 설정은 섞지 않는다
                {'FUT_MODE': 'testnet', 'BINANCE_API_KEY': 'spotkey', 'BINANCE_API_SECRET': 'spotsecret'},
                {'FUT_MODE': 'testnet', 'FUT_API_KEY': 'spotkey', 'FUT_API_SECRET': 'spotsecret',
                 'BINANCE_API_KEY': 'spotkey'}):
        try:
            fm.load_settings(env)
            raise AssertionError(f'should refuse {env}')
        except SystemExit:
            pass


def test_transfer_rebases_peak_after_first_run():
    bot, ex, store, msgs, _ = _bot()
    _settle(bot, 10_000)
    ex.income_rows = [{'incomeType': 'TRANSFER', 'income': '500', 'asset': 'USDT', 'time': DAY0}]
    bot.rebase_peak_on_transfers()                       # 첫 실행: 기준만 잡는다
    assert float(store.get_state('peak_equity')) == 10_000
    ex.income_rows = [{'incomeType': 'TRANSFER', 'income': '-2000', 'asset': 'USDT', 'time': DAY0 + 200_000}]
    bot.rebase_peak_on_transfers()
    assert float(store.get_state('peak_equity')) == 8_000
    assert any('출금' in m for m in msgs)


def test_load_settings_guards():
    s = fm.load_settings({})
    assert s.mode == 'observe' and not s.trading and s.base_url == cfg.FUT_LIVE_BASE_URL
    for env in ({'FUT_MODE': 'live', 'FUT_API_KEY': 'k', 'FUT_API_SECRET': 's'},
                {'FUT_MODE': 'testnet'},
                {'FUT_MODE': 'nope'},
                {'FUT_MODE': 'testnet', 'FUT_API_KEY': 'k', 'FUT_API_SECRET': 's',
                 'FUT_BASE_URL': 'https://fapi.binance.com'}):
        try:
            fm.load_settings(env)
            raise AssertionError(f'should refuse {env}')
        except SystemExit:
            pass
    t = fm.load_settings({'FUT_MODE': 'testnet', 'FUT_API_KEY': 'k', 'FUT_API_SECRET': 's'})
    assert t.base_url == cfg.FUT_TESTNET_BASE_URL and t.trading
    live = fm.load_settings({'FUT_MODE': 'live', 'FUT_API_KEY': 'k', 'FUT_API_SECRET': 's',
                             'FUT_LIVE_CONFIRM': cfg.FUT_LIVE_CONFIRM_PHRASE})
    assert live.base_url == cfg.FUT_LIVE_BASE_URL


def test_env_file_parser_handles_comments():
    import shutil
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    try:
        p = d / '.env'
        p.write_text('# 주석\nFUT_MODE=testnet   # 데모\nFUT_API_KEY="abc"\nBROKEN\n', encoding='utf-8')
        env = fm._read_env_file(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert env == {'FUT_MODE': 'testnet', 'FUT_API_KEY': 'abc'}
    assert fm._read_env_file(Path('/nonexistent/.env')) == {}


def test_check_report_lists_regime_and_small_symbols():
    bot, ex, _, _, _ = _bot()
    ex.equity = 500.0
    ex.rules = lambda s: replace(RULES, symbol=s, min_notional=Decimal('50') if s == 'BTCUSDT' else Decimal('5'))
    ex.mark.update(dict.fromkeys(cfg.FUT_UNIVERSE, 100.0))
    ex.mark['BTCUSDT'] = 65_000.0
    del ex.mark['AAVEUSDT']                                  # 시세가 빠진 종목이 있어도 0으로 나누지 않는다
    lines = fm.check_report(bot)
    assert any('장세' in ln for ln in lines)
    small = next(ln for ln in lines if '낼 수 없는 종목' in ln)
    assert 'BTCUSDT(자산 $1,30' in small and 'ETHUSDT' not in small and 'AAVEUSDT(시세 없음)' in small
    # 시세가 빠진 종목이 있으면 '전 종목' 값을 단정하지 않는다 (빠진 게 BTC·ETH면 $1,300이 $400처럼 보인다)
    full = next(ln for ln in lines if ln.startswith('전 종목이 열리는 자산'))
    assert '알 수 없음' in full and 'AAVEUSDT' in full and '19종목 기준 $1,30' in full
    ex.mark['AAVEUSDT'] = 100.0
    full = next(ln for ln in fm.check_report(bot) if ln.startswith('전 종목이 열리는 자산'))
    assert full.startswith('전 종목이 열리는 자산: $1,30') and 'BTCUSDT' in full
    del ex.mark['BTCUSDT']
    full = next(ln for ln in fm.check_report(bot) if ln.startswith('전 종목이 열리는 자산'))
    assert '알 수 없음' in full and 'BTCUSDT' in full and '$1,30' not in full
    ex.mark.clear()
    assert any(ln == '전 종목이 열리는 자산: 알 수 없음 (시세를 받지 못함)' for ln in fm.check_report(bot))


def test_check_uses_the_real_order_sizing_path():
    """--check 는 실제 주문 크기 계산(_size)과 같은 경로를 쓴다 (2026-11-17 재검증에서 찾은 오류).
    BTC는 최소 금액($50)보다 최소 수량(0.001개 ≈ $65)이 실제 하한이고, ETH는 금액이 딱 $20이어도
    수량을 내림하면($20 → 0.010개 = $18.2) 막힌다. 예전 표시는 둘 다 '가능'이었다."""
    bot, ex, _, _, _ = _bot(mode='observe')
    real = {'BTCUSDT': SymbolRules('BTCUSDT', Decimal('0.1'), Decimal('0.001'), Decimal('0.001'), Decimal('50'),
                                   Decimal('0.001'), Decimal('0.001')),
            'ETHUSDT': SymbolRules('ETHUSDT', Decimal('0.01'), Decimal('0.001'), Decimal('0.001'), Decimal('20'),
                                   Decimal('0.001'), Decimal('0.001'))}
    ex.rules = lambda s: real.get(s, replace(RULES, symbol=s))
    ex.mark.update({'BTCUSDT': 65_123.35, 'ETHUSDT': 1_820.48})
    marks = bot.marks()
    assert 'BTCUSDT' in fm.blocked_symbols(bot, 1_000.0, marks)         # 명목가 $50 = 최소 금액인데 0.001개 미만
    # 보류 사유도 실제 하한을 말해야 한다 — 최소 금액 $50이 아니라 최소 수량 0.001개 × 시세 ≈ $65
    why = bot._size('BTCUSDT', fm._probe_signal(65_123.35), 65_123.35, 1_000.0, 1.0, 0.0)[2]
    assert '필요 약 $65' in why and '필요 $50' not in why          # (뒤의 '계산 $50.00'은 위험 기준 금액)
    assert 'ETHUSDT' in fm.blocked_symbols(bot, 400.0, marks)           # 명목가 $20 → 0.010개 = $18.2
    assert 'ETHUSDT' not in fm.blocked_symbols(bot, 410.0, marks)
    btc = fm.min_equity_for(bot, 'BTCUSDT', 65_123.35)
    assert 1_302 < btc < 1_305                                           # 0.001개 × 시세 × (손절폭 10% ÷ 위험 0.5%)
    assert 'BTCUSDT' not in fm.blocked_symbols(bot, btc, marks)
    assert 'BTCUSDT' in fm.blocked_symbols(bot, btc * 0.999, marks)
    assert abs(fm.min_equity_for(bot, 'BTCUSDT', 65_123.35, dds=0.5) - 2 * btc) < 6   # 낙폭 15%(위험 절반)면 두 배
    # 금액이 최소 금액과 **정확히 같으면** 허용한다(바이낸스 기준도 '이상'): ETH 2,000 × 0.010개 = $20
    assert 'ETHUSDT' not in fm.blocked_symbols(bot, 400.0, {**marks, 'ETHUSDT': (2_000.0, 0.0)})


def test_store_roundtrip():
    st = fm.Store(':memory:')
    st.put_position('A', {'x': 1})
    st.put_entry('B', {'y': 2})
    st.set_state('k', 3)
    assert st.positions() == {'A': {'x': 1}} and st.entries() == {'B': {'y': 2}} and st.get_state('k') == '3'
    st.del_position('A')
    st.del_entry('B')
    st.add_trade({'symbol': 'A', 'net': 1.0, 'exit_ts': 5})
    assert st.positions() == {} and st.entries() == {} and st.trades(0)[0]['net'] == 1.0
    assert st.get_state('missing', 'd') == 'd'


def test_telegram_notifier_without_token_only_logs():
    n = fm.TelegramNotifier('', '')
    n('hello')
    assert n.q.empty()


# ══════════════════════════════════════════════════════════════
#  2026-11-17 외부 검증에서 나온 결함 — 회귀 테스트 (번호는 검증 보고서 기준)
# ══════════════════════════════════════════════════════════════

def _restart(bot, ex, store, msgs, clock):
    """같은 DB·거래소로 프로세스를 새로 띄운다."""
    bot2 = fm.FuturesBot(ex, store, msgs.append, mode='testnet', clock=clock, observe_equity=10_000)
    bot2.current_regime = bot.current_regime
    bot2.startup()
    return bot2


def test_r1_crash_after_entry_order_is_recovered_on_restart():
    """#1 주문은 거래소에 들어갔는데 기록 전에 죽음 → 재시작 후 그 주문을 찾아 이어 간다.
    예전: 재시작 후 '봇이 아는 진입 []', 고아 주문이 체결되면 손절 없는 '모르는 포지션'."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.crash_after_send = True
    try:
        bot.on_day_close()
        raise AssertionError('주문 직후 죽어야 한다')
    except RuntimeError:
        pass
    e = store.entries()['ETHUSDT']                        # 선기록 덕분에 남아 있다
    assert e['order_id'] is None and e['client_id'].startswith('atlE')
    bot2 = _restart(bot, ex, store, msgs, clock)
    (oid, o), = ex.orders.items()
    assert o['status'] == 'NEW'                           # 기록과 이어진 주문은 대조가 취소하지 않는다
    ex.fill(oid)
    clock.ms += 30_000
    bot2.poll()
    pos = store.positions()['ETHUSDT']
    assert ex.algos[pos['sl_algo_id']]['algoStatus'] == 'NEW' and ex.orders[pos['tp_order_id']]['status'] == 'NEW'
    assert store.entries() == {} and not any('모르는 포지션' in m for m in msgs)


def test_r1_untracked_bot_order_is_cancelled_but_manual_order_is_kept():
    """#1 기록이 없는 봇 주문(atl 접두사)은 재시작 때 취소, 사람이 낸 주문은 그대로."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    oid = store.entries()['ETHUSDT']['order_id']
    store.del_entry('ETHUSDT')                            # 예전 구조: 주문 후 기록 전에 죽어 기록이 없음
    ex.book['BTCUSDT'] = (60000.0, 60001.0)
    manual = ex.new_order('BTCUSDT', 'BUY', 'LIMIT', Decimal('0.001'), price=Decimal('50000'),
                          time_in_force='GTC', client_id='web_manual_1')['orderId']
    _restart(bot, ex, store, msgs, clock)
    assert ex.orders[oid]['status'] == 'CANCELED' and ex.orders[manual]['status'] == 'NEW'
    assert any('기록에 없던 봇 주문 1건' in m for m in msgs)


def test_r1_crash_before_send_drops_entry():
    """#1 선기록만 하고 주문은 못 나간 경우 → 거래소에 없음(-2013) → 진입 정리."""
    bot, ex, store, msgs, clock = _bot()
    store.put_entry('ETHUSDT', {'symbol': 'ETHUSDT', 'strategy': 'S6', 'side': 1, 'qty': '1', 'sl': 100.0,
                                'sl_dist': 5.0, 'rr': 2.0, 'ref': 105.0, 'order_id': None,
                                'client_id': 'atlEETHUSDT123', 'status': 'PENDING_NEW',
                                'expire_ms': clock.ms + 3600_000})
    bot.poll_entries()
    assert store.entries() == {} and any('보내기 전에 멈춰' in m for m in msgs)


def test_r1_crash_after_market_fallback_keeps_position_protected():
    """#1 시장가 전환 주문이 체결된 직후 죽음 → 재시작 후 포지션 기록·보호주문까지 마무리."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    ex.crash_after_send = True                            # 이번엔 시장가 주문 직후
    try:
        bot.poll_entries()
        raise AssertionError('시장가 주문 직후 죽어야 한다')
    except RuntimeError:
        pass
    assert store.entries()['ETHUSDT']['status'] == 'MARKET_SENT' and 'ETHUSDT' in ex.positions
    bot2 = _restart(bot, ex, store, msgs, clock)
    bot2.poll()
    pos = store.positions()['ETHUSDT']
    assert ex.algos[pos['sl_algo_id']]['algoStatus'] == 'NEW' and store.entries() == {}
    assert float(pos['qty']) == ex.positions['ETHUSDT'][0]


def test_r1_orphan_stop_is_adopted_not_duplicated():
    """#1 손절을 낸 직후 기록 전에 죽음 → 재시작 때 그 손절을 입양(중복 손절을 만들지 않는다)."""
    bot, ex, store, msgs, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    p = store.positions()['ETHUSDT']
    aid = p['sl_algo_id']
    p['sl_algo_id'] = None
    store.put_position('ETHUSDT', p)
    _restart(bot, ex, store, msgs, clock)
    assert store.positions()['ETHUSDT']['sl_algo_id'] == aid
    assert sum(1 for a in ex.algos.values() if a['algoStatus'] == 'NEW') == 1


def test_r2_repeg_after_crash_below_stop_is_abandoned():
    """#2 메이커 거부 뒤 가격이 손절가 아래로 급락(112 → 98) → 다시 걸지 않는다.
    예전: 손절 거리를 그대로 들고 97.97에 다시 걸고 손절을 92.5까지 끌어내렸다."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.reject_next_gtx = True
    bot.on_day_close()
    ex.book['ETHUSDT'] = (97.97, 97.99)
    ex.mark['ETHUSDT'] = 98.0
    clock.ms += 30_000
    bot.poll_entries()
    assert store.entries() == {} and store.positions() == {}
    assert not [o for o in ex.orders.values() if o['status'] == 'NEW']
    assert any('재주문 취소' in m and 'invalid_sl' in m for m in msgs)


def test_r2_repeg_keeps_absolute_stop_resizes_and_never_chases():
    """#2 다시 걸 때 손절가는 신호가 정한 가격 그대로, 수량은 새 손절 거리로, 가격은 처음 지정가보다 불리하지 않게."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.reject_next_gtx = True
    bot.on_day_close()
    e0 = store.entries()['ETHUSDT']
    ex.book['ETHUSDT'] = (115.0, 115.02)                  # 반등 — 쫓아가면 안 된다
    ex.reject_next_gtx = True
    clock.ms += 30_000
    bot.poll_entries()
    e1 = store.entries()['ETHUSDT']
    assert Decimal(e1['price']) == Decimal(e0['limit0'])
    ex.book['ETHUSDT'] = (109.99, 110.01)                 # 하락했지만 아직 손절가 위
    clock.ms += 30_000
    bot.poll_entries()
    e2 = store.entries()['ETHUSDT']
    assert Decimal(e2['price']) < Decimal(e0['price'])
    assert abs(float(e2['price']) - e2['sl_dist'] - e0['sl']) < 1e-6        # 절대 손절가 유지
    assert float(e2['qty']) * e2['sl_dist'] <= 10_000 * cfg.FUT_RISK_PCT + 1e-6
    ex.fill(e2['order_id'])
    bot.poll_entries()
    assert abs(store.positions()['ETHUSDT']['sl'] - e0['sl']) < 1e-6


def test_r2_entry_cancelled_in_app_is_not_revived():
    """#2 운영자가 앱에서 진입 주문을 취소하면 되살리지 않는다 (예전: 30초마다 4시간 동안 부활)."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    ex.cancel_order('ETHUSDT', store.entries()['ETHUSDT']['order_id'])
    for _ in range(3):
        clock.ms += 30_000
        bot.poll_entries()
    assert store.entries() == {} and not [o for o in ex.orders.values() if o['status'] == 'NEW']
    assert sum('밖에서 취소' in m for m in msgs) == 1


def test_r2_repeg_gives_up_after_limit():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.reject_next_gtx = True
    bot.on_day_close()
    for _ in range(cfg.ENTRY_MAX_REPEG + 1):
        ex.reject_next_gtx = True
        clock.ms += 30_000
        bot.poll_entries()
    assert store.entries() == {} and any('번 거부돼' in m for m in msgs)
    assert sum(1 for o in ex.orders.values() if o['type'] == 'LIMIT') == 1 + cfg.ENTRY_MAX_REPEG   # 처음 1번 + 재주문 10번


def test_r3_daily_loss_limit_sees_losses_before_midnight():
    """#3 어제 23:00 -6% 손절 → 오늘 00:01 판정에서 막힌다 (예전: 오늘 0시 이후만 세서 +0.0%)."""
    bot, ex, store, msgs, clock = _bot()
    store.add_trade({'symbol': 'SOLUSDT', 'strategy': 'S6', 'side': 1, 'entry_ts': DAY0 - 5 * 3600_000,
                     'exit_ts': DAY0 - 3600_000, 'net': -600.0, 'risk_usd': 50.0, 'mode': 'testnet'})
    ex.equity = 9_400
    assert bot.day_loss_pct(9_400) <= cfg.FUT_DAILY_LOSS_LIMIT
    _with_breakout(ex)
    rep = bot.on_day_close()
    assert rep['entries'] == [] and any('일일 손실' in why for _, _, why in rep['skipped'])
    clock.ms += DAY_MS                                    # 24시간이 지나면 다시 진입
    assert bot.day_loss_pct(9_400) == 0.0


def test_r5_repeated_stop_failure_alerts_are_throttled():
    """#5 손절 주문이 계속 실패해도 알림은 30분에 한 번 (예전: 30초마다 → 하루 2,880건)."""
    bot, ex, store, msgs, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    ex.algos[pos['sl_algo_id']]['algoStatus'] = 'CANCELED'
    real = ex.new_stop_market

    def boom(*a, **k):
        raise BinanceAPIError(400, -2021, 'Order would immediately trigger.', '/fapi/v1/algoOrder')

    ex.new_stop_market = boom
    n0 = len(msgs)
    for _ in range(10):
        clock.ms += 30_000
        bot.poll()
    assert sum('손절 주문 실패' in m for m in msgs[n0:]) == 1
    clock.ms += cfg.ALERT_COOLDOWN_SEC * 1000
    bot.poll()
    fails = [m for m in msgs[n0:] if '손절 주문 실패' in m]
    assert len(fails) == 2 and '번 더 반복' in fails[-1]
    ex.new_stop_market = real
    clock.ms += 30_000
    bot.poll()
    assert any('손절 주문 복구' in m for m in msgs)


def test_r6_late_start_runs_trend_exit_but_skips_entries():
    """#6 자정 후 10시간 늦게 켜짐 → 추세 이탈 청산은 하고, 지정가 창이 지난 신규 진입은 하지 않는다.
    예전: 2시간이 지나면 그날 판정을 통째로 건너뛰어 청산까지 다음 자정으로 밀렸다."""
    bot, ex, store, msgs, clock = _bot()
    _open_long_position(bot, ex, store)
    clock.ms = DAY0 + 3 * DAY_MS + 10 * 3600_000
    closes = _flat() + [112.0, 111.0, 95.0]               # 10일 신저가 이탈
    ex.daily['ETHUSDT'] = _daily(closes, [1000.0] * len(closes), DAY0 + 3 * DAY_MS)
    ex.book['ETHUSDT'] = (94.99, 95.01)
    _with_breakout(ex, 'SOLUSDT')
    ex.daily['SOLUSDT'] = _daily(_flat() + [112.0], [1000.0] * 70 + [6000.0], DAY0 + 3 * DAY_MS)
    assert bot.day_due() is True
    rep = bot.on_day_close()
    assert rep['exits'] == ['ETHUSDT'] and store.positions() == {}
    assert rep['entries'] == [] and any('지정가 창' in why for _, _, why in rep['skipped'])
    assert bot.day_due() is False


def test_r9_partial_take_profit_resizes_stop():
    """#9 익절이 일부만 체결돼 포지션이 줄면 손절 수량도 줄인다 (새 손절을 먼저 걸고 옛것을 취소)."""
    bot, ex, store, msgs, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    old = pos['sl_algo_id']
    amt, entry = ex.positions['ETHUSDT']
    ex.positions['ETHUSDT'] = (amt / 2, entry)
    clock.ms += 30_000
    bot.poll()
    p = store.positions()['ETHUSDT']
    assert p['sl_algo_id'] != old and ex.algos[old]['algoStatus'] == 'CANCELED'
    assert Decimal(ex.algos[p['sl_algo_id']]['quantity']) == Decimal(p['qty']) == Decimal(str(amt / 2))


def test_r9_stop_resize_failure_keeps_old_stop_and_retries():
    bot, ex, store, msgs, clock = _bot()
    _, pos = _open_long_position(bot, ex, store)
    old = pos['sl_algo_id']
    amt, entry = ex.positions['ETHUSDT']
    ex.positions['ETHUSDT'] = (amt / 2, entry)
    real = ex.new_stop_market

    def boom(*a, **k):
        raise BinanceAPIError(500, -1001, 'Internal error', '/fapi/v1/algoOrder')

    ex.new_stop_market = boom
    clock.ms += 30_000
    bot.poll()
    assert store.positions()['ETHUSDT']['sl_algo_id'] == old and ex.algos[old]['algoStatus'] == 'NEW'
    ex.new_stop_market = real
    clock.ms += 30_000
    bot.poll()
    assert store.positions()['ETHUSDT']['sl_algo_id'] != old and ex.algos[old]['algoStatus'] == 'CANCELED'


# ── 주문 경로의 나머지 분기 (라이브 커버리지 게이트 대상) ────────────────

def _raise(err):
    def f(*a, **k):
        raise err
    return f


def test_regime_uses_full_btc_history_with_paging():
    """라이브 장세 판정은 BTC 일봉을 상장일부터 끝까지(1,500개씩 나눠) 받는다 — EMA200 시작점이 백테스트와 같아야 한다.
    두 번째부터는 최근 10개만 받아 이어 붙인다."""
    clock = Clock(DAY0 + 90_000)
    ex = FakeEx(clock)
    start = cfg.FUT_BTC_HISTORY_START_MS
    days = (DAY0 - start) // DAY_MS + 1
    rows = [[start + i * DAY_MS, 100.0, 101.0, 99.0, 100.0 + i * 0.01, 1000.0] for i in range(days)]
    calls = []

    def klines(symbol, interval, limit=500, start_ms=None):
        calls.append((interval, start_ms))
        if interval == '4h':
            return [[DAY0 - (60 - i) * core.H4_MS, 100.0, 101.0, 99.0, 100.0, 10.0] for i in range(61)]
        if start_ms is None:
            return rows[-limit:]
        return [r for r in rows if r[0] >= start_ms][:limit]

    ex.klines = klines
    bot = fm.FuturesBot(ex, fm.Store(':memory:'), [].append, mode='testnet', clock=clock)
    regime, _, _ = bot.current_regime()
    assert sum(1 for i, s in calls if i == '1d' and s is not None) >= 2
    assert len(bot._btc_d1) == days and bot.last_regime == regime
    rows.append([DAY0 + DAY_MS, 100.0, 101.0, 99.0, 120.0, 1000.0])
    clock.ms += DAY_MS
    bot.current_regime()
    assert len(bot._btc_d1) == days + 1 and bot._btc_d1[-1].ts == DAY0 + DAY_MS


def test_partial_fill_then_app_cancel_protects_only_the_filled_part():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    o = ex.orders[store.entries()['ETHUSDT']['order_id']]
    ex._fill(o, float(o['price']), round(float(o['origQty']) / 2, 3))
    o['status'] = 'CANCELED'                               # 나머지는 운영자가 앱에서 취소
    bot.poll_entries()
    pos = store.positions()['ETHUSDT']
    assert Decimal(pos['qty']) == Decimal(o['executedQty'])
    assert Decimal(ex.algos[pos['sl_algo_id']]['quantity']) == Decimal(pos['qty'])
    assert store.entries() == {} and any('이미 체결된' in m for m in msgs)


def test_market_fallback_rejected_cleans_up():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    real = ex.new_order

    def reject_market(symbol, side, order_type, quantity, **kw):
        if order_type == 'MARKET':
            raise BinanceAPIError(400, -2019, 'Margin is insufficient.', '/fapi/v1/order')
        return real(symbol, side, order_type, quantity, **kw)

    ex.new_order = reject_market
    bot.poll_entries()
    assert store.entries() == {} and store.positions() == {} and any('시장가 진입 실패' in m for m in msgs)


def test_market_fallback_rechecks_absolute_stop():
    """4시간 뒤 가격이 이미 신호 손절가 아래면 시장가로 사지 않는다."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    ex.book['ETHUSDT'] = (97.97, 97.99)
    bot.poll_entries()
    assert store.entries() == {} and store.positions() == {}
    assert any('시장가 전환 보류' in m and 'invalid_sl' in m for m in msgs)


def test_close_failure_alert_is_throttled():
    bot, ex, store, msgs, clock = _bot()
    _open_long_position(bot, ex, store)
    real = ex.new_order

    def reject_close(symbol, side, order_type, quantity, **kw):
        if order_type == 'MARKET' and kw.get('reduce_only'):
            raise BinanceAPIError(400, -2022, 'ReduceOnly Order is rejected.', '/fapi/v1/order')
        return real(symbol, side, order_type, quantity, **kw)

    ex.new_order = reject_close
    for _ in range(3):
        bot.close_position('ETHUSDT', store.positions()['ETHUSDT'], 'CROSS')
    assert 'ETHUSDT' in store.positions() and sum('청산 주문 실패' in m for m in msgs) == 1


def test_reconcile_adopts_orphan_take_profit_and_cancels_stray_stop():
    bot, ex, store, msgs, clock = _bot()
    _open_long_position(bot, ex, store)
    p = store.positions()['ETHUSDT']
    tp = p['tp_order_id']
    p['tp_order_id'] = None                                # 익절을 낸 직후 기록 전에 죽음
    store.put_position('ETHUSDT', p)
    stray = ex.new_stop_market('SOLUSDT', 'SELL', Decimal('1'), Decimal('90'),
                               client_algo_id='atlSSOLUSDT1')['algoId']      # 포지션 없는 종목의 봇 손절
    bot.reconcile()
    assert store.positions()['ETHUSDT']['tp_order_id'] == tp
    assert ex.algos[stray]['algoStatus'] == 'CANCELED'
    assert sum(1 for o in ex.orders.values() if o['status'] == 'NEW') == 1   # 익절을 중복으로 걸지 않는다


def test_entry_rejected_by_exchange_leaves_no_record():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.new_order = _raise(BinanceAPIError(400, -4164, "Order's notional must be no smaller than 5", '/fapi/v1/order'))
    rep = bot.on_day_close()
    assert rep['entries'] == [] and store.entries() == {}          # 선기록도 지운다
    assert any('주문 거부' in why for _, _, why in rep['skipped'])


def test_repeg_order_failure_keeps_entry_for_retry():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.reject_next_gtx = True
    bot.on_day_close()
    first = store.entries()['ETHUSDT']['order_id']
    real = ex.new_order
    ex.new_order = _raise(BinanceAPIError(500, -1001, 'Internal error', '/fapi/v1/order'))
    clock.ms += 30_000
    bot.poll_entries()
    e = store.entries()['ETHUSDT']
    assert e['order_id'] == first and e['repegs'] == 1             # 다음 점검에서 다시
    ex.new_order = real
    clock.ms += 30_000
    bot.poll_entries()
    assert ex.orders[store.entries()['ETHUSDT']['order_id']]['status'] == 'NEW'


def test_startup_warns_multi_assets_and_keeps_going_on_margin_errors():
    bot, ex, store, msgs, clock = _bot()
    ex.get_multi_assets = lambda: True
    ex.set_margin_type = _raise(BinanceAPIError(400, -4168, 'Unable to adjust to isolated-margin mode', '/fapi/v1/marginType'))
    bot.startup()
    assert any('멀티에셋' in m for m in msgs)


def test_startup_in_observe_mode_does_not_touch_account():
    bot, ex, store, msgs, clock = _bot(mode='observe')
    ex.hedge = True
    bot.startup()
    assert ex.hedge is True


def test_crowded_funding_blocks_long():
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    ex.funding['ETHUSDT'] = 0.0006
    rep = bot.on_day_close()
    assert rep['entries'] == [] and any('펀딩비' in why for _, _, why in rep['skipped'])


def test_number_parsers_fall_back():
    assert fm._f('x', 7.0) == 7.0 and fm._f(None) == 0.0
    assert fm._dec('') is None and fm._dec(None) is None and fm._dec('abc') is None
    assert fm._dec('1.50') == Decimal('1.50')


def test_at_most_six_positions_live():
    """동시 보유는 6개까지. 숫자를 박아 둔다 — 6→7로 바꿔도 테스트가 통과하던 빈틈(2026-11-17 변이 시험)."""
    for held, want in ((5, True), (6, False)):
        bot, ex, store, msgs, clock = _bot()
        for i in range(held):
            store.put_position(f'P{i}USDT', {'strategy': 'S6', 'side': 1, 'qty': '1', 'entry': 100.0, 'sl': 90.0,
                                             'sl_dist': 10.0, 'rr': 2.0, 'tp': 120.0, 'sl_algo_id': None,
                                             'tp_order_id': None})
        _with_breakout(ex)
        rep = bot.on_day_close()
        assert (len(rep['entries']) == 1) is want, held
        if not want:
            assert any('6개 가득' in why for _, _, why in rep['skipped'])


def test_market_fallback_with_no_fill_is_cleaned_up():
    """시장가 전환 주문이 한 개도 안 채워지고 끝나면(EXPIRED, 호가 부족 등) 진입을 정리한다 — 0으로 나누지 않는다."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    real = ex.new_order

    def market_expires(symbol, side, order_type, quantity, **kw):
        if order_type != 'MARKET':
            return real(symbol, side, order_type, quantity, **kw)
        oid = ex._id()
        ex.orders[oid] = {'orderId': oid, 'symbol': symbol, 'side': side, 'type': 'MARKET', 'origQty': str(quantity),
                          'price': '0', 'status': 'EXPIRED', 'executedQty': '0', 'avgPrice': '0', 'reduceOnly': False,
                          'timeInForce': None, 'clientOrderId': kw.get('client_id')}
        return dict(ex.orders[oid])

    ex.new_order = market_expires
    bot.poll_entries()
    assert store.entries() == {} and store.positions() == {}


def test_market_fallback_rounding_to_zero_leaves_no_position():
    """시장가 수량은 시장가 단위(market_step)로 다시 내림한다. 그러다 0이 되면 진입을 접는다 — 0개짜리 포지션을 남기지 않는다."""
    bot, ex, store, msgs, clock = _bot()
    _with_breakout(ex)
    bot.on_day_close()
    ex.rules = lambda s: replace(RULES, symbol=s, market_step=Decimal('100'))
    clock.ms += cfg.ENTRY_LIMIT_SEC * 1000 + 1
    bot.poll_entries()
    assert store.entries() == {} and store.positions() == {}

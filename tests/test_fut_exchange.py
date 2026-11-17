"""atlas_fut_exchange — 서명·요청 형식·조건부 주문 경로·오류 처리."""

import json
import urllib.parse
from decimal import Decimal

import atlas_fut_exchange as fx


class _Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append((method, url, body, headers))
        status, data = self.responses.pop(0)
        if isinstance(data, Exception):
            raise data
        return status, data if isinstance(data, str) else json.dumps(data)


def _client(responses, key='k', secret='s'):
    t = _Transport(responses)
    c = fx.FuturesREST('https://demo-fapi.binance.com/', key, secret, transport=t,
                       clock=lambda: 1_700_000_000.0, sleep=lambda s: None)
    return c, t


def _params(call):
    method, url, body, _ = call
    q = urllib.parse.urlparse(url).query if method in ('GET', 'DELETE') else body.decode()
    return dict(urllib.parse.parse_qsl(q))


def test_signature_matches_binance_documentation_example():
    c = fx.FuturesREST('x', 'k', '2b5eb11e18796d12d88f13dc27dbbd02c2cc51ff7059765ed9821957d82bb4d9')
    q = 'symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1&price=9000&timeInForce=GTC&recvWindow=5000&timestamp=1591702613943'
    assert c.sign(q) == '3c661234138461fcc7a7d8746c6558c9842d4e10870d2ecbedf7777cad694af9'


def test_signed_get_puts_signature_last_and_sends_key_header():
    c, t = _client([(200, [])])
    c.position_risk()
    method, url, body, headers = t.calls[0]
    assert method == 'GET' and body is None and headers['X-MBX-APIKEY'] == 'k'
    assert url.startswith('https://demo-fapi.binance.com/fapi/v3/positionRisk?')
    query = urllib.parse.urlparse(url).query
    unsigned, sig = query.rsplit('&signature=', 1)
    assert c.sign(unsigned) == sig
    p = _params(t.calls[0])
    assert p['timestamp'] == '1700000000000' and p['recvWindow'] == '5000'


def test_new_order_post_body_is_form_encoded_without_exponents():
    c, t = _client([(200, {'orderId': 7, 'status': 'NEW'})])
    c.new_order('ETHUSDT', 'BUY', 'LIMIT', Decimal('0.00001'), price=Decimal('1800.50'),
                time_in_force='GTX', client_id='atlE1')
    p = _params(t.calls[0])
    assert t.calls[0][0] == 'POST' and t.calls[0][1].endswith('/fapi/v1/order')
    assert p['quantity'] == '0.00001' and p['price'] == '1800.5' and p['timeInForce'] == 'GTX'
    assert 'reduceOnly' not in p and p['newClientOrderId'] == 'atlE1'


def test_stop_loss_goes_to_algo_endpoint_with_trigger_price():
    c, t = _client([(200, {'algoId': 99, 'algoStatus': 'NEW'})])
    res = c.new_stop_market('BTCUSDT', 'SELL', Decimal('0.002'), Decimal('60000'), client_algo_id='x' * 50)
    method, url, _, _ = t.calls[0]
    p = _params(t.calls[0])
    assert method == 'POST' and url.endswith('/fapi/v1/algoOrder')
    assert p['algoType'] == 'CONDITIONAL' and p['type'] == 'STOP_MARKET'
    assert p['triggerPrice'] == '60000' and 'stopPrice' not in p        # stopPrice는 조용히 무시된다
    assert p['reduceOnly'] == 'true' and p['workingType'] == 'MARK_PRICE'
    assert len(p['clientAlgoId']) == 36
    assert fx.algo_id_of(res) == 99 and fx.algo_status(res) == 'NEW'


def test_algo_helpers_accept_both_response_shapes():
    c, _ = _client([(200, {'orders': [{'algoId': 1}]}), (200, [{'algoId': 2}])])
    assert c.open_algo_orders() == [{'algoId': 1}]
    assert c.open_algo_orders() == [{'algoId': 2}]
    assert fx.algo_status({'status': 'triggered'}) == 'TRIGGERED'
    assert fx.algo_id_of({}) is None


def test_error_is_parsed_and_orders_are_not_retried():
    c, t = _client([(400, {'code': -2019, 'msg': 'Margin is insufficient.'})])
    try:
        c.new_order('BTCUSDT', 'BUY', 'MARKET', Decimal('1'))
        raise AssertionError('should raise')
    except fx.BinanceAPIError as e:
        assert e.code == -2019 and e.status == 400
    assert len(t.calls) == 1


def test_get_retries_server_errors_then_succeeds():
    c, t = _client([(503, 'busy'), (200, {'serverTime': 1_700_000_000_500})])
    assert c.sync_time() == 500
    assert len(t.calls) == 2


def test_network_error_retries_for_reads():
    c, t = _client([(0, OSError('timeout')), (200, [])])
    assert c.position_risk() == []
    assert len(t.calls) == 2


def test_margin_type_no_change_is_ignored():
    c, _ = _client([(400, {'code': -4046, 'msg': 'No need to change margin type.'})])
    assert c.set_margin_type('BTCUSDT', 'ISOLATED')['code'] == -4046


def test_signed_call_without_keys_fails_fast():
    c, t = _client([], key='', secret='')
    try:
        c.account()
        raise AssertionError('should raise')
    except fx.BinanceAPIError as e:
        assert 'API' in e.msg
    assert t.calls == []


def test_non_json_success_is_an_error():
    c, _ = _client([(200, '<html>')])
    try:
        c.book_ticker('BTCUSDT')
        raise AssertionError('should raise')
    except fx.BinanceAPIError as e:
        assert 'JSON' in e.msg


def test_rounding_uses_decimal_steps():
    assert fx.round_to(0.018, Decimal('0.001')) == Decimal('0.018')   # 0.018*1000 부동소수 함정
    assert fx.round_to(1.23456, Decimal('0.01')) == Decimal('1.23')
    assert fx.round_to(1.23456, Decimal('0.01'), up=True) == Decimal('1.24')
    assert fx.fmt_dec(Decimal('1E-5')) == '0.00001'
    assert fx.fmt_dec(Decimal('100')) == '100'
    assert fx.round_to(5.5, Decimal('0')) == Decimal('5.5')


def test_parse_rules_reads_filters():
    info = {'symbols': [{'symbol': 'BTCUSDT', 'filters': [
        {'filterType': 'PRICE_FILTER', 'tickSize': '0.10'},
        {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'},
        {'filterType': 'MARKET_LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'},
        {'filterType': 'MIN_NOTIONAL', 'notional': '50'}]}]}
    r = fx.parse_rules(info)['BTCUSDT']
    assert (r.tick, r.step, r.min_notional) == (Decimal('0.10'), Decimal('0.001'), Decimal('50'))
    c, _ = _client([(200, info)])
    assert c.rules('BTCUSDT').min_qty == Decimal('0.001')


def test_endpoints_used_for_account_and_algo_management():
    ok = (200, {})
    c, t = _client([ok, (200, {'dualSidePosition': False}), ok, (200, {'multiAssetsMargin': False}), ok,
                    ok, ok, ok, ok, (200, []), (200, []), (200, []), ok, ok, ok])
    c.account()
    assert c.get_position_mode() is False
    c.set_position_mode(False)
    assert c.get_multi_assets() is False
    c.set_leverage('BTCUSDT', 3)
    c.query_order('BTCUSDT', 1)
    c.cancel_order('BTCUSDT', 1)
    c.query_algo_order(5)
    c.cancel_algo_order(5)
    c.open_orders()
    c.income(income_type='FUNDING_FEE')
    c.user_trades('BTCUSDT')
    c.klines('BTCUSDT', '1d', 10)
    c.premium_index()
    c.book_ticker('BTCUSDT')
    paths = [(m, urllib.parse.urlparse(u).path) for m, u, _, _ in t.calls]
    assert ('DELETE', '/fapi/v1/algoOrder') in paths and ('GET', '/fapi/v1/algoOrder') in paths
    # 조건부 주문 취소는 공식 문서상 algoId/clientAlgoId만 받는다 — symbol을 붙이지 않는다
    cancel = [(u, b) for m, u, b, _ in t.calls if m == 'DELETE' and '/fapi/v1/algoOrder' in u]
    assert cancel and all('symbol=' not in u and b'symbol=' not in (b or b'') for u, b in cancel)
    assert ('GET', '/fapi/v3/account') in paths and ('POST', '/fapi/v1/positionSide/dual') in paths
    assert ('GET', '/fapi/v1/premiumIndex') in paths and ('GET', '/fapi/v1/ticker/bookTicker') in paths

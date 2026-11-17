"""
ATLAS Futures — 바이낸스 USDⓈ-M REST 클라이언트 (표준 라이브러리만 사용)
========================================================================
ccxt를 쓰지 않는 이유:
  2025-12-09부터 STOP_MARKET 같은 조건부 주문은 /fapi/v1/algoOrder 로만 받는다.
  예전 경로(/fapi/v1/order)로 보내면 -4120 으로 거부된다. 2026-05 기준 ccxt의
  create_order()는 아직 예전 경로로 보낸다는 보고가 있어, 손절처럼 실계좌 안전을
  좌우하는 주문은 엔드포인트를 직접 지정해서 보낸다.

주의할 함정 (공식 변경 로그·이관 가이드 기준):
  · 조건부 주문 가격 파라미터는 triggerPrice (stopPrice 아님 — 넣어도 오류 없이 무시됨)
  · 응답 ID는 orderId 가 아니라 algoId, 클라이언트 ID는 clientAlgoId (36자 이하)
  · 미체결 조회도 두 갈래: 일반 주문 /fapi/v1/openOrders, 조건부 /fapi/v1/openAlgoOrders
  · 심볼당 조건부 주문 10개 한도(-4045)
  · 가격·수량은 exchangeInfo 의 tickSize·stepSize 로 맞춘다 (-1111 방지)

테스트넷(데모): https://demo-fapi.binance.com — 키는 demo.binance.com 에서 발급한다.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Callable, Optional

# (method, url, body, headers, timeout) -> (http_status, response_text)
Transport = Callable[[str, str, Optional[bytes], dict, float], tuple[int, str]]

log = logging.getLogger('atlas_fut')


class BinanceAPIError(Exception):
    def __init__(self, status: int, code: Optional[int], msg: str, path: str):
        super().__init__(f'HTTP {status} {path} code={code} msg={msg}')
        self.status = status
        self.code = code
        self.msg = msg
        self.path = path


def urllib_transport(method: str, url: str, body: Optional[bytes], headers: dict,
                     timeout: float) -> tuple[int, str]:  # pragma: no cover - 실네트워크
    h = dict(headers)
    if body is not None:
        h['Content-Type'] = 'application/x-www-form-urlencoded'
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 - 설정된 바이낸스 https 엔드포인트
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:      # 4xx/5xx는 예외가 아니라 상태코드로 넘긴다 → 호출측이 판정
        log.debug(f'HTTP {e.code} {method} {url.split("?")[0]}')
        return e.code, e.read().decode()


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    tick: Decimal
    step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    market_step: Decimal
    market_min_qty: Decimal


def _dec(x: Any) -> Decimal:
    return Decimal(str(x))


def round_to(value: float, step: Decimal, up: bool = False) -> Decimal:
    """step 배수로 내림(기본) 또는 올림. 부동소수 곱셈 대신 Decimal로 계산한다."""
    if step <= 0:
        return _dec(value)
    q = (_dec(value) / step).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return (q * step).normalize()


def fmt_dec(d: Decimal) -> str:
    """지수 표기 없는 문자열 (바이낸스는 1E-5 형식을 거부한다)."""
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s or '0'


def parse_rules(info: dict) -> dict[str, SymbolRules]:
    out: dict[str, SymbolRules] = {}
    for s in info.get('symbols', []):
        f = {x.get('filterType'): x for x in s.get('filters', [])}
        lot = f.get('LOT_SIZE', {})
        mlot = f.get('MARKET_LOT_SIZE', lot)
        mn = f.get('MIN_NOTIONAL', {})
        out[s['symbol']] = SymbolRules(
            symbol=s['symbol'],
            tick=_dec(f.get('PRICE_FILTER', {}).get('tickSize', '0')),
            step=_dec(lot.get('stepSize', '0')),
            min_qty=_dec(lot.get('minQty', '0')),
            min_notional=_dec(mn.get('notional', mn.get('minNotional', '0'))),
            market_step=_dec(mlot.get('stepSize', lot.get('stepSize', '0'))),
            market_min_qty=_dec(mlot.get('minQty', lot.get('minQty', '0'))),
        )
    return out


class FuturesREST:
    """USDⓈ-M 선물 REST. 주문은 재시도하지 않는다(중복 체결 방지) — 조회만 재시도."""

    def __init__(self, base_url: str, api_key: str = '', api_secret: str = '', *,
                 recv_window: int = 5000, timeout: float = 10.0,
                 transport: Optional[Transport] = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.timeout = timeout
        self.transport: Transport = transport or urllib_transport
        self.clock = clock
        self.sleep = sleep
        self.time_offset_ms = 0
        self._rules: dict[str, SymbolRules] = {}

    # ── 서명·전송 ────────────────────────────────────────────
    def sign(self, query: str) -> str:
        return hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def request(self, method: str, path: str, params: Optional[dict] = None, *,
                signed: bool = False, retries: int = 0) -> Any:
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            p = {k: v for k, v in (params or {}).items() if v is not None}
            if signed:
                if not self.api_key or not self.api_secret:
                    raise BinanceAPIError(0, None, 'API 키가 설정되지 않았습니다', path)
                p['recvWindow'] = self.recv_window
                p['timestamp'] = int(self.clock() * 1000) + self.time_offset_ms
            query = urllib.parse.urlencode(p)
            if signed:
                query = f'{query}&signature={self.sign(query)}'
            url = self.base_url + path
            body: Optional[bytes] = None
            if method in ('GET', 'DELETE'):
                if query:
                    url = f'{url}?{query}'
            else:
                body = query.encode()
            headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
            try:
                status, text = self.transport(method, url, body, headers, self.timeout)
            except OSError as e:   # 네트워크 단절·타임아웃 (URLError 포함)
                last = e
                if attempt < retries:
                    self.sleep(1 + attempt)
                    continue
                raise
            try:
                data = json.loads(text) if text else None
            except ValueError:
                log.debug(f'JSON 아닌 응답 HTTP {status} {path}')
                data = None
                if status < 400:
                    raise BinanceAPIError(status, None, f'JSON 아님: {text[:200]}', path) from None
            if status >= 400:
                code = data.get('code') if isinstance(data, dict) else None
                msg = data.get('msg', text) if isinstance(data, dict) else text
                err = BinanceAPIError(status, code, str(msg), path)
                if code == -1021 and signed:              # 시계 어긋남 → 동기화 후 재시도
                    self.sync_time()
                if attempt < retries and (status >= 500 or status == 429 or code == -1021):
                    last = err
                    self.sleep(1 + attempt)
                    continue
                raise err
            return data
        raise last if last else RuntimeError('unreachable')

    def sync_time(self) -> int:
        server = self.request('GET', '/fapi/v1/time', retries=2)['serverTime']
        self.time_offset_ms = int(server) - int(self.clock() * 1000)
        return self.time_offset_ms

    # ── 공개 데이터 ───────────────────────────────────────────
    def exchange_info(self) -> dict:
        info = self.request('GET', '/fapi/v1/exchangeInfo', retries=2)
        self._rules = parse_rules(info)
        return info

    def rules(self, symbol: str) -> SymbolRules:
        if symbol not in self._rules:
            self.exchange_info()
        return self._rules[symbol]

    def klines(self, symbol: str, interval: str, limit: int = 500,
               start_ms: Optional[int] = None) -> list[list]:
        return self.request('GET', '/fapi/v1/klines', {
            'symbol': symbol, 'interval': interval, 'limit': limit, 'startTime': start_ms}, retries=2)

    def premium_index(self, symbol: Optional[str] = None) -> Any:
        """마크가격·직전 펀딩비. symbol 없으면 전 종목 목록."""
        return self.request('GET', '/fapi/v1/premiumIndex', {'symbol': symbol}, retries=2)

    def book_ticker(self, symbol: str) -> dict:
        return self.request('GET', '/fapi/v1/ticker/bookTicker', {'symbol': symbol}, retries=2)

    # ── 계좌 ─────────────────────────────────────────────────
    def account(self) -> dict:
        return self.request('GET', '/fapi/v3/account', signed=True, retries=2)

    def position_risk(self, symbol: Optional[str] = None) -> list[dict]:
        return self.request('GET', '/fapi/v3/positionRisk', {'symbol': symbol}, signed=True, retries=2)

    def get_position_mode(self) -> bool:
        """True면 헤지 모드(양방향). 이 봇은 단방향(False)이어야 한다."""
        return bool(self.request('GET', '/fapi/v1/positionSide/dual', signed=True, retries=2)['dualSidePosition'])

    def set_position_mode(self, hedge: bool) -> Any:
        return self.request('POST', '/fapi/v1/positionSide/dual',
                            {'dualSidePosition': 'true' if hedge else 'false'}, signed=True)

    def get_multi_assets(self) -> bool:
        return bool(self.request('GET', '/fapi/v1/multiAssetsMargin', signed=True, retries=2)['multiAssetsMargin'])

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self.request('POST', '/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage}, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> Any:
        try:
            return self.request('POST', '/fapi/v1/marginType',
                                {'symbol': symbol, 'marginType': margin_type}, signed=True)
        except BinanceAPIError as e:
            if e.code == -4046:          # 이미 그 마진 방식이다
                return {'code': -4046, 'msg': 'no change'}
            raise

    def income(self, income_type: Optional[str] = None, symbol: Optional[str] = None,
               start_ms: Optional[int] = None, limit: int = 1000) -> list[dict]:
        return self.request('GET', '/fapi/v1/income', {
            'incomeType': income_type, 'symbol': symbol, 'startTime': start_ms, 'limit': limit},
            signed=True, retries=2)

    def user_trades(self, symbol: str, start_ms: Optional[int] = None, limit: int = 500) -> list[dict]:
        return self.request('GET', '/fapi/v1/userTrades', {
            'symbol': symbol, 'startTime': start_ms, 'limit': limit}, signed=True, retries=2)

    # ── 일반 주문 (지정가·시장가) ─────────────────────────────
    def new_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, *,
                  price: Optional[Decimal] = None, time_in_force: Optional[str] = None,
                  reduce_only: bool = False, client_id: Optional[str] = None) -> dict:
        return self.request('POST', '/fapi/v1/order', {
            'symbol': symbol, 'side': side, 'type': order_type,
            'quantity': fmt_dec(quantity),
            'price': fmt_dec(price) if price is not None else None,
            'timeInForce': time_in_force,
            'reduceOnly': 'true' if reduce_only else None,
            'newClientOrderId': client_id,
            'newOrderRespType': 'RESULT',
        }, signed=True)

    def query_order(self, symbol: str, order_id: Optional[int] = None,
                    client_id: Optional[str] = None) -> dict:
        return self.request('GET', '/fapi/v1/order', {
            'symbol': symbol, 'orderId': order_id, 'origClientOrderId': client_id}, signed=True, retries=2)

    def cancel_order(self, symbol: str, order_id: Optional[int] = None,
                     client_id: Optional[str] = None) -> dict:
        return self.request('DELETE', '/fapi/v1/order', {
            'symbol': symbol, 'orderId': order_id, 'origClientOrderId': client_id}, signed=True)

    def open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        return self.request('GET', '/fapi/v1/openOrders', {'symbol': symbol}, signed=True, retries=2)

    # ── 조건부 주문 (Algo Order API) ──────────────────────────
    def new_stop_market(self, symbol: str, side: str, quantity: Decimal, trigger: Decimal, *,
                        working_type: str = 'MARK_PRICE', client_algo_id: Optional[str] = None) -> dict:
        """손절용 STOP_MARKET (reduceOnly). closePosition 대신 수량을 명시한다 —
        일부 자료가 algo 경로에서 closePosition 미지원을 보고해서 양쪽에서 확실히 되는 형태를 쓴다."""
        return self.request('POST', '/fapi/v1/algoOrder', {
            'algoType': 'CONDITIONAL', 'symbol': symbol, 'side': side, 'type': 'STOP_MARKET',
            'quantity': fmt_dec(quantity), 'triggerPrice': fmt_dec(trigger),
            'workingType': working_type, 'reduceOnly': 'true', 'priceProtect': 'true',
            'clientAlgoId': client_algo_id[:36] if client_algo_id else None,
        }, signed=True)

    def query_algo_order(self, algo_id: Optional[int] = None, client_algo_id: Optional[str] = None) -> dict:
        return self.request('GET', '/fapi/v1/algoOrder', {
            'algoId': algo_id, 'clientAlgoId': client_algo_id}, signed=True, retries=2)

    def cancel_algo_order(self, algo_id: Optional[int] = None, client_algo_id: Optional[str] = None) -> dict:
        """조건부 주문 취소. 공식 문서의 파라미터는 algoId / clientAlgoId 뿐이다(symbol 없음).
        예전 /fapi/v1/order 습관대로 symbol을 붙이지 않는다 — 남는 파라미터는 엔드포인트에 따라
        -1104(읽지 않은 파라미터)로 거부될 수 있다."""
        return self.request('DELETE', '/fapi/v1/algoOrder', {
            'algoId': algo_id, 'clientAlgoId': client_algo_id}, signed=True)

    def open_algo_orders(self, symbol: Optional[str] = None) -> list[dict]:
        res = self.request('GET', '/fapi/v1/openAlgoOrders', {'symbol': symbol}, signed=True, retries=2)
        if isinstance(res, dict):
            return list(res.get('orders', []))
        return list(res or [])


def algo_status(order: dict) -> str:
    """조건부 주문 상태 문자열 (응답 키가 algoStatus / status 로 갈린다)."""
    return str(order.get('algoStatus') or order.get('status') or '').upper()


def algo_id_of(order: dict) -> Optional[int]:
    v = order.get('algoId')
    return int(v) if v is not None else None

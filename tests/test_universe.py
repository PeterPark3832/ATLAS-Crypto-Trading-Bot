"""
ATLAS — 유니버스 모듈 단위 테스트
================================
atlas_spot_universe.py의 심볼 필터링/랭킹 로직을 검증합니다.

실행:
  pytest tests/test_universe.py -v
"""



from atlas_spot_universe import (
    _is_excluded, _is_stable_price,
    discover_universe, filter_universe, filter_tradeable,
    rank_by_momentum, get_backtest_universe,
)


# ══════════════════════════════════════════════════════════════
#  헬퍼: 가짜 ccxt 거래소
# ══════════════════════════════════════════════════════════════

class _FakeExchange:
    def __init__(self, tickers=None, markets=None, fetch_tickers_raises=False,
                 load_markets_raises=False):
        self._tickers = tickers or {}
        self._markets = markets or {}
        self._fetch_tickers_raises = fetch_tickers_raises
        self._load_markets_raises = load_markets_raises

    def fetch_tickers(self):
        if self._fetch_tickers_raises:
            raise RuntimeError('network error')
        return self._tickers

    def load_markets(self):
        if self._load_markets_raises:
            raise RuntimeError('network error')
        return self._markets


def _make_ohlcv_const_return(n: int, daily_return: float, start: float = 100.0) -> list:
    """매일 동일 수익률로 움직이는 합성 1D OHLCV (모멘텀 테스트용)."""
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + daily_return))
    day_ms = 24 * 3600 * 1000
    return [[i * day_ms, c, c, c, c, 1000.0] for i, c in enumerate(closes)]


# ══════════════════════════════════════════════════════════════
#  _is_excluded / _is_stable_price
# ══════════════════════════════════════════════════════════════

class TestIsExcluded:
    def test_stablecoin_excluded(self):
        assert _is_excluded('USDC') is True

    def test_leveraged_token_excluded(self):
        assert _is_excluded('BTCUP') is True
        assert _is_excluded('ETH3L') is True

    def test_normal_symbol_not_excluded(self):
        assert _is_excluded('BTC') is False
        assert _is_excluded('ETH') is False


class TestIsStablePrice:
    def test_within_peg_range(self):
        assert _is_stable_price(1.0) is True
        assert _is_stable_price(0.98) is True
        assert _is_stable_price(1.02) is True

    def test_outside_peg_range(self):
        assert _is_stable_price(50000.0) is False
        assert _is_stable_price(0.5) is False

    def test_zero_or_negative_not_stable(self):
        assert _is_stable_price(0) is False
        assert _is_stable_price(-1.0) is False

    def test_none_not_stable(self):
        assert _is_stable_price(None) is False


# ══════════════════════════════════════════════════════════════
#  discover_universe
# ══════════════════════════════════════════════════════════════

class TestDiscoverUniverse:
    def test_filters_and_sorts_by_volume(self):
        tickers = {
            'BTC/USDT': {'last': 50000.0, 'quoteVolume': 50_000_000},
            'ETH/USDT': {'last': 3000.0, 'quoteVolume': 80_000_000},
            'LOW/USDT': {'last': 1.0, 'quoteVolume': 1_000_000},     # 거래량 미달
        }
        ex = _FakeExchange(tickers=tickers)
        result = discover_universe(ex, min_volume=10_000_000, max_symbols=50)
        assert result == ['ETHUSDT', 'BTCUSDT']

    def test_excludes_stablecoins_and_leveraged(self):
        tickers = {
            'BTC/USDT': {'last': 50000.0, 'quoteVolume': 50_000_000},
            'USDC/USDT': {'last': 1.0, 'quoteVolume': 100_000_000},
            'BTCUP/USDT': {'last': 10.0, 'quoteVolume': 50_000_000},
        }
        ex = _FakeExchange(tickers=tickers)
        result = discover_universe(ex, min_volume=10_000_000, max_symbols=50)
        assert result == ['BTCUSDT']

    def test_excludes_price_pegged_symbols(self):
        """리스트 외 토큰이라도 가격이 $1 페그 범위면 자동 제외."""
        tickers = {
            'BTC/USDT': {'last': 50000.0, 'quoteVolume': 50_000_000},
            'WEIRDSTABLE/USDT': {'last': 0.99, 'quoteVolume': 50_000_000},
        }
        ex = _FakeExchange(tickers=tickers)
        result = discover_universe(ex, min_volume=10_000_000, max_symbols=50)
        assert result == ['BTCUSDT']

    def test_ignores_non_usdt_pairs(self):
        tickers = {
            'BTC/USDT': {'last': 50000.0, 'quoteVolume': 50_000_000},
            'BTC/BUSD': {'last': 50000.0, 'quoteVolume': 50_000_000},
        }
        ex = _FakeExchange(tickers=tickers)
        result = discover_universe(ex, min_volume=10_000_000, max_symbols=50)
        assert result == ['BTCUSDT']

    def test_respects_max_symbols(self):
        tickers = {
            f'SYM{i}/USDT': {'last': 10.0, 'quoteVolume': 100_000_000 - i}
            for i in range(10)
        }
        ex = _FakeExchange(tickers=tickers)
        result = discover_universe(ex, min_volume=1_000_000, max_symbols=3)
        assert len(result) == 3
        assert result == ['SYM0USDT', 'SYM1USDT', 'SYM2USDT']

    def test_fetch_tickers_failure_returns_empty(self):
        ex = _FakeExchange(fetch_tickers_raises=True)
        result = discover_universe(ex)
        assert result == []


# ══════════════════════════════════════════════════════════════
#  filter_universe
# ══════════════════════════════════════════════════════════════

class TestFilterUniverse:
    def test_no_exclude_returns_same(self):
        symbols = ['BTCUSDT', 'ETHUSDT']
        assert filter_universe(symbols) == symbols

    def test_excludes_listed_symbols(self):
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
        assert filter_universe(symbols, exclude=['ETHUSDT']) == ['BTCUSDT', 'XRPUSDT']


# ══════════════════════════════════════════════════════════════
#  filter_tradeable
# ══════════════════════════════════════════════════════════════

class TestFilterTradeable:
    def test_filters_inactive_and_non_spot(self):
        markets = {
            'BTC/USDT': {'active': True, 'spot': True},
            'ETH/USDT': {'active': False, 'spot': True},
            'XRP/USDT': {'active': True, 'spot': False},
        }
        ex = _FakeExchange(markets=markets)
        result = filter_tradeable(ex, ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'])
        assert result == ['BTCUSDT']

    def test_missing_market_excluded(self):
        ex = _FakeExchange(markets={'BTC/USDT': {'active': True, 'spot': True}})
        result = filter_tradeable(ex, ['BTCUSDT', 'NOPE'])
        assert result == ['BTCUSDT']

    def test_load_markets_failure_returns_input_unchanged(self):
        ex = _FakeExchange(load_markets_raises=True)
        symbols = ['BTCUSDT', 'ETHUSDT']
        assert filter_tradeable(ex, symbols) == symbols


# ══════════════════════════════════════════════════════════════
#  rank_by_momentum
# ══════════════════════════════════════════════════════════════

class TestRankByMomentum:
    def test_higher_momentum_ranked_first(self):
        cache = {
            'STRONG': _make_ohlcv_const_return(80, 0.02),
            'WEAK':   _make_ohlcv_const_return(80, 0.001),
        }
        result = rank_by_momentum(['WEAK', 'STRONG'], cache, momentum_days=45)
        assert result[0] == 'STRONG'

    def test_missing_data_ranked_last(self):
        cache = {
            'HAS_DATA': _make_ohlcv_const_return(80, 0.01),
            'NO_DATA':  [],
        }
        result = rank_by_momentum(['NO_DATA', 'HAS_DATA'], cache, momentum_days=45)
        assert result == ['HAS_DATA', 'NO_DATA']

    def test_insufficient_history_ranked_last(self):
        cache = {
            'SHORT': _make_ohlcv_const_return(10, 0.01),  # momentum_days+30일 미달
            'LONG':  _make_ohlcv_const_return(80, 0.01),
        }
        result = rank_by_momentum(['SHORT', 'LONG'], cache, momentum_days=45)
        assert result == ['LONG', 'SHORT']


# ══════════════════════════════════════════════════════════════
#  get_backtest_universe
# ══════════════════════════════════════════════════════════════

class TestGetBacktestUniverse:
    def test_default_is_fixed_universe(self):
        result = get_backtest_universe()
        assert 'BTCUSDT' in result
        assert 'NEARUSDT' not in result

    def test_extended_includes_fixed_plus_more(self):
        fixed = get_backtest_universe(extended=False)
        extended = get_backtest_universe(extended=True)
        assert set(fixed).issubset(set(extended))
        assert len(extended) > len(fixed)

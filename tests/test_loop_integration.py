"""
ATLAS — 전략 루프 통합 스모크 테스트
================================
단위 테스트는 개별 함수를 목으로 검증하므로 **루프 배선 오류**를 놓친다.
(예: 배치 프리페치를 넣었는데 루프에서 호출하지 않아 여전히 개별 조회)

여기서는 가짜 거래소로 `_strategy_timeframe_loop` 1회전을 실제로 돌려
진입→보호주문→관리→청산 경로가 실제로 이어지는지 확인한다.

실행:
  pytest tests/test_loop_integration.py -v
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


SYMS = [f'SYM{i}USDT' for i in range(12)]


class _LoopEx:
    """한 패스를 돌리기에 충분한 가짜 거래소 + 호출 계측."""

    def __init__(self):
        self.n_ticker = 0
        self.n_batch = 0
        self.n_ohlcv = 0
        self.orders = []

    def fetch_last_prices(self, syms):
        self.n_batch += 1
        return {s: {'price': 100.0} for s in syms}

    def fetch_ticker(self, sym):
        self.n_ticker += 1
        return {'last': 100.0, 'close': 100.0}

    def fetch_ohlcv(self, ccxt_sym, timeframe, limit=None):
        self.n_ohlcv += 1
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        out = []
        for i in range(max(limit or 300, 300)):
            ts = int((base + timedelta(hours=4 * i)).timestamp() * 1000)
            px = 100.0 + (i % 7) * 0.5
            out.append([ts, px, px * 1.01, px * 0.99, px, 1000.0])
        return out

    def create_market_buy_order(self, ccxt_sym, qty):
        self.orders.append(('buy', ccxt_sym, qty))
        return {'average': 100.0, 'price': 100.0, 'filled': qty}

    def create_market_sell_order(self, ccxt_sym, qty):
        self.orders.append(('sell', ccxt_sym, qty))
        return {'average': 101.0, 'price': 101.0, 'filled': qty}

    def create_order(self, ccxt_sym, otype, side, qty, price, params=None):
        self.orders.append(('stop', ccxt_sym, qty))
        return {'id': 'STOP1'}

    def cancel_order(self, oid, ccxt_sym):
        self.orders.append(('cancel', str(oid)))

    def fetch_order(self, oid, ccxt_sym):
        return {'status': 'open'}

    def fetch_open_orders(self, ccxt_sym):
        return []

    def fetch_balance(self, params=None):
        return {'free': {'USDT': 10_000.0}, 'USDT': {'total': 10_000.0}}

    def amount_to_precision(self, ccxt_sym, qty):
        return f'{float(qty):.6f}'

    def price_to_precision(self, ccxt_sym, px):
        return f'{float(px):.2f}'


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'loop.db')
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'ABSENT')
    sm.init_spot_db()
    monkeypatch.setattr(sm, '_tg', lambda m: None)
    monkeypatch.setattr(sm, '_price_cache', {})
    monkeypatch.setattr(sm, '_last_known_price', {})
    monkeypatch.setattr(sm, '_candle_cache', sm.CandleCache(ttl=300))
    monkeypatch.setattr(sm, '_state', {
        'equity': 10_000.0, 'usdt_balance': 10_000.0, 'peak_equity': 10_000.0,
        'day_pnl': 0.0, 'day_start_eq': 10_000.0, 'paused': False,
        'dry_run': False, 'daily_loss_alerted': False, 'ratchet_alert_tier': 0,
        'universe': list(SYMS), 'universe_ranked': list(SYMS),
        'active_strategies': ['S3'],
    })
    ex = _LoopEx()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


def _run_one_pass(monkeypatch, strategies=('S3',), timeframe='4h'):
    """루프를 1회전만 돌리고 중단시킨다."""
    stop = threading.Event()
    real_wait = stop.wait

    def _wait(timeout=None):
        stop.set()
        return real_wait(0)
    stop.wait = _wait
    sm._strategy_timeframe_loop(timeframe, list(strategies), stop)


class TestLoopWiring:
    def test_pass_uses_batch_prices_not_per_symbol(self, env, monkeypatch):
        """배치 프리페치가 루프에 실제로 배선됐는지.
        (함수만 만들고 호출을 안 하면 단위 테스트는 통과하지만
         운영에서는 여전히 심볼×전략마다 개별 조회가 나간다)"""
        _run_one_pass(monkeypatch)
        assert env.n_batch == 1, '패스당 배치 조회 1회여야 한다'
        assert env.n_ticker == 0, f'개별 조회가 {env.n_ticker}회 발생 — 캐시 미적용'

    def test_multi_strategy_pass_still_single_batch(self, env, monkeypatch):
        sm._state['active_strategies'] = ['S3', 'S4', 'S5', 'S6']
        _run_one_pass(monkeypatch, strategies=('S3', 'S4', 'S5', 'S6'), timeframe='1d')
        assert env.n_batch == 1
        assert env.n_ticker == 0, (
            f'12심볼×4전략인데 개별 조회 {env.n_ticker}회 — 기존 구조라면 48회')

    def test_open_position_symbol_is_prefetched(self, env, monkeypatch):
        """유니버스에서 빠진 보유 심볼도 SL/TP 판정에 시세가 필요하다."""
        sm._save_position('S3', 'GONEUSDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        captured = {}
        orig = sm._prefetch_prices

        def _spy(syms):
            captured['syms'] = list(syms)
            return orig(syms)
        monkeypatch.setattr(sm, '_prefetch_prices', _spy)
        _run_one_pass(monkeypatch)
        assert 'GONE/USDT' in captured['syms']

    def test_pass_does_not_raise_and_db_intact(self, env, monkeypatch):
        _run_one_pass(monkeypatch)
        # 포지션 테이블과 거래 테이블이 정상 조회돼야 한다
        assert isinstance(sm._load_all_positions(), list)
        with sm._db_lock, sm._db_conn() as c:
            c.execute('SELECT COUNT(*) FROM spot_trades').fetchone()

    def test_candle_cache_shared_across_strategies(self, env, monkeypatch):
        """같은 심볼·타임프레임 캔들을 전략마다 다시 받지 않는다."""
        sm._state['active_strategies'] = ['S3', 'S4', 'S5', 'S6']
        _run_one_pass(monkeypatch, strategies=('S3', 'S4', 'S5', 'S6'), timeframe='1d')
        assert env.n_ohlcv == len(SYMS), (
            f'심볼당 1회여야 하는데 {env.n_ohlcv}회 (전략 수만큼 중복 조회 중)')


class TestCandleCacheEviction:
    def test_stale_entries_are_swept(self, env):
        cache = sm.CandleCache(ttl=1)
        cache._sweep_every = 0
        for i in range(30):
            cache._cache[f'OLD{i}/USDT_1d'] = ([[1, 2, 3, 4, 5, 6]], time.time() - 100)
            cache._locks[f'OLD{i}/USDT_1d'] = threading.Lock()
        removed = cache._sweep()
        assert removed == 30
        assert cache._cache == {} and cache._locks == {}

    def test_fresh_entries_survive(self, env):
        cache = sm.CandleCache(ttl=300)
        cache._sweep_every = 0
        cache._cache['BTC/USDT_1d'] = ([[1, 2, 3, 4, 5, 6]], time.time())
        assert cache._sweep() == 0
        assert 'BTC/USDT_1d' in cache._cache

    def test_sweep_is_rate_limited(self, env):
        cache = sm.CandleCache(ttl=1)
        cache._cache['X_1d'] = ([[1]], time.time() - 100)
        assert cache._sweep() == 0, '스윕 주기 전에는 돌지 않아야 한다'

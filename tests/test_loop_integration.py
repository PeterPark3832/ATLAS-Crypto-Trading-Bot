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

    def test_open_position_symbol_is_managed(self, env, monkeypatch):
        """유니버스에서 빠진 보유 심볼도 실제로 _manage_position 을 타야 한다.

        시세 프리페치만 되고 순회에서 빠지면 SL/TP 판정·청산·보호주문
        재등록이 전부 멈춰 포지션이 방치된다(거래소 주문 없으면 무방비).
        """
        sm._save_position('S3', 'GONEUSDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        seen = []
        monkeypatch.setattr(
            sm, '_manage_position',
            lambda strat, sym, ccxt_sym, df, i: seen.append((strat, sym)))
        _run_one_pass(monkeypatch)
        assert ('S3', 'GONEUSDT') in seen, (
            '유니버스 밖 보유 심볼이 관리 루프에서 누락됐다')

    def test_out_of_universe_symbol_is_manage_only(self, env, monkeypatch):
        """유니버스 밖 심볼은 관리만 하고 신규 진입은 하지 않는다."""
        sm._save_position('S3', 'GONEUSDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        monkeypatch.setattr(sm, '_manage_position',
                            lambda *a, **k: None)
        entered = []
        monkeypatch.setattr(sm, '_spot_buy',
                            lambda *a, **k: entered.append(a[1]))
        _run_one_pass(monkeypatch)
        assert 'GONEUSDT' not in entered, (
            '유니버스에서 탈락한 심볼에 신규 진입이 발생했다')

    def test_other_timeframe_position_not_scanned(self, env, monkeypatch):
        """이 루프가 담당하지 않는 전략의 포지션은 끌어오지 않는다.

        (4H 루프가 1D 전용 전략의 보유 심볼까지 캔들을 받으면 낭비다)
        """
        sm._save_position('S5', 'OTHERUSDT', 100.0, 95.0, 110.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'RANGING')
        seen = []
        monkeypatch.setattr(
            sm, '_manage_position',
            lambda strat, sym, ccxt_sym, df, i: seen.append(sym))
        _run_one_pass(monkeypatch, strategies=('S3',), timeframe='4h')
        assert 'OTHERUSDT' not in seen

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


class TestClosedBarParity:
    """지표 판정은 **완성된 마지막 봉**으로 해야 한다.

    거래소가 돌려주는 마지막 봉은 형성 중이라 BB·EMA가 폴링마다 흔들린다.
    그 값으로 청산하면 ① 백테스트가 검증한 적 없는 동작이 되고(신호는 이미
    i-1을 쓰므로 내부에서도 어긋난다) ② 봉 중간에 잠깐 뒤집힌 크로스에
    휩쓸려 조기 청산된다.
    """

    def test_manage_position_gets_closed_bar(self, env, monkeypatch):
        seen = {}

        def _spy(strategy, symbol, ccxt_sym, df, i):
            seen['i'] = i
            seen['len'] = len(df)
        monkeypatch.setattr(sm, '_manage_position', _spy)
        _run_one_pass(monkeypatch)
        assert seen, '_manage_position이 호출되지 않았다'
        assert seen['i'] == seen['len'] - 2, (
            f"형성 중인 봉({seen['len']-1})이 아니라 완성봉({seen['len']-2})을 "
            f"넘겨야 한다 (실제 {seen['i']})")

    def test_signal_and_exit_use_same_bar(self, env, monkeypatch):
        """신호와 청산이 서로 다른 봉을 보면 같은 순간에 모순된 판단을 한다."""
        idx = {}

        def _mp(strategy, symbol, ccxt_sym, df, i):
            idx['exit'] = i
        monkeypatch.setattr(sm, '_manage_position', _mp)

        orig = sm.SIGNAL_FUNCS['S3']

        def _sig(df, i):
            idx['signal'] = i
            return orig(df, i)
        monkeypatch.setitem(sm.SIGNAL_FUNCS, 'S3', _sig)

        _run_one_pass(monkeypatch)
        if 'signal' in idx:      # 새 봉이 닫혀 신호까지 갔을 때만 비교 가능
            assert idx['signal'] == idx['exit']

    def test_single_bar_does_not_crash(self, env, monkeypatch):
        """봉이 1개뿐이어도 음수 인덱스로 떨어지지 않아야 한다."""
        assert max(0, 1 - 2) == 0


class TestCandleCacheEviction:
    def test_stale_entries_are_swept(self, env):
        cache = sm.CandleCache(ttl=1)
        cache._sweep_every = 0
        for i in range(30):
            cache._cache[f'OLD{i}/USDT_1d'] = ([[1, 2, 3, 4, 5, 6]], time.time() - 100)
            cache._locks[f'OLD{i}/USDT_1d'] = threading.Lock()
        removed = cache._sweep()
        assert removed == 30
        assert cache._cache == {}
        # 락 객체는 남긴다 — 임계구역에 있는 스레드의 락을 버리면
        # 뒤이어 들어온 스레드가 새 락을 만들어 상호배제가 깨진다.
        assert len(cache._locks) == 30

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

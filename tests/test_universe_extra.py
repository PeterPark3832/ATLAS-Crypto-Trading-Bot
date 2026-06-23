"""
ATLAS — 유니버스 갱신 루프(atlas_spot_universe.py) 단위 테스트
================================
test_universe.py가 다루지 않는 universe_refresh_loop을 검증합니다.

실행:
  pytest tests/test_universe_extra.py -v
"""

import os
import sys
import threading
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import atlas_spot_universe as uv


class _SetOnWaitEvent(threading.Event):
    """body(); wait() 순서의 루프에서 본문을 정확히 1회만 실행시키기 위해,
    wait() 호출 시점에 즉시 set()한다."""
    def wait(self, timeout=None):
        self.set()
        return True


class TestUniverseRefreshLoop:
    def test_updates_shared_state_with_new_universe(self, monkeypatch):
        monkeypatch.setattr(uv, 'discover_universe', lambda ex: ['BTCUSDT', 'ETHUSDT'])
        state = {'universe': ['OLDUSDT']}
        ev = _SetOnWaitEvent()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert state['universe'] == ['BTCUSDT', 'ETHUSDT']
        assert state['universe_ranked'] == ['BTCUSDT', 'ETHUSDT']
        assert 'universe_updated' in state

    def test_empty_discovery_keeps_existing_universe(self, monkeypatch):
        monkeypatch.setattr(uv, 'discover_universe', lambda ex: [])
        state = {'universe': ['OLDUSDT']}
        ev = _SetOnWaitEvent()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert state['universe'] == ['OLDUSDT']
        assert 'universe_ranked' not in state

    def test_uses_momentum_ranking_when_cache_present(self, monkeypatch):
        monkeypatch.setattr(uv, 'discover_universe', lambda ex: ['BTCUSDT', 'ETHUSDT'])
        calls = []

        def _fake_rank(symbols, ohlcv_cache, momentum_days=None):
            calls.append((symbols, ohlcv_cache))
            return list(reversed(symbols))

        monkeypatch.setattr(uv, 'rank_by_momentum', _fake_rank)
        state = {'universe': [], 'ohlcv_1d_cache': {'BTCUSDT': [[0, 1, 1, 1, 1, 1]]}}
        ev = _SetOnWaitEvent()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert calls
        assert state['universe_ranked'] == ['ETHUSDT', 'BTCUSDT']

    def test_skips_momentum_ranking_when_no_cache(self, monkeypatch):
        monkeypatch.setattr(uv, 'discover_universe', lambda ex: ['BTCUSDT', 'ETHUSDT'])
        called = []
        monkeypatch.setattr(uv, 'rank_by_momentum', lambda *a, **k: called.append(1))
        state = {'universe': []}
        ev = _SetOnWaitEvent()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert called == []
        assert state['universe_ranked'] == ['BTCUSDT', 'ETHUSDT']

    def test_exception_during_discovery_is_caught_and_state_kept(self, monkeypatch):
        def _raise(ex):
            raise RuntimeError('network error')

        monkeypatch.setattr(uv, 'discover_universe', _raise)
        state = {'universe': ['OLDUSDT']}
        ev = _SetOnWaitEvent()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert state['universe'] == ['OLDUSDT']

    def test_stop_event_set_before_first_iteration_skips_body(self, monkeypatch):
        called = []
        monkeypatch.setattr(uv, 'discover_universe', lambda ex: called.append(1))
        state = {}
        ev = threading.Event()
        ev.set()
        uv.universe_refresh_loop(object(), state, stop_event=ev, state_lock=threading.Lock())
        assert called == []
        assert state == {}

"""
ATLAS — 텔레그램 비동기 전송 큐 단위 테스트
================================
_tg(논블로킹 enqueue), _tg_worker(백그라운드 전송), _tg_flush(동기 flush),
큐 포화 시 최신 우선 드롭을 검증합니다.

실행:
  pytest tests/test_telegram_queue.py -v
"""

import os
import queue
import sys
import threading
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


@pytest.fixture(autouse=True)
def _fresh_queue(monkeypatch):
    """각 테스트마다 빈 큐 + 토큰 존재 상태로 초기화."""
    monkeypatch.setattr(sm, '_tg_queue', queue.Queue(maxsize=5))
    monkeypatch.setattr(sm, 'TG_TOKEN', 'tok')
    monkeypatch.setattr(sm, 'TG_CHAT_ID', '111')


@pytest.fixture
def _sent(monkeypatch):
    """실제 HTTP 전송을 기록으로 대체."""
    sent = []
    monkeypatch.setattr(sm, '_tg_send_now', lambda msg: sent.append(msg))
    return sent


class TestTgEnqueue:
    def test_tg_enqueues_without_sending(self, _sent):
        sm._tg('hello')
        assert _sent == []                    # 전송은 워커가 — _tg는 큐만
        assert sm._tg_queue.get_nowait() == 'hello'

    def test_tg_noop_without_token(self, monkeypatch, _sent):
        monkeypatch.setattr(sm, 'TG_TOKEN', '')
        sm._tg('hello')
        assert sm._tg_queue.qsize() == 0

    def test_full_queue_drops_oldest(self, _sent):
        for i in range(5):
            sm._tg(f'msg{i}')                 # 큐 maxsize=5 → 가득
        sm._tg('newest')                      # 가장 오래된 msg0 축출
        drained = []
        while True:
            try:
                drained.append(sm._tg_queue.get_nowait())
            except queue.Empty:
                break
        assert 'msg0' not in drained
        assert 'newest' in drained
        assert len(drained) == 5


class TestTgWorker:
    def test_worker_drains_and_sends(self, _sent):
        import time as _time
        sm._tg('a')
        sm._tg('b')
        stop = threading.Event()
        t = threading.Thread(target=sm._tg_worker, args=(stop,), daemon=True)
        t.start()
        for _ in range(100):                  # 큐가 빌 때까지 폴링 (최대 1초)
            if sm._tg_queue.empty() and len(_sent) >= 2:
                break
            _time.sleep(0.01)
        stop.set()
        t.join(timeout=2)
        assert set(_sent) == {'a', 'b'}

    def test_worker_flushes_remaining_on_stop(self, _sent):
        stop = threading.Event()
        stop.set()                            # 즉시 종료 → 루프 미진입, flush만
        sm._tg_queue.put_nowait('leftover')
        sm._tg_worker(stop)
        assert _sent == ['leftover']


class TestTgFlush:
    def test_flush_sends_all_synchronously(self, _sent):
        sm._tg('x')
        sm._tg('y')
        sm._tg_flush()
        assert _sent == ['x', 'y']
        assert sm._tg_queue.qsize() == 0

    def test_flush_empty_queue_noop(self, _sent):
        sm._tg_flush()
        assert _sent == []

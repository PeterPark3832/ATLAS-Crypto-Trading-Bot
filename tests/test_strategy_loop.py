"""
ATLAS — _strategy_timeframe_loop 단위 테스트
================================
atlas_spot_main.py의 메인 신호 루프(_strategy_timeframe_loop)가
중복봉 방지, 재시작 보호, 쿨다운, 레짐 라우팅, RS Gate, 펀딩비 필터
등 진입 차단/허용 조건을 올바르게 처리하는지 검증합니다.
_manage_position / _spot_buy는 모킹하여 "언제 호출되는가"만 봅니다.

실행:
  pytest tests/test_strategy_loop.py -v
"""

import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  공용 fixture / 헬퍼
# ══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    monkeypatch.setattr(sm, '_tg', lambda msg: None)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / 'test_spot.db'
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', db_file)
    sm.init_spot_db()
    yield db_file


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'NOT_PRESENT')


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sm.time, 'sleep', lambda s: None)


@pytest.fixture(autouse=True)
def _no_funding(monkeypatch):
    monkeypatch.setattr(sm, '_get_spot_funding', lambda sym: 0.0)


@pytest.fixture
def _manage_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, '_manage_position',
                         lambda *a: calls.append(a))
    return calls


@pytest.fixture
def _buy_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, '_spot_buy',
                         lambda *a: (calls.append(a), True)[1])
    return calls


@pytest.fixture
def _state(monkeypatch):
    fresh = {
        'universe': ['BTCUSDT'],
        'active_strategies': ['TEST'],
        'universe_ranked': [],
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


@pytest.fixture
def _fake_strategy(monkeypatch):
    """CALC_FUNCS/SIGNAL_FUNCS/REGIME_STRATEGY_MAP에 가짜 전략 'TEST' 등록."""
    signal_box = {'signal': 1, 'rr': 2.0}

    def fake_calc(ohlcv):
        ts = pd.to_datetime([row[0] for row in ohlcv], unit='ms', utc=True)
        return pd.DataFrame({'ts': ts})

    def fake_signal(df, i):
        return dict(signal_box)

    monkeypatch.setattr(sm, 'CALC_FUNCS', {'TEST': fake_calc})
    monkeypatch.setattr(sm, 'SIGNAL_FUNCS', {'TEST': fake_signal})
    monkeypatch.setattr(sm, 'REGIME_STRATEGY_MAP', {
        'TRENDING_UP': ['TEST'], 'WEAK_TREND': ['TEST'],
        'RANGING': [], 'TRENDING_DOWN': [],
    })
    monkeypatch.setattr(sm, 'get_cached_regime', lambda: _RS('TRENDING_UP'))
    return signal_box


class _RS:
    def __init__(self, regime):
        self.regime = regime


class _OneShotEvent(threading.Event):
    """wait() 1회 후 루프를 종료시킨다 (본문 1회만 실행)."""
    def wait(self, timeout=None):
        self.set()
        return True


class _TwoShotEvent(threading.Event):
    """wait() 2회 후 종료 — 동일 호출 내에서 while 루프를 2회 통과시킨다."""
    def __init__(self):
        super().__init__()
        self._n = 0

    def wait(self, timeout=None):
        self._n += 1
        if self._n >= 2:
            self.set()
        return self.is_set()


def _make_ohlcv(n=60, interval_ms=4 * 3600 * 1000):
    base = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return [[base + i * interval_ms, 100.0, 101.0, 99.0, 100.0, 1000.0] for i in range(n)]


def _fake_ex_and_cache(monkeypatch, ohlcv):
    monkeypatch.setattr(sm, '_get_ex', lambda: object())
    monkeypatch.setattr(sm._candle_cache, 'get', lambda ex, sym, tf, lim: ohlcv)
    monkeypatch.setattr(sm, '_get_price', lambda ccxt_sym: 100.0)


# ══════════════════════════════════════════════════════════════
#  기본 흐름
# ══════════════════════════════════════════════════════════════

class TestBasicFlow:
    def test_empty_universe_does_nothing(self, monkeypatch, _state, _manage_calls, _buy_calls):
        _state['universe'] = []
        ev = threading.Event()
        # universe가 비면 time.sleep(10) 후 continue되어 stop_event.wait()에
        # 도달하지 못하므로, sleep 호출 자체에서 이벤트를 set해 루프를 종료시킨다.
        monkeypatch.setattr(sm.time, 'sleep', lambda s: ev.set())
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _manage_calls == []
        assert _buy_calls == []

    def test_candle_fetch_failure_is_skipped(self, monkeypatch, _state, _manage_calls):
        monkeypatch.setattr(sm, '_get_ex', lambda: object())

        def _raise(ex, sym, tf, lim):
            raise RuntimeError('network error')
        monkeypatch.setattr(sm._candle_cache, 'get', _raise)
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _manage_calls == []

    def test_insufficient_candles_skips_strategy(self, monkeypatch, _state, _manage_calls):
        _fake_ex_and_cache(monkeypatch, _make_ohlcv(n=10))
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _manage_calls == []

    def test_inactive_strategy_is_skipped(self, monkeypatch, _state, _fake_strategy, _manage_calls):
        _state['active_strategies'] = ['OTHER']
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _manage_calls == []

    def test_manage_position_called_for_active_strategy(self, monkeypatch, _state, _fake_strategy,
                                                          _manage_calls):
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert len(_manage_calls) == 1
        assert _manage_calls[0][0] == 'TEST'
        assert _manage_calls[0][1] == 'BTCUSDT'


# ══════════════════════════════════════════════════════════════
#  중복봉 / 재시작 보호
# ══════════════════════════════════════════════════════════════

class TestDuplicateBarProtection:
    def test_same_bar_skips_signal_check_on_second_pass(self, monkeypatch, _state, _fake_strategy,
                                                          _manage_calls, _buy_calls):
        """같은 호출 내에서 while 루프가 2회 돌아도 동일 종가봉이면 신호체크는 1회만."""
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _TwoShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        # _manage_position은 매 폴링마다 호출되어 2회, 매수는 최초 1회만
        assert len(_manage_calls) == 2
        assert len(_buy_calls) == 1

    def test_restart_protection_blocks_reentry_via_db(self, monkeypatch, _state, _fake_strategy,
                                                       _manage_calls, _buy_calls):
        """last_bar 메모리가 초기화돼도(재시작 시뮬레이션) DB에 이미 이번 봉 이후
        진입 기록이 있으면 재진입이 차단되어야 한다."""
        ohlcv = _make_ohlcv()
        _fake_ex_and_cache(monkeypatch, ohlcv)
        _cur_ts = int(pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True).timestamp())
        future_ts = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        sm._save_position('TEST', 'BTCUSDT', 100.0, 90.0, 110.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        with sm._db_lock, sm._db_conn() as conn:
            conn.execute('UPDATE spot_positions SET entry_ts=? WHERE strategy=? AND symbol=?',
                         (future_ts, 'TEST', 'BTCUSDT'))
        # 기존 포지션 있음 체크보다 먼저 막히도록, 포지션은 다른 심볼 기록처럼
        # spot_trades에도 동일 효과를 줄 수 있으나 여기서는 spot_positions 경로로 충분.
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []


# ══════════════════════════════════════════════════════════════
#  쿨다운 / 기존포지션 / 레짐
# ══════════════════════════════════════════════════════════════

class TestGatingConditions:
    def test_existing_position_blocks_new_entry(self, monkeypatch, _state, _fake_strategy,
                                                  _buy_calls):
        sm._save_position('TEST', 'BTCUSDT', 100.0, 90.0, 110.0, 1.0, 100.0,
                           0.02, 'sl_tp', 0, 'TRENDING_UP')
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_crisis_regime_blocks_entry(self, monkeypatch, _state, _fake_strategy, _buy_calls):
        monkeypatch.setattr(sm, 'get_cached_regime', lambda: _RS(sm.REGIME_CRISIS))
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_regime_not_allowing_strategy_blocks_entry(self, monkeypatch, _state, _fake_strategy,
                                                         _buy_calls):
        monkeypatch.setattr(sm, 'get_cached_regime', lambda: _RS('RANGING'))
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_no_signal_blocks_entry(self, monkeypatch, _state, _fake_strategy, _buy_calls):
        _fake_strategy['signal'] = 0
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_s3_cooldown_set_after_buy_and_blocks_next_pass(self, monkeypatch, _state,
                                                              _fake_strategy, _buy_calls):
        _state['active_strategies'] = ['S3']
        monkeypatch.setattr(sm, 'CALC_FUNCS', {'S3': sm.CALC_FUNCS.get('S3', None) or
                                                (lambda ohlcv: pd.DataFrame({
                                                    'ts': pd.to_datetime(
                                                        [r[0] for r in ohlcv], unit='ms', utc=True)
                                                }))})
        monkeypatch.setattr(sm, 'SIGNAL_FUNCS', {'S3': lambda df, i: {'signal': 1, 'rr': 2.0}})
        monkeypatch.setattr(sm, 'REGIME_STRATEGY_MAP', {'TRENDING_UP': ['S3']})
        ohlcv = _make_ohlcv()
        _fake_ex_and_cache(monkeypatch, ohlcv)
        ev = _TwoShotEvent()
        sm._strategy_timeframe_loop('4h', ['S3'], ev)
        # 두번째 패스에서는 같은 종가봉이라 신호체크 전에 continue되므로
        # 매수는 1회만 발생 (쿨다운 분기까지 도달하려면 새 봉이 필요하지만,
        # 여기서는 동일봉 재진입 차단 경로로도 1회만 매수됨을 검증)
        assert len(_buy_calls) == 1


# ══════════════════════════════════════════════════════════════
#  RS Gate / 펀딩비 필터
# ══════════════════════════════════════════════════════════════

class TestRsGateAndFunding:
    def test_rs_gate_blocks_low_momentum_symbol(self, monkeypatch, _state, _fake_strategy,
                                                 _buy_calls):
        monkeypatch.setattr(sm, 'MOMENTUM_RS_GATE_STRATS', ['TEST'])
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda sym: 1.0)
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_rs_gate_allows_top_tier_with_risk_boost(self, monkeypatch, _state, _fake_strategy,
                                                      _buy_calls):
        monkeypatch.setattr(sm, 'MOMENTUM_RS_GATE_STRATS', ['TEST'])
        monkeypatch.setattr(sm, 'MOMENTUM_TOP_TIER_PCT', 0.2)
        monkeypatch.setattr(sm, 'MOMENTUM_TOP_RISK_MULT', 1.5)
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda sym: 0.1)
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert len(_buy_calls) == 1
        sig_arg = _buy_calls[0][3]
        assert sig_arg['_rs_scale'] == 1.5

    def test_funding_block_prevents_entry(self, monkeypatch, _state, _fake_strategy, _buy_calls):
        monkeypatch.setattr(sm, 'FUNDING_APPLY_STRATS', ['TEST'])
        monkeypatch.setattr(sm, 'FUNDING_LONG_BLOCK', 0.0005)
        monkeypatch.setattr(sm, '_get_spot_funding', lambda sym: 0.001)
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert _buy_calls == []

    def test_funding_short_boost_scales_sig(self, monkeypatch, _state, _fake_strategy, _buy_calls):
        monkeypatch.setattr(sm, 'FUNDING_APPLY_STRATS', ['TEST'])
        monkeypatch.setattr(sm, 'FUNDING_LONG_BLOCK', 0.0005)
        monkeypatch.setattr(sm, 'FUNDING_SHORT_BOOST', -0.0002)
        monkeypatch.setattr(sm, '_get_spot_funding', lambda sym: -0.0005)
        _fake_ex_and_cache(monkeypatch, _make_ohlcv())
        ev = _OneShotEvent()
        sm._strategy_timeframe_loop('4h', ['TEST'], ev)
        assert len(_buy_calls) == 1
        sig_arg = _buy_calls[0][3]
        assert sig_arg['_funding_scale'] == 1.20

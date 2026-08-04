"""
ATLAS — S1/S2/S7 전략 신호 로직 단위 테스트
================================
test_strategies.py가 S3~S6만 다루므로, CALC_FUNCS/SIGNAL_FUNCS에 등록된
나머지 활성 전략 S1(Buy&Hold), S2(SMA Golden Cross), S7(MACD Momentum)을
검증합니다.

실행:
  pytest tests/test_strategies_extra.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from atlas_spot_strategies import (
    calc_s1, get_signal_s1,
    calc_s2, get_signal_s2, check_exit_s2,
    calc_s7, get_signal_s7, check_exit_s7,
)
from atlas_spot_config import S2_SMA_SLOW, S7_MACD_SLOW, S7_EMA_TREND, S7_RR_MIN, S7_RR_MAX


def _make_flat_ohlcv(n: int, price: float = 1000.0, interval_ms: int = 14_400_000) -> list:
    return [[i * interval_ms, price, price * 1.001, price * 0.999, price, 1000.0]
            for i in range(n)]


def _make_uptrend_ohlcv(n: int, start: float = 100.0, drift: float = 0.01,
                         interval_ms: int = 14_400_000) -> list:
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    result = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.003
        l = min(o, c) * 0.997
        result.append([i * interval_ms, o, h, l, c, 1000.0])
    return result


def _flat_then_uptrend(n_flat: int, n_up: int, price: float = 100.0,
                        drift: float = 0.02, interval_ms: int = 14_400_000) -> list:
    flat = _make_flat_ohlcv(n_flat, price=price, interval_ms=interval_ms)
    up = _make_uptrend_ohlcv(n_up, start=price, drift=drift, interval_ms=interval_ms)
    up = [[(n_flat + j) * interval_ms, *row[1:]] for j, row in enumerate(up)]
    return flat + up


def _scan_signals(df: pd.DataFrame, get_signal_fn, warmup: int) -> list[dict]:
    sigs = []
    for i in range(warmup, len(df)):
        sig = get_signal_fn(df, i)
        if sig['signal'] == 1:
            sigs.append(sig)
    return sigs


# ══════════════════════════════════════════════════════════════
#  S1: Buy & Hold
# ══════════════════════════════════════════════════════════════

class TestS1:
    def test_calc_s1_returns_plain_ohlcv_df(self):
        ohlcv = _make_flat_ohlcv(10)
        df = calc_s1(ohlcv)
        assert len(df) == 10
        assert 'close' in df.columns

    def test_signals_buy_on_first_bar_only(self):
        ohlcv = _make_flat_ohlcv(20)
        df = calc_s1(ohlcv)
        sig0 = get_signal_s1(df, 0)
        assert sig0['signal'] == 1
        assert sig0['exit_type'] == 'hold'
        assert sig0['sl'] == 0.0 and sig0['tp'] == 0.0
        for i in range(1, 20):
            assert get_signal_s1(df, i)['signal'] == 0


# ══════════════════════════════════════════════════════════════
#  S2: SMA Golden Cross
# ══════════════════════════════════════════════════════════════

class TestS2:
    def test_warmup_no_signal(self):
        ohlcv = _make_uptrend_ohlcv(50)
        df = calc_s2(ohlcv)
        for i in range(min(S2_SMA_SLOW + 5, len(df))):
            assert get_signal_s2(df, i)['signal'] == 0

    def test_golden_cross_after_flat_then_uptrend(self):
        ohlcv = _flat_then_uptrend(n_flat=S2_SMA_SLOW + 10, n_up=80, drift=0.02)
        df = calc_s2(ohlcv)
        sigs = _scan_signals(df, get_signal_s2, warmup=S2_SMA_SLOW + 5)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp'] or sig['tp'] == 0.0
            assert sig['exit_type'] == 'death_cross'

    def test_flat_series_no_signal(self):
        ohlcv = _make_flat_ohlcv(S2_SMA_SLOW + 60)
        df = calc_s2(ohlcv)
        sigs = _scan_signals(df, get_signal_s2, warmup=S2_SMA_SLOW + 5)
        assert sigs == []

    def test_check_exit_on_death_cross(self):
        up = _make_uptrend_ohlcv(S2_SMA_SLOW + 60, drift=0.015)
        down = _make_uptrend_ohlcv(S2_SMA_SLOW + 60,
                                    start=float(pd.DataFrame(up)[4].iloc[-1]), drift=-0.02)
        down = [[row[0] + up[-1][0], *row[1:]] for row in down]
        df = calc_s2(up + down)
        exits = [check_exit_s2(df, i) for i in range(S2_SMA_SLOW + 5, len(df))]
        assert any(exits)

    def test_check_exit_before_index_1_is_false(self):
        ohlcv = _make_uptrend_ohlcv(10)
        df = calc_s2(ohlcv)
        assert check_exit_s2(df, 0) is False


# ══════════════════════════════════════════════════════════════
#  S7: MACD Momentum
# ══════════════════════════════════════════════════════════════

class TestS7:
    _warmup = S7_MACD_SLOW + S7_EMA_TREND + 5

    def test_warmup_no_signal(self):
        ohlcv = _make_uptrend_ohlcv(50)
        df = calc_s7(ohlcv)
        for i in range(min(self._warmup, len(df))):
            assert get_signal_s7(df, i)['signal'] == 0

    def test_macd_cross_after_flat_then_uptrend(self):
        ohlcv = _flat_then_uptrend(n_flat=self._warmup + 10, n_up=100, drift=0.02)
        df = calc_s7(ohlcv)
        sigs = _scan_signals(df, get_signal_s7, warmup=self._warmup)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp']
            assert S7_RR_MIN - 0.01 <= sig['rr'] <= S7_RR_MAX + 0.01
            assert sig['exit_type'] == 'sl_tp'

    def test_flat_series_no_signal(self):
        ohlcv = _make_flat_ohlcv(self._warmup + 60)
        df = calc_s7(ohlcv)
        sigs = _scan_signals(df, get_signal_s7, warmup=self._warmup)
        assert sigs == []

    def test_check_exit_after_two_negative_hist_bars(self):
        ohlcv = _flat_then_uptrend(n_flat=self._warmup + 10, n_up=80, drift=0.02)
        down = _make_uptrend_ohlcv(60, start=ohlcv[-1][4], drift=-0.02,
                                    interval_ms=14_400_000)
        down = [[row[0] + ohlcv[-1][0], *row[1:]] for row in down]
        df = calc_s7(ohlcv + down)
        exits = [check_exit_s7(df, i) for i in range(self._warmup, len(df))]
        assert any(exits)

    def test_check_exit_below_min_bars_is_false(self):
        ohlcv = _make_uptrend_ohlcv(10)
        df = calc_s7(ohlcv)
        assert check_exit_s7(df, 0) is False

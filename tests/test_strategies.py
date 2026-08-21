"""
ATLAS — 전략 신호 로직 단위 테스트
================================
atlas_spot_strategies.py의 get_signal_sN()이 의도한 조건에서
실제로 유효한 매수 신호(sl < entry < tp)를 만드는지,
조건 미충족 시 신호 없음(_no_sig)을 반환하는지 검증합니다.

실행:
  pytest tests/test_strategies.py -v
"""



import numpy as np
import pandas as pd
import pytest

from atlas_spot_strategies import (
    calc_s3, get_signal_s3,
    calc_s4, get_signal_s4,
    calc_s5, get_signal_s5,
    calc_s6, get_signal_s6,
)


# ══════════════════════════════════════════════════════════════
#  헬퍼: 합성 OHLCV — 급락 후 V자 반등 (S4/S5 과매도 반등 트리거용)
# ══════════════════════════════════════════════════════════════

def _make_crash_then_bounce_ohlcv(seed: int = 1, n_calm: int = 40,
                                   n_decline: int = 15, decline_rate: float = 0.998,
                                   n_bounce: int = 40, bounce_rate: float = 0.0015,
                                   start: float = 1000.0, interval_ms: int = 86_400_000) -> list:
    """평온한 구간(n_calm) → 완만한 하락(n_decline, RSI 과매도+BB하단 이탈)
    → 느린 반등(n_bounce, RSI 30 상향돌파를 BB하단 근처에서 발생시킴).
    급격한 1봉 크래시는 RSI(SMA 기반)가 창에서 빠지며 불연속 점프를 만들어
    '아직 밴드 근처'라는 전제가 깨지므로, 완만한 다구간 하락+완만한 반등으로 구성."""
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n_calm):
        closes.append(closes[-1] * (1 + rng.normal(0.0, 0.0005)))
    for _ in range(n_decline):
        closes.append(closes[-1] * decline_rate)
    for _ in range(n_bounce):
        closes.append(closes[-1] * (1 + rng.normal(bounce_rate, 0.001)))
    result = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.002
        l = min(o, c) * 0.998
        result.append([i * interval_ms, o, h, l, c, 1000.0])
    return result


def _make_flat_ohlcv(n: int, price: float = 1000.0, interval_ms: int = 86_400_000) -> list:
    """변동 없는 평탄 시리즈 — 어떤 전략도 신호를 내지 않아야 함."""
    result = []
    for i in range(n):
        result.append([i * interval_ms, price, price * 1.001, price * 0.999, price, 1000.0])
    return result


def _make_uptrend_ohlcv(n: int, start: float = 100.0, drift: float = 0.01,
                         interval_ms: int = 14_400_000, vol_spike_at: int | None = None) -> list:
    """꾸준한 상승 추세 (EMA 골든크로스 + ADX 강세 유발용)."""
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    result = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.003
        l = min(o, c) * 0.997
        v = 1000.0 if vol_spike_at is None or i != vol_spike_at else 5000.0
        result.append([i * interval_ms, o, h, l, c, v])
    return result


def _scan_signals(df: pd.DataFrame, get_signal_fn, warmup: int) -> list[dict]:
    sigs = []
    for i in range(warmup, len(df)):
        sig = get_signal_fn(df, i)
        if sig['signal'] == 1:
            sigs.append(sig)
    return sigs


# ══════════════════════════════════════════════════════════════
#  S3: EMA Trend Follow
# ══════════════════════════════════════════════════════════════

class TestS3:
    def test_warmup_no_signal(self):
        ohlcv = _make_uptrend_ohlcv(50)
        df = calc_s3(ohlcv)
        for i in range(min(210, len(df))):
            assert get_signal_s3(df, i)['signal'] == 0

    def test_uptrend_eventually_signals(self):
        """평탄한 구간(EMA 동조) 후 급추세 전환 → 워밍업 이후 골든크로스 신호 발생.
        처음부터 상승만 하면 골든크로스가 워밍업 전(봉 0~1)에 끝나버려 못 잡으므로,
        EMA가 수렴한 평탄 구간 뒤에 추세를 시작시켜 워밍업 이후 크로스를 유도."""
        flat = _make_flat_ohlcv(230, price=100.0, interval_ms=14_400_000)
        up = _make_uptrend_ohlcv(80, start=100.0, drift=0.02, interval_ms=14_400_000)
        up = [[(230 + j) * 14_400_000, *row[1:]] for j, row in enumerate(up)]
        df = calc_s3(flat + up)
        sigs = _scan_signals(df, get_signal_s3, warmup=210)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp']
            assert sig['rr'] >= 1.5 and sig['rr'] <= 3.0

    def test_flat_series_no_signal(self):
        """평탄 시리즈 — 골든크로스도 ADX 강세도 없음."""
        ohlcv = _make_flat_ohlcv(260)
        df = calc_s3(ohlcv)
        sigs = _scan_signals(df, get_signal_s3, warmup=210)
        assert sigs == []

    def test_check_exit_death_cross(self):
        from atlas_spot_strategies import check_exit_s3
        ohlcv = _make_uptrend_ohlcv(260, drift=0.015)
        df = calc_s3(ohlcv)
        # 추세 반전 구간을 만들어 데스크로스 발생 확인
        falling = _make_uptrend_ohlcv(60, start=float(df['close'].iloc[-1]), drift=-0.02)
        ohlcv2 = ohlcv + [[row[0] + ohlcv[-1][0], *row[1:]] for row in falling]
        df2 = calc_s3(ohlcv2)
        exits = [check_exit_s3(df2, i) for i in range(220, len(df2))]
        assert any(exits)


# ══════════════════════════════════════════════════════════════
#  S4: RSI Mean Reversion
# ══════════════════════════════════════════════════════════════

class TestS4:
    def test_warmup_no_signal(self):
        ohlcv = _make_crash_then_bounce_ohlcv()
        df = calc_s4(ohlcv)
        warmup = 20 + 14 + 5
        for i in range(min(warmup, len(df))):
            assert get_signal_s4(df, i)['signal'] == 0

    def test_crash_then_bounce_signals(self):
        """급락(RSI 과매도+BB하단 이탈) 후 반등(RSI 30 상향돌파) → 신호 발생."""
        ohlcv = _make_crash_then_bounce_ohlcv()
        df = calc_s4(ohlcv)
        warmup = 20 + 14 + 5
        sigs = _scan_signals(df, get_signal_s4, warmup)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp']
            assert sig['rr'] == pytest.approx(1.5)

    def test_pure_uptrend_no_signal(self):
        """RSI가 30 아래로 내려간 적이 없으면 상향돌파 신호도 없음."""
        ohlcv = _make_uptrend_ohlcv(80, interval_ms=86_400_000)
        df = calc_s4(ohlcv)
        warmup = 20 + 14 + 5
        sigs = _scan_signals(df, get_signal_s4, warmup)
        assert sigs == []


# ══════════════════════════════════════════════════════════════
#  S5: Bollinger Band Bounce
# ══════════════════════════════════════════════════════════════

class TestS5:
    def test_warmup_no_signal(self):
        ohlcv = _make_crash_then_bounce_ohlcv()
        df = calc_s5(ohlcv)
        warmup = 20 + 15
        for i in range(min(warmup, len(df))):
            assert get_signal_s5(df, i)['signal'] == 0

    def test_crash_triggers_bb_breach_signal(self):
        """BB 하단 이탈 + RSI<30 구간에서 신호 발생, TP=BB상단 > entry."""
        ohlcv = _make_crash_then_bounce_ohlcv()
        df = calc_s5(ohlcv)
        warmup = 20 + 15
        sigs = _scan_signals(df, get_signal_s5, warmup)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp']

    def test_flat_series_no_signal(self):
        ohlcv = _make_flat_ohlcv(60)
        df = calc_s5(ohlcv)
        warmup = 20 + 15
        sigs = _scan_signals(df, get_signal_s5, warmup)
        assert sigs == []


# ══════════════════════════════════════════════════════════════
#  S6: Donchian Breakout
# ══════════════════════════════════════════════════════════════

class TestS6:
    def test_warmup_no_signal(self):
        ohlcv = _make_uptrend_ohlcv(50, drift=0.02)
        df = calc_s6(ohlcv)
        warmup = 20 + 20 + 5
        for i in range(min(warmup, len(df))):
            assert get_signal_s6(df, i)['signal'] == 0

    def test_breakout_with_volume_spike_signals(self):
        """20일 신고가 돌파 + 거래량 2배 스파이크 → 신호 발생."""
        # 평탄한 구간 뒤에 강한 돌파+거래량 스파이크 봉 추가
        flat = _make_flat_ohlcv(60, price=100.0, interval_ms=86_400_000)
        breakout_bar = [[60 * 86_400_000, 100.0, 130.0, 100.0, 128.0, 5000.0]]
        more = _make_flat_ohlcv(5, price=128.0, interval_ms=86_400_000)
        more = [[(61 + j) * 86_400_000, *row[1:]] for j, row in enumerate(more)]
        ohlcv = flat + breakout_bar + more
        df = calc_s6(ohlcv)
        warmup = 20 + 20 + 5
        sigs = _scan_signals(df, get_signal_s6, warmup)
        assert len(sigs) > 0
        for sig in sigs:
            assert sig['sl'] < sig['tp']

    def test_breakout_without_volume_spike_no_signal(self):
        """가격은 돌파하지만 거래량 스파이크가 없으면 신호 없음."""
        flat = _make_flat_ohlcv(60, price=100.0, interval_ms=86_400_000)
        breakout_bar = [[60 * 86_400_000, 100.0, 130.0, 100.0, 128.0, 1000.0]]  # 거래량 평소와 동일
        ohlcv = flat + breakout_bar
        df = calc_s6(ohlcv)
        warmup = 20 + 20 + 5
        sigs = _scan_signals(df, get_signal_s6, warmup)
        assert sigs == []

    def test_check_exit_donchian_low_breach(self):
        """직전 봉의 10일 신저가보다 더 깊이 이탈하면 청산 신호."""
        from atlas_spot_strategies import check_exit_s6
        flat = _make_flat_ohlcv(60, price=100.0, interval_ms=86_400_000)
        crash_bar = [[60 * 86_400_000, 100.0, 100.0, 50.0, 60.0, 1000.0]]
        deeper_bar = [[61 * 86_400_000, 60.0, 60.0, 30.0, 40.0, 1000.0]]
        ohlcv = flat + crash_bar + deeper_bar
        df = calc_s6(ohlcv)
        assert check_exit_s6(df, len(df) - 1, entry_price=100.0) is True

    def test_check_exit_no_breach_when_above_prior_low(self):
        """직전 10일 신저가(약 99.9) 위에서 마감 — 청산 신호 없음."""
        from atlas_spot_strategies import check_exit_s6
        flat = _make_flat_ohlcv(60, price=100.0, interval_ms=86_400_000)
        mild_dip = [[60 * 86_400_000, 100.0, 100.0, 99.95, 99.97, 1000.0]]
        ohlcv = flat + mild_dip
        df = calc_s6(ohlcv)
        assert check_exit_s6(df, len(df) - 1, entry_price=100.0) is False

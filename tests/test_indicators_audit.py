"""
ATLAS — 지표 계산 감사
================================
지표가 틀리면 그 위 전부가 틀린다. 세 가지를 확인·고정한다.

① **데이터 부족이 RANGING으로 위장되던 문제**
   `calc_adx`는 봉이 모자라면 0.0을 돌려주는데, `classify_regime`은 그걸
   '추세 없음'으로 읽어 RANGING으로 분류한다. `update_regime`의 가드가
   30봉이었던 반면 calc_adx는 42봉(14×3)을 요구해서, 30~41봉 구간에서
   데이터 문제가 정상 레짐으로 둔갑하고 S4·S5가 그대로 진입했다.

② **ADX는 Wilder 평활이 맞는지**

③ **ATR·RSI는 단순이동평균(SMA)** — ADX와 혼용이다. 라이브와 백테스트가
   같은 함수를 쓰므로 검증이 무효가 되지는 않지만, ATR_SL 계열 파라미터는
   이 정의에 맞춰진 값이라 외부 지표와 직접 비교할 수 없다. 사실을 고정한다.

실행:
  pytest tests/test_indicators_audit.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import atlas_indicators as ind
import atlas_regime as R


def _bars(n, seed=1, drift=0.002, vol=0.02):
    rng = np.random.default_rng(seed)
    px, ts, out = 100.0, 1609459200000, []
    for i in range(n):
        o = px
        px *= (1 + drift + rng.normal(0, vol))
        out.append([ts + i * 86400000, o, max(o, px) * 1.01,
                    min(o, px) * 0.99, px, 1e6])
    return out


# ══════════════════════════════════════════════════════════════
#  ① 데이터 부족 → RANGING 위장 방지
# ══════════════════════════════════════════════════════════════

class TestInsufficientData:
    def test_min_bars_helper_exists(self):
        assert ind.adx_min_bars(14) == 42
        assert ind.adx_min_bars(20) == 60

    def test_adx_returns_zero_below_minimum(self):
        for n in (30, 35, 41):
            assert ind.calc_adx(_bars(n), 14) == 0.0

    def test_adx_works_at_minimum(self):
        assert ind.calc_adx(_bars(42), 14) > 0.0

    def test_zero_adx_would_be_classified_as_ranging(self):
        """이게 위험의 본질 — 0.0은 '추세 없음'과 구분되지 않는다."""
        assert R.classify_regime(0.0, 100, 90, 0.02) == 'RANGING'

    def test_regime_guard_matches_adx_requirement(self):
        """가드가 ADX 요구치보다 낮으면 그 사이 구간이 조용히 오분류된다."""
        src = Path(R.__file__).read_text()
        assert 'adx_min_bars' in src, (
            'update_regime 가드가 calc_adx 요구 봉수를 참조해야 한다')
        assert 'len(ohlcv) < 30' not in src, '30봉 하드코딩 가드가 남아 있다'

    def test_short_series_keeps_previous_regime(self, monkeypatch):
        """봉이 모자라면 새 레짐을 만들지 말고 직전 값을 유지해야 한다."""
        prev = R.get_cached_regime()

        class _Ex:
            def fetch_ohlcv(self, sym, tf, limit=None):
                return _bars(35)          # 30 이상 42 미만 — 구멍 구간

        state = R.update_regime(_Ex())
        assert state.regime == prev.regime, (
            '데이터 부족인데 새 레짐(RANGING)을 만들어냈다')


# ══════════════════════════════════════════════════════════════
#  ② ADX — Wilder 평활 검증
# ══════════════════════════════════════════════════════════════

class TestAdxCorrectness:
    def test_matches_wilder_reference(self):
        """독립 구현(ewm alpha=1/n)과 근사해야 한다.
        초기화 방식 차이로 완전 일치하지는 않으므로 허용 오차를 둔다."""
        rows = _bars(300, seed=7)
        got = ind.calc_adx(rows, 14)

        df = ind._ohlcv_to_df(rows)
        h, l, c = df['high'], df['low'], df['close']
        up, dn = h.diff(), -l.diff()
        dm_p = ((up > dn) & (up > 0)) * up.clip(lower=0)
        dm_m = ((dn > up) & (dn > 0)) * dn.clip(lower=0)
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        a = 1 / 14
        tr_s = tr.ewm(alpha=a, adjust=False).mean()
        di_p = 100 * dm_p.ewm(alpha=a, adjust=False).mean() / tr_s
        di_m = 100 * dm_m.ewm(alpha=a, adjust=False).mean() / tr_s
        dx = ((di_p - di_m).abs() / (di_p + di_m) * 100).fillna(0)
        ref = float(dx.ewm(alpha=a, adjust=False).mean().iloc[-1])

        assert got == pytest.approx(ref, rel=0.15), (
            f'ADX가 Wilder 기준과 크게 다르다 (구현 {got} vs 기준 {ref})')

    def test_trending_higher_than_choppy(self):
        trend = ind.calc_adx(_bars(200, seed=2, drift=0.02, vol=0.005), 14)
        chop = ind.calc_adx(_bars(200, seed=2, drift=0.0, vol=0.03), 14)
        assert trend > chop, '추세 구간의 ADX가 더 높아야 한다'

    def test_bounded_range(self):
        for seed in (1, 5, 9):
            assert 0 <= ind.calc_adx(_bars(200, seed=seed), 14) <= 100


# ══════════════════════════════════════════════════════════════
#  ③ ATR·RSI 정의 (SMA — 의도적 고정)
# ══════════════════════════════════════════════════════════════

class TestSmoothingConvention:
    def test_atr_uses_sma_not_wilder(self):
        """현재 구현은 SMA다. 바꾸면 **모든 전략의 손절 거리**가 달라지므로
        검증 없이 건드리면 안 된다. 사실을 테스트로 못박는다."""
        rows = _bars(200, seed=4)
        df = ind._ohlcv_to_df(rows)
        pc = df['close'].shift(1)
        tr = pd.concat([df['high'] - df['low'],
                        (df['high'] - pc).abs(),
                        (df['low'] - pc).abs()], axis=1).max(axis=1)
        assert ind._calc_atr(df, 14).iloc[-1] == pytest.approx(
            tr.rolling(14).mean().iloc[-1])

    def test_atr_close_to_wilder_in_practice(self):
        """정의는 다르지만 실측 차이는 작다(오해 방지용 근거)."""
        rows = _bars(200, seed=4)
        df = ind._ohlcv_to_df(rows)
        pc = df['close'].shift(1)
        tr = pd.concat([df['high'] - df['low'],
                        (df['high'] - pc).abs(),
                        (df['low'] - pc).abs()], axis=1).max(axis=1)
        sma = ind._calc_atr(df, 14).iloc[-1]
        wilder = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        assert abs(sma / wilder - 1) < 0.15

    def test_rsi_zero_loss_handled(self):
        """무손실 구간에서 RSI가 NaN으로 비지 않아야 한다(강세장 공백 방지)."""
        close = pd.Series(np.linspace(100, 200, 60))
        rsi = ind._calc_rsi(close, 14)
        assert not rsi.iloc[-1] != rsi.iloc[-1]      # NaN 아님
        assert rsi.iloc[-1] == pytest.approx(100.0)

    def test_rsi_bounded(self):
        rng = np.random.default_rng(3)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)))
        rsi = ind._calc_rsi(close, 14).dropna()
        assert rsi.min() >= 0 and rsi.max() <= 100


# ══════════════════════════════════════════════════════════════
#  백테스트 레짐 윈도우는 충분한가
# ══════════════════════════════════════════════════════════════

class TestBacktestRegimeWindow:
    def test_window_exceeds_adx_minimum(self):
        """build_regime_map은 i>=50부터 51봉 윈도우를 쓰므로 42봉 요구를 만족."""
        import atlas_spot_backtest as bt
        m = bt.build_regime_map(_bars(200, seed=6))
        assert m, '레짐맵이 비었다'
        assert 'RANGING' in set(m.values()) or len(set(m.values())) >= 1
        # 모든 값이 RANGING이면 ADX가 0으로 죽었다는 신호
        assert set(m.values()) != {'RANGING'}, (
            '전부 RANGING — ADX가 계산되지 않았을 가능성'
        )

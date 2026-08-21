"""
ATLAS — 백테스트 프레임워크 단위 테스트
================================
atlas_spot_backtest.py의 순수 유틸리티 + bar-by-bar 시뮬레이터를 검증합니다.

실행:
  pytest tests/test_backtest.py -v
"""



from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import pytest

import atlas_spot_backtest as bt


# ══════════════════════════════════════════════════════════════
#  헬퍼: 합성 OHLCV
# ══════════════════════════════════════════════════════════════

# backtest_strategy의 start_date/end_date 필터(2021~2030)에 걸리지 않도록
# 1970-01-01 에폭이 아닌 2021-01-01 기준으로 타임스탬프를 생성한다.
_BASE_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _make_uptrend_ohlcv(n: int, start: float = 100.0, drift: float = 0.02,
                         interval_ms: int = 86_400_000) -> list:
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    result = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) * 1.003
        l = min(o, c) * 0.997
        result.append([_BASE_MS + i * interval_ms, o, h, l, c, 1000.0])
    return result


def _make_flat_ohlcv(n: int, price: float = 100.0, interval_ms: int = 86_400_000) -> list:
    return [[_BASE_MS + i * interval_ms, price, price * 1.001, price * 0.999, price, 1000.0]
            for i in range(n)]


def _make_breakout_ohlcv(flat_n: int = 60, flat_price: float = 100.0,
                         breakout_price: float = 130.0, after_n: int = 10) -> list:
    """평탄 구간 → 강한 돌파(거래량 스파이크) → 평탄 유지 (S6 트리거용)."""
    flat = _make_flat_ohlcv(flat_n, price=flat_price)
    breakout = [[_BASE_MS + flat_n * 86_400_000, flat_price, breakout_price, flat_price,
                 breakout_price * 0.98, 5000.0]]
    after = [[_BASE_MS + (flat_n + 1 + j) * 86_400_000, *row[1:]]
             for j, row in enumerate(_make_flat_ohlcv(after_n, price=breakout_price * 0.98))]
    return flat + breakout + after


# ══════════════════════════════════════════════════════════════
#  _since_ms
# ══════════════════════════════════════════════════════════════

class TestSinceMs:
    def test_known_date(self):
        assert bt._since_ms('2021-01-01') == int(
            datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def test_epoch_zero(self):
        assert bt._since_ms('1970-01-01') == 0


# ══════════════════════════════════════════════════════════════
#  _bucket_pnl_r
# ══════════════════════════════════════════════════════════════

class TestBucketPnlR:
    @pytest.mark.parametrize('pnl_r,bucket', [
        (-2.0,  '< -1.5R'),
        (-1.2,  '-1.5~-1R'),
        (-0.7,  '-1~-0.5R'),
        (-0.1,  '-0.5~0R'),
        (0.2,   '0~0.5R'),
        (0.7,   '0.5~1R'),
        (1.2,   '1~1.5R'),
        (1.7,   '1.5~2R'),
        (3.0,   '> 2R'),
    ])
    def test_buckets(self, pnl_r, bucket):
        buckets = defaultdict(int)
        bt._bucket_pnl_r(buckets, pnl_r)
        assert buckets[bucket] == 1

    def test_boundary_zero_goes_to_positive_bucket(self):
        buckets = defaultdict(int)
        bt._bucket_pnl_r(buckets, 0.0)
        assert buckets['0~0.5R'] == 1


# ══════════════════════════════════════════════════════════════
#  _get_slippage
# ══════════════════════════════════════════════════════════════

class TestGetSlippage:
    def test_tier1_symbol(self):
        sym = bt.BT_TIER1_SYMBOLS[0]
        assert bt._get_slippage(sym) == bt.BT_SLIPPAGE_BY_TIER['tier1']

    def test_tier2_symbol(self):
        sym = bt.BT_TIER2_SYMBOLS[0]
        assert bt._get_slippage(sym) == bt.BT_SLIPPAGE_BY_TIER['tier2']

    def test_unknown_symbol_falls_back_to_tier3(self):
        assert bt._get_slippage('SOMERANDOMUSDT') == bt.BT_SLIPPAGE_BY_TIER['tier3']


# ══════════════════════════════════════════════════════════════
#  load_ohlcv_csv
# ══════════════════════════════════════════════════════════════

class TestLoadOhlcvCsv:
    def test_loads_ms_timestamp_column(self, tmp_path):
        csv_path = tmp_path / 'data.csv'
        df = pd.DataFrame({
            'timestamp': [0, 86_400_000],
            'open':  [100.0, 101.0],
            'high':  [102.0, 103.0],
            'low':   [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [10.0, 20.0],
        })
        df.to_csv(csv_path, index=False)
        result = bt.load_ohlcv_csv(str(csv_path))
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 86_400_000
        assert result[0][4] == 101.0  # close

    def test_loads_datetime_string_column(self, tmp_path):
        csv_path = tmp_path / 'data.csv'
        df = pd.DataFrame({
            'open_time': ['2021-01-01', '2021-01-02'],
            'open':  [100.0, 101.0],
            'high':  [102.0, 103.0],
            'low':   [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [10.0, 20.0],
        })
        df.to_csv(csv_path, index=False)
        result = bt.load_ohlcv_csv(str(csv_path))
        assert len(result) == 2
        assert result[1][0] > result[0][0]

    def test_missing_timestamp_column_raises(self, tmp_path):
        csv_path = tmp_path / 'data.csv'
        pd.DataFrame({'open': [1.0], 'high': [1.0], 'low': [1.0],
                      'close': [1.0], 'volume': [1.0]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            bt.load_ohlcv_csv(str(csv_path))


# ══════════════════════════════════════════════════════════════
#  build_regime_map
# ══════════════════════════════════════════════════════════════

class TestBuildRegimeMap:
    def test_insufficient_data_returns_empty(self):
        ohlcv = _make_uptrend_ohlcv(30)
        assert bt.build_regime_map(ohlcv) == {}

    def test_strong_uptrend_classified_trending_up(self):
        ohlcv = _make_uptrend_ohlcv(260, drift=0.02)
        regime_map = bt.build_regime_map(ohlcv)
        assert len(regime_map) > 0
        last_key = str(pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True).date())
        assert regime_map[last_key] == 'TRENDING_UP'

    def test_flat_series_classified_ranging(self):
        ohlcv = _make_flat_ohlcv(260)
        regime_map = bt.build_regime_map(ohlcv)
        last_key = str(pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True).date())
        assert regime_map[last_key] == 'RANGING'


# ══════════════════════════════════════════════════════════════
#  calc_spot_metrics
# ══════════════════════════════════════════════════════════════

class TestCalcSpotMetrics:
    def test_empty_trades(self):
        assert bt.calc_spot_metrics([]) == {'total_trades': 0}

    def test_all_wins_metrics(self):
        trades = [bt.SpotTrade(symbol='BTCUSDT', strategy='S6', direction='LONG',
                                mode='', entry=100.0, exit_px=110.0, pnl_r=2.0,
                                risk_pct=0.02, reason='TP', entry_bar=i, exit_bar=i + 1)
                   for i in range(5)]
        metrics = bt.calc_spot_metrics(trades, initial_equity=10_000)
        assert metrics['total_trades'] == 5
        assert metrics['win_rate'] == 100.0
        assert metrics['profit_factor'] == 999.0  # 손실 없음 → PF 상한
        assert metrics['final_equity'] > 10_000

    def test_mixed_trades_profit_factor(self):
        trades = [
            bt.SpotTrade(symbol='BTCUSDT', strategy='S6', direction='LONG', mode='',
                         entry=100.0, exit_px=110.0, pnl_r=2.0, risk_pct=0.02,
                         reason='TP', entry_bar=0, exit_bar=1),
            bt.SpotTrade(symbol='BTCUSDT', strategy='S6', direction='LONG', mode='',
                         entry=100.0, exit_px=95.0, pnl_r=-1.0, risk_pct=0.02,
                         reason='SL', entry_bar=2, exit_bar=3),
        ]
        metrics = bt.calc_spot_metrics(trades, initial_equity=10_000)
        assert metrics['win_rate'] == 50.0
        assert metrics['profit_factor'] == pytest.approx(2.0)
        assert metrics['sl_count'] == 1
        assert metrics['tp_count'] == 1

    def test_drawdown_tracked_on_losses(self):
        trades = [
            bt.SpotTrade(symbol='BTCUSDT', strategy='S6', direction='LONG', mode='',
                         entry=100.0, exit_px=90.0, pnl_r=-1.0, risk_pct=0.10,
                         reason='SL', entry_bar=i, exit_bar=i + 1)
            for i in range(3)
        ]
        metrics = bt.calc_spot_metrics(trades, initial_equity=10_000)
        assert metrics['max_dd_pct'] > 0
        assert metrics['final_equity'] < 10_000

    def test_cagr_computed_with_dates(self):
        trades = [bt.SpotTrade(symbol='BTCUSDT', strategy='S6', direction='LONG', mode='',
                                entry=100.0, exit_px=120.0, pnl_r=2.0, risk_pct=0.02,
                                reason='TP', entry_bar=0, exit_bar=1)]
        metrics = bt.calc_spot_metrics(trades, initial_equity=10_000,
                                        start_date='2021-01-01', end_date='2022-01-01')
        assert metrics['cagr_pct'] != 0.0


# ══════════════════════════════════════════════════════════════
#  backtest_strategy (S6 Donchian Breakout 사용)
# ══════════════════════════════════════════════════════════════

class TestBacktestStrategy:
    def test_insufficient_bars_returns_empty(self):
        ohlcv = _make_flat_ohlcv(10)
        trades, diag = bt.backtest_strategy('S6', 'BTCUSDT', ohlcv, {},
                                             '2021-01-01', '2030-01-01')
        assert trades == []
        assert diag == {}

    def test_breakout_produces_entry_and_forced_end_exit(self):
        """돌파 신호 발생 후 가격이 SL/TP에 닿지 않으면 데이터 끝에서 강제청산(END)."""
        ohlcv = _make_breakout_ohlcv()
        trades, diag = bt.backtest_strategy('S6', 'BTCUSDT', ohlcv, {},
                                             '2021-01-01', '2030-01-01')
        assert diag.get('entries', 0) >= 1
        assert len(trades) >= 1
        assert trades[-1].reason == 'END'

    def test_regime_block_prevents_entry(self):
        """CRISIS 레짐(빈 허용전략 리스트)에서는 S6 신호가 있어도 진입 차단."""
        ohlcv = _make_breakout_ohlcv()
        # backtest_strategy는 regime_map.get(date_key, 'WEAK_TREND')로 조회하므로
        # defaultdict가 아닌, 모든 봉의 날짜를 명시적으로 매핑한 dict가 필요하다.
        regime_map = {
            str(pd.to_datetime(row[0], unit='ms', utc=True).date()): 'CRISIS'
            for row in ohlcv
        }
        trades, diag = bt.backtest_strategy('S6', 'BTCUSDT', ohlcv, regime_map,
                                             '2021-01-01', '2030-01-01')
        assert diag.get('entries', 0) == 0
        assert diag.get('regime_block', 0) > 0

    def test_sl_exit_realized(self, monkeypatch):
        """진입 직후 급락 → SL 청산, pnl_r은 음수."""
        flat = _make_flat_ohlcv(60, price=100.0)
        breakout = [[_BASE_MS + 60 * 86_400_000, 100.0, 130.0, 100.0, 128.0, 5000.0]]
        crash = [[_BASE_MS + 61 * 86_400_000, 128.0, 128.0, 50.0, 60.0, 1000.0]]
        more = [[_BASE_MS + (62 + j) * 86_400_000, *row[1:]]
                for j, row in enumerate(_make_flat_ohlcv(10, price=60.0))]
        ohlcv = flat + breakout + crash + more
        trades, diag = bt.backtest_strategy('S6', 'BTCUSDT', ohlcv, {},
                                             '2021-01-01', '2030-01-01')
        sl_trades = [t for t in trades if t.reason == 'SL']
        assert len(sl_trades) >= 1
        assert sl_trades[0].pnl_r < 0

    def test_start_date_filters_early_bars(self):
        ohlcv = _make_breakout_ohlcv()
        # 시작일을 데이터 끝보다 한참 뒤로 설정 → 어떤 봉도 처리되지 않음
        future_start = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True) + pd.Timedelta(days=100)
        trades, diag = bt.backtest_strategy(
            'S6', 'BTCUSDT', ohlcv, {},
            str(future_start.date()), '2099-01-01')
        assert diag.get('total_bars', 0) == 0
        assert trades == []

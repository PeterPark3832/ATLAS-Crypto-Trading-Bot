"""
ATLAS — 비용 대비 엣지 가드 / 슬리피지 계측
================================
거래 1건의 기대값 분해:

    포지션 명목가 = 리스크금액 / SL거리%
    왕복비용      = 명목가 × 비용률 = 리스크금액 × (비용률 / SL거리%)
    ∴ 순기대값    = 리스크금액 × [ avg_r − 비용률/SL거리% ]

즉 **SL이 좁게 잡힌 신호일수록 비용이 R을 크게 잠식**하며, 승률과 무관하게
`avg_r ≤ 비용률/SL거리%`이면 그 거래는 구조적으로 마이너스 기대값이다.

실행:
  pytest tests/test_cost_edge.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_main as sm


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(sm, '_tg', lambda msg: sent.append(msg))
    return sent


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'cost.db')
    sm.init_spot_db()


@pytest.fixture(autouse=True)
def _no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_KILL_SWITCH', tmp_path / 'ABSENT')


@pytest.fixture
def _state(monkeypatch):
    fresh = {
        'equity': 10_000.0, 'usdt_balance': 10_000.0, 'peak_equity': 10_000.0,
        'ratchet_alert_tier': 0, 'paused': False, 'dry_run': True,
        'day_pnl': 0.0, 'day_start_eq': 10_000.0, 'daily_loss_alerted': False,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    return fresh


class _Ex:
    def __init__(self, bid=99.95, ask=100.05, fill=100.0):
        self.bid, self.ask, self.fill = bid, ask, fill

    def fetch_ticker(self, sym):
        return {'bid': self.bid, 'ask': self.ask, 'last': 100.0, 'close': 100.0}

    def create_market_buy_order(self, sym, qty):
        return {'average': self.fill, 'price': self.fill, 'filled': qty}

    def create_market_sell_order(self, sym, qty):
        return {'average': self.fill, 'price': self.fill, 'filled': qty}

    def amount_to_precision(self, sym, q):
        return f'{float(q):.8f}'

    def price_to_precision(self, sym, p):
        return f'{float(p):.4f}'

    def fetch_balance(self, params=None):
        return {'free': {'BTC': 1e9}}


@pytest.fixture
def _ex(monkeypatch):
    ex = _Ex()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


def _sig(sl_pct, rr=2.0, entry=100.0):
    return {'sl': entry * (1 - sl_pct), 'tp': entry * (1 + sl_pct * rr),
            'rr': rr, 'exit_type': 'sl_tp', 'max_hold': 10}


def _insert(pnl_r, strategy='S3', n=1, dry=0):
    with sm._db_lock, sm._db_conn() as conn:
        for _ in range(n):
            conn.execute(
                "INSERT INTO spot_trades (strategy,symbol,pnl_usdt,pnl_r,fee_usdt,"
                "reason,entry_ts,exit_ts,dry_run) VALUES (?,?,?,?,0,'TP',"
                "'2026-07-01','2026-07-01',?)",
                (strategy, 'BTCUSDT', pnl_r * 10, pnl_r, dry))


# ══════════════════════════════════════════════════════════════
#  왕복 비용 추정
# ══════════════════════════════════════════════════════════════

class TestCostEstimate:
    def test_uses_live_spread(self, _ex):
        cost, spread = sm._estimate_round_trip_cost('BTC/USDT')
        assert spread == pytest.approx(0.001, rel=1e-3)      # (100.05-99.95)/100
        # 수수료 2회 + 스프레드 1회 + 슬리피지 2회
        expected = sm.BT_SPOT_FEE * 2 + spread + sm.SPOT_ASSUMED_SLIP_PCT * 2
        assert cost == pytest.approx(expected)

    def test_wide_spread_raises_cost(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(bid=99.0, ask=101.0))
        cost, spread = sm._estimate_round_trip_cost('BTC/USDT')
        assert spread == pytest.approx(0.02, rel=1e-3)
        assert cost > 0.02

    def test_falls_back_when_ticker_fails(self, monkeypatch):
        class _Boom:
            def fetch_ticker(self, s):
                raise Exception('rate limit')
        monkeypatch.setattr(sm, '_get_ex', lambda: _Boom())
        cost, spread = sm._estimate_round_trip_cost('BTC/USDT')
        assert spread == sm.SPOT_DEFAULT_SPREAD_PCT
        assert cost > 0


# ══════════════════════════════════════════════════════════════
#  비용 대비 엣지 판정
# ══════════════════════════════════════════════════════════════

class TestCostEdgeGuard:
    def test_tight_stop_rejected(self, _ex):
        """SL 1% → 비용률 0.004/0.01 = 0.40R 잠식 → 차단."""
        ok, why = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 1.0, 100.0)
        assert ok is False and '1R' in why

    def test_normal_stop_allowed(self, _ex):
        """SL 5% → 0.08R → 통과."""
        ok, _ = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is True

    def test_wide_spread_can_reject_normal_stop(self, monkeypatch):
        """스프레드가 벌어진 비유동 심볼은 정상 SL이어도 마이너스 기대값이 된다."""
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(bid=98.5, ask=101.5))
        ok, why = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is False and '스프레드' in why

    def test_proven_negative_edge_rejected(self, _ex):
        """표본이 충분하고 낙관치(avg_r+1SE)조차 비용을 못 넘으면 차단."""
        _insert(0.0, n=40)                       # avg_r=0, 분산 0 → SE=0
        ok, why = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is False and 'avg_r' in why

    def test_positive_edge_allowed(self, _ex):
        _insert(0.5, n=40)
        ok, _ = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is True

    def test_small_sample_not_blocked(self, _ex):
        """표본이 적으면 avg_r 추정 분산이 커 오차단 위험 → 판정하지 않는다."""
        _insert(-0.5, n=sm.SPOT_EDGE_MIN_TRADES - 1)
        ok, _ = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is True

    def test_noisy_sample_gets_benefit_of_doubt(self, _ex):
        """avg_r은 음수지만 분산이 커서 낙관치가 비용을 넘으면 통과시킨다.
        (표본 40건짜리 avg_r 추정으로 전략을 죽이지 않기 위한 안전장치)"""
        _insert(2.0, n=18)
        _insert(-2.0, n=22)                      # avg_r=-0.2, SE≈0.32
        avg, n, se = sm._get_realized_avg_r('S3')
        assert n >= sm.SPOT_EDGE_MIN_TRADES and avg < 0
        assert avg + se > 0.08, f'낙관치 {avg + se:.3f}가 비용 0.08을 넘어야 하는 케이스'
        ok, _ = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is True

    def test_clearly_negative_edge_blocked_despite_variance(self, _ex):
        """분산을 감안해도 명백히 비용을 못 넘으면 차단한다."""
        _insert(0.3, n=18)
        _insert(-0.5, n=22)                      # avg_r≈-0.14, SE≈0.064
        avg, n, se = sm._get_realized_avg_r('S3')
        assert avg + se < 0.08
        assert sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)[0] is False

    def test_dry_run_trades_excluded_from_edge(self, _ex):
        _insert(0.0, n=40, dry=1)                # 가상 거래만 있음
        _, n, _ = sm._get_realized_avg_r('S3')
        assert n == 0
        ok, _ = sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 5.0, 100.0)
        assert ok is True

    def test_guard_blocks_buy_end_to_end(self, _state, _ex):
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.01), 100.0,
                            'TRENDING_UP') is False
        assert sm._load_position('S3', 'BTCUSDT') is None

    def test_guard_allows_normal_buy_end_to_end(self, _state, _ex):
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.05), 100.0,
                            'TRENDING_UP') is True

    def test_zero_sl_dist_is_safe(self, _ex):
        assert sm._cost_edge_ok('S3', 'BTCUSDT', 'BTC/USDT', 0.0, 100.0)[0] is True


# ══════════════════════════════════════════════════════════════
#  슬리피지 계측
# ══════════════════════════════════════════════════════════════

class TestSlippageTracking:
    def test_entry_slippage_recorded(self, _state, monkeypatch):
        """신호가보다 비싸게 체결되면 양수로 기록된다."""
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(fill=100.5))
        _state['dry_run'] = False
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.05), 100.0,
                            'TRENDING_UP') is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['entry_slip_pct'] == pytest.approx(0.005, rel=1e-3)

    def test_round_trip_slippage_recorded(self, _state, monkeypatch):
        """왕복 = 진입(비싸게 삼) + 청산(싸게 팜)."""
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(fill=100.5))
        _state['dry_run'] = False
        sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.05), 100.0, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')

        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(fill=109.0))
        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 110.0, 'TP')

        with sm._db_lock, sm._db_conn() as conn:
            row = dict(conn.execute('SELECT * FROM spot_trades').fetchone())
        # 진입 +0.5%, 청산 (110-109)/110 = +0.909%
        assert row['slip_pct'] == pytest.approx(0.005 + 1.0 / 110, rel=1e-3)

    def test_clean_fill_has_zero_slippage(self, _state, monkeypatch):
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex(fill=100.0))
        _state['dry_run'] = False
        sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.05), 100.0, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['entry_slip_pct'] == pytest.approx(0.0)

    def test_dry_run_records_no_slippage(self, _state, _ex):
        """dry-run은 신호가로 체결됐다고 보므로 슬리피지가 없다."""
        assert sm._spot_buy('S3', 'BTCUSDT', 'BTC/USDT', _sig(0.05), 100.0,
                            'TRENDING_UP') is True
        pos = sm._load_position('S3', 'BTCUSDT')
        assert pos['entry_slip_pct'] == pytest.approx(0.0)

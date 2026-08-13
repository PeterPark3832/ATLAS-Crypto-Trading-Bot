"""
ATLAS — 웹 대시보드(atlas_web_dashboard.py) 단위 테스트
================================
순수 계산 함수(_metrics, _ratchet_scale, _kelly_stats, _rolling_wr,
_dd_curve, _monthly_pnl, _sym_stats, _regime_briefing, _alerts)와
인증 토큰(_check) 로직을 검증합니다.

실행:
  pytest tests/test_web_dashboard.py -v
"""

import time
from datetime import datetime, timedelta, timezone


import pandas as pd
import pytest

import atlas_web_dashboard as wd


# ══════════════════════════════════════════════════════════════
#  헬퍼: trades DataFrame (실제 _trades() 출력 형식)
# ══════════════════════════════════════════════════════════════

def _trades_df(rows, base_ts=None):
    base_ts = base_ts or datetime(2021, 1, 1, tzinfo=timezone.utc)
    data = []
    for i, r in enumerate(rows):
        data.append({
            'strategy': r.get('strategy', 'S6'),
            'symbol': r.get('symbol', 'BTCUSDT'),
            'entry_price': r.get('entry_price', 100.0),
            'exit_price': r.get('exit_price', 110.0),
            'qty': r.get('qty', 1.0),
            'cost_usdt': r.get('cost_usdt', 100.0),
            'pnl_usd': r['pnl_usd'],
            'pnl_pct': r.get('pnl_pct', 0.0),
            'pnl_r': r.get('pnl_r', 0.0),
            'hold_hours': r.get('hold_hours', 1.0),
            'reason': r.get('reason', 'TP'),
            'regime': r.get('regime', 'TRENDING_UP'),
            'entry_ts': base_ts + timedelta(days=i),
            'exit_ts': base_ts + timedelta(days=i, hours=1),
            'fee_usd': r.get('fee_usd', 0.0),
            'direction': 'LONG',
            'leverage': 1,
        })
    df = pd.DataFrame(data)
    df['net_pnl'] = df['pnl_usd'] - df['fee_usd']
    return df


# ══════════════════════════════════════════════════════════════
#  _check (인증 토큰)
# ══════════════════════════════════════════════════════════════

class TestCheck:
    def test_valid_token_passes(self, monkeypatch):
        monkeypatch.setattr(wd, '_tokens', {'tok1': time.time() + 100})
        assert wd._check('tok1') is True

    def test_unknown_token_fails(self, monkeypatch):
        monkeypatch.setattr(wd, '_tokens', {})
        assert wd._check('nope') is False

    def test_expired_token_fails_and_is_removed(self, monkeypatch):
        tokens = {'tok1': time.time() - 10}
        monkeypatch.setattr(wd, '_tokens', tokens)
        assert wd._check('tok1') is False
        assert 'tok1' not in tokens


# ══════════════════════════════════════════════════════════════
#  _metrics
# ══════════════════════════════════════════════════════════════

class TestMetrics:
    def test_empty_df_returns_defaults(self, monkeypatch):
        monkeypatch.setattr(wd, 'INITIAL_CAPITAL', 1000.0)
        m = wd._metrics(pd.DataFrame())
        assert m['total'] == 0
        assert m['equity'] == 1000.0

    def test_mixed_trades_metrics(self, monkeypatch):
        monkeypatch.setattr(wd, 'INITIAL_CAPITAL', 1000.0)
        df = _trades_df([
            {'pnl_usd': 100.0, 'fee_usd': 1.0, 'pnl_r': 2.0},
            {'pnl_usd': -50.0, 'fee_usd': 1.0, 'pnl_r': -1.0},
        ])
        m = wd._metrics(df)
        assert m['total'] == 2
        assert m['wins'] == 1
        assert m['wr'] == 50.0
        assert m['gross_pnl'] == pytest.approx(50.0)
        assert m['total_fee'] == pytest.approx(2.0)
        assert m['total_pnl'] == pytest.approx(48.0)  # net = gross - fee
        assert m['equity'] == pytest.approx(1048.0)

    def test_no_losses_pf_is_zero_not_inf(self, monkeypatch):
        """gl==0일 때 PF가 weekly_report와 달리 0으로 처리됨 (inf 아님) — 현재 동작 고정."""
        monkeypatch.setattr(wd, 'INITIAL_CAPITAL', 1000.0)
        df = _trades_df([{'pnl_usd': 10.0, 'pnl_r': 1.0}])
        m = wd._metrics(df)
        assert m['pf'] == 0


# ══════════════════════════════════════════════════════════════
#  _ratchet_scale
# ══════════════════════════════════════════════════════════════

class TestRatchetScale:
    @pytest.mark.parametrize('mdd,scale', [(0.0, 1.0), (4.9, 1.0), (5.0, 0.7),
                                            (7.9, 0.7), (8.0, 0.4), (15.0, 0.4)])
    def test_thresholds(self, mdd, scale):
        assert wd._ratchet_scale(mdd)['scale'] == scale


# ══════════════════════════════════════════════════════════════
#  _kelly_stats
# ══════════════════════════════════════════════════════════════

class TestKellyStats:
    def test_empty_returns_empty_list(self):
        assert wd._kelly_stats(pd.DataFrame()) == []

    def test_groups_by_strategy_with_full_kelly(self):
        df = _trades_df([
            {'strategy': 'S6', 'pnl_usd': 100.0, 'pnl_r': 2.0},
            {'strategy': 'S6', 'pnl_usd': 100.0, 'pnl_r': 2.0},
            {'strategy': 'S6', 'pnl_usd': -50.0, 'pnl_r': -1.0},
        ])
        rows = wd._kelly_stats(df)
        assert len(rows) == 1
        r = rows[0]
        assert r['trades'] == 3
        assert r['wr'] == pytest.approx(200 / 3, abs=0.1)
        assert r['rr'] == pytest.approx(2.0)
        assert r['reliable'] is False  # trades < 20

    def test_sorted_by_trade_count_descending(self):
        df = _trades_df(
            [{'strategy': 'S3', 'pnl_usd': 10.0, 'pnl_r': 1.0}] +
            [{'strategy': 'S6', 'pnl_usd': 10.0, 'pnl_r': 1.0} for _ in range(3)]
        )
        rows = wd._kelly_stats(df)
        assert rows[0]['trades'] == 3


# ══════════════════════════════════════════════════════════════
#  _rolling_wr
# ══════════════════════════════════════════════════════════════

class TestRollingWr:
    def test_empty_or_single_returns_empty(self):
        assert wd._rolling_wr(pd.DataFrame()) == []
        assert wd._rolling_wr(_trades_df([{'pnl_usd': 1.0}])) == []

    def test_window_caps_at_n(self):
        df = _trades_df([{'pnl_usd': 10.0} for _ in range(15)])
        result = wd._rolling_wr(df, n=5)
        assert len(result) == 15
        assert result[-1]['n'] == 5
        assert result[0]['n'] == 1

    def test_wr_reflects_recent_window(self):
        rows = [{'pnl_usd': -10.0} for _ in range(3)] + [{'pnl_usd': 10.0} for _ in range(3)]
        df = _trades_df(rows)
        result = wd._rolling_wr(df, n=3)
        assert result[-1]['wr'] == 100.0  # 마지막 3건 전부 승


# ══════════════════════════════════════════════════════════════
#  _dd_curve / _monthly_pnl / _sym_stats
# ══════════════════════════════════════════════════════════════

class TestDdCurve:
    def test_empty_returns_empty(self):
        assert wd._dd_curve(pd.DataFrame()) == []

    def test_drawdown_after_loss(self, monkeypatch):
        monkeypatch.setattr(wd, 'INITIAL_CAPITAL', 1000.0)
        df = _trades_df([{'pnl_usd': 100.0}, {'pnl_usd': -150.0}, {'pnl_usd': 20.0}])
        result = wd._dd_curve(df)
        assert result[0]['dd'] == 0.0
        assert result[1]['dd'] == pytest.approx(15.0)


class TestMonthlyPnl:
    def test_empty_returns_empty(self):
        assert wd._monthly_pnl(pd.DataFrame()) == []

    def test_groups_by_month(self):
        df = _trades_df([{'pnl_usd': 10.0}, {'pnl_usd': 20.0}],
                         base_ts=datetime(2021, 1, 30, tzinfo=timezone.utc))
        result = wd._monthly_pnl(df)
        assert len(result) >= 1
        total_pnl = sum(r['pnl'] for r in result)
        assert total_pnl == pytest.approx(30.0)


class TestSymStats:
    def test_empty_returns_empty(self):
        assert wd._sym_stats(pd.DataFrame()) == []

    def test_groups_by_symbol_strips_usdt(self):
        df = _trades_df([
            {'symbol': 'BTCUSDT', 'pnl_usd': 50.0},
            {'symbol': 'BTCUSDT', 'pnl_usd': -20.0},
            {'symbol': 'ETHUSDT', 'pnl_usd': 5.0},
        ])
        rows = wd._sym_stats(df)
        btc = next(r for r in rows if r['symbol'] == 'BTC')
        assert btc['trades'] == 2
        assert btc['best'] == 50.0
        assert btc['worst'] == -20.0
        # pnl 절대값 기준 내림차순 정렬 → BTC(30) 가 ETH(5) 보다 먼저
        assert rows[0]['symbol'] == 'BTC'


# ══════════════════════════════════════════════════════════════
#  _regime_briefing
# ══════════════════════════════════════════════════════════════

class TestRegimeBriefing:
    def test_unknown_regime_returns_waiting_state(self):
        result = wd._regime_briefing(None, True, 0)
        assert result['tone'] == 'neutral'
        assert result['active_strats'] == []

    def test_bot_dead_overrides_everything(self, monkeypatch):
        monkeypatch.setattr(wd, '_parse_regime_metrics',
                             lambda: {'adx_1d': None, 'adx_4h': None, 'atr_pct': None,
                                      'btc': None, 'ema200': None})
        result = wd._regime_briefing('TRENDING_UP', False, 0)
        assert result['tone'] == 'danger'
        assert '중단' in result['headline']

    def test_open_position_reports_active_tone(self, monkeypatch):
        monkeypatch.setattr(wd, '_parse_regime_metrics',
                             lambda: {'adx_1d': None, 'adx_4h': None, 'atr_pct': None,
                                      'btc': None, 'ema200': None})
        result = wd._regime_briefing('TRENDING_UP', True, 2)
        assert result['tone'] == 'active'
        assert '2개' in result['headline']

    def test_crisis_regime_blocks_entry(self, monkeypatch):
        monkeypatch.setattr(wd, '_parse_regime_metrics',
                             lambda: {'adx_1d': None, 'adx_4h': None, 'atr_pct': 9.0,
                                      'btc': None, 'ema200': None})
        result = wd._regime_briefing('CRISIS', True, 0)
        assert result['tone'] == 'danger'
        assert result['active_strats'] == []

    def test_ema_gap_computed_when_metrics_available(self, monkeypatch):
        monkeypatch.setattr(wd, '_parse_regime_metrics',
                             lambda: {'adx_1d': 30.0, 'adx_4h': 28.0, 'atr_pct': 2.0,
                                      'btc': 110.0, 'ema200': 100.0})
        result = wd._regime_briefing('TRENDING_UP', True, 0)
        assert result['ema_gap'] == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════
#  _alerts
# ══════════════════════════════════════════════════════════════

class TestAlerts:
    def test_no_alerts_when_healthy(self):
        m = {'mdd_pct': 2.0, 'total': 5, 'wr': 60, 'pf': 2.0}
        assert wd._alerts(m, True) == []

    def test_bot_dead_is_critical(self):
        m = {'mdd_pct': 0.0, 'total': 0, 'wr': 0, 'pf': 0}
        alerts = wd._alerts(m, False)
        assert any(a['level'] == 'critical' for a in alerts)

    def test_mdd_over_15_is_critical(self):
        m = {'mdd_pct': 16.0, 'total': 0, 'wr': 0, 'pf': 0}
        alerts = wd._alerts(m, True)
        assert alerts[0]['level'] == 'critical'

    def test_mdd_over_10_is_warn(self):
        m = {'mdd_pct': 11.0, 'total': 0, 'wr': 0, 'pf': 0}
        alerts = wd._alerts(m, True)
        assert alerts[0]['level'] == 'warn'

    def test_low_winrate_with_enough_trades_warns(self):
        m = {'mdd_pct': 0.0, 'total': 10, 'wr': 20, 'pf': 2.0}
        alerts = wd._alerts(m, True)
        assert any('승률' in a['msg'] for a in alerts)

    def test_low_winrate_ignored_with_few_trades(self):
        m = {'mdd_pct': 0.0, 'total': 5, 'wr': 20, 'pf': 2.0}
        assert wd._alerts(m, True) == []

    def test_pf_below_1_with_enough_trades_warns(self):
        m = {'mdd_pct': 0.0, 'total': 10, 'wr': 60, 'pf': 0.8}
        alerts = wd._alerts(m, True)
        assert any('PF' in a['msg'] for a in alerts)

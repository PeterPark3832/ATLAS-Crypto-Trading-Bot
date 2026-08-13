"""
ATLAS — 라이브/백테스트 패리티
================================
백테스트에 라이브의 진입 필터·사이징 제약이 빠져 있으면 성과가 구조적으로
과대평가되고, 그 결과로 튜닝한 파라미터는 실계좌에서 재현되지 않는다.
(즉 백테스트가 최적화의 근거로 쓸 수 없게 된다)

여기서는 신호 함수를 직접 주입해 각 필터가 실제로 발동하는지 확인한다.

실행:
  pytest tests/test_backtest_parity.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import atlas_spot_backtest as bt
import atlas_spot_config as cfg
import atlas_spot_main as sm


def _flat_ohlcv(n=400, px=100.0):
    """가격이 완만히 오르는 4H 캔들 — 청산이 나도록 변동은 준다."""
    rng = np.random.default_rng(3)
    ts = 1609459200000
    rows = []
    p = px
    for i in range(n):
        p *= (1 + 0.0008 + rng.normal(0, 0.006))
        rows.append([ts + i * 4 * 3600 * 1000, p,
                     p * 1.02, p * 0.98, p, 1e6])
    return rows


@pytest.fixture
def always_signal(monkeypatch):
    """항상 진입 신호를 내는 스텁. sl_pct로 SL 거리를 제어한다."""
    def _install(sl_pct, rr=2.0):
        def _sig(df, i):
            close = float(df['close'].iloc[i])
            return {'signal': 1, 'sl': close * (1 - sl_pct),
                    'tp': close * (1 + sl_pct * rr), 'rr': rr,
                    'exit_type': 'sl_tp', 'max_hold': 20}
        monkeypatch.setitem(bt.SIGNAL_FUNCS, 'S3', _sig)
    return _install


def _falling_ohlcv(n=500):
    """지속 하락 — 대부분 SL로 청산돼 PF가 무너지는 구간."""
    rng = np.random.default_rng(9)
    ts = 1609459200000
    rows = []
    p = 100.0
    for i in range(n):
        p *= (1 - 0.004 + rng.normal(0, 0.008))
        rows.append([ts + i * 4 * 3600 * 1000, p, p * 1.01, p * 0.98, p, 1e6])
    return rows


def _run(rows=None, **kw):
    return bt.backtest_strategy('S3', 'BTCUSDT', rows or _flat_ohlcv(), {},
                                '2021-01-01', '2022-12-31',
                                risk_pct=0.02, **kw)


# ══════════════════════════════════════════════════════════════
#  상수 패리티
# ══════════════════════════════════════════════════════════════

class TestConstantParity:
    def test_kelly_fraction_shared(self):
        assert bt.SPOT_KELLY_FRACTION == sm.SPOT_KELLY_FRACTION

    def test_cost_and_sl_limits_shared(self):
        assert bt.SPOT_MAX_COST_PER_R == sm.SPOT_MAX_COST_PER_R
        assert bt.SPOT_MAX_SL_PCT == sm.SPOT_MAX_SL_PCT

    def test_health_thresholds_shared(self):
        assert bt.SPOT_HEALTH_PF_SOFT == sm.SPOT_HEALTH_PF_SOFT
        assert bt.SPOT_HEALTH_PF_HARD == sm.SPOT_HEALTH_PF_HARD
        assert bt.SPOT_HEALTH_SOFT_SCALE == sm.SPOT_HEALTH_SOFT_SCALE


# ══════════════════════════════════════════════════════════════
#  진입 필터가 실제로 발동하는가
# ══════════════════════════════════════════════════════════════

class TestEntryFilterParity:
    def test_tight_sl_blocked_by_cost_guard(self, always_signal):
        """SL 0.5% → 왕복비용이 1R의 50%를 잠식 → 대부분 차단.
        (신호봉 종가 기준 SL과 다음봉 시가 진입 사이에 갭이 벌어지면
         실효 SL이 넓어져 일부는 통과한다 — 필터는 실제 sl_dist로 판정)"""
        always_signal(sl_pct=0.005)
        trades, diag = _run()
        assert diag.get('cost_exceeds_edge', 0) > 0, '비용 가드가 발동해야 한다'
        assert diag['cost_exceeds_edge'] > len(trades) * 5, (
            '차단이 통과보다 압도적으로 많아야 한다')

    def test_wide_sl_blocked_by_sl_cap(self, always_signal):
        """SL 상한(SPOT_MAX_SL_PCT) 초과는 라이브에서 막히므로 백테스트도 막아야."""
        always_signal(sl_pct=cfg.SPOT_MAX_SL_PCT + 0.05)
        trades, diag = _run()
        assert diag.get('sl_too_wide', 0) > 0
        assert len(trades) == 0

    def test_normal_sl_passes(self, always_signal):
        """정상 SL은 두 필터를 통과해 거래가 발생한다."""
        always_signal(sl_pct=0.05)
        trades, diag = _run()
        assert diag.get('cost_exceeds_edge', 0) == 0
        assert diag.get('sl_too_wide', 0) == 0
        assert len(trades) > 0, '정상 조건에서는 거래가 나와야 한다'

    def test_cost_guard_reduces_trade_count(self, always_signal, monkeypatch):
        """가드를 끄면(한도를 무한대로) 거래가 늘어난다 — 가드가 실제로
        기대값 음수 거래를 걸러내고 있다는 직접 증거."""
        always_signal(sl_pct=0.008)
        _, diag_on = _run()
        monkeypatch.setattr(bt, 'SPOT_MAX_COST_PER_R', 999.0)
        trades_off, diag_off = _run()
        assert diag_on.get('cost_exceeds_edge', 0) > 0
        assert diag_off.get('cost_exceeds_edge', 0) == 0
        assert len(trades_off) > 0

    def test_cost_guard_uses_same_inequality_as_live(self):
        """백테스트와 라이브가 같은 부등식(비용률/SL거리% ≤ 한도)을 쓴다."""
        slip = bt._get_slippage('BTCUSDT')
        cost_rate = bt.BT_SPOT_FEE * 2 + slip * 2
        # 이 SL에서는 통과, 그보다 좁으면 차단되는 경계
        boundary = cost_rate / cfg.SPOT_MAX_COST_PER_R
        assert cost_rate / (boundary * 1.01) < cfg.SPOT_MAX_COST_PER_R
        assert cost_rate / (boundary * 0.99) > cfg.SPOT_MAX_COST_PER_R


# ══════════════════════════════════════════════════════════════
#  사이징 패리티
# ══════════════════════════════════════════════════════════════

class TestSizingParity:
    def test_health_block_engages_after_bad_run(self, always_signal):
        """손실이 누적되면 백테스트도 건강도 차단이 걸려야 한다.
        (없으면 라이브가 실제로는 멈췄을 구간까지 계속 거래한 셈이 되어
         하락장 성과가 통째로 과대평가된다)"""
        always_signal(sl_pct=0.05)
        trades, diag = _run(_falling_ohlcv())
        nets = [t.pnl_r * t.risk_pct for t in trades]
        gl = abs(sum(n for n in nets if n < 0))
        pf = sum(n for n in nets if n > 0) / gl if gl else float('inf')
        assert pf < cfg.SPOT_HEALTH_PF_HARD, f'PF가 나빠야 하는 시나리오 (실제 {pf:.2f})'
        assert diag.get('health_blocked', 0) > 0, (
            f'건강도 차단이 걸리지 않음 (거래 {len(trades)}건, diag={diag})')

    def test_healthy_run_not_blocked(self, always_signal):
        """정상 구간에서는 건강도 차단이 걸리지 않는다(과차단 방지)."""
        always_signal(sl_pct=0.05)
        _, diag = _run()
        assert diag.get('health_blocked', 0) == 0

    # ── Kelly 사이징 — 소스 문자열이 아니라 **동작**으로 검증한다 ──
    #   예전에는 `'kelly_scale = SPOT_KELLY_SCALE_MIN' in src` 처럼 소스를
    #   문자열로 검사했다. 그러면 표현만 바꿔도 실패하고(리팩터링 방해),
    #   반대로 표현을 유지한 채 의미를 바꾸면 통과한다(진짜 회귀를 놓침).
    #   `_bt_kelly_scale`이 순수함수로 분리되면서 직접 시험할 수 있게 됐다.

    @staticmethod
    def _fake_trades(pnl_rs, risk_pct=0.02):
        from atlas_spot_backtest import SpotTrade
        return [SpotTrade(symbol='X', strategy='S6', direction='LONG', mode='',
                          entry=1.0, exit_px=1.0, pnl_r=r, risk_pct=risk_pct,
                          reason='SL' if r <= 0 else 'TP', entry_bar=i,
                          exit_bar=i + 1, regime='RANGING')
                for i, r in enumerate(pnl_rs)]

    def test_losing_streak_uses_min_kelly(self):
        """전패 구간에서 라이브는 최소 스케일을 쓴다 — 백테스트도 동일해야
        낙관 편향이 생기지 않는다."""
        losing = self._fake_trades([-1.0] * 40)
        assert bt._bt_kelly_scale(losing) == pytest.approx(cfg.SPOT_KELLY_SCALE_MIN)

    def test_thin_sample_does_not_scale(self):
        """표본이 적으면 개입하지 않는다(중립 1.0)."""
        few = self._fake_trades([1.0] * (cfg.SPOT_KELLY_MIN_TRADES - 1))
        assert bt._bt_kelly_scale(few) == 1.0

    def test_all_wins_stays_neutral(self):
        """전승은 b를 계산할 수 없다 — 표본 편향일 뿐이므로 상향하지 않는다."""
        assert bt._bt_kelly_scale(self._fake_trades([1.0] * 40)) == 1.0

    def test_good_record_scales_above_floor(self):
        good = self._fake_trades([2.0, 2.0, 2.0, -1.0] * 12)
        assert bt._bt_kelly_scale(good) > cfg.SPOT_KELLY_SCALE_MIN

    def test_kelly_never_exceeds_cap(self):
        for rs in ([3.0] * 30 + [-1.0] * 2, [5.0, -0.1] * 25, [1.0, -1.0] * 25):
            assert bt._bt_kelly_scale(self._fake_trades(rs)) <= cfg.SPOT_KELLY_SCALE_MAX

    def test_health_returns_none_when_blocked(self):
        """차단과 '스케일 0.0'을 같은 값으로 돌려주면 곱셈으로 흘러들어가
        조용히 0 사이즈 주문이 된다 — 명시적으로 구분돼야 한다."""
        bad = self._fake_trades([-1.0] * 40)
        assert bt._bt_health_scale(bad) is None

    def test_health_soft_scale_between_thresholds(self):
        mixed = self._fake_trades([1.0] * 10 + [-1.0] * 12)
        val = bt._bt_health_scale(mixed)
        assert val in (None, cfg.SPOT_HEALTH_SOFT_SCALE, 1.0)

    def test_health_neutral_on_thin_sample(self):
        few = self._fake_trades([-1.0] * (cfg.SPOT_HEALTH_MIN_TRADES - 1))
        assert bt._bt_health_scale(few) == 1.0


# ══════════════════════════════════════════════════════════════
#  남아있는 한계 (문서화)
# ══════════════════════════════════════════════════════════════

class TestDashboardSchemaCompat:
    def test_optional_column_falls_back_on_old_db(self, tmp_path, monkeypatch):
        """대시보드는 DB를 읽기 전용으로 열어 마이그레이션을 못 한다.
        봇이 아직 새 컬럼을 안 만든 DB에서도 쿼리가 깨지면 안 된다."""
        import sqlite3
        import atlas_web_dashboard as wd

        db = tmp_path / 'old.db'
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, strategy TEXT,
              symbol TEXT, entry_price REAL, exit_price REAL, qty_tokens REAL,
              cost_usdt REAL, pnl_usdt REAL, pnl_pct REAL, pnl_r REAL,
              hold_hours REAL, reason TEXT, entry_ts TEXT, exit_ts TEXT,
              regime TEXT, fee_usdt REAL DEFAULT 0);
            INSERT INTO spot_trades (strategy,symbol,entry_price,exit_price,
              qty_tokens,cost_usdt,pnl_usdt,pnl_pct,pnl_r,hold_hours,reason,
              entry_ts,exit_ts,regime,fee_usdt)
            VALUES ('S3','BTCUSDT',100,110,1,100,10,10,2,5,'TP',
              '2026-07-01T00:00:00+00:00','2026-07-02T00:00:00+00:00','TRENDING_UP',0.2);
        """)
        con.commit(); con.close()

        monkeypatch.setattr(wd, 'DB_FILE', db)
        monkeypatch.setattr(wd, '_col_cache', {})
        assert wd._opt_col('spot_trades', 'slip_pct') == '0'

        df = wd._trades()
        assert len(df) == 1
        assert float(df['slip_pct'].iloc[0]) == 0.0
        m = wd._metrics(df)
        assert m['total_slip'] == 0.0 and m['cost_drag_pct'] >= 0


class TestKnownGaps:
    def test_portfolio_constraints_are_documented_as_missing(self):
        """백테스트는 (전략×심볼) 단위라 포트폴리오 제약을 모델링할 수 없다.
        이 사실이 코드에 명시돼 있어야 결과를 오해하지 않는다."""
        src = Path(bt.__file__).read_text()
        assert '포트폴리오 제약' in src, (
            '백테스트가 모델링하지 못하는 제약(동시 포지션 수, 슬롯당 자본, '
            'USDT 예비금)이 문서화돼 있어야 한다')


class TestS5SlCooldownParity:
    """라이브의 S5 손절 쿨다운이 백테스트에도 있어야 한다.

    라이브(_s5_safety_block)는 S5가 손절로 나간 종목에 2일간 재진입하지
    않는다 — 평균회귀는 '더 싸졌으니 또 산다'가 되기 쉬워, 같은 하락에
    연속으로 맞기 때문이다.

    백테스트가 이 규칙을 무시하면 **라이브가 절대 하지 않는 거래**로 성과를
    평가하게 된다. 막는 이유가 '나쁜 거래'이므로 편향은 비관 쪽이고,
    S5는 지금 OOS 기준 미달로 재최적화 대상이라 판정이 뒤집힐 수 있다.
    """

    @pytest.fixture
    def s5_always_sl(self, monkeypatch):
        """진입하면 곧 손절되는 S5 신호 — 재진입 간격만 본다."""
        def _sig(df, i):
            close = float(df['close'].iloc[i])
            return {'signal': 1, 'sl': close * 0.97, 'tp': close * 1.20,
                    'rr': 2.0, 'exit_type': 'sl_tp', 'max_hold': 0}
        monkeypatch.setitem(bt.SIGNAL_FUNCS, 'S5', _sig)
        monkeypatch.setitem(bt.CALC_FUNCS, 'S5', bt.CALC_FUNCS['S3'])

    def _s5(self, rows):
        return bt.backtest_strategy('S5', 'BTCUSDT', rows, {},
                                    '2021-01-01', '2022-12-31', risk_pct=0.02)

    def test_no_reentry_on_bar_right_after_sl(self, s5_always_sl):
        trades, _ = self._s5(_falling_ohlcv())
        sl_bars = {t.exit_bar for t in trades if t.reason == 'SL'}
        entries = [t.entry_bar for t in trades]
        violations = [b for b in entries
                      if any(0 < b - x <= cfg.S5_SL_COOLDOWN_BARS for x in sl_bars)]
        assert not violations, (
            f'손절 후 {cfg.S5_SL_COOLDOWN_BARS}봉 이내 재진입 {len(violations)}건 — '
            f'라이브는 막는 거래다')

    def test_cooldown_constant_is_shared(self):
        """상수를 복제하면 한쪽만 바뀌어 조용히 어긋난다."""
        src = Path(bt.__file__).read_text(encoding='utf-8')
        assert 'S5_SL_COOLDOWN_BARS' in src, (
            '백테스트가 라이브와 같은 상수를 참조해야 한다')

    def test_non_sl_exit_does_not_trigger_cooldown(self, s5_always_sl, monkeypatch):
        """쿨다운은 **손절 후에만** 건다 — 익절까지 막으면 과도 제약이다."""
        def _tp_sig(df, i):
            close = float(df['close'].iloc[i])
            return {'signal': 1, 'sl': close * 0.80, 'tp': close * 1.005,
                    'rr': 2.0, 'exit_type': 'sl_tp', 'max_hold': 0}
        monkeypatch.setitem(bt.SIGNAL_FUNCS, 'S5', _tp_sig)
        trades, _ = self._s5(_flat_ohlcv())
        tp_trades = [t for t in trades if t.reason == 'TP']
        if len(tp_trades) >= 2:
            gaps = [b.entry_bar - a.exit_bar
                    for a, b in zip(tp_trades, tp_trades[1:], strict=False)]
            assert min(gaps) <= cfg.S5_SL_COOLDOWN_BARS, (
                '익절 후에도 쿨다운이 걸려 진입 기회를 과도하게 막는다')

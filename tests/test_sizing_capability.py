"""
ATLAS — 사이징 실행 가능성 진단
================================
포지션 크기는 여러 스케일의 **곱**이다:

    리스크 = 기본(2%) × Kelly × 래칫 × 레짐 × 건강도

1보다 작은 값이 겹치면 주문금액이 거래소 최소치($5) 아래로 내려간다.
그 조합은 신호가 나와도 **영원히 체결되지 않는데**, 로그 한 줄만 남아
운영자는 "전략이 돌고 있다"고 믿는다. 소액 계좌에서 하락장 커버리지가
통째로 죽는 것이 대표적이다(S4는 TRENDING_DOWN 유일 전략).

이 조용한 실패를 기동 시 명시적 경고로 바꾼다.

실행:
  pytest tests/test_sizing_capability.py -v
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
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'sizing.db')
    sm.init_spot_db()


def _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=5.0, dry=0):
    """sl_dist = pnl_usdt / (pnl_r × qty) 로 역산되도록 넣는다."""
    with sm._db_lock, sm._db_conn() as conn:
        conn.execute(
            "INSERT INTO spot_trades (strategy,symbol,entry_price,qty_tokens,"
            "pnl_usdt,pnl_r,fee_usdt,reason,entry_ts,exit_ts,dry_run) "
            "VALUES ('S3','BTCUSDT',?,?,?,?,0,'TP','2026-07-01','2026-07-01',?)",
            (entry, qty, pnl_usdt, pnl_r, dry))


# ══════════════════════════════════════════════════════════════
#  전형 SL 추정
# ══════════════════════════════════════════════════════════════

class TestTypicalSl:
    def test_default_without_samples(self):
        assert sm._typical_sl_pct() == pytest.approx(0.05)

    def test_derived_from_real_trades(self):
        # sl_dist = 5 / (1 × 1) = 5 → entry 100 대비 5%
        for _ in range(12):
            _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=5.0)
        assert sm._typical_sl_pct() == pytest.approx(0.05, rel=1e-6)

    def test_uses_median_not_mean(self):
        """이상치 하나가 진단을 왜곡하면 안 된다."""
        for _ in range(12):
            _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=3.0)   # 3%
        _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=40.0)      # 40% 이상치
        assert sm._typical_sl_pct() == pytest.approx(0.03, rel=1e-6)

    def test_small_sample_falls_back(self):
        for _ in range(5):
            _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=2.0)
        assert sm._typical_sl_pct() == pytest.approx(0.05)

    def test_dry_run_excluded(self):
        for _ in range(20):
            _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=1.0, dry=1)
        assert sm._typical_sl_pct() == pytest.approx(0.05)

    def test_absurd_values_filtered(self):
        for _ in range(20):
            _insert(entry=100.0, qty=1.0, pnl_r=1.0, pnl_usdt=90.0)   # SL 90% → 제외
        assert sm._typical_sl_pct() == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════
#  조합별 진입 가능성
# ══════════════════════════════════════════════════════════════

class TestSizingCapability:
    def test_covers_every_regime_strategy_pair(self):
        rows = sm._diagnose_sizing_capability(1000.0)
        pairs = {(r['strategy'], r['regime']) for r in rows}
        expected = {(s, reg) for reg, ss in sm.REGIME_STRATEGY_MAP.items() for s in ss}
        assert pairs == expected

    def test_small_account_has_dead_combinations(self):
        """$134 계좌에서 하락장 전략은 최소주문 미달로 실행 불가."""
        rows = sm._diagnose_sizing_capability(134.0)
        dead = [r for r in rows if not r['tradable']]
        assert any(r['strategy'] == 'S4' and r['regime'] == 'TRENDING_DOWN'
                   for r in dead), '하락장 유일 전략의 실행 불가를 잡아야 한다'

    def test_larger_account_has_none(self):
        assert all(r['tradable'] for r in sm._diagnose_sizing_capability(1000.0))

    def test_effective_risk_is_far_below_configured(self):
        """스케일이 곱해져 설정값보다 훨씬 낮아지는 것을 드러내야 한다."""
        rows = sm._diagnose_sizing_capability(1000.0)
        s3 = next(r for r in rows if r['strategy'] == 'S3')
        # S3는 WEAK_TREND 전용 → 항상 레짐 0.5배가 곱해진다
        assert s3['regime_scale'] == sm.WEAK_TREND_RISK_SCALE
        assert s3['risk_pct'] < sm.SPOT_BASE_RISK_PCT * 0.25

    def test_alloc_cap_applied(self):
        rows = sm._diagnose_sizing_capability(1_000_000.0)
        assert all(r['cost_usdt'] <= 1_000_000.0 * sm.SPOT_MAX_ALLOC_PCT + 1e-6
                   for r in rows)

    def test_zero_equity_safe(self):
        rows = sm._diagnose_sizing_capability(0.0)
        assert rows and all(not r['tradable'] for r in rows)


# ══════════════════════════════════════════════════════════════
#  보고
# ══════════════════════════════════════════════════════════════

class TestReport:
    def test_warns_on_dead_combination(self, _no_telegram):
        sm._report_sizing_capability(134.0)
        assert any('진입 불가' in m for m in _no_telegram)
        assert any('S4 / TRENDING_DOWN' in m for m in _no_telegram)

    def test_silent_when_all_tradable(self, _no_telegram):
        sm._report_sizing_capability(1000.0)
        assert not [m for m in _no_telegram if '진입 불가' in m]

    def test_failure_is_non_fatal(self, monkeypatch, _no_telegram):
        monkeypatch.setattr(sm, '_diagnose_sizing_capability',
                            lambda eq: (_ for _ in ()).throw(Exception('boom')))
        sm._report_sizing_capability(134.0)      # 예외가 새어나오면 기동이 죽는다
        assert not _no_telegram

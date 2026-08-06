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

class TestStrategyValidation:
    """`--strategies`는 아무 문자열이나 받는데 검증이 없어서, 오타 하나로
    봇이 **아무것도 거래하지 않으면서 "정상 기동"을 보고**하는 상태가 됐다.
    죽는 방식이 세 가지라 각각 구분해 알려야 한다."""

    def test_valid_strategies_pass(self):
        ok, problems = sm.validate_active_strategies(['S3', 'S4', 'S5', 'S6'])
        assert ok == ['S3', 'S4', 'S5', 'S6'] and problems == []

    def test_unknown_name_is_flagged(self):
        ok, problems = sm.validate_active_strategies(['S9'])
        assert ok == [] and len(problems) == 1
        assert '타임프레임' in problems[0]

    def test_lowercase_is_flagged(self):
        """소문자는 어느 루프에도 배정되지 않아 완전히 침묵한다."""
        ok, problems = sm.validate_active_strategies(['s3'])
        assert ok == [] and problems

    def test_strategy_without_regime_is_flagged(self):
        """레짐 배정이 없으면 매 봉 차단되어 진입이 0건이 된다."""
        ok, problems = sm.validate_active_strategies(['S7'])
        assert ok == []
        assert '레짐' in problems[0]

    def test_partial_valid_keeps_good_ones(self):
        ok, problems = sm.validate_active_strategies(['S3', 'S9', 'S6'])
        assert ok == ['S3', 'S6'] and len(problems) == 1

    def test_empty_input(self):
        assert sm.validate_active_strategies([]) == ([], [])

    def test_every_live_strategy_is_runnable(self):
        """설정에 선언된 전략은 전부 실제로 돌 수 있어야 한다.
        (이 단언이 깨지면 config 편집이 조용히 전략을 죽인 것)"""
        ok, problems = sm.validate_active_strategies(list(sm.LIVE_STRATEGIES))
        assert problems == [], f'선언됐지만 실행 불가: {problems}'
        assert sorted(ok) == sorted(sm.LIVE_STRATEGIES)

    def test_default_active_matches_live(self):
        from atlas_spot_config import DEFAULT_ACTIVE_STRATEGIES, LIVE_STRATEGIES
        assert sorted(DEFAULT_ACTIVE_STRATEGIES) == sorted(LIVE_STRATEGIES), (
            '기본 활성 목록과 레짐 배정 목록이 어긋나면 일부 전략이 놀게 된다')


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


class TestRegimeIdleDetection:
    """'지금 이 레짐에서는 아무것도 못 산다'를 알린다.

    기동 진단은 죽은 **조합**을 나열하지만 그 상태 자체는 말해주지 않는다.
    소액 계좌가 하락장에 들어가면 담당 전략이 통째로 최소주문액 아래로
    떨어져 봇이 **조용히 논다** — 로그도 정상이고 프로세스도 살아 있어
    운영자는 계속 매매 중이라 믿는다. 실측: 자산 $217에서 TRENDING_DOWN의
    유일한 전략 S4가 주문 $3.26으로 최소 $5 미달이었다.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(sm, '_regime_idle_alerted', set())

    def _rows(self, monkeypatch, tradable):
        monkeypatch.setattr(sm, '_diagnose_sizing_capability', lambda eq: [
            {'strategy': 'S4', 'regime': 'TRENDING_DOWN',
             'tradable': tradable, 'cost_usdt': 3.26, 'risk_pct': 0.0009},
            {'strategy': 'S6', 'regime': 'TRENDING_UP',
             'tradable': True, 'cost_usdt': 10.7, 'risk_pct': 0.003},
        ])

    def test_alerts_when_no_strategy_can_enter(self, monkeypatch):
        self._rows(monkeypatch, tradable=False)
        msg = sm.check_regime_idle('TRENDING_DOWN', 217.0)
        assert 'TRENDING_DOWN' in msg and '진입 가능한 전략이 없습니다' in msg

    def test_silent_when_some_strategy_works(self, monkeypatch):
        self._rows(monkeypatch, tradable=True)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0) == ''

    def test_alerts_once_per_regime(self, monkeypatch):
        """5초 루프에서 점검하므로 억제가 없으면 폭주한다."""
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0) == ''

    def test_rearms_after_recovery(self, monkeypatch):
        """자본이 늘어 해소됐다가 다시 나빠지면 또 알려야 한다."""
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0)
        self._rows(monkeypatch, tradable=True)
        assert sm.check_regime_idle('TRENDING_DOWN', 400.0) == ''
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0)

    def test_crisis_is_not_reported(self, monkeypatch):
        """CRISIS는 설계상 전면 차단 — 정상 동작이므로 알리지 않는다."""
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle(sm.REGIME_CRISIS, 217.0) == ''

    def test_empty_regime_is_safe(self, monkeypatch):
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('', 217.0) == ''

    def test_unknown_regime_is_not_reported(self, monkeypatch):
        """기동 직후 RegimeLoop가 첫 분류를 내기 전 상태.

        UNKNOWN은 REGIME_STRATEGY_MAP에 없어 '담당 전략 0개'로 읽힌다.
        걸러내지 않으면 **재시작할 때마다** 허위 경보가 나간다.
        실측: 11:14:56 '레짐(UNKNOWN)에서 진입 가능한 전략이 없습니다'.
        """
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('UNKNOWN', 217.0) == ''

    def test_unmapped_regime_is_not_reported(self, monkeypatch):
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('NOT_A_REGIME', 217.0) == ''

    def test_known_regime_still_reported(self, monkeypatch):
        """가드가 정상 경보까지 막으면 안 된다."""
        self._rows(monkeypatch, tradable=False)
        assert sm.check_regime_idle('TRENDING_DOWN', 217.0)

    def test_regimes_with_no_assigned_strategies_are_silent(self, monkeypatch):
        """맵에 빈 레짐이 셋 있다 — 전부 '설계상 쉬는 중'이지 자본 문제가 아니다.

        CRISIS(변동성 폭발) · MICRO_RANGING(동작 보존) · UNKNOWN(판별 실패).
        """
        self._rows(monkeypatch, tradable=False)
        empty = [rg for rg, strats in sm.REGIME_STRATEGY_MAP.items() if not strats]
        assert empty, '빈 레짐이 하나도 없다면 이 가드의 전제가 바뀐 것이다'
        for rg in empty:
            assert sm.check_regime_idle(rg, 217.0) == '', f'{rg}에서 허위 경보'

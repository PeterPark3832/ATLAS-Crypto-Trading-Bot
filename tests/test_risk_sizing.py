"""
ATLAS — 리스크/포지션 사이징 단위 테스트
================================
atlas_spot_main.py의 자금 관리 함수를 검증합니다.
  - _check_buying_power: 가용 USDT 잔고 확인
  - _get_kelly_scale: 최근 거래 기반 Kelly 스케일
  - _get_ratchet_scale: 드로다운 기반 리스크 스케일 (단계별 알림 dedup 포함)

실행:
  pytest tests/test_risk_sizing.py -v
"""



import pytest

import atlas_spot_main as sm


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    """모든 테스트에서 실제 텔레그램 호출을 차단."""
    monkeypatch.setattr(sm, '_tg', lambda msg: None)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """실제 운영 DB(atlas_spot.db) 대신 임시 DB 사용."""
    db_file = tmp_path / 'test_spot.db'
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', db_file)
    sm.init_spot_db()
    yield db_file


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """전역 _state를 테스트마다 깨끗하게 초기화."""
    fresh = {
        'equity': 0.0,
        'usdt_balance': 0.0,
        'peak_equity': 0.0,
        'ratchet_alert_tier': 0,
    }
    monkeypatch.setattr(sm, '_state', fresh)
    yield fresh


def _insert_trade(pnl_r: float, strategy: str = 'S3'):
    with sm._db_lock, sm._db_conn() as conn:
        conn.execute(
            "INSERT INTO spot_trades (strategy, symbol, pnl_r) VALUES (?, ?, ?)",
            (strategy, 'BTCUSDT', pnl_r),
        )


# ══════════════════════════════════════════════════════════════
#  _check_buying_power
# ══════════════════════════════════════════════════════════════

class TestCheckBuyingPower:
    def test_sufficient_funds_ok(self, _reset_state):
        _reset_state['equity'] = 1000.0
        _reset_state['usdt_balance'] = 500.0
        ok, reason = sm._check_buying_power(100.0)
        assert ok is True
        assert reason == ''

    def test_below_min_order_rejected(self, _reset_state):
        _reset_state['equity'] = 1000.0
        _reset_state['usdt_balance'] = 500.0
        ok, reason = sm._check_buying_power(1.0)
        assert ok is False
        assert 'NOTIONAL' in reason

    def test_reserve_eats_into_available(self, _reset_state):
        """예비금(10%) 차감 후 가용 USDT가 최소주문 미달이면 거부."""
        _reset_state['equity'] = 1000.0
        _reset_state['usdt_balance'] = 100.0  # 예비금 100 - 가용 0
        ok, reason = sm._check_buying_power(10.0)
        assert ok is False
        assert 'USDT 부족' in reason

    def test_cost_exceeds_available_rejected(self, _reset_state):
        _reset_state['equity'] = 1000.0
        _reset_state['usdt_balance'] = 200.0  # 예비금 100, 가용 100
        ok, reason = sm._check_buying_power(150.0)
        assert ok is False
        assert '가용' in reason

    def test_exact_min_order_boundary(self, _reset_state):
        _reset_state['equity'] = 1000.0
        _reset_state['usdt_balance'] = 500.0
        ok, _ = sm._check_buying_power(sm.SPOT_MIN_ORDER_USDT)
        assert ok is True


# ══════════════════════════════════════════════════════════════
#  _get_kelly_scale
# ══════════════════════════════════════════════════════════════

class TestGetKellyScale:
    def test_insufficient_trades_returns_min(self):
        for _ in range(sm.SPOT_KELLY_MIN_TRADES - 1):
            _insert_trade(0.5)
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN

    def test_no_losses_returns_one(self):
        for _ in range(sm.SPOT_KELLY_MIN_TRADES):
            _insert_trade(0.5)
        assert sm._get_kelly_scale('S3') == 1.0

    def test_no_wins_returns_min_scale(self):
        """전패 구간은 최소 스케일. 1.0을 주면 표본부족(0.30)보다 크게
        베팅하는 역전이 생긴다 — 연패 중인 전략이 최대로 베팅하는 꼴."""
        for _ in range(sm.SPOT_KELLY_MIN_TRADES):
            _insert_trade(-0.5)
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN

    def test_losing_streak_never_exceeds_small_sample_scale(self):
        for _ in range(15):
            _insert_trade(-0.4)
        assert sm._get_kelly_scale('S3') <= sm.SPOT_KELLY_SCALE_MIN

    def test_high_winrate_and_pf_allows_higher_cap(self):
        """승률>=55%, PF>=1.5 충족 → 상한 2.00까지 허용."""
        for _ in range(12):
            _insert_trade(2.0)   # 큰 승, wr 충족
        for _ in range(8):
            _insert_trade(-0.3)  # 작은 패
        scale = sm._get_kelly_scale('S3')
        assert sm.SPOT_KELLY_SCALE_MIN <= scale <= sm.SPOT_KELLY_SCALE_MAX

    def test_weak_performance_capped_at_1_5(self):
        """승률/PF 조건 미충족 시 상한 1.50."""
        for _ in range(6):
            _insert_trade(0.3)
        for _ in range(6):
            _insert_trade(-0.4)
        scale = sm._get_kelly_scale('S3')
        assert scale <= 1.50

    def test_strategy_isolation(self):
        """전략별 거래만 집계 — 다른 전략 거래는 영향 없음."""
        for _ in range(sm.SPOT_KELLY_MIN_TRADES):
            _insert_trade(0.5, strategy='S4')
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN


# ══════════════════════════════════════════════════════════════
#  _get_ratchet_scale
# ══════════════════════════════════════════════════════════════

class TestGetRatchetScale:
    def test_no_peak_returns_one(self, _reset_state):
        _reset_state['equity'] = 1000.0
        _reset_state['peak_equity'] = 0.0
        assert sm._get_ratchet_scale() == 1.0

    def test_normal_below_threshold(self, _reset_state):
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 980.0  # -2%
        assert sm._get_ratchet_scale() == 1.0
        assert _reset_state['ratchet_alert_tier'] == 0

    def test_soft_drawdown_scale_70pct(self, _reset_state):
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 940.0  # -6% (>=5% 임계)
        scale = sm._get_ratchet_scale()
        assert scale == 0.70
        assert _reset_state['ratchet_alert_tier'] == 1

    def test_hard_drawdown_scale_40pct(self, _reset_state):
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 900.0  # -10% (>=8% 임계)
        scale = sm._get_ratchet_scale()
        assert scale == 0.40
        assert _reset_state['ratchet_alert_tier'] == 2

    def test_hard_dd_recovery_restores_to_70pct(self, _reset_state):
        """하드DD 바닥 대비 +16% 회복(여전히 하드DD 구간 내) → 0.70로 복원."""
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 750.0  # -25%, 하드DD 진입
        sm._get_ratchet_scale()  # floor=750 기록
        _reset_state['equity'] = 750.0 * 1.16  # 바닥 대비 +16% 회복, 여전히 피크 대비 -13% (하드DD 구간 내)
        scale = sm._get_ratchet_scale()
        assert scale == 0.70
        assert _reset_state['ratchet_alert_tier'] == 1

    def test_recovery_to_normal_resets_tier(self, _reset_state):
        """소프트DD 진입 후 회복 시 tier=0으로 리셋."""
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 940.0
        sm._get_ratchet_scale()
        assert _reset_state['ratchet_alert_tier'] == 1
        _reset_state['equity'] = 1000.0
        scale = sm._get_ratchet_scale()
        assert scale == 1.0
        assert _reset_state['ratchet_alert_tier'] == 0

    def test_hard_dd_floor_persists_across_calls(self, _reset_state):
        """ratchet_floor가 최초 진입 시 즉시 저장되어야 함 (저장 안 되면 recover_pct가
        항상 0이 되어 회복이 영원히 불가능해지는 회귀 방지)."""
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 750.0
        sm._get_ratchet_scale()
        assert _reset_state.get('ratchet_floor') == 750.0
        # 동일 equity로 재호출해도 floor가 유지되어야 함
        sm._get_ratchet_scale()
        assert _reset_state.get('ratchet_floor') == 750.0

    def test_alert_tier_dedup_no_duplicate_calls(self, monkeypatch, _reset_state):
        """동일 단계 유지 중에는 _tg가 재호출되지 않음 (알림 중복 방지)."""
        calls = []
        monkeypatch.setattr(sm, '_tg', lambda msg: calls.append(msg))
        _reset_state['peak_equity'] = 1000.0
        _reset_state['equity'] = 940.0
        sm._get_ratchet_scale()
        sm._get_ratchet_scale()
        sm._get_ratchet_scale()
        assert len(calls) == 1

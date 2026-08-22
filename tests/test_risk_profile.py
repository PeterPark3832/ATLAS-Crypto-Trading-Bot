"""
ATLAS — 실효 리스크 프로파일 진단
================================
리스크 기반 사이징의 전제는 **"SL 거리와 무관하게 리스크가 일정하다"** 이다.
배분 상한(`SPOT_MAX_ALLOC_PCT`)이 걸리는 구간에서는 그 전제가 깨지고,
리스크가 SL 거리에 **비례**하게 된다.

현재 설정(리스크 2.0% / 상한 14%)에서 상한이 걸리는 조건은
SL거리 < 0.020/0.14 = **14.3%** — 전형 SL이 5% 안팎이므로 거의 모든
거래가 여기 해당한다:
    SL  2%   → 0.28%   ← 배분상한
    SL  5%   → 0.70%   ← 배분상한 (전형)
    SL 10%   → 1.40%   ← 배분상한
    SL 14.3%+ → 2.00%  (설정값 그대로)

이 상태를 "고쳐야 할 결함"으로 단정하지 않는다 — 포트폴리오 관점에서는
6종목 × 0.70% = 4.2%로 일간 손실 한도(-4%)와 눈금이 맞고, 값을 내리면
소액 구간에서 조합이 죽는다. 판단 근거는 atlas_spot_config.py의 사이징
주석에 있다. 여기서 하는 일은 **그 상태를 숫자로 고정**해, 누가 리스크나
상한을 바꾸면 테스트가 실패하며 주석을 함께 갱신하도록 강제하는 것이다.

실행:
  pytest tests/test_risk_profile.py -v
"""

from pathlib import Path

import pytest

import atlas_spot_config as cfg
import atlas_spot_main as sm


class TestRiskProfile:
    def test_reports_every_sl_bucket(self):
        rows = sm._risk_profile(1000.0)
        assert len(rows) >= 5
        assert all({'sl_pct', 'risk_pct', 'capped'} <= set(r) for r in rows)

    def test_risk_matches_size_position(self):
        """진단이 실제 사이징과 다른 값을 보고하면 진단이 아니라 거짓말이다."""
        eq = 1000.0
        for r in sm._risk_profile(eq):
            adj, _, _ = sm._size_position(eq, 100.0 * r['sl_pct'], 100.0)
            assert r['risk_pct'] == pytest.approx(adj)

    def test_capped_flag_is_accurate(self):
        eq = 1000.0
        for r in sm._risk_profile(eq):
            want = eq * cfg.SPOT_BASE_RISK_PCT / r['sl_pct']
            _, _, cost = sm._size_position(eq, 100.0 * r['sl_pct'], 100.0)
            assert r['capped'] == (cost < want - 1e-9)

    def test_scale_invariant(self):
        """리스크는 비율이므로 자본이 달라져도 프로파일은 같아야 한다."""
        a = [r['risk_pct'] for r in sm._risk_profile(500.0)]
        b = [r['risk_pct'] for r in sm._risk_profile(50_000.0)]
        assert a == pytest.approx(b)


class TestDetectsTheInversion:
    """현재의 사이징 성격을 **테스트로 고정**하는 래칫.

    지금은 '전형 SL 구간에서 배분 상한이 사이징을 결정한다'가 참이다.
    누가 리스크나 상한을 바꾸면 이 테스트들이 먼저 실패해서,
    config 사이징 주석(판단 기준·소액 구간 위험)을 함께 갱신하도록 만든다.
    실패 = 버그가 아니라 "문서도 같이 고쳐라"는 신호다.
    """

    def test_currently_all_buckets_are_capped(self):
        rows = sm._risk_profile(1000.0)
        capped = [r for r in rows if r['capped']]
        assert capped, (
            '배분 상한이 더 이상 걸리지 않는다 — 리스크 설정이 바뀌었다면 '
            'config 주석과 이 테스트를 함께 갱신할 것')

    def test_risk_rises_with_sl_distance_while_capped(self):
        """상한이 걸리는 구간에서는 리스크가 SL에 비례한다 — 역전."""
        capped = [r for r in sm._risk_profile(1000.0) if r['capped']]
        if len(capped) < 2:
            pytest.skip('상한 구간이 2개 미만')
        risks = [r['risk_pct'] for r in capped]
        assert risks == sorted(risks), '상한 구간에서 리스크가 단조 증가해야 한다'
        assert risks[-1] > risks[0] * 2, (
            f'역전 폭이 {risks[-1]/risks[0]:.1f}배 — 이 값이 1에 가까워지면 '
            f'결함이 해소된 것이다')

    def test_effective_risk_equals_alloc_times_sl(self):
        """상한 구간의 실효 리스크 = 배분상한 × SL거리. 설정값과 무관하다."""
        for r in sm._risk_profile(1000.0):
            if r['capped']:
                assert r['risk_pct'] == pytest.approx(
                    cfg.SPOT_MAX_ALLOC_PCT * r['sl_pct'])

    def test_cap_binds_exactly_below_risk_over_alloc(self):
        """상한이 걸리는 문턱 = 리스크 ÷ 배분상한. config 주석의 14.3%가
        이 식에서 나온 값임을 고정한다(둘 중 하나가 바뀌면 문턱도 바뀐다)."""
        threshold = cfg.SPOT_BASE_RISK_PCT / cfg.SPOT_MAX_ALLOC_PCT
        eq = 1000.0
        for sl_pct, expect_capped in ((threshold * 0.9, True),
                                      (threshold * 1.1, False)):
            _, _, cost = sm._size_position(eq, 100.0 * sl_pct, 100.0)
            want = eq * cfg.SPOT_BASE_RISK_PCT / sl_pct
            assert (cost < want - 1e-9) is expect_capped, (
                f'SL {sl_pct*100:.2f}%에서 상한 적용 여부가 예상과 다르다 '
                f'(문턱 {threshold*100:.1f}%)')

    def test_effective_risk_never_exceeds_the_setting(self):
        """상한은 리스크를 **줄이기만** 한다 — 설정값이 천장이다.
        (config 주석이 SPOT_BASE_RISK_PCT를 '상한'이라 부르는 근거)"""
        for r in sm._risk_profile(1000.0):
            assert r['risk_pct'] <= cfg.SPOT_BASE_RISK_PCT + 1e-12


class TestReportedRiskIsEffective:
    """`_diagnose_sizing_capability`가 **의도값**을 보고하면, 배분 상한이
    리스크를 3분의 1로 줄이고 있다는 사실이 진단에서 보이지 않는다.
    실제로 그래서 이 문제가 오래 숨어 있었다."""

    def test_reports_effective_not_intended(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_kelly_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_get_strategy_health_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_typical_sl_pct', lambda: 0.05)
        rows = sm._diagnose_sizing_capability(1000.0)
        assert rows
        for r in rows:
            assert 'intended_risk_pct' in r, '의도값도 함께 보고해야 비교가 된다'
            if r['alloc_capped']:
                assert r['risk_pct'] < r['intended_risk_pct']
            # 보고된 리스크는 실제 주문금액과 정합해야 한다
            assert r['risk_pct'] == pytest.approx(
                r['cost_usdt'] * 0.05 / 1000.0)

    def test_capped_flag_present(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_kelly_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_get_strategy_health_scale', lambda s: 1.0)
        monkeypatch.setattr(sm, '_typical_sl_pct', lambda: 0.05)
        assert any(r['alloc_capped'] for r in sm._diagnose_sizing_capability(1000.0))


class TestMaxPositionsIsHonest:
    def test_configured_max_is_unreachable(self):
        """가용자본 ÷ 종목당 상한 = 실제 한계. 설정값이 그보다 크면 표시가
        거짓이 된다 — config 주석에 그 사실이 적혀 있어야 한다."""
        real = (1 - cfg.SPOT_RESERVE_PCT) / cfg.SPOT_MAX_ALLOC_PCT
        if real < cfg.SPOT_MAX_POSITIONS:
            # encoding 명시: config에 한글 주석이 있어 Windows 기본
            # 로케일(cp949)로 읽으면 UnicodeDecodeError가 난다.
            src = Path(cfg.__file__).read_text(encoding='utf-8')
            assert '실제 한계는' in src, (
                f'설정 {cfg.SPOT_MAX_POSITIONS}개 vs 실제 {real:.0f}개 — '
                f'그 사실이 config에 적혀 있어야 한다')


class TestAllSlotsAreReachable:
    """슬롯 상한이 **자본으로 실제 채워지는 수**여야 한다.

    `SPOT_MAX_POSITIONS × SPOT_MAX_ALLOC_PCT + SPOT_RESERVE_PCT`가 100%에
    닿으면 수수료가 들어갈 자리가 없어 마지막 슬롯이 영영 안 채워진다.
    실제로 상한 15% × 6슬롯 + 예비금 10% = 정확히 100%였을 때, 6번째
    진입이 가용 $86.57 vs 주문 $87.00 으로 **$0.43 차이로 차단**됐다.
    설정은 6인데 동작은 5였다.
    """

    def test_slot_budget_leaves_room_for_fees(self):
        used = cfg.SPOT_MAX_POSITIONS * cfg.SPOT_MAX_ALLOC_PCT + cfg.SPOT_RESERVE_PCT
        assert used < 0.99, (
            f'슬롯 {cfg.SPOT_MAX_POSITIONS} × 배분 {cfg.SPOT_MAX_ALLOC_PCT:.0%} + '
            f'예비금 {cfg.SPOT_RESERVE_PCT:.0%} = {used:.1%} — 수수료 여유가 없어 '
            f'마지막 슬롯이 채워지지 않는다')

    def test_every_slot_actually_fills(self, monkeypatch):
        """_check_buying_power를 실제로 태워 슬롯을 순서대로 채워 본다."""
        eq, fee = 1000.0, 0.001
        state = {'equity': eq, 'usdt_balance': eq}
        monkeypatch.setattr(sm, '_state', state)
        filled = 0
        for _ in range(cfg.SPOT_MAX_POSITIONS):
            cost = eq * cfg.SPOT_MAX_ALLOC_PCT          # Kelly 1.0 → 상한에 걸린 크기
            ok, why = sm._check_buying_power(cost)
            if not ok:
                break
            state['usdt_balance'] -= cost * (1 + fee)
            filled += 1
        assert filled == cfg.SPOT_MAX_POSITIONS, (
            f'{cfg.SPOT_MAX_POSITIONS}슬롯 설정인데 {filled}개만 채워졌다 — '
            f'배분상한 × 슬롯수 + 예비금이 100%에 너무 가깝다')

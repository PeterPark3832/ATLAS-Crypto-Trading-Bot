"""
ATLAS — 실효 리스크 프로파일 진단
================================
리스크 기반 사이징의 전제는 **"SL 거리와 무관하게 리스크가 일정하다"** 이다.
배분 상한(`SPOT_MAX_ALLOC_PCT`)이 걸리는 구간에서는 그 전제가 깨지고,
리스크가 SL 거리에 **비례**하게 된다 — 불확실한 거래(넓은 SL)일수록
크게 거는 셈으로, 사이징이 존재하는 이유의 정반대다.

현재 설정(리스크 2.0% / 상한 15%)에서 실제로 그 상태다:
    SL  2%(타이트=확신)  → 0.30%
    SL 10%(넓음=불확실)  → 1.50%

설정만 봐서는 드러나지 않는다. `SPOT_BASE_RISK_PCT`는 2%라고 적혀 있지만
SL 15% 미만에서는 아무 일도 하지 않는다. 그래서 기동 시 숫자로 남긴다.

실행:
  pytest tests/test_risk_profile.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """이 클래스가 현재 결함을 **테스트로 고정**한다.

    지금은 '역전이 존재한다'가 참이다. 나중에 리스크를 0.0075로 내리면
    이 테스트들이 실패하며 "결함이 해소됐으니 문서를 갱신하라"고 알린다.
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
        if cfg.SPOT_MAX_POSITIONS > real:
            src = Path(cfg.__file__).read_text()
            assert '실제 한계는' in src, (
                f'설정 {cfg.SPOT_MAX_POSITIONS}개 vs 실제 {real:.0f}개 — '
                f'그 사실이 config에 적혀 있어야 한다')

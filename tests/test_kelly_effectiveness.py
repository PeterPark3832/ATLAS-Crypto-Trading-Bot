"""
ATLAS — Kelly 스케일이 실제로 '작동'하는지
================================
Kelly의 목적은 **성과가 좋은 전략에 자본을 더 배분**하는 것이다.
그런데 하한(SPOT_KELLY_SCALE_MIN)이 배수(SPOT_KELLY_FRACTION)와 정합하지
않으면, 현실적인 모든 성과 구간을 하한이 흡수해 Kelly가 **상수**가 된다.
그러면 코드는 멀쩡히 돌지만 기능은 죽어 있고, 테스트도 통과한다.

실제로 half-Kelly(×0.5) 도입 시 하한 0.30이 그대로 남아 이 상태가 됐었다:
  WR45%/RR1.5 → 0.30,  WR60%/RR2.5 → 0.30  (구분 없음)

여기서는 "차등이 살아있는가"를 직접 단언해 재발을 막는다.

실행:
  pytest tests/test_kelly_effectiveness.py -v
"""



import pytest

import atlas_spot_backtest as bt
import atlas_spot_main as sm


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'kelly.db')
    sm.init_spot_db()


def _load(wr: float, rr: float, n: int = 100, strategy: str = 'S3'):
    """승률 wr, 손익비 rr 인 거래 n건을 넣는다."""
    wins = int(round(n * wr))
    with sm._db_lock, sm._db_conn() as conn:
        for i in range(n):
            r = rr if i < wins else -1.0
            conn.execute(
                "INSERT INTO spot_trades (strategy,symbol,pnl_r,pnl_usdt,fee_usdt,"
                "reason,entry_ts,exit_ts,dry_run) VALUES (?,?,?,?,0,'TP',"
                "'2026-07-01','2026-07-01',0)",
                (strategy, 'BTCUSDT', r, r * 10))


def _theoretical(wr: float, b: float) -> float:
    raw = wr - (1 - wr) / b
    return max(sm.SPOT_KELLY_SCALE_MIN,
               min(sm.SPOT_KELLY_SCALE_MAX, raw * sm.SPOT_KELLY_FRACTION))


# ══════════════════════════════════════════════════════════════
#  기능이 죽어있지 않은가 (핵심)
# ══════════════════════════════════════════════════════════════

class TestKellyDifferentiates:
    def test_better_strategy_gets_larger_scale(self):
        """우수 전략이 평범한 전략보다 큰 배분을 받아야 한다.
        이 단언이 깨지면 Kelly가 상수가 된 것이다."""
        _load(0.45, 1.5, strategy='S3')      # 평범
        _load(0.60, 2.5, strategy='S6')      # 우수
        weak = sm._get_kelly_scale('S3')
        good = sm._get_kelly_scale('S6')
        assert good > weak, (
            f'우수({good:.3f})가 평범({weak:.3f})보다 커야 한다 — '
            f'하한 {sm.SPOT_KELLY_SCALE_MIN}이 배수 {sm.SPOT_KELLY_FRACTION}와 '
            f'정합하지 않으면 둘 다 하한에 붙어 기능이 죽는다')

    def test_scale_increases_monotonically_with_edge(self):
        """엣지가 커질수록 배분도 커져야 한다(적어도 감소하지 않음)."""
        prev = None
        for i, (wr, rr) in enumerate([(0.45, 1.5), (0.55, 2.0), (0.60, 2.5), (0.65, 3.0)]):
            sid = f'X{i}'
            _load(wr, rr, strategy=sid)
            k = sm._get_kelly_scale(sid)
            if prev is not None:
                assert k >= prev - 1e-9, f'엣지가 커졌는데 배분이 줄었다 ({prev} → {k})'
            prev = k
        assert prev > sm.SPOT_KELLY_SCALE_MIN, '최상위 구간은 하한을 넘어야 한다'

    def test_floor_is_consistent_with_fraction(self):
        """하한이 배수와 정합해야 한다. 정합하지 않으면 현실 구간 전체가
        하한에 흡수돼 Kelly가 상수가 된다."""
        # 승률 60% / RR 2.5 — 이 전략군에서 '우수'에 해당하는 현실적 상단
        raw = 0.60 - 0.40 / 2.5
        assert raw * sm.SPOT_KELLY_FRACTION > sm.SPOT_KELLY_SCALE_MIN, (
            f'우수 전략의 half-Kelly({raw * sm.SPOT_KELLY_FRACTION:.3f})조차 '
            f'하한({sm.SPOT_KELLY_SCALE_MIN})을 못 넘으면 차등이 불가능하다')

    def test_matches_theory(self):
        _load(0.60, 2.5, n=120)
        assert sm._get_kelly_scale('S3') == pytest.approx(_theoretical(0.60, 2.5), abs=0.02)


# ══════════════════════════════════════════════════════════════
#  경계 동작
# ══════════════════════════════════════════════════════════════

class TestKellyBounds:
    def test_insufficient_sample_uses_floor(self):
        _load(0.60, 2.5, n=sm.SPOT_KELLY_MIN_TRADES - 1)
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN

    def test_all_losses_uses_floor(self):
        _load(0.0, 1.0, n=40)
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN

    def test_never_exceeds_max(self):
        _load(0.90, 5.0, n=120)
        assert sm._get_kelly_scale('S3') <= sm.SPOT_KELLY_SCALE_MAX

    def test_weak_edge_clamped_to_floor(self):
        _load(0.40, 1.2, n=100)
        assert sm._get_kelly_scale('S3') == sm.SPOT_KELLY_SCALE_MIN


# ══════════════════════════════════════════════════════════════
#  백테스트 패리티
# ══════════════════════════════════════════════════════════════

class TestBacktestParity:
    def test_same_floor_and_fraction(self):
        assert bt.SPOT_KELLY_SCALE_MIN == sm.SPOT_KELLY_SCALE_MIN
        assert bt.SPOT_KELLY_FRACTION == sm.SPOT_KELLY_FRACTION
        assert bt.SPOT_KELLY_SCALE_MAX == sm.SPOT_KELLY_SCALE_MAX

    def test_backtest_would_also_differentiate(self):
        """백테스트가 라이브와 다른 배분을 쓰면 검증 결과를 믿을 수 없다."""
        weak = max(bt.SPOT_KELLY_SCALE_MIN,
                   (0.45 - 0.55 / 1.5) * bt.SPOT_KELLY_FRACTION)
        good = max(bt.SPOT_KELLY_SCALE_MIN,
                   (0.60 - 0.40 / 2.5) * bt.SPOT_KELLY_FRACTION)
        assert good > weak

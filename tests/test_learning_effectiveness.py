"""
ATLAS — 학습기가 **실제로 돈을 더 버는가**
================================
앞의 `test_learning.py`는 학습기가 *설계대로 동작하는지*를 본다. 이 파일은
다른 질문을 한다: **그래서 수익이 늘어나는가?**

둘은 전혀 다른 질문이다. 통계적으로 흠 없는 학습기도 실제로는
  · 학습 지연 때문에 이미 지나간 국면에 반응하거나
  · 수축이 과해 차이를 못 살리거나
  · 탐색 하한 때문에 나쁜 팔에 계속 돈을 넣어
전체 성과를 **떨어뜨릴** 수 있다. 그래서 진짜 참값(ground truth)을 아는
합성 시장에서 균등 배분과 정면으로 비교한다.

선행편향 방지: 매 시점의 배분은 **그 시점까지의 거래만** 보고 계산한다.
(이걸 어기면 학습기가 미래를 보고 배분해 당연히 이긴다 — 무의미한 승리)

실행:
  pytest tests/test_learning_effectiveness.py -v
  python tests/test_learning_effectiveness.py        # 성과 비교표 출력
"""

import os
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_learning as L

START = datetime(2025, 1, 1, tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════
#  합성 시장
# ══════════════════════════════════════════════════════════════

def simulate(truth: dict, n_per_arm: int, seed: int,
             sd: float = 0.9, days_between: float = 1.0,
             relearn_every: int = 10, cfg: L.LearnConfig = L.DEFAULT,
             regime_shift_at: float | None = None,
             truth_after: dict | None = None) -> dict:
    """참값을 아는 시장에서 균등 배분 vs 학습 배분을 비교한다.

    각 거래의 손익 기여 = 배분배수 × 실현 R.
    배분배수는 **그 거래 직전까지의 이력**으로만 계산한다(선행편향 없음).

    Returns: {'flat': 총R, 'learned': 총R, 'n': 거래수, ...}
    """
    rnd = random.Random(seed)
    arms = list(truth)

    # 거래 일정 생성 — 팔을 번갈아 가며 발생
    events = []
    for i in range(n_per_arm):
        for a in arms:
            events.append((i, a))
    rnd.shuffle(events)

    history: list[L.Observation] = []
    flat_total = 0.0
    learned_total = 0.0
    scales: dict = {}
    switched = False

    for idx, (i, (sid, rg)) in enumerate(events):
        now = START + timedelta(days=idx * days_between)

        # 국면 전환 — 참값이 도중에 바뀐다(비정상 시장)
        active = truth
        if regime_shift_at is not None and idx >= len(events) * regime_shift_at:
            active = truth_after or truth
            switched = True

        r = rnd.gauss(active[(sid, rg)], sd)

        # 배분은 **이전까지의** 이력으로만
        if idx % relearn_every == 0:
            res = L.learn(history, now=now, cfg=cfg)
            scales = {k: v['scale'] for k, v in res.items()}
        s = scales.get((sid, rg), cfg.unproven_scale)

        flat_total += r
        learned_total += s * r
        history.append(L.Observation(sid, rg, r, now))

    return {
        'flat': flat_total, 'learned': learned_total,
        'n': len(events), 'switched': switched,
        'edge': learned_total - flat_total,
    }


def repeated(truth, seeds, **kw) -> dict:
    """여러 시드로 반복해 우연을 배제한다."""
    runs = [simulate(truth, seed=s, **kw) for s in seeds]
    wins = sum(1 for r in runs if r['learned'] > r['flat'])
    return {
        'runs': runs, 'wins': wins, 'n_runs': len(runs),
        'mean_flat': sum(r['flat'] for r in runs) / len(runs),
        'mean_learned': sum(r['learned'] for r in runs) / len(runs),
    }


# ══════════════════════════════════════════════════════════════
#  ① 핵심 주장 — 팔이 다르면 학습이 이긴다
# ══════════════════════════════════════════════════════════════

MIXED = {
    ('S6', 'TRENDING_UP'):   0.45,
    ('S6', 'RANGING'):      -0.55,
    ('S4', 'RANGING'):       0.20,
    ('S3', 'WEAK_TREND'):   -0.35,
    ('S5', 'WEAK_TREND'):    0.05,
}


class TestBeatsFlatAllocation:
    def test_wins_on_average(self):
        out = repeated(MIXED, seeds=range(12), n_per_arm=60)
        assert out['mean_learned'] > out['mean_flat'], (
            f'학습 {out["mean_learned"]:.1f}R vs 균등 {out["mean_flat"]:.1f}R — '
            f'학습기가 가치를 더하지 못한다')

    def test_wins_most_runs(self):
        out = repeated(MIXED, seeds=range(12), n_per_arm=60)
        assert out['wins'] >= 9, (
            f'12회 중 {out["wins"]}회만 승 — 우연과 구별되지 않는다')

    def test_avoids_the_worst_arm(self):
        """가장 나쁜 팔의 배분이 가장 낮아야 한다."""
        rnd = random.Random(3)
        now = START + timedelta(days=200)
        hist = [L.Observation(s, r, rnd.gauss(mu, 0.9),
                              now - timedelta(days=i * 0.7))
                for (s, r), mu in MIXED.items() for i in range(50)]
        res = L.learn(hist, now=now)
        worst = min(MIXED, key=MIXED.get)
        best = max(MIXED, key=MIXED.get)
        assert res[worst]['scale'] < res[best]['scale']

    def test_ranking_correlates_with_truth(self):
        """배분 순위가 참값 순위와 같은 방향이어야 한다."""
        rnd = random.Random(5)
        now = START + timedelta(days=200)
        hist = [L.Observation(s, r, rnd.gauss(mu, 0.9),
                              now - timedelta(days=i * 0.7))
                for (s, r), mu in MIXED.items() for i in range(50)]
        res = L.learn(hist, now=now)
        pairs = [(MIXED[k], res[k]['scale']) for k in MIXED]
        conc = sum(1 for i in range(len(pairs)) for j in range(i + 1, len(pairs))
                   if (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1]) >= 0)
        total = len(pairs) * (len(pairs) - 1) / 2
        assert conc / total >= 0.8, f'순위 일치율 {conc/total:.0%}'


# ══════════════════════════════════════════════════════════════
#  ② 해를 끼치지 않는가 — 더 중요한 질문
# ══════════════════════════════════════════════════════════════

IDENTICAL = {('A', 'X'): 0.15, ('B', 'X'): 0.15, ('C', 'X'): 0.15}


class TestDoesNoHarm:
    """팔이 **전부 같을 때** 학습기가 성과를 깎으면 안 된다.

    이게 더 중요한 시험이다. 실제 전략들이 비슷비슷할 가능성이 높고,
    그때 학습기가 잡음을 신호로 착각해 배분을 흔들면 순손실이 난다.
    """

    def test_no_significant_loss_when_arms_identical(self):
        out = repeated(IDENTICAL, seeds=range(12), n_per_arm=60)
        ratio = out['mean_learned'] / out['mean_flat'] if out['mean_flat'] else 1.0
        assert ratio > 0.75, (
            f'동일한 팔들에서 학습 배분이 균등 대비 {ratio:.0%} — '
            f'잡음을 신호로 착각해 손해를 내고 있다')

    def test_all_losing_arms_are_scaled_down(self):
        """전부 손실이면 전체 노출을 줄여야 한다 — 균등보다 덜 잃어야 한다."""
        losing = {('A', 'X'): -0.30, ('B', 'X'): -0.25, ('C', 'X'): -0.35}
        out = repeated(losing, seeds=range(8), n_per_arm=60)
        assert out['mean_learned'] > out['mean_flat'], (
            '전부 지는 시장에서 손실을 줄이지 못한다')

    def test_never_amplifies_a_losing_book(self):
        losing = {('A', 'X'): -0.30, ('B', 'X'): -0.25}
        for r in repeated(losing, seeds=range(6), n_per_arm=60)['runs']:
            assert r['learned'] > r['flat']


# ══════════════════════════════════════════════════════════════
#  ③ 비정상 시장 — 참값이 바뀌면 따라가는가
# ══════════════════════════════════════════════════════════════

class TestAdaptsToRegimeShift:
    BEFORE = {('A', 'X'): 0.45, ('B', 'X'): -0.45}
    AFTER = {('A', 'X'): -0.45, ('B', 'X'): 0.45}     # 완전 역전

    def test_recovers_after_truth_flips(self):
        """참값이 뒤집힌 뒤에도 옛 배분을 고집하면 감쇠가 작동하지 않는 것이다."""
        rnd = random.Random(11)
        now = START
        hist = []
        # 1단계: A가 좋은 국면 90일
        for i in range(120):
            for (s, r), mu in self.BEFORE.items():
                hist.append(L.Observation(s, r, rnd.gauss(mu, 0.9),
                                          now + timedelta(days=i * 0.75)))
        mid = now + timedelta(days=90)
        res1 = L.learn(hist, now=mid)
        assert res1[('A', 'X')]['scale'] > res1[('B', 'X')]['scale']

        # 2단계: 역전 후 90일
        for i in range(120):
            for (s, r), mu in self.AFTER.items():
                hist.append(L.Observation(s, r, rnd.gauss(mu, 0.9),
                                          mid + timedelta(days=i * 0.75)))
        end = mid + timedelta(days=90)
        res2 = L.learn(hist, now=end)
        assert res2[('B', 'X')]['scale'] > res2[('A', 'X')]['scale'], (
            '참값이 뒤집혔는데 배분이 따라오지 않는다 — 감쇠가 무력하다')

    def test_no_decay_fails_to_adapt(self):
        """감쇠를 끄면 적응하지 못한다는 것을 보여 감쇠의 필요성을 고정한다."""
        rnd = random.Random(11)
        now = START
        hist = []
        for i in range(120):
            for (s, r), mu in self.BEFORE.items():
                hist.append(L.Observation(s, r, rnd.gauss(mu, 0.9),
                                          now + timedelta(days=i * 0.75)))
        mid = now + timedelta(days=90)
        for i in range(120):
            for (s, r), mu in self.AFTER.items():
                hist.append(L.Observation(s, r, rnd.gauss(mu, 0.9),
                                          mid + timedelta(days=i * 0.75)))
        end = mid + timedelta(days=90)
        no_decay = L.LearnConfig(half_life_days=100_000.0)
        res = L.learn(hist, now=end, cfg=no_decay)
        gap_nodecay = res[('B', 'X')]['scale'] - res[('A', 'X')]['scale']
        res_d = L.learn(hist, now=end)
        gap_decay = res_d[('B', 'X')]['scale'] - res_d[('A', 'X')]['scale']
        assert gap_decay >= gap_nodecay


# ══════════════════════════════════════════════════════════════
#  ④ 탐색 하한의 비용 — 공짜가 아니라는 것을 명시
# ══════════════════════════════════════════════════════════════

class TestExplorationCost:
    def test_floor_costs_money_on_bad_arms(self):
        """하한은 흡수상태를 막는 대신 **나쁜 팔에 계속 돈을 넣는다.**
        이 비용을 숫자로 드러내 운영자가 값을 정할 수 있게 한다.

        격리를 켜 두면 확실히 나쁜 팔은 0이 되어 하한이 아예 발동하지
        않는다. 그러면 두 설정이 같은 결과를 내 **비용이 0으로 보인다** —
        하한만 놓고 비교하려면 격리를 꺼야 한다.
        """
        losing = {('A', 'X'): -0.40, ('B', 'X'): 0.40}
        hi = repeated(losing, seeds=range(6), n_per_arm=60,
                      cfg=L.LearnConfig(explore_floor=0.50, quarantine=False))
        lo = repeated(losing, seeds=range(6), n_per_arm=60,
                      cfg=L.LearnConfig(explore_floor=0.05, quarantine=False))
        assert lo['mean_learned'] > hi['mean_learned'], (
            '하한이 낮을수록 나쁜 팔 노출이 줄어 성과가 좋아야 한다')

    def test_quarantine_makes_floor_free_on_clear_losers(self):
        """격리가 켜져 있으면 확실한 손실 팔에서는 하한 비용이 들지 않는다."""
        losing = {('A', 'X'): -0.40, ('B', 'X'): 0.40}
        hi = repeated(losing, seeds=range(4), n_per_arm=60,
                      cfg=L.LearnConfig(explore_floor=0.50))
        lo = repeated(losing, seeds=range(4), n_per_arm=60,
                      cfg=L.LearnConfig(explore_floor=0.05))
        assert hi['mean_learned'] == pytest.approx(lo['mean_learned'], rel=1e-6)

    def test_floor_still_keeps_arm_alive(self):
        cfg = L.LearnConfig(explore_floor=0.05, quarantine=False)
        rnd = random.Random(2)
        now = START + timedelta(days=100)
        hist = [L.Observation('A', 'X', rnd.gauss(-0.5, 0.9),
                              now - timedelta(days=i * 0.5)) for i in range(60)]
        assert L.learn(hist, now=now, cfg=cfg)[('A', 'X')]['scale'] >= 0.05


# ══════════════════════════════════════════════════════════════
#  콘솔 리포트
# ══════════════════════════════════════════════════════════════

def _report():
    scenarios = [
        ('팔이 뚜렷이 다름', MIXED, {}),
        ('팔이 전부 동일', IDENTICAL, {}),
        ('전부 손실', {('A', 'X'): -0.30, ('B', 'X'): -0.25, ('C', 'X'): -0.35}, {}),
    ]
    w = 76
    print('═' * w)
    print('  학습 배분 vs 균등 배분 — 참값을 아는 합성 시장 (시드 12회 평균)')
    print('═' * w)
    print(f'  {"시나리오":<20}{"균등(R)":>12}{"학습(R)":>12}{"차이":>12}{"승률":>10}')
    print(f'  {"─" * (w - 4)}')
    for name, truth, kw in scenarios:
        out = repeated(truth, seeds=range(12), n_per_arm=60, **kw)
        d = out['mean_learned'] - out['mean_flat']
        wr = f'{out["wins"]}/{out["n_runs"]}'
        print(f'  {name:<20}{out["mean_flat"]:>12.1f}{out["mean_learned"]:>12.1f}'
              f'{d:>+12.1f}{wr:>10}')
    print('═' * w)
    print('  ※ 손익은 R배수 합계. 배분배수 × 실현R 로 계산하며, 배분은 항상')
    print('    그 거래 **직전까지의** 이력만 보고 정한다(선행편향 없음).')


if __name__ == '__main__':
    _report()

"""
ATLAS — 자기주도 학습 (atlas_learning)
================================
학습 시스템은 **틀려도 조용하다** — 잘못 배운 배분은 예외를 던지지 않고
그냥 돈을 잃는다. 그래서 이 테스트는 결과값보다 **성질(property)** 을
고정하는 데 집중한다:

  · 격리는 반드시 유한 시간 내에 풀린다 (흡수상태 부재)
  · 증거가 적을수록 배분이 전체 평균 쪽으로 당겨진다 (수축)
  · 증거가 낡을수록 확신이 줄어든다 (감쇠)
  · 운 좋은 소표본이 큰 배분을 받지 못한다 (과적합 방지)
  · 비용을 못 넘는 엣지는 이득으로 세지 않는다

초안의 실제 결함: Kish ESS로 최소표본 게이트를 걸었는데 ESS는 가중치에
**비율 불변**이라 시간이 지나도 값이 변하지 않았다. 격리된 팔이 영원히
격리되는 — 이 모듈이 막으려던 바로 그 흡수상태였다. `test_ess_is_time_
invariant`가 그 사실을, `TestNoAbsorbingState`가 수정을 고정한다.

실행:
  pytest tests/test_learning.py -v
"""

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


import pytest

import atlas_learning as L

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def obs(strategy='S6', regime='TRENDING_UP', rs=(), age_days=0, now=NOW):
    """나이가 같은 관측 묶음."""
    return [L.Observation(strategy, regime, float(r),
                          now - timedelta(days=age_days)) for r in rs]


def spread(strategy, regime, rs, now=NOW, step=1.0):
    """하루 간격으로 흩어진 관측(가장 최근이 age 0)."""
    return [L.Observation(strategy, regime, float(r),
                          now - timedelta(days=i * step))
            for i, r in enumerate(rs)]


# ══════════════════════════════════════════════════════════════
#  ① 감쇠 · 정보량
# ══════════════════════════════════════════════════════════════

class TestDecay:
    def test_half_life_halves(self):
        assert L.decay_weight(45, 45) == pytest.approx(0.5)
        assert L.decay_weight(90, 45) == pytest.approx(0.25)

    def test_fresh_is_full_weight(self):
        assert L.decay_weight(0, 45) == 1.0

    def test_negative_age_is_clamped(self):
        """시계 왜곡으로 미래 타임스탬프가 들어와도 가중치가 폭발하면 안 된다."""
        assert L.decay_weight(-10, 45) == 1.0
        o = L.Observation('S6', 'X', 1.0, NOW + timedelta(days=30))
        assert o.age_days(NOW) == 0.0

    def test_zero_half_life_disables_decay(self):
        assert L.decay_weight(1000, 0) == 1.0

    def test_info_decays_to_zero(self):
        o = obs(rs=[1.0] * 20)
        i0 = L.fit_arm(o, NOW)['info']
        i1 = L.fit_arm(o, NOW + timedelta(days=45))['info']
        assert i1 == pytest.approx(i0 / 2, rel=1e-6)


class TestEssVsInfo:
    """초안이 뒤섞었던 두 개념을 분리해 고정한다."""

    def test_ess_is_time_invariant(self):
        """ESS는 시간이 지나도 변하지 않는다 — 게이트로 쓰면 안 되는 이유."""
        o = obs(rs=[1.0] * 20)
        e0 = L.fit_arm(o, NOW)['ess']
        e1 = L.fit_arm(o, NOW + timedelta(days=365))['ess']
        assert e1 == pytest.approx(e0, rel=1e-9)

    def test_ess_measures_concentration(self):
        assert L.effective_sample_size([1, 1, 1, 1]) == pytest.approx(4.0)
        assert L.effective_sample_size([1, 0.001, 0.001]) < 1.1

    def test_ess_scale_invariant(self):
        a = L.effective_sample_size([1.0, 0.5, 0.25])
        b = L.effective_sample_size([0.1, 0.05, 0.025])
        assert a == pytest.approx(b)

    def test_ess_empty_is_zero(self):
        assert L.effective_sample_size([]) == 0.0
        assert L.effective_sample_size([0, 0]) == 0.0


class TestWeightedStats:
    def test_uniform_weights_match_plain(self):
        import statistics
        vals = [1.0, -1.0, 2.0, -0.5, 0.3]
        m, sd, ess = L.weighted_stats(vals, [1.0] * len(vals))
        assert m == pytest.approx(sum(vals) / len(vals))
        assert sd == pytest.approx(statistics.stdev(vals), rel=1e-6)
        assert ess == pytest.approx(len(vals))

    def test_recent_dominates(self):
        m, _, _ = L.weighted_stats([10.0, 0.0], [1.0, 0.01])
        assert m > 9.0

    def test_identical_values_have_zero_sd(self):
        _, sd, _ = L.weighted_stats([2.0] * 5, [1.0] * 5)
        assert sd == pytest.approx(0.0)

    def test_mismatched_lengths_safe(self):
        assert L.weighted_stats([1.0, 2.0], [1.0]) == (0.0, 0.0, 0.0)

    def test_empty_safe(self):
        assert L.weighted_stats([], []) == (0.0, 0.0, 0.0)

    def test_single_observation_sd_zero_not_nan(self):
        """1건은 분산을 추정할 수 없다. NaN이 흘러나가면 배분이 오염된다."""
        m, sd, ess = L.weighted_stats([1.0], [1.0])
        assert m == 1.0 and sd == 0.0 and not math.isnan(sd)


# ══════════════════════════════════════════════════════════════
#  ② 수축 — 소표본 과적합 방지
# ══════════════════════════════════════════════════════════════

class TestShrinkage:
    def test_no_evidence_returns_prior(self):
        assert L.shrink_mean(5.0, 0.0, 0.1) == pytest.approx(0.1)

    def test_much_evidence_returns_own_mean(self):
        assert L.shrink_mean(0.5, 10_000.0, 0.0) == pytest.approx(0.5, abs=1e-3)

    def test_k_equals_info_is_midpoint(self):
        cfg = L.LearnConfig(shrink_k=10.0)
        assert L.shrink_mean(1.0, 10.0, 0.0, cfg) == pytest.approx(0.5)

    def test_shrunk_lies_between(self):
        for info in (1, 5, 20, 100):
            s = L.shrink_mean(0.8, info, -0.2)
            assert -0.2 <= s <= 0.8

    def test_lucky_small_sample_is_pulled_down(self):
        """3연승한 팔이 곧바로 최대 배분을 받으면 안 된다."""
        lucky = spread('S6', 'TRENDING_UP', [2.0, 2.0, 2.0])
        many = spread('S4', 'RANGING', [0.2] * 60)
        res = L.learn(lucky + many, now=NOW)
        assert res[('S6', 'TRENDING_UP')]['scale'] <= res[('S4', 'RANGING')]['scale']

    def test_pooled_prior_is_evidence_weighted(self):
        arms = {('A', 'X'): {'mean_r': 1.0, 'info': 90.0},
                ('B', 'X'): {'mean_r': 0.0, 'info': 10.0}}
        assert L.pooled_prior(arms) == pytest.approx(0.9)

    def test_pooled_prior_ignores_empty_arms(self):
        arms = {('A', 'X'): {'mean_r': 1.0, 'info': 10.0},
                ('B', 'X'): {'mean_r': 99.0, 'info': 0.0}}
        assert L.pooled_prior(arms) == pytest.approx(1.0)

    def test_pooled_prior_no_arms(self):
        assert L.pooled_prior({}) == 0.0

    def test_prior_can_be_negative(self):
        """전부 지고 있으면 prior도 음수여야 한다 — 소표본을 낙관 보정하면 안 된다."""
        losers = (spread('A', 'X', [-0.5] * 40) + spread('B', 'X', [-0.4] * 40))
        stats = {k: L.fit_arm(v, NOW) for k, v in _bucket(losers).items()}
        assert L.pooled_prior(stats) < 0


class TestAdaptiveShrinkage:
    """고정 수축계수의 실제 피해: **잘 버는 팔의 자본이 깎인다.**

    손실 팔이 여럿이면 전체 평균(prior)이 크게 음수가 되고, 고정 k는 진짜로
    다른 팔까지 거기로 끌어내린다. 이 모듈이 하려던 일의 정반대다.
    분산분해로 '진짜 차이 τ²'를 추정해 수축 강도를 데이터가 정하게 한다.
    """

    @staticmethod
    def _noisy(sid, rg, mean, n, seed):
        import random
        rnd = random.Random(seed)
        return [L.Observation(sid, rg, mean + rnd.gauss(0, 0.8),
                              NOW - timedelta(days=i)) for i in range(n)]

    @pytest.fixture
    def mixed(self):
        good = self._noisy('G', 'X', 0.45, 45, 1)
        losers = []
        for i in range(6):
            losers += self._noisy(f'L{i}', 'Y', -0.55, 45, 10 + i)
        return good, losers

    def test_fixed_k_drags_winner_estimate_down(self, mixed):
        """고정 k의 피해는 **추정치**에서 먼저 나타난다.

        배분은 상한(max_scale)에 걸리면 차이가 안 보이므로, 피해를 재는
        올바른 지점은 수축된 평균이다. 이 테스트가 피해를 재현하지 못하면
        아래 `test_adaptive_protects_winner`는 아무것도 증명하지 못한다.
        """
        good, losers = mixed
        cfg = L.LearnConfig(adaptive_shrink=False)
        alone = L.learn(good, now=NOW, cfg=cfg)[('G', 'X')]['shrunk_r']
        withl = L.learn(good + losers, now=NOW, cfg=cfg)[('G', 'X')]['shrunk_r']
        assert withl < alone

    def test_adaptive_protects_winner(self, mixed):
        good, losers = mixed
        fixed = L.learn(good + losers, now=NOW,
                        cfg=L.LearnConfig(adaptive_shrink=False))[('G', 'X')]
        adapt = L.learn(good + losers, now=NOW,
                        cfg=L.LearnConfig(adaptive_shrink=True))[('G', 'X')]
        assert adapt['shrink_k'] < fixed['shrink_k']
        assert adapt['shrunk_r'] > fixed['shrunk_r']
        assert adapt['lcb'] > fixed['lcb']

    def test_adaptive_shows_up_in_scale_when_not_capped(self):
        """상한에 걸리지 않는 완만한 승자에서는 배분에도 차이가 나야 한다."""
        good = self._noisy('G', 'X', 0.12, 45, 21)
        losers = []
        for i in range(4):
            losers += self._noisy(f'L{i}', 'Y', -0.30, 45, 30 + i)
        fixed = L.learn(good + losers, now=NOW,
                        cfg=L.LearnConfig(adaptive_shrink=False))[('G', 'X')]
        adapt = L.learn(good + losers, now=NOW,
                        cfg=L.LearnConfig(adaptive_shrink=True))[('G', 'X')]
        if fixed['scale'] < L.DEFAULT.max_scale:
            assert adapt['scale'] >= fixed['scale']

    def test_k_never_exceeds_configured_cap(self, mixed):
        good, losers = mixed
        a = L.learn(good + losers, now=NOW)[('G', 'X')]
        assert a['shrink_k'] <= L.DEFAULT.shrink_k

    def test_pure_noise_gives_strong_shrinkage(self):
        """팔 간 차이가 전부 잡음이면 **강하게** 수축해야 한다.

        필요한 k는 수천 단위다(k = 관측분산/τ², τ²→0). 초안은 이걸
        shrink_k=12로 잘라 수축을 사실상 무효화했고, 참값이 같은 팔들이
        0.25 대 1.50으로 갈렸다. 기본값이 아니라 발산 방지 상한만 건다.
        """
        arms = []
        for i in range(5):
            arms += self._noisy(f'A{i}', 'X', 0.0, 45, 100 + i)
        k = L.estimate_shrink_k({kk: L.fit_arm(v, NOW)
                                 for kk, v in _bucket(arms).items()})
        assert k > 100.0, f'k={k:.1f} — 잡음뿐인데 수축이 약하다'
        assert k <= L.DEFAULT.max_shrink_k

    def test_single_arm_falls_back_to_fixed(self):
        """팔이 하나면 τ를 추정할 **대상이 없다** — 최대 수축이 아니라 기본값.

        '못 구했다'와 '구했더니 0'을 뭉치면 팔 하나짜리 계좌가 이유 없이
        최대 수축을 맞는다.
        """
        arms = {('A', 'X'): {'mean_r': 0.5, 'info': 20.0, 'sd_r': 1.0}}
        assert L.estimate_shrink_k(arms) == pytest.approx(L.DEFAULT.shrink_k)

    def test_no_arms_falls_back_to_fixed(self):
        assert L.estimate_shrink_k({}) == pytest.approx(L.DEFAULT.shrink_k)

    def test_disabled_uses_fixed(self):
        arms = {('A', 'X'): {'mean_r': 1.0, 'info': 20.0, 'sd_r': 0.1},
                ('B', 'X'): {'mean_r': -1.0, 'info': 20.0, 'sd_r': 0.1}}
        cfg = L.LearnConfig(adaptive_shrink=False)
        assert L.estimate_shrink_k(arms, cfg) == pytest.approx(cfg.shrink_k)

    def test_k_is_never_negative(self):
        arms = {('A', 'X'): {'mean_r': 9.0, 'info': 50.0, 'sd_r': 5.0},
                ('B', 'X'): {'mean_r': -9.0, 'info': 50.0, 'sd_r': 5.0}}
        assert L.estimate_shrink_k(arms) >= 0.0


class TestEdgeReference:
    """상대 배분의 분모·중심. 여기가 틀리면 배분 전체가 조용히 틀어진다."""

    def test_spread_has_a_floor(self):
        """차이가 **거의 없는** 팔들이 최대 재배분을 유발하면 안 된다.

        분모가 실제 표준편차뿐이면, 엣지 차이가 0.001R이어도 z가 폭발해
        한쪽은 상한, 한쪽은 하한으로 갈린다. 잡음에 최대로 반응하는 셈이다.
        ref_edge를 분모 하한으로 둬 이걸 막는다.
        """
        arms = {('A', 'X'): {'net_edge': 0.1000, 'info': 30.0, 'state': 'trusted'},
                ('B', 'X'): {'net_edge': 0.1001, 'info': 30.0, 'state': 'trusted'}}
        mean, spread = L.edge_reference(arms, tau=0.0)
        assert spread >= L.DEFAULT.ref_edge
        z = (0.1001 - mean) / spread
        assert abs(L.DEFAULT.gain * z) < 0.01, (
            '무의미한 차이가 큰 배분 변화를 만든다')

    def test_near_identical_arms_stay_near_one(self):
        """위 성질의 종단 확인 — 배분이 실제로 1.0 근처에 머물러야 한다."""
        import random
        rnd = random.Random(9)
        o = []
        for sid in ('A', 'B', 'C'):
            o += [L.Observation(sid, 'X', 0.15 + rnd.gauss(0, 0.9),
                                NOW - timedelta(days=i)) for i in range(45)]
        for a in L.learn(o, now=NOW).values():
            if a['state'] not in ('unproven', 'quarantined'):
                assert 0.5 < a['scale'] < 1.6

    def test_unproven_arms_excluded_from_reference(self):
        """증거가 없는 팔의 net_edge는 0으로 채워져 있다.

        그걸 기준 계산에 넣으면, 검증되지도 않은 0이 평균을 끌어당겨
        **증명된 팔들의 상대 위치를 왜곡한다.**
        """
        arms = {
            ('A', 'X'): {'net_edge': 0.20, 'info': 30.0, 'state': 'trusted'},
            ('B', 'X'): {'net_edge': 0.30, 'info': 30.0, 'state': 'trusted'},
            ('C', 'X'): {'net_edge': 0.00, 'info': 30.0, 'state': 'unproven'},
        }
        mean, _ = L.edge_reference(arms, tau=0.2)
        assert mean == pytest.approx(0.25), (
            f'평균 {mean:.3f} — 미검증 팔이 기준을 끌어내렸다')

    def test_all_unproven_falls_back(self):
        arms = {('A', 'X'): {'net_edge': 0.0, 'info': 30.0, 'state': 'unproven'}}
        mean, spread = L.edge_reference(arms, tau=0.0)
        assert mean == 0.0 and spread >= L.DEFAULT.ref_edge

    def test_empty_reference_is_safe(self):
        mean, spread = L.edge_reference({}, tau=0.0)
        assert math.isfinite(mean) and spread > 0

    def test_normalizer_scales_with_real_dispersion(self):
        """τ로 정규화하므로 **차이의 절대 크기**가 배분을 좌우하면 안 된다.

        고정 분모를 쓰면 팔들이 서로 멀수록 무조건 상·하한으로 saturate해,
        '누가 더 나은가'의 구조가 아니라 '얼마나 벌어졌나'라는 무관한 양이
        배분을 정하게 된다. 엣지를 통째로 3배 늘려도 상대 위치는 그대로이니
        배분도 그대로여야 한다.
        """
        import random

        def build(mult):
            rnd = random.Random(4)
            o = []
            for sid, mu in (('A', 0.30), ('B', 0.00), ('C', -0.30)):
                o += [L.Observation(sid, 'X', mu * mult + rnd.gauss(0, 0.5 * mult),
                                    NOW - timedelta(days=i)) for i in range(45)]
            return L.learn(o, now=NOW, cfg=L.LearnConfig(quarantine=False))

        a, b = build(1.0), build(3.0)
        for k in a:
            assert a[k]['scale'] == pytest.approx(b[k]['scale'], abs=0.12), (
                f'{k}: 1배 {a[k]["scale"]:.3f} vs 3배 {b[k]["scale"]:.3f} — '
                f'차이의 절대 크기가 배분을 바꾸고 있다')

    def test_slightly_above_average_arm_is_only_slightly_boosted(self):
        """팔들이 크게 벌어져 있을 때, **살짝** 나은 팔은 살짝만 올려야 한다.

        고정 분모를 쓰면 이 팔의 z가 폭발해 곧바로 상한(1.50)으로 튄다.
        차이의 절대 크기(τ)로 정규화해야 '살짝 나음'이 '살짝 더'가 된다.
        (상·하한에 걸린 팔들은 어떤 분모에서도 같은 값이라 이 성질을
         드러내지 못한다 — 중간 영역에서만 검출된다)
        """
        import random
        rnd = random.Random(6)
        o = []
        for sid, mu in (('A', 1.20), ('B', 0.15), ('C', -1.20)):
            o += [L.Observation(sid, 'X', mu + rnd.gauss(0, 0.30),
                                NOW - timedelta(days=i)) for i in range(45)]
        res = L.learn(o, now=NOW, cfg=L.LearnConfig(quarantine=False))
        b = res[('B', 'X')]['scale']
        assert 0.90 < b < 1.25, (
            f'평균보다 조금 나은 팔의 배분이 {b:.3f} — '
            f'분모가 너무 작아 작은 차이가 최대 배분으로 증폭됐다')

    def test_reference_is_info_weighted(self):
        arms = {('A', 'X'): {'net_edge': 1.0, 'info': 90.0, 'state': 'trusted'},
                ('B', 'X'): {'net_edge': 0.0, 'info': 10.0, 'state': 'trusted'}}
        assert L.edge_reference(arms, tau=0.2)[0] == pytest.approx(0.9)


class TestConfigValidation:
    """오설정은 예외를 던지지 않고 **그냥 돈을 잃는다** — 기동 시 막는다."""

    def test_cap_below_floor_rejected(self):
        with pytest.raises(ValueError, match='explore_floor'):
            L.LearnConfig(max_scale=0.10, explore_floor=0.25)

    def test_cap_below_floor_would_break_the_floor(self):
        """왜 막아야 하는지 — min(cap, max(floor, raw))가 하한을 덮어쓴다."""
        floor, cap, raw = 0.25, 0.10, 5.0
        assert min(cap, max(floor, raw)) < floor

    def test_negative_z_rejected(self):
        with pytest.raises(ValueError, match='z'):
            L.LearnConfig(z=-1.0)

    def test_negative_ref_edge_rejected(self):
        with pytest.raises(ValueError, match='ref_edge'):
            L.LearnConfig(ref_edge=-0.1)

    def test_zero_min_info_rejected(self):
        with pytest.raises(ValueError, match='min_info'):
            L.LearnConfig(min_info=0.0)

    def test_negative_floor_rejected(self):
        with pytest.raises(ValueError):
            L.LearnConfig(explore_floor=-0.1)

    def test_defaults_are_valid(self):
        assert L.LearnConfig().max_scale >= L.LearnConfig().explore_floor


class TestCorruptInput:
    """NaN 오염이 가장 위험한 이유: 배분이 **멀쩡해 보이는 값**으로 나온다."""

    def test_nan_is_dropped(self):
        o = spread('A', 'X', [0.5] * 39) + [L.Observation('A', 'X', float('nan'), NOW)]
        a = L.learn(o, now=NOW)[('A', 'X')]
        assert a['n_raw'] == 39

    def test_inf_is_dropped(self):
        o = spread('A', 'X', [0.5] * 39) + [L.Observation('A', 'X', float('inf'), NOW)]
        assert L.learn(o, now=NOW)[('A', 'X')]['n_raw'] == 39

    def test_nan_would_silently_pass_quarantine_check(self):
        """가드가 없으면 왜 위험한지 — NaN 비교는 전부 False다."""
        assert not (float('nan') < 0)
        assert max(0.25, float('nan')) == 0.25   # 하한이 그대로 나온다

    def test_all_nan_arm_disappears(self):
        o = [L.Observation('A', 'X', float('nan'), NOW) for _ in range(30)]
        assert L.learn(o, now=NOW) == {}

    def test_result_stays_finite_with_corruption(self):
        o = (spread('A', 'X', [0.5] * 30)
             + [L.Observation('A', 'X', float('nan'), NOW)] * 5
             + spread('B', 'Y', [-0.3] * 30))
        for a in L.learn(o, now=NOW).values():
            assert all(math.isfinite(v) for v in a.values() if isinstance(v, float))


def _bucket(observations):
    out = {}
    for o in observations:
        out.setdefault(L.arm_key(o.strategy, o.regime), []).append(o)
    return out


# ══════════════════════════════════════════════════════════════
#  ③ 신뢰구간 — 증거가 낡으면 넓어져야 한다
# ══════════════════════════════════════════════════════════════

class TestConfidenceBounds:
    def test_lcb_below_ucb(self):
        lo, hi = L.confidence_bounds(0.2, 1.0, 20.0)
        assert lo < 0.2 < hi

    def test_zero_sd_collapses_interval(self):
        lo, hi = L.confidence_bounds(0.2, 0.0, 20.0)
        assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)

    def test_more_evidence_narrows(self):
        w_few = L.confidence_bounds(0.2, 1.0, 5.0)
        w_many = L.confidence_bounds(0.2, 1.0, 500.0)
        assert (w_many[1] - w_many[0]) < (w_few[1] - w_few[0])

    def test_stale_evidence_widens(self):
        """같은 거래라도 오래되면 '모른다'로 돌아가야 한다."""
        o = spread('S6', 'TRENDING_UP', [1.0, -1.0] * 20)
        def width(t):
            s = L.fit_arm(o, t)
            lo, hi = L.confidence_bounds(s['mean_r'], s['sd_r'], s['info'])
            return hi - lo
        assert width(NOW + timedelta(days=180)) > width(NOW)

    def test_zero_info_does_not_divide_by_zero(self):
        cfg = L.LearnConfig(shrink_k=0.0)
        lo, hi = L.confidence_bounds(0.2, 1.0, 0.0, cfg)
        assert math.isfinite(lo) and math.isfinite(hi)


# ══════════════════════════════════════════════════════════════
#  ④ 흡수상태 부재 — 이 모듈의 핵심 보장
# ══════════════════════════════════════════════════════════════

class TestNoAbsorbingState:
    LOSER = None

    @pytest.fixture
    def loser(self):
        return spread('S6', 'TRENDING_UP', [-1.0] * 20)

    def test_confident_loser_is_quarantined(self, loser):
        a = L.learn(loser, now=NOW)[('S6', 'TRENDING_UP')]
        assert a['state'] == 'quarantined' and a['scale'] == 0.0

    def test_quarantine_releases_in_finite_time(self, loser):
        """거래가 멈춰도 반드시 풀려야 한다 — 학습기 최대 실패 방지."""
        a = L.learn(loser, now=NOW)[('S6', 'TRENDING_UP')]
        d = L.days_until_revival(a)
        assert math.isfinite(d) and d > 0
        later = L.learn(loser, now=NOW + timedelta(days=d + 1))[('S6', 'TRENDING_UP')]
        assert later['scale'] > 0.0, f'{d:.0f}일 뒤에도 격리 — 흡수상태'

    def test_predicted_revival_is_accurate(self, loser):
        """예측 시점 직전에는 아직 격리, 직후에는 해제 — 예측이 맞아야 한다."""
        a = L.learn(loser, now=NOW)[('S6', 'TRENDING_UP')]
        d = L.days_until_revival(a)
        before = L.learn(loser, now=NOW + timedelta(days=d - 1))[('S6', 'TRENDING_UP')]
        after = L.learn(loser, now=NOW + timedelta(days=d + 1))[('S6', 'TRENDING_UP')]
        assert before['scale'] == 0.0 and after['scale'] > 0.0

    def test_revival_zero_when_already_below(self):
        a = {'info': 1.0}
        assert L.days_until_revival(a) == 0.0

    def test_no_decay_means_no_revival_and_says_so(self):
        """감쇠를 끄면 복귀가 보장되지 않는다 — 조용히 유한값을 내면 안 된다."""
        cfg = L.LearnConfig(half_life_days=0.0)
        assert L.days_until_revival({'info': 100.0}, cfg) == math.inf

    def test_weak_arm_keeps_exploration_floor(self):
        """확신이 없는 약한 팔은 0이 아니라 하한을 받아야 증거가 계속 쌓인다."""
        mixed = spread('S6', 'TRENDING_UP', [0.9, -1.0] * 15)
        a = L.learn(mixed, now=NOW)[('S6', 'TRENDING_UP')]
        if a['state'] != 'quarantined':
            assert a['scale'] >= L.DEFAULT.explore_floor

    def test_quarantine_can_be_disabled(self, loser):
        cfg = L.LearnConfig(quarantine=False)
        a = L.learn(loser, now=NOW, cfg=cfg)[('S6', 'TRENDING_UP')]
        assert a['scale'] >= cfg.explore_floor


# ══════════════════════════════════════════════════════════════
#  ⑤ 배분 — 비용·증거 반영
# ══════════════════════════════════════════════════════════════

class TestAllocation:
    def test_insufficient_evidence_is_neutral(self):
        a = L.learn(spread('S6', 'TRENDING_UP', [1.0, 0.5]), now=NOW)[('S6', 'TRENDING_UP')]
        assert a['state'] == 'unproven'
        assert a['scale'] == L.DEFAULT.unproven_scale

    def test_strong_arm_gets_more_than_weak(self):
        both = (spread('A', 'X', [0.9, 0.8, 1.0] * 12)
                + spread('B', 'X', [0.1, -0.1] * 18))
        res = L.learn(both, now=NOW)
        assert res[('A', 'X')]['scale'] > res[('B', 'X')]['scale']

    def test_cost_reduces_edge(self):
        o = spread('A', 'X', [0.3, 0.25, 0.35] * 12)
        free = L.learn(o, now=NOW, cfg=L.LearnConfig(cost_per_r=0.0))
        paid = L.learn(o, now=NOW, cfg=L.LearnConfig(cost_per_r=0.20))
        assert paid[('A', 'X')]['net_edge'] < free[('A', 'X')]['net_edge']

    def test_cost_changes_absolute_allocation(self):
        o = spread('A', 'X', [0.3, 0.25, 0.35] * 12)
        free = L.learn(o, now=NOW, cfg=L.LearnConfig(relative=False, cost_per_r=0.0))
        paid = L.learn(o, now=NOW, cfg=L.LearnConfig(relative=False, cost_per_r=0.20))
        assert paid[('A', 'X')]['scale'] < free[('A', 'X')]['scale']

    def test_cost_is_neutral_in_relative_mode(self):
        """상대 모드에서 비용은 배분 **비율**을 바꾸지 않는다.

        모든 팔이 같은 비용을 내므로 분자에서 상쇄된다. 비용은 '아예 거래할
        가치가 있는가'(격리)에만 관여한다. 이 성질을 모르면 cost_per_r을
        조정했는데 배분이 안 변한다고 버그로 오해한다.
        """
        o = spread('A', 'X', [0.5] * 30) + spread('B', 'Y', [0.1] * 30)
        lo = L.learn(o, now=NOW, cfg=L.LearnConfig(cost_per_r=0.0))
        hi = L.learn(o, now=NOW, cfg=L.LearnConfig(cost_per_r=0.03))
        for k in lo:
            if lo[k]['state'] != 'quarantined' and hi[k]['state'] != 'quarantined':
                assert lo[k]['scale'] == pytest.approx(hi[k]['scale'], abs=1e-9)

    def test_scale_respects_cap(self):
        o = spread('A', 'X', [5.0] * 40)
        a = L.learn(o, now=NOW)[('A', 'X')]
        assert a['scale'] <= L.DEFAULT.max_scale

    def test_scale_never_exceeds_cap_for_any_config(self):
        cfg = L.LearnConfig(max_scale=1.2, ref_edge=0.01)
        o = spread('A', 'X', [3.0] * 40)
        assert L.learn(o, now=NOW, cfg=cfg)[('A', 'X')]['scale'] <= 1.2

    def test_every_arm_reports_reason(self):
        o = spread('A', 'X', [0.5] * 30) + spread('B', 'Y', [1.0])
        for a in L.learn(o, now=NOW).values():
            assert a['reason'] and isinstance(a['reason'], str)

    def test_states_are_known(self):
        o = (spread('A', 'X', [1.0] * 40) + spread('B', 'Y', [-1.0] * 40)
             + spread('C', 'Z', [0.1]))
        valid = {'boosted', 'trusted', 'floored', 'quarantined', 'unproven'}
        assert {a['state'] for a in L.learn(o, now=NOW).values()} <= valid

    def test_floored_state_only_when_raw_below_floor(self):
        """클램프 후 값으로 판정하면 'floored'가 영원히 안 나온다(초안 결함)."""
        o = spread('A', 'X', [0.10, 0.09, 0.11] * 12)   # 비용 겨우 넘는 수준
        a = L.learn(o, now=NOW)[('A', 'X')]
        if a['state'] not in ('unproven', 'quarantined'):
            raw = a['net_edge'] / L.DEFAULT.ref_edge
            assert (a['state'] == 'floored') == (raw < L.DEFAULT.explore_floor)


# ══════════════════════════════════════════════════════════════
#  ⑥ 레짐 분리 — 이 모듈의 존재 이유
# ══════════════════════════════════════════════════════════════

class TestRegimeSeparation:
    def test_same_strategy_split_by_regime(self):
        """상승장에서 벌고 횡보장에서 잃는 전략은 **분리돼야** 한다.
        전략 단위로 뭉치면 상쇄돼 '평범한 전략'으로 보인다."""
        o = (spread('S6', 'TRENDING_UP', [1.0, 0.9, 1.1] * 12)
             + spread('S6', 'RANGING', [-1.0] * 36))
        res = L.learn(o, now=NOW)
        assert res[('S6', 'TRENDING_UP')]['scale'] > res[('S6', 'RANGING')]['scale']

    def test_blending_would_hide_the_signal(self):
        """레짐을 무시하면 두 팔이 하나로 합쳐져 신호가 사라진다는 것을 보인다."""
        good = [0.9] * 36
        bad = [-1.0] * 36
        blended = L.learn(spread('S6', 'ALL', good + bad), now=NOW)[('S6', 'ALL')]
        split = L.learn(spread('S6', 'TRENDING_UP', good)
                        + spread('S6', 'RANGING', bad), now=NOW)
        assert split[('S6', 'TRENDING_UP')]['scale'] > blended['scale']

    def test_empty_regime_becomes_unknown(self):
        o = [L.Observation('S6', '', 1.0, NOW)]
        assert ('S6', 'UNKNOWN') in L.learn(o, now=NOW)

    def test_scale_for_lookup(self):
        res = L.learn(spread('S6', 'TRENDING_UP', [0.5] * 40), now=NOW)
        assert L.scale_for(res, 'S6', 'TRENDING_UP') == res[('S6', 'TRENDING_UP')]['scale']

    def test_scale_for_unknown_arm_is_neutral(self):
        assert L.scale_for({}, 'S9', 'NOWHERE') == L.DEFAULT.unproven_scale


# ══════════════════════════════════════════════════════════════
#  ⑦ 결정론 · 견고성
# ══════════════════════════════════════════════════════════════

class TestRobustness:
    def test_deterministic(self):
        o = spread('A', 'X', [0.5, -0.3, 1.2] * 12)
        assert L.learn(o, now=NOW) == L.learn(o, now=NOW)

    def test_order_independent(self):
        """입력 순서가 **결정**을 바꾸면 안 된다.

        부동소수 합산 순서 때문에 마지막 자리(~1e-16)는 달라진다. 그건
        무해하므로 값 동일성이 아니라 배분·상태의 동일성을 고정한다.
        """
        o = spread('A', 'X', [0.5, -0.3, 1.2] * 12)
        a = L.learn(o, now=NOW)[('A', 'X')]
        b = L.learn(list(reversed(o)), now=NOW)[('A', 'X')]
        assert a['state'] == b['state']
        assert a['scale'] == pytest.approx(b['scale'], rel=1e-9)

    def test_empty_input(self):
        assert L.learn([], now=NOW) == {}

    def test_all_zero_returns(self):
        a = L.learn(spread('A', 'X', [0.0] * 40), now=NOW)[('A', 'X')]
        assert math.isfinite(a['scale'])

    def test_no_nan_anywhere(self):
        o = (spread('A', 'X', [0.0] * 40) + spread('B', 'Y', [1.0])
             + spread('C', 'Z', [-1.0] * 40) + spread('D', 'W', [3.0, -3.0] * 20))
        for k, a in L.learn(o, now=NOW).items():
            for f, v in a.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f'{k}.{f} = {v}'

    def test_extreme_outlier_does_not_explode(self):
        o = spread('A', 'X', [0.1] * 39 + [1e6])
        a = L.learn(o, now=NOW)[('A', 'X')]
        assert 0.0 <= a['scale'] <= L.DEFAULT.max_scale

    def test_now_defaults_to_utc_without_crashing(self):
        assert isinstance(L.learn(spread('A', 'X', [0.5] * 30)), dict)


# ══════════════════════════════════════════════════════════════
#  ⑧ DB 어댑터 · 리포트
# ══════════════════════════════════════════════════════════════

class TestDbAdapter:
    @pytest.fixture
    def db(self, tmp_path):
        import sqlite3
        p = tmp_path / 'spot.db'
        conn = sqlite3.connect(str(p))
        conn.execute('CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, '
                     'strategy TEXT, regime TEXT, pnl_r REAL, exit_ts TEXT, '
                     'dry_run INTEGER DEFAULT 0)')
        now = datetime.now(timezone.utc)
        rows = [('S6', 'TRENDING_UP', 1.0, now.isoformat(), 0),
                ('S6', 'TRENDING_UP', -1.0, now.isoformat(), 0),
                ('S4', 'RANGING', 0.5, now.isoformat(), 1),          # 드라이런
                ('S4', 'RANGING', 0.5, None, 0),                     # 잘못된 ts
                ('S5', 'RANGING', None, now.isoformat(), 0)]         # pnl_r 없음
        conn.executemany('INSERT INTO spot_trades (strategy, regime, pnl_r, '
                         'exit_ts, dry_run) VALUES (?,?,?,?,?)', rows)
        conn.commit(); conn.close()
        return p

    def test_reads_live_trades_only(self, db):
        o = L.load_observations(db)
        assert len(o) == 2
        assert all(x.strategy == 'S6' for x in o)

    def test_missing_db_is_empty(self, tmp_path):
        assert L.load_observations(tmp_path / 'nope.db') == []

    def test_bad_schema_is_empty(self, tmp_path):
        import sqlite3
        p = tmp_path / 'bad.db'
        sqlite3.connect(str(p)).execute('CREATE TABLE x (y INT)').connection.close()
        assert L.load_observations(p) == []

    def test_timestamps_are_tz_aware(self, db):
        assert all(x.exit_at.tzinfo is not None for x in L.load_observations(db))

    def test_parse_ts_variants(self):
        assert L._parse_ts('2026-01-01T00:00:00Z').tzinfo is not None
        assert L._parse_ts('2026-01-01T00:00:00').tzinfo is not None
        assert L._parse_ts('') is None
        assert L._parse_ts('garbage') is None
        assert L._parse_ts(None) is None


class TestReport:
    def test_prints(self, capsys):
        L.print_report(L.learn(spread('A', 'X', [0.5] * 40), now=NOW))
        assert '배분' in capsys.readouterr().out

    def test_empty_report_explains(self, capsys):
        L.print_report({})
        assert '관측 없음' in capsys.readouterr().out

    def test_quarantine_is_surfaced(self, capsys):
        L.print_report(L.learn(spread('A', 'X', [-1.0] * 40), now=NOW))
        assert '격리' in capsys.readouterr().out

    def test_cli_json(self, tmp_path):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / 'atlas_learning.py'),
             '--json', '--db', str(tmp_path / 'none.db')],
            capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == '{}'


class TestNoPep563:
    """이 모듈은 PEP 563(from __future__ import annotations)을 쓰면 안 된다.

    애노테이션이 전부 문자열이 되면 @dataclass가 KW_ONLY를 찾느라
    dataclasses._is_type()을 타는데, CPython 3.11의 그 함수는
    `sys.modules.get(cls.__module__).__dict__`를 방어 없이 호출한다.
    CI(3.11)에서 이 파일 수집 중 간헐적으로 터졌다 —
      AttributeError: 'NoneType' object has no attribute '__dict__'
    같은 커밋이 PR 실행은 통과하고 push 실행만 실패하는 플래키였고,
    플래키 게이트는 진짜 실패를 가린다.
    """

    def test_module_does_not_use_future_annotations(self):
        src = Path(L.__file__).read_text(encoding='utf-8')
        code = '\n'.join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith('#'))
        assert 'from __future__ import annotations' not in code, (
            'PEP 563을 켜면 @dataclass가 CPython 3.11의 취약한 _is_type 경로를 '
            '타 CI가 간헐적으로 실패한다')

    def test_dataclass_annotations_are_real_objects(self):
        """문자열이 아니어야 _is_type 경로가 아예 도달 불가다."""
        import dataclasses
        for cls in (L.LearnConfig, L.Observation):
            for f in dataclasses.fields(cls):
                assert not isinstance(f.type, str), (
                    f'{cls.__name__}.{f.name} 애노테이션이 문자열이다')

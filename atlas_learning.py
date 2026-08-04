"""
ATLAS Spot — 자기주도 학습 (Self-Directed Allocation Learner)
==============================================================
봇이 **자기 실적으로부터 배워** 자본 배분을 스스로 고치게 한다.

기존 학습 장치(Kelly, 건강도)의 한계
------------------------------------
둘 다 '전략' 단위로만 배운다. 그런데 이 봇은 이미 레짐별로 전략을
라우팅하고 있다 — 즉 **같은 전략이 장세에 따라 전혀 다른 성과**를 낸다.
전략 단위로 뭉뚱그리면 상승장에서 번 돈과 횡보장에서 잃은 돈이 상쇄돼
"평범한 전략" 하나로 보이고, 어느 쪽을 늘리고 줄일지 알 수 없다.

그 외 세 가지 결함:
  ① **불확실성을 모른다** — Kelly는 승률·손익비의 점추정을 참값처럼 쓴다.
     20건 표본의 표준오차는 매우 큰데, 아는 척하고 베팅한다.
  ② **시간을 모른다** — 200건 전 거래와 어제 거래의 가중치가 같다.
     시장은 비정상(non-stationary)이다.
  ③ **표본이 적을수록 과감해진다** — 우연히 3연승한 조합이 최고 성과로
     보인다. 다중검정 인플레와 같은 함정이다.

이 모듈의 접근
--------------
(전략 × 레짐)을 **팔(arm)** 로 두고, 각 팔의 *순* 기대 R을 추정한다.

  1. **시간 감쇠**   — 반감기 기반 지수가중. 오래된 증거는 저절로 힘을 잃는다.
  2. **수축(shrinkage)** — 표본이 적은 팔은 전체 평균 쪽으로 당겨진다.
     우연한 연승이 곧바로 큰 베팅으로 이어지지 않게 하는 장치다.
  3. **하한신뢰구간(LCB)** — 평균이 아니라 '확신할 수 있는 하한'으로 베팅한다.
     운이 아니라 **증거**가 있어야 자본을 준다.
  4. **비용 차감**   — 순엣지 = LCB − 왕복비용/R. 총이익이 아니라 순이익을
     최적화한다. (`capital_plan.cost_profile`과 같은 정의)

흡수상태(absorbing state)를 만들지 않는다
------------------------------------------
학습 시스템의 가장 위험한 실패는 **한 번 0을 주면 영원히 0**이 되는 것이다.
배분이 0이면 거래가 없고, 거래가 없으면 새 증거가 없고, 증거가 없으면
영원히 0이다. 이 모듈은 두 겹으로 막는다:

  - 배분 하한(`explore_floor`) — 확신이 없으면 완전히 끄지 않는다.
  - 격리(quarantine)는 **감쇠가 스스로 푼다** — 손실이 확실한 팔은 0으로
    격리되지만, 거래가 멈추면 남은 증거의 정보량이 감쇠해 최소치 아래로
    내려가고 자동으로 중립 복귀한다. 복귀 시점은 `days_until_revival()`로
    **정확히 계산된다**(특례 코드가 아니라 감쇠의 귀결이라, '복구 로직이
    안 도는' 실패가 원리적으로 없다).

    ⚠️ 여기서 '정보량'은 Kish 유효표본수(ESS)가 **아니라** 가중치 합 Σw다.
       ESS = (Σw)²/Σw² 는 가중치에 **비율 불변**이라, 모든 관측이 같은
       비율로 감쇠하면 값이 전혀 변하지 않는다(20건이 1년 뒤에도 19.8).
       초안에서 ESS로 게이트를 걸었다가 격리가 영구화되는 것을 수치로
       확인하고 고쳤다 — 막으려던 흡수상태를 정작 학습기가 갖고 있었다.
       ESS는 '가중치가 얼마나 쏠렸는가'를 재는 값이라 분산 불편보정에만
       쓰고, '증거가 얼마나 남았는가'는 Σw로 잰다.

왜 상대 배분인가 (측정으로 찾은 결함)
--------------------------------------
초안은 **절대 기준**으로 배분했다 — "확신 순엣지가 ref_edge(0.10R)면 1.0".
합성 시장으로 균등 배분과 비교해 보니 참값 +0.15R짜리 **확실히 수익인**
포트폴리오에서 학습기가 균등 대비 **38%** 밖에 못 벌었다. 원인:

  · 실전 규모의 표본(정보량 ~20)에서 LCB 페널티가 0.13R 정도인데,
    현실적인 엣지도 0.1~0.2R이다. 페널티가 엣지를 통째로 삼킨다.
  · 그 결과 증거가 **없는** 팔은 1.00, 증거를 45건 쌓은 팔은 0.25가 됐다.
    학습기가 '증거를 가진 죄'로 벌을 주는 셈이다.

그래서 배분을 **포트폴리오 평균 대비 상대값**으로 바꿨다:

    배분 = 1 + gain × (이 팔의 순엣지 − 전체 평균 순엣지) / 팔 간 표준편차

  · 팔이 전부 같으면 분자가 0 → 전부 정확히 1.00 → 균등 배분과 동일.
    **해를 끼칠 수 없다**(위 38% 같은 실패가 구조적으로 불가능).
  · 팔이 다르면 좋은 쪽 >1, 나쁜 쪽 <1로 **재배분**한다. 총 노출은
    대체로 유지되므로, 학습기를 켠다고 전체 리스크가 달라지지 않는다.
  · 분모가 데이터에서 오므로(τ) 튜닝 상수를 하나 줄인다.

'확실히 손실'인 팔의 격리는 여전히 **절대 기준**이다 — 전부 지고 있을 때
상대값만 보면 아무도 못 줄인다.

따름 성질: 상대 모드에서 `cost_per_r`은 **배분 비율을 바꾸지 않는다.**
모든 팔이 같은 비용을 내므로 분자에서 상쇄되기 때문이다. 비용은 오직
'아예 거래할 가치가 있는가'(격리)에만 관여한다. 절대 모드에서는 배분에
직접 영향을 준다.

이 모듈은 **계산만** 한다. 라이브 사이징에 자동으로 반영되지 않는다.
반영 여부는 운영자가 결정한다(`--apply` 아님 — config 배선이 필요).

사용법:
  python atlas_learning.py                 # 학습 결과 리포트
  python atlas_learning.py --half-life 30
  python atlas_learning.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'LEARN')

sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LearnConfig:
    """학습기 하이퍼파라미터.

    기본값은 **보수적** 으로 잡았다. 학습기가 틀렸을 때의 비용이
    맞았을 때의 이득보다 크기 때문이다(자본은 복리로 줄어든다).
    """
    half_life_days:  float = 45.0   # 증거의 반감기. 시장 비정상성 대응
    shrink_k:        float = 12.0   # 수축 강도 **기본값**(가상 관측 수).
                                    # τ를 추정할 수 없을 때(팔 1개 등)만 쓴다.
                                    # ⚠️ 상한으로 쓰면 안 된다 — 차이가 전부
                                    # 잡음일 때 필요한 k는 수천 단위인데
                                    # 12로 자르면 수축이 사실상 무효가 되고,
                                    # 참값이 같은 팔들이 0.25 대 1.50으로 갈렸다
    max_shrink_k:    float = 5000.0 # 발산 방지용 절대 상한(τ→0에서 k→∞)
    adaptive_shrink: bool  = True   # James–Stein 방식으로 수축 강도 자동 추정
    min_info:        float = 8.0    # 이보다 정보량(Σw)이 적으면 개입하지 않는다.
                                    # ESS가 아니라 Σw인 이유는 모듈 docstring 참조
    z:               float = 1.0    # 신뢰구간 폭(1.0 ≈ 84% 단측). 음수 불가
    relative:        bool  = True   # 배분을 **포트폴리오 평균 대비**로 정한다.
                                    # 아래 '왜 상대 배분인가' 참조
    gain:            float = 0.50   # 상대 모드 민감도. 팔 간 차이 1σ당 배분 변화
    ref_edge:        float = 0.10   # 절대 모드의 '배분 1.0' 기준 순엣지(R).
                                    # 상대 모드에서는 분모 하한으로만 쓴다
    explore_floor:   float = 0.25   # 배분 하한 — 흡수상태 방지
    max_scale:       float = 1.50   # 배분 상한. explore_floor 이상이어야 한다
    unproven_scale:  float = 0.25   # 표본 부족 팔의 배분.
                                    # 운영자 결정: 증명 전엔 작게, 증거가
                                    # 쌓이면 키운다. 1.00이면 격리 해제 직후
                                    # 0.00 → 1.00으로 점프하는 구간이 생긴다.
                                    # ⚠️ 소액 계좌에서는 주문이 거래소 최소액
                                    # 아래로 내려갈 수 있다 — capital_plan.py로
                                    # 문턱을 확인하고 켤 것
    cost_per_r:      float = 0.04   # 수수료가 1R에서 차지하는 비중.
                                    # ⚠️ 슬리피지·스프레드는 **넣지 말 것** —
                                    # pnl_r은 실제 체결가로 계산되므로 이미
                                    # 반영돼 있다. 왕복비용 전체(0.08)를 빼면
                                    # 슬리피지를 두 번 세어 엣지를 과소평가한다
    quarantine:      bool  = True   # 손실이 확실한 팔을 0으로 격리할지

    def __post_init__(self):
        """오설정을 조용히 통과시키지 않는다.

        학습기의 잘못된 설정은 예외를 던지지 않고 **그냥 돈을 잃는다**.
        특히 max_scale < explore_floor면 `min(cap, max(floor, raw))`가
        하한을 조용히 무시해, 흡수상태 방지 장치가 무력화된다.
        """
        problems = []
        if self.max_scale < self.explore_floor:
            problems.append(
                f'max_scale({self.max_scale}) < explore_floor({self.explore_floor}) '
                f'— 상한이 하한을 덮어써 탐색 하한이 무효가 된다')
        if self.z < 0:
            problems.append(f'z({self.z}) < 0 — 신뢰구간이 뒤집혀 '
                            f'불확실할수록 크게 베팅하게 된다')
        if self.explore_floor < 0:
            problems.append(f'explore_floor({self.explore_floor}) < 0')
        if self.ref_edge < 0:
            problems.append(f'ref_edge({self.ref_edge}) < 0 — 엣지 부호가 뒤집힌다')
        if self.min_info <= 0:
            problems.append(f'min_info({self.min_info}) <= 0 — 증거 없이 개입하게 된다')
        if self.shrink_k < 0:
            problems.append(f'shrink_k({self.shrink_k}) < 0')
        if self.max_shrink_k < self.shrink_k:
            problems.append(
                f'max_shrink_k({self.max_shrink_k}) < shrink_k({self.shrink_k}) '
                f'— 기본 수축 강도조차 상한에 잘린다')
        if problems:
            raise ValueError('LearnConfig 오설정: ' + '; '.join(problems))


DEFAULT = LearnConfig()


# ══════════════════════════════════════════════════════════════
#  관측
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Observation:
    """청산된 거래 하나. 학습기의 유일한 입력."""
    strategy: str
    regime:   str
    pnl_r:    float
    exit_at:  datetime

    def age_days(self, now: datetime) -> float:
        return max(0.0, (now - self.exit_at).total_seconds() / 86400.0)

    def is_valid(self) -> bool:
        """NaN·무한대를 걸러낸다.

        **조용한 오염이 가장 위험하다.** NaN이 하나 섞이면 평균이 NaN이 되고,
        NaN과의 비교는 전부 False라 격리 판정을 그냥 통과한다. 그 다음
        `max(floor, nan)`이 floor를 돌려주므로 최종 배분은 0.25처럼
        **멀쩡해 보이는 값**이 된다 — 오염됐다는 신호가 어디에도 없다.
        """
        return math.isfinite(self.pnl_r)


def arm_key(strategy: str, regime: str) -> tuple:
    return (strategy, regime or 'UNKNOWN')


# ══════════════════════════════════════════════════════════════
#  가중치 / 통계
# ══════════════════════════════════════════════════════════════

def decay_weight(age_days: float, half_life_days: float) -> float:
    """반감기 지수 감쇠. 나이가 반감기면 0.5."""
    if half_life_days <= 0:
        return 1.0
    if age_days <= 0:
        return 1.0
    return float(2.0 ** (-age_days / half_life_days))


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish 유효표본수: (Σw)² / Σw².

    가중치가 모두 같으면 n과 일치하고, 한 건에 쏠릴수록 1에 가까워진다.
    즉 **가중치의 쏠림**을 재는 값이다.

    ⚠️ 시간 경과를 재지 **못한다** — 분자·분모가 모두 2차 동차라
    모든 가중치에 같은 상수를 곱해도 값이 변하지 않는다. 20건이 1년 뒤에도
    19.8로 그대로다. '증거가 얼마나 남았는가'는 `Σw`(정보량)로 재야 한다.
    이 함수는 분산 불편보정에만 쓴다.
    """
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def weighted_stats(values: Sequence[float],
                   weights: Sequence[float]) -> tuple[float, float, float]:
    """(가중평균, 가중표준편차, 유효표본수).

    표준편차는 유효표본 기반 불편보정을 쓴다. ESS ≤ 1이면 분산을 추정할
    수 없으므로 0.0을 반환한다 — 이때 호출자는 **불확실성을 0으로 믿으면
    안 된다**(아래 `fit_arm`이 최소표본 게이트로 막는다).
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0, 0.0, 0.0
    s1 = sum(weights)
    if s1 <= 0:
        return 0.0, 0.0, 0.0
    mean = sum(v * w for v, w in zip(values, weights, strict=True)) / s1
    ess = effective_sample_size(weights)
    if ess <= 1.0:
        return float(mean), 0.0, float(ess)
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights, strict=True)) / s1
    var *= ess / (ess - 1.0)          # 불편보정
    return float(mean), float(math.sqrt(max(var, 0.0))), float(ess)


# ══════════════════════════════════════════════════════════════
#  팔 적합
# ══════════════════════════════════════════════════════════════

def fit_arm(obs: Sequence[Observation], now: datetime,
            cfg: LearnConfig = DEFAULT) -> dict:
    """한 팔의 감쇠가중 통계.

    `info`(=Σw)와 `ess`(Kish)를 **둘 다** 낸다. 둘은 다른 것을 잰다:
      info — 남은 증거의 양. 시간이 지나면 줄어든다. 게이트·수축에 쓴다.
      ess  — 가중치 쏠림. 분산 불편보정에만 쓴다.
    """
    ws = [decay_weight(o.age_days(now), cfg.half_life_days) for o in obs]
    rs = [o.pnl_r for o in obs]
    mean, sd, ess = weighted_stats(rs, ws)
    return {
        'n_raw':  len(obs),
        'info':   float(sum(ws)),
        'ess':    ess,
        'mean_r': mean,
        'sd_r':   sd,
    }


def pooled_prior(arms: dict, cfg: LearnConfig = DEFAULT) -> float:
    """전체 팔을 합친 평균 R — 수축의 목표점.

    개별 팔이 아니라 **포트폴리오 전체가 어떤 상태인가**를 나타낸다.
    전부 지고 있으면 prior도 음수라, 표본이 적은 팔을 낙관적으로
    끌어올리지 않는다(이게 고정 prior 0.0보다 안전한 이유다).
    """
    num = sum(a['mean_r'] * a['info'] for a in arms.values() if a['info'] > 0)
    den = sum(a['info'] for a in arms.values() if a['info'] > 0)
    return float(num / den) if den > 0 else 0.0


def between_arm_variance(arms: dict) -> tuple:
    """(τ², 표집분산 V, 정보량가중 평균 μ) — 팔 간 차이의 분산분해.

        관측된 팔 간 분산 T  =  진짜 차이 τ²  +  표집오차 V

    τ² = max(T − V, 0)이 '진짜 차이'의 추정치다. 수축 강도와 상대 배분의
    분모가 **같은 τ**를 써야 한다 — 다른 값을 쓰면 한쪽이 줄인 차이를
    다른 쪽이 되살려 수축이 무효가 된다.
    """
    usable = [a for a in arms.values() if a.get('info', 0) > 0]
    if len(usable) < 2:
        return 0.0, 0.0, 0.0
    tot = sum(a['info'] for a in usable)
    mu = sum(a['mean_r'] * a['info'] for a in usable) / tot
    t_obs = sum(a['info'] * (a['mean_r'] - mu) ** 2 for a in usable) / tot
    v_samp = sum(a['sd_r'] ** 2 for a in usable) / len(usable)
    return float(max(t_obs - v_samp / max(tot / len(usable), 1e-9), 0.0)), \
           float(v_samp), float(mu)


def estimate_shrink_k(arms: dict, cfg: LearnConfig = DEFAULT) -> float:
    """수축 강도를 팔들의 **실제 분산**에서 추정한다 (James–Stein).

    고정 k의 문제: k를 크게 잡으면 진짜로 다른 팔까지 전체 평균으로
    끌려간다. 손실 팔이 여럿 있으면 **잘 버는 팔의 자본이 깎인다** —
    이 모듈이 하려던 일의 정반대다. 반대로 작게 잡으면 우연한 연승이
    곧바로 큰 베팅이 된다.

    해법은 데이터에게 묻는 것이다. 분산분해:

        관측된 팔 간 분산 T  =  진짜 차이 τ²  +  표집오차 V

    τ² = max(T − V, 0) 이 '진짜 차이'의 추정치다.
      · 팔들이 실제로 다르다(τ² 큼) → 수축을 약하게
      · 차이가 잡음으로 설명된다(τ² ≈ 0) → 수축을 세게

    개별 팔의 수축계수 B = v/(v+τ²)를 k 표기로 옮기면 k = sd²/τ² 이다.
    (v = sd²/info 이므로 k = info·B/(1−B) = sd²/τ²)

    팔이 하나뿐이면 비교 대상이 없어 τ²를 추정할 수 없다 → 고정값 사용.
    """
    if not cfg.adaptive_shrink:
        return max(cfg.shrink_k, 0.0)
    # '팔이 하나뿐이라 τ를 못 구한다'와 'τ를 구했더니 0이다'는 전혀 다르다.
    # 전자는 비교 대상이 없는 것이고(기본값), 후자는 차이가 잡음이라는
    # 적극적 증거다(최대 수축). 둘을 뭉치면 팔 하나짜리 계좌가 이유 없이
    # 최대 수축을 맞는다.
    if len([a for a in arms.values() if a.get('info', 0) > 0]) < 2:
        return max(cfg.shrink_k, 0.0)
    tau2, v_samp, _ = between_arm_variance(arms)
    if tau2 <= 0:
        return float(cfg.max_shrink_k)     # 차이가 전부 잡음 → 최대 수축
    # 차이가 잡음으로 설명될수록 k가 커진다(수천까지). 여기서 cfg.shrink_k로
    # 자르면 정작 필요한 강한 수축이 막힌다 — 발산 방지 상한만 건다.
    return float(min(cfg.max_shrink_k, max(v_samp / tau2, 0.0)))


def shrink_mean(mean_r: float, info: float, prior: float,
                cfg: LearnConfig = DEFAULT,
                k_override: Optional[float] = None) -> float:
    """경험적 베이즈 수축: 증거가 적을수록 전체 평균 쪽으로 당긴다.

    (info·자기평균 + k·전체평균) / (info + k)
    info → ∞ 이면 자기평균, info → 0 이면 전체평균.
    """
    k = max(cfg.shrink_k, 0.0) if k_override is None else max(k_override, 0.0)
    if info + k <= 0:
        return prior
    return float((mean_r * info + prior * k) / (info + k))


def confidence_bounds(shrunk: float, sd: float, info: float,
                      cfg: LearnConfig = DEFAULT,
                      k_override: Optional[float] = None) -> tuple[float, float]:
    """(하한, 상한). 수축 후 정보량(info + k)으로 표준오차를 잡는다.

    수축이 이미 분산을 줄였으므로 분모에 k를 더하지 않으면 불확실성을
    이중으로 세게 된다. 증거가 낡을수록 info가 줄어 구간이 **넓어진다** —
    "예전엔 알았지만 지금은 모른다"가 자동으로 표현된다.
    """
    k = max(cfg.shrink_k, 0.0) if k_override is None else max(k_override, 0.0)
    denom = max(info + k, 1e-9)
    se = sd / math.sqrt(denom)
    return float(shrunk - cfg.z * se), float(shrunk + cfg.z * se)


# ══════════════════════════════════════════════════════════════
#  배분
# ══════════════════════════════════════════════════════════════

def allocation_scale(stats: dict, prior: float,
                     cfg: LearnConfig = DEFAULT,
                     k_shrink: Optional[float] = None,
                     ref: tuple = (0.0, 1.0)) -> dict:
    """한 팔의 배분 배수와 그 근거.

    반환 dict의 `scale`이 결론이고 나머지는 **왜 그렇게 됐는지**다.
    학습기가 설명 불가능하면 운영자는 끌 수밖에 없으므로 근거를 함께 낸다.
    """
    info, mean, sd = stats['info'], stats['mean_r'], stats['sd_r']
    out = {
        **stats, 'prior': prior,
        'shrunk_r': mean, 'lcb': mean, 'ucb': mean,
        'net_edge': 0.0, 'scale': cfg.unproven_scale,
        'state': 'unproven',
        'reason': f'정보량 {info:.1f} < 최소 {cfg.min_info:.1f} — 개입하지 않음',
    }
    if info < cfg.min_info:
        return out

    shrunk = shrink_mean(mean, info, prior, cfg, k_shrink)
    lcb, ucb = confidence_bounds(shrunk, sd, info, cfg, k_shrink)
    out['shrink_k'] = float(max(cfg.shrink_k, 0.0) if k_shrink is None
                            else max(k_shrink, 0.0))
    net_edge = lcb - cfg.cost_per_r
    out.update({'shrunk_r': shrunk, 'lcb': lcb, 'ucb': ucb,
                'net_edge': net_edge})

    # 손실이 **확실한** 팔 — 상한마저 비용을 못 넘는다
    if cfg.quarantine and (ucb - cfg.cost_per_r) < 0:
        out.update({
            'scale': 0.0, 'state': 'quarantined',
            'reason': (f'상한 {ucb:+.3f}R 마저 비용 {cfg.cost_per_r:.3f}R 미달 — '
                       f'격리. 거래가 멈추면 {days_until_revival(out, cfg):.0f}일 뒤 '
                       f'정보량이 {cfg.min_info:.1f} 아래로 내려가 자동 복귀한다'),
        })
        return out

    if cfg.relative:
        # 포트폴리오 평균 대비 몇 σ인가 → 1.0 기준으로 가감
        raw = 1.0 + cfg.gain * (net_edge - ref[0]) / ref[1]
    else:
        raw = net_edge / cfg.ref_edge if cfg.ref_edge > 0 else 0.0
    scale = min(cfg.max_scale, max(cfg.explore_floor, raw))
    # 클램프 뒤에는 scale >= explore_floor가 항상 참이므로, '하한에 걸렸는가'는
    # 클램프 **전** 값(raw)으로 판정해야 한다.
    state = ('boosted' if scale > 1.0
             else 'trusted' if raw >= cfg.explore_floor
             else 'floored')
    if cfg.relative:
        reason = (f'확신 순엣지 {net_edge:+.3f}R, 전체 평균 {ref[0]:+.3f}R '
                  f'대비 {(net_edge - ref[0]) / ref[1]:+.2f}σ (σ={ref[1]:.3f}) '
                  f'→ {scale:.2f}')
    else:
        reason = (f'확신 순엣지 {net_edge:+.3f}R / 기준 {cfg.ref_edge:.3f}R '
                  f'= {raw:.2f} → {scale:.2f}')
    out.update({'scale': float(scale), 'state': state, 'reason': reason})
    return out


def edge_reference(arms: dict, cfg: LearnConfig = DEFAULT,
                   tau: Optional[float] = None) -> tuple:
    """상대 배분의 기준 (평균 순엣지, 정규화 분모).

    분모는 **수축 후 관측된 퍼짐이 아니라** 추정된 진짜 차이 τ다.
    수축 후 퍼짐으로 나누면, 수축이 눌러 놓은 차이를 도로 부풀려
    수축이 통째로 무효가 된다(참값이 같은 팔들이 0.36 대 1.5로 갈렸다).

    τ가 0에 가까우면(차이가 전부 잡음) 분자도 0에 가까우므로 배분은 1.0
    근처가 된다. 0으로 나누는 것만 `ref_edge` 하한으로 막는다.
    """
    usable = [a for a in arms.values()
              if a['state'] not in ('unproven',) and a['info'] > 0]
    if not usable:
        return 0.0, max(cfg.ref_edge, 1e-9)
    tot = sum(a['info'] for a in usable)
    mean = sum(a['net_edge'] * a['info'] for a in usable) / tot
    if tau is None:
        tau = math.sqrt(between_arm_variance(arms)[0])
    return float(mean), float(max(tau, cfg.ref_edge))


def learn(observations: Iterable[Observation], now: Optional[datetime] = None,
          cfg: LearnConfig = DEFAULT) -> dict:
    """관측 전체 → {(전략, 레짐): 배분 정보}.

    `now`를 인자로 받는 이유: 시간을 내부에서 읽으면 테스트가 시계에
    의존해 재현되지 않는다. 감쇠 로직은 시간에 전적으로 의존하므로
    **결정론적으로 검증 가능**해야 한다.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    buckets: dict = {}
    for o in observations:
        if not o.is_valid():          # NaN/inf는 조용히 배분을 오염시킨다
            continue
        buckets.setdefault(arm_key(o.strategy, o.regime), []).append(o)

    stats = {k: fit_arm(v, now, cfg) for k, v in buckets.items()}
    prior = pooled_prior(stats, cfg)
    k_shrink = estimate_shrink_k(stats, cfg)
    # 상대 배분은 '전체 대비 어디쯤인가'가 필요하므로 2패스로 계산한다.
    # 1패스: 기준 없이 순엣지만 구한다. 2패스: 그 분포로 배분을 정한다.
    first = {k: allocation_scale(s, prior, cfg, k_shrink) for k, s in stats.items()}
    tau = math.sqrt(between_arm_variance(stats)[0])
    ref = edge_reference(first, cfg, tau)
    return {k: allocation_scale(s, prior, cfg, k_shrink, ref)
            for k, s in stats.items()}


def scale_for(result: dict, strategy: str, regime: str,
              cfg: LearnConfig = DEFAULT) -> float:
    """라이브 사이징이 호출할 단일 진입점. 미학습 팔은 중립."""
    arm = result.get(arm_key(strategy, regime))
    return float(arm['scale']) if arm else cfg.unproven_scale


# ══════════════════════════════════════════════════════════════
#  복구 보장 — 격리가 언제 풀리는지 계산
# ══════════════════════════════════════════════════════════════

def days_until_revival(stats: dict, cfg: LearnConfig = DEFAULT) -> float:
    """추가 거래 없이 정보량이 최소치 아래로 내려가기까지의 일수.

    격리가 **반드시 풀린다**는 것을 숫자로 증명하는 함수다.
    거래가 멈추면 새 관측이 없으므로 모든 가중치가 반감기 h로 감쇠하고,
    정보량은 info(t) = info₀ · 2^(−t/h) 이다. 이것을 min_info와 같게 두면

        t = h · log₂(info₀ / min_info)

    이 값은 항상 **유한**하다 — 격리는 영구화될 수 없다.
    (초안은 Kish ESS로 게이트를 걸어 이 성질이 없었다. ESS는 비율 불변이라
     아무리 시간이 지나도 값이 그대로여서 격리가 영영 풀리지 않았다)
    """
    info = float(stats.get('info', 0.0))
    if info <= cfg.min_info:
        return 0.0
    if cfg.half_life_days <= 0 or cfg.min_info <= 0:
        return math.inf          # 감쇠가 없으면 복귀도 없다 — 정직하게 알린다
    return float(cfg.half_life_days * math.log2(info / cfg.min_info))


# ══════════════════════════════════════════════════════════════
#  DB 어댑터
# ══════════════════════════════════════════════════════════════

def load_observations(db_path: Optional[Path] = None,
                      lookback_days: int = 365) -> list[Observation]:
    """거래 DB에서 관측을 읽는다. 드라이런은 제외한다."""
    import sqlite3
    if db_path is None:
        from atlas_spot_config import SPOT_DB_FILE
        db_path = Path(SPOT_DB_FILE)
    if not Path(db_path).exists():
        return []
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT strategy, regime, pnl_r, exit_ts FROM spot_trades '
            'WHERE COALESCE(dry_run,0)=0 AND pnl_r IS NOT NULL '
            'AND exit_ts >= ? ORDER BY id DESC LIMIT 5000', (since,)
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    out = []
    for r in rows:
        ts = _parse_ts(r['exit_ts'])
        if ts is None:
            continue
        out.append(Observation(
            strategy=r['strategy'] or '', regime=r['regime'] or 'UNKNOWN',
            pnl_r=float(r['pnl_r']), exit_at=ts))
    return out


def _parse_ts(raw) -> Optional[datetime]:
    """ISO 문자열 → tz-aware datetime. 실패하면 None(조용히 버린다)."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════
#  리포트
# ══════════════════════════════════════════════════════════════

_STATE_MARK = {
    'boosted':     '▲',
    'trusted':     ' ',
    'floored':     '▽',
    'quarantined': '✗',
    'unproven':    '·',
}


def print_report(result: dict, cfg: LearnConfig = DEFAULT) -> None:
    w = 88
    print('═' * w)
    print(f'  ATLAS 자기주도 학습 — (전략 × 레짐) 배분')
    # 적응 수축이 켜져 있으면 **실제로 쓰인** k를 보여야 한다.
    # 설정값을 그대로 찍으면 운영자는 튜닝하지도 않은 값을 원인으로 오해한다.
    k_used = next((a['shrink_k'] for a in result.values() if 'shrink_k' in a), None)
    k_txt = (f'수축 k={k_used:.1f}(자동)'
             if cfg.adaptive_shrink and k_used is not None
             else f'수축 k={cfg.shrink_k:.0f}(고정)')
    mode_txt = '상대배분' if cfg.relative else '절대배분'
    print(f'  {mode_txt} · 반감기 {cfg.half_life_days:.0f}일 · {k_txt} · '
          f'최소정보량 {cfg.min_info:.0f} · 비용 {cfg.cost_per_r:.3f}R')
    # τ = 팔 간 '진짜' 성과 차이. **학습기의 가치를 결정하는 단 하나의 변수**다.
    # τ가 작으면 재배분할 것이 없어 학습기는 기껏해야 중립이고, 격리가 켜져
    # 있으면 오히려 손해다. 합성 시장 기준 손익분기는 τ ≈ 0.25R.
    tau = math.sqrt(between_arm_variance(result)[0]) if len(result) >= 2 else 0.0
    verdict = ('재배분 가치 있음' if tau >= 0.25 else
               '차이가 작다 — 학습기 이득이 거의 없거나 마이너스')
    print(f'  팔 간 실제 차이 τ = {tau:.3f}R  →  {verdict}')
    print('═' * w)
    if not result:
        print('\n  관측 없음 — 학습할 거래 기록이 없다. 배분은 전부 중립(1.00)이다.')
        print('═' * w)
        return

    print(f'\n  {"팔":<22}{"건수":>5}{"유효":>7}{"평균R":>8}'
          f'{"수축R":>8}{"하한R":>8}{"순엣지":>8}{"배분":>7}')
    print(f'  {"─" * (w - 4)}')
    for (sid, rg), a in sorted(result.items(),
                               key=lambda kv: -kv[1]['scale']):
        mark = _STATE_MARK.get(a['state'], ' ')
        print(f'  {mark}{sid}/{rg:<17}{a["n_raw"]:>5}{a["info"]:>7.1f}'
              f'{a["mean_r"]:>+8.3f}{a["shrunk_r"]:>+8.3f}{a["lcb"]:>+8.3f}'
              f'{a["net_edge"]:>+8.3f}{a["scale"]:>7.2f}')

    print(f'\n  근거')
    for (sid, rg), a in sorted(result.items(), key=lambda kv: -kv[1]['scale']):
        print(f'   {sid}/{rg}: {a["reason"]}')

    q = [k for k, a in result.items() if a['state'] == 'quarantined']
    if q:
        print(f'\n  ⚠ 격리 {len(q)}개: {", ".join(f"{s}/{r}" for s, r in q)}')
        print(f'    배분 0이면 새 증거가 생기지 않는다. 이 상태가 오래가면 '
              f'운영자가 원인을 봐야 한다.')
    print('═' * w)


def main() -> None:
    ap = argparse.ArgumentParser(description='ATLAS 자기주도 학습')
    ap.add_argument('--half-life', type=float, default=DEFAULT.half_life_days)
    ap.add_argument('--min-info', type=float, default=DEFAULT.min_info)
    ap.add_argument('--cost-per-r', type=float, default=DEFAULT.cost_per_r)
    ap.add_argument('--lookback', type=int, default=365)
    ap.add_argument('--db', type=str, default=None)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()

    cfg = LearnConfig(half_life_days=a.half_life, min_info=a.min_info,
                      cost_per_r=a.cost_per_r)
    obs = load_observations(Path(a.db) if a.db else None, a.lookback)
    res = learn(obs, cfg=cfg)
    if a.json:
        print(json.dumps({f'{s}/{r}': v for (s, r), v in res.items()},
                         ensure_ascii=False, indent=2, default=str))
    else:
        print_report(res, cfg)


if __name__ == '__main__':
    main()

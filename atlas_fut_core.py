"""
ATLAS Futures — 핵심 규칙 (순수 파이썬, 표준 라이브러리만 사용)
================================================================
라이브 엔진(atlas_fut_main.py)과 백테스트(atlas_fut_backtest.py)가 **같은 함수**를
쓴다. 규칙이 두 벌이면 검증한 적 없는 봇이 된다 — 현물 봇에서 실제로 겪은 문제다
(레짐 EMA200 계산 기간, 하루 앞선 레짐, S3 쿨다운이 라이브·백테스트에서 달랐다).

pandas·numpy를 쓰지 않는 이유:
  · 서버(1 vCPU / 1GB)에서 현물 봇과 메모리를 나눠 쓴다
  · 표준 라이브러리만으로 테스트가 돌아 어디서든 검증할 수 있다
  · 지표 계산 순서를 직접 통제해 백테스트·라이브 결과를 비트 단위로 맞춘다

판정은 모두 **완성봉**으로만 한다. 형성 중인 봉은 호출하는 쪽에서 잘라서 넘긴다.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import atlas_fut_config as cfg

NAN = float('nan')
DAY_MS = 86_400_000
H4_MS = 14_400_000

TRENDING_UP = 'TRENDING_UP'
TRENDING_DOWN = 'TRENDING_DOWN'
RANGING = 'RANGING'
WEAK_TREND = 'WEAK_TREND'
MICRO_RANGING = 'MICRO_RANGING'
CRISIS = 'CRISIS'
UNKNOWN = 'UNKNOWN'


@dataclass(slots=True)
class Bar:
    ts: int          # 봉 시작 시각(ms, UTC)
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass(slots=True)
class Signal:
    strategy: str
    side: int        # +1 롱 / -1 숏
    sl: float        # 신호봉 기준 손절가 (진입 시 '거리'를 유지한 채 체결가로 옮긴다)
    rr: float        # 손익비 (익절 = 체결가 ± 손절거리 × rr)
    ref: float       # 신호봉 종가
    atr: float


# ══════════════════════════════════════════════════════════════
#  수치 유틸
# ══════════════════════════════════════════════════════════════

def js_round2(x: float) -> float:
    """소수 둘째 자리 반올림(0.5는 올림). round()는 은행가 반올림이라 쓰지 않는다."""
    return math.floor(x * 100 + 0.5) / 100


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _plain_sum(xs: Sequence[float]) -> float:
    """왼쪽부터 단순 누적. 파이썬 3.12+의 sum()은 보정 합산이라 버전마다 값이 달라진다."""
    s = 0.0
    for x in xs:
        s += x
    return s


# ══════════════════════════════════════════════════════════════
#  지표
# ══════════════════════════════════════════════════════════════

def sma(x: Sequence[float], n: int) -> list[float]:
    """단순이동평균. NaN을 만나면 창을 처음부터 다시 채운다."""
    out = [NAN] * len(x)
    s = 0.0
    valid = 0
    for i, v in enumerate(x):
        if math.isnan(v):
            s = 0.0
            valid = 0
            continue
        s += v
        valid += 1
        if valid > n:
            s -= x[i - n]
            valid = n
        if valid == n:
            out[i] = s / n
    return out


def rolling_std(x: Sequence[float], n: int) -> list[float]:
    """표본 표준편차(ddof=1)."""
    out = [NAN] * len(x)
    for i in range(n - 1, len(x)):
        m = _plain_sum(x[i - n + 1:i + 1]) / n
        ss = 0.0
        for k in range(i - n + 1, i + 1):
            ss += (x[k] - m) ** 2
        out[i] = math.sqrt(ss / (n - 1))
    return out


def ema(x: Sequence[float], span: int) -> list[float]:
    """지수이동평균(adjust=False). 첫 값에서 시작한다 — 시작점이 다르면 값이 다르다."""
    if not x:
        return []
    a = 2 / (span + 1)
    out = [0.0] * len(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1 - a) * out[i - 1] + a * x[i]
    return out


def ewm_alpha(x: Sequence[float], a: float) -> list[float]:
    out = [NAN] * len(x)
    prev = NAN
    for i, v in enumerate(x):
        if math.isnan(v):
            out[i] = prev
            continue
        prev = v if math.isnan(prev) else (1 - a) * prev + a * v
        out[i] = prev
    return out


def rolling_max(x: Sequence[float], n: int) -> list[float]:
    out = [NAN] * len(x)
    for i in range(n - 1, len(x)):
        out[i] = max(x[i - n + 1:i + 1])
    return out


def rolling_min(x: Sequence[float], n: int) -> list[float]:
    out = [NAN] * len(x)
    for i in range(n - 1, len(x)):
        out[i] = min(x[i - n + 1:i + 1])
    return out


def true_range(bars: Sequence[Bar]) -> list[float]:
    tr = [0.0] * len(bars)
    for i, b in enumerate(bars):
        if i == 0:
            tr[i] = b.h - b.l
        else:
            pc = bars[i - 1].c
            tr[i] = max(b.h - b.l, abs(b.h - pc), abs(b.l - pc))
    return tr


def atr(bars: Sequence[Bar], n: int = cfg.ATR_PERIOD) -> list[float]:
    return sma(true_range(bars), n)


def adx_series(bars: Sequence[Bar], p: int = cfg.ATR_PERIOD) -> list[float]:
    """지수평활 ADX 시계열 (동적 손익비 계산용 — 현물 S6와 같은 식)."""
    n = len(bars)
    dmp = [0.0] * n
    dmm = [0.0] * n
    for i in range(1, n):
        up = bars[i].h - bars[i - 1].h
        dn = bars[i - 1].l - bars[i].l
        dmp[i] = up if (up > dn and up > 0) else 0.0
        dmm[i] = dn if (dn > up and dn > 0) else 0.0
    a = 1 / p
    ts = ewm_alpha(true_range(bars), a)
    ps = ewm_alpha(dmp, a)
    ms = ewm_alpha(dmm, a)
    dx = [NAN] * n
    for i in range(n):
        t = ts[i]
        if not t or math.isnan(t):
            continue
        dip = 100 * ps[i] / t
        dim = 100 * ms[i] / t
        den = dip + dim
        if den and not math.isnan(den):
            dx[i] = 100 * abs(dip - dim) / den
    return ewm_alpha(dx, a)


def calc_adx(bars: Sequence[Bar], period: int = 14) -> float:
    """Wilder ADX 단일값 (장세 판정용 — atlas_indicators.calc_adx와 같은 식).

    데이터가 period×3봉보다 짧으면 0.0. 호출측은 봉 수를 먼저 확인할 것 —
    0.0은 '추세 없음(RANGING)'과 구분되지 않는다.
    """
    if len(bars) < period * 3:
        return 0.0
    dmp: list[float] = []
    dmm: list[float] = []
    trs: list[float] = []
    for i in range(1, len(bars)):
        up = bars[i].h - bars[i - 1].h
        down = bars[i - 1].l - bars[i].l
        dmp.append(up if (up > down and up > 0) else 0.0)
        dmm.append(down if (down > up and down > 0) else 0.0)
        trs.append(max(bars[i].h - bars[i].l, abs(bars[i].h - bars[i - 1].c), abs(bars[i].l - bars[i - 1].c)))

    def wilder(lst: list[float], p: int) -> list[float]:
        s = _plain_sum(lst[:p])
        res = [s]
        for v in lst[p:]:
            s = s - s / p + v
            res.append(s)
        return res

    tw = wilder(trs, period)
    pw = wilder(dmp, period)
    mw = wilder(dmm, period)
    dip = [100 * d / t if t else 0.0 for d, t in zip(pw, tw, strict=True)]
    dim = [100 * d / t if t else 0.0 for d, t in zip(mw, tw, strict=True)]
    dx = [abs(p - m) / (p + m) * 100 if (p + m) else 0.0 for p, m in zip(dip, dim, strict=True)]
    if len(dx) < period:
        return 0.0
    adx = _plain_sum(dx[:period]) / period
    for v in dx[period:]:
        adx = adx - adx / period + v / period
    return js_round2(adx)


def dyn_rr(adx: float, gap_pct: float = 0.0, rr_min: float = cfg.RR_MIN,
           rr_max: float = cfg.RR_MAX, fallback: float = cfg.RR_FALLBACK) -> float:
    """추세 강도(ADX)에 비례하는 손익비. 지표 결측이면 정적 기본값으로 물러선다."""
    if not _finite(adx, gap_pct):
        return js_round2(max(rr_min, min(rr_max, fallback)))
    an = max(0.0, min(1.0, (adx - 15) / 30))
    gn = max(0.0, min(1.0, (gap_pct - 0.5) / 2.5))
    return js_round2(max(rr_min, min(rr_max, rr_min + (rr_max - rr_min) * (an * 2 / 3 + gn * 1 / 3))))


# ══════════════════════════════════════════════════════════════
#  봉 변환
# ══════════════════════════════════════════════════════════════

def aggregate_daily(h4: Sequence[Bar], now_ms: int) -> list[Bar]:
    """4시간봉 → UTC 일봉. now_ms 기준으로 **끝난 날**만 돌려준다."""
    out: list[Bar] = []
    cur: Optional[Bar] = None
    for b in h4:
        day = b.ts // DAY_MS * DAY_MS
        if cur is None or cur.ts != day:
            if cur is not None:
                out.append(cur)
            cur = Bar(day, b.o, b.h, b.l, b.c, b.v)
        else:
            cur.h = max(cur.h, b.h)
            cur.l = min(cur.l, b.l)
            cur.c = b.c
            cur.v += b.v
    if cur is not None and cur.ts + DAY_MS <= now_ms:
        out.append(cur)
    return out


def drop_forming(bars: Sequence[Bar], interval_ms: int, now_ms: int) -> list[Bar]:
    """아직 끝나지 않은 봉을 버린다 (거래소는 형성 중인 봉을 마지막에 붙여 준다)."""
    return [b for b in bars if b.ts + interval_ms <= now_ms]


# ══════════════════════════════════════════════════════════════
#  일봉 지표 묶음 + 신호
# ══════════════════════════════════════════════════════════════

@dataclass
class DailyPack:
    bars: list[Bar]
    c: list[float]
    v: list[float]
    atr: list[float]
    don_h_entry: list[float]    # 20일 최고가 (S6 진입)
    don_l_exit: list[float]     # 10일 최저가 (S6 청산·손절)
    don_l_entry: list[float]    # 20일 최저가 (SX 진입)
    don_h_exit: list[float]     # 10일 최고가 (SX 청산·손절)
    vol_ma: list[float]
    vwap: list[float]
    adx: list[float]


def pack_daily(d1: Sequence[Bar]) -> DailyPack:
    bars = list(d1)
    c = [b.c for b in bars]
    v = [b.v for b in bars]
    hi = [b.h for b in bars]
    lo = [b.l for b in bars]
    spv = sma([x * y for x, y in zip(c, v, strict=True)], cfg.VOL_MA)
    sv = sma(v, cfg.VOL_MA)
    vwap = [a / b if (b and not math.isnan(b)) else NAN for a, b in zip(spv, sv, strict=True)]
    return DailyPack(
        bars=bars, c=c, v=v, atr=atr(bars, cfg.ATR_PERIOD),
        don_h_entry=rolling_max(hi, cfg.DON_ENTRY), don_l_exit=rolling_min(lo, cfg.DON_EXIT),
        don_l_entry=rolling_min(lo, cfg.DON_ENTRY), don_h_exit=rolling_max(hi, cfg.DON_EXIT),
        vol_ma=sv, vwap=vwap, adx=adx_series(bars, cfg.ATR_PERIOD),
    )


def signal_s6(p: DailyPack, j: int) -> Optional[Signal]:
    """S6 롱: 종가가 직전 20일 최고가 돌파 + 거래량 2배 + 종가 > 20일 VWAP."""
    if j < cfg.SIGNAL_WARMUP:
        return None
    dh, dl, vm = p.don_h_entry[j - 1], p.don_l_exit[j - 1], p.vol_ma[j - 1]
    if not _finite(dh, dl, vm, p.atr[j], p.vwap[j]):
        return None
    if not (p.c[j] > dh and p.v[j] > vm * cfg.S6_VOL_MULT and p.c[j] > p.vwap[j]):
        return None
    e = p.c[j]
    sl = max(dl, e - p.atr[j] * cfg.ATR_SL_MULT)
    if sl >= e or sl <= 0:
        return None
    return Signal('S6', 1, sl, dyn_rr(p.adx[j]), e, p.atr[j])


def signal_sx(p: DailyPack, j: int) -> Optional[Signal]:
    """SX 숏: 종가가 직전 20일 최저가 이탈 + 종가 < 20일 VWAP (거래량 조건 없음)."""
    if j < cfg.SIGNAL_WARMUP:
        return None
    dl, dh = p.don_l_entry[j - 1], p.don_h_exit[j - 1]
    if not _finite(dl, dh, p.atr[j], p.vwap[j]):
        return None
    if not (p.c[j] < dl and p.c[j] < p.vwap[j]):
        return None
    e = p.c[j]
    sl = min(dh, e + cfg.ATR_SL_MULT * p.atr[j])
    if sl <= e:
        return None
    return Signal('SX', -1, sl, dyn_rr(p.adx[j]), e, p.atr[j])


SIGNALS = {'S6': signal_s6, 'SX': signal_sx}


def exit_cross(strategy: str, p: DailyPack, j: int) -> bool:
    """추세 이탈 청산: S6는 종가 < 직전 10일 최저가, SX는 종가 > 직전 10일 최고가."""
    if j < 1:
        return False
    if strategy == 'S6':
        ref = p.don_l_exit[j - 1]
        return _finite(ref) and p.c[j] < ref
    if strategy == 'SX':
        ref = p.don_h_exit[j - 1]
        return _finite(ref) and p.c[j] > ref
    return False


# ══════════════════════════════════════════════════════════════
#  장세 판정 (BTC)
# ══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class RegimeInputs:
    px: float
    ema: float
    adx: float
    slope: float
    atr_pct: float


def classify(adx: float, px: float, ema200: float, atr_pct: float,
             adx_4h: float = 0.0, slope: float = 0.0) -> str:
    if atr_pct >= cfg.REGIME_CRISIS_ATR:
        return CRISIS
    if adx >= cfg.REGIME_ADX_TREND:
        base = TRENDING_UP if px >= ema200 else TRENDING_DOWN
        if slope < 0 and adx < 30:
            return WEAK_TREND
        if base == TRENDING_UP and 0 < adx_4h < cfg.REGIME_MICRO_ADX4H:
            return MICRO_RANGING
        return base
    if adx < cfg.REGIME_ADX_WEAK:
        return RANGING
    return WEAK_TREND


def regime_table(btc_d1: Sequence[Bar]) -> list[RegimeInputs]:
    """BTC 일봉(완성봉, **상장일부터 전부**) → 날짜별 장세 입력값.

    EMA200은 전체 기록으로 계산한다. 현물 봇은 일봉 70개만 받아 EMA(200)을
    돌려서 '69일 전 가격이 절반 섞인 값'이 됐고, 백테스트와 레짐이 5% 날짜에서
    달랐다 — 그 버그를 여기서 고친다.
    """
    closes = [b.c for b in btc_d1]
    e = ema(closes, cfg.REGIME_EMA_SPAN)
    atr14 = sma(true_range(btc_d1), cfg.REGIME_ATR_PERIOD)
    lb = cfg.REGIME_ADX_LOOKBACK
    adx_d = [calc_adx(btc_d1[i - lb + 1:i + 1], 14) if i >= lb - 1 else 0.0 for i in range(len(btc_d1))]
    out: list[RegimeInputs] = []
    for i, b in enumerate(btc_d1):
        a = atr14[i] if not math.isnan(atr14[i]) else 0.0
        out.append(RegimeInputs(px=b.c, ema=e[i], adx=adx_d[i],
                                slope=(adx_d[i] - adx_d[i - 2]) if i >= 2 else 0.0,
                                atr_pct=a / b.c))
    return out


def adx4h_table(btc_h4: Sequence[Bar]) -> list[float]:
    lb = cfg.REGIME_ADX_LOOKBACK
    return [calc_adx(btc_h4[i - lb + 1:i + 1], 14) if i >= lb - 1 else 0.0 for i in range(len(btc_h4))]


def regime_from_bars(btc_d1_closed: Sequence[Bar], btc_h4_closed: Sequence[Bar]) -> tuple[str, RegimeInputs, float]:
    """라이브용: 마지막 완성 일봉·4시간봉 기준 장세. 백테스트 regime_table과 같은 값을 낸다."""
    if len(btc_d1_closed) < 61:
        return UNKNOWN, RegimeInputs(0.0, 0.0, 0.0, 0.0, 0.0), 0.0
    r = regime_table(btc_d1_closed)[-1]
    lb = cfg.REGIME_ADX_LOOKBACK
    a4 = calc_adx(btc_h4_closed[-lb:], 14) if len(btc_h4_closed) >= lb else 0.0
    return classify(r.adx, r.px, r.ema, r.atr_pct, a4, r.slope), r, a4


def strategies_for(regime: str) -> tuple[str, ...]:
    return cfg.REGIME_ROUTE.get(regime, ())


# ══════════════════════════════════════════════════════════════
#  진입 판정·사이징
# ══════════════════════════════════════════════════════════════

def funding_scale(strategy: str, funding_rate: float) -> Optional[float]:
    """펀딩비 필터. 차단이면 None, 아니면 위험 배수."""
    if strategy == 'S6':
        if funding_rate >= cfg.FUND_LONG_BLOCK:
            return None
        if funding_rate <= cfg.FUND_LONG_BOOST:
            return cfg.FUND_LONG_BOOST_MULT
        return 1.0
    if strategy == 'SX' and funding_rate <= cfg.FUND_SHORT_BLOCK:
        return None
    return 1.0


def sl_distance(sig: Signal, ref_px: float) -> float:
    """참조가(진입 예정가) 기준 손절 거리. 손절이 이미 지나간 가격이면 0 이하."""
    return ref_px - sig.sl if sig.side == 1 else sig.sl - ref_px


def entry_blocked(sl_dist: float, ref_px: float, symbol: str, taker: float = cfg.FUT_TAKER_FEE,
                  slip: Optional[float] = None) -> Optional[str]:
    """손절폭·비용 게이트. 막히면 사유 문자열."""
    if not (sl_dist > 0) or not (ref_px > 0):
        return 'invalid_sl'
    sl_pct = sl_dist / ref_px
    if sl_pct > cfg.FUT_MAX_SL_PCT:
        return 'sl_too_wide'
    s = cfg.slip_for(symbol) if slip is None else slip
    if (2 * taker + 2 * s) / sl_pct > cfg.FUT_MAX_COST_PER_R:
        return 'cost_exceeds_edge'
    return None


def position_qty(equity: float, sl_dist: float, ref_px: float, risk: float,
                 dd_scale: float = 1.0, f_scale: float = 1.0,
                 cap_frac: Optional[float] = None) -> float:
    """손절 시 잃는 금액 = 자산 × 위험률이 되도록 수량을 정하고, 명목가 상한으로 자른 뒤 낙폭 배수를 곱한다.

    낙폭 배수를 상한보다 먼저 곱하면, 상한이 걸리는 거래(손절폭 4% 미만)에서 '위험 절반'이
    사라진다 — 줄이라는 규칙을 나중에 적용되는 상한이 먹어 버린다. 펀딩 배수(×1.2)는 반대로
    상한 안에서만 늘어야 하므로 상한 전에 곱한다.
    """
    if equity <= 0 or sl_dist <= 0 or ref_px <= 0:
        return 0.0
    qty = equity * risk * f_scale / sl_dist
    cap = cfg.notional_cap_frac(risk) if cap_frac is None else cap_frac
    if qty * ref_px > equity * cap:
        qty = equity * cap / ref_px
    return qty * dd_scale


def trailing_day_loss(exits: Iterable[tuple[int, float]], now_ms: int, equity: float) -> float:
    """직전 24시간에 청산된 거래의 순손익 / 그 손익 전 자산 (일일 손실 한도 판정용, 손실이면 음수).

    '오늘 0시 이후'로 세면 판정 시각(00:01 UTC)에는 1분치만 보여 한도가 영영 걸리지 않는다.
    """
    pnl = sum(net for ts, net in exits if now_ms - DAY_MS < ts <= now_ms)
    base = equity - pnl
    return pnl / base if base > 0 else 0.0


def dd_scale(drawdown: float) -> float:
    """낙폭 규칙: 15% → 위험 절반, 20% → 신규 진입 중단(0)."""
    if drawdown >= cfg.FUT_DD_STOP:
        return 0.0
    if drawdown >= cfg.FUT_DD_HALF:
        return 0.5
    return 1.0


def entry_limit_price(side: int, ref: float, market_px: float, offset: float = cfg.ENTRY_POST_OFFSET) -> float:
    """메이커 지정가: 롱은 (신호 종가·현재가 중 낮은 값)보다 조금 아래, 숏은 반대."""
    if side == 1:
        return min(ref, market_px) * (1 - offset)
    return max(ref, market_px) * (1 + offset)


def take_profit_price(side: int, fill: float, sl_dist: float, rr: float) -> float:
    return fill + side * sl_dist * rr


def stop_price(side: int, fill: float, sl_dist: float) -> float:
    return fill - side * sl_dist


def funding_asof(times: Sequence[int], rates: Sequence[float], ts: int) -> float:
    """ts 시점까지 확정된 마지막 펀딩비 (없으면 0)."""
    i = bisect_right(times, ts)
    return rates[i - 1] if i > 0 else 0.0

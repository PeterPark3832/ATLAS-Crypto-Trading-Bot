"""atlas_fut_core — 지표·장세·신호·사이징 규칙 (표준 라이브러리만으로 돈다)."""

import math

import atlas_fut_config as cfg
import atlas_fut_core as core
from atlas_fut_core import DAY_MS, H4_MS, Bar


def _flat_then(last_close: float, last_vol: float, n: int = 70, base: float = 100.0) -> list[Bar]:
    """n-1일 횡보(±1) 뒤 마지막 날 종가·거래량을 지정한 일봉."""
    bars = []
    for i in range(n - 1):
        c = base + (1 if i % 2 else -1) * 0.5
        bars.append(Bar(i * DAY_MS, c, c + 1, c - 1, c, 1000.0))
    o = bars[-1].c
    bars.append(Bar((n - 1) * DAY_MS, o, max(o, last_close) + 0.5, min(o, last_close) - 0.5, last_close, last_vol))
    return bars


def test_js_round2_rounds_half_up_unlike_python_round():
    assert core.js_round2(25.125) == 25.13
    assert round(25.125, 2) == 25.12          # 파이썬은 은행가 반올림 — 그래서 쓰지 않는다
    assert core.js_round2(-2.5 / 100) == -0.02


def test_sma_resets_on_nan_and_matches_mean():
    x = [1.0, 2.0, 3.0, float('nan'), 4.0, 5.0, 6.0]
    out = core.sma(x, 3)
    assert math.isnan(out[1]) and out[2] == 2.0
    assert math.isnan(out[4]) and math.isnan(out[5]) and out[6] == 5.0


def test_ema_starts_at_first_value():
    out = core.ema([10.0, 20.0, 20.0], 3)
    assert out[0] == 10.0
    assert out[1] == 15.0 and out[2] == 17.5


def test_rolling_extremes_and_atr():
    bars = [Bar(i, 10, 10 + i, 10 - i, 10, 1) for i in range(5)]
    assert core.rolling_max([b.h for b in bars], 3)[4] == 14
    assert core.rolling_min([b.l for b in bars], 3)[4] == 6
    tr = core.true_range(bars)
    assert tr[0] == 0 and tr[1] == 2
    assert core.atr(bars, 2)[1] == 1.0


def test_calc_adx_needs_enough_bars_and_detects_trend():
    assert core.calc_adx([Bar(i, 1, 1, 1, 1, 1) for i in range(10)]) == 0.0
    up = [Bar(i, 100 + i, 101 + i, 99 + i, 100.5 + i, 1) for i in range(60)]
    chop = [Bar(i, 100, 101 + (i % 2), 99 - (i % 2), 100, 1) for i in range(60)]
    assert core.calc_adx(up) > 50
    assert core.calc_adx(chop) < 20


def test_classify_priority_and_downgrades():
    assert core.classify(40, 100, 90, 0.08) == core.CRISIS
    assert core.classify(35, 100, 90, 0.03) == core.TRENDING_UP
    assert core.classify(35, 80, 90, 0.03) == core.TRENDING_DOWN
    assert core.classify(27, 100, 90, 0.03, slope=-1) == core.WEAK_TREND      # 꺾이는 추세
    assert core.classify(35, 100, 90, 0.03, adx_4h=15) == core.MICRO_RANGING
    assert core.classify(35, 80, 90, 0.03, adx_4h=15) == core.TRENDING_DOWN    # 하락에는 MICRO 없음
    assert core.classify(15, 100, 90, 0.03) == core.RANGING
    assert core.classify(22, 100, 90, 0.03) == core.WEAK_TREND


def test_regime_route_matches_validated_table():
    assert core.strategies_for(core.TRENDING_UP) == ('S6',)
    assert core.strategies_for(core.RANGING) == ('S6',)
    assert core.strategies_for(core.WEAK_TREND) == ('SX',)
    assert core.strategies_for(core.TRENDING_DOWN) == ('SX',)
    for r in (core.MICRO_RANGING, core.CRISIS, core.UNKNOWN):
        assert core.strategies_for(r) == ()


def test_regime_table_uses_full_history_ema():
    bars = [Bar(i * DAY_MS, 100 + i, 101 + i, 99 + i, 100 + i, 1) for i in range(300)]
    full = core.regime_table(bars)[-1].ema
    assert full == core.ema([b.c for b in bars], 200)[-1]
    short = core.regime_table(bars[-70:])[-1].ema      # 현물 봇 버그: 70개만으로 EMA200
    assert abs(full - short) > 1


def test_regime_from_bars_matches_table_and_needs_history():
    bars = [Bar(i * DAY_MS, 100 + i, 102 + i, 99 + i, 101 + i, 1) for i in range(120)]
    h4 = [Bar(i * H4_MS, 100, 101, 99, 100, 1) for i in range(60)]
    regime, r, a4 = core.regime_from_bars(bars, h4)
    t = core.regime_table(bars)[-1]
    assert (r.adx, r.ema, r.slope) == (t.adx, t.ema, t.slope)
    assert regime == core.classify(t.adx, t.px, t.ema, t.atr_pct, a4, t.slope)
    assert core.regime_from_bars(bars[:30], h4)[0] == core.UNKNOWN


def test_signal_s6_requires_breakout_volume_and_sets_stop():
    bars = _flat_then(110.0, 5000.0)
    p = core.pack_daily(bars)
    sig = core.signal_s6(p, len(bars) - 1)
    assert sig is not None and sig.side == 1 and sig.strategy == 'S6'
    j = len(bars) - 1
    assert sig.sl == max(p.don_l_exit[j - 1], 110.0 - 2 * p.atr[j])
    assert cfg.RR_MIN <= sig.rr <= cfg.RR_MAX
    quiet = core.pack_daily(_flat_then(110.0, 1000.0))       # 거래량 스파이크 없음
    assert core.signal_s6(quiet, len(bars) - 1) is None
    assert core.signal_s6(p, 10) is None                     # 워밍업


def test_signal_sx_needs_no_volume_and_stop_above():
    bars = _flat_then(90.0, 1000.0)
    p = core.pack_daily(bars)
    j = len(bars) - 1
    sig = core.signal_sx(p, j)
    assert sig is not None and sig.side == -1
    assert sig.sl == min(p.don_h_exit[j - 1], 90.0 + 2 * p.atr[j])
    assert core.signal_s6(p, j) is None


def test_exit_cross_both_sides():
    p = core.pack_daily(_flat_then(90.0, 1000.0))
    j = len(p.c) - 1
    assert core.exit_cross('S6', p, j) is True
    assert core.exit_cross('SX', p, j) is False
    p2 = core.pack_daily(_flat_then(110.0, 1000.0))
    assert core.exit_cross('SX', p2, len(p2.c) - 1) is True
    assert core.exit_cross('XX', p2, 5) is False


def test_dyn_rr_fallback_and_range():
    assert core.dyn_rr(float('nan')) == cfg.RR_FALLBACK
    assert core.dyn_rr(10) == cfg.RR_MIN
    assert core.dyn_rr(45) == 2.5            # 갭 0 → ADX 몫(2/3)까지만
    assert core.dyn_rr(45, 3.0) == cfg.RR_MAX


def test_funding_filter():
    assert core.funding_scale('S6', 0.0006) is None
    assert core.funding_scale('S6', -0.0002) == cfg.FUND_LONG_BOOST_MULT
    assert core.funding_scale('S6', 0.0001) == 1.0
    assert core.funding_scale('SX', -0.0006) is None
    assert core.funding_scale('SX', 0.001) == 1.0


def test_entry_blocked_gates():
    assert core.entry_blocked(0, 100, 'BTCUSDT') == 'invalid_sl'
    assert core.entry_blocked(25, 100, 'BTCUSDT') == 'sl_too_wide'
    assert core.entry_blocked(0.5, 100, 'AAVEUSDT') == 'cost_exceeds_edge'   # 손절 0.5% → 비용이 R 잠식
    assert core.entry_blocked(5, 100, 'BTCUSDT') is None


def test_position_qty_risk_and_cap():
    q = core.position_qty(10_000, 10, 100, 0.005)          # 손절 10% → 위험 $50 → 5개
    assert abs(q - 5.0) < 1e-12
    capped = core.position_qty(10_000, 1, 100, 0.005)      # 손절 1% → 명목가 상한 12.5%
    assert abs(capped * 100 - 1250) < 1e-9
    assert core.position_qty(0, 1, 1, 0.005) == 0.0
    assert core.position_qty(10_000, 10, 100, 0.005, dd_scale=0.5) == q / 2


def test_dd_scale_thresholds():
    assert core.dd_scale(0.10) == 1.0
    assert core.dd_scale(0.15) == 0.5
    assert core.dd_scale(0.2) == 0.0


def test_entry_prices_and_brackets():
    assert core.entry_limit_price(1, 100, 101) == 100 * (1 - cfg.ENTRY_POST_OFFSET)
    assert core.entry_limit_price(-1, 100, 99) == 100 * (1 + cfg.ENTRY_POST_OFFSET)
    assert core.take_profit_price(-1, 100, 5, 2) == 90
    assert core.stop_price(-1, 100, 5) == 105
    sig = core.Signal('SX', -1, 105.0, 2.0, 100.0, 1.0)
    assert core.sl_distance(sig, 100) == 5


def test_aggregate_daily_drops_unfinished_day():
    h4 = [Bar(i * H4_MS, 10 + i, 11 + i, 9 + i, 10.5 + i, 1) for i in range(9)]   # 1일 + 3봉
    d = core.aggregate_daily(h4, now_ms=9 * H4_MS)
    assert len(d) == 1
    assert (d[0].o, d[0].h, d[0].l, d[0].c, d[0].v) == (10, 16, 9, 15.5, 6)
    assert len(core.aggregate_daily(h4, now_ms=2 * DAY_MS)) == 2
    assert core.drop_forming(h4, H4_MS, 3 * H4_MS)[-1].ts == 2 * H4_MS


def test_funding_asof():
    assert core.funding_asof([10, 20, 30], [0.1, 0.2, 0.3], 25) == 0.2
    assert core.funding_asof([10], [0.1], 5) == 0.0


def test_config_invariants():
    # 손절폭 상한 < 격리 레버리지 청산 거리 — 청산이 손절보다 먼저 오면 안 된다
    assert cfg.FUT_MAX_SL_PCT < 1 / cfg.FUT_LEVERAGE - cfg.FUT_MMR
    assert set(cfg.FUT_STRATEGIES) == {s for v in cfg.REGIME_ROUTE.values() for s in v}
    assert cfg.slip_for('BTCUSDT') < cfg.slip_for('SOLUSDT') < cfg.slip_for('AAVEUSDT')
    assert cfg.notional_cap_frac(0.005) == 0.125


def test_r4_drawdown_half_survives_notional_cap():
    """#4 낙폭 15% '위험 절반'이 명목가 상한에 먹히지 않는다.
    예전: 상한을 나중에 적용해 손절폭 1%·2%에서 수량이 ×1.00 그대로였다."""
    for sl_pct in (0.01, 0.02, 0.04, 0.08):
        full = core.position_qty(10_000, 100 * sl_pct, 100, 0.005)
        half = core.position_qty(10_000, 100 * sl_pct, 100, 0.005, dd_scale=0.5)
        assert abs(half - full / 2) < 1e-12, sl_pct
    # 펀딩 가산(×1.2)은 반대로 상한 안에서만 늘어난다
    assert core.position_qty(10_000, 1, 100, 0.005, f_scale=1.2) == core.position_qty(10_000, 1, 100, 0.005)


def test_r3_trailing_day_loss_window():
    """#3 일일 손실은 '오늘 0시 이후'가 아니라 직전 24시간으로 센다."""
    now = 10 * DAY_MS
    exits = [(now - DAY_MS, -100.0), (now - DAY_MS + 1, -200.0), (now, -100.0), (now + 1, -999.0)]
    # 정확히 24시간 전은 빼고, 지금 시각은 넣고, 미래는 뺀다. 기준 = 그 손실 전 자산
    assert core.trailing_day_loss(exits, now, 9_700) == -300.0 / 10_000
    assert core.trailing_day_loss([], now, 10_000) == 0.0
    assert core.trailing_day_loss([(now, 5.0)], now, 5.0) == 0.0          # 기준 자산 0 이하 → 판정 안 함


def test_s6_volume_rule_is_strictly_more_than_twice_the_average():
    """S6 거래량 조건 = 직전 20일 평균의 **2배 초과**. 경계 바로 양쪽을 고정한다.
    다른 테스트는 거래량을 평균의 6배로 넣어서 2.0 → 1.95 같은 규칙 변경을 잡지 못했다(2026-11-17 재검증)."""
    for mult, want in ((1.99, False), (2.0, False), (2.01, True)):
        p = core.pack_daily(_flat_then(112.0, 1000.0 * mult))
        assert p.vol_ma[-2] == 1000.0
        assert (core.signal_s6(p, len(p.c) - 1) is not None) is want, mult


# ── 규칙 경계 (2026-11-17 변이 시험에서 아무 테스트도 잡지 못한 것들) ─────────────
# 값은 일부러 숫자로 박는다. cfg 값을 읽어 기대값을 만들면 규칙이 바뀌어도 테스트가 같이 따라가 버린다.

def test_classify_boundaries():
    """장세 임계값은 '이상'(>=)과 '미만'(<)이 섞여 있다 — 경계 바로 위·아래를 고정한다."""
    C = core.classify
    up, down = (110.0, 100.0), (90.0, 100.0)                       # (BTC 가격, EMA200)
    assert C(30.0, *up, 0.07) == core.CRISIS and C(30.0, *up, 0.0699) == core.TRENDING_UP
    assert C(25.0, *up, 0.02) == core.TRENDING_UP and C(24.99, *up, 0.02) == core.WEAK_TREND
    assert C(25.0, *down, 0.02) == core.TRENDING_DOWN
    assert C(25.0, 100.0, 100.0, 0.02) == core.TRENDING_UP          # EMA200과 같으면 위로 본다
    assert C(20.0, *up, 0.02) == core.WEAK_TREND and C(19.99, *up, 0.02) == core.RANGING
    # 꺾이는 추세: ADX 30 미만에서 ADX 기울기가 음수면 약한 추세
    assert C(29.99, *up, 0.02, slope=-0.1) == core.WEAK_TREND
    assert C(30.0, *up, 0.02, slope=-0.1) == core.TRENDING_UP and C(25.0, *up, 0.02, slope=0.0) == core.TRENDING_UP
    # 단기 횡보: 일봉 상승 추세인데 4시간봉 ADX가 20 미만 (0은 '값 없음'이라 판정하지 않음, 하락 추세엔 적용 안 함)
    assert C(30.0, *up, 0.02, adx_4h=19.99) == core.MICRO_RANGING
    assert C(30.0, *up, 0.02, adx_4h=20.0) == core.TRENDING_UP and C(30.0, *up, 0.02, adx_4h=0.0) == core.TRENDING_UP
    assert C(30.0, *down, 0.02, adx_4h=10.0) == core.TRENDING_DOWN


def test_funding_filter_boundaries():
    """펀딩비는 거래소가 끊어 주는 값이라 기준값과 **정확히 같은** 날이 실제로 나온다."""
    fs = core.funding_scale
    assert fs('S6', 0.0005) is None and fs('S6', 0.00049) == 1.0          # 롱 과밀: +0.05% 이상 차단
    assert fs('S6', -0.0001) == 1.2 and fs('S6', -0.00009) == 1.0         # 롱 가산: -0.01% 이하 위험 ×1.2
    assert fs('SX', -0.0005) is None and fs('SX', -0.00049) == 1.0        # 숏 과밀: -0.05% 이하 차단
    assert fs('SX', 0.003) == 1.0                                         # 숏은 플러스 펀딩에 차단·가산 없음


def test_entry_gate_boundaries_and_zero_distance():
    eb = core.entry_blocked
    assert eb(0.0, 100.0, 'BTCUSDT') == 'invalid_sl'                      # 0으로 나누기 전에 막는다
    assert eb(-1.0, 100.0, 'BTCUSDT') == 'invalid_sl' and eb(1.0, 0.0, 'BTCUSDT') == 'invalid_sl'
    assert eb(20.0, 100.0, 'BTCUSDT') is None                             # 손절폭 정확히 20%는 허용
    assert eb(20.01, 100.0, 'BTCUSDT') == 'sl_too_wide'
    # 비용 게이트: 왕복 비용(테이커 0.05%×2 + BTC 슬리피지 0.03%×2 = 0.16%)이 1R의 20%를 넘으면 차단
    # → BTC는 손절폭 0.8%가 경계 (정확히 0.8%는 부동소수 오차로 0.1999…라 양옆으로 고정)
    assert eb(0.81, 100.0, 'BTCUSDT') is None and eb(0.79, 100.0, 'BTCUSDT') == 'cost_exceeds_edge'


def test_position_qty_never_divides_by_zero():
    assert core.position_qty(10_000, 0.0, 100, 0.005) == 0.0
    assert core.position_qty(10_000, 1.0, 0.0, 0.005) == 0.0
    assert core.position_qty(-5.0, 1.0, 100, 0.005) == 0.0


def test_signal_warmup_starts_at_bar_45():
    """신호는 기록의 45번째 봉(j=45)부터 낸다 — 그 앞은 지표가 덜 쌓였다."""
    for n, want in ((46, True), (45, False)):
        j = n - 1
        assert (core.signal_s6(core.pack_daily(_flat_then(112.0, 6000.0, n=n)), j) is not None) is want, n
        assert (core.signal_sx(core.pack_daily(_flat_then(88.0, 1000.0, n=n)), j) is not None) is want, n


def test_breakout_needs_close_strictly_beyond_20_day_channel():
    """돌파·이탈은 '넘어야' 한다. 틱 단위 가격이라 종가가 직전 고가·저가와 정확히 같은 날이 실제로 있다.
    (횡보 구간의 직전 20일 최고가 101.5, 최저가 98.5)"""
    s6 = core.signal_s6
    sx = core.signal_sx
    assert s6(core.pack_daily(_flat_then(101.5, 6000.0)), 69) is None
    assert s6(core.pack_daily(_flat_then(101.51, 6000.0)), 69) is not None
    assert sx(core.pack_daily(_flat_then(98.5, 1000.0)), 69) is None
    assert sx(core.pack_daily(_flat_then(98.49, 1000.0)), 69) is not None


def test_trend_exit_needs_close_strictly_beyond_10_day_extreme():
    ec = core.exit_cross
    assert ec('S6', core.pack_daily(_flat_then(98.5, 1000.0)), 69) is False      # 직전 10일 최저가와 같음
    assert ec('S6', core.pack_daily(_flat_then(98.49, 1000.0)), 69) is True
    assert ec('SX', core.pack_daily(_flat_then(101.5, 1000.0)), 69) is False     # 직전 10일 최고가와 같음
    assert ec('SX', core.pack_daily(_flat_then(101.51, 1000.0)), 69) is True

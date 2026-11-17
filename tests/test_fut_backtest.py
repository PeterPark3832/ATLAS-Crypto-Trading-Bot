"""atlas_fut_backtest — 합성 데이터로 체결·청산·비용 규칙을 고정한다."""

import shutil
import tempfile
from pathlib import Path

import atlas_fut_backtest as bt
import atlas_fut_config as cfg
import atlas_fut_core as core
from atlas_fut_core import DAY_MS, H4_MS, Bar

T0 = 1609459200000 - 70 * DAY_MS       # 2021-01-01 70일 전부터


class _FixedRegime:
    regime = 'TRENDING_UP'

    def __init__(self, btc):
        self.btc = btc

    def at(self, t):
        return _FixedRegime.regime


def _h4_series(day_closes, vols=None, wick=0.5):
    """일별 종가 목록 → 하루 6개 4시간봉 (봉 안에서 종가로 선형 이동)."""
    bars, prev = [], day_closes[0]
    for d, c in enumerate(day_closes):
        v = (vols[d] if vols else 1000.0) / 6
        for k in range(6):
            o = prev + (c - prev) * k / 6
            cl = prev + (c - prev) * (k + 1) / 6
            bars.append(Bar(T0 + d * DAY_MS + k * H4_MS, o, max(o, cl) + wick, min(o, cl) - wick, cl, v))
        prev = c
    return bars


def _universe(alt_closes, alt_vols, funding=None):
    n = len(alt_closes)
    now = T0 + n * DAY_MS
    btc = bt.build_symbol(_h4_series([100.0] * n), [], [], now)
    ft = [T0 + i * 8 * 3600_000 for i in range(n * 3)] if funding is not None else []
    fr = [funding] * len(ft) if funding is not None else []
    alt = bt.build_symbol(_h4_series(alt_closes, alt_vols), ft, fr, now)
    return {'BTCUSDT': btc, 'ETHUSDT': alt}


def _run(data, regime='TRENDING_UP', **kw):
    orig = bt.RegimeClock
    bt.RegimeClock = _FixedRegime
    _FixedRegime.regime = regime
    try:
        return bt.simulate(data, bt.BTConfig(start_ms=T0 + 60 * DAY_MS, **kw))
    finally:
        bt.RegimeClock = orig


def _breakout_path(after):
    closes = [100 + (0.5 if i % 2 else -0.5) for i in range(65)] + [112.0] + after
    vols = [1000.0] * 65 + [6000.0] + [1000.0] * len(after)
    return closes, vols


def test_long_breakout_enters_next_day_and_takes_profit():
    closes, vols = _breakout_path([111.0, 125.0, 140.0, 150.0])
    r = _run(_universe(closes, vols))
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t.st, t.side, t.reason) == ('S6', 1, 'TP')
    signal_day = T0 + 65 * DAY_MS
    assert t.entry_ts >= signal_day + DAY_MS           # 신호 다음 날부터 (선행 편향 없음)
    assert t.net > 0 and r.orders['created'] == 1


def test_regime_gate_blocks_entries():
    closes, vols = _breakout_path([111.0, 125.0, 140.0])
    for regime in ('MICRO_RANGING', 'CRISIS', 'TRENDING_DOWN'):
        assert _run(_universe(closes, vols), regime=regime).trades == []


def test_short_breakdown_stops_out_on_squeeze():
    closes = [100 + (0.5 if i % 2 else -0.5) for i in range(65)] + [88.0, 88.5, 120.0, 121.0]
    r = _run(_universe(closes, None), regime='TRENDING_DOWN')
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t.st, t.side, t.reason) == ('SX', -1, 'SL')
    assert t.net < 0 and -1.6 < t.net_r < -0.9          # 1R 안팎 손실(수수료·슬리피지 포함)


def test_funding_is_charged_to_longs():
    closes, vols = _breakout_path([111.0, 111.5, 112.0, 112.5, 113.0, 112.8, 113.2, 113.0])
    paid = _run(_universe(closes, vols, funding=0.0004))
    free = _run(_universe(closes, vols, funding=0.0))
    assert paid.costs['fund'] > 0 and free.costs['fund'] == 0
    assert paid.final < free.final


def test_market_entry_pays_taker_and_slippage():
    closes, vols = _breakout_path([111.0, 125.0, 140.0])
    limit = _run(_universe(closes, vols))
    market = _run(_universe(closes, vols), entry='market')
    assert market.costs['slip'] > limit.costs['slip']
    assert market.orders['created'] == 0


def test_summarize_and_csv_roundtrip():
    closes, vols = _breakout_path([111.0, 125.0, 140.0])
    data = _universe(closes, vols)
    p = Path(tempfile.mkdtemp())
    try:
        for s, sd in data.items():
            (p / f'{s}_4h.csv').write_text('ts,o,h,l,c,v\n' + '\n'.join(
                f'{b.ts},{b.o},{b.h},{b.l},{b.c},{b.v}' for b in sd.h4) + '\n')
        (p / 'ETHUSDT_funding.csv').write_text('t,r\n1,0.0001\n')
        loaded = bt.load_universe(p, cfg.FUT_UNIVERSE, T0 + len(closes) * DAY_MS)
    finally:
        shutil.rmtree(p, ignore_errors=True)
    assert set(loaded) == {'BTCUSDT', 'ETHUSDT'}
    assert loaded['ETHUSDT'].fund_r == [0.0001]
    s = bt.summarize(_run(data))
    assert s['trades'] == 1 and 'years' in s and 'by' in s


def test_r3_backtest_daily_loss_limit_blocks_next_entries():
    """#3 직전 24시간 손실 한도가 백테스트에서도 실제로 작동한다 (라이브와 같은 함수).
    예전: 날짜가 바뀌는 봉에서 0으로 리셋한 직후 진입을 판정해 한도가 한 번도 걸리지 않았다.
    ETH가 66일차에 손절 → 같은 날 마감에 SOL 돌파 → 67일차 SOL 진입이 한도에 걸리는가."""
    flat = [100 + (0.5 if i % 2 else -0.5) for i in range(65)]
    eth = flat + [112.0, 100.0, 100.0, 100.0]
    sol = flat + [100.0, 112.0, 112.5, 113.0]
    n = len(eth)
    now = T0 + n * DAY_MS
    data = {'BTCUSDT': bt.build_symbol(_h4_series([100.0] * n), [], [], now),
            'ETHUSDT': bt.build_symbol(_h4_series(eth, [1000.0] * 65 + [6000.0] + [1000.0] * 3), [], [], now),
            'SOLUSDT': bt.build_symbol(_h4_series(sol, [1000.0] * 66 + [6000.0] + [1000.0] * 2), [], [], now)}
    loose = _run(data)                                   # 기본 -4%: 손절 한 번(약 -0.5%)으론 안 걸린다
    tight = _run(data, daily_loss=-0.003)                # -0.3%: 바로 전 손절 때문에 SOL 진입이 막혀야 한다
    assert [t.reason for t in loose.trades if t.symbol == 'ETHUSDT'][:1] == ['SL']
    assert any(t.symbol == 'SOLUSDT' for t in loose.trades)
    assert not any(t.symbol == 'SOLUSDT' for t in tight.trades)


def test_backtest_holds_at_most_six():
    """동시 보유는 6개까지 — 7종목이 같은 날 신호를 내도 6개만 산다 (숫자를 박아 둔다)."""
    closes, vols = _breakout_path([113.0, 114.0, 115.0])
    n = len(closes)
    now = T0 + n * DAY_MS
    data = {'BTCUSDT': bt.build_symbol(_h4_series([100.0] * n), [], [], now)}
    for s in ('ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT'):
        data[s] = bt.build_symbol(_h4_series(closes, vols), [], [], now)
    assert len({t.symbol for t in _run(data).trades}) == 6


def test_bar_touching_limit_or_stop_exactly_counts():
    """4시간봉이 문턱에 **정확히 닿으면** 체결로 본다. 지정가는 0.05% 더 뚫어야 체결(L × 0.9995),
    손절은 닿으면 손절. 손절가는 10일 최저가에서 오는 경우가 많아 같은 가격을 다시 찍는 쌍바닥이 실제로 생긴다 —
    '<'로 바꾸면 백테스트가 낙관적으로 변하는데, 2026-11-17 변이 시험에서 어떤 테스트도 잡지 못했다."""
    closes, vols = _breakout_path([113.0, 113.0, 113.0, 113.0])
    n = len(closes)
    now = T0 + n * DAY_MS
    h4 = _h4_series(closes, vols)
    sd = bt.build_symbol(h4, [], [], now)
    sig = core.signal_s6(sd.pack, sd.idx1[T0 + 65 * DAY_MS])
    first = next(b for b in h4 if b.ts == T0 + 66 * DAY_MS)       # 신호 다음 날 첫 봉 (지정가를 거는 봉)
    lp = core.entry_limit_price(1, sig.ref, first.o)
    first.l = lp * (1 - 0.0005)                                     # 체결 문턱까지만 내려옴
    sl = core.stop_price(1, lp, core.sl_distance(sig, lp))          # 백테스트가 잡는 손절가와 같은 계산
    hit = next(b for b in h4 if b.ts == T0 + 67 * DAY_MS + H4_MS)
    hit.l = sl                                                       # 손절가까지만 내려옴
    data = {'BTCUSDT': bt.build_symbol(_h4_series([100.0] * n), [], [], now),
            'ETHUSDT': bt.build_symbol(h4, [], [], now)}
    t = _run(data).trades[0]
    assert t.entry_ts == first.ts and t.entry == lp
    assert t.reason == 'SL' and t.exit_ts == hit.ts


def test_bar_touching_limit_or_stop_exactly_counts_short_side():
    """위 테스트의 숏 쪽 거울상 — 지정가는 L × 1.0005까지 올라와야 체결, 손절은 고가가 닿으면 손절."""
    flat = [100 + (0.5 if i % 2 else -0.5) for i in range(65)]
    closes = flat + [88.0, 87.0, 87.0, 87.0, 87.0]
    n = len(closes)
    now = T0 + n * DAY_MS
    h4 = _h4_series(closes)
    sd = bt.build_symbol(h4, [], [], now)
    sig = core.signal_sx(sd.pack, sd.idx1[T0 + 65 * DAY_MS])
    first = next(b for b in h4 if b.ts == T0 + 66 * DAY_MS)
    lp = core.entry_limit_price(-1, sig.ref, first.o)
    first.h = lp * (1 + 0.0005)
    sl = core.stop_price(-1, lp, core.sl_distance(sig, lp))
    hit = next(b for b in h4 if b.ts == T0 + 67 * DAY_MS + H4_MS)
    hit.h = sl
    data = {'BTCUSDT': bt.build_symbol(_h4_series([100.0] * n), [], [], now),
            'ETHUSDT': bt.build_symbol(h4, [], [], now)}
    t = _run(data, regime='TRENDING_DOWN').trades[0]
    assert t.st == 'SX' and t.entry_ts == first.ts and t.entry == lp
    assert t.reason == 'SL' and t.exit_ts == hit.ts


def test_drawdown_halt_does_not_resume_when_open_positions_recover():
    """낙폭 20% 정지 뒤 보유 포지션이 회복해 낙폭이 20% 아래로 돌아와도 새로 사지 않는다 — 라이브와 같다
    (라이브는 사람이 점검 후 --resume). 예전 백테스트는 여기서 저절로 다시 들어갔다.
    위험 25%로 일부러: ETH 손절(낙폭 약 24%) → 정지, SOL 익절(낙폭 회복) → XRP 돌파 신호는 무시돼야 한다."""
    flat = [100 + (0.5 if i % 2 else -0.5) for i in range(65)]
    eth = flat + [112.0, 100.0] + [100.0] * 5            # 66일차 손절
    sol = flat + [112.0, 112.5, 130.0] + [130.0] * 4     # 67일차 익절 (손익비 2.5 → 익절가 약 126)
    xrp = flat + [100.0] * 4 + [112.0, 113.0, 114.0]     # 69일차 돌파 → 70일차에 살 수 있는가
    n = len(eth)
    now = T0 + n * DAY_MS

    def vols(k):
        return [1000.0] * k + [6000.0] + [1000.0] * (n - k - 1)

    data = {'BTCUSDT': bt.build_symbol(_h4_series([100.0] * n), [], [], now),
            'ETHUSDT': bt.build_symbol(_h4_series(eth, vols(65)), [], [], now),
            'SOLUSDT': bt.build_symbol(_h4_series(sol, vols(65)), [], [], now),
            'XRPUSDT': bt.build_symbol(_h4_series(xrp, vols(69)), [], [], now)}
    r = _run(data, risk=0.25)
    assert r.halted_at is not None and r.max_dd >= 0.20
    assert [t.reason for t in r.trades if t.symbol == 'ETHUSDT'] == ['SL']
    assert [t.reason for t in r.trades if t.symbol == 'SOLUSDT'] == ['TP']     # 보유 포지션은 정지 뒤에도 정상 청산
    assert not any(t.symbol == 'XRPUSDT' for t in r.trades)

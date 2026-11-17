"""
ATLAS Futures — 포트폴리오 백테스트
===================================
라이브 엔진과 같은 규칙(atlas_fut_core)으로 20종목 포트폴리오를 4시간봉 단위로 재생한다.

반영하는 현실:
  · 신호는 일봉 마감 후 판정, 진입은 다음 봉부터 (선행 편향 없음)
  · 장세는 '진입 시점에 이미 끝난' BTC 일봉·4시간봉으로만 판정
  · 진입: 메이커 지정가(현재가보다 0.02% 유리) 4시간 대기 → 못 사면 시장가
          지정가는 가격이 0.05% 더 뚫고 지나가야 체결로 본다(대기열 순서 보수 가정)
  · 손절: 스탑 시장가 — 티어 슬리피지 × 1.5, 봉이 손절가를 건너뛰면 시가로 체결
  · 익절: 지정가(메이커) — 0.05% 더 뚫어야 체결, 지정가로 진입한 봉에서는 익절 안 함
  · 한 봉에서 손절·익절이 다 닿으면 손절로 본다
  · 실제 펀딩비 기록을 8시간마다 정산
  · 종목당 1포지션(단방향), 동시 6개, 일일 손실 한도, 낙폭 규칙

실행:
  python atlas_fut_backtest.py --data-dir data/futures --fetch     # 데이터 받기 + 실행
  python atlas_fut_backtest.py --data-dir data/futures --risk 0.005
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import atlas_fut_config as cfg
import atlas_fut_core as core
from atlas_fut_core import DAY_MS, H4_MS, Bar


# ══════════════════════════════════════════════════════════════
#  데이터
# ══════════════════════════════════════════════════════════════

@dataclass
class SymbolData:
    h4: list[Bar]
    d1: list[Bar]
    pack: core.DailyPack
    idx4: dict[int, int]
    idx1: dict[int, int]
    fund_t: list[int]
    fund_r: list[float]


def load_csv_bars(path: Path) -> list[Bar]:
    out: list[Bar] = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            out.append(Bar(int(row['ts']), float(row['o']), float(row['h']), float(row['l']),
                           float(row['c']), float(row['v'])))
    out.sort(key=lambda b: b.ts)
    return out


def load_csv_funding(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        return [], []
    rows: list[tuple[int, float]] = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            rows.append((int(row['t']), float(row['r'])))
    rows.sort()
    return [t for t, _ in rows], [r for _, r in rows]


def build_symbol(h4: list[Bar], fund_t: list[int], fund_r: list[float], now_ms: int) -> SymbolData:
    h4 = core.drop_forming(h4, H4_MS, now_ms)
    d1 = core.aggregate_daily(h4, now_ms)
    return SymbolData(h4=h4, d1=d1, pack=core.pack_daily(d1),
                      idx4={b.ts: i for i, b in enumerate(h4)}, idx1={b.ts: i for i, b in enumerate(d1)},
                      fund_t=fund_t, fund_r=fund_r)


def _http_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={'User-Agent': 'atlas-fut-backtest'})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 - 고정된 https 바이낸스 공개 엔드포인트
        return json.loads(r.read().decode())


def fetch_to_csv(data_dir: Path, symbols: list[str], end_ms: Optional[int] = None) -> None:  # pragma: no cover - 네트워크
    """바이낸스 공개 API에서 4시간봉·펀딩비를 받아 CSV로 저장(인증 불필요)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    end_ms = end_ms or int(time.time() * 1000)
    for s in symbols:
        start = cfg.FUT_BTC_HISTORY_START_MS if s == cfg.FUT_REGIME_SYMBOL else 1577836800000  # 2020-01-01
        rows: list[list] = []
        st = start
        while st < end_ms:
            j = _http_json(f'{cfg.FUT_LIVE_BASE_URL}/fapi/v1/klines?symbol={s}&interval=4h&startTime={st}&limit=1500')
            if not j:
                break
            rows.extend(j)
            st = int(j[-1][0]) + 1
            if len(j) < 1500:
                break
            time.sleep(0.1)
        with (data_dir / f'{s}_4h.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ts', 'o', 'h', 'l', 'c', 'v'])
            for k in rows:
                w.writerow([k[0], k[1], k[2], k[3], k[4], k[5]])
        frows: list[tuple[int, str]] = []
        st = start
        while True:
            j = _http_json(f'{cfg.FUT_LIVE_BASE_URL}/fapi/v1/fundingRate?symbol={s}&startTime={st}&limit=1000')
            if not j:
                break
            frows.extend((int(x['fundingTime']), x['fundingRate']) for x in j)
            st = int(j[-1]['fundingTime']) + 1
            if len(j) < 1000:
                break
            time.sleep(0.12)
        with (data_dir / f'{s}_funding.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'r'])
            w.writerows(frows)
        print(f'  {s}: 4h {len(rows)}봉, 펀딩 {len(frows)}건')


def load_universe(data_dir: Path, symbols: list[str], now_ms: int) -> dict[str, SymbolData]:
    out: dict[str, SymbolData] = {}
    for s in symbols:
        p = data_dir / f'{s}_4h.csv'
        if not p.exists():
            continue
        ft, fr = load_csv_funding(data_dir / f'{s}_funding.csv')
        out[s] = build_symbol(load_csv_bars(p), ft, fr, now_ms)
    return out


# ══════════════════════════════════════════════════════════════
#  장세 (BTC)
# ══════════════════════════════════════════════════════════════

class RegimeClock:
    """시각 T(봉 시작)에 '이미 끝난' BTC 봉으로 장세를 판정한다."""

    def __init__(self, btc: SymbolData):
        self.btc = btc
        self.table = core.regime_table(btc.d1)
        self.adx4 = core.adx4h_table(btc.h4)

    def at(self, t: int) -> str:
        day_start = t // DAY_MS * DAY_MS
        di = self.btc.idx1.get(day_start - DAY_MS)
        if di is None or di < 60:
            return core.UNKNOWN
        k = self.btc.idx4.get(t - H4_MS)
        a4 = self.adx4[k] if k is not None else 0.0
        r = self.table[di]
        return core.classify(r.adx, r.px, r.ema, r.atr_pct, a4, r.slope)


# ══════════════════════════════════════════════════════════════
#  시뮬레이터
# ══════════════════════════════════════════════════════════════

@dataclass
class BTConfig:
    start_ms: int = 1609459200000          # 2021-01-01
    end_ms: int = 2 ** 62
    eq0: float = 10_000.0
    strategies: tuple[str, ...] = cfg.FUT_STRATEGIES
    route: Optional[dict[str, tuple[str, ...]]] = None       # None → cfg.REGIME_ROUTE
    scale_long: dict[str, float] = field(default_factory=dict)   # 장세별 롱 위험 배수 (기본 1.0)
    scale_short: dict[str, float] = field(default_factory=dict)
    risk: float = cfg.FUT_RISK_PCT
    cap_frac: Optional[float] = None
    max_pos: int = cfg.FUT_MAX_POSITIONS
    taker: float = cfg.FUT_TAKER_FEE
    maker: float = cfg.FUT_MAKER_FEE
    funding: bool = True
    slip_mult: float = 1.0
    stop_mult: float = cfg.STOP_SLIP_MULT
    through: float = cfg.LIMIT_THROUGH
    entry: str = 'hybrid'                  # 'market' | 'limit' | 'hybrid'
    limit_bars: int = cfg.ENTRY_LIMIT_SEC // (H4_MS // 1000)
    dd_rule: str = 'live'                  # 'live'(15%→½, 20%→정지 — 라이브처럼 다시 풀리지 않음) | 'none'
    daily_loss: float = cfg.FUT_DAILY_LOSS_LIMIT
    lev: float = cfg.FUT_LEVERAGE
    mmr: float = cfg.FUT_MMR


@dataclass
class Position:
    st: str
    side: int
    qty: float
    entry: float
    sl: float
    tp: float
    sl_dist: float
    entry_ts: int
    fee_in: float
    regime: str
    last: float
    mid_bar: bool
    fund_paid: float = 0.0

    @property
    def risk_usd(self) -> float:
        return self.qty * self.sl_dist


@dataclass
class Trade:
    symbol: str
    st: str
    side: int
    entry_ts: int
    exit_ts: int
    entry: float
    exit_px: float
    reason: str
    net: float
    net_r: float
    risk_usd: float
    fund: float
    regime: str


@dataclass
class BTResult:
    cfg: BTConfig
    trades: list[Trade]
    equity: list[tuple[int, float]]
    max_dd: float
    final: float
    costs: dict[str, float]
    orders: dict[str, int]
    halted_at: Optional[int] = None        # 낙폭 20% 정지가 걸린 봉 (없으면 None)


def simulate(data: dict[str, SymbolData], c: BTConfig) -> BTResult:
    btc = data[cfg.FUT_REGIME_SYMBOL]
    regime_clock = RegimeClock(btc)
    route = c.route if c.route is not None else cfg.REGIME_ROUTE
    symbols = [s for s in cfg.FUT_UNIVERSE if s in data]
    clock = [b.ts for b in btc.h4 if c.start_ms - DAY_MS <= b.ts <= c.end_ms]

    wallet = c.eq0
    peak = c.eq0
    max_dd = 0.0
    dds = 1.0
    halted_at: Optional[int] = None
    pos: dict[str, Position] = {}
    orders: dict[str, dict] = {}
    trades: list[Trade] = []
    equity: list[tuple[int, float]] = []
    pending_sig: list[tuple[str, core.Signal]] = []
    pending_exit: dict[str, str] = {}
    fallback: list[tuple[str, core.Signal]] = []
    recent_exits: list[tuple[int, float]] = []      # (청산 시각, 순손익) — 직전 24시간 손실 한도 판정용
    fptr = dict.fromkeys(symbols, 0)
    costs = {'fee': 0.0, 'slip': 0.0, 'fund': 0.0, 'liq': 0.0}
    ost = {'created': 0, 'filled': 0, 'expired': 0, 'fallback': 0}

    def bar(s: str, t: int) -> Optional[Bar]:
        i = data[s].idx4.get(t)
        return data[s].h4[i] if i is not None else None

    def mtm(t: int, field_name: str) -> float:
        u = 0.0
        for s, p in pos.items():
            b = bar(s, t)
            q = getattr(b, field_name) if b is not None else p.last
            p.last = q
            u += p.side * (q - p.entry) * p.qty
        return wallet + u

    def open_pos(s: str, it: dict, t: int, fill: float, fee: float, slip_cost: float, mid_bar: bool) -> None:
        nonlocal wallet
        wallet -= fee
        costs['fee'] += fee
        costs['slip'] += slip_cost
        side = it['side']
        tp = core.take_profit_price(side, fill, it['sl_dist'], it['rr'])
        pos[s] = Position(st=it['st'], side=side, qty=it['qty'], entry=fill,
                          sl=core.stop_price(side, fill, it['sl_dist']), tp=tp, sl_dist=it['sl_dist'],
                          entry_ts=t, fee_in=fee, regime=it['regime'], last=fill, mid_bar=mid_bar)

    def close(s: str, t: int, px: float, reason: str, maker: bool, slip_cost: float = 0.0) -> None:
        nonlocal wallet
        p = pos[s]
        fee = abs(px * p.qty) * (c.maker if maker else c.taker)
        pnl = p.side * (px - p.entry) * p.qty
        wallet += pnl - fee
        costs['fee'] += fee
        costs['slip'] += slip_cost
        net = pnl - fee - p.fee_in - p.fund_paid
        recent_exits.append((t, net))
        trades.append(Trade(s, p.st, p.side, p.entry_ts, t, p.entry, px, reason, net,
                            net / p.risk_usd if p.risk_usd else 0.0, p.risk_usd, p.fund_paid, p.regime))
        del pos[s]
        pending_exit.pop(s, None)

    def make_intent(s: str, sig: core.Signal, t: int, regime: str, ref_px: float) -> Optional[dict]:
        if dds <= 0:                       # 낙폭 20% 정지 — 수량 0 비교에 기대지 않고 명시적으로 막는다
            return None
        sl_dist = core.sl_distance(sig, ref_px)
        if core.entry_blocked(sl_dist, ref_px, s, c.taker, cfg.slip_for(s) * c.slip_mult):
            return None
        fr = core.funding_asof(data[s].fund_t, data[s].fund_r, t)
        f_scale = core.funding_scale(sig.strategy, fr)
        if f_scale is None:
            return None
        r_scale = (c.scale_long if sig.side == 1 else c.scale_short).get(regime, 1.0)
        eq = mtm(t, 'o')
        qty = core.position_qty(eq, sl_dist, ref_px, c.risk, dds * r_scale, f_scale, c.cap_frac)
        if not (qty > 0):
            return None
        return {'st': sig.strategy, 'side': sig.side, 'sl_dist': sl_dist, 'qty': qty, 'rr': sig.rr,
                'regime': regime}

    for ci, t in enumerate(clock):
        trading = t >= c.start_ms
        day = t // DAY_MS * DAY_MS

        # 1) 펀딩비 정산
        for s in symbols:
            ft, fr = data[s].fund_t, data[s].fund_r
            i = fptr[s]
            p = pos.get(s)
            while i < len(ft) and ft[i] <= t + 60_000:
                if p is not None and c.funding and ft[i] > p.entry_ts + 60_000:
                    b = bar(s, t)
                    q = b.o if b is not None else p.last
                    amt = p.side * p.qty * q * fr[i]
                    wallet -= amt
                    p.fund_paid += amt
                    costs['fund'] += amt
                i += 1
            fptr[s] = i

        # 2) 예약된 청산(추세 이탈) — 시가 시장가
        for s, reason in list(pending_exit.items()):
            if s not in pos:
                continue
            b = bar(s, t)
            if b is None:
                continue
            p = pos[s]
            f = b.o * (1 - p.side * cfg.slip_for(s) * c.slip_mult)
            close(s, t, f, reason, False, abs(b.o - f) * p.qty)
        pending_exit = {}

        # 3) 진입: 시장가 전환분 → 새 신호
        if trading:
            regime = regime_clock.at(t)
            allowed = route.get(regime, ())
            for s, sig in fallback:
                if s in pos or len(pos) + len(orders) >= c.max_pos or sig.strategy not in allowed:
                    continue
                b = bar(s, t)
                if b is None:
                    continue
                it = make_intent(s, sig, t, regime, b.o)
                if it is None:
                    continue
                fill = b.o * (1 + it['side'] * cfg.slip_for(s) * c.slip_mult)
                ost['fallback'] += 1
                open_pos(s, it, t, fill, fill * it['qty'] * c.taker, abs(fill - b.o) * it['qty'], False)
            fallback = []
            if pending_sig:
                recent_exits = [x for x in recent_exits if x[0] > t - DAY_MS]
            loss_block = bool(pending_sig) and core.trailing_day_loss(recent_exits, t, mtm(t, 'o')) <= c.daily_loss
            for s, sig in pending_sig:
                if s in pos or s in orders or len(pos) + len(orders) >= c.max_pos or sig.strategy not in allowed:
                    continue
                if loss_block:
                    continue
                b = bar(s, t)
                if b is None:
                    continue
                if c.entry == 'market':
                    it = make_intent(s, sig, t, regime, b.o)
                    if it is None:
                        continue
                    fill = b.o * (1 + it['side'] * cfg.slip_for(s) * c.slip_mult)
                    open_pos(s, it, t, fill, fill * it['qty'] * c.taker, abs(fill - b.o) * it['qty'], False)
                    continue
                lp = core.entry_limit_price(sig.side, sig.ref, b.o)
                it = make_intent(s, sig, t, regime, lp)
                if it is None:
                    continue
                orders[s] = {**it, 'L': lp, 'expire': ci + c.limit_bars, 'sig': sig}
                ost['created'] += 1
        pending_sig = []

        # 4) 지정가 체결 / 만료
        for s, o in list(orders.items()):
            b = bar(s, t)
            if b is not None and (b.l <= o['L'] * (1 - c.through) if o['side'] == 1 else b.h >= o['L'] * (1 + c.through)):
                del orders[s]
                ost['filled'] += 1
                open_pos(s, o, t, o['L'], o['L'] * o['qty'] * c.maker, 0.0, True)
                continue
            if ci + 1 >= o['expire']:
                del orders[s]
                ost['expired'] += 1
                if c.entry == 'hybrid':
                    fallback.append((s, o['sig']))

        # 5) 장중 손절·익절
        for s, p in list(pos.items()):
            b = bar(s, t)
            if b is None:
                continue
            lg = p.side == 1
            sslip = cfg.slip_for(s) * c.slip_mult * c.stop_mult
            liq = p.entry * (1 - p.side * (1 / c.lev - c.mmr))
            if (b.o <= liq) if lg else (b.o >= liq):
                costs['liq'] += 1
                close(s, t, liq, 'LIQ', False)
                continue
            if (b.o <= p.sl) if lg else (b.o >= p.sl):
                f = b.o * (1 - p.side * sslip)
                close(s, t, f, 'SL', False, abs(b.o - f) * p.qty)
                continue
            if (b.l <= p.sl) if lg else (b.h >= p.sl):
                f = p.sl * (1 - p.side * sslip)
                close(s, t, f, 'SL', False, abs(p.sl - f) * p.qty)
                continue
            if not p.mid_bar and p.tp > 0 and ((b.h >= p.tp * (1 + c.through)) if lg else (b.l <= p.tp * (1 - c.through))):
                close(s, t, p.tp, 'TP', True)
                continue
            p.mid_bar = False

        # 6) 봉 마감 평가 + 낙폭
        eq_c = mtm(t, 'c')
        if trading:
            equity.append((t, eq_c))
        peak = max(peak, eq_c)
        dd = (peak - eq_c) / peak
        if trading and dd > max_dd:
            max_dd = dd
        if c.dd_rule == 'live':
            # 라이브와 같이 한 번 걸리면 풀리지 않는다(라이브는 사람이 --resume 해야 재개). 예전엔 보유 포지션이
            # 회복해 낙폭이 20% 아래로 돌아오면 저절로 다시 들어갔다.
            if halted_at is None and trading and core.dd_scale(dd) <= 0:
                halted_at = t
            dds = 0.0 if halted_at is not None else core.dd_scale(dd)
        else:
            dds = 1.0

        # 7) 일봉 마감 → 추세 이탈 청산 예약 + 신호 수집
        if (t + H4_MS) % DAY_MS == 0:
            for s in symbols:
                sd = data[s]
                j = sd.idx1.get(day)
                if j is None:
                    continue
                p = pos.get(s)
                if p is not None:
                    if core.exit_cross(p.st, sd.pack, j):
                        pending_exit[s] = 'CROSS'
                elif s not in orders:
                    for st in c.strategies:
                        new_sig = core.SIGNALS[st](sd.pack, j)
                        if new_sig is not None:
                            pending_sig.append((s, new_sig))

    last_t = clock[-1]
    for s, p in list(pos.items()):
        b = bar(s, last_t)
        close(s, last_t, b.c if b is not None else p.last, 'END', False)
    return BTResult(cfg=c, trades=trades, equity=equity, max_dd=max_dd, final=wallet, costs=costs, orders=ost,
                    halted_at=halted_at)


# ══════════════════════════════════════════════════════════════
#  리포트
# ══════════════════════════════════════════════════════════════

def _segment(equity: list[tuple[int, float]], a: int, b: int) -> Optional[tuple[float, float, float]]:
    pts = [(t, v) for t, v in equity if a <= t < b]
    if len(pts) < 2:
        return None
    s0, s1 = pts[0][1], pts[-1][1]
    pk, md = s0, 0.0
    for _, v in pts:
        pk = max(pk, v)
        md = max(md, (pk - v) / pk)
    yrs = (pts[-1][0] - pts[0][0]) / (365.25 * DAY_MS)
    cagr = (s1 / s0) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    return s1 / s0 - 1, cagr, md


def _pf(trades: list[Trade]) -> float:
    gw = sum(t.net for t in trades if t.net > 0)
    gl = -sum(t.net for t in trades if t.net <= 0)
    return gw / gl if gl > 0 else math.inf


def summarize(r: BTResult) -> dict:
    yrs = (r.equity[-1][0] - r.equity[0][0]) / (365.25 * DAY_MS)
    out: dict = {
        'final': round(r.final, 2),
        'cagr': (r.final / r.cfg.eq0) ** (1 / yrs) - 1,
        'max_dd': r.max_dd,
        'trades': len(r.trades),
        'win_rate': sum(1 for t in r.trades if t.net > 0) / max(len(r.trades), 1),
        'pf': _pf(r.trades),
        'avg_net_r': sum(t.net_r for t in r.trades) / max(len(r.trades), 1),
        'costs_pct': {k: v / r.cfg.eq0 for k, v in r.costs.items() if k != 'liq'},
        'orders': r.orders,
        'halted_at': r.halted_at,
    }
    years: dict[int, tuple[float, float]] = {}
    for y in range(2021, 2031):
        a = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        b = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        seg = _segment(r.equity, a, b)
        if seg:
            years[y] = (seg[0], seg[2])
    out['years'] = years
    split = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    out['is'] = _segment(r.equity, r.equity[0][0], split)
    out['oos'] = _segment(r.equity, split, r.equity[-1][0] + 1)
    by: dict[str, list[Trade]] = {}
    for t in r.trades:
        by.setdefault(f'{t.st}@{t.regime}', []).append(t)
        by.setdefault(t.st, []).append(t)
    out['by'] = {k: (len(v), _pf(v), sum(t.net_r for t in v) / len(v)) for k, v in by.items()}
    return out


def print_summary(s: dict, label: str = '') -> None:  # pragma: no cover - 출력 전용
    print(f'\n=== {label} ===')
    print(f"최종 ${s['final']:,.0f} | 연 {s['cagr'] * 100:.1f}% | MDD {s['max_dd'] * 100:.1f}% | "
          f"{s['trades']}건 승률 {s['win_rate'] * 100:.0f}% PF {s['pf']:.2f} 평균 {s['avg_net_r']:.3f}R")
    if s['is'] and s['oos']:
        print(f"2021~23 연 {s['is'][1] * 100:.1f}% (MDD {s['is'][2] * 100:.1f}%) | "
              f"2024~ 연 {s['oos'][1] * 100:.1f}% (MDD {s['oos'][2] * 100:.1f}%)")
    print('연도별: ' + ', '.join(f'{y} {v[0] * 100:+.1f}%/{v[1] * 100:.1f}' for y, v in s['years'].items()))
    print('비용(시작자금 대비): ' + ', '.join(f'{k} {v * 100:.1f}%' for k, v in s['costs_pct'].items()))
    print('전략·장세별: ' + ', '.join(f'{k} n{n} PF{pf:.2f}' for k, (n, pf, _) in sorted(s['by'].items())))
    if s.get('halted_at'):
        when = datetime.fromtimestamp(s['halted_at'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        print(f'⚠️ 낙폭 20% 정지: {when} UTC 부터 신규 진입 없음 (라이브는 점검 후 --resume)')


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description='ATLAS 선물 포트폴리오 백테스트')
    ap.add_argument('--data-dir', default='data/futures')
    ap.add_argument('--fetch', action='store_true', help='바이낸스 공개 API에서 데이터 받기')
    ap.add_argument('--risk', type=float, default=cfg.FUT_RISK_PCT)
    ap.add_argument('--entry', choices=('market', 'limit', 'hybrid'), default='hybrid')
    ap.add_argument('--slip-mult', type=float, default=1.0)
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--now-ms', type=int, default=0, help='이 시각 기준으로 끝난 봉만 사용(재현용)')
    a = ap.parse_args(argv)
    d = Path(a.data_dir)
    if a.fetch:
        fetch_to_csv(d, cfg.FUT_UNIVERSE)
    now_ms = a.now_ms or int(time.time() * 1000)
    data = load_universe(d, cfg.FUT_UNIVERSE, now_ms)
    start = int(datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    r = simulate(data, BTConfig(start_ms=start, risk=a.risk, entry=a.entry, slip_mult=a.slip_mult))
    print_summary(summarize(r), f'위험 {a.risk * 100:.2f}% · 진입 {a.entry} · 슬리피지 ×{a.slip_mult}')


if __name__ == '__main__':  # pragma: no cover
    main()

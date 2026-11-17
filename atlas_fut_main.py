"""
ATLAS Futures — 라이브 엔진 (바이낸스 USDⓈ-M, 롱·숏)
=====================================================
규칙은 전부 atlas_fut_core 에 있고(백테스트와 같은 함수), 이 파일은 주문과 상태만 다룬다.

하루 흐름 (UTC):
  00:01  전날 일봉 마감 판정
         ① 장세(BTC) 판정 → 켤 전략 결정 (상승·박스권: S6 롱 / 약한추세·하락: SX 숏)
         ② 보유 포지션 추세 이탈 청산 (S6: 10일 신저가 / SX: 10일 신고가) — 시장가
         ③ 새 신호 → 메이커 지정가 주문 (현재가보다 0.02% 유리한 가격, 4시간 대기)
  상시   30초마다 점검
         · 지정가 체결 → 손절(STOP_MARKET, 조건부 주문 API) + 익절(지정가 reduceOnly) 등록
         · 4시간 안 사지면 지정가 취소 → 조건 재확인 후 시장가
         · 익절·손절 체결 → 반대쪽 보호주문 취소, 거래 기록(실제 체결가·수수료·펀딩비)
         · 보호주문이 사라졌으면 다시 걸고, 손절가를 넘었는데 살아 있으면 시장가 청산
         · 자산 고점 대비 낙폭 15% → 위험 절반, 20% → 신규 진입 정지(수동 재개)

모드 (.env 의 FUT_MODE):
  observe  주문 없음. 매일 장세·신호·가상 수량만 기록 (API 키 불필요) ← 기본값
  testnet  바이낸스 선물 데모(demo-fapi.binance.com)에서 실제 주문
  live     실계좌. FUT_LIVE_CONFIRM 에 확인 문구가 있어야만 켜진다

실행:
  python atlas_fut_main.py            # 상시 실행 (systemd: deploy/atlas-fut.service)
  python atlas_fut_main.py --check    # 설정·계좌·주문 가능 여부 점검만 (읽기 전용)
  python atlas_fut_main.py --once     # 지금 시점으로 일봉 판정 1회 (observe 모드 확인용)
  python atlas_fut_main.py --resume   # 낙폭 20% 정지 해제
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

import atlas_fut_config as cfg
import atlas_fut_core as core
from atlas_fut_core import DAY_MS, H4_MS, Bar
from atlas_fut_exchange import BinanceAPIError, FuturesREST, algo_id_of, algo_status, fmt_dec, round_to

APP_DIR = Path(__file__).parent
log = logging.getLogger('atlas_fut')

CID_PREFIX = 'atl'          # 봇이 낸 주문의 client id 접두사 (E 진입, M 시장가 전환, S 손절, T 익절, X 청산)
ORDER_NOT_FOUND = -2013     # 바이낸스 'Order does not exist'


# ══════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════

def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        v = v.split(' #', 1)[0].strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


@dataclass
class Settings:
    mode: str
    base_url: str
    api_key: str
    api_secret: str
    tg_token: str
    tg_chat_id: str
    db_file: Path
    observe_equity: float
    key_source: str = 'FUT_API_KEY'

    @property
    def trading(self) -> bool:
        return self.mode in ('testnet', 'live')


def load_settings(env: Optional[dict[str, str]] = None) -> Settings:
    """.env → Settings. 이미 설정된 환경변수가 .env보다 우선한다(systemd Environment= 존중)."""
    if env is None:
        env = {**_read_env_file(APP_DIR / '.env'), **os.environ}
    mode = env.get('FUT_MODE', 'observe').strip().lower() or 'observe'
    if mode not in ('observe', 'testnet', 'live'):
        raise SystemExit(f'FUT_MODE={mode!r} — observe / testnet / live 중 하나여야 합니다')
    if mode == 'live' and env.get('FUT_LIVE_CONFIRM', '') != cfg.FUT_LIVE_CONFIRM_PHRASE:
        raise SystemExit('실계좌(live) 모드는 .env에 FUT_LIVE_CONFIRM='
                         f'{cfg.FUT_LIVE_CONFIRM_PHRASE} 가 있어야 켜집니다')
    default_url = cfg.FUT_LIVE_BASE_URL if mode in ('live', 'observe') else cfg.FUT_TESTNET_BASE_URL
    base_url = env.get('FUT_BASE_URL', '').strip() or default_url
    if mode == 'testnet' and 'fapi.binance.com' in base_url and 'demo' not in base_url and 'testnet' not in base_url:
        raise SystemExit('testnet 모드인데 실계좌 주소가 설정돼 있습니다 — FUT_BASE_URL을 확인하세요')
    key, secret = env.get('FUT_API_KEY', '').strip(), env.get('FUT_API_SECRET', '').strip()
    spot_key = env.get('BINANCE_API_KEY', '').strip()
    spot_secret = env.get('BINANCE_API_SECRET', '').strip()
    key_source = 'FUT_API_KEY'
    if mode == 'live' and not (key or secret) and spot_key and spot_secret:
        # 현물 봇 키에 Futures 권한이 켜져 있으면 그대로 같이 쓴다 (같은 계좌, 같은 서버 IP 제한).
        # FUT 키를 하나만 적은 경우는 섞지 않고 아래에서 멈춘다 — 반쯤 설정은 실수일 가능성이 크다.
        key, secret, key_source = spot_key, spot_secret, 'BINANCE_API_KEY (현물 봇과 공유)'
    if mode in ('testnet', 'live') and (not key or not secret):
        hint = (' — 데모 키는 demo.binance.com 에서 발급합니다' if mode == 'testnet'
                else ' (둘 다 비우면 현물 봇의 BINANCE_API_KEY / SECRET 을 씁니다 — 그 키에 Futures 권한 필요)')
        raise SystemExit(f'{mode} 모드에는 FUT_API_KEY / FUT_API_SECRET 이 필요합니다{hint}')
    if mode == 'testnet' and spot_key and key == spot_key:
        raise SystemExit('testnet 모드에 실계좌 키(BINANCE_API_KEY와 같은 값)가 들어 있습니다 — '
                         '데모 서버는 demo.binance.com 에서 발급한 데모 키만 받습니다')
    db = Path(env.get('FUT_DB_FILE', '').strip() or (APP_DIR / cfg.FUT_DB_FILE))
    return Settings(mode=mode, base_url=base_url, api_key=key, api_secret=secret,
                    tg_token=env.get('TG_TOKEN', '').strip(), tg_chat_id=env.get('TG_CHAT_ID', '').strip(),
                    db_file=db, observe_equity=float(env.get('FUT_OBSERVE_EQUITY', '1000') or 1000),
                    key_source=key_source)


# ══════════════════════════════════════════════════════════════
#  저장소 (SQLite)
# ══════════════════════════════════════════════════════════════

class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS fut_positions (
                symbol TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fut_entries (
                symbol TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fut_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, strategy TEXT, side INTEGER,
                entry_ts INTEGER, exit_ts INTEGER, entry REAL, exit REAL, qty REAL, reason TEXT,
                pnl REAL, fees REAL, funding REAL, net REAL, net_r REAL, risk_usd REAL, regime TEXT,
                mode TEXT);
            CREATE TABLE IF NOT EXISTS fut_state (key TEXT PRIMARY KEY, value TEXT);
            """)

    def _all(self, table: str) -> dict[str, dict]:
        with self.lock:
            rows = self.conn.execute(f'SELECT symbol, data FROM {table}').fetchall()  # nosec B608 - 고정 테이블명
        return {r['symbol']: json.loads(r['data']) for r in rows}

    def _put(self, table: str, symbol: str, data: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute(f'INSERT OR REPLACE INTO {table}(symbol, data) VALUES(?, ?)',  # nosec B608
                              (symbol, json.dumps(data)))

    def _del(self, table: str, symbol: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(f'DELETE FROM {table} WHERE symbol=?', (symbol,))  # nosec B608

    def positions(self) -> dict[str, dict]:
        return self._all('fut_positions')

    def put_position(self, symbol: str, data: dict) -> None:
        self._put('fut_positions', symbol, data)

    def del_position(self, symbol: str) -> None:
        self._del('fut_positions', symbol)

    def entries(self) -> dict[str, dict]:
        return self._all('fut_entries')

    def put_entry(self, symbol: str, data: dict) -> None:
        self._put('fut_entries', symbol, data)

    def del_entry(self, symbol: str) -> None:
        self._del('fut_entries', symbol)

    def add_trade(self, t: dict) -> None:
        cols = ('symbol', 'strategy', 'side', 'entry_ts', 'exit_ts', 'entry', 'exit', 'qty', 'reason',
                'pnl', 'fees', 'funding', 'net', 'net_r', 'risk_usd', 'regime', 'mode')
        with self.lock, self.conn:
            self.conn.execute(f'INSERT INTO fut_trades({",".join(cols)}) VALUES({",".join("?" * len(cols))})',  # nosec B608
                              tuple(t.get(c) for c in cols))

    def trades(self, since_ms: int = 0) -> list[dict]:
        with self.lock:
            rows = self.conn.execute('SELECT * FROM fut_trades WHERE exit_ts >= ? ORDER BY id', (since_ms,)).fetchall()
        return [dict(r) for r in rows]

    def get_state(self, key: str, default: str = '') -> str:
        with self.lock:
            row = self.conn.execute('SELECT value FROM fut_state WHERE key=?', (key,)).fetchone()
        return row['value'] if row else default

    def set_state(self, key: str, value: Any) -> None:
        with self.lock, self.conn:
            self.conn.execute('INSERT OR REPLACE INTO fut_state(key, value) VALUES(?, ?)', (key, str(value)))


# ══════════════════════════════════════════════════════════════
#  알림
# ══════════════════════════════════════════════════════════════

class TelegramNotifier:
    """큐 기반 비동기 전송 — 전송 실패가 매매 루프를 막지 않는다."""

    def __init__(self, token: str, chat_id: str, prefix: str = ''):
        self.token, self.chat_id, self.prefix = token, chat_id, prefix
        self.q: queue.Queue[str] = queue.Queue(maxsize=200)
        if token and chat_id:
            threading.Thread(target=self._worker, name='FutTg', daemon=True).start()

    def __call__(self, msg: str) -> None:
        log.info(msg.replace('\n', ' | '))
        if not (self.token and self.chat_id):
            return
        try:
            self.q.put_nowait(self.prefix + msg)
        except queue.Full:
            log.warning('텔레그램 대기열이 가득 차 알림 1건을 버렸습니다 (내용은 위 로그에 있음)')

    def _worker(self) -> None:  # pragma: no cover - 네트워크
        while True:
            msg = self.q.get()
            data = urllib.parse.urlencode({'chat_id': self.chat_id, 'text': msg}).encode()
            try:
                req = urllib.request.Request(f'https://api.telegram.org/bot{self.token}/sendMessage', data=data)
                urllib.request.urlopen(req, timeout=10).close()  # nosec B310 - 고정된 텔레그램 https 주소
            except OSError as e:
                log.warning(f'텔레그램 전송 실패: {e}')


# ══════════════════════════════════════════════════════════════
#  엔진
# ══════════════════════════════════════════════════════════════

def _bars(raw: list[list]) -> list[Bar]:
    return [Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in raw]


def _dec(x: Any) -> Optional[Decimal]:
    """수량 문자열 → Decimal (비었거나 잘못된 값이면 None)."""
    if x is None or x == '':
        return None
    try:
        return Decimal(str(x))
    except ArithmeticError:          # decimal.InvalidOperation
        log.debug(f'수량 변환 실패({x!r})')
        return None


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        log.debug(f'숫자 변환 실패({x!r}) → {default}')
        return default


class FuturesBot:
    SIDE_NAME = {1: 'BUY', -1: 'SELL'}

    def __init__(self, ex: FuturesREST, store: Store, notify: Callable[[str], None], *,
                 mode: str = 'observe', clock: Callable[[], float] = time.time,
                 observe_equity: float = 1000.0):
        self.ex = ex
        self.store = store
        self.notify = notify
        self.mode = mode
        self.trading = mode in ('testnet', 'live')
        self.clock = clock
        self.observe_equity = observe_equity
        self._btc_d1: list[Bar] = []
        self.last_regime = core.UNKNOWN
        self._alerts: dict[str, tuple[float, int]] = {}     # 반복 알림 억제: 키 → (마지막 전송 시각, 억제 횟수)

    # ── 공용 ────────────────────────────────────────────────
    def now_ms(self) -> int:
        return int(self.clock() * 1000)

    def _cid(self, kind: str, symbol: str) -> str:
        return f'{CID_PREFIX}{kind}{symbol}{self.now_ms() % 10**10}'[:36]

    def _rules(self, symbol: str):
        return self.ex.rules(symbol)

    def alert(self, key: str, msg: str) -> None:
        """점검마다 되풀이되는 문제는 ALERT_COOLDOWN_SEC에 한 번만 알린다(로그는 호출부가 매번 남긴다).
        30초마다 알리면 하루 2,880건이 되어 알림 대기열(200)에서 진짜 알림을 밀어낸다."""
        now = self.clock()
        last, skipped = self._alerts.get(key, (0.0, 0))
        if last and now - last < cfg.ALERT_COOLDOWN_SEC:
            self._alerts[key] = (last, skipped + 1)
            return
        self._alerts[key] = (now, 0)
        more = f' (지난 {cfg.ALERT_COOLDOWN_SEC // 60}분 동안 {skipped}번 더 반복)' if skipped else ''
        self.notify(msg + more)

    def alert_resolved(self, key: str, msg: str) -> None:
        """알렸던 문제가 풀렸으면 한 번 알린다."""
        if self._alerts.pop(key, None) is not None:
            self.notify(msg)

    # ── 시세 ────────────────────────────────────────────────
    def daily_closed(self, symbol: str, limit: int = 500) -> list[Bar]:
        return core.drop_forming(_bars(self.ex.klines(symbol, '1d', limit=limit)), DAY_MS, self.now_ms())

    def btc_daily_full(self) -> list[Bar]:
        """BTC 일봉을 상장일부터 전부 (EMA200을 백테스트와 같은 시작점에서 계산하기 위해)."""
        now = self.now_ms()
        if not self._btc_d1:
            out: list[Bar] = []
            start = cfg.FUT_BTC_HISTORY_START_MS
            while start < now:
                chunk = _bars(self.ex.klines(cfg.FUT_REGIME_SYMBOL, '1d', limit=1500, start_ms=start))
                if not chunk:
                    break
                out.extend(chunk)
                start = chunk[-1].ts + DAY_MS
                if len(chunk) < 1500:
                    break
            self._btc_d1 = out
        else:
            recent = _bars(self.ex.klines(cfg.FUT_REGIME_SYMBOL, '1d', limit=10))
            known = {b.ts: i for i, b in enumerate(self._btc_d1)}
            for b in recent:
                if b.ts in known:
                    self._btc_d1[known[b.ts]] = b
                elif b.ts > self._btc_d1[-1].ts:
                    self._btc_d1.append(b)
        return core.drop_forming(self._btc_d1, DAY_MS, now)

    def current_regime(self) -> tuple[str, core.RegimeInputs, float]:
        d1 = self.btc_daily_full()
        h4 = core.drop_forming(_bars(self.ex.klines(cfg.FUT_REGIME_SYMBOL, '4h', limit=60)), H4_MS, self.now_ms())
        regime, r, a4 = core.regime_from_bars(d1, h4)
        self.last_regime = regime
        return regime, r, a4

    def marks(self) -> dict[str, tuple[float, float]]:
        """심볼 → (마크가격, 직전 펀딩비)."""
        out: dict[str, tuple[float, float]] = {}
        for x in self.ex.premium_index() or []:
            out[x['symbol']] = (_f(x.get('markPrice')), _f(x.get('lastFundingRate')))
        return out

    # ── 계좌 ────────────────────────────────────────────────
    def equity(self) -> float:
        if not self.trading:
            return self.observe_equity
        return _f(self.ex.account().get('totalMarginBalance'))

    def exchange_positions(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for p in self.ex.position_risk() or []:
            amt = _f(p.get('positionAmt'))
            if amt != 0:
                out[p['symbol']] = (amt, _f(p.get('entryPrice')))
        return out

    # ── 낙폭·일일 손실 ─────────────────────────────────────
    def _dd_scale_now(self) -> float:
        """직전 판정의 위험 배수 (낙폭 정지 중이면 0)."""
        if self.store.get_state('dd_halt', '0') == '1':
            return 0.0
        return _f(self.store.get_state('dd_scale', '1'), 1.0)

    def update_drawdown(self, eq: float) -> float:
        """고점 대비 낙폭 갱신 후 위험 배수를 돌려준다 (0이면 신규 진입 정지).

        직전 점검 대비 자산이 FUT_EQUITY_JUMP_PCT 넘게 바뀌었으면 이번엔 판정을 미룬다.
        이체 직후에는 잔고가 먼저 바뀌고 이체 기록(income)은 늦게 보일 수 있어서, 그 틈에
        고점을 올리면 입금이 두 번 세이고(가짜 낙폭), 낙폭 정지를 걸면 출금을 손실로 본다.
        """
        if eq <= 0:
            return 1.0
        last = _f(self.store.get_state('last_equity', '0'))
        self.store.set_state('last_equity', f'{eq:.8f}')
        if last <= 0 or abs(eq - last) / last >= cfg.FUT_EQUITY_JUMP_PCT:
            log.info(f'[낙폭] 자산 변동 ${last:,.2f} → ${eq:,.2f} — 이체 반영을 기다려 다음 점검에서 판정')
            return self._dd_scale_now()
        peak = max(_f(self.store.get_state('peak_equity', '0')), eq)
        self.store.set_state('peak_equity', f'{peak:.8f}')
        dd = (peak - eq) / peak
        scale = core.dd_scale(dd)
        tier = '2' if scale == 0 else ('1' if scale < 1 else '0')
        if tier != self.store.get_state('dd_tier', '0'):
            self.store.set_state('dd_tier', tier)
            if tier == '2':
                self.store.set_state('dd_halt', '1')
                self.notify(f'🛑 낙폭 {dd * 100:.1f}% (고점 ${peak:,.2f} → ${eq:,.2f}) — 신규 진입 정지.\n'
                            f'   보유 포지션은 손절·익절 그대로 둡니다. 점검 후 --resume 으로 재개하세요.')
            elif tier == '1':
                self.notify(f'⚠️ 낙폭 {dd * 100:.1f}% — 거래당 위험을 절반으로 줄입니다')
            else:
                self.notify(f'✅ 낙폭 회복 ({dd * 100:.1f}%) — 정상 위험으로 복귀')
        self.store.set_state('dd_scale', f'{scale}')
        if self.store.get_state('dd_halt', '0') == '1':
            return 0.0
        return scale

    def rebase_peak_on_transfers(self) -> None:
        """입출금(선물 지갑 이체)을 낙폭으로 오인하지 않도록 고점을 같은 금액만큼 옮긴다."""
        if not self.trading:
            return
        since = int(self.store.get_state('transfer_checked_ms', '0') or 0)
        if not since:
            # 최초 실행: 지금 잔고가 곧 기준이다. 과거 이체를 더하면 고점이 부풀어 가짜 낙폭이 생긴다.
            self.store.set_state('transfer_checked_ms', self.now_ms())
            return
        rows = self.ex.income(income_type='TRANSFER', start_ms=since + 1) or []
        moved = sum(_f(r.get('income')) for r in rows if str(r.get('asset', 'USDT')) in ('USDT', 'USDC'))
        if rows:
            self.store.set_state('transfer_checked_ms', max(int(r.get('time', since)) for r in rows))
        if moved:
            peak = _f(self.store.get_state('peak_equity', '0'))
            self.store.set_state('peak_equity', f'{max(peak + moved, 0):.8f}')
            self.notify(f'ℹ️ 선물 지갑 {"입금" if moved > 0 else "출금"} ${abs(moved):,.2f} 반영 — 낙폭 기준 조정')

    def day_loss_pct(self, eq: float) -> float:
        """직전 24시간 실현손익 비율 (일일 손실 한도 판정용 — 백테스트와 같은 함수)."""
        now = self.now_ms()
        exits = [(int(t['exit_ts']), float(t['net'] or 0.0)) for t in self.store.trades(now - DAY_MS)]
        return core.trailing_day_loss(exits, now, eq)

    # ── 기동 ────────────────────────────────────────────────
    def startup(self) -> None:
        self.ex.sync_time()
        self.ex.exchange_info()
        if not self.trading:
            log.info('[기동] observe 모드 — 주문 없이 판정만 기록합니다')
            return
        if self.ex.get_position_mode():
            if self.exchange_positions():
                raise SystemExit('헤지(양방향) 모드인데 열린 포지션이 있습니다 — 수동 정리 후 단방향으로 바꾸세요')
            self.ex.set_position_mode(False)
            self.notify('ℹ️ 포지션 모드를 단방향으로 바꿨습니다')
        if self.ex.get_multi_assets():
            self.notify('⚠️ 멀티에셋 모드가 켜져 있습니다 — 격리 마진을 쓰려면 단일 자산 모드여야 합니다')
        held = self.exchange_positions()
        for s in cfg.FUT_UNIVERSE:
            if s in held:
                continue      # 포지션이 있으면 마진 방식·레버리지를 바꿀 수 없다
            try:
                self.ex.set_margin_type(s, cfg.FUT_MARGIN_TYPE)
                self.ex.set_leverage(s, cfg.FUT_LEVERAGE)
            except BinanceAPIError as e:
                log.warning(f'[기동] {s} 마진/레버리지 설정 실패: {e}')
        self.reconcile()

    def reconcile(self) -> None:
        """DB ↔ 거래소 대조. 봇이 꺼진 사이 체결된 것을 정리하고, 기록하기 전에 죽어 남은 주문을
        정리한 뒤, 빠진 보호주문을 다시 건다."""
        held = self.exchange_positions()
        for s, p in self.store.positions().items():
            if s not in held:
                self.finalize_closed(s, p)
        self.reconcile_orders()
        known = set(self.store.positions())
        pending = set(self.store.entries())        # 선기록된 진입 — 다음 점검에서 주문을 찾아 마무리한다
        for s, (amt, px) in held.items():
            if s not in known and s not in pending and s in cfg.FUT_UNIVERSE:
                self.notify(f'⚠️ {s} 봇이 모르는 포지션 {amt} @ {px} — 직접 연 것이면 직접 관리하세요(봇은 건드리지 않음)')
        self.ensure_protection()

    def reconcile_orders(self) -> list[str]:
        """봇 접두사(atl)가 붙은 주문 중 DB가 모르는 것을 정리한다 — 주문을 낸 뒤 기록하기 전에
        죽은 경우다. 보호주문은 같은 종목 포지션 자리가 비어 있으면 입양하고, 나머지는 취소한다.
        (방치하면 남은 GTX 진입 주문이 나중에 체결돼 손절 없는 포지션이 생긴다.) 취소한 목록을 돌려준다."""
        positions = self.store.positions()
        entries = self.store.entries()
        known = {str(v) for e in entries.values() for v in (e.get('order_id'), e.get('client_id')) if v}
        known |= {str(p['tp_order_id']) for p in positions.values() if p.get('tp_order_id')}
        known_algo = {str(p['sl_algo_id']) for p in positions.values() if p.get('sl_algo_id')}
        open_ord = list(self.ex.open_orders() or [])
        open_algo = list(self.ex.open_algo_orders() or [])
        open_ord_ids = {str(o.get('orderId')) for o in open_ord}
        open_algo_ids = {str(algo_id_of(a)) for a in open_algo}
        canceled: list[str] = []
        for o in open_ord:
            cid, s, oid = str(o.get('clientOrderId') or ''), str(o.get('symbol') or ''), o.get('orderId')
            if not cid.startswith(CID_PREFIX) or str(oid) in known or cid in known:
                continue
            p = positions.get(s)
            if cid.startswith(CID_PREFIX + 'T') and p is not None and str(p.get('tp_order_id')) not in open_ord_ids:
                p['tp_order_id'] = oid
                self.store.put_position(s, p)
                log.info(f'[대조] {s} 기록에 없던 익절 주문 {oid} 입양')
                continue
            try:
                self.ex.cancel_order(s, oid)
                canceled.append(f'{s}({cid[:4]})')
            except BinanceAPIError as err:
                log.warning(f'[대조] {s} 기록에 없는 주문 {oid} 취소 실패: {err}')
        for a in open_algo:
            cid, s, aid = str(a.get('clientAlgoId') or ''), str(a.get('symbol') or ''), algo_id_of(a)
            if not cid.startswith(CID_PREFIX) or str(aid) in known_algo:
                continue
            p = positions.get(s)
            if p is not None and str(p.get('sl_algo_id')) not in open_algo_ids:
                p['sl_algo_id'] = aid
                p['sl_qty'] = str(a.get('quantity') or '')
                self.store.put_position(s, p)
                log.info(f'[대조] {s} 기록에 없던 손절 주문 {aid} 입양')
                continue
            try:
                self.ex.cancel_algo_order(aid)
                canceled.append(f'{s}({cid[:4]})')
            except BinanceAPIError as err:
                log.warning(f'[대조] {s} 기록에 없는 손절 주문 {aid} 취소 실패: {err}')
        if canceled:
            self.notify(f'🧹 기록에 없던 봇 주문 {len(canceled)}건 취소(재시작 전 기록 누락): {", ".join(canceled)}')
        return canceled

    # ── 일봉 마감 판정 ─────────────────────────────────────
    def on_day_close(self) -> dict:
        now = self.now_ms()
        regime, r, a4 = self.current_regime()
        allowed = core.strategies_for(regime)
        report: dict = {'regime': regime, 'allowed': allowed, 'exits': [], 'entries': [], 'skipped': []}
        packs: dict[str, core.DailyPack] = {}
        for s in cfg.FUT_UNIVERSE:
            try:
                d1 = self.daily_closed(s)
            except BinanceAPIError as e:
                log.warning(f'[일봉] {s} 조회 실패: {e}')
                continue
            if len(d1) > cfg.SIGNAL_WARMUP:
                packs[s] = core.pack_daily(d1)

        if self.trading:
            try:
                self.reconcile_orders()
            except BinanceAPIError as e:
                log.warning(f'[일봉] 주문 대조 실패: {e}')
        positions = self.store.positions()
        for s, p in positions.items():
            pk = packs.get(s)
            if pk is not None and core.exit_cross(p['strategy'], pk, len(pk.c) - 1):
                report['exits'].append(s)
                if self.trading:
                    self.close_position(s, p, 'CROSS')

        eq = self.equity()
        dds = self.update_drawdown(eq) if self.trading else 1.0
        since_close = now - now // DAY_MS * DAY_MS
        day_loss = self.day_loss_pct(eq) if self.trading else 0.0
        blocked = ''
        if not allowed:
            blocked = f'장세 {regime} — 신규 진입 없음'
        elif since_close >= cfg.ENTRY_LIMIT_SEC * 1000:
            # 늦게 켜졌다. 추세 이탈 청산은 위에서 이미 했고, 지정가 창(마감 후 4시간)이 지난 신호는 쓰지 않는다
            blocked = f'마감 {since_close / 3_600_000:.1f}시간 뒤 판정 — 지정가 창이 지나 신규 진입 생략(청산만)'
        elif dds <= 0:
            blocked = '낙폭 20% 정지 상태'
        elif day_loss <= cfg.FUT_DAILY_LOSS_LIMIT:
            blocked = f'직전 24시간 손실 {day_loss * 100:.1f}% — 일일 손실 한도'
        report['blocked'] = blocked
        marks = self.marks()
        positions = self.store.positions()
        entries = self.store.entries()
        for s in cfg.FUT_UNIVERSE:
            pk = packs.get(s)
            if pk is None or s in positions or s in entries:
                continue
            for st in cfg.FUT_STRATEGIES:
                sig = core.SIGNALS[st](pk, len(pk.c) - 1)
                if sig is None:
                    continue
                if blocked or st not in allowed:
                    report['skipped'].append((s, st, blocked or f'{regime}에서 {st} 꺼짐'))
                    continue
                paper = 0 if self.trading else len(report['entries'])   # observe: 가상 진입도 자리를 차지한다
                if len(positions) + len(entries) + paper >= cfg.FUT_MAX_POSITIONS:
                    report['skipped'].append((s, st, '동시 보유 6개 가득'))
                    continue
                res = self.open_entry(s, sig, regime, eq, dds, marks.get(s, (0.0, 0.0)))
                if res.get('ok'):
                    report['entries'].append(res)
                    entries = self.store.entries()
                else:
                    report['skipped'].append((s, st, res.get('why', '')))
        self.store.set_state('last_day_processed', now // DAY_MS * DAY_MS)
        self.notify(self._day_report(report, r, a4, eq))
        return report

    def _day_report(self, rep: dict, r: core.RegimeInputs, a4: float, eq: float) -> str:
        lines = [f'📅 일봉 마감 판정 [{self.mode}] 장세 {rep["regime"]} '
                 f'(ADX {r.adx:.1f}, 4H ADX {a4:.1f}, BTC {r.px:,.0f} / EMA200 {r.ema:,.0f})',
                 f'   켤 전략: {", ".join(rep["allowed"]) or "없음"} | 자산 ${eq:,.2f}']
        for e in rep['entries']:
            lines.append(f'   ➕ {e["symbol"]} {e["strategy"]} {"롱" if e["side"] == 1 else "숏"} '
                         f'{e["qty"]} @ {e["price"]} (손절 {e["sl"]}, 위험 ${e["risk_usd"]:.2f})')
        for s in rep['exits']:
            lines.append(f'   ➖ {s} 추세 이탈 청산')
        for s, st, why in rep['skipped'][:8]:
            lines.append(f'   · {s} {st} 신호 — 보류: {why}')
        if rep.get('blocked') and not rep['skipped']:
            lines.append(f'   · {rep["blocked"]}')
        return '\n'.join(lines)

    # ── 진입 ────────────────────────────────────────────────
    def _size(self, symbol: str, sig: core.Signal, ref_px: float, eq: float, dds: float,
              funding: float) -> tuple[Optional[Decimal], float, str]:
        sl_dist = core.sl_distance(sig, ref_px)
        why = core.entry_blocked(sl_dist, ref_px, symbol)
        if why:
            return None, sl_dist, why
        f_scale = core.funding_scale(sig.strategy, funding)
        if f_scale is None:
            return None, sl_dist, f'펀딩비 {funding * 100:.3f}% 과밀'
        qty = core.position_qty(eq, sl_dist, ref_px, cfg.FUT_RISK_PCT, dds, f_scale)
        rules = self._rules(symbol)
        q = round_to(qty, rules.step)
        if q < rules.min_qty or q * Decimal(str(ref_px)) < rules.min_notional:
            # 필요한 최소 금액은 '최소 금액'과 '최소 수량 × 가격' 중 큰 쪽 (BTC: 0.001개 ≈ $65 > $50)
            need = max(float(rules.min_notional), float(rules.min_qty) * ref_px)
            return None, sl_dist, f'최소 주문 미달(필요 약 ${need:,.0f}, 계산 ${qty * ref_px:.2f})'
        return q, sl_dist, ''

    def open_entry(self, symbol: str, sig: core.Signal, regime: str, eq: float, dds: float,
                   mark_funding: tuple[float, float]) -> dict:
        mark, funding = mark_funding
        rules = self._rules(symbol)
        if self.trading:
            bt = self.ex.book_ticker(symbol)
            bid, ask = _f(bt.get('bidPrice')), _f(bt.get('askPrice'))
            mid = (bid + ask) / 2 if bid and ask else mark
        else:
            bid = ask = mid = mark or sig.ref
        lp = core.entry_limit_price(sig.side, sig.ref, mid)
        if self.trading:
            lp = min(lp, bid) if sig.side == 1 else max(lp, ask)       # 메이커 보장
        price = round_to(lp, rules.tick, up=(sig.side == -1))
        qty, sl_dist, why = self._size(symbol, sig, float(price), eq, dds, funding)
        if qty is None:
            return {'ok': False, 'why': why}
        now = self.now_ms()
        info = {'symbol': symbol, 'strategy': sig.strategy, 'side': sig.side, 'qty': fmt_dec(qty),
                'price': fmt_dec(price), 'limit0': fmt_dec(price), 'sl': round(sig.sl, 8), 'sl_dist': sl_dist,
                'rr': sig.rr, 'ref': sig.ref, 'regime': regime, 'created_ms': now,
                # 일봉 마감 기준 — 늦게 켜져도 지정가 창이 늘어나지 않는다(백테스트: 첫 4시간봉)
                'expire_ms': now // DAY_MS * DAY_MS + (cfg.ENTRY_LIMIT_SEC + cfg.DAY_CLOSE_DELAY_SEC) * 1000,
                'risk_usd': float(qty) * sl_dist, 'ok': True}
        if not self.trading:
            return info
        cid = self._cid('E', symbol)
        info.update(order_id=None, client_id=cid, status='PENDING_NEW')
        self.store.put_entry(symbol, info)            # 선기록 — 주문 직후 멈춰도 재시작 때 client id로 찾는다
        try:
            o = self.ex.new_order(symbol, self.SIDE_NAME[sig.side], 'LIMIT', qty, price=price,
                                  time_in_force='GTX', client_id=cid)
        except BinanceAPIError as e:
            self.store.del_entry(symbol)
            log.warning(f'[진입] {symbol} 지정가 주문 거부: {e}')
            return {'ok': False, 'why': f'주문 거부 {e.code} {e.msg}'}
        info.update(order_id=o.get('orderId'), status=o.get('status'))
        self.store.put_entry(symbol, info)
        return info

    def _entry_order(self, symbol: str, e: dict) -> dict:
        """진입 기록이 가리키는 주문. 주문 번호를 받기 전에 멈췄으면 미리 적어 둔 client id로 찾는다."""
        if e.get('order_id'):
            return self.ex.query_order(symbol, e['order_id'])
        return self.ex.query_order(symbol, client_id=e.get('client_id'))

    def _entry_done(self, symbol: str, e: dict, fill: float, qty: Decimal, maker: bool) -> None:
        """체결 → 포지션 기록·보호주문 → 진입 기록 삭제. 이 순서라야 중간에 멈춰도 포지션을 잃지 않는다."""
        self.on_filled(symbol, e, fill, qty, maker)
        self.store.del_entry(symbol)

    def _entry_missing(self, symbol: str, e: dict) -> None:
        """미리 적어 둔 주문이 거래소에 없다 = 보내기 전에 멈췄다. 앞서 체결된 지정가 분량만 지킨다."""
        part = Decimal(str(e.get('limit_filled') or '0'))
        log.warning(f'[진입] {symbol} 기록만 있고 거래소에 주문이 없음 — 진입 정리')
        if part > 0:
            self._entry_done(symbol, e, _f(e.get('limit_avg')), part, maker=True)
            return
        self.store.del_entry(symbol)
        self.notify(f'· {symbol} {e.get("strategy", "")} 주문을 보내기 전에 멈춰 이번 진입은 건너뜁니다')

    def _finish_market(self, symbol: str, e: dict, o: dict) -> None:
        """시장가 전환 주문의 결과로 진입을 마무리한다 (보낸 직후 멈췄다가 재시작한 경우도 여기로 온다)."""
        if str(o.get('status', '')) in ('NEW', 'PARTIALLY_FILLED'):
            return                      # 시장가는 곧 끝난다 — 다음 점검에서 다시 본다
        part = Decimal(str(e.get('limit_filled') or '0'))
        q = Decimal(str(o.get('executedQty') or '0'))
        total = part + q
        if total <= 0:
            self.store.del_entry(symbol)
            return
        avg = (Decimal(str(_f(e.get('limit_avg')))) * part + Decimal(str(_f(o.get('avgPrice')))) * q) / total
        self._entry_done(symbol, e, float(avg), total, maker=(q == 0))

    def poll_entries(self) -> None:
        now = self.now_ms()
        positions = self.store.positions()
        for s, e in self.store.entries().items():
            if s in positions:               # 포지션을 기록한 직후(진입 기록을 지우기 전) 멈춘 경우
                self.store.del_entry(s)
                continue
            try:
                o = self._entry_order(s, e)
            except BinanceAPIError as err:
                if err.code == ORDER_NOT_FOUND and not e.get('order_id'):
                    self._entry_missing(s, e)
                else:
                    log.warning(f'[진입] {s} 주문 조회 실패: {err}')
                continue
            if e.get('status') == 'MARKET_SENT':
                self._finish_market(s, e, o)
                continue
            if not e.get('order_id') and o.get('orderId'):
                e['order_id'] = o.get('orderId')
                self.store.put_entry(s, e)
            st = str(o.get('status', ''))
            filled = Decimal(str(o.get('executedQty') or '0'))
            avg = _f(o.get('avgPrice'))
            if st == 'FILLED':
                self._entry_done(s, e, avg, filled, maker=True)
                continue
            expired = now >= int(e['expire_ms'])
            if st in ('NEW', 'PARTIALLY_FILLED'):
                if not expired:
                    continue
                try:
                    o = self.ex.cancel_order(s, e.get('order_id'))
                except BinanceAPIError as err:
                    log.warning(f'[진입] {s} 취소 실패(체결됐을 수 있음): {err}')
                    o = self.ex.query_order(s, e.get('order_id'))
                filled = Decimal(str(o.get('executedQty') or '0'))
                avg = _f(o.get('avgPrice'))
                if str(o.get('status')) == 'FILLED':
                    self._entry_done(s, e, avg, filled, maker=True)
                else:
                    self.fallback_market(s, e, filled, avg)
                continue
            if st == 'EXPIRED':
                if filled == 0 and not expired:
                    self.repeg(s, e)          # 메이커 거부: 거는 순간 가격이 닿아 GTX가 취소됐다
                else:
                    self.fallback_market(s, e, filled, avg)
                continue
            # CANCELED·REJECTED — 봇이 취소한 것이 아니다(앱에서 직접 취소 등). 되살리지 않는다.
            if filled > 0:
                self._entry_done(s, e, avg, filled, maker=True)
            else:
                self.store.del_entry(s)
            self.notify(f'· {s} 진입 주문이 밖에서 취소됨({st}) — 다시 걸지 않습니다'
                        + (f' (이미 체결된 {fmt_dec(filled)}에는 손절·익절을 겁니다)' if filled > 0 else ''))

    def _resize(self, symbol: str, e: dict, px: float) -> tuple[Optional[Decimal], float, str]:
        """진입 가격이 바뀔 때(재주문·시장가 전환) 신호의 **절대 손절가**로 손절 거리를 다시 재고,
        손절폭·비용 게이트와 수량을 처음부터 다시 계산한다."""
        sig = core.Signal(e['strategy'], int(e['side']), float(e['sl']), float(e['rr']), float(e['ref']), 0.0)
        eq = self.equity()
        dds = self.update_drawdown(eq)
        funding = self.marks().get(symbol, (0.0, 0.0))[1]
        return self._size(symbol, sig, px, eq, dds, funding)

    def repeg(self, symbol: str, e: dict) -> None:
        """메이커 거부(EXPIRED) → 지금 호가로 다시 건다. 처음 지정가보다 불리하게 쫓아가지 않고,
        가격이 움직였으면 시장가 전환과 같은 가드(절대 손절가 기준 재계산)를 거친다.
        예전에는 손절 거리를 그대로 들고 가서, 급락 중 손절가 아래에서 롱을 다시 걸고 손절도 함께 끌어내렸다."""
        side = int(e['side'])
        n = int(e.get('repegs', 0)) + 1
        if n > cfg.ENTRY_MAX_REPEG:
            self.store.del_entry(symbol)
            self.notify(f'· {symbol} {e["strategy"]} 메이커 주문이 {cfg.ENTRY_MAX_REPEG}번 거부돼 이번 진입은 건너뜁니다')
            return
        bt = self.ex.book_ticker(symbol)
        rules = self._rules(symbol)
        px = _f(bt.get('bidPrice')) if side == 1 else _f(bt.get('askPrice'))
        price = round_to(px * (1 - side * cfg.ENTRY_POST_OFFSET), rules.tick, up=(side == -1))
        first = _dec(e.get('limit0'))
        if first is not None:
            price = min(price, first) if side == 1 else max(price, first)
        qty, sl_dist, why = self._resize(symbol, e, float(price))
        if qty is None:
            self.store.del_entry(symbol)
            self.notify(f'· {symbol} {e["strategy"]} 재주문 취소 — {why} (호가 {fmt_dec(price)}, 손절 {float(e["sl"]):g})')
            return
        prev = dict(e)
        cid = self._cid('E', symbol)
        e.update(order_id=None, client_id=cid, price=fmt_dec(price), qty=fmt_dec(qty), sl_dist=sl_dist,
                 risk_usd=float(qty) * sl_dist, repegs=n, status='PENDING_NEW')
        self.store.put_entry(symbol, e)               # 선기록
        try:
            o = self.ex.new_order(symbol, self.SIDE_NAME[side], 'LIMIT', qty, price=price,
                                  time_in_force='GTX', client_id=cid)
        except BinanceAPIError as err:
            log.warning(f'[진입] {symbol} 재주문 실패(다음 점검에서 다시): {err}')
            self.store.put_entry(symbol, {**prev, 'repegs': n})
            return
        e.update(order_id=o.get('orderId'), status=o.get('status'))
        self.store.put_entry(symbol, e)

    def fallback_market(self, symbol: str, e: dict, filled: Decimal, avg: float) -> None:
        """4시간 안에 못 산 진입 → 조건을 다시 확인하고 남은 수량을 시장가로. 진입 기록은 여기서 끝낸다."""
        side = int(e['side'])
        regime, _, _ = self.current_regime()
        still_ok = e['strategy'] in core.strategies_for(regime) and symbol not in self.store.positions()
        others = len(self.store.positions()) + len([k for k in self.store.entries() if k != symbol])
        if not still_ok or others >= cfg.FUT_MAX_POSITIONS:
            if filled > 0:
                self._entry_done(symbol, e, avg, filled, maker=True)
            else:
                self.store.del_entry(symbol)
                self.notify(f'· {symbol} {e["strategy"]} 지정가 미체결 — 장세({regime})·자리 조건이 바뀌어 진입 취소')
            return
        bt = self.ex.book_ticker(symbol)
        px = _f(bt.get('askPrice')) if side == 1 else _f(bt.get('bidPrice'))
        remaining = Decimal(e['qty']) - filled
        if filled == 0:
            qty, sl_dist, why = self._resize(symbol, e, px)
            if qty is None:
                self.store.del_entry(symbol)
                self.notify(f'· {symbol} {e["strategy"]} 시장가 전환 보류: {why}')
                return
            e = {**e, 'sl_dist': sl_dist, 'qty': fmt_dec(qty)}
            remaining = qty
        rules = self._rules(symbol)
        remaining = round_to(float(remaining), rules.market_step)
        if remaining <= 0 or remaining * Decimal(str(px)) < rules.min_notional:
            if filled > 0:
                self._entry_done(symbol, e, avg, filled, maker=True)
            else:
                self.store.del_entry(symbol)
            return
        cid = self._cid('M', symbol)
        e = {**e, 'status': 'MARKET_SENT', 'order_id': None, 'client_id': cid,
             'limit_filled': fmt_dec(filled), 'limit_avg': avg}
        self.store.put_entry(symbol, e)               # 선기록 — 보낸 직후 멈춰도 재시작 때 이 주문을 찾는다
        try:
            o = self.ex.new_order(symbol, self.SIDE_NAME[side], 'MARKET', remaining, client_id=cid)
        except BinanceAPIError as err:
            log.error(f'[진입] {symbol} 시장가 진입 실패: {err}')
            self.notify(f'⚠️ {symbol} 시장가 진입 실패: {err.code} {err.msg}')
            if filled > 0:
                self._entry_done(symbol, e, avg, filled, maker=True)
            else:
                self.store.del_entry(symbol)
            return
        self._finish_market(symbol, e, o)

    def on_filled(self, symbol: str, e: dict, fill: float, qty: Decimal, maker: bool) -> None:
        side = int(e['side'])
        sl_dist = float(e['sl_dist'])
        pos = {'strategy': e['strategy'], 'side': side, 'qty': fmt_dec(qty), 'entry': fill,
               'sl': core.stop_price(side, fill, sl_dist), 'sl_dist': sl_dist, 'rr': float(e['rr']),
               'tp': core.take_profit_price(side, fill, sl_dist, float(e['rr'])),
               'entry_ms': self.now_ms(), 'regime': e.get('regime', ''), 'maker': maker,
               'sl_algo_id': None, 'tp_order_id': None}
        self.store.put_position(symbol, pos)
        self.place_protection(symbol, pos)
        self.notify(f'✅ {symbol} {pos["strategy"]} {"롱" if side == 1 else "숏"} 진입 {pos["qty"]} @ {fill:g}\n'
                    f'   손절 {pos["sl"]:g} / 익절 {pos["tp"]:g} (위험 ${float(qty) * sl_dist:.2f})')

    # ── 보호주문 ────────────────────────────────────────────
    def place_protection(self, symbol: str, pos: dict) -> None:
        rules = self._rules(symbol)
        side = int(pos['side'])
        close_side = self.SIDE_NAME[-side]
        qty = Decimal(pos['qty'])
        if not pos.get('sl_algo_id'):
            trig = round_to(pos['sl'], rules.tick, up=(side == 1))      # 롱 손절은 올림(진입가 쪽) — 위험이 계획보다 커지지 않게
            try:
                a = self.ex.new_stop_market(symbol, close_side, qty, trig, working_type=cfg.SL_WORKING_TYPE,
                                            client_algo_id=self._cid('S', symbol))
                pos['sl_algo_id'] = algo_id_of(a)
                pos['sl_qty'] = fmt_dec(qty)
                self.alert_resolved(f'sl_{symbol}', f'✅ {symbol} 손절 주문 복구')
            except BinanceAPIError as err:
                log.error(f'[보호] {symbol} 손절 주문 실패: {err}')
                self.alert(f'sl_{symbol}', f'🚨 {symbol} 손절 주문 실패 — 다음 점검에서 재시도, 그 사이 소프트 손절 감시: '
                                          f'{err.code} {err.msg}')
        if not pos.get('tp_order_id') and pos.get('tp', 0) > 0:
            tp = round_to(pos['tp'], rules.tick, up=(side == -1))
            try:
                o = self.ex.new_order(symbol, close_side, 'LIMIT', qty, price=tp, time_in_force='GTC',
                                      reduce_only=True, client_id=self._cid('T', symbol))
                pos['tp_order_id'] = o.get('orderId')
            except BinanceAPIError as err:
                log.warning(f'[보호] {symbol} 익절 주문 실패(재시도 예정): {err}')
        self.store.put_position(symbol, pos)

    def ensure_protection(self) -> None:
        positions = self.store.positions()
        if not positions or not self.trading:
            return
        open_algo = {algo_id_of(a) for a in self.ex.open_algo_orders() or []}
        open_ord = {o.get('orderId') for o in self.ex.open_orders() or []}
        for s, p in positions.items():
            changed = False
            if p.get('sl_algo_id') and p['sl_algo_id'] not in open_algo:
                p['sl_algo_id'] = None
                changed = True
            if p.get('tp_order_id') and p['tp_order_id'] not in open_ord:
                p['tp_order_id'] = None
                changed = True
            if changed or not p.get('sl_algo_id') or not p.get('tp_order_id'):
                self.place_protection(s, p)

    def _cancel_protection(self, symbol: str, pos: dict) -> None:
        if pos.get('tp_order_id'):
            try:
                self.ex.cancel_order(symbol, pos['tp_order_id'])
            except BinanceAPIError as err:
                log.info(f'[보호] {symbol} 익절 주문 취소 생략: {err.code}')
        if pos.get('sl_algo_id'):
            try:
                self.ex.cancel_algo_order(pos['sl_algo_id'])
            except BinanceAPIError as err:
                log.info(f'[보호] {symbol} 손절 주문 취소 생략: {err.code}')

    # ── 청산 ────────────────────────────────────────────────
    def close_position(self, symbol: str, pos: dict, reason: str) -> None:
        side = int(pos['side'])
        held = self.exchange_positions().get(symbol)
        if held:
            qty = round_to(abs(held[0]), self._rules(symbol).market_step)
            try:
                self.ex.new_order(symbol, self.SIDE_NAME[-side], 'MARKET', qty, reduce_only=True,
                                  client_id=self._cid('X', symbol))
            except BinanceAPIError as err:
                log.error(f'[청산] {symbol} 청산 주문 실패({reason}): {err}')
                self.alert(f'close_{symbol}', f'🚨 {symbol} 청산 주문 실패({reason}): {err.code} {err.msg}')
                return
        self._cancel_protection(symbol, pos)
        self.finalize_closed(symbol, pos, reason)

    def finalize_closed(self, symbol: str, pos: dict, reason: str = '') -> None:
        """포지션이 사라졌을 때: 사유 판정 → 실제 체결가·수수료·펀딩비로 거래 기록."""
        side = int(pos['side'])
        if not reason:
            reason = 'EXTERNAL'
            if pos.get('tp_order_id'):
                try:
                    if str(self.ex.query_order(symbol, pos['tp_order_id']).get('status')) == 'FILLED':
                        reason = 'TP'
                except BinanceAPIError as err:
                    log.info(f'[기록] {symbol} 익절 주문 조회 실패: {err.code}')
            if reason == 'EXTERNAL' and pos.get('sl_algo_id'):
                try:
                    if algo_status(self.ex.query_algo_order(pos['sl_algo_id'])) in ('TRIGGERED', 'FINISHED', 'FILLED'):
                        reason = 'SL'
                except BinanceAPIError as err:
                    log.info(f'[기록] {symbol} 손절 주문 조회 실패: {err.code}')
            self._cancel_protection(symbol, pos)
        entry_ms = int(pos.get('entry_ms', 0))
        # userTrades는 한 번에 7일까지만 돌려준다(기본: 최근 7일). 보유가 더 길면 진입 체결이 빠지므로
        # 청산 체결은 최근 7일에서 찾고, 진입 수수료는 체결 방식(메이커/테이커)으로 추정한다.
        recent = [t for t in (self.ex.user_trades(symbol) or []) if int(t.get('time', 0)) >= entry_ms]
        exit_fills = [t for t in recent if str(t.get('side')) == self.SIDE_NAME[-side]]
        qty = sum(_f(t.get('qty')) for t in exit_fills)
        exit_px = (sum(_f(t.get('price')) * _f(t.get('qty')) for t in exit_fills) / qty) if qty else _f(pos['entry'])
        pnl = sum(_f(t.get('realizedPnl')) for t in exit_fills)
        entry_fee = _f(pos['entry']) * _f(pos['qty']) * (cfg.FUT_MAKER_FEE if pos.get('maker') else cfg.FUT_TAKER_FEE)
        fees = entry_fee + sum(_f(t.get('commission')) for t in exit_fills)
        funding = 0.0
        try:
            funding = sum(_f(x.get('income')) for x in
                          self.ex.income(income_type='FUNDING_FEE', symbol=symbol, start_ms=entry_ms) or [])
        except BinanceAPIError as err:
            log.info(f'[기록] {symbol} 펀딩비 조회 실패: {err.code}')
        risk = _f(pos['qty']) * _f(pos['sl_dist'])
        net = pnl - fees + funding
        self.store.add_trade({'symbol': symbol, 'strategy': pos['strategy'], 'side': side, 'entry_ts': entry_ms,
                              'exit_ts': self.now_ms(), 'entry': _f(pos['entry']), 'exit': exit_px,
                              'qty': _f(pos['qty']), 'reason': reason, 'pnl': pnl, 'fees': fees,
                              'funding': funding, 'net': net, 'net_r': net / risk if risk else 0.0,
                              'risk_usd': risk, 'regime': pos.get('regime', ''), 'mode': self.mode})
        self.store.del_position(symbol)
        emoji = '✅' if net > 0 else '❌'
        self.notify(f'{emoji} {symbol} {pos["strategy"]} 청산({reason}) {_f(pos["entry"]):g} → {exit_px:g} | '
                    f'순손익 ${net:+.2f} ({net / risk if risk else 0:+.2f}R, 수수료 ${fees:.2f}, 펀딩 ${funding:+.2f})')

    def poll_positions(self) -> None:
        positions = self.store.positions()
        if not positions:
            return
        held = self.exchange_positions()
        marks = self.marks()
        for s, p in positions.items():
            if s not in held:
                self.finalize_closed(s, p)
                continue
            amt = abs(held[s][0])
            if abs(amt - _f(p['qty'])) > 1e-12:
                p['qty'] = fmt_dec(Decimal(str(amt)))      # 익절 일부 체결 등
                self.store.put_position(s, p)
            sl_qty = _dec(p.get('sl_qty'))
            if p.get('sl_algo_id') and sl_qty is not None and sl_qty != _dec(p['qty']):
                self._rearm_stop(s, p)
            mark = marks.get(s, (0.0, 0.0))[0]
            side = int(p['side'])
            if mark and ((side == 1 and mark <= p['sl']) or (side == -1 and mark >= p['sl'])):
                # 거래소 손절이 살아 있으면 곧 체결된다 — 한 번 더 기다리고, 두 번 연속이면 직접 청산
                k = f'softsl_{s}'
                if self.store.get_state(k) == '1':
                    self.alert(f'softsl_{s}', f'🚨 {s} 손절가를 넘었는데 포지션이 남아 있어 시장가로 청산합니다')
                    self.store.set_state(k, '0')
                    self.close_position(s, p, 'SOFT_SL')
                else:
                    self.store.set_state(k, '1')
            else:
                self.store.set_state(f'softsl_{s}', '0')
        self.ensure_protection()

    def _rearm_stop(self, symbol: str, pos: dict) -> None:
        """포지션 수량이 바뀌면(익절 일부 체결 등) 손절 수량을 맞춘다. 새 손절을 먼저 걸고 옛것을
        지운다 — 반대 순서면 그 사이 손절이 없다. 새 주문이 실패하면 옛 손절을 남기고 다음 점검에서
        다시 한다(reduceOnly라 넘치게 닫지는 않지만, 포지션보다 큰 수량은 발동 때 거부될 수 있다)."""
        old = pos.get('sl_algo_id')
        pos['sl_algo_id'] = None
        self.place_protection(symbol, pos)          # 익절 주문은 남은 수량으로 그대로 걸려 있다
        if pos.get('sl_algo_id'):
            try:
                self.ex.cancel_algo_order(old)
            except BinanceAPIError as err:
                log.info(f'[보호] {symbol} 옛 손절 취소 생략: {err.code}')
            log.info(f'[보호] {symbol} 포지션 수량 변경 → 손절 수량 {pos["sl_qty"]}로 다시 걸었습니다')
        else:
            pos['sl_algo_id'] = old
            self.store.put_position(symbol, pos)

    def poll(self) -> None:
        if not self.trading:
            return
        self.poll_entries()
        self.poll_positions()
        # 이체를 먼저 반영해야 낙폭 판정이 입출금을 손익으로 보지 않는다 (매 점검, income weight 30)
        try:
            self.rebase_peak_on_transfers()
        except BinanceAPIError as e:
            log.warning(f'[낙폭] 이체 기록 조회 실패 — 이번 점검은 이체 반영 없이 판정: {e}')
        self.update_drawdown(self.equity())

    # ── 루프 ────────────────────────────────────────────────
    def day_due(self) -> bool:
        """오늘 일봉 판정을 아직 안 했고 마감 후 대기시간이 지났는가. 늦게 켜져도 그날 판정은 한다 —
        추세 이탈 청산은 늦어도 해야 하고, 신규 진입만 on_day_close가 지정가 창(4시간)으로 거른다.
        (예전에는 2시간이 지나면 그날을 통째로 건너뛰어 청산까지 다음 자정으로 밀렸다.)"""
        now = self.now_ms()
        day0 = now // DAY_MS * DAY_MS
        done = int(self.store.get_state('last_day_processed', '0') or 0)
        return done < day0 and now - day0 >= cfg.DAY_CLOSE_DELAY_SEC * 1000

    def run_forever(self, kill_switch: Path, sleep: Callable[[float], None] = time.sleep) -> None:  # pragma: no cover
        while not kill_switch.exists():
            try:
                if self.trading:
                    self.poll()
                if self.day_due():
                    self.on_day_close()
            except BinanceAPIError as e:
                log.warning(f'[루프] 거래소 오류: {e}')
            except OSError as e:
                log.warning(f'[루프] 네트워크 오류: {e}')
            except Exception as e:   # 루프가 죽으면 보호주문 감시도 멈춘다 — 알리고 계속 돈다
                log.exception('[루프] 예상 못 한 오류')
                self.alert(f'loop_{type(e).__name__}', f'🚨 선물봇 루프 오류: {type(e).__name__}: {e}')
            sleep(cfg.POLL_SEC)
        self.notify('🔴 선물봇 종료 (킬스위치)')


# ══════════════════════════════════════════════════════════════
#  점검 리포트 (--check)
# ══════════════════════════════════════════════════════════════

CHECK_SL_PCT = 0.10    # 주문 가능 여부를 재는 기준 거래: 손절폭 10% (명목가 = 위험 ÷ 10% = 자산의 5%)


def _probe_signal(px: float) -> core.Signal:
    """주문 가능 여부를 재는 가상 롱 신호 (손절폭 10%)."""
    return core.Signal('S6', 1, px * (1 - CHECK_SL_PCT), cfg.RR_FALLBACK, px, 0.0)


def blocked_symbols(bot: FuturesBot, eq: float, marks: dict[str, tuple[float, float]],
                    dds: float = 1.0) -> list[str]:
    """손절폭 10% 거래를 이 자산으로 낼 수 없는 종목. 실제 주문과 **같은 계산 경로**(_size)를 써서
    최소 수량(BTC 0.001개 ≈ $65 > 최소 금액 $50)과 수량 내림(ETH $20 → 0.010개 = $18.2)까지 반영한다.
    예전엔 명목가와 최소 금액만 비교해서 자산 $1,000에서 BTC를 '가능'으로 잘못 표시했다."""
    out = []
    for s in cfg.FUT_UNIVERSE:
        px = marks.get(s, (0.0, 0.0))[0]
        if px <= 0 or bot._size(s, _probe_signal(px), px, eq, dds, 0.0)[0] is None:
            out.append(s)
    return out


def min_equity_for(bot: FuturesBot, symbol: str, px: float, dds: float = 1.0) -> float:
    """이 종목에서 손절폭 10% 거래를 낼 수 있는 최소 자산. 식으로 구한 뒤 실제 계산 경로(_size)로
    확인한다 — 수량 내림과 부동소수 오차 때문에 식의 값에서 딱 한 단계 모자랄 수 있다."""
    r = bot._rules(symbol)
    step, min_qty, min_notional = float(r.step), float(r.min_qty), float(r.min_notional)
    q = max(min_qty, math.ceil(min_notional / px / step - 1e-9) * step if step > 0 else min_notional / px)
    eq = q * px * CHECK_SL_PCT / (cfg.FUT_RISK_PCT * dds)
    for _ in range(200):
        if bot._size(symbol, _probe_signal(px), px, eq, dds, 0.0)[0] is not None:
            break
        eq *= 1.001
    return eq


def check_report(bot: FuturesBot) -> list[str]:
    lines = [f'모드: {bot.mode} | 주소: {bot.ex.base_url}']
    regime, r, a4 = bot.current_regime()
    lines.append(f'장세: {regime} (BTC {r.px:,.0f}, EMA200 {r.ema:,.0f}, ADX {r.adx:.1f}, 4H ADX {a4:.1f}) '
                 f'→ 켤 전략 {", ".join(core.strategies_for(regime)) or "없음"}')
    eq = bot.equity()
    lines.append(f'선물 지갑 자산: ${eq:,.2f}' + (' (observe 가정값)' if not bot.trading else ''))
    marks = bot.marks()
    need = {s: min_equity_for(bot, s, marks[s][0]) for s in cfg.FUT_UNIVERSE if marks.get(s, (0.0, 0.0))[0] > 0}
    blocked = blocked_symbols(bot, eq, marks)
    later = [s for s in blocked_symbols(bot, eq, marks, dds=0.5) if s not in blocked]
    lines.append(f'거래당 위험 ${eq * cfg.FUT_RISK_PCT:.2f} — 손절폭 10% 거래를 낼 수 없는 종목(최소 주문 미달): '
                 + (', '.join(f'{s}(자산 ${need[s]:,.0f} 필요)' if s in need else f'{s}(시세 없음)' for s in blocked)
                    or '없음'))
    if later:
        lines.append(f'   낙폭 15%(위험 절반)가 되면 더 빠지는 종목: {", ".join(later)}')
    missing = [s for s in cfg.FUT_UNIVERSE if s not in need]
    if not need:
        lines.append('전 종목이 열리는 자산: 알 수 없음 (시세를 받지 못함)')
    elif missing:
        # 시세가 빠진 종목을 빼고 구한 값을 '전 종목' 값처럼 보이면 안 된다 — BTC·ETH가 빠지면 $1,300이 $400으로 보인다
        top = max(need, key=lambda s: need[s])
        lines.append(f'전 종목이 열리는 자산: 알 수 없음 (시세 없는 {len(missing)}종목: {", ".join(missing)}) — '
                     f'시세 있는 {len(need)}종목 기준 ${need[top]:,.0f} '
                     f'(낙폭 15% 상태 ${min_equity_for(bot, top, marks[top][0], dds=0.5):,.0f})')
    else:
        top = max(need, key=lambda s: need[s])
        lines.append(f'전 종목이 열리는 자산: ${need[top]:,.0f} (낙폭 15% 상태 '
                     f'${min_equity_for(bot, top, marks[top][0], dds=0.5):,.0f}) — 가장 비싼 {top}, 그 가격에 비례')
    if bot.trading:
        lines.append(f'포지션 모드: {"헤지(바꿔야 함)" if bot.ex.get_position_mode() else "단방향 OK"} | '
                     f'멀티에셋: {"켜짐(끄는 것 권장)" if bot.ex.get_multi_assets() else "꺼짐 OK"}')
        lines.append(f'열린 포지션: {bot.exchange_positions() or "없음"}')
        lines.append(f'봇 기록 포지션: {list(bot.store.positions()) or "없음"} | 대기 진입: {list(bot.store.entries()) or "없음"}')
    lines.append(f'낙폭 정지: {"예 (--resume 필요)" if bot.store.get_state("dd_halt", "0") == "1" else "아니오"}')
    return lines


def _setup_logging() -> None:  # pragma: no cover
    (APP_DIR / cfg.FUT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(APP_DIR / cfg.FUT_LOG_DIR / 'atlas_fut.log', maxBytes=20 * 1024 * 1024,
                             backupCount=5, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[fh, logging.StreamHandler(sys.stdout)])


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description='ATLAS 선물 봇')
    ap.add_argument('--check', action='store_true', help='설정·계좌 점검만 (주문 없음)')
    ap.add_argument('--once', action='store_true', help='지금 시점으로 일봉 판정 1회')
    ap.add_argument('--resume', action='store_true', help='낙폭 20% 정지 해제')
    a = ap.parse_args(argv)
    _setup_logging()
    st = load_settings()
    kill = Path(cfg.FUT_KILL_SWITCH)
    if kill.exists() and not (a.check or a.resume):
        log.warning(f'킬스위치({kill}) 있음 — 기동 중단')
        return
    ex = FuturesREST(st.base_url, st.api_key, st.api_secret)
    store = Store(st.db_file)
    notify = TelegramNotifier(st.tg_token, st.tg_chat_id, prefix=f'[선물·{st.mode}] ')
    bot = FuturesBot(ex, store, notify, mode=st.mode, observe_equity=st.observe_equity)
    if a.resume:
        store.set_state('dd_halt', '0')
        store.set_state('dd_tier', '0')
        print('낙폭 정지를 해제했습니다. 고점은 현재 자산으로 다시 잡힙니다.')
        if st.trading:
            store.set_state('peak_equity', f'{bot.equity():.8f}')
        return
    ex.sync_time()
    ex.exchange_info()
    if a.check:
        print('\n'.join(check_report(bot) + ([f'API 키: {st.key_source}'] if st.trading else [])))
        return
    bot.startup()
    if a.once:
        rep = bot.on_day_close()
        print(json.dumps({k: v for k, v in rep.items() if k != 'entries'}, ensure_ascii=False, default=str))
        time.sleep(2)
        return
    notify(f'🟢 선물봇 시작 — 모드 {st.mode}, 거래당 위험 {cfg.FUT_RISK_PCT * 100:.1f}%, 종목 {len(cfg.FUT_UNIVERSE)}개'
           + (f', API 키 {st.key_source}' if st.trading else ''))
    bot.run_forever(kill)


if __name__ == '__main__':  # pragma: no cover
    main()

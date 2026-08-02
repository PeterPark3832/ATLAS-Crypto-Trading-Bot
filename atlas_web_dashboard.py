"""
ATLAS Web Dashboard — FastAPI backend
uvicorn atlas_web_dashboard:app --host 0.0.0.0 --port 8080
"""
import io, os, time, json, secrets, shlex, sqlite3, subprocess, logging, hashlib, hmac, threading
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

# 전략/레짐 표시 메타데이터는 config를 단일 출처로 사용 (수동 복사 금지 — drift 방지)
from atlas_spot_config import (
    REGIME_STRATEGY_MAP as _REGIME_STRAT_MAP,
    STRATEGY_NAMES_KR   as _STRAT_FULL,
    STRATEGY_CONDITIONS as _STRAT_CONDITION,
    REGIME_DESCRIPTIONS as _REGIME_DESC,
)

load_dotenv(Path(__file__).parent / '.env')

DB_FILE         = Path(__file__).parent / 'state' / 'atlas_spot.db'
LOG_FILE        = Path(__file__).parent / 'logs'  / 'spot_main.log'
LOG_DIR         = Path(__file__).parent / 'logs'
BOT_DIR         = Path(__file__).parent
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
DASH_PASSWORD   = os.getenv('DASH_PASSWORD') or secrets.token_urlsafe(16)
REFRESH_SEC     = int(os.getenv('DASH_REFRESH_SEC', '60'))
# 봇 기동 인자 보존: 대시보드 start가 원래 운용 구성(--dry-run, --strategies 등)
# 그대로 재기동하도록 .env에 명시 (예: BOT_START_ARGS="--strategies S3,S5,S6")
# 미설정 시 인자 없이(라이브 모드 + 기본 전략) 기동되므로 반드시 확인할 것.
BOT_START_ARGS  = os.getenv('BOT_START_ARGS', '')
API_KEY         = os.getenv('BINANCE_API_KEY', '')
API_SECRET      = os.getenv('BINANCE_API_SECRET', '')
TG_TOKEN        = os.getenv('TG_TOKEN', '')
TG_CHAT_ID      = os.getenv('TG_CHAT_ID', '')
KILL_SWITCH     = Path('/tmp/ATLAS_SPOT_STOP')
HTML_PATH       = BOT_DIR / 'dashboard_ui.html'

STRAT_LABEL = {'S3': 'EMA Trend (4H)', 'S5': 'BB Bounce (1D)', 'S6': 'Donchian (1D)'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'dashboard_errors.log'), encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

if not os.getenv('DASH_PASSWORD'):
    log.warning('DASH_PASSWORD 미설정 — 무작위 비밀번호가 생성되어 사실상 로그인 불가 '
                '상태입니다. .env에 DASH_PASSWORD를 설정하세요.')

app = FastAPI(docs_url=None, redoc_url=None)

# ── 인증 ──────────────────────────────────────────────────────────
_tokens: dict[str, float] = {}

# 로그인 브루트포스 방어: IP당 실패 기록 (윈도 내 초과 시 429)
_login_failures: dict[str, list[float]] = {}
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SEC   = 900   # 15분

def _new_token() -> str:
    tok = secrets.token_hex(32)
    _tokens[tok] = time.time() + 86400
    return tok

def _check(tok: str) -> bool:
    exp = _tokens.get(tok)
    if not exp or time.time() > exp:
        _tokens.pop(tok, None); return False
    return True

def _auth(token: str):
    if not _check(token):
        raise HTTPException(401, 'Unauthorized')

# ── 가격 배치 조회 ────────────────────────────────────────────────
def _fetch_prices(symbols) -> dict:
    """여러 심볼 현재가를 단일 요청으로 조회 (Binance batch ticker)."""
    syms = [str(s) for s in symbols if s]
    if not syms:
        return {}
    try:
        r = requests.get('https://api.binance.com/api/v3/ticker/price',
                         params={'symbols': json.dumps(syms, separators=(',', ':'))},
                         timeout=6)
        if r.ok:
            return {d['symbol']: float(d['price']) for d in r.json()}
    except Exception as e:
        log.error(f'가격 배치 조회 실패: {e}')
    return {}


def _try_login(password: str, client_ip: str) -> str:
    """패스워드 검증 후 토큰 발급. 브루트포스 차단(429)/실패(401) 시 예외."""
    now = time.time()
    fails = [t for t in _login_failures.get(client_ip, []) if now - t < LOGIN_WINDOW_SEC]
    if len(fails) >= LOGIN_MAX_FAILURES:
        _login_failures[client_ip] = fails
        log.warning(f'로그인 차단(브루트포스 의심): {client_ip}')
        raise HTTPException(429, 'Too many attempts — try again later')
    if hmac.compare_digest(str(password or ''), str(DASH_PASSWORD)):
        _login_failures.pop(client_ip, None)
        return _new_token()
    fails.append(now)
    _login_failures[client_ip] = fails
    raise HTTPException(401, 'Invalid password')

# ── 총 자산 (USDT + 오픈 포지션 시가) ───────────────────────────────
_bal_cache: dict = {'val': None, 'ts': 0.0}

def _spot_balance(pos_df: pd.DataFrame | None = None) -> float | None:
    """USDT 잔고 + 오픈 포지션 현재가 합산 = 총 자산 (60s TTL).
    pos_df를 넘기면 DB 재조회를 생략한다."""
    now = time.time()
    if _bal_cache['val'] is not None and now - _bal_cache['ts'] < 60:
        return _bal_cache['val']
    if not API_KEY or not API_SECRET:
        return None
    try:
        ts  = int(time.time() * 1000)
        qs  = f'timestamp={ts}'
        sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
        r   = requests.get(
            'https://api.binance.com/api/v3/account',
            params={'timestamp': ts, 'signature': sig},
            headers={'X-MBX-APIKEY': API_KEY}, timeout=8)
        if not r.ok:
            raise ValueError(f'HTTP {r.status_code}')
        usdt = 0.0
        for b in r.json().get('balances', []):
            if b['asset'] == 'USDT':
                usdt = float(b['free']) + float(b['locked'])
                break

        # 오픈 포지션 시가 합산 (가격 1회 배치 조회)
        total = usdt
        if pos_df is None:
            pos_df = _positions()
        if not pos_df.empty:
            open_pos = [(str(p['symbol']), float(p['qty']), float(p['risk_usd']))
                        for _, p in pos_df.iterrows() if float(p['qty']) > 0]
            prices = _fetch_prices([s for s, _, _ in open_pos])
            for sym, qty, risk in open_pos:
                pr = prices.get(sym)
                total += qty * pr if pr else risk

        val = round(total, 2)
        _bal_cache.update({'val': val, 'ts': now})
        return val
    except Exception as e:
        log.error(f'total equity 조회 실패: {e}')
    return _bal_cache['val']

# ── Telegram ──────────────────────────────────────────────────────
def _tg(msg: str):
    if not TG_TOKEN or not TG_CHAT_ID: return
    try:
        requests.post(f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
                      data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=8)
    except Exception as e:
        log.error(f'TG 전송 실패: {e}')

# ── DB 쿼리 ──────────────────────────────────────────────────────
def _q(sql: str, params=()) -> pd.DataFrame:
    if not DB_FILE.exists(): return pd.DataFrame()
    try:
        con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, timeout=10)
        df  = pd.read_sql_query(sql, con, params=params)
        con.close()
        return df
    except Exception as e:
        log.error(f'DB 쿼리 실패: {e}')
        return pd.DataFrame()

_col_cache: dict = {}


def _opt_col(table: str, col: str, default: str = '0') -> str:
    """컬럼이 있으면 COALESCE(col,0), 없으면 상수를 반환.

    대시보드는 DB를 읽기 전용으로 열기 때문에 마이그레이션을 할 수 없다.
    봇이 아직 새 컬럼을 추가하지 않은 상태(대시보드 먼저 기동 등)에서도
    쿼리가 깨지지 않아야 한다.
    """
    key = (table, col)
    if key not in _col_cache:
        try:
            df = _q(f'PRAGMA table_info({table})')
            _col_cache[key] = col in set(df['name']) if not df.empty else False
        except Exception:
            _col_cache[key] = False
    return f'COALESCE({col}, {default})' if _col_cache[key] else default


def _trades(days: int = 9999) -> pd.DataFrame:
    w = '' if days >= 9999 else f"WHERE exit_ts >= datetime('now','-{days} days')"
    df = _q(f"""SELECT strategy, symbol,
                       entry_price, exit_price, qty_tokens AS qty,
                       cost_usdt,
                       pnl_usdt   AS pnl_usd,
                       pnl_pct, pnl_r,
                       hold_hours, reason, regime,
                       entry_ts, exit_ts,
                       COALESCE(fee_usdt, 0) AS fee_usd,
                       {_opt_col('spot_trades', 'slip_pct')} AS slip_pct
                FROM spot_trades {w} ORDER BY exit_ts ASC""")
    if not df.empty:
        df['direction'] = 'LONG'
        df['leverage']  = 1
        df['exit_ts']   = pd.to_datetime(df['exit_ts'],  utc=True, errors='coerce')
        df['entry_ts']  = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
        df['net_pnl']   = df['pnl_usd'] - df['fee_usd']
        # 슬리피지 비용(USD) = 왕복 슬리피지율 × 명목가.
        # 수수료와 달리 net_pnl에는 이미 체결가로 반영돼 있으므로 중복 차감하지
        # 않고, "실행 비용이 얼마였는지" 보여주기 위한 표시용으로만 집계한다.
        df['slip_usd'] = df['slip_pct'] * df['cost_usdt']
    return df

def _positions() -> pd.DataFrame:
    df = _q("""SELECT strategy, symbol,
                      entry_price, sl, tp,
                      qty_tokens AS qty,
                      cost_usdt  AS risk_usd,
                      risk_pct, exit_type, max_hold_bars,
                      bars_held, peak_price, entry_ts, regime
               FROM spot_positions ORDER BY entry_ts DESC""")
    if not df.empty:
        df['direction'] = 'LONG'
        df['leverage']  = 1
    return df

def _metrics(df: pd.DataFrame) -> dict:
    empty = dict(total=0, wins=0, wr=0, total_pnl=0, gross_pnl=0, total_fee=0,
                 pf=0, avg_r=0, mdd_pct=0, mdd_usd=0, sharpe=0, equity=INITIAL_CAPITAL)
    if df.empty: return empty
    net   = df['net_pnl'] if 'net_pnl' in df.columns else df['pnl_usd']
    total = len(df); wins = int((net > 0).sum())
    pnl   = float(net.sum())
    gross = float(df['pnl_usd'].sum())
    fee   = float(df['fee_usd'].sum()) if 'fee_usd' in df.columns else 0
    gw    = float(net[net > 0].sum())
    gl    = abs(float(net[net < 0].sum()))
    pf    = round(gw / gl, 2) if gl > 0 else 0
    cum   = net.cumsum().values
    dd    = np.maximum.accumulate(cum) - cum
    mdd_u = float(dd.max()); mdd_p = mdd_u / INITIAL_CAPITAL * 100
    sharpe = 0.0
    if not df['exit_ts'].isna().all():
        d = df.set_index('exit_ts')['net_pnl'].resample('1D').sum().fillna(0)
        if len(d) > 1 and d.std() > 0:
            sharpe = round(d.mean() / d.std() * (365 ** 0.5), 2)
    # 실행 비용 가시화: 수수료 + 슬리피지가 총수익(gross)의 몇 %를 잠식했는가.
    # 이 비율이 높으면 전략이 아니라 **실행**이 수익을 갉아먹고 있다는 뜻이다.
    slip = float(df['slip_usd'].sum()) if 'slip_usd' in df.columns else 0.0
    cost_total = fee + max(slip, 0.0)
    gross_win = float(df.loc[df['pnl_usd'] > 0, 'pnl_usd'].sum())
    cost_drag = round(cost_total / gross_win * 100, 1) if gross_win > 0 else 0.0
    return dict(total=total, wins=wins, wr=round(wins / total * 100, 1),
                total_pnl=round(pnl, 2), gross_pnl=round(gross, 2), total_fee=round(fee, 2),
                total_slip=round(slip, 2), cost_drag_pct=cost_drag,
                pf=pf, avg_r=round(float(df['pnl_r'].mean()), 3),
                mdd_pct=round(mdd_p, 1), mdd_usd=round(mdd_u, 2),
                sharpe=sharpe, equity=round(INITIAL_CAPITAL + pnl, 2))

def _ratchet_scale(mdd_pct: float) -> dict:
    if mdd_pct >= 8.0:   scale, label = 0.4, '낙폭 8%+ → 리스크 40%'
    elif mdd_pct >= 5.0: scale, label = 0.7, '낙폭 5%+ → 리스크 70%'
    else:                scale, label = 1.0, '정상'
    return {'scale': scale, 'label': label, 'mdd': round(mdd_pct, 1)}

def _kelly_stats(df: pd.DataFrame) -> list:
    if df.empty: return []
    rows = []
    for strat, g in df.groupby('strategy'):
        t = len(g); w = int((g['pnl_usd'] > 0).sum()); wr = w / t if t > 0 else 0
        avg_win  = float(g[g['pnl_r'] > 0]['pnl_r'].mean()) if (g['pnl_r'] > 0).any() else 1.0
        avg_loss = abs(float(g[g['pnl_r'] <= 0]['pnl_r'].mean())) if (g['pnl_r'] <= 0).any() else 1.0
        rr = avg_win / avg_loss if avg_loss > 0 else 2.0
        full_k = max(0.0, wr - (1 - wr) / rr) if rr > 0 else 0.0
        rows.append({'strategy': STRAT_LABEL.get(strat, strat), 'trades': t,
                     'wr': round(wr * 100, 1), 'rr': round(rr, 2),
                     'full_k': round(full_k * 100, 1), 'half_k': round(full_k * 50, 1),
                     'reliable': t >= 20})
    return sorted(rows, key=lambda x: x['trades'], reverse=True)

# 곡선은 차트 폭(~600px)보다 훨씬 많은 점을 보내봐야 화면에 표현되지 않는다.
# 5,000건이면 곡선 3종이 응답의 98%(1MB)를 차지하고 Chart.js 렌더도 그만큼 느려진다.
MAX_CURVE_POINTS = 400


def _downsample(rows: list, key: str, max_points: int = MAX_CURVE_POINTS) -> list:
    """구간별 최소·최대를 남기는 다운샘플 — 피크와 저점(최대낙폭 등)이
    깎이지 않도록 단순 n번째 추출 대신 극값을 보존한다."""
    n = len(rows)
    if n <= max_points:
        return rows
    buckets = max(1, max_points // 2)
    size = n / buckets
    keep = {0, n - 1}
    for b in range(buckets):
        lo, hi = int(b * size), min(int((b + 1) * size), n)
        if lo >= hi:
            continue
        vals = [rows[i].get(key) or 0 for i in range(lo, hi)]
        keep.add(lo + vals.index(min(vals)))
        keep.add(lo + vals.index(max(vals)))
    return [rows[i] for i in sorted(keep)]


def _iso_list(series: pd.Series) -> list:
    """타임스탬프 열 → ISO 문자열 리스트 (NaT는 None)."""
    return [t.isoformat() if pd.notna(t) else None for t in series]


def _rolling_wr(df: pd.DataFrame, n: int = 10) -> list:
    if df.empty or len(df) < 2: return []
    d = df.sort_values('exit_ts')
    wins = (d['pnl_usd'] > 0).astype(float)
    wr   = (wins.rolling(n, min_periods=1).mean() * 100).round(1).tolist()
    cnt  = np.minimum(np.arange(1, len(d) + 1), n).tolist()
    ts   = _iso_list(d['exit_ts'])
    return [{'ts': ts[i], 'wr': wr[i], 'n': cnt[i]} for i in range(len(wr))]

def _dd_curve(df: pd.DataFrame) -> list:
    if df.empty: return []
    d   = df.sort_values('exit_ts')
    cum = d['pnl_usd'].cumsum().values
    dd  = np.round((np.maximum.accumulate(cum) - cum) / INITIAL_CAPITAL * 100, 2)
    ts  = _iso_list(d['exit_ts'])
    return [{'ts': ts[i], 'dd': float(dd[i])} for i in range(len(dd))]

def _monthly_pnl(df: pd.DataFrame) -> list:
    if df.empty or df['exit_ts'].isna().all(): return []
    d = df.copy(); d['month'] = d['exit_ts'].dt.strftime('%Y-%m')
    rows = []
    for month, g in d.groupby('month'):
        t = len(g); w = int((g['pnl_usd'] > 0).sum())
        rows.append({'month': month, 'pnl': round(float(g['pnl_usd'].sum()), 2),
                     'trades': t, 'wr': round(w / t * 100, 0)})
    return sorted(rows, key=lambda x: x['month'])

def _sym_stats(df: pd.DataFrame) -> list:
    if df.empty: return []
    rows = []
    for sym, g in df.groupby('symbol'):
        t = len(g); w = int((g['pnl_usd'] > 0).sum())
        rows.append({'symbol': sym.replace('USDT', ''), 'direction': 'LONG',
                     'trades': t, 'wr': round(w / t * 100, 0),
                     'pnl': round(float(g['pnl_usd'].sum()), 2),
                     'avg_r': round(float(g['pnl_r'].mean()), 3),
                     'best': round(float(g['pnl_usd'].max()), 2),
                     'worst': round(float(g['pnl_usd'].min()), 2)})
    return sorted(rows, key=lambda x: abs(x['pnl']), reverse=True)

def _bot_alive() -> bool:
    try:
        r = subprocess.run(['pgrep', '-f', 'atlas_spot_main.py'], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception as e:
        log.error(f'bot_alive 실패: {e}'); return False

_watchdog_state = {'alive': True, 'down_since': None}

def _bot_watchdog_loop():
    """대시보드 프로세스와 독립적으로 60초마다 봇 생존 확인 — 다운 감지 시 텔레그램 알림."""
    while True:
        try:
            alive = _bot_alive()
            now = time.time()
            if not alive and _watchdog_state['alive']:
                _watchdog_state['alive'] = False
                _watchdog_state['down_since'] = now
                _tg('🔴 [감시] ATLAS Spot 봇 프로세스 중단 감지 — 즉시 확인이 필요합니다.')
                log.warning('[감시] 봇 프로세스 중단 감지')
            elif alive and not _watchdog_state['alive']:
                down_min = (now - _watchdog_state['down_since']) / 60 if _watchdog_state['down_since'] else 0
                _watchdog_state['alive'] = True
                _watchdog_state['down_since'] = None
                _tg(f'✅ [감시] 봇 프로세스 복구 확인 (중단 {down_min:.0f}분)')
                log.info('[감시] 봇 프로세스 복구 확인')
        except Exception as e:
            log.error(f'[감시루프] 오류: {e}')
        time.sleep(60)

threading.Thread(target=_bot_watchdog_loop, name='BotWatchdog', daemon=True).start()


def _last_log_time() -> str:
    try:
        if not LOG_FILE.exists(): return None
        diff = (datetime.now(timezone.utc) - datetime.fromtimestamp(
            LOG_FILE.stat().st_mtime, tz=timezone.utc)).total_seconds()
        if diff < 120:  return f'{int(diff)}초 전'
        if diff < 3600: return f'{int(diff / 60)}분 전'
        return f'{int(diff / 3600)}시간 전'
    except Exception as e:
        log.error(f'last_log_time 실패: {e}'); return None

# 로그 tail 공유 캐시 — 봇 로그는 24시간 계속 자라므로(수십~수백 MB)
# 요청마다 전체를 읽으면 파싱 비용이 응답시간을 지배한다. 끝부분만 읽고
# 두 파서가 같은 1회 읽기를 공유한다. (mtime+size가 같으면 재사용)
_LOG_TAIL_BYTES = 512 * 1024
# 레짐 지표 블록(ADX/BTC/EMA200)은 regime_loop가 1시간에 한 번만 찍는다.
# 로그가 많은 날엔 512KB 안에 안 들어올 수 있으므로, 못 찾으면 단계적으로
# 더 읽는다(흔한 경우는 여전히 512KB로 빠르게 끝난다).
_LOG_TAIL_MAX = 8 * 1024 * 1024
_log_tail_cache: dict = {'key': None, 'text': ''}


def _log_tail(min_bytes: int = _LOG_TAIL_BYTES) -> str:
    try:
        if not LOG_FILE.exists():
            return ''
        st = LOG_FILE.stat()
        key = (st.st_mtime, st.st_size, min_bytes)
        if _log_tail_cache['key'] == key:
            return _log_tail_cache['text']
        want = min(min_bytes, _LOG_TAIL_MAX)
        with LOG_FILE.open('rb') as f:
            if st.st_size > want:
                f.seek(-want, os.SEEK_END)
            data = f.read()
        text = data.decode('utf-8', errors='ignore')
        _log_tail_cache.update({'key': key, 'text': text})
        return text
    except Exception as e:
        log.error(f'log_tail 실패: {e}')
        return _log_tail_cache['text']


def _regime_from_log() -> str:
    try:
        if not LOG_FILE.exists(): return None
        lines = _log_tail().splitlines()[-300:]
        for line in reversed(lines):
            for regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'WEAK_TREND', 'CRISIS']:
                if regime in line:
                    return regime
    except Exception as e:
        log.error(f'regime_from_log 실패: {e}')
    return None


def _parse_regime_metrics() -> dict:
    """최근 RegimeLoop 로그 블록에서 ADX/BTC/EMA200/ATR% 파싱."""
    out = {'adx_1d': None, 'adx_4h': None, 'atr_pct': None,
           'btc': None, 'ema200': None}
    try:
        if not LOG_FILE.exists():
            return out
        import re
        rx_adx = re.compile(r'ADX\(1D\)=([\d.]+).*?ADX\(4H\)=([\d.]+).*?ATR%=([\d.]+)')
        rx_btc = re.compile(r'BTC=([\d,]+)\s+EMA200=([\d,]+)')
        # 지표 블록은 1시간에 한 번만 찍히므로 기본 tail에 없을 수 있다.
        # 못 찾으면 읽는 범위를 넓혀 재시도한다(대부분 첫 시도에서 끝난다).
        m_adx = m_btc = []
        for want in (_LOG_TAIL_BYTES, _LOG_TAIL_BYTES * 8, _LOG_TAIL_MAX):
            text = _log_tail(want)
            m_adx = list(rx_adx.finditer(text))
            m_btc = list(rx_btc.finditer(text))
            if m_adx and m_btc:
                break
            if len(text) < want:      # 파일 전체를 이미 읽음 — 더 읽어도 소용없다
                break
        if m_adx:
            g = m_adx[-1]
            out['adx_1d']  = float(g.group(1))
            out['adx_4h']  = float(g.group(2))
            out['atr_pct'] = float(g.group(3))
        if m_btc:
            g = m_btc[-1]
            out['btc']    = float(g.group(1).replace(',', ''))
            out['ema200'] = float(g.group(2).replace(',', ''))
    except Exception as e:
        log.error(f'parse_regime_metrics 실패: {e}')
    return out


def _regime_briefing(regime: str, bot_alive: bool, open_pos_cnt: int) -> dict:
    """현재 봇 상태를 사람이 읽기 쉬운 브리핑으로 생성."""
    if not regime or regime not in _REGIME_DESC:
        return {'headline': '레짐 판별 대기 중',
                'detail': '봇이 BTC 레짐을 아직 계산하지 않았습니다.',
                'reason': '', 'active_strats': [], 'conditions': [], 'metrics': {}, 'tone': 'neutral'}

    kor, character = _REGIME_DESC[regime]
    strats = _REGIME_STRAT_MAP.get(regime, [])
    strat_names = [f'{s}({_STRAT_FULL.get(s, s)})' for s in strats]
    conditions = [{'strategy': s, 'name': _STRAT_FULL.get(s, s),
                   'condition': _STRAT_CONDITION.get(s, '')} for s in strats]
    mx = _parse_regime_metrics()

    # EMA200 대비 BTC 위치(%)
    ema_gap = None
    if mx['btc'] and mx['ema200']:
        ema_gap = (mx['btc'] - mx['ema200']) / mx['ema200'] * 100

    # "왜 지금 진입/대기인가" 판단
    tone = 'neutral'
    if not bot_alive:
        reason = '⚠️ 봇 프로세스가 중단된 상태 — 신규 진입 불가. 즉시 점검 필요.'
        tone = 'danger'
    elif open_pos_cnt > 0:
        reason = f'현재 {open_pos_cnt}개 포지션 보유 중. 활성 전략의 진입 신호를 계속 감시합니다.'
        tone = 'active'
    elif regime == 'CRISIS':
        atr_s = f"{mx['atr_pct']:.1f}%" if mx['atr_pct'] else '높음'
        reason = f'변동성 위기(ATR {atr_s})로 모든 전략 진입이 차단된 상태입니다. 시장 안정 시 자동 해제됩니다.'
        tone = 'danger'
    elif not strats:
        reason = '이 레짐에 배정된 전략이 없어 신규 진입이 없습니다.'
        tone = 'warn'
    elif regime == 'TRENDING_DOWN':
        reason = ('하락 추세 — 과매도 반등 전략 S4(RSI 과매도)만 보수적으로 운용합니다. '
                  'RSI가 과매도 구간에 진입한 코인을 감시하되, 강한 하락에서는 신규 진입을 '
                  '자제합니다. (반등 전략 S5는 하락장 실전 성과가 나빠 제외됨)')
        tone = 'warn'
    else:
        reason = (f'활성 전략({", ".join(strats)})이 진입 조건을 만족하는 코인을 감시 중입니다. '
                  f'아직 신호가 없어 현금 보유 상태입니다.')
        tone = 'active'

    # 헤드라인
    if not bot_alive:
        headline = '봇 중단됨 — 점검 필요'
    elif open_pos_cnt > 0:
        headline = f'{kor} · 포지션 {open_pos_cnt}개 운용 중'
    elif not strats or regime == 'CRISIS':
        headline = f'{kor} · 신규 진입 차단'
    else:
        headline = f'{kor} · 진입 신호 대기 중'

    return {
        'headline': headline,
        'detail': character,
        'reason': reason,
        'active_strats': strat_names,
        'conditions': conditions,
        'regime_kor': kor,
        'ema_gap': round(ema_gap, 1) if ema_gap is not None else None,
        'metrics': mx,
        'tone': tone,
    }


def _alerts(m: dict, bot_alive: bool) -> list:
    alerts = []
    if not bot_alive:
        alerts.append({'level': 'critical', 'msg': '봇 프로세스 중단 감지 — 즉시 확인 필요'})
    if m['mdd_pct'] > 15:
        alerts.append({'level': 'critical', 'msg': f'MDD {m["mdd_pct"]:.1f}% 초과 — 즉시 중단 검토'})
    elif m['mdd_pct'] > 10:
        alerts.append({'level': 'warn', 'msg': f'MDD {m["mdd_pct"]:.1f}% 경고선 초과'})
    if m['total'] >= 10:
        if m['wr'] < 30:
            alerts.append({'level': 'warn', 'msg': f'승률 {m["wr"]:.0f}% 경고선 미달 (기준 30%)'})
        if m['pf'] < 1.0:
            alerts.append({'level': 'warn', 'msg': f'PF {m["pf"]:.2f} — 손실 구간'})
    return alerts

# ══════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════

@app.post('/api/auth')
async def auth(req: Request):
    body = await req.json()
    client_ip = req.client.host if req.client else 'unknown'
    return {'token': _try_login(body.get('password'), client_ip)}

@app.get('/api/status')
def status(token: str):
    _auth(token)
    return {'bot_alive': _bot_alive(), 'last_log': _last_log_time(), 'regime': _regime_from_log()}

@app.post('/api/control')
async def control(req: Request):
    body   = await req.json()
    _auth(body.get('token', ''))
    action = body.get('action', '')
    if action == 'stop':
        KILL_SWITCH.touch()
        _tg('🛑 [스팟 대시보드] 봇 중지 명령')
        return {'ok': True, 'msg': '봇 중지 신호 전송 완료'}
    elif action == 'start':
        if _bot_alive():
            return {'ok': False, 'msg': '봇이 이미 실행 중입니다'}
        if KILL_SWITCH.exists(): KILL_SWITCH.unlink()
        log_path = LOG_DIR / 'spot_main.log'
        # BOT_START_ARGS로 원래 운용 구성 보존 — 인자 없이 띄우면 dry-run으로
        # 돌리던 봇이 실주문 모드로 재기동되는 사고가 날 수 있다.
        cmd = ['python3', '-u', 'atlas_spot_main.py'] + shlex.split(BOT_START_ARGS)
        subprocess.Popen(
            cmd,
            cwd=str(BOT_DIR),
            stdout=open(log_path, 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        args_desc = BOT_START_ARGS or '(기본 인자 — 라이브 모드)'
        _tg(f'▶️ [스팟 대시보드] 봇 시작 명령: {args_desc}')
        return {'ok': True, 'msg': f'봇 시작 완료 ({args_desc})'}
    return {'ok': False, 'msg': f'알 수 없는 명령: {action}'}

@app.post('/api/panic')
async def panic(req: Request):
    body = await req.json()
    _auth(body.get('token', ''))
    if body.get('confirm') != 'PANIC':
        raise HTTPException(400, '확인 코드 오류 (PANIC 입력 필요)')
    results = []
    KILL_SWITCH.touch()
    results.append('✅ 봇 중지 신호 전송')
    if API_KEY and API_SECRET:
        try:
            import ccxt
            ex = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True})
            ex.options['defaultType'] = 'spot'
            ex.load_markets()
            pos_df = _positions()
            sold = 0
            for _, p in pos_df.iterrows():
                try:
                    sym = p['symbol'].replace('USDT', '/USDT')
                    qty = float(p['qty'])
                    if qty > 0:
                        # 열린 스탑 주문이 잔고를 잠그므로 먼저 전부 취소
                        try:
                            ex.cancel_all_orders(sym)
                        except Exception:
                            pass
                        ex.create_order(sym, 'market', 'sell', qty)
                        sold += 1
                        log.warning(f'강제 매도: {sym} {qty}')
                except Exception as e:
                    results.append(f'⚠️ {p["symbol"]} 매도 실패: {str(e)[:40]}')
            results.append(f'✅ 보유 코인 {sold}개 시장가 매도')
        except Exception as e:
            results.append(f'❌ CCXT 오류: {str(e)[:60]}')
    _tg('🚨 [패닉] 스팟 강제 매도\n' + '\n'.join(results))
    return {'ok': True, 'results': results}

@app.get('/api/dashboard')
def dashboard(token: str, period: int = 0):
    _auth(token)
    df  = _trades(period if period > 0 else 9999)
    pos = _positions()
    m   = _metrics(df)
    bot_alive = _bot_alive()
    log_time  = _last_log_time()
    regime = _regime_from_log() or 'UNKNOWN'
    alerts = _alerts(m, bot_alive)
    actual_bal = _spot_balance(pos)

    # 수익 곡선
    eq_curve = []
    if not df.empty:
        d = df.sort_values('exit_ts')
        net_col = 'net_pnl' if 'net_pnl' in d.columns else 'pnl_usd'
        eq     = (INITIAL_CAPITAL + d[net_col].cumsum()).round(2).tolist()
        pnl    = d[net_col].round(2).tolist()
        fee    = (d['fee_usd'].round(2).tolist() if 'fee_usd' in d.columns
                  else [0.0] * len(d))
        reason = d['reason'].tolist()
        ts     = _iso_list(d['exit_ts'])
        eq_curve = [{'ts': ts[i], 'eq': eq[i], 'reason': reason[i],
                     'pnl': pnl[i], 'fee': fee[i]} for i in range(len(eq))]

    # 일별 PnL
    daily = []
    if not df.empty and not df['exit_ts'].isna().all():
        net_col = 'net_pnl' if 'net_pnl' in df.columns else 'pnl_usd'
        for ts, v in df.set_index('exit_ts')[net_col].resample('1D').sum().tail(30).items():
            daily.append({'date': ts.strftime('%m/%d'), 'pnl': round(float(v), 2)})

    # 전략별 모듈 통계
    module_stats = []
    if not df.empty:
        net_col = 'net_pnl' if 'net_pnl' in df.columns else 'pnl_usd'
        for strat, g in df.groupby('strategy'):
            t = len(g); w = int((g[net_col] > 0).sum())
            gw = float(g[g[net_col] > 0][net_col].sum())
            gl = abs(float(g[g[net_col] < 0][net_col].sum()))
            module_stats.append({
                'module':  STRAT_LABEL.get(strat, strat),
                'trades':  t, 'wr': round(w / t * 100, 0),
                'pf':      round(gw / gl, 2) if gl > 0 else 0,
                'pnl':     round(float(g[net_col].sum()), 2),
                'fee':     round(float(g['fee_usd'].sum()), 2) if 'fee_usd' in g.columns else 0,
                'avg_r':   round(float(g['pnl_r'].mean()), 3),
                'reliable': t >= 20,
            })

    # 열린 포지션
    open_pos = []; now_utc = datetime.now(timezone.utc)
    if not pos.empty:
        for _, p in pos.iterrows():
            entry = float(p['entry_price']); sl = float(p['sl'])
            hold_str = '—'
            try:
                et = pd.to_datetime(p['entry_ts'], utc=True, errors='coerce')
                if pd.notna(et):
                    diff = (now_utc - et).total_seconds(); h, mr = divmod(int(diff), 3600)
                    hold_str = f'{h}h {mr // 60}m'
            except Exception: pass
            sl_dist = f'{abs(entry - sl) / entry * 100:.2f}%' if entry > 0 else '—'
            open_pos.append({
                'strategy': STRAT_LABEL.get(str(p['strategy']), str(p['strategy'])),
                'symbol':   p['symbol'], 'direction': 'LONG',
                'entry':    round(entry, 4), 'sl': round(sl, 4),
                'tp':       round(float(p['tp']), 4),
                'qty':      round(float(p['qty']), 6), 'leverage': 1,
                'entry_ts': str(p['entry_ts']), 'hold': hold_str, 'sl_dist': sl_dist,
                'cost_usdt': round(float(p['risk_usd']), 2),
                'regime':   str(p.get('regime', '')),
            })

    # 최근 거래 50건
    recent = []
    if not df.empty:
        for _, r in df.sort_values('exit_ts', ascending=False).head(50).iterrows():
            recent.append({
                'ts':        r['exit_ts'].strftime('%m-%d %H:%M') if pd.notna(r['exit_ts']) else '-',
                'strategy':  STRAT_LABEL.get(str(r['strategy']), str(r['strategy'])),
                'symbol':    r['symbol'], 'direction': 'LONG',
                'entry':     round(float(r['entry_price']), 4),
                'exit':      round(float(r['exit_price']), 4),
                'pnl':       round(float(r['pnl_usd']), 2),
                'r':         round(float(r['pnl_r']), 2),
                'reason':    r['reason'], 'regime': r['regime'] or '',
            })

    # 연속 스트릭
    streak = {'count': 0, 'kind': 'none'}
    if not df.empty:
        s = df.sort_values('exit_ts', ascending=False)
        is_win = float(s.iloc[0]['pnl_usd']) > 0; cnt = 0
        for _, r in s.iterrows():
            if (float(r['pnl_usd']) > 0) == is_win: cnt += 1
            else: break
        streak = {'count': cnt, 'kind': 'win' if is_win else 'loss'}

    briefing = _regime_briefing(regime, bot_alive, len(open_pos))

    return {
        'metrics': m, 'regime': regime, 'streak': streak,
        'briefing': briefing,
        'actual_balance': actual_bal,
        'bot_alive': bot_alive, 'log_time': log_time, 'alerts': alerts,
        'eq_curve': _downsample(eq_curve, 'eq'), 'daily': daily,
        'module_stats': module_stats,
        'sym_dir': _sym_stats(df), 'recent': recent, 'open_pos': open_pos,
        'ratchet': _ratchet_scale(m['mdd_pct']), 'kelly': _kelly_stats(df),
        'rolling_wr': _downsample(_rolling_wr(df, n=10), 'wr'),
        'dd_curve': _downsample(_dd_curve(df), 'dd'),
        'monthly_pnl': _monthly_pnl(df),
        'refresh_sec': REFRESH_SEC,
        'initial_capital': INITIAL_CAPITAL,
        'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    }

@app.get('/api/logs')
def get_logs(token: str, lines: int = 200, filter: str = ''):
    _auth(token)
    try:
        if not LOG_FILE.exists():
            return {'lines': [], 'file': '로그 없음'}
        with open(LOG_FILE, 'rb') as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - lines * 120))
            raw = f.read().decode('utf-8', errors='replace').splitlines()[-lines:]
        if filter:
            kws = [k.strip().lower() for k in filter.split(',') if k.strip()]
            raw = [l for l in raw if any(k in l.lower() for k in kws)]
        return {'lines': raw, 'file': LOG_FILE.name}
    except Exception as e:
        log.error(f'로그 조회 실패: {e}'); return {'lines': [], 'file': '오류'}

@app.get('/api/trades/csv')
def trades_csv(token: str, period: int = 0):
    _auth(token)
    df = _trades(period if period > 0 else 9999)
    if df.empty:
        return StreamingResponse(iter(['no data']), media_type='text/plain')
    _fml = ('=', '+', '-', '@', '\t', '\r', '\n')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda v: ("'" + v) if isinstance(v, str) and v and v[0] in _fml else v)
    buf = io.StringIO(); df.to_csv(buf, index=False, encoding='utf-8-sig'); buf.seek(0)
    fname = f'spot_trades_{datetime.now().strftime("%Y%m%d")}.csv'
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv; charset=utf-8-sig',
                             headers={'Content-Disposition': f'attachment; filename={fname}'})

@app.get('/api/prices')
def prices(token: str, symbols: str = ''):
    _auth(token)
    return _fetch_prices(symbols.split(',') if symbols else [])

@app.get('/', response_class=HTMLResponse)
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding='utf-8'))
    return HTMLResponse('<h1>dashboard_ui.html not found</h1>', 500)

"""
ATLAS Spot Web Dashboard — FastAPI backend
uvicorn atlas_spot_web_dashboard:app --host 0.0.0.0 --port 8080
"""
import io, os, time, secrets, sqlite3, subprocess, logging, hashlib, hmac
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv(Path(__file__).parent / '.env')

DB_FILE         = Path(__file__).parent / 'state' / 'atlas_spot.db'
LOG_FILE        = Path(__file__).parent / 'logs'  / 'spot_main.log'
LOG_DIR         = Path(__file__).parent / 'logs'
BOT_DIR         = Path(__file__).parent
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
DASH_PASSWORD   = os.getenv('DASH_PASSWORD', 'atlas2026')
REFRESH_SEC     = int(os.getenv('DASH_REFRESH_SEC', '60'))
API_KEY         = os.getenv('BINANCE_API_KEY', '')
API_SECRET      = os.getenv('BINANCE_API_SECRET', '')
TG_TOKEN        = os.getenv('TG_TOKEN', '')
TG_CHAT_ID      = os.getenv('TG_CHAT_ID', '')
KILL_SWITCH     = Path('/tmp/ATLAS_SPOT_STOP')
HTML_PATH       = BOT_DIR / 'dashboard_spot_ui.html'

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
app = FastAPI(docs_url=None, redoc_url=None)

# ── 인증 ──────────────────────────────────────────────────────────
_tokens: dict[str, float] = {}

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

# ── Spot 잔고 (Binance REST, 5분 TTL) ────────────────────────────
_bal_cache: dict = {'val': None, 'ts': 0.0}

def _spot_balance() -> float | None:
    now = time.time()
    if _bal_cache['val'] is not None and now - _bal_cache['ts'] < 300:
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
        for b in r.json().get('balances', []):
            if b['asset'] == 'USDT':
                val = float(b['free']) + float(b['locked'])
                _bal_cache.update({'val': val, 'ts': now})
                return val
    except Exception as e:
        log.error(f'spot balance 조회 실패: {e}')
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

def _trades(days: int = 9999) -> pd.DataFrame:
    w = '' if days >= 9999 else f"WHERE exit_ts >= datetime('now','-{days} days')"
    df = _q(f"""SELECT strategy, symbol,
                       entry_price, exit_price, qty_tokens AS qty,
                       cost_usdt,
                       pnl_usdt   AS pnl_usd,
                       pnl_pct, pnl_r,
                       hold_hours, reason, regime,
                       entry_ts, exit_ts,
                       COALESCE(fee_usdt, 0) AS fee_usd
                FROM spot_trades {w} ORDER BY exit_ts ASC""")
    if not df.empty:
        df['direction'] = 'LONG'
        df['leverage']  = 1
        df['exit_ts']   = pd.to_datetime(df['exit_ts'],  utc=True, errors='coerce')
        df['entry_ts']  = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
        df['net_pnl']   = df['pnl_usd'] - df['fee_usd']
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
    return dict(total=total, wins=wins, wr=round(wins / total * 100, 1),
                total_pnl=round(pnl, 2), gross_pnl=round(gross, 2), total_fee=round(fee, 2),
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

def _rolling_wr(df: pd.DataFrame, n: int = 10) -> list:
    if df.empty or len(df) < 2: return []
    df = df.sort_values('exit_ts')
    wins = (df['pnl_usd'] > 0).astype(int).values
    result = []
    for i in range(len(wins)):
        start = max(0, i - n + 1); window = wins[start:i + 1]
        ts_val = df.iloc[i]['exit_ts']
        result.append({'ts': ts_val.isoformat() if pd.notna(ts_val) else None,
                       'wr': round(float(window.mean()) * 100, 1), 'n': len(window)})
    return result

def _dd_curve(df: pd.DataFrame) -> list:
    if df.empty: return []
    df = df.sort_values('exit_ts').reset_index(drop=True)
    cum = df['pnl_usd'].cumsum().values
    dd  = (np.maximum.accumulate(cum) - cum) / INITIAL_CAPITAL * 100
    result = []
    for i, row in df.iterrows():
        ts_val = row['exit_ts']
        result.append({'ts': ts_val.isoformat() if pd.notna(ts_val) else None,
                       'dd': round(float(dd[i]), 2)})
    return result

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

def _regime_from_log() -> str:
    try:
        if not LOG_FILE.exists(): return None
        lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()[-300:]
        for line in reversed(lines):
            for regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'WEAK_TREND', 'CRISIS']:
                if regime in line:
                    return regime
    except Exception as e:
        log.error(f'regime_from_log 실패: {e}')
    return None

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
    if body.get('password') == DASH_PASSWORD:
        return {'token': _new_token()}
    raise HTTPException(401, 'Invalid password')

@app.get('/api/status')
async def status(token: str):
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
        subprocess.Popen(
            ['python3', '-u', 'atlas_spot_main.py'],
            cwd=str(BOT_DIR),
            stdout=open(log_path, 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _tg('▶️ [스팟 대시보드] 봇 시작 명령')
        return {'ok': True, 'msg': '봇 시작 완료'}
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
async def dashboard(token: str, period: int = 0):
    _auth(token)
    df  = _trades(period if period > 0 else 9999)
    pos = _positions()
    m   = _metrics(df)
    bot_alive = _bot_alive()
    log_time  = _last_log_time()
    regime = _regime_from_log() or 'UNKNOWN'
    alerts = _alerts(m, bot_alive)
    actual_bal = _spot_balance()

    # 수익 곡선
    eq_curve = []
    if not df.empty:
        d = df.sort_values('exit_ts').copy()
        net_col = 'net_pnl' if 'net_pnl' in d.columns else 'pnl_usd'
        d['eq'] = INITIAL_CAPITAL + d[net_col].cumsum()
        for _, r in d.iterrows():
            eq_curve.append({'ts': r['exit_ts'].isoformat() if pd.notna(r['exit_ts']) else None,
                             'eq': round(float(r['eq']), 2), 'reason': r['reason'],
                             'pnl': round(float(r[net_col]), 2),
                             'fee': round(float(r.get('fee_usd', 0)), 2)})

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

    return {
        'metrics': m, 'regime': regime, 'streak': streak,
        'actual_balance': actual_bal,
        'bot_alive': bot_alive, 'log_time': log_time, 'alerts': alerts,
        'eq_curve': eq_curve, 'daily': daily, 'module_stats': module_stats,
        'sym_dir': _sym_stats(df), 'recent': recent, 'open_pos': open_pos,
        'ratchet': _ratchet_scale(m['mdd_pct']), 'kelly': _kelly_stats(df),
        'rolling_wr': _rolling_wr(df, n=10), 'dd_curve': _dd_curve(df),
        'monthly_pnl': _monthly_pnl(df),
        'refresh_sec': REFRESH_SEC,
        'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    }

@app.get('/api/logs')
async def get_logs(token: str, lines: int = 200, filter: str = ''):
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
async def trades_csv(token: str, period: int = 0):
    _auth(token)
    df = _trades(period if period > 0 else 9999)
    if df.empty:
        return StreamingResponse(iter(['no data']), media_type='text/plain')
    buf = io.StringIO(); df.to_csv(buf, index=False, encoding='utf-8-sig'); buf.seek(0)
    fname = f'spot_trades_{datetime.now().strftime("%Y%m%d")}.csv'
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv; charset=utf-8-sig',
                             headers={'Content-Disposition': f'attachment; filename={fname}'})

@app.get('/api/prices')
async def prices(token: str, symbols: str = ''):
    _auth(token)
    result = {}
    for sym in (symbols.split(',') if symbols else []):
        try:
            r = requests.get('https://api.binance.com/api/v3/ticker/price',
                             params={'symbol': sym}, timeout=5)
            if r.ok: result[sym] = float(r.json().get('price', 0))
        except Exception as e:
            log.error(f'가격 조회 실패 {sym}: {e}')
    return result

@app.get('/', response_class=HTMLResponse)
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding='utf-8'))
    return HTMLResponse('<h1>dashboard_spot_ui.html not found</h1>', 500)

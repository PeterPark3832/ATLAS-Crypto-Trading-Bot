"""
ATLAS v2 Web Dashboard — FastAPI 백엔드  (Stage 1)
uvicorn atlas_web_dashboard:app --host 0.0.0.0 --port 8080
"""
import os, time, secrets, sqlite3, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

load_dotenv(Path(__file__).parent / '.env')

DB_FILE         = Path(os.getenv('DB_FILE', str(Path(__file__).parent / 'state' / 'atlas_v2.db')))
LOG_DIR         = Path(os.getenv('REMOTE_LOG_DIR', str(Path(__file__).parent / 'logs')))
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
DASH_PASSWORD   = os.getenv('DASH_PASSWORD', 'atlas2026')
REFRESH_SEC     = int(os.getenv('DASH_REFRESH_SEC', '60'))
HTML_PATH       = Path(__file__).parent / 'dashboard_ui.html'

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
        _tokens.pop(tok, None)
        return False
    return True

def _auth(token: str):
    if not _check(token):
        raise HTTPException(401, 'Unauthorized')

# ── DB ───────────────────────────────────────────────────────────
def _q(sql: str, params=()) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, timeout=10)
        df  = pd.read_sql_query(sql, con, params=params)
        con.close()
        return df
    except Exception:
        return pd.DataFrame()

def _trades(days: int = 9999) -> pd.DataFrame:
    w = '' if days >= 9999 else f"WHERE exit_ts >= datetime('now','-{days} days')"
    df = _q(f"""SELECT strategy,symbol,direction,entry_price,exit_price,
                       qty,leverage,pnl_usd,pnl_r,reason,regime,entry_ts,exit_ts
                FROM trades {w} ORDER BY exit_ts ASC""")
    if not df.empty:
        df['exit_ts']  = pd.to_datetime(df['exit_ts'],  utc=True, errors='coerce')
        df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
    return df

def _positions() -> pd.DataFrame:
    return _q("""SELECT strategy,symbol,direction,entry_price,sl,tp,
                        qty,leverage,risk_usd,rr,bep_done,entry_ts
                 FROM positions ORDER BY entry_ts DESC""")

def _metrics(df: pd.DataFrame) -> dict:
    empty = dict(total=0,wins=0,wr=0,total_pnl=0,pf=0,avg_r=0,
                 mdd_pct=0,mdd_usd=0,sharpe=0,equity=INITIAL_CAPITAL)
    if df.empty:
        return empty
    total = len(df); wins = int((df['pnl_usd']>0).sum())
    pnl   = float(df['pnl_usd'].sum())
    gw    = float(df[df['pnl_usd']>0]['pnl_usd'].sum())
    gl    = abs(float(df[df['pnl_usd']<0]['pnl_usd'].sum()))
    pf    = round(gw/gl, 2) if gl>0 else 0
    cum   = df['pnl_usd'].cumsum().values
    dd    = np.maximum.accumulate(cum) - cum
    mdd_u = float(dd.max()); mdd_p = mdd_u/INITIAL_CAPITAL*100
    sharpe = 0.0
    if not df['exit_ts'].isna().all():
        d = df.set_index('exit_ts')['pnl_usd'].resample('1D').sum().fillna(0)
        if len(d)>1 and d.std()>0:
            sharpe = round(d.mean()/d.std()*(365**0.5), 2)
    return dict(total=total,wins=wins,wr=round(wins/total*100,1),total_pnl=round(pnl,2),
                pf=pf,avg_r=round(float(df['pnl_r'].mean()),3),
                mdd_pct=round(mdd_p,1),mdd_usd=round(mdd_u,2),
                sharpe=sharpe,equity=round(INITIAL_CAPITAL+pnl,2))

# ── 봇 상태 ──────────────────────────────────────────────────────
def _bot_alive() -> bool:
    try:
        r = subprocess.run(['pgrep','-f','atlas_v2_main.py'],
                           capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False

def _last_log_time() -> str:
    try:
        logs = sorted(LOG_DIR.glob('atlas_v2_*.log'),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if not logs:
            return None
        mtime = logs[0].stat().st_mtime
        dt    = datetime.fromtimestamp(mtime, tz=timezone.utc)
        diff  = (datetime.now(timezone.utc) - dt).total_seconds()
        if diff < 120:   return f'{int(diff)}초 전'
        if diff < 3600:  return f'{int(diff/60)}분 전'
        return f'{int(diff/3600)}시간 전'
    except Exception:
        return None

def _regime_from_log() -> str:
    """로그 파일 마지막 200줄에서 최신 레짐 파싱."""
    try:
        logs = sorted(LOG_DIR.glob('atlas_v2_*.log'),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if not logs:
            return None
        lines = logs[0].read_text(encoding='utf-8', errors='ignore').splitlines()[-300:]
        for line in reversed(lines):
            for regime in ['TRENDING_UP','TRENDING_DOWN','RANGING','WEAK_TREND','CRISIS']:
                if regime in line:
                    return regime
    except Exception:
        pass
    return None

# ── 심볼×방향 통계 ────────────────────────────────────────────────
def _sym_dir_stats(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    rows = []
    for (sym, direction), g in df.groupby(['symbol','direction']):
        t = len(g); w = int((g['pnl_usd']>0).sum())
        rows.append({
            'symbol':    sym.replace('USDT',''),
            'direction': direction,
            'trades':    t,
            'wr':        round(w/t*100, 0),
            'pnl':       round(float(g['pnl_usd'].sum()), 2),
            'avg_r':     round(float(g['pnl_r'].mean()), 3),
            'best':      round(float(g['pnl_usd'].max()), 2),
            'worst':     round(float(g['pnl_usd'].min()), 2),
        })
    return sorted(rows, key=lambda x: abs(x['pnl']), reverse=True)

# ── 경고 판정 ─────────────────────────────────────────────────────
def _alerts(m: dict, bot_alive: bool) -> list:
    alerts = []
    if not bot_alive:
        alerts.append({'level':'critical','msg':'봇 프로세스 중단 감지 — 즉시 확인 필요'})
    if m['mdd_pct'] > 20:
        alerts.append({'level':'critical','msg':f'MDD {m["mdd_pct"]:.1f}% 초과 — 즉시 중단 검토'})
    elif m['mdd_pct'] > 15:
        alerts.append({'level':'warn','msg':f'MDD {m["mdd_pct"]:.1f}% 경고선 초과'})
    if m['total'] >= 10:
        if m['wr'] < 30:
            alerts.append({'level':'warn','msg':f'승률 {m["wr"]:.0f}% 경고선 미달 (기준 30%)'})
        if m['pf'] < 1.0:
            alerts.append({'level':'warn','msg':f'PF {m["pf"]:.2f} — 손실 구간'})
    return alerts

# ── API ──────────────────────────────────────────────────────────
@app.post('/api/auth')
async def auth(req: Request):
    body = await req.json()
    if body.get('password') == DASH_PASSWORD:
        return {'token': _new_token()}
    raise HTTPException(401, 'Invalid password')

@app.get('/api/status')
async def status(token: str):
    _auth(token)
    alive    = _bot_alive()
    log_time = _last_log_time()
    regime   = _regime_from_log()
    return {'bot_alive': alive, 'last_log': log_time, 'regime': regime}

@app.get('/api/dashboard')
async def dashboard(token: str, period: int = 0):
    _auth(token)
    df  = _trades(period if period > 0 else 9999)
    pos = _positions()
    m   = _metrics(df)

    # 봇 상태
    bot_alive = _bot_alive()
    log_time  = _last_log_time()

    # 레짐 (로그 우선, fallback: 마지막 거래)
    regime = _regime_from_log()
    if not regime and not df.empty:
        last = df.sort_values('exit_ts').dropna(subset=['regime'])
        if not last.empty:
            regime = str(last.iloc[-1]['regime'])
    regime = regime or 'UNKNOWN'

    # 경고
    alerts = _alerts(m, bot_alive)

    # 수익 곡선
    eq_curve = []
    if not df.empty:
        d = df.sort_values('exit_ts').copy()
        d['eq'] = INITIAL_CAPITAL + d['pnl_usd'].cumsum()
        for _, r in d.iterrows():
            eq_curve.append({'ts':  r['exit_ts'].isoformat() if pd.notna(r['exit_ts']) else None,
                             'eq':  round(float(r['eq']), 2),
                             'reason': r['reason'],
                             'pnl': round(float(r['pnl_usd']), 2)})

    # 일별 PnL
    daily = []
    if not df.empty and not df['exit_ts'].isna().all():
        for ts, v in df.set_index('exit_ts')['pnl_usd'].resample('1D').sum().tail(30).items():
            daily.append({'date': ts.strftime('%m/%d'), 'pnl': round(float(v), 2)})

    # 모듈별
    MOD = {'V2_MA_LONG':'A 추세','V2_MA_SHORT':'A 추세',
           'V2_MR_LONG':'B 평균회귀','V2_MR_SHORT':'B 평균회귀',
           'V2_MC_LONG':'C 브레이크아웃','V2_MC_SHORT':'C 브레이크아웃',
           'V2_MD_SHORT':'D 펀딩비'}
    module_stats = []
    if not df.empty:
        d2 = df.copy()
        d2['mod'] = d2['strategy'].map(MOD).fillna(d2['strategy'])
        for mod, g in d2.groupby('mod'):
            t=len(g); w=int((g['pnl_usd']>0).sum())
            gw=float(g[g['pnl_usd']>0]['pnl_usd'].sum())
            gl=abs(float(g[g['pnl_usd']<0]['pnl_usd'].sum()))
            module_stats.append({'module':mod,'trades':t,'wr':round(w/t*100,0),
                                 'pf':round(gw/gl,2) if gl>0 else 0,
                                 'pnl':round(float(g['pnl_usd'].sum()),2),
                                 'avg_r':round(float(g['pnl_r'].mean()),3),
                                 'reliable':t>=20})

    # 심볼×방향
    sym_dir = _sym_dir_stats(df)

    # 최근 거래 50건
    recent = []
    if not df.empty:
        for _, r in df.sort_values('exit_ts', ascending=False).head(50).iterrows():
            recent.append({'ts':        r['exit_ts'].strftime('%m-%d %H:%M') if pd.notna(r['exit_ts']) else '-',
                           'strategy':  r['strategy'],
                           'symbol':    r['symbol'],
                           'direction': r['direction'],
                           'entry':     round(float(r['entry_price']), 4),
                           'exit':      round(float(r['exit_price']),  4),
                           'pnl':       round(float(r['pnl_usd']), 2),
                           'r':         round(float(r['pnl_r']),   2),
                           'reason':    r['reason'],
                           'regime':    r['regime'] or ''})

    # 열린 포지션 (보유시간 + SL거리 추가)
    open_pos = []
    now_utc  = datetime.now(timezone.utc)
    if not pos.empty:
        for _, p in pos.iterrows():
            entry = float(p['entry_price'])
            sl    = float(p['sl'])
            # 보유시간
            hold_str = '—'
            try:
                et  = pd.to_datetime(p['entry_ts'], utc=True, errors='coerce')
                if pd.notna(et):
                    diff = (now_utc - et).total_seconds()
                    h, m_rem = divmod(int(diff), 3600)
                    hold_str = f'{h}h {m_rem//60}m'
            except Exception:
                pass
            # SL까지 거리 %
            sl_dist = '—'
            try:
                if entry > 0:
                    pct = abs(entry - sl) / entry * 100
                    sl_dist = f'{pct:.2f}%'
            except Exception:
                pass
            open_pos.append({'strategy':  p['strategy'],
                             'symbol':    p['symbol'],
                             'direction': p['direction'],
                             'entry':     round(entry, 4),
                             'sl':        round(sl, 4),
                             'tp':        round(float(p['tp']), 4),
                             'qty':       float(p['qty']),
                             'leverage':  int(p['leverage']),
                             'entry_ts':  str(p['entry_ts']),
                             'hold':      hold_str,
                             'sl_dist':   sl_dist})

    # 스트릭
    streak = {'count':0,'kind':'none'}
    if not df.empty:
        s = df.sort_values('exit_ts', ascending=False)
        is_win = float(s.iloc[0]['pnl_usd']) > 0
        cnt = 0
        for _, r in s.iterrows():
            if (float(r['pnl_usd'])>0) == is_win: cnt+=1
            else: break
        streak = {'count':cnt,'kind':'win' if is_win else 'loss'}

    return {'metrics':     m,
            'regime':      regime,
            'streak':      streak,
            'bot_alive':   bot_alive,
            'log_time':    log_time,
            'alerts':      alerts,
            'eq_curve':    eq_curve,
            'daily':       daily,
            'module_stats':module_stats,
            'sym_dir':     sym_dir,
            'recent':      recent,
            'open_pos':    open_pos,
            'refresh_sec': REFRESH_SEC,
            'updated_at':  datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

@app.get('/api/prices')
async def prices(token: str, symbols: str = ''):
    _auth(token)
    result = {}
    for sym in (symbols.split(',') if symbols else []):
        try:
            r = requests.get('https://fapi.binance.com/fapi/v1/ticker/price',
                             params={'symbol': sym}, timeout=5)
            if r.ok: result[sym] = float(r.json().get('price', 0))
        except Exception: pass
    return result

@app.get('/', response_class=HTMLResponse)
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding='utf-8'))
    return HTMLResponse('<h1>dashboard_ui.html not found</h1>', 500)

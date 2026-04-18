"""
ATLAS v2 Dashboard v2
=====================
탭 기반 레이아웃 + 모바일 최적화 + 전체 지표

[실행]
  streamlit run atlas_dashboard.py --server.port 8501 --server.address 0.0.0.0
"""

import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════
#  0. 설정
# ══════════════════════════════════════════════════════════════

load_dotenv(Path(__file__).parent / '.env')

DB_FILE         = Path(os.getenv('DB_FILE',
                    str(Path(__file__).parent / 'state' / 'atlas_v2.db')))
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000'))
DASH_PASSWORD   = os.getenv('DASH_PASSWORD', 'atlas2026')
REFRESH_SEC     = int(os.getenv('DASH_REFRESH_SEC', '60'))

st.set_page_config(
    page_title='ATLAS v2',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ══════════════════════════════════════════════════════════════
#  1. 모바일 CSS 주입
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── 공통 ── */
[data-testid="stAppViewContainer"] { background: #0e0e1a; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem !important; }

/* ── KPI 카드 ── */
.kpi-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    border-left: 4px solid;
}
.kpi-label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
.kpi-value { font-size: 22px; font-weight: 700; margin-top: 2px; }

/* ── 레짐 배지 ── */
.regime-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* ── 스트릭 ── */
.streak-box {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}

/* ── PC: 최대 너비 제한 (차트 무한 늘어짐 방지) ── */
@media (min-width: 769px) {
    .block-container {
        max-width: 1280px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
}

/* ── 모바일 대응 ── */
@media (max-width: 640px) {
    /* 컬럼 세로 쌓기 */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* 테이블 가로 스크롤 */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }
    /* 탭 텍스트 크기 */
    [data-testid="stTab"] button { font-size: 12px !important; padding: 6px 8px !important; }
    /* KPI 값 크기 축소 */
    .kpi-value { font-size: 18px !important; }
    .block-container { padding: 0.5rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  2. 인증
# ══════════════════════════════════════════════════════════════

def _auth() -> bool:
    if st.session_state.get('authed'):
        return True
    st.markdown('<h2>🔒 ATLAS v2</h2>', unsafe_allow_html=True)
    pw = st.text_input('비밀번호', type='password')
    if st.button('로그인', use_container_width=True):
        if pw == DASH_PASSWORD:
            st.session_state['authed'] = True
            st.rerun()
        else:
            st.error('비밀번호가 틀렸습니다.')
    return False

if not _auth():
    st.stop()

# ══════════════════════════════════════════════════════════════
#  3. DB 헬퍼
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_SEC)
def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, timeout=10)
        df  = pd.read_sql_query(sql, con, params=params)
        con.close()
        return df
    except Exception as e:
        st.toast(f'DB 오류: {e}', icon='⚠️')
        return pd.DataFrame()


def load_trades(period: str = '전체') -> pd.DataFrame:
    cutoff_map = {'오늘': 1, '7일': 7, '30일': 30}
    where = ''
    if period in cutoff_map:
        days = cutoff_map[period]
        where = f"WHERE exit_ts >= datetime('now', '-{days} days')"

    df = _query(f"""
        SELECT strategy, symbol, direction, mode,
               entry_price, exit_price, qty, leverage,
               pnl_usd, pnl_r, rr, hold_hours,
               reason, regime, entry_ts, exit_ts
        FROM trades {where}
        ORDER BY exit_ts ASC
    """)
    if df.empty:
        return df
    df['exit_ts']  = pd.to_datetime(df['exit_ts'],  utc=True, errors='coerce')
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
    return df


def load_positions() -> pd.DataFrame:
    return _query("""
        SELECT strategy, symbol, direction, mode,
               entry_price, sl, tp, qty, leverage,
               risk_usd, rr, bep_done, entry_ts
        FROM positions ORDER BY entry_ts DESC
    """)


# ══════════════════════════════════════════════════════════════
#  4. 실시간 가격 (공개 API, 인증 불필요)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def get_prices(symbols: list) -> dict:
    prices = {}
    for sym in symbols:
        try:
            r = requests.get(
                'https://fapi.binance.com/fapi/v1/ticker/price',
                params={'symbol': sym}, timeout=5)
            if r.ok:
                prices[sym] = float(r.json().get('price', 0))
        except Exception:
            pass
    return prices


# ══════════════════════════════════════════════════════════════
#  5. 지표 계산
# ══════════════════════════════════════════════════════════════

def calc_metrics(df: pd.DataFrame) -> dict:
    empty = dict(total=0, wins=0, wr=0, total_pnl=0,
                 pf=0, avg_r=0, mdd_pct=0, mdd_usd=0, sharpe=0)
    if df.empty:
        return empty

    total     = len(df)
    wins      = int((df['pnl_usd'] > 0).sum())
    wr        = wins / total * 100
    total_pnl = float(df['pnl_usd'].sum())
    gw        = float(df[df['pnl_usd'] > 0]['pnl_usd'].sum())
    gl        = abs(float(df[df['pnl_usd'] < 0]['pnl_usd'].sum()))
    pf        = round(gw / gl, 2) if gl > 0 else float('inf')
    avg_r     = float(df['pnl_r'].mean())

    cum        = df['pnl_usd'].cumsum().values
    peak       = np.maximum.accumulate(cum)
    dd         = peak - cum
    mdd_usd    = float(dd.max())
    mdd_pct    = mdd_usd / INITIAL_CAPITAL * 100

    sharpe = 0.0
    if not df['exit_ts'].isna().all():
        daily = (df.set_index('exit_ts')['pnl_usd']
                   .resample('1D').sum().fillna(0))
        if len(daily) > 1 and daily.std() > 0:
            sharpe = round(daily.mean() / daily.std() * (365 ** 0.5), 2)

    return dict(total=total, wins=wins, wr=wr, total_pnl=total_pnl,
                pf=pf, avg_r=avg_r, mdd_pct=mdd_pct, mdd_usd=mdd_usd,
                sharpe=sharpe)


def calc_streak(df: pd.DataFrame) -> tuple:
    """현재 연속 승/패. Returns (count, 'win'|'loss'|'none')"""
    if df.empty:
        return 0, 'none'
    s = df.sort_values('exit_ts', ascending=False)
    is_win = float(s.iloc[0]['pnl_usd']) > 0
    cnt = 0
    for _, row in s.iterrows():
        if (float(row['pnl_usd']) > 0) == is_win:
            cnt += 1
        else:
            break
    return cnt, ('win' if is_win else 'loss')


def get_regime(df: pd.DataFrame) -> str:
    if df.empty or 'regime' not in df.columns:
        return 'UNKNOWN'
    last = df.sort_values('exit_ts').dropna(subset=['regime'])
    return str(last.iloc[-1]['regime']) if not last.empty else 'UNKNOWN'


# ══════════════════════════════════════════════════════════════
#  6. 차트
# ══════════════════════════════════════════════════════════════

_DARK = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=8, r=8, t=30, b=8),
)


def equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text='거래 데이터 없음', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font_size=14, font_color='#888')
        fig.update_layout(**_DARK, height=260)
        return fig

    df = df.sort_values('exit_ts').copy()
    df['equity'] = INITIAL_CAPITAL + df['pnl_usd'].cumsum()

    fig.add_trace(go.Scatter(
        x=df['exit_ts'], y=df['equity'],
        mode='lines', name='잔고',
        line=dict(color='#00b4d8', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,180,216,0.07)',
    ))
    fig.add_hline(y=INITIAL_CAPITAL,
                  line_dash='dash', line_color='rgba(255,255,255,0.25)', line_width=1)

    for subset, color, symbol in [
        (df[df['reason'] == 'TP'], '#2ecc71', 'triangle-up'),
        (df[df['reason'] == 'SL'], '#e74c3c', 'triangle-down'),
    ]:
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['exit_ts'], y=subset['equity'],
                mode='markers',
                name='TP' if color == '#2ecc71' else 'SL',
                marker=dict(color=color, size=9, symbol=symbol),
            ))

    fig.update_layout(
        **_DARK, height=260,
        title='수익 곡선',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(title='USDT', gridcolor='rgba(255,255,255,0.06)'),
    )
    return fig


def mdd_gauge(mdd_pct: float) -> go.Figure:
    color = '#e74c3c' if mdd_pct > 10 else ('#f39c12' if mdd_pct > 5 else '#2ecc71')
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(mdd_pct, 1),
        number={'suffix': '%', 'font': {'size': 30, 'color': color}},
        title={'text': 'MDD', 'font': {'size': 13, 'color': '#888'}},
        gauge={
            'axis': {'range': [0, 20], 'tickcolor': '#555'},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0,  5], 'color': 'rgba(46,204,113,0.15)'},
                {'range': [5, 10], 'color': 'rgba(243,156,18,0.15)'},
                {'range': [10,20], 'color': 'rgba(231,76,60,0.15)'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 2},
                          'thickness': 0.75, 'value': 10},
        },
    ))
    fig.update_layout(**_DARK, height=200)
    return fig


def daily_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or df['exit_ts'].isna().all():
        fig.update_layout(**_DARK, height=180, title='일별 PnL')
        return fig

    daily = (df.set_index('exit_ts')['pnl_usd']
               .resample('1D').sum()
               .reset_index()
               .tail(30))
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in daily['pnl_usd']]

    fig.add_trace(go.Bar(
        x=daily['exit_ts'].dt.strftime('%m/%d'),
        y=daily['pnl_usd'],
        marker_color=colors,
        hovertemplate='%{x}: %{y:+.2f}$<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1)
    fig.update_layout(
        **_DARK, height=200,
        title='일별 PnL (최근 30일)',
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
    )
    return fig


def pnl_by_strategy_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(**_DARK, height=200, title='전략별 누적 PnL')
        return fig

    grp = df.groupby('strategy')['pnl_usd'].sum().reset_index()
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in grp['pnl_usd']]
    fig.add_trace(go.Bar(
        x=grp['strategy'], y=grp['pnl_usd'],
        marker_color=colors,
        hovertemplate='%{x}: %{y:+.2f}$<extra></extra>',
    ))
    fig.update_layout(
        **_DARK, height=220,
        title='전략별 누적 PnL',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  7. 공통 컴포넌트
# ══════════════════════════════════════════════════════════════

def kpi_card(col, label: str, value: str, color: str = '#888888'):
    col.markdown(
        f'<div class="kpi-card" style="border-color:{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def regime_badge(regime: str):
    cfg = {
        'TRENDING_UP':   ('#2ecc71', '🟢 TRENDING UP'),
        'TRENDING_DOWN': ('#e74c3c', '🔴 TRENDING DOWN'),
        'RANGING':       ('#f39c12', '🟡 RANGING'),
        'WEAK_TREND':    ('#3498db', '🔵 WEAK TREND'),
        'CRISIS':        ('#9b59b6', '🟣 CRISIS'),
    }
    color, label = cfg.get(regime, ('#888', f'⚪ {regime}'))
    st.markdown(
        f'<div class="regime-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{label}</div>',
        unsafe_allow_html=True,
    )


def streak_badge(cnt: int, kind: str):
    if kind == 'none' or cnt == 0:
        label, color = '기록 없음', '#888'
    elif kind == 'win':
        label, color = f'🔥 {cnt}연승 중', '#2ecc71'
    else:
        label, color = f'❄️ {cnt}연패 중', '#e74c3c'
    st.markdown(
        f'<div class="streak-box" style="border:1px solid {color}44">'
        f'<div class="kpi-label">연속 스트릭</div>'
        f'<div style="color:{color};font-size:20px;font-weight:700;margin-top:4px">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def module_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mod_map = {
        'V2_MA_LONG': 'A 추세', 'V2_MA_SHORT': 'A 추세',
        'V2_MR_LONG': 'B 평균회귀', 'V2_MR_SHORT': 'B 평균회귀',
        'V2_MC_LONG': 'C 브레이크아웃', 'V2_MC_SHORT': 'C 브레이크아웃',
    }
    df = df.copy()
    df['module'] = df['strategy'].map(mod_map).fillna(df['strategy'])
    rows = []
    for mod, grp in df.groupby('module'):
        total = len(grp)
        wins  = int((grp['pnl_usd'] > 0).sum())
        gw    = float(grp[grp['pnl_usd'] > 0]['pnl_usd'].sum())
        gl    = abs(float(grp[grp['pnl_usd'] < 0]['pnl_usd'].sum()))
        rows.append({
            '모듈':  mod,
            '거래':  total,
            '승률':  f'{wins/total*100:.0f}%',
            'PF':    round(gw/gl, 2) if gl > 0 else '∞',
            '총PnL': f'${grp["pnl_usd"].sum():+.2f}',
            'avgR':  f'{grp["pnl_r"].mean():+.3f}',
            '신뢰도': '⚠️' if total < 20 else '✅',
        })
    return pd.DataFrame(rows)


def symbol_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for sym, grp in df.groupby('symbol'):
        total = len(grp)
        wins  = int((grp['pnl_usd'] > 0).sum())
        rows.append({
            '심볼':  sym,
            '거래':  total,
            '승률':  f'{wins/total*100:.0f}%',
            '총PnL': f'${grp["pnl_usd"].sum():+.2f}',
            '최대손실': f'${grp["pnl_usd"].min():+.2f}',
            '최대수익': f'${grp["pnl_usd"].max():+.2f}',
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  8. 사이드바: 기간 필터
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('### ⚙️ 설정')
    period = st.selectbox('기간', ['전체', '30일', '7일', '오늘'], index=0)
    st.markdown('---')
    st.markdown(f'갱신 주기: **{REFRESH_SEC}초**')
    if st.button('🔄 지금 갱신'):
        st.cache_data.clear()
        st.rerun()
    st.markdown('---')
    if st.button('🔒 로그아웃'):
        st.session_state['authed'] = False
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  9. 데이터 로드
# ══════════════════════════════════════════════════════════════

trades    = load_trades(period)
positions = load_positions()
metrics   = calc_metrics(trades)
streak_cnt, streak_kind = calc_streak(trades)
regime    = get_regime(load_trades('전체'))  # 레짐은 전체 기준 최신값

syms = list(positions['symbol'].unique()) if not positions.empty else []
prices = get_prices(syms)

# ── 헤더 ────────────────────────────────────────────────────
now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
st.markdown(
    f'<h2 style="margin:0 0 4px 0">📈 ATLAS v2</h2>'
    f'<p style="color:#555;font-size:12px;margin:0">{now_str} · {period} 기준</p>',
    unsafe_allow_html=True,
)
regime_badge(regime)

# ══════════════════════════════════════════════════════════════
#  10. 탭 레이아웃
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(['📊 Overview', '📋 포지션', '📈 분석', '🚦 상태'])


# ── Tab 1: Overview ─────────────────────────────────────────
with tab1:
    # KPI 카드 (2행 × 3열)
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    pnl_color = '#2ecc71' if metrics['total_pnl'] >= 0 else '#e74c3c'
    wr_color  = '#2ecc71' if metrics['wr'] >= 40 else ('#f39c12' if metrics['wr'] >= 33 else '#e74c3c')
    pf_val    = metrics['pf']
    pf_color  = '#2ecc71' if pf_val >= 1.2 else ('#f39c12' if pf_val >= 1.0 else '#e74c3c')
    mdd_color = '#e74c3c' if metrics['mdd_pct'] > 10 else ('#f39c12' if metrics['mdd_pct'] > 5 else '#2ecc71')

    kpi_card(r1c1, '총 PnL',    f'${metrics["total_pnl"]:+.2f}', pnl_color)
    kpi_card(r1c2, '승률',       f'{metrics["wr"]:.0f}%  ({metrics["wins"]}W/{metrics["total"]-metrics["wins"]}L)', wr_color)
    kpi_card(r1c3, 'PF',        f'{pf_val:.2f}' if pf_val != float("inf") else '∞', pf_color)
    kpi_card(r2c1, '총 거래수', f'{metrics["total"]}건', '#888')
    kpi_card(r2c2, 'Sharpe',    f'{metrics["sharpe"]:.2f}', '#00b4d8')
    kpi_card(r2c3, 'avg R',     f'{metrics["avg_r"]:+.3f}', '#888')

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # 스트릭 + MDD 게이지 + 일별 PnL (3열)
    col_streak, col_gauge, col_daily = st.columns([1, 1, 2])
    with col_streak:
        streak_badge(streak_cnt, streak_kind)
    with col_gauge:
        st.plotly_chart(mdd_gauge(metrics['mdd_pct']),
                        use_container_width=True, config={'displayModeBar': False})
    with col_daily:
        st.plotly_chart(daily_bar(trades),
                        use_container_width=True, config={'displayModeBar': False})

    # 수익 곡선 (전체 폭)
    st.plotly_chart(equity_chart(trades),
                    use_container_width=True, config={'displayModeBar': False})


# ── Tab 2: 포지션 ───────────────────────────────────────────
with tab2:
    st.markdown(f'#### 열린 포지션 ({len(positions)}개)')
    if positions.empty:
        st.info('현재 열린 포지션 없음')
    else:
        rows = []
        for _, p in positions.iterrows():
            sym       = p['symbol']
            cur_price = prices.get(sym, 0)
            entry     = float(p['entry_price'])
            qty       = float(p['qty'])
            direction = p['direction']

            if cur_price > 0 and entry > 0:
                if direction == 'LONG':
                    upnl = (cur_price - entry) * qty
                else:
                    upnl = (entry - cur_price) * qty
                upnl_str = f'${upnl:+.2f}'
            else:
                upnl_str = '—'

            rows.append({
                '전략':     p['strategy'],
                '심볼':     sym,
                '방향':     '🟢 L' if direction == 'LONG' else '🔴 S',
                '진입가':   f'{entry:,.4f}',
                '현재가':   f'{cur_price:,.4f}' if cur_price else '—',
                '미실현PnL': upnl_str,
                'SL':       f'{float(p["sl"]):,.4f}',
                'TP':       f'{float(p["tp"]):,.4f}',
                '레버리지': f'{int(p["leverage"])}x',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('#### 최근 거래 (50건)')
    if trades.empty:
        st.info('거래 내역 없음')
    else:
        disp = trades.sort_values('exit_ts', ascending=False).head(50).copy()
        disp['exit_ts']     = disp['exit_ts'].dt.strftime('%m-%d %H:%M')
        disp['pnl_usd']     = disp['pnl_usd'].map(lambda x: f'${x:+.2f}')
        disp['pnl_r']       = disp['pnl_r'].map(lambda x: f'{x:+.2f}R')
        disp['entry_price'] = disp['entry_price'].map(lambda x: f'{x:,.4f}')
        disp['exit_price']  = disp['exit_price'].map(lambda x: f'{x:,.4f}')
        disp['direction']   = disp['direction'].map({'LONG': '🟢 L', 'SHORT': '🔴 S'})
        st.dataframe(
            disp[['exit_ts', 'strategy', 'symbol', 'direction',
                   'entry_price', 'exit_price', 'pnl_usd', 'pnl_r', 'reason', 'regime']],
            use_container_width=True, hide_index=True,
            column_config={
                'exit_ts':    '청산시각',
                'strategy':   '전략',
                'symbol':     '심볼',
                'direction':  '방향',
                'entry_price':'진입가',
                'exit_price': '청산가',
                'pnl_usd':    'PnL',
                'pnl_r':      'R',
                'reason':     '사유',
                'regime':     '레짐',
            }
        )


# ── Tab 3: 분석 ─────────────────────────────────────────────
with tab3:
    st.markdown('#### 모듈별 성과')
    mod_df = module_table(trades)
    if mod_df.empty:
        st.info('데이터 없음')
    else:
        st.dataframe(mod_df, use_container_width=True, hide_index=True)

    st.markdown('#### 심볼별 성과')
    sym_df = symbol_table(trades)
    if sym_df.empty:
        st.info('데이터 없음')
    else:
        st.dataframe(sym_df, use_container_width=True, hide_index=True)

    # 전략 PnL 바 + 방향별 성과 나란히
    ch_col, dir_col = st.columns([3, 2])
    with ch_col:
        st.plotly_chart(pnl_by_strategy_bar(trades),
                        use_container_width=True, config={'displayModeBar': False})
    with dir_col:
        st.markdown('#### 방향별 성과')
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        for direction, label in [('LONG', '🟢 LONG'), ('SHORT', '🔴 SHORT')]:
            if trades.empty:
                st.info(f'{label} 거래 없음')
            else:
                sub = trades[trades['direction'] == direction]
                if sub.empty:
                    st.info(f'{label} 거래 없음')
                else:
                    wins = int((sub['pnl_usd'] > 0).sum())
                    st.markdown(
                        f'<div class="kpi-card" style="border-color:#888">'
                        f'<div class="kpi-label">{label}</div>'
                        f'<div style="font-size:14px;margin-top:6px">'
                        f'{len(sub)}건 · 승률 {wins/len(sub)*100:.0f}% · '
                        f'${sub["pnl_usd"].sum():+.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Tab 4: 상태 ────────────────────────────────────────────
with tab4:
    m = metrics

    # 알림 판정
    alerts, stops = [], []
    if m['total'] >= 10:
        if m['wr'] < 30:
            alerts.append(f"승률 {m['wr']:.0f}% < 30% 경고선")
        if m['pf'] < 1.0:
            alerts.append(f"PF {m['pf']:.2f} < 1.0 — 손실 구간")
    if m['mdd_pct'] > 20:
        stops.append(f"MDD {m['mdd_pct']:.1f}% > 20% — 즉시 중단 검토")
    elif m['mdd_pct'] > 15:
        alerts.append(f"MDD {m['mdd_pct']:.1f}% > 15% — 허용치 초과")

    st.markdown('#### 🚦 봇 개입 판단')
    if stops:
        st.error('\n'.join(stops))
        st.error('→ 즉시 /stop 또는 포지션 정리 검토')
    elif alerts:
        st.warning('\n'.join(alerts))
        st.warning('→ 경고 상태: /pause 고려')
    else:
        st.success(f'✅ 모든 지표 정상 범위 (거래 {m["total"]}건 기준)')

    if m['total'] < 10:
        st.info(f'📌 {m["total"]}건 — 10건 이상부터 통계 신뢰도 확보')

    st.markdown('---')
    st.markdown('#### 📊 지표 상세')

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f"""
| 지표 | 현재값 | 기준 |
|------|--------|------|
| 거래수 | {m['total']}건 | 10건+ 권장 |
| 승률 | {m['wr']:.1f}% | 30% 이상 |
| PF | {m['pf']:.2f} | 1.0 이상 |
""")
    with ic2:
        st.markdown(f"""
| 지표 | 현재값 | 기준 |
|------|--------|------|
| MDD | {m['mdd_pct']:.1f}% | 경고 15% / 중단 20% |
| Sharpe | {m['sharpe']:.2f} | 0.5 이상 |
| avg R | {m['avg_r']:+.3f} | 양수 유지 |
""")

    st.markdown('---')
    st.markdown(f'**DB 경로:** `{DB_FILE}`')
    st.markdown(f'**초기 자본:** `${INITIAL_CAPITAL:,.0f}`')
    st.markdown(f'**갱신 주기:** `{REFRESH_SEC}초`')

# ══════════════════════════════════════════════════════════════
#  11. 자동 갱신
# ══════════════════════════════════════════════════════════════

time.sleep(REFRESH_SEC)
st.rerun()

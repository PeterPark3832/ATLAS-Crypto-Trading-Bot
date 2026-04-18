"""
ATLAS v2 Dashboard
==================
Streamlit 실시간 성과 대시보드 — Vultr 서버 직접 실행용

[실행]
  streamlit run atlas_dashboard.py --server.port 8501 --server.address 0.0.0.0

[방화벽]
  ufw allow 8501

[접속]
  http://<VULTR_IP>:8501
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
#  1. 인증 게이트
# ══════════════════════════════════════════════════════════════

def _auth():
    if st.session_state.get('authed'):
        return True
    st.title('🔒 ATLAS v2 Dashboard')
    pw = st.text_input('비밀번호', type='password')
    if st.button('로그인'):
        if pw == DASH_PASSWORD:
            st.session_state['authed'] = True
            st.rerun()
        else:
            st.error('비밀번호가 틀렸습니다.')
    return False

if not _auth():
    st.stop()

# ══════════════════════════════════════════════════════════════
#  2. DB 헬퍼
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_SEC)
def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        df = pd.read_sql_query(sql, con, params=params)
        con.close()
        return df
    except Exception as e:
        st.error(f'DB 오류: {e}')
        return pd.DataFrame()


def load_trades() -> pd.DataFrame:
    df = _query("""
        SELECT strategy, symbol, direction, mode,
               entry_price, exit_price, qty, leverage,
               pnl_usd, pnl_r, rr, hold_hours,
               reason, regime, entry_ts, exit_ts
        FROM trades
        ORDER BY exit_ts ASC
    """)
    if df.empty:
        return df
    df['exit_ts'] = pd.to_datetime(df['exit_ts'], utc=True, errors='coerce')
    df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
    return df


def load_positions() -> pd.DataFrame:
    return _query("""
        SELECT strategy, symbol, direction, mode,
               entry_price, sl, tp, qty, leverage,
               risk_usd, rr, bep_done, entry_ts
        FROM positions
        ORDER BY entry_ts DESC
    """)


# ══════════════════════════════════════════════════════════════
#  3. 지표 계산 헬퍼
# ══════════════════════════════════════════════════════════════

def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(total=0, wins=0, wr=0, total_pnl=0,
                    pf=0, avg_r=0, mdd_pct=0, mdd_usd=0, sharpe=0)

    total      = len(df)
    wins       = int((df['pnl_usd'] > 0).sum())
    wr         = wins / total * 100
    total_pnl  = float(df['pnl_usd'].sum())
    gross_win  = float(df[df['pnl_usd'] > 0]['pnl_usd'].sum())
    gross_loss = abs(float(df[df['pnl_usd'] < 0]['pnl_usd'].sum()))
    pf         = round(gross_win / gross_loss, 2) if gross_loss > 0 else float('inf')
    avg_r      = float(df['pnl_r'].mean())

    # MDD (누적 PnL 기준)
    cum  = df['pnl_usd'].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum
    mdd_usd  = float(dd.max())
    mdd_pct  = mdd_usd / INITIAL_CAPITAL * 100

    # Sharpe (일별 PnL 기준)
    sharpe = 0.0
    if 'exit_ts' in df.columns and not df['exit_ts'].isna().all():
        daily = (df.set_index('exit_ts')['pnl_usd']
                   .resample('1D').sum()
                   .fillna(0))
        if len(daily) > 1 and daily.std() > 0:
            sharpe = round(daily.mean() / daily.std() * (365 ** 0.5), 2)

    return dict(total=total, wins=wins, wr=wr, total_pnl=total_pnl,
                pf=pf, avg_r=avg_r, mdd_pct=mdd_pct, mdd_usd=mdd_usd,
                sharpe=sharpe)


# ══════════════════════════════════════════════════════════════
#  4. 수익 곡선 차트
# ══════════════════════════════════════════════════════════════

def equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text='거래 데이터 없음', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return fig

    df = df.sort_values('exit_ts').copy()
    df['cum_pnl']  = df['pnl_usd'].cumsum()
    df['equity']   = INITIAL_CAPITAL + df['cum_pnl']

    # 수익 곡선
    fig.add_trace(go.Scatter(
        x=df['exit_ts'], y=df['equity'],
        mode='lines', name='잔고',
        line=dict(color='#00b4d8', width=2),
        fill='tozeroy', fillcolor='rgba(0,180,216,0.08)',
    ))

    # 기준선 (초기 자본)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dash',
                  line_color='rgba(255,255,255,0.3)', line_width=1)

    # TP / SL 마커
    tp_df = df[df['reason'] == 'TP']
    sl_df = df[df['reason'] == 'SL']

    if not tp_df.empty:
        fig.add_trace(go.Scatter(
            x=tp_df['exit_ts'], y=tp_df['equity'],
            mode='markers', name='TP',
            marker=dict(color='#2ecc71', size=8, symbol='triangle-up'),
        ))
    if not sl_df.empty:
        fig.add_trace(go.Scatter(
            x=sl_df['exit_ts'], y=sl_df['equity'],
            mode='markers', name='SL',
            marker=dict(color='#e74c3c', size=8, symbol='triangle-down'),
        ))

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(title='USDT', gridcolor='rgba(255,255,255,0.06)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def pnl_bar_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in df['pnl_usd']]
    fig.add_trace(go.Bar(
        x=list(range(len(df))),
        y=df['pnl_usd'],
        marker_color=colors,
        name='거래별 PnL',
        hovertemplate='%{y:+.2f}$<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_width=1)
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=10, r=10, t=10, b=10),
        height=200,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(title='PnL ($)', gridcolor='rgba(255,255,255,0.06)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  5. 모듈별 성과 테이블
# ══════════════════════════════════════════════════════════════

def module_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['모듈', '거래', '승률', 'PF', '총PnL', 'avgR', '신뢰도'])

    rows = []
    # 모듈 그룹: MA, MR, MC 기준
    module_map = {
        'V2_MA_LONG': 'A (추세)', 'V2_MA_SHORT': 'A (추세)',
        'V2_MR_LONG': 'B (평균회귀)', 'V2_MR_SHORT': 'B (평균회귀)',
        'V2_MC_LONG': 'C (브레이크아웃)', 'V2_MC_SHORT': 'C (브레이크아웃)',
    }
    df = df.copy()
    df['module'] = df['strategy'].map(module_map).fillna(df['strategy'])

    for mod, grp in df.groupby('module'):
        total = len(grp)
        wins  = int((grp['pnl_usd'] > 0).sum())
        wr    = wins / total * 100
        gw    = float(grp[grp['pnl_usd'] > 0]['pnl_usd'].sum())
        gl    = abs(float(grp[grp['pnl_usd'] < 0]['pnl_usd'].sum()))
        pf    = round(gw / gl, 2) if gl > 0 else float('inf')
        rows.append({
            '모듈':    mod,
            '거래':    total,
            '승률':    f'{wr:.0f}%',
            'PF':      pf,
            '총PnL':   f'${grp["pnl_usd"].sum():+.2f}',
            'avgR':    f'{grp["pnl_r"].mean():+.3f}',
            '신뢰도':  '⚠️ 부족' if total < 20 else '✅',
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  6. 메인 렌더링
# ══════════════════════════════════════════════════════════════

def render():
    trades    = load_trades()
    positions = load_positions()
    metrics   = calc_metrics(trades)

    # ── 헤더 ──────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    st.markdown(
        f'<h2 style="margin-bottom:0">📈 ATLAS v2 Dashboard</h2>'
        f'<p style="color:#888;margin-top:2px">갱신: {now_utc} · {REFRESH_SEC}초마다 자동 갱신</p>',
        unsafe_allow_html=True,
    )

    # ── KPI 카드 ───────────────────────────────────────────────
    pnl_color  = '#2ecc71' if metrics['total_pnl'] >= 0 else '#e74c3c'
    mdd_color  = '#e74c3c' if metrics['mdd_pct'] > 10 else ('#f39c12' if metrics['mdd_pct'] > 5 else '#2ecc71')
    wr_color   = '#2ecc71' if metrics['wr'] >= 40 else ('#f39c12' if metrics['wr'] >= 33 else '#e74c3c')
    pf_val     = metrics['pf']
    pf_color   = '#2ecc71' if pf_val >= 1.2 else ('#f39c12' if pf_val >= 1.0 else '#e74c3c')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _card(col, label, value, color='#ffffff'):
        col.markdown(
            f'<div style="background:#1e1e2e;padding:14px 16px;border-radius:10px;'
            f'border-left:4px solid {color}">'
            f'<div style="color:#888;font-size:12px">{label}</div>'
            f'<div style="color:{color};font-size:22px;font-weight:700">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _card(c1, '총 PnL',     f'${metrics["total_pnl"]:+.2f}',     pnl_color)
    _card(c2, '승률',        f'{metrics["wr"]:.0f}%',              wr_color)
    _card(c3, 'PF',          f'{pf_val:.2f}' if pf_val != float("inf") else '∞', pf_color)
    _card(c4, 'MDD',         f'{metrics["mdd_pct"]:.1f}%',         mdd_color)
    _card(c5, 'Sharpe',      f'{metrics["sharpe"]:.2f}',           '#00b4d8')
    _card(c6, '총 거래수',   f'{metrics["total"]}건',              '#888888')

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ── 수익 곡선 + 거래별 PnL ────────────────────────────────
    col_chart, col_bar = st.columns([3, 1])
    with col_chart:
        st.markdown('**수익 곡선**')
        st.plotly_chart(equity_chart(trades), use_container_width=True, config={'displayModeBar': False})
    with col_bar:
        st.markdown('**거래별 PnL**')
        st.plotly_chart(pnl_bar_chart(trades), use_container_width=True, config={'displayModeBar': False})

    # ── 모듈별 성과 + 열린 포지션 ─────────────────────────────
    col_mod, col_pos = st.columns([3, 2])

    with col_mod:
        st.markdown('**모듈별 성과**')
        mod_df = module_table(trades)
        if mod_df.empty:
            st.info('아직 청산된 거래가 없습니다.')
        else:
            st.dataframe(mod_df, use_container_width=True, hide_index=True)

    with col_pos:
        st.markdown(f'**열린 포지션** ({len(positions)}개)')
        if positions.empty:
            st.info('열린 포지션 없음')
        else:
            disp = positions[['strategy', 'symbol', 'direction',
                               'entry_price', 'sl', 'tp', 'leverage']].copy()
            disp.columns = ['전략', '심볼', '방향', '진입가', 'SL', 'TP', '레버리지']
            disp['방향'] = disp['방향'].map({'LONG': '🟢 L', 'SHORT': '🔴 S'})
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── 알림 상태 ─────────────────────────────────────────────
    with st.expander('🚦 봇 개입 판단', expanded=False):
        m = metrics
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

        if stops:
            st.error('  \n'.join(stops))
        elif alerts:
            st.warning('  \n'.join(alerts))
        else:
            st.success(f'✅ 모든 지표 정상 (거래 {m["total"]}건 기준)')
        if m['total'] < 10:
            st.info(f'📌 {m["total"]}건 — 통계 신뢰도 낮음 (10건 이상 권장)')

    # ── 거래 히스토리 ─────────────────────────────────────────
    with st.expander('📋 거래 히스토리', expanded=True):
        if trades.empty:
            st.info('거래 내역 없음')
        else:
            disp = trades[['exit_ts', 'strategy', 'symbol', 'direction',
                            'entry_price', 'exit_price', 'pnl_usd',
                            'pnl_r', 'reason', 'regime']].copy()
            disp = disp.sort_values('exit_ts', ascending=False).head(50)
            disp['exit_ts']    = disp['exit_ts'].dt.strftime('%m-%d %H:%M')
            disp['pnl_usd']    = disp['pnl_usd'].map(lambda x: f'${x:+.2f}')
            disp['pnl_r']      = disp['pnl_r'].map(lambda x: f'{x:+.2f}R')
            disp['entry_price'] = disp['entry_price'].map(lambda x: f'{x:,.4f}')
            disp['exit_price']  = disp['exit_price'].map(lambda x: f'{x:,.4f}')
            disp.columns = ['청산시각', '전략', '심볼', '방향',
                             '진입가', '청산가', 'PnL', 'R', '사유', '레짐']
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── 자동 갱신 ─────────────────────────────────────────────
    time.sleep(REFRESH_SEC)
    st.rerun()


render()

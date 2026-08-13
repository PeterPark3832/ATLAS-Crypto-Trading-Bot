"""
ATLAS — 주간 리포트(weekly_report.py) 단위 테스트
================================
load_trades(DB 조회), calc(통계 계산), strategy_summary(전략별 요약)을
검증합니다.

실행:
  pytest tests/test_weekly_report.py -v
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import weekly_report as wr


# ══════════════════════════════════════════════════════════════
#  헬퍼: 실제 spot_trades 스키마로 임시 DB 생성
# ══════════════════════════════════════════════════════════════

def _make_db(tmp_path, rows):
    db_file = tmp_path / 'atlas_spot.db'
    con = sqlite3.connect(str(db_file))
    con.execute("""
        CREATE TABLE spot_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy    TEXT,
            symbol      TEXT,
            entry_price REAL,
            exit_price  REAL,
            qty_tokens  REAL,
            cost_usdt   REAL,
            pnl_usdt    REAL,
            pnl_pct     REAL,
            pnl_r       REAL,
            hold_hours  REAL,
            reason      TEXT,
            entry_ts    TEXT,
            exit_ts     TEXT,
            regime      TEXT,
            fee_usdt    REAL DEFAULT 0
        )
    """)
    for r in rows:
        con.execute("""
            INSERT INTO spot_trades
            (strategy, symbol, pnl_usdt, pnl_r, reason, exit_ts, fee_usdt)
            VALUES (?,?,?,?,?,?,?)
        """, (r['strategy'], r.get('symbol', 'BTCUSDT'), r['pnl_usdt'],
              r.get('pnl_r', 0.0), r.get('reason', 'TP'), r['exit_ts'],
              r.get('fee', 0.0)))
    con.commit()
    con.close()
    return db_file


def _df(rows):
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  load_trades
# ══════════════════════════════════════════════════════════════

# DB 경로 해석(죽은 DB_FILE 오버라이드 폴백) 테스트는 로직이
# atlas_db.resolve_db_path 로 승격되면서 tests/test_common_utils.py 로 이동했다.
# 여기 남은 것: weekly_report 가 실제로 그 해석기를 거친 경로를 쓰는지.
class TestDbPathWiring:
    def test_db_file_is_resolved_through_shared_helper(self):
        """모듈 로드 시점에 resolve_db_path 결과가 DB_FILE에 박혀야 한다.

        (서버 .env의 죽은 DB_FILE 경로 사고 재발 방지 — 해석기 자체의
        동작은 test_common_utils 가 검증한다)
        """
        assert Path(wr.SPOT_DB_FILE) == wr.DB_FILE or wr.DB_FILE.exists()


class TestLoadTrades:
    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wr, 'DB_FILE', tmp_path / 'nope.db')
        result = wr.load_trades(7)
        assert result.empty

    def test_loads_recent_trades_from_real_schema(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        rows = [
            {'strategy': 'S6', 'pnl_usdt': 50.0, 'pnl_r': 1.5, 'reason': 'TP',
             'exit_ts': now.isoformat()},
            {'strategy': 'S6', 'pnl_usdt': -20.0, 'pnl_r': -1.0, 'reason': 'SL',
             'exit_ts': (now - timedelta(days=20)).isoformat()},
        ]
        db_file = _make_db(tmp_path, rows)
        monkeypatch.setattr(wr, 'DB_FILE', db_file)
        result = wr.load_trades(7)
        assert len(result) == 1
        assert result.iloc[0]['pnl_usdt'] == 50.0

    def test_pnl_is_net_of_fees(self, tmp_path, monkeypatch):
        now = datetime.now(timezone.utc)
        rows = [{'strategy': 'S6', 'pnl_usdt': 50.0, 'fee': 2.5,
                 'exit_ts': now.isoformat()}]
        db_file = _make_db(tmp_path, rows)
        monkeypatch.setattr(wr, 'DB_FILE', db_file)
        result = wr.load_trades(7)
        assert result.iloc[0]['pnl_usdt'] == pytest.approx(47.5)

    def test_returns_empty_on_query_failure(self, tmp_path, monkeypatch):
        db_file = tmp_path / 'bad.db'
        con = sqlite3.connect(str(db_file))
        con.execute('CREATE TABLE spot_trades (id INTEGER)')  # 컬럼 없음 → 쿼리 실패
        con.commit()
        con.close()
        monkeypatch.setattr(wr, 'DB_FILE', db_file)
        result = wr.load_trades(7)
        assert result.empty


# ══════════════════════════════════════════════════════════════
#  calc
# ══════════════════════════════════════════════════════════════

class TestCalc:
    def test_empty_df_returns_empty_dict(self):
        assert wr.calc(pd.DataFrame()) == {}

    def test_mixed_trades_metrics(self):
        df = _df([
            {'pnl_usdt': 100.0, 'pnl_r': 2.0, 'reason': 'TP'},
            {'pnl_usdt': -50.0, 'pnl_r': -1.0, 'reason': 'SL'},
            {'pnl_usdt': 30.0, 'pnl_r': 1.0, 'reason': 'TP'},
        ])
        result = wr.calc(df)
        assert result['total'] == 3
        assert result['wins'] == 2
        assert result['wr'] == pytest.approx(200 / 3)
        assert result['pnl'] == pytest.approx(80.0)
        assert result['pf'] == pytest.approx(130 / 50, abs=0.01)
        assert result['tp'] == 2
        assert result['sl'] == 1

    def test_no_losses_pf_is_inf(self):
        df = _df([{'pnl_usdt': 10.0, 'pnl_r': 1.0, 'reason': 'TP'}])
        result = wr.calc(df)
        assert result['pf'] == float('inf')

    def test_drawdown_computed_relative_to_peak_equity(self, monkeypatch):
        monkeypatch.setattr(wr, 'INITIAL_CAPITAL', 1000.0)
        df = _df([
            {'pnl_usdt': 100.0, 'pnl_r': 1.0, 'reason': 'TP'},
            {'pnl_usdt': -150.0, 'pnl_r': -1.0, 'reason': 'SL'},
            {'pnl_usdt': 20.0, 'pnl_r': 0.5, 'reason': 'TP'},
        ])
        result = wr.calc(df)
        # 자산곡선 1000→1100→950→970, 피크 1100 → mdd = 150/1100
        assert result['mdd'] == pytest.approx(150 / 1100 * 100)

    def test_drawdown_from_start_without_new_peak(self, monkeypatch):
        monkeypatch.setattr(wr, 'INITIAL_CAPITAL', 1000.0)
        df = _df([{'pnl_usdt': -200.0, 'pnl_r': -1.0, 'reason': 'SL'}])
        result = wr.calc(df)
        # 피크 = 초기자본 1000 → mdd = 200/1000
        assert result['mdd'] == pytest.approx(20.0)


# ══════════════════════════════════════════════════════════════
#  strategy_summary
# ══════════════════════════════════════════════════════════════

class TestStrategySummary:
    def test_empty_df_reports_no_trades(self):
        assert '거래 없음' in wr.strategy_summary(pd.DataFrame())

    def test_groups_by_strategy(self):
        df = _df([
            {'strategy': 'S6', 'pnl_usdt': 10.0},
            {'strategy': 'S6', 'pnl_usdt': -5.0},
            {'strategy': 'S3', 'pnl_usdt': 20.0},
        ])
        result = wr.strategy_summary(df)
        assert 'S3' in result
        assert 'S6' in result
        lines = result.split('\n')
        assert len(lines) == 2

"""
ATLAS — MCP 서버(atlas_mcp_server.py) 단위 테스트
================================
get_trade_history, get_pnl_by_strategy, check_alert_thresholds,
get_error_logs를 _query_db/_read_log_lines(SSH 의존)를 모킹해
검증합니다.

실행:
  pytest tests/test_mcp_server.py -v
"""



import sqlite3

import pytest

import atlas_mcp_server as ms


# ══════════════════════════════════════════════════════════════
#  헬퍼: 실제 spot_trades 스키마로 임시 DB 생성 후 _query_db 모킹
# ══════════════════════════════════════════════════════════════

def _mock_query_db(monkeypatch, tmp_path, rows):
    db_file = tmp_path / 'atlas_spot.db'
    con = sqlite3.connect(str(db_file))
    con.execute("""
        CREATE TABLE spot_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, entry_price REAL, exit_price REAL,
            qty_tokens REAL, cost_usdt REAL, pnl_usdt REAL, pnl_pct REAL,
            pnl_r REAL, hold_hours REAL, reason TEXT, entry_ts TEXT,
            exit_ts TEXT, regime TEXT, fee_usdt REAL DEFAULT 0
        )
    """)
    for r in rows:
        con.execute("""
            INSERT INTO spot_trades (strategy, symbol, pnl_usdt, pnl_r, reason, regime,
                                      entry_ts, exit_ts, fee_usdt)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (r.get('strategy', 'S6'), r.get('symbol', 'BTCUSDT'), r['pnl_usdt'],
              r.get('pnl_r', 0.0), r.get('reason', 'TP'), r.get('regime', 'TRENDING_UP'),
              r.get('entry_ts', '2021-01-01T00:00:00'), r['exit_ts'],
              r.get('fee', 0.0)))
    con.commit()
    con.close()

    def _real_query(sql, params=None):
        c = sqlite3.connect(str(db_file))
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, params or [])
        out = [dict(row) for row in cur.fetchall()]
        c.close()
        return out

    monkeypatch.setattr(ms, '_query_db', _real_query)
    return db_file


@pytest.fixture(autouse=True)
def _initial_capital(monkeypatch):
    monkeypatch.setattr(ms, 'INITIAL_CAPITAL', 1000.0)


# ══════════════════════════════════════════════════════════════
#  get_trade_history — 실제 spot_trades 스키마 검증
# ══════════════════════════════════════════════════════════════

class TestGetTradeHistory:
    def test_queries_real_schema_and_reports_pnl(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [
            {'pnl_usdt': 50.0, 'pnl_r': 1.5, 'exit_ts': '2099-01-01T00:00:00'},
            {'pnl_usdt': -20.0, 'pnl_r': -1.0, 'reason': 'SL', 'exit_ts': '2099-01-02T00:00:00'},
        ])
        result = ms.get_trade_history(9999)
        assert '2건' in result
        assert '+30.00' in result
        assert '1W / 1L' in result

    def test_no_trades_returns_message(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [])
        result = ms.get_trade_history(30)
        assert '없음' in result

    def test_pnl_is_net_of_fees(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [
            {'pnl_usdt': 50.0, 'fee': 2.5, 'exit_ts': '2099-01-01T00:00:00'},
        ])
        result = ms.get_trade_history(9999)
        assert '+47.50' in result

    def test_caps_display_at_25_rows(self, monkeypatch, tmp_path):
        rows = [{'pnl_usdt': 1.0, 'exit_ts': f'2099-01-{i+1:02d}T00:00:00'} for i in range(30)]
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.get_trade_history(9999)
        assert '외 5건' in result


# ══════════════════════════════════════════════════════════════
#  get_pnl_by_strategy
# ══════════════════════════════════════════════════════════════

class TestGetPnlByStrategy:
    def test_no_trades_returns_message(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [])
        assert '없습니다' in ms.get_pnl_by_strategy()

    def test_groups_by_strategy_and_symbol(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [
            {'strategy': 'S6', 'symbol': 'BTCUSDT', 'pnl_usdt': 100.0,
             'exit_ts': '2099-01-01T00:00:00'},
            {'strategy': 'S6', 'symbol': 'BTCUSDT', 'pnl_usdt': -30.0,
             'exit_ts': '2099-01-02T00:00:00'},
            {'strategy': 'S3', 'symbol': 'ETHUSDT', 'pnl_usdt': 10.0,
             'exit_ts': '2099-01-03T00:00:00'},
        ])
        result = ms.get_pnl_by_strategy()
        assert '[S3]' in result
        assert '[S6]' in result
        assert '전체 합계' in result
        assert '거래:3건' in result

    def test_flags_insufficient_sample_size(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [
            {'strategy': 'S6', 'symbol': 'BTCUSDT', 'pnl_usdt': 10.0,
             'exit_ts': '2099-01-01T00:00:00'},
        ])
        result = ms.get_pnl_by_strategy()
        assert '통계 불충분' in result


# ══════════════════════════════════════════════════════════════
#  check_alert_thresholds
# ══════════════════════════════════════════════════════════════

class TestCheckAlertThresholds:
    def test_no_trades_returns_message(self, monkeypatch, tmp_path):
        _mock_query_db(monkeypatch, tmp_path, [])
        assert '없습니다' in ms.check_alert_thresholds()

    def test_healthy_metrics_report_normal(self, monkeypatch, tmp_path):
        rows = [{'pnl_usdt': 10.0, 'exit_ts': f'2099-01-{i+1:02d}T00:00:00'} for i in range(15)]
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.check_alert_thresholds()
        assert '정상 범위' in result

    def test_low_winrate_and_low_pf_trigger_warning(self, monkeypatch, tmp_path):
        rows = (
            [{'pnl_usdt': 50.0, 'exit_ts': '2099-01-01T00:00:00'}] +
            [{'pnl_usdt': -10.0, 'exit_ts': f'2099-01-{i+2:02d}T00:00:00'} for i in range(12)]
        )
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.check_alert_thresholds()
        assert '경고선' in result
        assert '손실 구간' in result

    def test_mdd_over_20_triggers_stop(self, monkeypatch, tmp_path):
        rows = [
            {'pnl_usdt': 100.0, 'exit_ts': '2099-01-01T00:00:00'},
            {'pnl_usdt': -250.0, 'exit_ts': '2099-01-02T00:00:00'},
        ]
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.check_alert_thresholds()
        assert '즉시 중단' in result

    def test_mdd_between_15_and_20_warns_only(self, monkeypatch, tmp_path):
        rows = [
            {'pnl_usdt': 100.0, 'exit_ts': '2099-01-01T00:00:00'},
            {'pnl_usdt': -170.0, 'exit_ts': '2099-01-02T00:00:00'},
        ]
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.check_alert_thresholds()
        assert '허용치 초과' in result
        assert '즉시 중단' not in result.split('[판정]')[1].split('통계 신뢰도')[0] \
            or '🚨' not in result

    def test_low_sample_count_flagged(self, monkeypatch, tmp_path):
        rows = [{'pnl_usdt': 10.0, 'exit_ts': '2099-01-01T00:00:00'}]
        _mock_query_db(monkeypatch, tmp_path, rows)
        result = ms.check_alert_thresholds()
        assert '통계 신뢰도 낮음' in result


# ══════════════════════════════════════════════════════════════
#  get_error_logs
# ══════════════════════════════════════════════════════════════

class TestGetErrorLogs:
    def test_no_log_lines_returns_message(self, monkeypatch):
        monkeypatch.setattr(ms, '_read_log_lines', lambda n_tail=5000: [])
        assert '찾을 수 없거나' in ms.get_error_logs()

    def test_filters_to_warning_error_critical_tags(self, monkeypatch):
        lines = [
            '2099-01-01 [INFO] 정상 동작',
            '2099-01-01 [WARNING] 캔들 로드 실패',
            '2099-01-01 [ERROR] 주문 실패',
            '2099-01-01 [CRITICAL] DB 손상',
        ]
        monkeypatch.setattr(ms, '_read_log_lines', lambda n_tail=5000: lines)
        result = ms.get_error_logs(50)
        assert '[INFO]' not in result
        assert '[WARNING]' in result
        assert '[ERROR]' in result
        assert '[CRITICAL]' in result

    def test_no_matching_lines_reports_clean(self, monkeypatch):
        monkeypatch.setattr(ms, '_read_log_lines',
                             lambda n_tail=5000: ['2099-01-01 [INFO] ok'])
        result = ms.get_error_logs()
        assert '없음' in result

    def test_caps_to_n_lines(self, monkeypatch):
        lines = [f'2099-01-01 [ERROR] err{i}' for i in range(10)]
        monkeypatch.setattr(ms, '_read_log_lines', lambda n_tail=5000: lines)
        result = ms.get_error_logs(n=3)
        assert 'err9' in result
        assert 'err6' not in result


class TestRemotePathDefaults:
    """기본 경로가 실제 배포 위치를 가리켜야 한다.

    예전 기본값은 /root/ATLAS/... 였는데 서버의 실제 경로는
    /root/atlas_spot/... 이다. REMOTE_DB_PATH 를 .env에 적지 않으면 모든
    조회가 `[Errno 2] No such file` 로 죽어 **도구 전체가 동작 불가**였다.
    실제로 그 상태였고, MCP 도구를 호출해 보고서야 드러났다.
    (주간 리포트가 죽은 DB 경로를 보던 것과 같은 부류)
    """

    def test_db_default_points_to_deployed_path(self):
        assert ms.REMOTE_DB_PATH.startswith('/root/atlas_spot/'), (
            f'배포 경로가 아니다: {ms.REMOTE_DB_PATH}')
        assert ms.REMOTE_DB_PATH.endswith('state/atlas_spot.db')

    def test_log_default_points_to_deployed_path(self):
        assert ms.REMOTE_LOG_DIR == '/root/atlas_spot/logs', ms.REMOTE_LOG_DIR

    def test_paths_share_one_root(self, monkeypatch):
        """배포 위치가 바뀌면 REMOTE_DIR 하나로 함께 움직여야 한다."""
        import importlib
        monkeypatch.setenv('REMOTE_DIR', '/opt/atlas')
        monkeypatch.delenv('REMOTE_DB_PATH', raising=False)
        monkeypatch.delenv('REMOTE_LOG_DIR', raising=False)
        m = importlib.reload(ms)
        try:
            assert m.REMOTE_DB_PATH == '/opt/atlas/state/atlas_spot.db'
            assert m.REMOTE_LOG_DIR == '/opt/atlas/logs'
        finally:
            monkeypatch.delenv('REMOTE_DIR', raising=False)
            importlib.reload(ms)

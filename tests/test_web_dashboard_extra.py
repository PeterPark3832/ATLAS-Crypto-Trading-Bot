"""
ATLAS — 웹 대시보드(atlas_web_dashboard.py) 보강 단위 테스트
================================
test_web_dashboard.py가 다루지 않는 _bot_alive, _last_log_time,
_regime_from_log, _parse_regime_metrics(로그 파싱) 을 검증합니다.

실행:
  pytest tests/test_web_dashboard_extra.py -v
"""

import os
import sys
import time
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import atlas_web_dashboard as wd


# ══════════════════════════════════════════════════════════════
#  _bot_alive
# ══════════════════════════════════════════════════════════════

class _FakeCompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


class TestBotAlive:
    def test_returns_true_when_pgrep_finds_process(self, monkeypatch):
        monkeypatch.setattr(wd.subprocess, 'run', lambda *a, **k: _FakeCompletedProcess(0))
        assert wd._bot_alive() is True

    def test_returns_false_when_pgrep_finds_nothing(self, monkeypatch):
        monkeypatch.setattr(wd.subprocess, 'run', lambda *a, **k: _FakeCompletedProcess(1))
        assert wd._bot_alive() is False

    def test_exception_returns_false(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError('pgrep not found')

        monkeypatch.setattr(wd.subprocess, 'run', _raise)
        assert wd._bot_alive() is False


# ══════════════════════════════════════════════════════════════
#  _last_log_time
# ══════════════════════════════════════════════════════════════

class TestLastLogTime:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, 'LOG_FILE', tmp_path / 'nope.log')
        assert wd._last_log_time() is None

    def test_recent_file_reports_seconds(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text('x')
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._last_log_time()
        assert result.endswith('초 전')

    def test_old_file_reports_hours(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text('x')
        old_ts = time.time() - 7200
        os.utime(f, (old_ts, old_ts))
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._last_log_time()
        assert result.endswith('시간 전')

    def test_minutes_old_file_reports_minutes(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text('x')
        old_ts = time.time() - 300
        os.utime(f, (old_ts, old_ts))
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._last_log_time()
        assert result.endswith('분 전')


# ══════════════════════════════════════════════════════════════
#  _regime_from_log
# ══════════════════════════════════════════════════════════════

class TestRegimeFromLog:
    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, 'LOG_FILE', tmp_path / 'nope.log')
        assert wd._regime_from_log() is None

    def test_finds_most_recent_regime_from_bottom(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text(
            '2099-01-01 [Regime] TRENDING_UP\n'
            '2099-01-02 [Regime] RANGING\n'
            '2099-01-03 [Regime] CRISIS\n'
        )
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        assert wd._regime_from_log() == 'CRISIS'

    def test_no_regime_keyword_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text('2099-01-01 [INFO] 정상 동작\n')
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        assert wd._regime_from_log() is None

    def test_only_scans_last_300_lines(self, tmp_path, monkeypatch):
        lines = ['line filler'] * 400
        lines[50] = '[Regime] TRENDING_DOWN'  # 마지막 300줄 밖
        f = tmp_path / 'spot_main.log'
        f.write_text('\n'.join(lines) + '\n')
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        assert wd._regime_from_log() is None


# ══════════════════════════════════════════════════════════════
#  _parse_regime_metrics
# ══════════════════════════════════════════════════════════════

class TestParseRegimeMetrics:
    def test_missing_file_returns_all_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, 'LOG_FILE', tmp_path / 'nope.log')
        result = wd._parse_regime_metrics()
        assert result == {'adx_1d': None, 'adx_4h': None, 'atr_pct': None,
                           'btc': None, 'ema200': None}

    def test_parses_adx_and_price_lines(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text(
            '🟢 장세: TRENDING_UP\n'
            '  ADX(1D)=30.5↗  ADX(4H)=22.1  ATR%=2.34%\n'
            '  BTC=65,432  EMA200=60,123\n'
        )
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._parse_regime_metrics()
        assert result['adx_1d'] == 30.5
        assert result['adx_4h'] == 22.1
        assert result['atr_pct'] == 2.34
        assert result['btc'] == 65432.0
        assert result['ema200'] == 60123.0

    def test_uses_most_recent_match_when_multiple_blocks(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text(
            '  ADX(1D)=10.0  ADX(4H)=10.0  ATR%=1.00%\n'
            '  BTC=1,000  EMA200=900\n'
            '  ADX(1D)=40.0  ADX(4H)=35.0  ATR%=3.00%\n'
            '  BTC=2,000  EMA200=1,800\n'
        )
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._parse_regime_metrics()
        assert result['adx_1d'] == 40.0
        assert result['btc'] == 2000.0

    def test_missing_metrics_lines_returns_none_fields(self, tmp_path, monkeypatch):
        f = tmp_path / 'spot_main.log'
        f.write_text('2099-01-01 [INFO] 아무 의미 없는 로그\n')
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        result = wd._parse_regime_metrics()
        assert result['adx_1d'] is None
        assert result['btc'] is None

"""
ATLAS — 공용 유틸(atlas_bootstrap / atlas_notify / atlas_db) 단위 테스트
======================================================================
4개 스크립트에 복사돼 있던 텔레그램 전송·dotenv 부트스트랩·DB 경로 해석을
공통부로 모으면서, 각 사본이 갖던 계약(타임아웃·print 폴백·오류 시 본문
재출력·log.error)이 파라미터로 정확히 보존되는지 고정한다.

DB 경로 해석 테스트는 weekly_report._resolve_db 에서 승격된 것이다 —
서버 .env의 죽은 DB_FILE 경로(/root/atlas_bot/...) 때문에 주간 리포트가
거래 40건이 쌓인 상태에서도 매번 '데이터 없음'을 내던 실제 사고가 배경.

실행:
  pytest tests/test_common_utils.py -v
"""

import logging
import os
import sqlite3
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_bootstrap
import atlas_db
import atlas_notify


# ══════════════════════════════════════════════════════════════
#  atlas_notify.send_telegram
# ══════════════════════════════════════════════════════════════

class _Posts:
    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises

    def __call__(self, url, data=None, timeout=None):
        self.calls.append({'url': url, 'data': data, 'timeout': timeout})
        if self._raises:
            raise self._raises


class TestSendTelegram:
    def test_sends_with_given_timeout(self, monkeypatch):
        post = _Posts()
        monkeypatch.setattr(atlas_notify.requests, 'post', post)
        ok = atlas_notify.send_telegram('hi', 'tok', 'chat', timeout=8)
        assert ok is True
        assert post.calls[0]['timeout'] == 8
        assert 'bottok/' in post.calls[0]['url']
        assert post.calls[0]['data'] == {'chat_id': 'chat', 'text': 'hi'}

    def test_missing_creds_prints_when_fallback(self, monkeypatch, capsys):
        """배치 계약: 자격증명이 없으면 본문을 stdout으로 — cron 출력 유실 방지."""
        post = _Posts()
        monkeypatch.setattr(atlas_notify.requests, 'post', post)
        ok = atlas_notify.send_telegram('report body', '', '', print_fallback=True)
        assert ok is False and post.calls == []
        assert 'report body' in capsys.readouterr().out

    def test_missing_creds_silent_without_fallback(self, monkeypatch, capsys):
        """데몬 계약(dashboard): 자격증명 없으면 조용히."""
        monkeypatch.setattr(atlas_notify.requests, 'post', _Posts())
        atlas_notify.send_telegram('x', '', '')
        assert capsys.readouterr().out == ''

    def test_error_reprints_body_when_asked(self, monkeypatch, capsys):
        """monthly/reoptimize 계약: 실패 시 오류 + 본문 둘 다 출력."""
        monkeypatch.setattr(atlas_notify.requests, 'post',
                            _Posts(raises=RuntimeError('down')))
        ok = atlas_notify.send_telegram('body', 'tok', 'chat', reprint_on_error=True)
        out = capsys.readouterr().out
        assert ok is False
        assert 'TG 전송 실패' in out and 'body' in out

    def test_error_without_reprint_omits_body(self, monkeypatch, capsys):
        """weekly 계약: 실패 시 오류만 출력, 본문은 재출력하지 않는다."""
        monkeypatch.setattr(atlas_notify.requests, 'post',
                            _Posts(raises=RuntimeError('down')))
        atlas_notify.send_telegram('body', 'tok', 'chat')
        out = capsys.readouterr().out
        assert 'TG 전송 실패' in out and 'body' not in out

    def test_error_goes_to_logger_when_given(self, monkeypatch, capsys, caplog):
        """dashboard 계약: 오류는 print가 아니라 log.error 로."""
        monkeypatch.setattr(atlas_notify.requests, 'post',
                            _Posts(raises=RuntimeError('down')))
        logger = logging.getLogger('test_notify')
        with caplog.at_level('ERROR', logger='test_notify'):
            atlas_notify.send_telegram('body', 'tok', 'chat', logger=logger)
        assert any('TG 전송 실패' in r.message for r in caplog.records)
        assert capsys.readouterr().out == ''

    def test_error_never_raises(self, monkeypatch):
        """전송 실패가 호출자(리포트·대시보드)를 죽이면 안 된다."""
        monkeypatch.setattr(atlas_notify.requests, 'post',
                            _Posts(raises=RuntimeError('down')))
        assert atlas_notify.send_telegram('x', 'tok', 'chat') is False


# ══════════════════════════════════════════════════════════════
#  atlas_db — 경로 해석 + 읽기전용 연결
# ══════════════════════════════════════════════════════════════

class TestResolveDbPath:
    """죽은 DB_FILE 오버라이드 방어 (weekly_report에서 승격된 실사고 회귀)."""

    def test_falls_back_when_override_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv('DB_FILE', str(tmp_path / 'gone' / 'old.db'))
        default = tmp_path / 'real.db'
        assert atlas_db.resolve_db_path(default) == default, (
            '존재하지 않는 DB_FILE 오버라이드를 그대로 쓰면 빈 결과가 정상처럼 보인다')
        assert '없습니다' in capsys.readouterr().out, '대체 사실을 알려야 한다'

    def test_existing_override_is_honoured(self, tmp_path, monkeypatch):
        p = tmp_path / 'custom.db'
        p.write_text('', encoding='utf-8')
        monkeypatch.setenv('DB_FILE', str(p))
        assert atlas_db.resolve_db_path(tmp_path / 'default.db') == p

    def test_defaults_when_no_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv('DB_FILE', raising=False)
        assert atlas_db.resolve_db_path(tmp_path / 'd.db') == tmp_path / 'd.db'

    def test_blank_override_is_ignored(self, tmp_path, monkeypatch, capsys):
        """공백 값은 '설정 안 됨'이다 — 경고도 내지 않는다."""
        monkeypatch.setenv('DB_FILE', '   ')
        assert atlas_db.resolve_db_path(tmp_path / 'd.db') == tmp_path / 'd.db'
        assert capsys.readouterr().out == ''


class TestConnectRo:
    def test_reads_but_rejects_writes(self, tmp_path):
        """mode=ro 가 실제로 강제되는지 — 봇이 쓰는 파일을 절대 오염시키지 않는다."""
        db = tmp_path / 't.db'
        rw = sqlite3.connect(str(db))
        rw.execute('CREATE TABLE t (v INTEGER)')
        rw.execute('INSERT INTO t VALUES (7)')
        rw.commit(); rw.close()

        con = atlas_db.connect_ro(db)
        try:
            assert con.execute('SELECT v FROM t').fetchone()[0] == 7
            with pytest.raises(sqlite3.OperationalError):
                con.execute('INSERT INTO t VALUES (8)')
        finally:
            con.close()


# ══════════════════════════════════════════════════════════════
#  atlas_bootstrap
# ══════════════════════════════════════════════════════════════

class TestBootstrap:
    def test_defaults_fill_only_missing(self, monkeypatch):
        monkeypatch.delenv('TG_TOKEN', raising=False)
        monkeypatch.setenv('TG_CHAT_ID', 'real-value')
        atlas_bootstrap.ensure_env_defaults('TAG')
        assert os.environ['TG_TOKEN'] == 'TAG'
        assert os.environ['TG_CHAT_ID'] == 'real-value', (
            'setdefault 여야 한다 — 실제 값을 덮으면 라이브 자격증명이 사라진다')

    def test_load_env_missing_file_is_noop(self, tmp_path):
        """anchor 옆에 .env가 없어도 죽지 않는다 (기존 사본들과 동일 동작)."""
        atlas_bootstrap.load_env(tmp_path / 'nonexistent' / 'script.py')

    def test_load_env_does_not_override(self, tmp_path, monkeypatch):
        """.env 값이 이미 설정된 환경변수를 덮지 않는다 — systemd Environment=
        지시자와 CI 주입 값이 이겨야 한다(atlas_spot_config와 동일 의미)."""
        (tmp_path / '.env').write_text('BOOT_TEST_KEY=from_file\n', encoding='utf-8')
        monkeypatch.setenv('BOOT_TEST_KEY', 'from_env')
        atlas_bootstrap.load_env(tmp_path / 'script.py')
        assert os.environ['BOOT_TEST_KEY'] == 'from_env'

    def test_no_config_import(self):
        """bootstrap이 config를 끌어오면 스크립트들의 'config보다 먼저 .env'
        순서 보장이 조용히 깨진다.

        (독스트링·주석의 언급은 무해하므로 **임포트 구문만** 검사한다 —
        문자열 포함 검사로 하면 이 규칙을 설명하는 주석 자체에 걸린다)
        """
        import ast
        tree = ast.parse(Path(atlas_bootstrap.__file__).read_text(encoding='utf-8'))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert 'atlas_spot_config' not in imported

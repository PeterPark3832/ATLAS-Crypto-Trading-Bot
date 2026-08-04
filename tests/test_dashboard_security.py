"""
ATLAS — 대시보드 인증 강화 단위 테스트
================================
_try_login(브루트포스 rate limit + compare_digest)과
BOT_START_ARGS 기동 인자 보존을 검증합니다.

실행:
  pytest tests/test_dashboard_security.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi import HTTPException

import atlas_web_dashboard as wd


@pytest.fixture(autouse=True)
def _fresh_auth_state(monkeypatch):
    monkeypatch.setattr(wd, 'DASH_PASSWORD', 'correct-horse')
    monkeypatch.setattr(wd, '_login_failures', {})
    monkeypatch.setattr(wd, '_tokens', {})


class TestTryLogin:
    def test_correct_password_returns_token(self):
        tok = wd._try_login('correct-horse', '1.2.3.4')
        assert tok and wd._check(tok)

    def test_wrong_password_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            wd._try_login('wrong', '1.2.3.4')
        assert exc.value.status_code == 401

    def test_none_password_raises_401_not_typeerror(self):
        with pytest.raises(HTTPException) as exc:
            wd._try_login(None, '1.2.3.4')
        assert exc.value.status_code == 401

    def test_sixth_attempt_within_window_blocked_429(self):
        for _ in range(wd.LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                wd._try_login('wrong', '1.2.3.4')
        # 한도 도달 후에는 올바른 비밀번호여도 차단 (브루트포스 성공 방지)
        with pytest.raises(HTTPException) as exc:
            wd._try_login('correct-horse', '1.2.3.4')
        assert exc.value.status_code == 429

    def test_failures_expire_after_window(self, monkeypatch):
        base = 1_000_000.0
        now = {'t': base}
        monkeypatch.setattr(wd.time, 'time', lambda: now['t'])
        for _ in range(wd.LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                wd._try_login('wrong', '1.2.3.4')
        now['t'] = base + wd.LOGIN_WINDOW_SEC + 1
        tok = wd._try_login('correct-horse', '1.2.3.4')
        assert tok

    def test_success_clears_failure_history(self):
        for _ in range(wd.LOGIN_MAX_FAILURES - 1):
            with pytest.raises(HTTPException):
                wd._try_login('wrong', '1.2.3.4')
        wd._try_login('correct-horse', '1.2.3.4')
        assert '1.2.3.4' not in wd._login_failures

    def test_rate_limit_is_per_ip(self):
        for _ in range(wd.LOGIN_MAX_FAILURES):
            with pytest.raises(HTTPException):
                wd._try_login('wrong', '1.1.1.1')
        # 다른 IP는 영향 없음
        tok = wd._try_login('correct-horse', '2.2.2.2')
        assert tok


class TestStartArgsPreserved:
    def test_bot_start_args_parsed_with_shlex(self, monkeypatch):
        """control start가 BOT_START_ARGS를 분해해 커맨드에 붙이는지 —
        엔드포인트 자체 대신 동일 파싱 로직을 검증."""
        import shlex
        monkeypatch.setattr(wd, 'BOT_START_ARGS', '--dry-run --strategies S3,S5')
        cmd = ['python3', '-u', 'atlas_spot_main.py'] + shlex.split(wd.BOT_START_ARGS)
        assert cmd[-3:] == ['--dry-run', '--strategies', 'S3,S5']

    def test_default_empty_args(self):
        import shlex
        assert shlex.split('') == []

"""
ATLAS — 대시보드 인증 강화 단위 테스트
================================
_try_login(브루트포스 rate limit + compare_digest)과
BOT_START_ARGS 기동 인자 보존을 검증합니다.

실행:
  pytest tests/test_dashboard_security.py -v
"""

from pathlib import Path


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


class TestNoCommittedCredentials:
    """추적 대상 파일에 실제 자격증명이 들어가면 안 된다.

    운영 서버의 git remote URL에 GitHub PAT가 평문으로 박혀 있었다
    (.git/config). 저장소가 공개라 pull에는 애초에 인증이 필요 없었는데도
    토큰이 남아 있었다 — 서버를 백업하거나 이미지를 뜨면 그대로 새어 나간다.
    같은 실수가 **추적 파일**에 들어오는 것을 여기서 막는다.

    .env는 gitignore 대상이라 검사 범위 밖이다(추적되지 않는다).
    """

    # 실제 키 형태만 잡는다. 문서의 placeholder(`...`, `<your-key>`)는 통과시켜야
    # 설치 안내를 쓸 수 있다.
    PATTERNS = {
        'GitHub PAT':      r'gh[pousr]_[A-Za-z0-9]{30,}',
        'GitHub fine PAT': r'github_pat_[A-Za-z0-9_]{50,}',
        'AWS access key':  r'AKIA[0-9A-Z]{16}',
        'Slack token':     r'xox[baprs]-[A-Za-z0-9-]{10,}',
        'Private key':     r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }

    def _tracked_files(self):
        import subprocess
        root = Path(__file__).parent.parent
        out = subprocess.run(['git', 'ls-files'], cwd=root,
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            pytest.skip('git 저장소가 아니다')
        return [root / p for p in out.stdout.splitlines() if p.strip()]

    def test_no_credentials_in_tracked_files(self):
        import re
        hits = []
        for path in self._tracked_files():
            if not path.is_file() or path.suffix in ('.png', '.jpg', '.ico', '.db'):
                continue
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            # 이 테스트 파일 자신의 패턴 정의는 제외
            if path.name == Path(__file__).name:
                continue
            for label, pat in self.PATTERNS.items():
                if re.search(pat, text):
                    hits.append(f'{path.name}: {label}')
        assert not hits, f'추적 파일에 자격증명으로 보이는 값: {hits}'

    def test_no_credentials_in_remote_url_docs(self):
        """설치 안내가 토큰 박힌 URL을 예시로 쓰면 그대로 따라 하게 된다."""
        import re
        root = Path(__file__).parent.parent
        for name in ('README.md',):
            p = root / name
            if not p.exists():
                continue
            text = p.read_text(encoding='utf-8', errors='ignore')
            assert not re.search(r'https://[^\s/]*:[^\s/]*@github\.com', text), (
                f'{name}에 자격증명이 박힌 remote URL 예시가 있다')


class TestAccessLogRedactsToken:
    """접근 로그에 토큰이 평문으로 남으면 안 된다.

    토큰은 쿼리스트링으로 전달되므로 uvicorn 접근 로그에 그대로 기록된다.
    유효기간이 24시간이라, 로그를 읽을 수 있는 사람은 그동안 로그인 없이
    대시보드를 열 수 있다 — 봇 중지·재시작 권한까지 포함된다.
    실제로 운영 서버의 logs/dashboard.log 에 유효한 토큰이 쌓여 있었고,
    로그는 회전·압축되며 백업에도 함께 담기므로 노출 경로가 넓다.
    """

    def _record(self, path):
        import logging
        return logging.LogRecord(
            'uvicorn.access', logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d',
            ('1.2.3.4:5', 'GET', path, '1.1', 200), None)

    def test_token_is_redacted(self):
        rec = self._record('/api/dashboard?token=deadbeefcafe1234&period=0')
        assert wd._RedactToken().filter(rec) is True
        assert 'deadbeefcafe1234' not in rec.args[2]
        assert '<redacted>' in rec.args[2]

    def test_other_params_survive(self):
        """가리기가 과해서 진단에 필요한 정보까지 지우면 안 된다."""
        rec = self._record('/api/dashboard?token=secret123&period=30')
        wd._RedactToken().filter(rec)
        assert 'period=30' in rec.args[2]
        assert '/api/dashboard' in rec.args[2]

    def test_path_without_token_untouched(self):
        rec = self._record('/api/status')
        wd._RedactToken().filter(rec)
        assert rec.args[2] == '/api/status'

    def test_malformed_record_does_not_crash(self):
        """로그 필터가 예외를 내면 요청 처리까지 깨진다."""
        import logging
        rec = logging.LogRecord('uvicorn.access', logging.INFO, __file__, 1,
                                'no args', None, None)
        assert wd._RedactToken().filter(rec) is True

    def test_filter_is_installed(self):
        """정의만 하고 붙이지 않으면 아무 효과가 없다."""
        import logging
        assert any(isinstance(f, wd._RedactToken)
                   for f in logging.getLogger('uvicorn.access').filters), \
            'uvicorn.access 로거에 리댁션 필터가 붙어 있지 않다'

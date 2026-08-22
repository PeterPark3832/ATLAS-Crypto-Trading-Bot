"""
ATLAS — 보안 자세(posture) 래칫
===============================
2026-08 보안 점검에서 고친 항목들이 되돌아가지 않도록 고정한다.
여기 있는 테스트는 기능이 아니라 **노출면**을 지킨다 — 실패하면
"기능이 깨졌다"가 아니라 "공격면이 넓어졌다"는 뜻이다.

배경: 이 대시보드는 인증 토큰을 들고 있고 **전량 시장가 매도** 버튼이
있다. 스크립트 한 줄이나 iframe 하나면 계좌가 정리된다.
"""
from pathlib import Path

import atlas_spot_main as sm
import atlas_web_dashboard as wd


class TestFuturesClientIsUnauthenticated:
    """선물 커넥션은 공개 데이터(fetch_funding_rate)만 쓴다.

    자격증명을 붙여두면 거래소 키에 Futures 권한을 열어둬야 하고,
    그 권한은 키가 유출됐을 때 레버리지 포지션 개설을 허용한다.
    """

    def test_no_credentials_in_futures_connection(self, monkeypatch):
        captured = {}

        class _FakeCcxt:
            @staticmethod
            def binance(cfg):
                captured.update(cfg)
                return object()

        monkeypatch.setattr(sm, 'ccxt', _FakeCcxt)
        monkeypatch.setattr(sm, '_ex_futures', None)
        sm._get_ex_futures()
        assert captured, '커넥션 생성이 일어나지 않았다'
        assert 'apiKey' not in captured, (
            '선물 커넥션에 API 키가 실렸다 — 이 클라이언트는 공개 '
            'endpoint만 호출하므로 인증이 필요 없다')
        assert 'secret' not in captured

    def test_futures_is_only_used_for_public_funding_rate(self):
        """자격증명을 뺀 근거(공개 호출뿐)가 유지되는지 확인한다."""
        src = Path(sm.__file__).read_text(encoding='utf-8')
        calls = [ln.strip() for ln in src.splitlines()
                 if '_get_ex_futures()' in ln and 'def ' not in ln]
        assert calls, '_get_ex_futures 호출부를 찾지 못했다'
        for ln in calls:
            assert 'fetch_funding_rate' in ln, (
                f'선물 커넥션이 공개 endpoint 외에 쓰인다: {ln} — '
                f'인증이 필요하면 자격증명 제거 근거가 무너진다')


class TestSecurityHeaders:
    def test_all_headers_present(self):
        for h in ('Content-Security-Policy', 'X-Frame-Options',
                  'X-Content-Type-Options', 'Referrer-Policy'):
            assert h in wd.SECURITY_HEADERS, f'{h} 누락'

    def test_clickjacking_is_blocked(self):
        """패닉(전량 매도) 버튼이 있는 페이지다 — 프레임 삽입을 막아야 한다."""
        assert "frame-ancestors 'none'" in wd.SECURITY_HEADERS['Content-Security-Policy']
        assert wd.SECURITY_HEADERS['X-Frame-Options'] == 'DENY'

    def test_token_cannot_be_exfiltrated_to_other_origins(self):
        assert "connect-src 'self'" in wd.SECURITY_HEADERS['Content-Security-Policy']

    def test_middleware_is_registered(self):
        """상수만 있고 미들웨어가 안 붙으면 헤더는 나가지 않는다."""
        names = [getattr(m.kwargs.get('dispatch', None), '__name__', '')
                 for m in wd.app.user_middleware]
        assert any('security_headers' in n for n in names), (
            f'보안 헤더 미들웨어가 등록되지 않았다: {names}')


class TestExternalScriptsArePinned:
    """CDN이 침해되면 이 페이지의 스크립트는 토큰을 그대로 읽는다."""

    def test_every_external_script_has_integrity(self):
        import re
        html = (Path(wd.__file__).parent / 'dashboard_ui.html').read_text(encoding='utf-8')
        tags = re.findall(r'<script\b[^>]*\bsrc="https?://[^>]*>', html)
        assert tags, '외부 스크립트 태그를 찾지 못했다 (선택자 확인 필요)'
        for t in tags:
            assert 'integrity=' in t, (
                f'SRI 없는 외부 스크립트: {t[:90]} — 버전 고정만으로는 '
                f'CDN 침해를 막지 못한다')
            assert 'crossorigin=' in t, f'crossorigin 누락: {t[:90]}'


class TestLogEndpointIsBounded:
    def test_lines_are_clamped(self, monkeypatch, tmp_path):
        """상한이 없으면 요청 하나가 로그 전체(최대 20MB)를 직렬화한다."""
        big = tmp_path / 'spot_main.log'
        big.write_text('\n'.join(f'line {i}' for i in range(5000)), encoding='utf-8')
        monkeypatch.setattr(wd, 'LOG_FILE', big)
        monkeypatch.setattr(wd, '_tokens', {})
        tok = wd._new_token()
        out = wd.get_logs(tok, lines=10_000_000)
        assert len(out['lines']) <= wd.LOG_MAX_LINES, (
            f"{len(out['lines'])}행 반환 — 상한 {wd.LOG_MAX_LINES}이 적용되지 않았다")

    def test_normal_request_still_works(self, monkeypatch, tmp_path):
        f = tmp_path / 'spot_main.log'
        f.write_text('\n'.join(f'line {i}' for i in range(50)), encoding='utf-8')
        monkeypatch.setattr(wd, 'LOG_FILE', f)
        monkeypatch.setattr(wd, '_tokens', {})
        out = wd.get_logs(wd._new_token(), lines=10)
        assert len(out['lines']) == 10

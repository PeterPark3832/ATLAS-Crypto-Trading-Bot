"""
ATLAS — 예외 처리 규율
================================
이 저장소에서 반복해서 나온 결함은 전부 같은 모양이었다:
**코드는 돌고 로그는 정상인데 기능이 죽어 있다.**

`except Exception: pass`는 그 실패를 만드는 가장 빠른 방법이다. 예외를
삼키면 봇은 계속 돌지만, 삼킨 그 지점의 기능은 사라진다 — 그리고 아무도
모른다. 특히 라이브 주문·포지션 경로에서는 삼킨 예외 하나가
"손절이 걸린 줄 알았는데 안 걸린" 상태를 만든다.

여기서는 두 가지를 강제한다:
  ① 라이브 주문 경로의 예외 처리는 **반드시 흔적을 남긴다**(로그 or 재발생)
  ② 전체 조용한 삼킴 수가 현재보다 늘지 않는다(래칫)

실행:
  pytest tests/test_error_handling.py -v
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent   # 소스 스캔 기준 경로 (경로 삽입은 conftest)

import pytest

# 현재 실측치. **목표가 아니라 래칫**이다 — 줄이는 건 환영, 늘리는 건 실패.
# 남아 있는 것들은 전수 검토했고 대부분 정상 흐름제어(queue.Empty)이거나
# 문서화된 기본값으로 물러나는 파싱 폴백이다.
MAX_SILENT_TOTAL = 21

# 라이브에서 주문을 내거나 포지션을 관리하는 모듈 — 여기서는 삼킴을 허용하되
# **흔적 없는** 삼킴만 센다. 정상 흐름제어는 아래 화이트리스트로 뺀다.
LIVE_PATH = ['atlas_spot_main.py']

# 예외 자체가 곧 신호라서 로그가 불필요한 경우
CONTROL_FLOW_EXC = {'Empty', 'Full', 'StopIteration', 'KeyboardInterrupt',
                    'ImportError', 'ModuleNotFoundError'}


def _exc_names(handler: ast.ExceptHandler) -> set:
    t = handler.type
    if t is None:
        return set()
    nodes = t.elts if isinstance(t, ast.Tuple) else [t]
    out = set()
    for n in nodes:
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _leaves_trace(handler: ast.ExceptHandler) -> bool:
    """로그를 남기거나 예외를 다시 던지는가."""
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise):
            return True
        dumped = ast.dump(stmt)
        if 'log' in dumped or 'print' in dumped or '_tg' in dumped:
            return True
    return False


def _handlers(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            yield node


def _silent(path: Path) -> list:
    out = []
    for h in _handlers(path):
        if _leaves_trace(h):
            continue
        if _exc_names(h) & CONTROL_FLOW_EXC:
            continue
        out.append(h.lineno)
    return out


PROD_FILES = sorted(p for p in ROOT.glob('*.py'))


class TestNoSilentSwallowingRatchet:
    def test_total_silent_handlers_do_not_grow(self):
        found = {p.name: _silent(p) for p in PROD_FILES}
        total = sum(len(v) for v in found.values())
        detail = ', '.join(f'{k}:{len(v)}' for k, v in sorted(found.items()) if v)
        assert total <= MAX_SILENT_TOTAL, (
            f'조용한 예외 삼킴이 {total}건으로 늘었다(허용 {MAX_SILENT_TOTAL}). '
            f'예외를 삼키면 봇은 계속 돌지만 그 기능은 사라진다. [{detail}]')

    def test_ratchet_is_tight(self):
        """래칫이 실제보다 헐거우면 아무것도 막지 못한다."""
        total = sum(len(_silent(p)) for p in PROD_FILES)
        assert total >= MAX_SILENT_TOTAL - 4, (
            f'실측 {total} ≪ 래칫 {MAX_SILENT_TOTAL} — MAX_SILENT_TOTAL을 '
            f'{total}로 낮춰 조여야 한다')


class TestLiveOrderPathIsLoud:
    """주문·포지션 경로에서 흔적 없는 삼킴은 **손절이 걸린 줄 아는** 상태를
    만든다. 실제로 이번 감사에서 비상 손절 실패가 조용히 넘어가고 있었다."""

    RISKY = {
        '_spot_buy', '_spot_buy_locked', '_spot_sell', '_manage_position',
        '_place_stop_loss_order', '_place_protective_orders',
        '_cancel_orphan_sell_orders', '_rearm_missing_protection',
    }

    @pytest.mark.parametrize('fname', LIVE_PATH)
    def test_risky_functions_leave_a_trace(self, fname):
        tree = ast.parse((ROOT / fname).read_text())
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name not in self.RISKY:
                continue
            for h in ast.walk(fn):
                if not isinstance(h, ast.ExceptHandler):
                    continue
                if _leaves_trace(h) or (_exc_names(h) & CONTROL_FLOW_EXC):
                    continue
                offenders.append(f'{fn.name}():{h.lineno}')
        assert not offenders, (
            f'주문/포지션 경로에서 흔적 없이 예외를 삼킨다: {offenders}\n'
            f'실패해도 봇은 계속 돌지만 그 보호는 사라진다.')

    def test_risky_function_list_is_current(self):
        """대상 함수가 이름이 바뀌면 이 테스트가 조용히 무의미해진다."""
        import atlas_spot_main as sm
        missing = [n for n in self.RISKY if not hasattr(sm, n)]
        assert not missing, f'존재하지 않는 함수를 검사 중: {missing}'


class TestBareExceptForbidden:
    def test_no_bare_except(self):
        """`except:`는 KeyboardInterrupt·SystemExit까지 잡아 종료를 막는다."""
        bad = []
        for p in PROD_FILES:
            bad += [f'{p.name}:{h.lineno}' for h in _handlers(p) if h.type is None]
        assert not bad, f'bare except 발견: {bad}'

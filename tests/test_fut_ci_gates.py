"""
#7 CI 게이트가 선물 모듈을 실제로 본다.

ci.yml 의 라이브 커버리지 단계 주석이 경고하는 실패 유형 그대로다 — 주문을 내는 코드가
게이트 대상 파일 목록에 없으면, 테스트를 지워도·타입이 깨져도 CI는 초록불을 켠다.
실제로 선물 모듈이 mypy 대상에서 빠져 있는 동안 atlas_fut_backtest.py 의 타입 오류 1건이
CI에 보이지 않았다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CI = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
FUT_MODULES = sorted(p.name for p in ROOT.glob('atlas_fut_*.py'))


def test_futures_modules_exist():
    assert {'atlas_fut_main.py', 'atlas_fut_exchange.py', 'atlas_fut_core.py'} <= set(FUT_MODULES)


def test_mypy_covers_every_futures_module():
    mypy_lines = [ln for ln in CI.splitlines() if re.match(r'\s*run:\s*mypy\b', ln)]
    assert mypy_lines, 'ci.yml 에서 mypy 단계를 찾지 못했다'
    missing = [m for m in FUT_MODULES if not any(m in ln for ln in mypy_lines)]
    assert not missing, f'mypy 가 보지 않는 선물 모듈: {missing}'


def test_live_coverage_gate_includes_futures_order_path():
    m = re.search(r"coverage report --include='([^']+)'", CI)
    assert m, 'ci.yml 에서 라이브 경로 커버리지 게이트를 찾지 못했다'
    included = set(m.group(1).split(','))
    missing = {'atlas_fut_main.py', 'atlas_fut_exchange.py', 'atlas_fut_core.py'} - included
    assert not missing, f'주문 경로인데 라이브 커버리지 게이트에서 빠진 모듈: {sorted(missing)}'

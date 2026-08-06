"""
ATLAS — WFO → 재최적화 트리거 체인
================================
재최적화기는 '실패 구제' 도구로 만들어졌지만, 지금은 **신규 기능의 검증
통로**이기도 하다(추적 손절 ON/OFF 등이 그리드에 있다).

문제: 트리거 조건이 "OOS 미달 전략이 있을 때"뿐이었다. 전 전략이 통과
중인 정상 상태에서는 재최적화가 **한 번도 돌지 않으므로**, 기능을 넣어두고
데이터가 판단하게 하겠다는 계획이 그대로 멈춘다.
(실제로 합성 데이터 실행에서 "2/2 PASS"가 나와 트리거되지 않았다)

실행:
  pytest tests/test_wfo_trigger.py -v
"""

import os
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import monthly_wfo_report as wfo


@pytest.fixture
def calls(monkeypatch):
    got = []
    monkeypatch.setattr(wfo.subprocess, 'run',
                        lambda *a, **k: got.append(a[0]) or None)
    monkeypatch.delenv('WFO_AUTO_REOPT', raising=False)
    monkeypatch.delenv('WFO_REOPT_ALWAYS', raising=False)
    return got


class TestTriggerChain:
    def test_no_trigger_without_opt_in(self, calls):
        """로컬/수동 실행에서는 systemctl을 건드리지 않는다."""
        wfo.maybe_trigger_reopt(['S4'])
        assert calls == []

    def test_triggers_on_failing_strategy(self, calls, monkeypatch):
        monkeypatch.setenv('WFO_AUTO_REOPT', '1')
        wfo.maybe_trigger_reopt(['S4'])
        assert len(calls) == 1
        assert 'atlas-wfo-reopt.service' in calls[0]
        assert '--no-block' in calls[0], '리포트 잡을 붙잡지 않아야 한다'

    def test_silent_when_all_pass_by_default(self, calls, monkeypatch):
        """기본값은 실패 시에만 — 자원이 제한된 서버를 배려한 선택."""
        monkeypatch.setenv('WFO_AUTO_REOPT', '1')
        wfo.maybe_trigger_reopt([])
        assert calls == []

    def test_always_flag_triggers_when_all_pass(self, calls, monkeypatch):
        """이 경로가 없으면 신규 기능(추적 손절 등) 검증이 영원히 안 돈다."""
        monkeypatch.setenv('WFO_AUTO_REOPT', '1')
        monkeypatch.setenv('WFO_REOPT_ALWAYS', '1')
        wfo.maybe_trigger_reopt([])
        assert len(calls) == 1

    def test_always_flag_needs_auto_reopt(self, calls, monkeypatch):
        """ALWAYS만 켜도 AUTO_REOPT가 없으면 동작하지 않는다(오발동 방지)."""
        monkeypatch.setenv('WFO_REOPT_ALWAYS', '1')
        wfo.maybe_trigger_reopt([])
        assert calls == []

    def test_systemctl_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setenv('WFO_AUTO_REOPT', '1')
        monkeypatch.setattr(wfo.subprocess, 'run',
                            lambda *a, **k: (_ for _ in ()).throw(OSError('no systemd')))
        wfo.maybe_trigger_reopt(['S4'])      # 예외가 리포트를 죽이면 안 된다


class TestDeployUnits:
    ROOT = Path(__file__).parent.parent / 'deploy'

    def test_report_has_timer(self):
        assert (self.ROOT / 'atlas-wfo-report.timer').exists()

    def test_reopt_has_no_timer_by_design(self):
        """재최적화는 리포트가 트리거한다 — 독립 타이머가 있으면 중복 실행된다.

        지키려는 불변식은 '재최적화에 타이머가 없다'는 것 하나다. 예전에는
        deploy 디렉터리의 타이머가 **정확히 1개**인지로 검사했는데, 그러면
        무관한 잡(주간 리포트 등)을 타이머로 등록하는 것만으로 깨진다.
        의도한 적 없는 제약이라 실제로 정상 추가를 막았다.
        """
        assert not (self.ROOT / 'atlas-wfo-reopt.timer').exists(), (
            '재최적화에 독립 타이머가 생기면 리포트 트리거와 겹쳐 '
            '메모리 피크가 중복된다(RAM 951MB 서버)')

    def test_reopt_runs_after_report(self):
        unit = (self.ROOT / 'atlas-wfo-reopt.service').read_text()
        assert 'After=' in unit and 'atlas-wfo-report.service' in unit, (
            '리포트와 동시에 돌면 메모리 피크가 겹친다(RAM 951MB 서버)')

    def test_always_option_documented(self):
        """운영자가 이 스위치의 존재와 트레이드오프를 알 수 있어야 한다."""
        unit = (self.ROOT / 'atlas-wfo-report.service').read_text()
        assert 'WFO_REOPT_ALWAYS' in unit
        assert '분기' in unit or '수동' in unit, '권장 운용법이 적혀 있어야 한다'

    def test_memory_isolation_present(self):
        """라이브 봇을 지키기 위한 cgroup 상한이 두 잡 모두에 있어야 한다."""
        for name in ('atlas-wfo-report.service', 'atlas-wfo-reopt.service'):
            assert 'MemoryMax=' in (self.ROOT / name).read_text(), name

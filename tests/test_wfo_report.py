"""
ATLAS — 월간 WFO 리포트 + 재최적화 단위 테스트
================================================
백테스트 실행 없이 순수 로직만 검증한다:
  monthly_wfo_report: evaluate(PASS/FAIL 판정), format_message
  reoptimize:         override_params(치환/원복), current_params,
                      pick_targets, format_proposal

실행:
  pytest tests/test_wfo_report.py -v
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import monthly_wfo_report as mr
import reoptimize as ro
import atlas_spot_strategies as strat
from atlas_spot_config import WF_OOS_MIN_PF, WF_OOS_MIN_SHARPE


def _m(pf, sharpe, pnl, trades, cagr=5.0, wr=55.0):
    return {'profit_factor': pf, 'sharpe': sharpe, 'total_pnl_pct': pnl,
            'total_trades': trades, 'cagr_pct': cagr, 'win_rate': wr}


# ── monthly_wfo_report.evaluate ────────────────────────────────

def _wf_fixture():
    return {
        'IS': {
            'S4': _m(2.0, 1.0, 40, 60),
            'S5': _m(1.8, 0.9, 30, 50),
            'S3': _m(1.5, 0.8, 20, 40),
            'S6': _m(1.0, 0.1, 0, 0),
        },
        'OOS': {
            'S4': _m(1.5, 0.5, 10, 30),    # PASS
            'S5': _m(0.9, 0.4, -5, 30),    # FAIL: PF < 1.10
            'S3': _m(1.2, 0.2, 3, 30),     # FAIL: sharpe < 0.30
            'S6': _m(0.0, 0.0, 0, 0),      # has_data False
        },
    }


def test_evaluate_verdicts():
    rows = mr.evaluate(_wf_fixture(), ['S4', 'S5', 'S3', 'S6'])
    by = {r['sid']: r for r in rows}

    assert by['S4']['verdict'] == 'PASS'
    assert by['S4']['has_data'] is True
    assert by['S4']['ratio_pf'] == round(1.5 / 2.0, 2)

    assert by['S5']['verdict'] == 'FAIL' and by['S5']['pass_pf'] is False
    assert by['S3']['verdict'] == 'FAIL' and by['S3']['pass_sharpe'] is False

    # 거래 0 → 판정 불가
    assert by['S6']['has_data'] is False
    assert by['S6']['verdict'] == 'FAIL'


def test_evaluate_threshold_boundary():
    # OOS PF 가 정확히 임계값이면 PASS (>=)
    wf = {'IS': {'S4': _m(1.5, 1.0, 20, 40)},
          'OOS': {'S4': _m(WF_OOS_MIN_PF, WF_OOS_MIN_SHARPE, 1, 25)}}
    row = mr.evaluate(wf, ['S4'])[0]
    assert row['verdict'] == 'PASS'


def test_format_message_flags_failures():
    rows = mr.evaluate(_wf_fixture(), ['S4', 'S5', 'S3', 'S6'])
    msg = mr.format_message(rows, datetime(2026, 8, 1, tzinfo=timezone.utc), '2026-08-01')
    assert 'WFO 월간 리포트' in msg
    assert '재최적화 후보' in msg
    # FAIL 전략은 후보에 포함, PASS 는 미포함
    assert 'S5' in msg and 'S3' in msg
    assert '1/3 PASS' in msg  # S4 pass, S5/S3 fail, S6 데이터없음 → 평가 3건 중 1 PASS


def test_format_message_all_pass():
    wf = {'IS': {'S4': _m(2.0, 1.0, 40, 60)},
          'OOS': {'S4': _m(1.5, 0.5, 10, 30)}}
    rows = mr.evaluate(wf, ['S4'])
    msg = mr.format_message(rows, datetime.now(timezone.utc), '2026-08-01')
    assert '전 전략 OOS 기준 통과' in msg


# ── reoptimize.override_params / current_params ────────────────

def test_override_params_restores():
    key = 'S4_RSI_ENTRY'
    original = getattr(strat, key)
    sentinel = original + 7
    with ro.override_params({key: sentinel}):
        assert getattr(strat, key) == sentinel
    assert getattr(strat, key) == original


def test_override_params_restores_on_error():
    key = 'S4_ATR_SL'
    original = getattr(strat, key)
    with pytest.raises(RuntimeError):
        with ro.override_params({key: 99.0}):
            assert getattr(strat, key) == 99.0
            raise RuntimeError('boom')
    assert getattr(strat, key) == original


def test_override_params_rejects_unknown():
    with pytest.raises(KeyError):
        with ro.override_params({'NOPE_NOT_A_PARAM': 1}):
            pass


def test_current_params_matches_module():
    cur = ro.current_params('S4')
    assert set(cur.keys()) == set(ro.GRIDS['S4'].keys())
    for k, v in cur.items():
        assert v == getattr(strat, k)


# ── reoptimize.pick_targets ────────────────────────────────────

def test_pick_targets_explicit():
    assert ro.pick_targets('s4, s5') == ['S4', 'S5']


def test_pick_targets_from_latest_fail(tmp_path, monkeypatch):
    latest = {
        'summary': [
            {'sid': 'S4', 'has_data': True, 'verdict': 'PASS'},
            {'sid': 'S5', 'has_data': True, 'verdict': 'FAIL'},
            {'sid': 'S6', 'has_data': False, 'verdict': 'FAIL'},
        ]
    }
    (tmp_path / 'wfo_latest.json').write_text(json.dumps(latest), encoding='utf-8')
    monkeypatch.setattr(ro, 'SPOT_RESULTS_DIR', tmp_path)
    # has_data=True 인 FAIL 만 (S5). S6 는 데이터 없어 제외.
    assert ro.pick_targets('') == ['S5']


def test_pick_targets_no_fail_returns_empty(tmp_path, monkeypatch):
    latest = {'summary': [{'sid': 'S4', 'has_data': True, 'verdict': 'PASS'}]}
    (tmp_path / 'wfo_latest.json').write_text(json.dumps(latest), encoding='utf-8')
    monkeypatch.setattr(ro, 'SPOT_RESULTS_DIR', tmp_path)
    assert ro.pick_targets('') == []


# ── reoptimize.format_proposal ─────────────────────────────────

def test_format_proposal_accepted():
    props = [{
        'sid': 'S4', 'name': 'RSI Mean Reversion', 'accepted': True, 'reason': '',
        'current': {'S4_RSI_ENTRY': 30, 'S4_ATR_SL': 2.0},
        'proposed': {'S4_RSI_ENTRY': 25, 'S4_ATR_SL': 2.0},
        'baseline': {'is': _m(1.5, 1.0, 20, 40), 'oos': _m(0.9, 0.4, -5, 30)},
        'candidate': {'is': _m(2.1, 1.2, 30, 45), 'oos': _m(1.4, 0.6, 12, 28)},
    }]
    msg = ro.format_proposal(props, datetime.now(timezone.utc))
    assert '재최적화 제안' in msg
    assert 'S4_RSI_ENTRY = 30  →  25' in msg
    # 값이 바뀌지 않은 파라미터는 diff 에 안 나온다
    assert 'S4_ATR_SL' not in msg
    assert '과최적화 위험' in msg


def test_format_proposal_none_accepted():
    props = [{'sid': 'S5', 'accepted': False, 'reason': 'OOS 개선 없음'}]
    msg = ro.format_proposal(props, datetime.now(timezone.utc))
    assert '개선 제안 없음' in msg
    assert 'OOS 개선 없음' in msg


class TestPortfolioCaveat:
    """리포트는 이 수치가 '상한선'이라는 사실을 함께 알려야 한다.

    backtest_strategy 는 (전략 × 심볼) 단위로 독립 실행되므로 동시 포지션
    한도·슬롯당 최소자본·USDT 예비금을 반영하지 못한다 — 코드에는 한계로
    명시돼 있는데(atlas_spot_backtest 의 '포트폴리오 제약' 주석) 정작 판정을
    전달하는 리포트에는 단서가 없었다. PASS/FAIL로 전략 존폐를 정하는
    사람이 수치를 액면 그대로 믿게 된다.
    """

    def test_caveat_present_in_message(self):
        wf = {'IS': {'S4': _m(2.0, 1.0, 40, 60)},
              'OOS': {'S4': _m(1.5, 0.5, 10, 30)}}
        rows = mr.evaluate(wf, ['S4'])
        msg = mr.format_message(rows, datetime.now(timezone.utc), '2026-08-13', 564.0)
        assert '낙관적' in msg, '결과를 상한선으로 읽어야 한다는 단서가 없다'

    def test_slots_computed_from_equity(self):
        """추상적 경고가 아니라 **현재 한도**를 숫자로 보여야 판단에 쓸 수 있다."""
        txt = mr.portfolio_caveat(564.0)
        assert '15슬롯' in txt, txt      # min(15, 564//20=28) = 15 (상한)

    def test_small_account_slots_are_capital_bound(self):
        txt = mr.portfolio_caveat(217.0)
        assert '10슬롯' in txt, txt      # min(15, 217//20=10) = 10

    def test_unknown_equity_falls_back_to_config_limit(self):
        """자산 조회가 실패해도 단서 자체는 남아야 한다."""
        txt = mr.portfolio_caveat(None)
        assert '낙관적' in txt and '15슬롯' in txt

    def test_zero_equity_does_not_crash(self):
        assert '낙관적' in mr.portfolio_caveat(0.0)

"""
ATLAS — BNB 수수료 잔고 감시
================================
바이낸스의 "BNB로 수수료 지불"은 **잔고가 있을 때만** 동작한다. 비면
토글이 켜져 있어도 조용히 기초자산 차감으로 되돌아간다. 그 순간:

  ① 25% 할인 상실 — 왕복 0.15% → 0.2% (1R 잠식 7% → 8%)
  ② **보호주문이 깨진다** — 매수 체결량에서 수수료만큼 기초자산이 빠져
     '기록 수량 > 실제 보유량'이 되고, 손절 주문이 -2010으로 거부된다

②가 훨씬 비싸다. 할인 몇 %가 아니라 **손절이 안 걸린 포지션**이 생긴다.
실제로 라이브에서 ADA·RIF·FET 3종목이 이 상태가 됐고, DB 수량을 손으로
교정해야만 해소됐다(3fa6074).

토글이 켜져 있어도 안심할 수 없다는 것이 핵심이다 — 잔고를 봐야 한다.

실행:
  pytest tests/test_bnb_fee_balance.py -v
"""

import os
import sys
import time
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_config as cfg
import atlas_spot_main as sm


@pytest.fixture
def tg(monkeypatch):
    sent = []
    monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
    monkeypatch.setattr(sm, '_bnb_alert', {'at': 0.0})

    class _Ex:
        def fetch_ticker(self, sym):
            return {'last': 600.0}
    monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
    return sent


def _bal(bnb_qty):
    return {'BNB': {'total': bnb_qty}, 'USDT': {'total': 500.0}}


class TestAlerting:
    def test_empty_balance_alerts(self, tg):
        sm._check_bnb_fee_balance(_bal(0.0))
        assert len(tg) == 1
        assert 'BNB' in tg[0]

    def test_low_balance_alerts(self, tg):
        # $5 미만 → 0.008 BNB × $600 = $4.8
        sm._check_bnb_fee_balance(_bal(0.008))
        assert len(tg) == 1

    def test_sufficient_balance_is_silent(self, tg):
        sm._check_bnb_fee_balance(_bal(0.05))     # $30
        assert tg == []

    def test_alert_mentions_the_real_consequence(self, tg):
        """'할인 못 받음'만 알리면 급하지 않게 느껴진다. 진짜 위험은
        손절 주문이 거부되는 것이다."""
        sm._check_bnb_fee_balance(_bal(0.0))
        assert '손절' in tg[0], f'실제 결과가 안 적혀 있다: {tg[0]}'

    def test_threshold_matches_config(self, tg):
        px = 600.0
        just_under = (cfg.SPOT_BNB_MIN_USD - 0.5) / px
        just_over = (cfg.SPOT_BNB_MIN_USD + 0.5) / px
        sm._check_bnb_fee_balance(_bal(just_over))
        assert tg == []
        sm._check_bnb_fee_balance(_bal(just_under))
        assert len(tg) == 1


class TestNoSpam:
    def test_repeat_within_window_is_suppressed(self, tg):
        sm._check_bnb_fee_balance(_bal(0.0))
        sm._check_bnb_fee_balance(_bal(0.0))
        sm._check_bnb_fee_balance(_bal(0.0))
        assert len(tg) == 1, '잔고폴러는 60초마다 도는데 매번 알리면 무시하게 된다'

    def test_alerts_again_after_window(self, tg, monkeypatch):
        sm._check_bnb_fee_balance(_bal(0.0))
        monkeypatch.setattr(
            sm, '_bnb_alert',
            {'at': time.time() - cfg.SPOT_BNB_ALERT_HOURS * 3600 - 1})
        sm._check_bnb_fee_balance(_bal(0.0))
        assert len(tg) == 2


class TestFailSafe:
    def test_missing_bnb_key_is_treated_as_empty(self, tg):
        sm._check_bnb_fee_balance({'USDT': {'total': 500.0}})
        assert len(tg) == 1, 'BNB 키가 아예 없으면 잔고 0과 같다'

    def test_ticker_failure_does_not_crash_poller(self, tg, monkeypatch):
        class _Bad:
            def fetch_ticker(self, sym):
                raise RuntimeError('네트워크')
        monkeypatch.setattr(sm, '_get_ex', lambda: _Bad())
        sm._check_bnb_fee_balance(_bal(0.05))     # 예외가 새어나가면 안 된다

    def test_malformed_balance_is_safe(self, tg):
        for bad in ({'BNB': None}, {'BNB': {'total': 'x'}}, {}):
            sm._check_bnb_fee_balance(bad)


class TestWiring:
    def test_called_from_balance_poller_path(self):
        """`_get_spot_equity`가 이미 전 잔고를 가져오므로 추가 API 호출이 없다."""
        src = Path(sm.__file__).read_text()
        body = src[src.index('def _get_spot_equity'):]
        assert '_check_bnb_fee_balance(bal)' in body[:800]

    def test_config_constants_exist(self):
        assert cfg.SPOT_BNB_MIN_USD > 0
        assert cfg.SPOT_BNB_ALERT_HOURS > 0

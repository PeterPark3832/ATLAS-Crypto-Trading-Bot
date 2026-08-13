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

import time
from pathlib import Path


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


# ══════════════════════════════════════════════════════════════
#  자본 규모에 따른 확장 (2026-08)
# ══════════════════════════════════════════════════════════════

class TestThresholdScalesWithConsumption:
    """고정 달러 임계값은 자본이 커지면 무의미해진다.

    자본 $10,000이면 명목가도 수수료도 10배라, $5는 **하루치**가 된다.
    경고를 받아도 반응할 시간이 없다. 그래서 임계값을 '남은 소모일수'로
    잡는다 — 자본이 아니라 실제 소모 속도를 따라간다.
    """

    def test_high_consumption_raises_threshold(self):
        low = sm._bnb_alert_threshold(6.0)      # 월 $6 (소액 계좌)
        high = sm._bnb_alert_threshold(600.0)   # 월 $600 (대형 계좌)
        assert high > low * 10

    def test_threshold_equals_configured_days(self):
        monthly = 300.0
        expect = monthly / 30.0 * cfg.SPOT_BNB_MIN_DAYS
        assert sm._bnb_alert_threshold(monthly) == pytest.approx(expect)

    def test_small_account_uses_dollar_floor(self):
        """소모가 미미하면 임계값이 몇 센트가 된다 — 하한이 필요하다."""
        assert sm._bnb_alert_threshold(0.5) == pytest.approx(cfg.SPOT_BNB_MIN_USD)
        assert sm._bnb_alert_threshold(0.0) == pytest.approx(cfg.SPOT_BNB_MIN_USD)

    def test_gives_configured_days_of_warning(self):
        """임계값에 도달한 시점에 남은 여유가 설정한 일수여야 한다."""
        for monthly in (30.0, 300.0, 3000.0):
            thr = sm._bnb_alert_threshold(monthly)
            days = thr / (monthly / 30.0)
            assert days >= cfg.SPOT_BNB_MIN_DAYS - 1e-6


class TestRefillCapacityLimit:
    """자동 충전에는 속도 상한이 있다 — 1회 상한 ÷ 쿨다운.
    자본이 커지면 소모가 이걸 넘어서고, 그러면 켜 뒀어도 잔고가 계속 준다.
    **켜 뒀다는 사실이 오히려 방심을 만들기 때문에** 감지해서 알려야 한다."""

    def test_small_account_keeps_up(self):
        _, _, ok = sm._bnb_refill_capacity(monthly_usd=6.0, equity=500.0)
        assert ok

    def test_reports_shortfall_numbers(self):
        cap, use, ok = sm._bnb_refill_capacity(monthly_usd=6.0, equity=500.0)
        assert cap > 0 and use > 0
        assert ok == (cap >= use)

    def test_capacity_scales_with_equity(self):
        small, _, _ = sm._bnb_refill_capacity(1000.0, 1_000.0)
        large, _, _ = sm._bnb_refill_capacity(1000.0, 500_000.0)
        assert large > small, '자산이 크면 1회 상한도 커져 보충 속도가 오른다'

    def test_shortfall_is_detectable(self, monkeypatch):
        """쿨다운을 길게 잡으면 반드시 못 따라가는 지점이 생긴다."""
        monkeypatch.setattr(sm, 'SPOT_BNB_REFILL_COOLDOWN_H', 24 * 30)
        _, _, ok = sm._bnb_refill_capacity(monthly_usd=5000.0, equity=1_000.0)
        assert ok is False

    def test_warns_once_per_window(self, monkeypatch):
        sent = []
        monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', True)
        monkeypatch.setattr(sm, 'SPOT_BNB_REFILL_COOLDOWN_H', 24 * 30)
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_capacity_alert', {'at': 0.0})
        sm._warn_if_refill_cannot_keep_up(5000.0, 1_000.0)
        sm._warn_if_refill_cannot_keep_up(5000.0, 1_000.0)
        assert len(sent) == 1
        assert '따라가지 못' in sent[0] or '소모' in sent[0]

    def test_silent_when_refill_disabled(self, monkeypatch):
        """자동 충전이 꺼져 있으면 용량 경고는 의미가 없다."""
        sent = []
        monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', False)
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_capacity_alert', {'at': 0.0})
        sm._warn_if_refill_cannot_keep_up(5000.0, 1_000.0)
        assert sent == []


class TestPriceCache:
    """잔고폴러는 60초마다 돈다. 매번 티커를 치면 하루 1,440회를 낭비한다."""

    def test_price_is_cached(self, monkeypatch):
        calls = []

        class _Ex:
            def fetch_ticker(self, s):
                calls.append(1)
                return {'last': 600.0}
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        monkeypatch.setattr(sm, '_bnb_price', {'usd': 0.0, 'at': 0.0})
        for _ in range(5):
            sm._bnb_price_usd()
        assert len(calls) == 1

    def test_cache_expires(self, monkeypatch):
        calls = []

        class _Ex:
            def fetch_ticker(self, s):
                calls.append(1)
                return {'last': 600.0}
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        monkeypatch.setattr(sm, '_bnb_price',
                            {'usd': 600.0,
                             'at': time.time() - cfg.SPOT_BNB_PRICE_TTL_SEC - 1})
        sm._bnb_price_usd()
        assert len(calls) == 1

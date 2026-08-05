"""
ATLAS — BNB 자동 충전 안전장치
================================
이건 봇이 **운영자 돈으로 스스로 매수하는 유일한 기능**이다. 매매와 무관한
지출이라 잘못 돌면 USDT가 조용히 BNB로 빠져나간다. 그래서 상한을 여러 겹
걸고, 여기서 **각 겹을 따로** 시험한다 — 한 겹만 검사하면 나머지가 뚫려도
모른다.

  ① 기본 OFF                    ② 드라이런에서 실행 금지
  ③ 1회 매수 상한               ④ 재매수 쿨다운
  ⑤ USDT 예비금 침범 금지        ⑥ 최소 매수액 미만이면 아예 안 삼

필요량은 **실제 거래 이력**(최근 30일 명목가 합 × 수수료율 × 2)에서
추정한다. 이력이 없으면 최소 금액만 산다 — 과대 추정으로 매매 자금을
묶는 것보다 적게 사고 다음에 또 사는 편이 낫다.

실행:
  pytest tests/test_bnb_autorefill.py -v
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

EQUITY = 1000.0
USDT_FREE = 500.0          # 예비금 $100(10%)을 크게 넘는 여유


@pytest.fixture
def on(monkeypatch):
    """자동 충전 ON + 쿨다운 해제 + 소모량 추정 고정."""
    monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', True)
    monkeypatch.setattr(sm, '_bnb_refill', {'at': 0.0})
    monkeypatch.setattr(sm, '_state', {'dry_run': False,
                                       'usdt_balance': USDT_FREE,
                                       'equity': EQUITY})
    monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 8.0)
    return monkeypatch


def _amt(current=0.0, usdt=USDT_FREE, equity=EQUITY):
    return sm._bnb_refill_amount(current, usdt, equity)


# ══════════════════════════════════════════════════════════════
#  ① 기본은 사지 않는다
# ══════════════════════════════════════════════════════════════

class TestDefaultOff:
    def test_config_default_is_off(self):
        """돈을 쓰는 기능이라 기본값이 OFF여야 한다."""
        assert cfg.SPOT_BNB_AUTO_REFILL is False

    def test_disabled_never_buys(self, monkeypatch):
        monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', False)
        amt, why = _amt(current=0.0)
        assert amt == 0.0 and '꺼짐' in why

    def test_dry_run_never_buys(self, on, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'dry_run': True,
                                           'usdt_balance': USDT_FREE,
                                           'equity': EQUITY})
        amt, why = _amt(current=0.0)
        assert amt == 0.0 and '드라이런' in why

    def test_sufficient_balance_never_buys(self, on):
        amt, why = _amt(current=cfg.SPOT_BNB_MIN_USD + 1)
        assert amt == 0.0 and '충분' in why


# ══════════════════════════════════════════════════════════════
#  ③④⑤⑥ 각 상한을 개별로 확인
# ══════════════════════════════════════════════════════════════

class TestSpendingCaps:
    def test_respects_per_purchase_cap(self, on, monkeypatch):
        """소모량이 아무리 커도 1회 상한을 넘지 않는다."""
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 10_000.0)
        amt, _ = _amt(current=0.0, usdt=100_000.0, equity=200_000.0)
        assert amt == pytest.approx(cfg.SPOT_BNB_MAX_BUY_USD)

    def test_cooldown_blocks_repeat(self, on, monkeypatch):
        monkeypatch.setattr(sm, '_bnb_refill', {'at': time.time()})
        amt, why = _amt(current=0.0)
        assert amt == 0.0 and '쿨다운' in why

    def test_cooldown_expires(self, on, monkeypatch):
        past = time.time() - cfg.SPOT_BNB_REFILL_COOLDOWN_H * 3600 - 1
        monkeypatch.setattr(sm, '_bnb_refill', {'at': past})
        amt, _ = _amt(current=0.0)
        assert amt > 0

    def test_reserve_is_never_touched(self, on):
        """예비금을 깎아 BNB를 사면 매매 자금을 잠식하는 본말전도다."""
        reserve = EQUITY * cfg.SPOT_RESERVE_PCT
        amt, why = _amt(current=0.0, usdt=reserve + 1.0, equity=EQUITY)
        assert amt == 0.0 and '예비금' in why

    def test_buys_when_above_reserve(self, on):
        reserve = EQUITY * cfg.SPOT_RESERVE_PCT
        amt, _ = _amt(current=0.0,
                      usdt=reserve + cfg.SPOT_BNB_MAX_BUY_USD + 1, equity=EQUITY)
        assert amt > 0

    def test_below_min_buy_does_not_trade(self, on, monkeypatch):
        """푼돈을 사면 수수료·NOTIONAL만 낭비한다."""
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 0.5)
        amt, why = _amt(current=0.0)
        assert amt == 0.0 and '최소' in why

    def test_amount_never_exceeds_spendable(self, on):
        for usdt in (150.0, 200.0, 400.0, 5000.0):
            amt, _ = _amt(current=0.0, usdt=usdt, equity=EQUITY)
            assert amt <= max(usdt - EQUITY * cfg.SPOT_RESERVE_PCT, 0.0) + 1e-9


# ══════════════════════════════════════════════════════════════
#  필요량 추정
# ══════════════════════════════════════════════════════════════

class TestSizing:
    def test_scales_with_trading_volume(self, on, monkeypatch):
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 3.0)
        small, _ = _amt(current=0.0)
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 9.0)
        big, _ = _amt(current=0.0)
        assert big > small

    def test_targets_configured_months(self, on, monkeypatch):
        monthly = 7.0
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: monthly)
        amt, _ = _amt(current=0.0)
        expect = min(monthly * cfg.SPOT_BNB_TARGET_MONTHS, cfg.SPOT_BNB_MAX_BUY_USD)
        assert amt == pytest.approx(expect)

    def test_existing_balance_is_deducted(self, on, monkeypatch):
        """이미 가진 만큼은 빼고 산다.

        1회 상한에 걸리지 않는 구간에서 봐야 한다 — 둘 다 상한에 걸리면
        같은 값이 나와 이 성질을 잴 수 없다.
        """
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 8.0)
        none_held, _ = _amt(current=0.0)
        some_held, _ = _amt(current=4.0)
        assert none_held < cfg.SPOT_BNB_MAX_BUY_USD, '상한에 걸리지 않아야 한다'
        assert some_held == pytest.approx(none_held - 4.0)

    def test_no_history_buys_minimum_only(self, on, monkeypatch):
        """이력이 없을 때 크게 사면 매매 자금이 BNB로 묶인다."""
        monkeypatch.setattr(sm, '_estimate_monthly_bnb_usd', lambda: 0.0)
        amt, _ = _amt(current=0.0)
        assert amt == pytest.approx(cfg.SPOT_BNB_MIN_BUY_USD)


class TestEstimateFromHistory:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime, timedelta, timezone
        p = tmp_path / 'spot.db'
        conn = sqlite3.connect(str(p))
        conn.execute('CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, '
                     'cost_usdt REAL, exit_ts TEXT, dry_run INTEGER DEFAULT 0)')
        now = datetime.now(timezone.utc)
        rows = [(100.0, now.isoformat(), 0),
                (200.0, (now - timedelta(days=10)).isoformat(), 0),
                (500.0, (now - timedelta(days=90)).isoformat(), 0),   # 창 밖
                (900.0, now.isoformat(), 1)]                          # 드라이런
        conn.executemany('INSERT INTO spot_trades (cost_usdt, exit_ts, dry_run) '
                         'VALUES (?,?,?)', rows)
        conn.commit(); conn.close()
        monkeypatch.setattr(sm, 'SPOT_DB_FILE', p)
        monkeypatch.setattr(sm, '_detect_fee_rate', lambda: 0.00075)
        return p

    def test_uses_recent_live_trades_only(self, db):
        # (100 + 200) × 0.00075 × 2 = 0.45
        assert sm._estimate_monthly_bnb_usd() == pytest.approx(0.45)

    def test_db_failure_returns_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'nope.db')
        assert sm._estimate_monthly_bnb_usd() >= 0.0


# ══════════════════════════════════════════════════════════════
#  실행 경로
# ══════════════════════════════════════════════════════════════

class TestExecution:
    @pytest.fixture
    def ex(self, on, monkeypatch):
        calls = []
        sent = []

        class _Ex:
            def create_order(self, sym, typ, side, amt, price, params):
                calls.append((sym, typ, side, params))
                return {'filled': 0.05}
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        return calls, sent

    def test_places_quote_denominated_market_buy(self, ex):
        calls, _ = ex
        sm._refill_bnb(0.0)
        assert len(calls) == 1
        sym, typ, side, params = calls[0]
        assert (sym, typ, side) == ('BNB/USDT', 'market', 'buy')
        assert 'quoteOrderQty' in params, (
            '수량이 아니라 **금액**으로 사야 상한이 정확히 지켜진다')
        assert params['quoteOrderQty'] <= cfg.SPOT_BNB_MAX_BUY_USD

    def test_notifies_operator(self, ex):
        _, sent = ex
        sm._refill_bnb(0.0)
        assert len(sent) == 1 and 'BNB' in sent[0]

    def test_cooldown_starts_on_success(self, ex):
        """성공 후 쿨다운이 안 걸리면 폴러가 60초마다 계속 사들인다."""
        sm._refill_bnb(0.0)
        assert sm._bnb_refill['at'] > 0.0

    def test_second_call_is_blocked_after_success(self, ex):
        """종단 확인 — 연속 호출에도 주문은 한 번뿐이어야 한다."""
        calls, _ = ex
        sm._refill_bnb(0.0)
        sm._refill_bnb(0.0)
        sm._refill_bnb(0.0)
        assert len(calls) == 1, f'{len(calls)}회 매수 — 반복 지출 위험'

    def test_cooldown_does_not_start_on_failure(self, on, monkeypatch):
        """실패로 쿨다운을 걸면 다음 기회까지 72시간을 그냥 버린다."""
        class _Bad:
            def create_order(self, *a, **k):
                raise RuntimeError('잔고 부족')
        monkeypatch.setattr(sm, '_get_ex', lambda: _Bad())
        monkeypatch.setattr(sm, '_tg', lambda m: None)
        sm._refill_bnb(0.0)
        assert sm._bnb_refill['at'] == 0.0

    def test_order_failure_does_not_crash(self, on, monkeypatch):
        class _Bad:
            def create_order(self, *a, **k):
                raise RuntimeError('네트워크')
        monkeypatch.setattr(sm, '_get_ex', lambda: _Bad())
        monkeypatch.setattr(sm, '_tg', lambda m: None)
        sm._refill_bnb(0.0)          # 예외가 새면 잔고폴러가 죽는다

    def test_no_order_when_amount_is_zero(self, on, monkeypatch):
        calls = []

        class _Ex:
            def create_order(self, *a, **k):
                calls.append(1)
                return {}
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', False)
        sm._refill_bnb(0.0)
        assert calls == []


class TestWiring:
    def test_refill_is_attempted_before_alerting(self):
        """충전에 성공하면 경고가 자연히 멎어야 한다 — 순서가 중요하다."""
        src = Path(sm.__file__).read_text()
        body = src[src.index('def _check_bnb_fee_balance'):]
        body = body[:body.index('def _estimate_monthly_bnb_usd')] \
            if 'def _estimate_monthly_bnb_usd' in body else body[:2000]
        assert '_refill_bnb(usd)' in body

    def test_alert_path_survives_when_refill_off(self, monkeypatch):
        """자동 충전이 꺼져 있어도 경고는 그대로 나가야 한다."""
        sent = []
        monkeypatch.setattr(sm, 'SPOT_BNB_AUTO_REFILL', False)
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_alert', {'at': 0.0})

        class _Ex:
            def fetch_ticker(self, s):
                return {'last': 600.0}
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        sm._check_bnb_fee_balance({'BNB': {'total': 0.0}})
        assert len(sent) == 1

"""
ATLAS — BNB 할인 판정 (오탐 수정)
================================
운영자가 "Use BNB to pay fees"를 **켜 둔 상태**에서 봇이 "꺼져 있는 것
같습니다" 알림을 반복 발송했다. 원인:

  바이낸스 API(`tradeFee`, `commissionRates`)는 **VIP 등급 기본 요율**만
  돌려준다. BNB 할인은 **체결 시점**에 적용되므로 이 응답에 반영되지
  않는다 — 토글을 켜도 영원히 0.1%로 답한다.

즉 요율만 보고 판단하면 **구조적으로 항상 오탐**이다. 할인의 흔적은
개별 체결의 `commissionAsset`에만 남는다.

게다가 수수료율 재확인에 6시간 TTL을 넣으면서(e6c1c39) 그 오탐이
하루 1회에서 **4회로 늘었다.** 알림에 발송 간격이 없었기 때문이다.

실행:
  pytest tests/test_bnb_discount_detection.py -v
"""

import time


import pytest

import atlas_spot_config as cfg
import atlas_spot_main as sm


@pytest.fixture
def db(tmp_path, monkeypatch):
    import sqlite3
    p = tmp_path / 'spot.db'
    conn = sqlite3.connect(str(p))
    conn.execute('CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, '
                 'symbol TEXT, dry_run INTEGER DEFAULT 0)')
    conn.execute("INSERT INTO spot_trades (symbol, dry_run) VALUES ('ADAUSDT', 0)")
    conn.commit(); conn.close()
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', p)
    return p


def _ex(fills):
    class _E:
        def fetch_my_trades(self, sym, limit=None):
            return fills
    return _E()


class TestDiscountDetection:
    def test_bnb_commission_means_active(self, db, monkeypatch):
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _ex([{'fee': {'currency': 'BNB'}}]))
        assert sm._bnb_discount_active() is True

    def test_base_asset_commission_means_inactive(self, db, monkeypatch):
        """수수료가 기초자산에서 빠졌다 = 할인이 실제로 안 걸렸다."""
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _ex([{'fee': {'currency': 'ADA'}}]))
        assert sm._bnb_discount_active() is False

    def test_reads_info_commission_asset_fallback(self, db, monkeypatch):
        monkeypatch.setattr(sm, '_get_ex',
                            lambda: _ex([{'info': {'commissionAsset': 'BNB'}}]))
        assert sm._bnb_discount_active() is True

    def test_uses_most_recent_fill(self, db, monkeypatch):
        """오래된 체결이 아니라 **최근** 상태를 봐야 한다."""
        monkeypatch.setattr(sm, '_get_ex', lambda: _ex([
            {'fee': {'currency': 'ADA'}},      # 예전 — BNB 없던 시절
            {'fee': {'currency': 'BNB'}},      # 최근 — 충전 후
        ]))
        assert sm._bnb_discount_active() is True

    def test_no_trade_history_is_unknown(self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / 'empty.db'
        conn = sqlite3.connect(str(p))
        conn.execute('CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, '
                     'symbol TEXT, dry_run INTEGER DEFAULT 0)')
        conn.commit(); conn.close()
        monkeypatch.setattr(sm, 'SPOT_DB_FILE', p)
        assert sm._bnb_discount_active() is False

    def test_api_failure_does_not_crash(self, db, monkeypatch):
        class _Bad:
            def fetch_my_trades(self, *a, **k):
                raise RuntimeError('권한 없음')
        monkeypatch.setattr(sm, '_get_ex', lambda: _Bad())
        assert sm._bnb_discount_active() is False


class TestNoFalseAlarm:
    """이 클래스가 원래 버그를 고정한다 — 켜 둔 상태에서 경고가 나가면 실패."""

    @pytest.fixture
    def caught(self, monkeypatch, db):
        sent = []
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_fee_rate',
                            {'taker': sm.BT_SPOT_FEE, 'checked': False, 'at': 0.0})
        monkeypatch.setattr(sm, '_bnb_off_alert', {'at': 0.0})
        return sent

    def test_no_warning_when_discount_is_active(self, caught, monkeypatch):
        """API가 0.1%로 답해도 체결이 BNB면 경고하면 안 된다."""
        class _E:
            def fetch_trading_fee(self, s):
                return {'taker': 0.001}          # VIP 기본 요율 (할인 미반영)

            def fetch_my_trades(self, s, limit=None):
                return [{'fee': {'currency': 'BNB'}}]
        monkeypatch.setattr(sm, '_get_ex', lambda: _E())
        sm._detect_fee_rate()
        assert not any('BNB' in m for m in caught), (
            f'켜 둔 상태에서 경고가 나갔다: {caught}')

    def test_effective_rate_reflects_discount(self, caught, monkeypatch):
        """할인이 걸려 있으면 비용 모델도 25% 낮아져야 한다.
        아니면 진입 가드가 실제보다 비싸다고 믿어 멀쩡한 신호를 막는다."""
        class _E:
            def fetch_trading_fee(self, s):
                return {'taker': 0.001}

            def fetch_my_trades(self, s, limit=None):
                return [{'fee': {'currency': 'BNB'}}]
        monkeypatch.setattr(sm, '_get_ex', lambda: _E())
        assert sm._detect_fee_rate() == pytest.approx(0.001 * sm.BNB_FEE_DISCOUNT)

    def test_warns_when_genuinely_off(self, caught, monkeypatch):
        class _E:
            def fetch_trading_fee(self, s):
                return {'taker': 0.001}

            def fetch_my_trades(self, s, limit=None):
                return [{'fee': {'currency': 'ADA'}}]
        monkeypatch.setattr(sm, '_get_ex', lambda: _E())
        sm._detect_fee_rate()
        assert any('BNB' in m for m in caught)


class TestAlertRateLimit:
    """수수료율 재확인이 6시간 주기가 되면서, 간격 제한이 없는 경고는
    하루 4번 나가게 됐다. 반복되는 알림은 곧 무시되는 알림이다."""

    def test_second_warning_is_suppressed(self, monkeypatch):
        sent = []
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_off_alert', {'at': 0.0})
        sm._warn_bnb_discount_off()
        sm._warn_bnb_discount_off()
        sm._warn_bnb_discount_off()
        assert len(sent) == 1

    def test_warns_again_after_window(self, monkeypatch):
        sent = []
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_off_alert', {'at': 0.0})
        sm._warn_bnb_discount_off()
        monkeypatch.setattr(
            sm, '_bnb_off_alert',
            {'at': time.time() - cfg.SPOT_BNB_ALERT_HOURS * 3600 - 1})
        sm._warn_bnb_discount_off()
        assert len(sent) == 2

    def test_message_explains_the_real_cause(self, monkeypatch):
        """'토글을 켜세요'는 이미 켠 사람에게 무의미하다.
        진짜 원인(잔고 부족)을 알려야 행동으로 이어진다."""
        sent = []
        monkeypatch.setattr(sm, '_tg', lambda m: sent.append(m))
        monkeypatch.setattr(sm, '_bnb_off_alert', {'at': 0.0})
        sm._warn_bnb_discount_off()
        assert '잔고' in sent[0]

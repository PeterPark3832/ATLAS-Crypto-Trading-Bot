"""
ATLAS — 포지션 관리(_manage_position) 단위 테스트
================================
atlas_spot_main.py의 _manage_position이 SL/TP/시간청산/긴급청산 등
올바른 조건에서 올바른 사유로 _spot_sell을 호출하는지 검증합니다.
실제 매도 실행(_spot_sell)은 별도 테스트(test_order_execution.py)에서
다루므로, 여기서는 _spot_sell을 모킹해 "언제 어떤 사유로 호출되는가"만 검증합니다.

실행:
  pytest tests/test_position_management.py -v
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  공용 fixture
# ══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    """전송을 막고 **보낸 내용을 모은다**(알림 횟수 검증용)."""
    sent: list = []
    monkeypatch.setattr(sm, '_tg', lambda msg: sent.append(msg))
    return sent


@pytest.fixture(autouse=True)
def _reset_alert_throttle(monkeypatch):
    """알림 간격 제한은 모듈 전역이라 테스트 간에 남는다.

    초기화하지 않으면 앞선 테스트가 같은 (전략, 심볼)로 한 번 보냈다는
    이유로 뒤 테스트의 알림이 조용히 억제된다.
    """
    monkeypatch.setattr(sm, '_stop_alert_at', {})


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / 'test_spot.db'
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', db_file)
    sm.init_spot_db()
    yield db_file


@pytest.fixture
def _sell_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, '_spot_sell',
                         lambda strategy, symbol, ccxt_sym, pos, price, reason:
                         calls.append((strategy, symbol, price, reason)))
    return calls


@pytest.fixture(autouse=True)
def _clear_price_cache(monkeypatch):
    monkeypatch.setattr(sm, '_last_known_price', {})


def _set_price(monkeypatch, value):
    monkeypatch.setattr(sm, '_get_price', lambda ccxt_sym: value)


def _open_position(strategy='S4', symbol='BTCUSDT', entry=100.0, sl=90.0, tp=120.0,
                    qty=10.0, max_hold=0, entry_hours_ago=2.0, peak=None):
    sm._save_position(strategy, symbol, entry, sl, tp, qty, entry * qty,
                       0.02, 'sl_tp', max_hold, 'TRENDING_UP')
    if peak is not None:
        sm._update_position_sl(strategy, symbol, sl, peak)
    if entry_hours_ago:
        ts = (datetime.now(timezone.utc) - timedelta(hours=entry_hours_ago)).isoformat()
        with sm._db_lock, sm._db_conn() as conn:
            conn.execute('UPDATE spot_positions SET entry_ts=? WHERE strategy=? AND symbol=?',
                         (ts, strategy, symbol))


# ══════════════════════════════════════════════════════════════
#  기본 동작
# ══════════════════════════════════════════════════════════════

class TestManagePositionBasics:
    def test_no_position_does_nothing(self, monkeypatch, _sell_calls):
        _set_price(monkeypatch, 100.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []


# ══════════════════════════════════════════════════════════════
#  가격 조회 실패 / 긴급 청산
# ══════════════════════════════════════════════════════════════

class TestEmergencyExit:
    def test_price_unavailable_no_cache_does_not_sell(self, monkeypatch, _sell_calls):
        _open_position()
        _set_price(monkeypatch, 0.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []

    def test_price_unavailable_fresh_cache_waits(self, monkeypatch, _sell_calls):
        """캐시가 있지만 TTL(2분) 이내면 아직 긴급청산하지 않음."""
        _open_position()
        sm._last_known_price['BTC/USDT'] = (100.0, time.time())
        _set_price(monkeypatch, 0.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []

    def test_price_unavailable_stale_cache_triggers_emergency_sell(self, monkeypatch, _sell_calls):
        _open_position(entry=100.0)
        sm._last_known_price['BTC/USDT'] = (100.0, time.time() - sm._PRICE_CACHE_TTL - 1)
        _set_price(monkeypatch, 0.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert len(_sell_calls) == 1
        assert _sell_calls[0][3] == 'EMERGENCY'
        assert _sell_calls[0][2] == pytest.approx(100.0 * 0.99)

    def test_exchange_stop_present_blocks_emergency_sell(self, monkeypatch,
                                                         _sell_calls, _no_telegram):
        """거래소 스탑이 있으면 던지지 않는다.

        이 경로의 목적은 'SL이 작동하지 못하는 상태' 방지인데, 거래소 스탑이
        걸려 있으면 SL은 봇의 시야와 무관하게 거래소가 집행한다.
        오히려 _spot_sell 은 매도 **전에 그 스탑을 취소**하므로, 이어지는
        시장가 매도가 실패하면(가격 조회를 막은 그 장애로 실패하기 쉽다)
        보호가 통째로 사라진 채 포지션만 남는다 — 의도와 정반대다.
        """
        _open_position(entry=100.0)
        sm._update_position_order_id('S4', 'BTCUSDT', 'STOP-1')
        sm._last_known_price['BTC/USDT'] = (100.0, time.time() - sm._PRICE_CACHE_TTL - 1)
        _set_price(monkeypatch, 0.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == [], (
            '거래소 보호주문이 있는데 긴급 청산했다 — 스탑을 취소하고 '
            '장애 중 시장가로 던지는 최악의 조합이 된다')

    def test_hold_is_reported_once(self, monkeypatch, _sell_calls, _no_telegram):
        """보류는 알려야 하지만 5분마다 반복하면 알림이 마비된다."""
        _open_position(entry=100.0)
        sm._update_position_order_id('S4', 'BTCUSDT', 'STOP-1')
        sm._last_known_price['BTC/USDT'] = (100.0, time.time() - sm._PRICE_CACHE_TTL - 1)
        _set_price(monkeypatch, 0.0)
        for _ in range(4):
            sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert len(_no_telegram) == 1, f'알림 {len(_no_telegram)}건 — 반복 발송'

    def test_no_protection_still_liquidates(self, monkeypatch, _sell_calls,
                                            _no_telegram):
        """보호가 없으면 기존대로 청산한다 — 가드가 과하게 막으면 안 된다."""
        _open_position(entry=100.0)
        sm._last_known_price['BTC/USDT'] = (100.0, time.time() - sm._PRICE_CACHE_TTL - 1)
        _set_price(monkeypatch, 0.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert len(_sell_calls) == 1 and _sell_calls[0][3] == 'EMERGENCY'


# ══════════════════════════════════════════════════════════════
#  SL / TP
# ══════════════════════════════════════════════════════════════

class TestSlTp:
    def test_price_below_sl_triggers_sell(self, monkeypatch, _sell_calls):
        _open_position(entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 89.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == [('S4', 'BTCUSDT', 89.0, 'SL')]

    def test_sl_breach_below_notional_keeps_position(self, monkeypatch, _sell_calls):
        """SL가 발생했지만 포지션 가치가 최소주문금액 미달이면 매도 시도하지 않음."""
        _open_position(entry=100.0, sl=90.0, tp=120.0, qty=0.01)  # 가치 ~$0.89
        _set_price(monkeypatch, 89.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []
        assert sm._load_position('S4', 'BTCUSDT') is not None

    def test_price_above_tp_triggers_sell(self, monkeypatch, _sell_calls):
        _open_position(entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 121.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == [('S4', 'BTCUSDT', 121.0, 'TP')]

    def test_sl_checked_before_tp(self, monkeypatch, _sell_calls):
        """SL과 TP가 동시에 깨지는 비정상 상황에선 SL을 우선 처리."""
        _open_position(entry=100.0, sl=90.0, tp=80.0, qty=10.0)  # tp<sl인 비정상 데이터
        _set_price(monkeypatch, 85.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls[0][3] == 'SL'


# ══════════════════════════════════════════════════════════════
#  추가 청산 조건 (exit_fn / CROSS)
# ══════════════════════════════════════════════════════════════

class TestExitFn:
    def test_exit_fn_true_triggers_cross_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S3', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 105.0)
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', lambda df, i: True)
        df = pd.DataFrame({'close': [105.0]})
        sm._manage_position('S3', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert _sell_calls == [('S3', 'BTCUSDT', 105.0, 'CROSS')]

    def test_exit_fn_false_does_not_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S3', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 105.0)
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', lambda df, i: False)
        df = pd.DataFrame({'close': [105.0]})
        sm._manage_position('S3', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert _sell_calls == []

    def test_exit_fn_exception_is_swallowed(self, monkeypatch, _sell_calls):
        """exit_fn에서 예외가 나도 _manage_position이 죽지 않고 다음 체크로 진행."""
        _open_position(strategy='S3', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 105.0)

        def _boom(df, i):
            raise ValueError('broken indicator')
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S3', _boom)
        df = pd.DataFrame({'close': [105.0]})
        sm._manage_position('S3', 'BTCUSDT', 'BTC/USDT', df, 0)  # 예외 없이 정상 종료
        assert _sell_calls == []

    def test_s6_exit_fn_receives_entry_price(self, monkeypatch, _sell_calls):
        """S6은 exit_fn(df, i, entry) 형태로 호출됨."""
        _open_position(strategy='S6', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 105.0)
        received = {}

        def _check(df, i, entry):
            received['entry'] = entry
            return True
        monkeypatch.setitem(sm.EXIT_CHECK_FUNCS, 'S6', _check)
        df = pd.DataFrame({'close': [105.0]})
        sm._manage_position('S6', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert received['entry'] == 100.0
        assert _sell_calls[0][3] == 'CROSS'


# ══════════════════════════════════════════════════════════════
#  시간 기반 청산
# ══════════════════════════════════════════════════════════════

class TestTimeExit:
    def test_max_hold_reached_triggers_time_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0,
                        max_hold=2, entry_hours_ago=60.0)  # S4는 1D(24h/bar) → 60h=2bars
        _set_price(monkeypatch, 105.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == [('S4', 'BTCUSDT', 105.0, 'TIME')]

    def test_below_max_hold_does_not_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0,
                        max_hold=5, entry_hours_ago=24.0)  # 1바만 경과
        _set_price(monkeypatch, 105.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []

    def test_max_hold_zero_disables_time_exit(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0,
                        max_hold=0, entry_hours_ago=1000.0)
        _set_price(monkeypatch, 105.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        assert _sell_calls == []


# ══════════════════════════════════════════════════════════════
#  전략별 특수 청산 (S4 BB_MID, S5 BB_UPPER 실시간 TP)
# ══════════════════════════════════════════════════════════════

class TestStrategySpecificExits:
    def test_s4_bb_mid_reached_triggers_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 105.0)
        df = pd.DataFrame({'bb_mid': [104.0]})
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert _sell_calls == [('S4', 'BTCUSDT', 105.0, 'BB_MID')]

    def test_s4_below_bb_mid_does_not_sell(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0)
        _set_price(monkeypatch, 100.0)
        df = pd.DataFrame({'bb_mid': [110.0]})
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert _sell_calls == []

    def test_s5_live_bb_upper_updates_tp_and_used_for_check(self, monkeypatch, _sell_calls):
        """S5는 bb_upper를 실시간 TP로 사용 — DB의 TP보다 낮은 신규 bb_upper를
        가격이 넘으면 즉시 매도되어야 함."""
        _open_position(strategy='S5', entry=100.0, sl=90.0, tp=130.0, qty=10.0)
        _set_price(monkeypatch, 106.0)
        df = pd.DataFrame({'bb_upper': [105.0]})  # 원래 DB tp(130)보다 훨씬 낮음
        sm._manage_position('S5', 'BTCUSDT', 'BTC/USDT', df, 0)
        assert _sell_calls == [('S5', 'BTCUSDT', 106.0, 'TP')]
        # bb_upper 갱신이 DB에도 반영됐는지 (포지션이 삭제되지 않았다면 확인)
        # 매도가 모킹되어 있으므로 포지션은 남아 있고 tp는 갱신되어 있어야 함
        pos = sm._load_position('S5', 'BTCUSDT')
        assert float(pos['tp']) == pytest.approx(105.0)


# ══════════════════════════════════════════════════════════════
#  Peak 업데이트 (트레일링 기준점)
# ══════════════════════════════════════════════════════════════

class TestPeakUpdate:
    def test_new_high_updates_peak(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0, peak=100.0)
        _set_price(monkeypatch, 110.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        pos = sm._load_position('S4', 'BTCUSDT')
        assert float(pos['peak_price']) == pytest.approx(110.0)
        assert _sell_calls == []

    def test_lower_price_does_not_update_peak(self, monkeypatch, _sell_calls):
        _open_position(strategy='S4', entry=100.0, sl=90.0, tp=120.0, qty=10.0, peak=110.0)
        _set_price(monkeypatch, 105.0)
        sm._manage_position('S4', 'BTCUSDT', 'BTC/USDT', None, 0)
        pos = sm._load_position('S4', 'BTCUSDT')
        assert float(pos['peak_price']) == pytest.approx(110.0)

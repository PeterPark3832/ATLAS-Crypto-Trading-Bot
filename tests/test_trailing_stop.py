"""
ATLAS — 추적 손절 (기본 OFF)
================================
`peak_price`는 이미 기록되고 있었지만 손절을 따라 올리는 데 쓰이지 않아
효과가 0이었다(같은 sl을 다시 쓰기만 함). 추세추종은 청산에서 수익이
갈리므로 기능을 살리되, **검증되지 않은 매매 변경**이라 기본값은 OFF다.

가장 중요한 불변식: **R배수의 분모는 진입 시점의 위험(orig_sl)이어야 한다.**
추적으로 sl이 올라간 뒤 그 값을 분모로 쓰면 R이 부풀고, 그 pnl_r이
Kelly·건강도·avg_r·비용 가드까지 연쇄 오염시킨다.

실행:
  pytest tests/test_trailing_stop.py -v
"""

import time
from pathlib import Path


import pytest

import atlas_rules as rules
import atlas_spot_main as sm


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(sm, '_tg', lambda msg: sent.append(msg))
    return sent


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'SPOT_DB_FILE', tmp_path / 'trail.db')
    sm.init_spot_db()


@pytest.fixture
def _on(monkeypatch):
    # trailing_sl 이 atlas_rules 로 이사해 호출 시점 조회처도 rules 전역이다
    monkeypatch.setattr(rules, 'SPOT_TRAIL_ENABLED', True)
    monkeypatch.setattr(rules, 'SPOT_TRAIL_ACTIVATE_R', 1.0)
    monkeypatch.setattr(rules, 'SPOT_TRAIL_MULT', 1.0)


# ══════════════════════════════════════════════════════════════
#  기본값은 꺼져 있어야 한다
# ══════════════════════════════════════════════════════════════

class TestDisabledByDefault:
    def test_config_default_off(self):
        import atlas_spot_config as cfg
        assert cfg.SPOT_TRAIL_ENABLED is False, (
            '검증 전 기본 활성화는 실계좌 매매를 바꾸는 것이다')

    def test_no_movement_when_disabled(self, monkeypatch):
        monkeypatch.setattr(rules, 'SPOT_TRAIL_ENABLED', False)
        # 진입 100, SL 95(=1R 5), 가격이 130까지 갔어도 SL 불변
        assert sm.trailing_sl(100.0, 95.0, 130.0, 5.0) == 95.0


# ══════════════════════════════════════════════════════════════
#  추적 규칙
# ══════════════════════════════════════════════════════════════

class TestTrailingRule:
    def test_inactive_before_threshold(self, _on):
        """+1R에 못 미치면 손절을 건드리지 않는다."""
        assert sm.trailing_sl(100.0, 95.0, 104.9, 5.0) == 95.0

    def test_activates_at_breakeven(self, _on):
        """ACTIVATE_R = TRAIL_MULT = 1.0 이면 활성화 시점이 곧 본전 이동."""
        assert sm.trailing_sl(100.0, 95.0, 105.0, 5.0) == pytest.approx(100.0)

    def test_follows_peak(self, _on):
        """최고가 115 → 손절 110 (1R 아래)."""
        assert sm.trailing_sl(100.0, 95.0, 115.0, 5.0) == pytest.approx(110.0)

    def test_never_moves_down(self, _on):
        """가격이 되밀려도 손절은 내려가지 않는다."""
        assert sm.trailing_sl(100.0, 112.0, 115.0, 5.0) == pytest.approx(112.0)
        # peak가 낮아도 기존 sl 유지
        assert sm.trailing_sl(100.0, 112.0, 106.0, 5.0) == 112.0

    def test_mult_widens_trail(self, _on, monkeypatch):
        monkeypatch.setattr(rules, 'SPOT_TRAIL_MULT', 2.0)
        assert sm.trailing_sl(100.0, 95.0, 120.0, 5.0) == pytest.approx(110.0)

    def test_activate_threshold_respected(self, _on, monkeypatch):
        monkeypatch.setattr(rules, 'SPOT_TRAIL_ACTIVATE_R', 2.0)
        assert sm.trailing_sl(100.0, 95.0, 109.9, 5.0) == 95.0     # +2R 미만
        assert sm.trailing_sl(100.0, 95.0, 110.0, 5.0) == pytest.approx(105.0)

    def test_zero_sl_dist_safe(self, _on):
        assert sm.trailing_sl(100.0, 100.0, 120.0, 0.0) == 100.0


# ══════════════════════════════════════════════════════════════
#  R배수 오염 방지 (가장 중요)
# ══════════════════════════════════════════════════════════════

class _Ex:
    def __init__(self):
        self.free = {'BTC': 1e9}

    def create_market_sell_order(self, s, q):
        return {'average': 115.0, 'price': 115.0, 'filled': q}

    def fetch_balance(self, params=None):
        return {'free': dict(self.free)}

    def cancel_order(self, oid, s):
        pass

    def fetch_open_orders(self, s):
        return []

    def amount_to_precision(self, s, q):
        return f'{float(q):.8f}'

    def price_to_precision(self, s, p):
        return f'{float(p):.4f}'


class TestRDenominator:
    def _pos(self, sl_now):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 130.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        # 추적으로 손절이 올라간 상태를 만든다
        sm._update_position_sl('S3', 'BTCUSDT', sl_now, 120.0)
        return sm._load_position('S3', 'BTCUSDT')

    def test_orig_sl_persisted_on_entry(self):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 130.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        pos = sm._load_position('S3', 'BTCUSDT')
        assert float(pos['orig_sl']) == pytest.approx(95.0)

    def test_r_uses_original_risk_not_trailed_sl(self, monkeypatch):
        """손절이 95→110으로 올라가도 R 분모는 5(=100-95)여야 한다.
        분모가 좁아지면 R이 부풀어 Kelly·건강도가 오염된다."""
        monkeypatch.setattr(sm, '_state', {
            'dry_run': False, 'day_pnl': 0.0, 'equity': 1000.0,
            'usdt_balance': 1000.0, 'peak_equity': 1000.0,
            'day_start_eq': 1000.0, 'daily_loss_alerted': False,
        })
        monkeypatch.setattr(sm, '_get_ex', lambda: _Ex())
        pos = self._pos(sl_now=110.0)
        assert float(pos['sl']) == pytest.approx(110.0)

        sm._spot_sell('S3', 'BTCUSDT', 'BTC/USDT', pos, 115.0, 'TP')
        with sm._db_lock, sm._db_conn() as c:
            row = dict(c.execute('SELECT * FROM spot_trades').fetchone())
        # (115 - 100) / 5 = 3.0R  ← 원래 위험 기준
        # 추적된 손절(110)을 분모로 쓰면 (115-100)/(100-110) 로 부호까지 깨진다
        assert row['pnl_r'] == pytest.approx(3.0, abs=0.01)

    def test_legacy_row_without_orig_sl_falls_back(self, monkeypatch):
        """구 DB(orig_sl=0)에서는 현재 sl로 폴백해 죽지 않아야 한다."""
        monkeypatch.setattr(sm, '_state', {
            'dry_run': True, 'day_pnl': 0.0, 'equity': 1000.0,
            'usdt_balance': 1000.0, 'peak_equity': 1000.0,
            'day_start_eq': 1000.0, 'daily_loss_alerted': False,
        })
        sm._save_position('S3', 'ETHUSDT', 100.0, 95.0, 130.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        with sm._db_lock, sm._db_conn() as c:
            c.execute("UPDATE spot_positions SET orig_sl=0 WHERE symbol='ETHUSDT'")
        pos = sm._load_position('S3', 'ETHUSDT')
        sm._spot_sell('S3', 'ETHUSDT', 'ETH/USDT', pos, 110.0, 'TP')
        with sm._db_lock, sm._db_conn() as c:
            row = dict(c.execute("SELECT * FROM spot_trades WHERE symbol='ETHUSDT'").fetchone())
        assert row['pnl_r'] == pytest.approx(2.0, abs=0.01)   # (110-100)/5


# ══════════════════════════════════════════════════════════════
#  거래소 스탑 재등록 (churn 방지)
# ══════════════════════════════════════════════════════════════

class TestRearm:
    def _pos(self):
        sm._save_position('S3', 'BTCUSDT', 100.0, 95.0, 130.0, 1.0, 100.0,
                          0.02, 'sl_tp', 0, 'TRENDING_UP')
        sm._update_position_order_id('S3', 'BTCUSDT', 'OLD-SL', 'OLD-TP')
        return sm._load_position('S3', 'BTCUSDT')

    @pytest.fixture
    def _live(self, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'dry_run': False})
        calls = []
        monkeypatch.setattr(sm, '_cancel_stop_order',
                            lambda st, sy, cs, oid: calls.append(('cancel', oid)))
        monkeypatch.setattr(sm, '_place_protective_orders',
                            lambda *a, **k: (calls.append(('place', a[4])), ('NEW-SL', 'NEW-TP'))[1])
        return calls

    def test_small_move_does_not_rearm(self, _live):
        """미세 이동마다 주문을 갈아끼우면 API 낭비 + 체결 공백이 생긴다."""
        pos = self._pos()
        sm._rearm_trailing_stop('S3', 'BTCUSDT', 'BTC/USDT', pos, 95.5)
        assert _live == []

    def test_material_move_rearms(self, _live):
        pos = self._pos()
        sm._rearm_trailing_stop('S3', 'BTCUSDT', 'BTC/USDT', pos, 100.0)
        assert ('place', 100.0) in _live
        assert ('cancel', 'OLD-SL') in _live and ('cancel', 'OLD-TP') in _live
        assert sm._load_position('S3', 'BTCUSDT')['sl_order_id'] == 'NEW-SL'

    def test_dry_run_skips(self, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'dry_run': True})
        called = []
        monkeypatch.setattr(sm, '_cancel_stop_order',
                            lambda *a: called.append(a))
        pos = self._pos()
        sm._rearm_trailing_stop('S3', 'BTCUSDT', 'BTC/USDT', pos, 120.0)
        assert called == []

    def test_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'dry_run': False})
        monkeypatch.setattr(sm, '_cancel_stop_order',
                            lambda *a: (_ for _ in ()).throw(Exception('net')))
        pos = self._pos()
        sm._rearm_trailing_stop('S3', 'BTCUSDT', 'BTC/USDT', pos, 120.0)   # 예외 전파 금지


# ══════════════════════════════════════════════════════════════
#  백테스트 패리티
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def _ex(monkeypatch):
    ex = _Ex()
    monkeypatch.setattr(sm, '_get_ex', lambda: ex)
    return ex


class TestRearmBackoff:
    """자가복구는 '언젠가 성공할 실패'에만 재시도해야 한다.

    주문금액이 거래소 최소치에 못 미치면 몇 번을 시도해도 영원히 실패한다
    (소액 계좌에서 흔하다 — 사이징 진단 참조). 구분하지 않으면 5분마다
    무한 재시도하며 API만 쓰고, 해결 불가능한 경고를 계속 계산한다.
    """

    @pytest.fixture(autouse=True)
    def _live(self, monkeypatch):
        monkeypatch.setattr(sm, '_state', {'dry_run': False})
        monkeypatch.setattr(sm, '_rearm_attempts', {})
        self.placed = []
        monkeypatch.setattr(sm, '_place_protective_orders',
                            lambda *a, **k: (self.placed.append(a), ('', ''))[1])

    def _pos(self, qty):
        sm._save_position('S4', 'BTCUSDT', 100.0, 95.0, 110.0, qty, qty * 100,
                          0.02, 'sl_tp', 0, 'TRENDING_DOWN')
        return sm._load_position('S4', 'BTCUSDT')

    def test_impossible_detected(self, _ex):
        assert sm._protection_impossible('BTC/USDT', 0.04, 95.0) is True   # $3.8
        assert sm._protection_impossible('BTC/USDT', 1.0, 95.0) is False

    def test_structural_failure_backs_off(self, _ex, _no_telegram):
        pos = self._pos(0.04)                       # 명목가 $4 < 최소 $5
        for _ in range(5):
            sm._rearm_attempts[('S4', 'BTCUSDT')] = (
                0.0, sm._rearm_attempts.get(('S4', 'BTCUSDT'), (0, 0))[1])
            sm._rearm_missing_protection('S4', 'BTCUSDT', 'BTC/USDT', pos)
        assert self.placed == [], '불가능한 조건에서 주문을 시도하면 안 된다'
        last, _ = sm._rearm_attempts[('S4', 'BTCUSDT')]
        assert last > time.time() + 3600, '장기 보류로 전환돼야 한다'

    def test_alerts_once_about_impossibility(self, _ex, _no_telegram):
        pos = self._pos(0.04)
        for _ in range(5):
            sm._rearm_attempts[('S4', 'BTCUSDT')] = (
                0.0, sm._rearm_attempts.get(('S4', 'BTCUSDT'), (0, 0))[1])
            sm._rearm_missing_protection('S4', 'BTCUSDT', 'BTC/USDT', pos)
        msgs = [m for m in _no_telegram if '보호주문 불가' in m]
        assert len(msgs) == 1, '반복 경고는 무시당한다'
        assert '소프트웨어 SL' in msgs[0], '실제 위험을 알려야 한다'

    def test_transient_failure_still_retries(self, _ex, _no_telegram):
        """금액이 충분하면(=일시적 실패) 재시도를 계속해야 한다."""
        pos = self._pos(1.0)
        for _ in range(3):
            sm._rearm_attempts[('S4', 'BTCUSDT')] = (
                0.0, sm._rearm_attempts.get(('S4', 'BTCUSDT'), (0, 0))[1])
            sm._rearm_missing_protection('S4', 'BTCUSDT', 'BTC/USDT', pos)
        assert len(self.placed) == 3

    def test_record_cleared_on_position_close(self, _ex):
        self._pos(1.0)
        sm._rearm_attempts[('S4', 'BTCUSDT')] = (123.0, 2)
        sm._delete_position('S4', 'BTCUSDT')
        assert ('S4', 'BTCUSDT') not in sm._rearm_attempts, (
            '남겨두면 항목이 쌓이고, 재진입 시 옛 실패 횟수를 물려받는다')


class TestBacktestEffect:
    """트레일링이 백테스트 청산에 실제로 개입하는지 결정적으로 확인한다.
    (규칙 함수만 맞고 루프에 배선되지 않으면 단위 테스트는 통과한다)"""

    def _run(self, enabled, monkeypatch):
        import atlas_spot_backtest as bt

        ts = 1609459200000
        warm = [100 + (i % 5) * 0.2 for i in range(300)]     # 지표 워밍업
        # 진입 100 → 최고 120 → 되밀림 100 (트레일링 없으면 이익 전부 반납)
        scen = [100, 104, 108, 112, 116, 120, 116, 112, 108, 104, 100, 100, 100]
        prices = warm + scen
        rows = [[ts + i * 4 * 3600 * 1000, p, p + 0.4, p - 0.4, p, 1e6]
                for i, p in enumerate(prices)]
        entry_bar = len(warm)

        def stub(df, i):
            if i == entry_bar - 1:
                return {'signal': 1, 'sl': 95.0, 'tp': 200.0, 'rr': 20.0,
                        'exit_type': 'sl_tp', 'max_hold': len(scen) - 1}
            return {'signal': 0}

        monkeypatch.setitem(bt.SIGNAL_FUNCS, 'S3', stub)
        monkeypatch.setitem(bt.EXIT_CHECK_FUNCS, 'S3', None)
        monkeypatch.setattr(bt, 'SPOT_TRAIL_ENABLED', enabled)     # bt 자체 게이트
        monkeypatch.setattr(rules, 'SPOT_TRAIL_ENABLED', enabled)  # trailing_sl 조회처
        trades, _ = bt.backtest_strategy('S3', 'BTCUSDT', rows, {},
                                         '2021-01-01', '2022-12-31', risk_pct=0.02)
        return trades

    def test_off_gives_back_the_run(self, monkeypatch):
        t = self._run(False, monkeypatch)
        assert len(t) == 1 and t[0].reason == 'TIME'
        assert t[0].pnl_r == pytest.approx(0.0, abs=0.1), '되밀림을 그대로 맞는다'

    def test_on_locks_in_profit(self, monkeypatch):
        t = self._run(True, monkeypatch)
        assert len(t) == 1 and t[0].reason == 'SL', '추적된 손절로 청산돼야 한다'
        assert t[0].pnl_r > 2.0, f'이익이 확보돼야 한다 (실제 {t[0].pnl_r:+.2f}R)'

    def test_r_denominator_stays_original(self, monkeypatch):
        """추적으로 sl이 111까지 올라갔어도 R 분모는 원래 위험(5)이다.
        분모가 바뀌면 R이 부풀어 Kelly·건강도가 오염된다."""
        t = self._run(True, monkeypatch)
        expected = (t[0].exit_px - t[0].entry) / 5.0
        assert t[0].pnl_r == pytest.approx(expected, abs=0.02)


class TestBacktestParity:
    def test_shares_same_rule_function(self):
        import atlas_spot_backtest as bt
        assert bt.trailing_sl is sm.trailing_sl, '규칙이 두 벌이면 패리티가 깨진다'
        assert bt.trailing_sl is rules.trailing_sl, (
            '단일 출처는 atlas_rules 다 — 어느 쪽이든 복제본이 생기면 안 된다')

    def test_backtest_import_has_no_live_side_effects(self):
        """backtest 를 import해도 atlas_spot_main 이 실행되면 안 된다.

        예전에는 backtest:70 이 main 에서 trailing_sl 을 가져와, 백테스트·
        리포트·최적화기가 import만 해도 라이브 봇의 로그 핸들러 설치·mkdir·
        캐시 생성을 물려받았다. atlas_rules 로 결합을 끊은 것이 이 테스트의
        보호 대상이다. (여기서는 소스 검사 — 프로세스 격리 검증은 무겁다)
        """
        import atlas_spot_backtest as bt
        src = Path(bt.__file__).read_text(encoding='utf-8')
        assert 'from atlas_spot_main import' not in src, (
            'backtest 가 다시 main 에 결합됐다 — import 부수효과가 되살아난다')
        assert 'import atlas_spot_main' not in src

    def test_backtest_reads_same_flag(self):
        import atlas_spot_backtest as bt
        import atlas_spot_config as cfg
        assert bt.SPOT_TRAIL_ENABLED == cfg.SPOT_TRAIL_ENABLED

    def test_backtest_keeps_orig_sl(self):
        """백테스트 포지션도 원본 SL을 따로 보존해야 R이 부풀지 않는다."""
        import atlas_spot_backtest as bt
        src = Path(bt.__file__).read_text()
        assert "'orig_sl':" in src
        assert "position.get('orig_sl', position['sl'])" in src

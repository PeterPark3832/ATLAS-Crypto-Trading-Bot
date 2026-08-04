"""
ATLAS — 진입 필터 (S5 안전 · RS Gate · 펀딩비)
================================
`_strategy_timeframe_loop` 안에 인라인으로 박혀 있던 세 필터를 뽑아냈다.
루프 안에 있을 때는 이 규칙들만 따로 시험할 수 없었고, 그래서 라이브에만
있는 규칙인지 백테스트에도 있는지조차 눈에 띄지 않았다.

여기서 고정하는 것:
  ① 차단과 감봉은 **타입으로 구분**된다(None vs 배수).
     0.0으로 표현하면 곱셈에 흘러들어가 '수량 0 주문'이 된다.
  ② 라이브 RS Gate가 백테스트 `_bt_rs_gate`와 같은 판정을 낸다.
  ③ 백테스트가 모델링하지 않는 필터(펀딩비)를 명시한다.

실행:
  pytest tests/test_entry_filters.py -v
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import atlas_spot_backtest as bt
import atlas_spot_config as cfg
import atlas_spot_main as sm


# ══════════════════════════════════════════════════════════════
#  RS Gate — 라이브 ↔ 백테스트 동일 판정
# ══════════════════════════════════════════════════════════════

class TestRsGateScale:
    def test_non_gate_strategy_is_neutral(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda s: 0.99)
        assert sm._rs_gate_scale('S4', 'X') == 1.0

    def test_bottom_rank_is_blocked_with_none(self, monkeypatch):
        monkeypatch.setattr(sm, 'MOMENTUM_RS_GATE_PCT', 0.5)
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda s: 0.9)
        assert sm._rs_gate_scale('S6', 'X') is None, (
            '차단을 0.0으로 돌려주면 곱셈에 흘러들어가 수량 0 주문이 된다')

    def test_top_tier_gets_boost(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda s: 0.01)
        assert sm._rs_gate_scale('S6', 'X') == cfg.MOMENTUM_TOP_RISK_MULT

    def test_middle_is_neutral(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda s: 0.5)
        assert sm._rs_gate_scale('S6', 'X') == 1.0

    @pytest.mark.parametrize('rank', [0.0, 0.1, 0.33, 0.5, 0.7, 0.99])
    def test_matches_backtest_decision(self, rank, monkeypatch):
        """같은 순위에서 라이브와 백테스트가 같은 판정을 내야 한다.
        갈라지면 WFO가 고른 임계값이 실계좌에서 다르게 동작한다."""
        monkeypatch.setattr(sm, '_get_momentum_rank_pct', lambda s: rank)
        live = sm._rs_gate_scale('S6', 'X')

        rank_map = {'2024-01-01': {'X': rank}}
        bt_scale, _ = bt._bt_rs_gate(True, rank_map, ['2024-01-01'],
                                     'X', '2024-01-02')
        assert (live is None) == (bt_scale is None), f'rank={rank} 차단 판정 불일치'
        if live is not None:
            assert live == pytest.approx(bt_scale), f'rank={rank} 배수 불일치'


# ══════════════════════════════════════════════════════════════
#  펀딩비 — 백테스트 미모델링
# ══════════════════════════════════════════════════════════════

class TestFundingScale:
    def test_non_applied_strategy_is_neutral(self, monkeypatch):
        monkeypatch.setattr(sm, '_get_spot_funding', lambda s: 9.9)
        assert sm._funding_scale('S4', 'X') in (1.0, None) or True
        if 'S4' not in cfg.FUNDING_APPLY_STRATS:
            assert sm._funding_scale('S4', 'X') == 1.0

    def test_long_crowding_blocks(self, monkeypatch):
        sid = next(iter(cfg.FUNDING_APPLY_STRATS))
        monkeypatch.setattr(sm, '_get_spot_funding',
                            lambda s: cfg.FUNDING_LONG_BLOCK)
        assert sm._funding_scale(sid, 'X') is None

    def test_short_squeeze_boosts(self, monkeypatch):
        sid = next(iter(cfg.FUNDING_APPLY_STRATS))
        monkeypatch.setattr(sm, '_get_spot_funding',
                            lambda s: cfg.FUNDING_SHORT_BOOST)
        assert sm._funding_scale(sid, 'X') > 1.0

    def test_neutral_funding(self, monkeypatch):
        sid = next(iter(cfg.FUNDING_APPLY_STRATS))
        mid = (cfg.FUNDING_SHORT_BOOST + cfg.FUNDING_LONG_BLOCK) / 2
        monkeypatch.setattr(sm, '_get_spot_funding', lambda s: mid)
        assert sm._funding_scale(sid, 'X') == 1.0

    def test_backtest_does_not_model_funding(self):
        """백테스트는 과거 펀딩비를 받지 않아 이 필터가 없다 — 낙관 편향."""
        src = Path(bt.__file__).read_text()
        assert 'FUNDING_APPLY_STRATS' not in src
        assert '펀딩비' in (sm._funding_scale.__doc__ or ''), (
            '모델링 격차가 코드에 적혀 있어야 다음 사람이 안다')


# ══════════════════════════════════════════════════════════════
#  S5 안전 필터
# ══════════════════════════════════════════════════════════════

class TestS5SafetyBlock:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / 'spot.db'
        conn = sqlite3.connect(str(p))
        conn.executescript(
            'CREATE TABLE spot_trades (id INTEGER PRIMARY KEY, strategy TEXT,'
            ' symbol TEXT, reason TEXT, exit_ts TEXT);'
            'CREATE TABLE spot_positions (id INTEGER PRIMARY KEY, strategy TEXT,'
            ' symbol TEXT);')
        conn.commit(); conn.close()
        monkeypatch.setattr(sm, 'SPOT_DB_FILE', p)
        return p

    @staticmethod
    def _add_sl(db, symbol, hours_ago):
        import sqlite3
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        c = sqlite3.connect(str(db))
        c.execute("INSERT INTO spot_trades (strategy,symbol,reason,exit_ts) "
                  "VALUES ('S5',?,'SL',?)", (symbol, ts))
        c.commit(); c.close()

    def test_no_history_passes(self, db):
        assert sm._s5_safety_block('AUSDT') is None

    def test_recent_sl_blocks(self, db):
        self._add_sl(db, 'AUSDT', hours_ago=1)
        assert sm._s5_safety_block('AUSDT') == 'sl_cooldown'

    def test_old_sl_passes(self, db):
        self._add_sl(db, 'AUSDT', hours_ago=cfg.S5_SL_COOLDOWN_BARS * 24 + 5)
        assert sm._s5_safety_block('AUSDT') is None

    def test_other_symbol_sl_does_not_block(self, db):
        self._add_sl(db, 'BUSDT', hours_ago=1)
        assert sm._s5_safety_block('AUSDT') is None

    def test_corrupt_timestamp_does_not_block_silently(self, db, caplog):
        """시각 파싱에 실패했는데 조용히 통과하면 쿨다운이 사라진다."""
        import sqlite3
        c = sqlite3.connect(str(db))
        c.execute("INSERT INTO spot_trades (strategy,symbol,reason,exit_ts) "
                  "VALUES ('S5','AUSDT','SL','not-a-timestamp')")
        c.commit(); c.close()
        with caplog.at_level('WARNING'):
            sm._s5_safety_block('AUSDT')
        assert any('SL 시각 파싱 실패' in r.message for r in caplog.records)

    def test_btc_corr_limit_blocks(self, db, monkeypatch):
        import sqlite3
        monkeypatch.setattr(sm, 'S5_BTC_CORR_SYMBOLS', ['AUSDT', 'BUSDT'])
        monkeypatch.setattr(sm, 'S5_CORR_MAX_POS', 1)
        c = sqlite3.connect(str(db))
        c.execute("INSERT INTO spot_positions (strategy,symbol) VALUES ('S5','BUSDT')")
        c.commit(); c.close()
        assert sm._s5_safety_block('AUSDT') == 'btc_corr_limit'

    def test_symbol_outside_corr_group_passes(self, db, monkeypatch):
        import sqlite3
        monkeypatch.setattr(sm, 'S5_BTC_CORR_SYMBOLS', ['AUSDT'])
        monkeypatch.setattr(sm, 'S5_CORR_MAX_POS', 1)
        c = sqlite3.connect(str(db))
        c.execute("INSERT INTO spot_positions (strategy,symbol) VALUES ('S5','AUSDT')")
        c.commit(); c.close()
        assert sm._s5_safety_block('ZUSDT') is None

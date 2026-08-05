"""
ATLAS — 보호주문 점검 도구
================================
`check_protection.py`는 운영자가 **실계좌 상태를 직접 확인**하는 도구다.
DB의 sl_order_id가 채워져 있다는 것만으로는 부족하다 — 그 주문이 거래소에
살아 있는지, 수량이 실제 보유량과 맞는지 봐야 한다.

여기서 고정하는 것:
  ① 수량 불일치(수수료 차감)를 실제로 잡는가
  ② 기본은 **읽기 전용** — --fix 없이 DB를 건드리면 안 된다
  ③ 미체결 주문 조회가 실패해도 죽지 않는가(부분 정보로 보고)

실행:
  pytest tests/test_check_protection.py -v
"""

import os
import sqlite3
import sys
from pathlib import Path

for _k in ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID'):
    os.environ.setdefault(_k, 'TEST')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import check_protection as cp


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / 'spot.db'
    conn = sqlite3.connect(str(p))
    conn.execute(
        'CREATE TABLE spot_positions (id INTEGER PRIMARY KEY, strategy TEXT, '
        'symbol TEXT, qty_tokens REAL, sl REAL, tp REAL, '
        'sl_order_id TEXT, tp_order_id TEXT, entry_price REAL)')
    rows = [
        # ADA: 기록 > 보유 (수수료가 기초자산에서 빠진 상태) + SL 없음
        ('S4', 'ADAUSDT', 44.9, 1.0, 1.5, '', '', 1.2),
        # 정상: 수량 일치 + SL 등록됨
        ('S6', 'SOLUSDT', 2.0, 100.0, 150.0, '111', '', 120.0),
    ]
    conn.executemany(
        'INSERT INTO spot_positions (strategy,symbol,qty_tokens,sl,tp,'
        'sl_order_id,tp_order_id,entry_price) VALUES (?,?,?,?,?,?,?,?)', rows)
    conn.commit(); conn.close()
    monkeypatch.setattr(cp, 'SPOT_DB_FILE', p)
    return p


class _Ex:
    def __init__(self, free, orders=None, fail_orders=False):
        self._free, self._orders = free, orders or []
        self._fail = fail_orders

    def fetch_balance(self):
        return {'free': self._free}

    def fetch_open_orders(self):
        if self._fail:
            raise RuntimeError('권한 없음')
        return self._orders


@pytest.fixture
def ex(monkeypatch):
    def _set(free, orders=None, fail_orders=False):
        e = _Ex(free, orders, fail_orders)
        monkeypatch.setattr(cp, '_exchange', lambda: e)
        return e
    return _set


class TestDetection:
    def test_detects_quantity_shortfall(self, db, ex, capsys):
        """수수료가 기초자산에서 빠져 보유량이 모자란 상태를 잡는다."""
        ex({'ADA': 44.8551, 'SOL': 2.0}, [{'id': '111'}])
        rc = cp.main([])
        out = capsys.readouterr().out
        assert rc == 1
        assert '수량 초과' in out
        assert 'ADAUSDT' in out

    def test_healthy_position_passes(self, db, ex, capsys):
        ex({'ADA': 44.9, 'SOL': 2.0},
           [{'id': '111'}])
        # ADA는 SL 주문 ID가 비어 있어 여전히 문제로 잡혀야 한다
        cp.main([])
        out = capsys.readouterr().out
        assert 'SL주문 ID 없음' in out

    def test_detects_dead_order_id(self, db, ex, capsys):
        """DB에는 주문 ID가 있는데 거래소에 없으면 보호가 없는 것이다."""
        ex({'ADA': 44.9, 'SOL': 2.0}, [])      # 미체결 주문 없음
        cp.main([])
        assert '거래소에 주문 없음' in capsys.readouterr().out

    def test_flags_structurally_impossible(self, db, ex, capsys):
        """NOTIONAL 미달이면 재시도해도 영원히 실패한다 — 구분해서 알린다."""
        ex({'ADA': 0.001, 'SOL': 2.0}, [{'id': '111'}])
        cp.main([])
        assert '구조적 불가' in capsys.readouterr().out

    def test_survives_open_order_query_failure(self, db, ex, capsys):
        """미체결 조회가 막혀도 수량 점검은 계속돼야 한다."""
        ex({'ADA': 44.8551, 'SOL': 2.0}, fail_orders=True)
        cp.main([])
        out = capsys.readouterr().out
        assert '수량 초과' in out and '확인할 수 없' in out


class TestReadOnlyByDefault:
    def _qty(self, db, symbol):
        conn = sqlite3.connect(str(db))
        try:
            return conn.execute('SELECT qty_tokens FROM spot_positions '
                                'WHERE symbol=?', (symbol,)).fetchone()[0]
        finally:
            conn.close()

    def test_does_not_modify_db_without_fix(self, db, ex, capsys):
        """운영자가 점검만 하려 했는데 DB가 바뀌면 안 된다."""
        ex({'ADA': 44.8551, 'SOL': 2.0}, [{'id': '111'}])
        cp.main([])
        assert self._qty(db, 'ADAUSDT') == pytest.approx(44.9)
        assert '--fix' in capsys.readouterr().out

    def test_fix_corrects_to_actual(self, db, ex):
        ex({'ADA': 44.8551, 'SOL': 2.0}, [{'id': '111'}])
        cp.main(['--fix'])
        assert self._qty(db, 'ADAUSDT') == pytest.approx(44.8551)

    def test_fix_leaves_healthy_rows_alone(self, db, ex):
        ex({'ADA': 44.8551, 'SOL': 2.0}, [{'id': '111'}])
        cp.main(['--fix'])
        assert self._qty(db, 'SOLUSDT') == pytest.approx(2.0)

    def test_never_places_orders(self):
        """이 도구는 주문을 내지 않는다 — 자가복구는 봇의 일이다."""
        src = Path(cp.__file__).read_text()
        assert 'create_order' not in src and 'privatePost' not in src


class TestEdgeCases:
    def test_no_positions_is_clean_exit(self, tmp_path, monkeypatch, capsys):
        p = tmp_path / 'empty.db'
        conn = sqlite3.connect(str(p))
        conn.execute('CREATE TABLE spot_positions (id INTEGER PRIMARY KEY, '
                     'strategy TEXT, symbol TEXT, qty_tokens REAL, sl REAL, '
                     'tp REAL, sl_order_id TEXT, tp_order_id TEXT, '
                     'entry_price REAL)')
        conn.commit(); conn.close()
        monkeypatch.setattr(cp, 'SPOT_DB_FILE', p)
        assert cp.main([]) == 0

    def test_missing_db_is_clean_exit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, 'SPOT_DB_FILE', tmp_path / 'nope.db')
        assert cp.main([]) == 0

    def test_tolerance_ignores_rounding_noise(self, db, ex, capsys):
        """거래소 정밀도 반올림까지 '문제'로 잡으면 경고가 무의미해진다.

        허용오차를 상수 기준으로 만들면(예: MISMATCH_PCT/2) 상수를 0으로
        바꿔도 차이가 0이 되어 검사가 통과해 버린다 — **고정된 미세 차이**를
        써야 허용오차가 실제로 동작하는지 잡는다.
        """
        ex({'ADA': 44.9 * (1 - 1e-9), 'SOL': 2.0}, [{'id': '111'}])
        cp.main([])
        assert '수량 초과' not in capsys.readouterr().out

    def test_tolerance_is_not_zero(self):
        """0이면 부동소수 잡음까지 전부 경고가 되어 아무도 안 보게 된다."""
        assert cp.MISMATCH_PCT > 0

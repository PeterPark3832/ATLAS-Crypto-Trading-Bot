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

import sqlite3
from pathlib import Path


import pytest

import check_protection as cp


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    """이 도구는 프라이빗 API를 쓰므로 자격증명 가드가 있다.

    로직 테스트에서는 통과시키고, 가드 자체는 TestCredentialGuard가 본다.
    """
    monkeypatch.setattr(cp, 'BINANCE_API_KEY', 'live-key')
    monkeypatch.setattr(cp, 'BINANCE_API_SECRET', 'live-secret')


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
    """가짜 거래소.

    total은 free + locked다. 보호주문이 걸린 포지션은 물량이 잠겨 있어
    free가 0에 가깝다 — 도구는 total로 판정해야 한다.
    """

    def __init__(self, free, orders=None, fail_orders=False, locked=None):
        self._free, self._orders = free, orders or []
        self._fail = fail_orders
        self._locked = locked or {}

    def fetch_balance(self):
        total = {k: v + self._locked.get(k, 0.0) for k, v in self._free.items()}
        for k, v in self._locked.items():
            total.setdefault(k, v)
        return {'free': self._free, 'used': self._locked, 'total': total}

    def fetch_open_orders(self, symbol=None):
        if self._fail:
            raise RuntimeError('권한 없음')
        return self._orders


@pytest.fixture
def ex(monkeypatch):
    def _set(free, orders=None, fail_orders=False, locked=None):
        e = _Ex(free, orders, fail_orders, locked)
        monkeypatch.setattr(cp, '_exchange', lambda: e)
        return e
    return _set


class TestCredentialGuard:
    """환경변수 스텁이 .env의 실제 키를 가리면 도구 전체가 죽는다.

    실제로 배포판에서 발생했다 — 모듈 상단의
    `os.environ.setdefault('BINANCE_API_KEY', 'CHECK')`가 먼저 실행되고,
    atlas_spot_config는 load_dotenv를 override 없이 부르기 때문에 더미가
    실제 키를 가렸다. 키는 _opt라 검증 없이 통과해 그대로 거래소로 나갔고,
    운영자에게는 원인을 알 수 없는 ccxt -2008 스택트레이스만 남았다.
    """

    def test_missing_key_reports_clearly(self, monkeypatch):
        monkeypatch.setattr(cp, 'BINANCE_API_KEY', '')
        assert 'API 키가 없습니다' in cp.credential_error()

    def test_dummy_key_is_detected(self, monkeypatch):
        monkeypatch.setattr(cp, 'BINANCE_API_KEY', 'CHECK')
        assert '더미' in cp.credential_error()

    def test_valid_credentials_pass(self):
        assert cp.credential_error() == ''

    def test_main_exits_before_touching_exchange(self, db, monkeypatch):
        """가드는 네트워크 호출 **전에** 걸려야 한다."""
        monkeypatch.setattr(cp, 'BINANCE_API_KEY', 'CHECK')

        def _boom():
            raise AssertionError('자격증명 확인 전에 거래소를 호출했다')
        monkeypatch.setattr(cp, '_exchange', _boom)
        assert cp.main([]) == 2

    def test_module_never_stubs_credentials(self):
        """회귀 방지 — 스텁이 다시 들어오면 잡는다(주석은 제외)."""
        src = Path(cp.__file__).read_text(encoding='utf-8')
        code = '\n'.join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith('#'))
        assert 'environ.setdefault' not in code, (
            '자격증명 스텁이 다시 추가됐다 — .env의 실제 키를 가려 '
            '도구가 -2008로 죽는다')


class TestLockedBalanceCountsAsHeld:
    """보호주문이 잠근 물량도 그 포지션의 보유량이다.

    free만 보면 **제대로 보호된 포지션일수록** free가 0에 가까워
    '수량 초과 99.9%'로 판정된다. 그 상태에서 --fix를 돌리면 DB 수량이
    먼지 값으로 덮여 포지션 기록이 파괴된다(청산 시 그만큼만 팔고
    나머지는 거래소에 방치). 라이브 3포지션에서 실제로 재현됐다.
    """

    def test_locked_position_is_healthy(self, db, ex, capsys):
        # SOL 2.0 전량이 자기 손절 주문에 잠겨 있다 (free 0)
        ex({'ADA': 44.9, 'SOL': 0.0}, [{'id': '111'}], locked={'SOL': 2.0})
        cp.main([])
        out = capsys.readouterr().out
        assert '수량 초과' not in out.split('SOLUSDT')[1].split('\n')[0], (
            '잠긴 물량을 보유량에서 누락해 정상 포지션을 문제로 판정했다')

    def test_locked_position_is_not_offered_for_fix(self, db, ex, capsys):
        ex({'ADA': 44.9, 'SOL': 0.0}, [{'id': '111'}], locked={'SOL': 2.0})
        cp.main([])
        out = capsys.readouterr().out
        assert 'SOLUSDT: 2.00000000 → 0.00000000' not in out, (
            '--fix가 포지션 수량을 0으로 덮어쓰려 한다 — 데이터 파괴'
        )

    def test_real_shortfall_still_detected_when_locked(self, db, ex, capsys):
        """잠금을 감안해도 실제로 모자라면 여전히 잡아야 한다."""
        # DB 44.9인데 free 0.05 + locked 44.0 = 44.05 → 실제 부족
        ex({'ADA': 0.05, 'SOL': 2.0}, [{'id': '111'}], locked={'ADA': 44.0})
        cp.main([])
        assert '수량 초과' in capsys.readouterr().out


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

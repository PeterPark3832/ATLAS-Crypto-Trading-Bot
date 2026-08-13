"""
ATLAS — 조회 전용 DB 접근 공통부 (경로 해석 + 읽기전용 연결)
==========================================================
weekly/monthly/dashboard 가 각자 굴리던 두 관용구를 모은다:

  1) `sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=10)` — 3벌
  2) DB_FILE 환경변수 오버라이드의 **실재 검증 폴백** — weekly에만 있었다

2)는 실제 운영 사고에서 나왔다: 서버 .env에 옛 선물봇 경로
(/root/atlas_bot/state/atlas_v2.db)가 남아 있었고, 그 파일이 없으니 주간
리포트는 거래 40건이 쌓인 상태에서도 매번 '데이터 없음'을 냈다 — 실패처럼
보이지 않아서 아무도 눈치채지 못했다. 같은 부류(MCP 서버의 REMOTE_DB_PATH)
가 한 번 더 재발한 뒤에야 이 방어를 공통부로 승격한다.

⚠️ `atlas_spot_main._db_conn` 은 통합 대상이 **아니다.** 봇은 쓰기 연결 +
   `_db_lock` 직렬화 + commit-on-exit 계약이고, 여기는 조회 전용이다.
"""

import os
import sqlite3
from pathlib import Path


def resolve_db_path(default: str | os.PathLike) -> Path:
    """봇이 쓰는 DB 경로. DB_FILE 환경변수는 **실재할 때만** 우선한다.

    오버라이드가 가리키는 파일이 없으면 그 사실을 알리고 정식 경로로
    되돌린다 — 죽은 경로를 조용히 믿으면 '빈 결과'가 '정상'처럼 보인다.
    """
    override = os.getenv('DB_FILE', '').strip()
    if override:
        p = Path(override)
        if p.exists():
            return p
        print(f'⚠️ DB_FILE={p} 이(가) 없습니다 — 봇 DB로 대체합니다: {default}')
    return Path(default)


def connect_ro(path: str | os.PathLike, timeout: float = 10) -> sqlite3.Connection:
    """읽기전용 연결. 봇이 같은 파일에 쓰는 중에도 안전하다(mode=ro URI)."""
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=timeout)

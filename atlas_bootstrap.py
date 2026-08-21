"""
ATLAS — 스크립트 공용 부트스트랩 (dotenv + 환경변수 기본값)
==========================================================
대시보드·리포트·최적화기가 각자 복사해 쓰던 두 관용구를 한 곳으로 모은다:

  1) `load_dotenv(Path(__file__).parent / '.env')` — 4개 스크립트 사본
  2) `os.environ.setdefault(키, 태그)` 자격증명 스텁 — 테스트 51개 사본

⚠️ 이 모듈은 **atlas_spot_config 를 import하지 않는다.**
   스크립트들은 config import *전에* .env를 로드해 raw os.getenv로
   TG 자격증명을 읽는 순서에 의존한다. 이 모듈이 config를 끌어오면
   그 순서 보장이 조용히 깨진다.

⚠️ `check_protection.py` 에서는 ensure_env_defaults 를 호출하지 말 것.
   그 도구는 프라이빗 API라 진짜 키가 필요하다 — 스텁이 .env의 실제 키를
   가리면 -2008로 전부 죽는다(tests/test_check_protection.py 가 소스 검사로
   강제한다).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

#: 자격증명 스텁 대상 키. 값이 없어도 import는 되게 하되(_opt),
#: 백테스트·테스트처럼 네트워크에 안 나가는 실행에서 KeyError를 막는다.
CREDENTIAL_KEYS = ('BINANCE_API_KEY', 'BINANCE_API_SECRET', 'TG_TOKEN', 'TG_CHAT_ID')


def load_env(anchor: str | os.PathLike | None = None) -> None:
    """anchor 파일과 같은 디렉터리의 .env 를 로드한다 (override 없음).

    override=False 는 의도다 — 이미 설정된 환경변수를 .env가 덮어쓰지
    않아야 systemd 유닛의 Environment= 지시자와 CI의 주입 값이 이긴다.
    atlas_spot_config 의 로드 의미와 동일하다.
    """
    base = Path(anchor).parent if anchor else Path(__file__).parent
    env = base / '.env'
    load_dotenv(env if env.exists() else None)


def ensure_env_defaults(tag: str = 'TEST') -> None:
    """자격증명 환경변수가 비어 있으면 더미 태그로 채운다.

    setdefault 이므로 실제 값이 있으면 절대 덮지 않는다. tag는 어느 경로로
    스텁이 들어왔는지 로그에서 구분하기 위한 표식이다(BACKTEST/TEST 등).
    """
    for key in CREDENTIAL_KEYS:
        os.environ.setdefault(key, tag)

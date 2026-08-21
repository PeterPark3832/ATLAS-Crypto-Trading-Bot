"""
ATLAS 테스트 공용 부트스트랩
============================
52개 테스트 파일이 똑같이 복사해 쓰던 4줄 프리앰블(자격증명 스텁 +
sys.path 삽입)을 한 곳으로 모은다. conftest는 pytest가 **테스트 모듈을
import하기 전에** 실행하므로 순서 보장이 프리앰블과 동일하다.

스텁이 필요한 이유: atlas_* 모듈들은 import 시점에 os.getenv로 자격증명을
읽는다(_opt라 없어도 죽지는 않지만, 테스트가 빈 문자열 대신 식별 가능한
더미('TEST')를 보게 해 실수로 실네트워크에 나가는 경로를 조기에 드러낸다).

setdefault이므로 CI가 주입한 값(BINANCE_API_KEY=CI 등)을 절대 덮지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import atlas_bootstrap  # noqa: E402  (sys.path 삽입 후여야 한다)

atlas_bootstrap.ensure_env_defaults('TEST')

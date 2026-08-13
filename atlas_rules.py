"""
ATLAS — 라이브·백테스트 공유 규칙 (leaf 모듈)
============================================
"규칙이 두 벌이면 패리티가 깨진다" — trailing_sl 은 라이브와 백테스트가
**같은 함수 객체**를 써야 하고, tests/test_trailing_stop.py 가 `is` 동일성
으로 이를 고정한다.

이전에는 이 함수가 atlas_spot_main 에 있어 백테스트가
`from atlas_spot_main import trailing_sl` 로 가져왔다. 그 한 줄 때문에
**백테스트를 import만 해도 라이브 봇 모듈 전체가 실행**됐다 — 로그 핸들러
설치(root logger), logs/ mkdir, 날짜별 로그 파일 생성, 캐시 인스턴스 생성.
reoptimize·monthly_wfo 같은 배치 잡이 전부 그 부수효과를 물려받았다.
이 모듈은 그 결합을 끊는 새 거처다: config 외에는 아무것도 import하지
않는 leaf 라서, 어느 방향에서 가져와도 부수효과가 없다.

⚠️ reoptimize.override_params 는 (strategies, main, backtest, rules) 모듈
   튜플을 hasattr 로 순회하며 파라미터를 치환한다. 여기 함수가 읽는 config
   상수는 **flat import** 를 유지해야 한다(속성 접근으로 바꾸면 치환이
   조용히 무력화된다 — ruff F401 죽은상수 탐지도 같은 이유로 flat 유지).
"""

from atlas_spot_config import (
    SPOT_TRAIL_ENABLED, SPOT_TRAIL_ACTIVATE_R, SPOT_TRAIL_MULT,
)


# ── 심볼 표기 변환 ────────────────────────────────────────────────
# 'BTCUSDT'(DB·로그 표기) ↔ 'BTC/USDT'(ccxt 표기) 변환이 main에만 ~20곳
# 흩어져 있었다. 순수 문자열 규칙이므로 여기가 단일 출처다.

def to_ccxt(symbol: str) -> str:
    """DB 표기 → ccxt 표기. 'BTCUSDT' → 'BTC/USDT'."""
    return symbol.replace('USDT', '/USDT')


def base_of(ccxt_sym: str) -> str:
    """ccxt 표기에서 기초자산. 'BTC/USDT' → 'BTC'."""
    return ccxt_sym.split('/')[0]


# ── 추적 손절 ────────────────────────────────────────────────────

def trailing_sl(entry: float, cur_sl: float, peak: float,
                sl_dist: float) -> float:
    """추적 손절 계산. 순수 함수 — 라이브/백테스트가 공유한다.

    peak가 entry + sl_dist × ACTIVATE_R 를 넘은 뒤부터
    `peak − sl_dist × TRAIL_MULT` 로 손절을 따라 올린다(내리지 않음).
    SL 거리를 기준으로 하므로 심볼·변동성에 무관하게 스케일이 맞고,
    ACTIVATE_R = TRAIL_MULT 이면 활성화 시점이 자연스럽게 본전 이동이 된다.
    """
    if not SPOT_TRAIL_ENABLED or sl_dist <= 0:
        return cur_sl
    if peak < entry + sl_dist * SPOT_TRAIL_ACTIVATE_R:
        return cur_sl                       # 아직 활성화 전
    return max(cur_sl, peak - sl_dist * SPOT_TRAIL_MULT)

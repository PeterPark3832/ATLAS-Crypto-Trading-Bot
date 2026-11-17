"""
ATLAS Futures — 설정 (단일 출처)
================================
바이낸스 USDⓈ-M 선물용 롱·숏 봇의 모든 매매 파라미터.
현물 봇(atlas_spot_config.py)과 완전히 분리돼 있고, 서로 import하지 않는다.

[설계 근거 — 2026-11 재검토]
  · 데이터: 바이낸스 선물 대형 20종목, 2021-01 ~ 2026-11, 4시간봉으로 장중 손절·익절 판정
  · 장세(BTC 일봉 ADX + EMA200 + ATR%)마다 전략을 하나씩만 켜서 따로 돌리고,
    2021~23 / 2024~26 두 기간 모두 PF 1.15 이상인 조합만 남겼다.

      장세            전략                    PF(2021~23 / 2024~26)
      TRENDING_UP     S6 롱(20일 신고가 돌파)   1.28 / 1.78
      RANGING         S6 롱                     3.11 / 1.52
      WEAK_TREND      SX 숏(20일 신저가 이탈)   1.18 / 1.73
      TRENDING_DOWN   SX 숏                     1.43 / 1.68
      MICRO_RANGING   쉼                        (롱 1.39 → 0.63, 뒤집힘)
      CRISIS          쉼                        (숏 0.71 / 0.00)

  · 거래당 위험 0.5%: 백테스트 연 9.7%, MDD 11.7% (수수료·슬리피지·실제 펀딩비 반영)
  · ⚠️ 위 장세표는 6년 전체를 보고 골랐다. 2021~23만으로 고른 표는 2024~26에서
    이전 안보다 나빴다(6.5% vs 9.8%). 현실 기대치는 연 6~9%로 본다.

[하지 않기로 한 것과 이유]
  · 평균회귀(과매도 매수·과매수 매도): 거의 모든 장세에서 PF < 1
  · 15분봉 스캘핑(예전 v2 Module C 재현): 비용 전 -0.03R, 비용 후 -0.27R/거래
  · 펀딩비 수확 숏(예전 Module D 재현): PF 0.82
  · 파라미터 튜닝: 2021~23에 맞춘 후보가 2024~26에서 무너지는 패턴이 반복됐다
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# 거래 종목 (2021년 초부터 상장된 대형 USDT 무기한 선물 — 백테스트와 동일)
# ─────────────────────────────────────────────────────────────
FUT_UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
    'LINKUSDT', 'AVAXUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'TRXUSDT', 'ATOMUSDT',
    'ETCUSDT', 'FILUSDT', 'UNIUSDT', 'NEARUSDT', 'XLMUSDT', 'AAVEUSDT',
]
FUT_REGIME_SYMBOL = 'BTCUSDT'
# BTC 무기한 선물 상장일. 장세 판정용 EMA200을 백테스트와 **같은 시작점**에서
# 계산하려면 라이브도 이 날부터 일봉을 받아야 한다(시작점이 다르면 값이 다르다).
FUT_BTC_HISTORY_START_MS = 1567900800000   # 2019-09-08 00:00 UTC

# 심볼별 슬리피지(편도) — 기존 현물 백테스트 티어를 그대로 쓴다
FUT_SLIP_TIER1 = ('BTCUSDT', 'ETHUSDT')
FUT_SLIP_TIER2 = ('BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
                  'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT')
FUT_SLIP_BY_TIER = (0.0003, 0.0008, 0.0015)

# ─────────────────────────────────────────────────────────────
# 장세 판정 (atlas_regime.py 와 같은 임계값 — 이 모듈은 그쪽을 import하지 않는다)
# ─────────────────────────────────────────────────────────────
REGIME_ADX_TREND = 25.0
REGIME_ADX_WEAK = 20.0
REGIME_CRISIS_ATR = 0.07
REGIME_ADX_LOOKBACK = 50          # ADX 계산에 쓰는 완성봉 수 (일봉·4시간봉 공통)
REGIME_EMA_SPAN = 200
REGIME_ATR_PERIOD = 14
REGIME_MICRO_ADX4H = 20.0         # 일봉 상승 추세인데 4시간봉 ADX가 이보다 낮으면 MICRO_RANGING

# 장세 → 켤 전략. 여기 없는 장세(MICRO_RANGING·CRISIS·UNKNOWN)는 신규 진입 없음.
REGIME_ROUTE = {
    'TRENDING_UP': ('S6',),
    'RANGING': ('S6',),
    'WEAK_TREND': ('SX',),
    'TRENDING_DOWN': ('SX',),
}
FUT_STRATEGIES = ('S6', 'SX')

# ─────────────────────────────────────────────────────────────
# 전략 파라미터 (튜닝하지 않은 기본값 — 현물 봇 S6와 동일)
# ─────────────────────────────────────────────────────────────
DON_ENTRY = 20            # S6: 20일 신고가 / SX: 20일 신저가
DON_EXIT = 10             # S6: 10일 신저가 이탈 시 청산 / SX: 10일 신고가 돌파 시 청산
VOL_MA = 20               # 거래량·VWAP 이동평균 기간
S6_VOL_MULT = 2.0         # S6만 거래량 스파이크 요구 (숏은 요구하지 않는다 — 아래 주석)
ATR_PERIOD = 14
ATR_SL_MULT = 2.0         # 손절 = 진입가 ∓ ATR × 2 (단, 10일 저가/고가가 더 가까우면 그쪽)
RR_MIN = 1.5
RR_MAX = 3.0
RR_FALLBACK = 2.0
SIGNAL_WARMUP = 45        # 이보다 짧은 기록에서는 신호를 내지 않는다
# ⚠️ 숏에 거래량 조건을 붙이면 성과가 무너진다(PF 1.60 → 0.96).
#    하락장에서 거래량이 터지며 신저가를 깨는 날은 투매(바닥 근처)인 경우가 많다.

# ─────────────────────────────────────────────────────────────
# 위험 관리
# ─────────────────────────────────────────────────────────────
FUT_RISK_PCT = 0.005            # 거래당 위험(손절 시 잃는 금액) = 자산의 0.5%
FUT_NOTIONAL_CAP_MULT = 25.0    # 종목당 명목가 상한 = 위험 × 25 (0.5% → 자산의 12.5%)
FUT_MAX_POSITIONS = 6           # 동시 보유 상한 (종목당 1개 — 단방향 모드)
FUT_MAX_SL_PCT = 0.20           # 손절폭이 이보다 넓으면 진입하지 않는다
FUT_MAX_COST_PER_R = 0.20       # 왕복 비용이 1R의 20%를 넘으면 진입하지 않는다
# 직전 24시간 실현손익 -4% 이하 → 그날 신규 진입 중단. 정상 운용(6개 x 0.5%)의 하루 최대 손실은
# 약 3%라, 이 한도는 평소엔 걸리지 않고 슬리피지 폭발·갭·버그 같은 이상 상황에서만 걸리는 차단기다.
FUT_DAILY_LOSS_LIMIT = -0.04
FUT_DD_HALF = 0.15              # 고점 대비 낙폭 15% → 거래당 위험 절반
FUT_DD_STOP = 0.20              # 고점 대비 낙폭 20% → 신규 진입 중단, 점검 알림
# 30초 사이 자산이 이만큼 바뀌면 이체일 수 있어 낙폭 판정을 한 번 미룬다. 명목가 합계가 최대
# 자산의 75%(6개 x 12.5%)라, 시장이 30초 만에 일제히 6.7% 넘게 움직여야 손익만으로 이 값에 닿는다.
FUT_EQUITY_JUMP_PCT = 0.05
FUT_LEVERAGE = 3                # 격리 3배. 손절폭 상한 20% < 청산 거리(약 32%)
FUT_MARGIN_TYPE = 'ISOLATED'
FUT_MMR = 0.005                 # 백테스트 청산가 근사용 유지증거금률

# 펀딩비 필터 (현물 봇 S6와 같은 규칙, 숏은 대칭으로 차단만)
FUND_LONG_BLOCK = 0.0005        # 롱: 펀딩 ≥ +0.05%/8h → 진입 차단 (롱 과밀)
FUND_LONG_BOOST = -0.0001       # 롱: 펀딩 ≤ -0.01%/8h → 위험 ×1.2
FUND_LONG_BOOST_MULT = 1.2
FUND_SHORT_BLOCK = -0.0005      # 숏: 펀딩 ≤ -0.05%/8h → 진입 차단 (숏 과밀)

# ─────────────────────────────────────────────────────────────
# 주문 실행
# ─────────────────────────────────────────────────────────────
FUT_TAKER_FEE = 0.0005
FUT_MAKER_FEE = 0.0002
ENTRY_POST_OFFSET = 0.0002      # 진입 지정가 = 현재가보다 0.02% 유리한 쪽 (메이커 전용)
ENTRY_LIMIT_SEC = 4 * 3600      # 지정가 대기 4시간 → 못 사면 시장가로 전환 (일봉 마감 기준, 늦게 켜져도 늘어나지 않음)
ENTRY_MAX_REPEG = 10            # 메이커 거부로 다시 거는 횟수 상한 — 넘으면 이번 진입 포기
STOP_SLIP_MULT = 1.5            # 백테스트: 손절(스탑 시장가) 슬리피지 = 티어 × 1.5
LIMIT_THROUGH = 0.0005          # 백테스트: 지정가는 가격이 0.05% 더 뚫고 지나가야 체결로 본다
SL_WORKING_TYPE = 'MARK_PRICE'  # 손절 트리거 기준 (꼬리 한 번에 안 털리도록 마크가격)
DAY_CLOSE_DELAY_SEC = 60        # 일봉 마감(00:00 UTC) 후 이만큼 기다렸다가 판정 (봉 확정 대기)
POLL_SEC = 30                   # 주문·포지션 점검 주기
ALERT_COOLDOWN_SEC = 1800       # 같은 문제 알림은 30분에 한 번 (로그는 매번 남는다)

# ─────────────────────────────────────────────────────────────
# 실행 환경
# ─────────────────────────────────────────────────────────────
FUT_LIVE_BASE_URL = 'https://fapi.binance.com'
FUT_TESTNET_BASE_URL = 'https://demo-fapi.binance.com'   # 바이낸스 선물 데모(테스트넷)
FUT_DB_FILE = 'state/atlas_fut.db'
FUT_LOG_DIR = 'logs'
FUT_KILL_SWITCH = '/tmp/ATLAS_FUT_STOP'
# 실계좌 모드는 이 값을 .env의 FUT_LIVE_CONFIRM에 그대로 적어야만 켜진다(실수 방지).
FUT_LIVE_CONFIRM_PHRASE = 'I_UNDERSTAND_REAL_MONEY'


def slip_for(symbol: str) -> float:
    """심볼별 편도 슬리피지 가정."""
    if symbol in FUT_SLIP_TIER1:
        return FUT_SLIP_BY_TIER[0]
    if symbol in FUT_SLIP_TIER2:
        return FUT_SLIP_BY_TIER[1]
    return FUT_SLIP_BY_TIER[2]


def notional_cap_frac(risk: float = FUT_RISK_PCT) -> float:
    """종목당 명목가 상한(자산 대비 비율). 손절폭 4% 미만일 때만 걸린다."""
    return risk * FUT_NOTIONAL_CAP_MULT

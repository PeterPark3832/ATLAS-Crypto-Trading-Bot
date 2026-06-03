"""
ATLAS Spot Trading Bot — 설정
==============================
현물(Spot) 전용 파라미터. 선물 config(atlas_config.py)와 완전히 분리.

전략 목록:
  S1: Buy & Hold (벤치마크)
  S2: SMA Golden Cross       1D
  S3: EMA Trend Follow       4H  (v2 Module A 롱 전용 변환)
  S4: RSI Mean Reversion     1D  (v2 Module B 롱 전용 변환)
  S5: Bollinger Band Bounce  1D
  S6: Donchian Breakout      1D  (터틀 트레이딩)
  S7: MACD Momentum          4H

레짐별 활성 전략:
  TRENDING_UP  : S2, S3, S6, S7 (추세추종)
  RANGING      : S4, S5         (평균회귀)
  WEAK_TREND   : 전체 (리스크 70%)
  TRENDING_DOWN: S4, S5         (과매도 반등 한정)
  CRISIS       : 전면 차단
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / '.env'
load_dotenv(_env_path if _env_path.exists() else None)


def _opt(key: str, default: str = '') -> str:
    return os.getenv(key, default).strip()


def _req(key: str) -> str:
    v = os.getenv(key, '').strip()
    if not v:
        raise RuntimeError(f"[설정 오류] 환경변수 '{key}' 없음. .env 파일을 확인하세요.")
    return v


# ─────────────────────────────────────────────────────────────
# API / Telegram (선물봇과 동일 키 재사용 가능)
# ─────────────────────────────────────────────────────────────
BINANCE_API_KEY    = _opt('BINANCE_API_KEY')
BINANCE_API_SECRET = _opt('BINANCE_API_SECRET')
TG_TOKEN           = _opt('TG_TOKEN')
TG_CHAT_ID         = _opt('TG_CHAT_ID')

# ─────────────────────────────────────────────────────────────
# 파일 경로
# ─────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
SPOT_DB_FILE      = BASE_DIR / 'state' / 'atlas_spot.db'
SPOT_LOG_DIR      = BASE_DIR / 'logs'
SPOT_DATA_DIR     = BASE_DIR / 'data'
SPOT_RESULTS_DIR  = BASE_DIR / 'results'
SPOT_KILL_SWITCH  = Path('/tmp/ATLAS_SPOT_STOP')

# ─────────────────────────────────────────────────────────────
# 유니버스 설정
# ─────────────────────────────────────────────────────────────
UNIVERSE_QUOTE_CURRENCY   = 'USDT'
UNIVERSE_MIN_VOLUME_USD   = 10_000_000   # 24h 최소 거래량 $10M
UNIVERSE_MAX_SYMBOLS      = 50           # 최대 심볼 수
UNIVERSE_REFRESH_HOURS    = 4            # 갱신 주기 (24→4시간: 주도주 전환 빠르게 포착)
UNIVERSE_MOMENTUM_DAYS    = 45           # 모멘텀 랭킹 기간 (90→45일: 단기 주도주 포착)

# 스테이블코인 기초자산 제외 목록
UNIVERSE_STABLECOIN_BASE = {
    # USD 페그
    'USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'USDP', 'FDUSD',
    'PYUSD', 'UST', 'LUSD', 'FRAX', 'USDD', 'SUSD', 'GUSD',
    'HUSD', 'USTC', 'EUROC', 'USDX', 'USDY', 'USDM',
    'RLUSD',           # Ripple USD (2025년 출시, $1 페그)
    'CRVUSD',          # Curve USD
    'CUSD',            # Celo USD
    'USDE', 'SUSDE',   # Ethena USD
    'FBTC', 'LBTC',    # BTC 래핑 (가격≈BTC, 변동성 제로에 가까움)
    # 귀금속 페그
    'PAXG', 'XAUT',
    # EUR 페그
    'EURS',
}
# 레버리지/인버스 토큰 키워드 제외
UNIVERSE_LEVERAGED_KEYWORDS = ['3L', '3S', '2L', '2S', 'UP', 'DOWN', 'BULL', 'BEAR']

# 가격 기반 스테이블코인 자동 감지 범위 ($0.97~$1.03)
# discover_universe()에서 현재가가 이 범위 내면 자동 제외
UNIVERSE_STABLE_PRICE_MIN = 0.97
UNIVERSE_STABLE_PRICE_MAX = 1.03

# ─────────────────────────────────────────────────────────────
# 포트폴리오 리스크
# ─────────────────────────────────────────────────────────────
SPOT_MAX_POSITIONS     = 15      # 최대 동시 포지션 수 (10→15: 유니버스 집중 베팅 수용)
SPOT_BASE_RISK_PCT     = 0.020   # 거래당 기본 리스크 (1%→2%: 레버리지 없는 현물 적정 수준)
SPOT_MAX_ALLOC_PCT     = 0.15    # 단일 종목 최대 배분 15%
SPOT_MIN_ALLOC_PCT     = 0.02    # 단일 종목 최소 배분 2%
SPOT_RESERVE_PCT       = 0.10    # USDT 최소 예비금 10%
SPOT_DAILY_LOSS_LIMIT  = -0.04   # 일간 손실 한도 -4%
SPOT_MIN_ORDER_USDT    = 5.0     # 최소 주문 금액 (Binance NOTIONAL 실제 기준 $5)
SPOT_MAX_SL_PCT        = 0.20    # SL 거리 상한 20% (초과 시 포지션이 너무 작아 차단)

# Kelly / Ratchet
SPOT_KELLY_MIN_TRADES  = 10      # 최소 거래수 (20→10: Kelly 조기 활성화)
SPOT_KELLY_SCALE_MIN   = 0.30
SPOT_KELLY_SCALE_MAX   = 2.00    # 상한 (1.50→2.00, WR>55% AND PF>1.5 조건부)
SPOT_KELLY_WR_THRESH   = 0.55    # Kelly 상한 2.00 허용 최소 승률
SPOT_KELLY_PF_THRESH   = 1.50    # Kelly 상한 2.00 허용 최소 PF
SPOT_RATCHET_DD_THRESH = 0.05    # 5% 하락 → 70% 리스크
SPOT_RATCHET_DD_HARD   = 0.08    # 8% 하락 → 40% 리스크
SPOT_RATCHET_RECOVER   = 0.15    # 회복 기준 (25%→15%: DD 바닥 대비 +15% 회복 시 복원)

# 모멘텀 집중 베팅 (Relative Strength Gate)
MOMENTUM_TOP_TIER_PCT   = 0.33   # 상위 33% 심볼 = "주도주 티어"
MOMENTUM_TOP_RISK_MULT  = 1.30   # 주도주 티어 리스크 30% 상향
MOMENTUM_RS_GATE_STRATS = ['S6', 'S7']  # RS Gate 적용 전략 (추세돌파 계열만)

# ─────────────────────────────────────────────────────────────
# S2: SMA Golden Cross (1D)
# ─────────────────────────────────────────────────────────────
S2_SMA_FAST    = 50
S2_SMA_SLOW    = 200
S2_ATR_PERIOD  = 14
S2_ATR_SL      = 2.5     # 손절: Entry - ATR × 2.5
S2_EXIT_TYPE   = 'death_cross'  # 데스크로스 or SL 발생 시 청산

# ─────────────────────────────────────────────────────────────
# S3: EMA Trend Follow (4H)
# ─────────────────────────────────────────────────────────────
S3_EMA_FAST    = 20
S3_EMA_SLOW    = 50
S3_EMA_TREND   = 200
S3_ADX_PERIOD  = 14
S3_ADX_MIN     = 25      # ADX 최소값 (Wilder 원본 기준; 20은 약추세 과진입)
S3_ATR_PERIOD  = 14
S3_ATR_SL      = 2.5
S3_RR          = 2.0     # 기본 손익비 (동적 RR 미계산 시 폴백)
S3_RR_MIN      = 1.5     # 동적 RR 하한
S3_RR_MAX      = 3.0     # 동적 RR 상한
S3_COOLDOWN    = 2       # 청산 후 대기 봉 수 (TRENDING_UP 레짐)
S3_COOLDOWN_WEAK = 6     # WEAK_TREND 쿨다운 (4H봉 6개=24시간: 휩쏘 방지)

# ─────────────────────────────────────────────────────────────
# S4: RSI Mean Reversion (1D)
# ─────────────────────────────────────────────────────────────
S4_RSI_PERIOD  = 14
S4_RSI_ENTRY   = 30      # RSI가 이 수준을 다시 상향돌파 시 진입
S4_BB_PERIOD   = 20
S4_BB_SIGMA    = 2.0
S4_ATR_PERIOD  = 14
S4_ATR_SL      = 2.0     # SL 배수 (1.5→2.0: 과매도 구간 조기 SL 탈출 감소)
S4_RR          = 1.5     # 손익비 (기존 1:1→1.5: 기대가치 개선)
S4_MAX_HOLD    = 20      # 최대 보유 일수

# ─────────────────────────────────────────────────────────────
# S5: Bollinger Band Bounce (1D)
# ─────────────────────────────────────────────────────────────
S5_BB_PERIOD   = 20
S5_BB_SIGMA    = 2.2
S5_RSI_CONFIRM = 30      # RSI 확인 필터 (40 미만 시 과매도 확인)
S5_ATR_PERIOD  = 14
S5_ATR_SL      = 1.2
S5_EXIT_TYPE   = 'bb_upper'   # 상단밴드 터치 시 청산
S5_MAX_HOLD    = 12      # 최대 보유 일수 (MDD 제한)

# ─────────────────────────────────────────────────────────────
# S6: Donchian Breakout (1D, 터틀 트레이딩)
# ─────────────────────────────────────────────────────────────
S6_ENTRY_PERIOD = 20     # 20일 신고가 돌파
S6_EXIT_PERIOD  = 10     # 10일 신저가 이탈
S6_VOL_MA       = 20     # 거래량 이동평균 기간
S6_VOL_MULT     = 2.0    # 거래량 스파이크 배수 (1.3→2.0: 일상 노이즈 제거)
S6_VOL_VWAP_CONFIRM = True  # VWAP 상단 확증 필터 (가짜 브레이크아웃 차단)
S6_ATR_PERIOD   = 14
S6_ATR_SL       = 2.0
S6_RR           = 2.0    # 기본 손익비
S6_RR_MIN       = 1.5    # 동적 RR 하한
S6_RR_MAX       = 3.0    # 동적 RR 상한

# ─────────────────────────────────────────────────────────────
# S7: MACD Momentum (4H)
# ─────────────────────────────────────────────────────────────
S7_MACD_FAST   = 12
S7_MACD_SLOW   = 26
S7_MACD_SIG    = 9
S7_EMA_TREND   = 200
S7_ATR_PERIOD  = 14
S7_ATR_SL      = 2.0
S7_RR          = 2.0     # 기본 손익비
S7_RR_MIN      = 1.5     # 동적 RR 하한
S7_RR_MAX      = 3.0     # 동적 RR 상한
S7_HIST_EXIT_BARS = 2    # MACD 히스토그램 음수 연속 N봉 시 청산 (1→2: 노이즈 청산 방지)

# ─────────────────────────────────────────────────────────────
# 캔들 캐시
# ─────────────────────────────────────────────────────────────
SPOT_CANDLE_4H       = 300
SPOT_CANDLE_1D       = 600
SPOT_CANDLE_CACHE_TTL = 290   # 초 (4분 50초)
SPOT_PRICE_POLL_SEC  = 60

# ─────────────────────────────────────────────────────────────
# 백테스트 비용 모델
# ─────────────────────────────────────────────────────────────
BT_SPOT_FEE      = 0.001    # 스팟 수수료 0.1% (Taker)
BT_SPOT_SLIPPAGE = 0.0003   # 슬리피지 0.03% (BTCUSDT 기준 폴백 — 티어 미매칭 시)
BT_INITIAL_EQ    = 10_000.0 # 초기 자본 $10,000

# 심볼별 슬리피지 티어 (백테스트 비용 현실화)
BT_SLIPPAGE_BY_TIER = {
    'tier1': 0.0003,   # BTC, ETH: 0.03%
    'tier2': 0.0008,   # 주요 대형 코인: 0.08%
    'tier3': 0.0015,   # 중소형 나머지: 0.15%
}
BT_TIER1_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
BT_TIER2_SYMBOLS = [
    'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT',
    'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'MATICUSDT', 'UNIUSDT',
]

# 선물 펀딩비 → 현물 진입 필터 (추세추종 전략 오진입 방지)
FUNDING_LONG_BLOCK    = 0.0005   # 0.05%/8h 이상 → 롱 과밀, 추세추종 진입 차단
FUNDING_SHORT_BOOST   = -0.0001  # -0.01%/8h 이하 → 숏 쏠림, 리스크 스케일 +20%
FUNDING_APPLY_STRATS  = ['S3', 'S6', 'S7']  # 추세추종 전략에만 적용

# Walk-Forward 기간
WF_IS_START      = '2021-01-01'
WF_IS_END        = '2023-12-31'
WF_OOS_START     = '2024-01-01'
WF_OOS_MIN_SHARPE = 0.30
WF_OOS_MIN_PF     = 1.10

# ─────────────────────────────────────────────────────────────
# 전략 메타데이터
# ─────────────────────────────────────────────────────────────
STRATEGY_TIMEFRAMES = {
    'S1': '1d',
    'S2': '1d',
    'S3': '4h',
    'S4': '1d',
    'S5': '1d',
    'S6': '1d',
    'S7': '4h',
    'S7V4': '4h',
}

STRATEGY_NAMES = {
    'S1': 'Buy & Hold',
    'S2': 'SMA Golden Cross',
    'S3': 'EMA Trend Follow',
    'S4': 'RSI Mean Reversion',
    'S5': 'Bollinger Band Bounce',
    'S6': 'Donchian Breakout',
    'S7': 'MACD Momentum',
    'S7V4': 'MACD Momentum Enhanced',
}

# 레짐별 허용 전략
# 레짐별 전략 배치 원칙:
#   TRENDING_UP  : S6 (돌파 전용 — S3 고점 진입 제거)
#   RANGING      : S4, S5 (평균회귀 — ADX<20 구간, 추세추종 불가)
#   WEAK_TREND   : S3, S5, S6 (추세 형성 초기 혼합)
#   TRENDING_DOWN: S4, S5 (과매도 반등 한정, 30% 리스크 스케일)
#   CRISIS       : 전면 차단
# NOTE: S7V4는 ADX≥25 필터를 내부적으로 요구하므로 RANGING(ADX<20) 배치 불가
REGIME_STRATEGY_MAP = {
    'TRENDING_UP':   ['S6'],
    'RANGING':       ['S4', 'S5'],
    'WEAK_TREND':    ['S3', 'S5', 'S6'],
    'TRENDING_DOWN': ['S4', 'S5'],
    'CRISIS':        [],
}

# WEAK_TREND 리스크 스케일 (50% — 약추세에서 리스크 절반)
WEAK_TREND_RISK_SCALE    = 0.50
TRENDING_DOWN_RISK_SCALE = 0.30  # TRENDING_DOWN 구간 리스크 30% (백테스트 검증)

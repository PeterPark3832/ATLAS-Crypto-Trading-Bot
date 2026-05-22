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
UNIVERSE_REFRESH_HOURS    = 24           # 갱신 주기 (시간)
UNIVERSE_MOMENTUM_DAYS    = 90           # 모멘텀 랭킹 기간 (일)

# 스테이블코인 기초자산 제외 목록
UNIVERSE_STABLECOIN_BASE = {
    'USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'USDP', 'FDUSD',
    'PYUSD', 'UST', 'LUSD', 'FRAX', 'USDD', 'SUSD', 'GUSD',
    'HUSD', 'EURS', 'USTC', 'EUROC', 'PAXG', 'XAUT',
}
# 레버리지/인버스 토큰 키워드 제외
UNIVERSE_LEVERAGED_KEYWORDS = ['3L', '3S', '2L', '2S', 'UP', 'DOWN', 'BULL', 'BEAR']

# ─────────────────────────────────────────────────────────────
# 포트폴리오 리스크
# ─────────────────────────────────────────────────────────────
SPOT_MAX_POSITIONS     = 10      # 최대 동시 포지션 수
SPOT_BASE_RISK_PCT     = 0.010   # 거래당 기본 리스크 1%
SPOT_MAX_ALLOC_PCT     = 0.15    # 단일 종목 최대 배분 15%
SPOT_MIN_ALLOC_PCT     = 0.02    # 단일 종목 최소 배분 2%
SPOT_RESERVE_PCT       = 0.10    # USDT 최소 예비금 10%
SPOT_DAILY_LOSS_LIMIT  = -0.04   # 일간 손실 한도 -4%
SPOT_MIN_ORDER_USDT    = 10.0    # 최소 주문 금액 (Binance 기준)

# Kelly / Ratchet (선물봇과 동일 로직)
SPOT_KELLY_MIN_TRADES  = 20
SPOT_KELLY_SCALE_MIN   = 0.30
SPOT_KELLY_SCALE_MAX   = 1.50
SPOT_RATCHET_DD_THRESH = 0.05    # 5% 하락 → 70% 리스크
SPOT_RATCHET_DD_HARD   = 0.08    # 8% 하락 → 40% 리스크
SPOT_RATCHET_RECOVER   = 0.25    # 25% 회복 시 +10% 리스크 스케일

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
S3_ADX_MIN     = 20      # ADX 최소값 (추세 강도 필터)
S3_ATR_PERIOD  = 14
S3_ATR_SL      = 2.5
S3_RR          = 2.0     # 고정 손익비
S3_COOLDOWN    = 2       # 청산 후 대기 봉 수

# ─────────────────────────────────────────────────────────────
# S4: RSI Mean Reversion (1D)
# ─────────────────────────────────────────────────────────────
S4_RSI_PERIOD  = 14
S4_RSI_ENTRY   = 30      # RSI가 이 수준을 다시 상향돌파 시 진입
S4_BB_PERIOD   = 20
S4_BB_SIGMA    = 2.0
S4_ATR_PERIOD  = 14
S4_ATR_SL      = 1.5
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
S6_VOL_MULT     = 1.3    # 거래량 스파이크 배수 (1.3x 이상)
S6_ATR_PERIOD   = 14
S6_ATR_SL       = 2.0
S6_RR           = 2.0

# ─────────────────────────────────────────────────────────────
# S7: MACD Momentum (4H)
# ─────────────────────────────────────────────────────────────
S7_MACD_FAST   = 12
S7_MACD_SLOW   = 26
S7_MACD_SIG    = 9
S7_EMA_TREND   = 200
S7_ATR_PERIOD  = 14
S7_ATR_SL      = 2.0
S7_RR          = 2.0

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
BT_SPOT_SLIPPAGE = 0.0003   # 슬리피지 0.03%
BT_INITIAL_EQ    = 10_000.0 # 초기 자본 $10,000

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
# Live 운용 (C안 최적화, 백테스트 검증):
#   TRENDING_UP  : S6만  (S3 고점진입 제거)
#   RANGING      : S7V4  (MACD 강화필터, PF=2.17, 승률57%)
#   WEAK_TREND   : S3+S5+S6 (추세 형성 초기 전 전략)
#   TRENDING_DOWN: S5+S7V4 30%scale (과매도 반등)
# 결과: PF 1.76→1.80 / MDD 7.2% 유지 / CAGR +8.5%→+9.6% / Sharpe 2.54→2.59
# WF: IS PF=1.12 / OOS PF=3.52 ✅
REGIME_STRATEGY_MAP = {
    'TRENDING_UP':   ['S6'],
    'RANGING':       ['S7V4'],
    'WEAK_TREND':    ['S3', 'S5', 'S6'],
    'TRENDING_DOWN': ['S5', 'S7V4'],
    'CRISIS':        [],
}

# WEAK_TREND 리스크 스케일 (70%)
WEAK_TREND_RISK_SCALE    = 0.70
TRENDING_DOWN_RISK_SCALE = 0.30  # TRENDING_DOWN 구간 리스크 30% (백테스트 검증)

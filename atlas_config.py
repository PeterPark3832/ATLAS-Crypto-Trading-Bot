"""
ATLAS — 통합 트레이딩 봇 설정
==============================
Phoenix 4H (BTC/SOL/BNB) + Equinox 1D (ETH) + Bounce 1D (ETH)
장세별 전략 자동 라우팅 · 통합 리스크 관리 · 연 20% / MDD 10% 목표

[심볼 역할 분리]
  BTC, SOL, BNB → Phoenix 4H  M1(눌림목) + M2(BB스퀴즈) LONG/SHORT
  ETH           → Equinox 1D LONG  + Bounce 1D LONG

[제거 근거 (백테스트)]
  SOLSTICE SHORT : 전 심볼 음수 (BTC -5.9%, ETH -3.6%, SOL -3.7%, XRP -15.7%)
  XRP            : Equinox LONG -25%, Solstice SHORT -15.7%
  DOGE           : Phoenix -10.55%, PF 0.87
  SOL Equinox    : OOS -0.8% (Phoenix SOL +29.2% → Phoenix로 대체)
  ETH Phoenix    : MDD 14.9% → Equinox/Bounce로 대체
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / '.env'
load_dotenv(_env_path if _env_path.exists() else None)

# ─────────────────────────────────────────────────────────────
# API / Telegram
# ─────────────────────────────────────────────────────────────
def _req(key: str) -> str:
    v = os.getenv(key, '').strip()
    if not v:
        raise RuntimeError(f"[설정 오류] 환경변수 '{key}' 없음. .env 파일을 확인하세요.")
    return v

BINANCE_API_KEY    = _req('BINANCE_API_KEY')
BINANCE_API_SECRET = _req('BINANCE_API_SECRET')
TG_TOKEN           = _req('TG_TOKEN')
TG_CHAT_ID         = _req('TG_CHAT_ID')

# ─────────────────────────────────────────────────────────────
# 파일 경로
# ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DB_FILE          = BASE_DIR / 'state' / 'atlas.db'
LOG_DIR          = BASE_DIR / 'logs'
KILL_SWITCH_FILE = Path('/tmp/ATLAS_STOP')

# ─────────────────────────────────────────────────────────────
# 전략 심볼 배분
# ─────────────────────────────────────────────────────────────
PHX_SYMBOLS = ['BTCUSDT', 'SOLUSDT', 'BNBUSDT']   # Phoenix 4H 롱+숏
EQ_SYMBOLS  = ['ETHUSDT']                           # Equinox 1D 롱
BN_SYMBOLS  = ['ETHUSDT']                           # Bounce  1D 롱
ALL_SYMBOLS = list(dict.fromkeys(PHX_SYMBOLS + EQ_SYMBOLS))  # 중복 제거

# ─────────────────────────────────────────────────────────────
# 포트폴리오 전역 리스크
# ─────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS  = 5       # 전략·심볼 합산 최대 동시 포지션
PORTFOLIO_VAR_CAP   = 0.10    # 포트폴리오 총 리스크 10% 초과 시 신규 진입 차단
DAILY_LOSS_LIMIT    = -0.03   # 당일 -3% 손실 시 봇 정지
CORR_SCALE_FLOOR    = 0.40    # 상관계수 보정 최소 배율 (corr=1 → ×0.40)
VOL_PARITY_FLOOR    = 0.50    # 변동성 패리티 최소 배율
VOL_PARITY_CAP      = 1.20    # 변동성 패리티 최대 배율

# ─────────────────────────────────────────────────────────────
# Phoenix 4H 파라미터 (과최적화 방지: 심볼별 RSI/BB 미분리)
# ─────────────────────────────────────────────────────────────
# 공통 지표
PHX_EMA_TREND     = 200
PHX_MACD_FAST     = 12
PHX_MACD_SLOW     = 26
PHX_MACD_SIGNAL   = 9
PHX_RSI_PERIOD    = 14
PHX_ATR_PERIOD    = 14
PHX_BB_PERIOD     = 20
PHX_BB_STD        = 2.0
PHX_SQUEEZE_LB    = 20    # BB 스퀴즈 최솟값 룩백
PHX_VOL_LB        = 20    # 거래량 평균 룩백

# 공통 신호 임계값 (심볼별 미분리)
PHX_RSI_LONG_MIN  = 30
PHX_RSI_LONG_MAX  = 65
PHX_RSI_SHORT_MIN = 35
PHX_RSI_SHORT_MAX = 75
PHX_SQUEEZE_MULT  = 1.05  # BB폭이 최소의 1.05배 이하면 스퀴즈
PHX_VOL_MULT      = 1.5   # 거래량 평균의 1.5배 초과 = 스파이크
PHX_ATR_MIN_PCT   = 0.003 # 진입가 대비 ATR 최소 0.3% (저변동 필터)

# 동적 RR 범위
PHX_RR_MIN = 1.5
PHX_RR_MAX = 3.5

# 재진입 쿨다운
PHX_COOLDOWN_CANDLES = 2   # 청산 후 2캔들(8H) 재진입 금지

# BEP (손익분기 SL 이동)
PHX_BEP_ENABLED     = True
PHX_BEP_TRIGGER_RR  = 1.0  # 리스크의 100% 이익 도달 시 SL → 진입가 (0.5→1.0: 조기 BEP 과다 방지)

# 추가 필터
PHX_BLOCK_WEEKEND       = True   # 토/일 UTC 신규 진입 금지
PHX_FUNDING_LONG_MAX    = 0.0005  # 펀딩비 +0.05% 초과 → LONG 차단
PHX_FUNDING_SHORT_MIN   = -0.0003 # 펀딩비 -0.03% 미만 → SHORT 차단
PHX_DAILY_TREND_FILTER  = True   # 1D EMA200 역행 차단
PHX_DAILY_TREND_BUF     = 0.03   # EMA200 경계 3% 완충

# 심볼별 파라미터 (leverage/risk/atr_sl/ema_entry만 심볼별 설정)
# atr_sl_mult 조정 근거: 진단 결과 평균 SL 보유시간 5.3봉(21H) → SL이 너무 타이트
# BTC/SOL 2.0→3.0, BNB 1.5→2.5 (구조적 수정, 커브피팅 아님)
PHX_SYMBOL_PARAMS = {
    'BTCUSDT': {'leverage': 5,  'risk_pct': 0.010, 'atr_sl_mult': 3.0, 'ema_entry': 30},
    'SOLUSDT': {'leverage': 2,  'risk_pct': 0.008, 'atr_sl_mult': 3.0, 'ema_entry': 30},
    'BNBUSDT': {'leverage': 3,  'risk_pct': 0.006, 'atr_sl_mult': 2.5, 'ema_entry': 30},
}

# ─────────────────────────────────────────────────────────────
# Equinox 1D 파라미터
# ─────────────────────────────────────────────────────────────
EQ_EMA_FAST  = 20
EQ_EMA_SLOW  = 60
EQ_SL_ATR    = 2.0
EQ_TP_ATR    = 4.0
EQ_RISK_PCT  = 0.012  # 1.2% 리스크
EQ_LEVERAGE  = 3      # 3배 레버리지 (Equinox 원본 10x → 보수적 조정)

# OOS 대응 추가 필터
EQ_ATR_FILTER_PCT   = 0.060  # ATR/종가 > 6.0% → 고변동성 구간 진입 차단
EQ_EMA_SLOPE_BARS   = 3      # EMA fast가 N봉 전보다 낮으면 진입 차단
EQ_EMA200_SLOPE_BARS = 10    # ETH EMA200이 N봉 전보다 낮으면(장기하락) 진입 차단
                              # BTC 레짐과 독립적으로 ETH 자체 추세 확인
EQ_MOMENTUM_BARS     = 20    # ETH 종가가 N봉 전보다 낮으면(최근 하락세) 진입 차단
                              # 2024-2025 BTC랠리-ETH약세 구간 고점추격 방지

# Walk-Forward EMA 최적화 (매주 일요일)
EQ_EMA_FAST_CANDIDATES  = [10, 15, 20, 25]
EQ_EMA_SLOW_CANDIDATES  = [40, 50, 60, 80]
EQ_WF_OOS_MIN_SHARPE    = 0.30   # OOS Sharpe 최소 기준
EQ_WF_OOS_IMPROVE       = 0.05   # 현행 대비 개선 최소폭

# 4H 멀티타임프레임 필터 (1D 신호 감지 후 4H EMA 방향 확인)
EQ_MTF_FILTER = True

# ─────────────────────────────────────────────────────────────
# Bounce 1D 파라미터 (ETH 과매도 반등)
# ─────────────────────────────────────────────────────────────
BN_EMA_FAST   = 20
BN_EMA_SLOW   = 60
BN_RSI_ENTRY  = 35    # RSI 35 미만 = 과매도
BN_BB_PERIOD  = 20
BN_BB_SIGMA   = 2.0
BN_SL_ATR     = 1.5
BN_TP_ATR     = 3.0
BN_RISK_PCT   = 0.008  # 0.8% 리스크
BN_LEVERAGE   = 2

# ─────────────────────────────────────────────────────────────
# 장세 분류 임계값 (BTC 기준, get_regime() 사용)
# ─────────────────────────────────────────────────────────────
REGIME_ADX_TREND    = 25    # ADX ≥ 25 = 추세장
REGIME_ADX_WEAK     = 20    # ADX < 20 = 횡보장
REGIME_CRISIS_ATR   = 0.07  # ATR/Price ≥ 7% = 위기 (신규 진입 전면 차단)
REGIME_BTC_LOOKBACK = 50    # ADX 계산 캔들 수

# ─────────────────────────────────────────────────────────────
# Kelly + Drawdown Ratchet
# ─────────────────────────────────────────────────────────────
KELLY_MIN_TRADES    = 20    # 최소 거래수 미만 → 기본 리스크 사용
KELLY_SCALE_MIN     = 0.30
KELLY_SCALE_MAX     = 1.50
RATCHET_DD_THRESH   = 0.05  # 낙폭 5% → 리스크 70%
RATCHET_DD_HARD     = 0.07  # 낙폭 7% → 리스크 40%
RATCHET_RECOVER_PCT = 0.20  # 저점 대비 20% 회복마다 +10% 복원

# ─────────────────────────────────────────────────────────────
# 실행 엔진
# ─────────────────────────────────────────────────────────────
LIMIT_OFFSET_PCT    = 0.0002   # 지정가 오프셋 0.02%
LIMIT_TIMEOUT_SEC   = 60       # 지정가 미체결 → 시장가 폴백
TWAP_RISK_THRESHOLD = 0.012    # 리스크 ≥ 1.2% 시 TWAP 3분할
TWAP_SPLITS         = 3
TWAP_INTERVAL_SEC   = 10

# ─────────────────────────────────────────────────────────────
# 백테스트 현실화 비용
# ─────────────────────────────────────────────────────────────
BT_FEE_TAKER   = 0.0005   # Binance 선물 테이커 수수료 0.05% (진입·청산 각 1회)
BT_SLIPPAGE    = 0.0003   # 시장가 슬리피지 추정 0.03% (진입 시 1회)

# ─────────────────────────────────────────────────────────────
# 캔들 캐시 / 루프 타이머
# ─────────────────────────────────────────────────────────────
CANDLE_CACHE_TTL    = 290      # 4분 50초 (캔들 중복 API 방지)
PRICE_POLL_SEC      = 30       # 포지션 감시 주기
CANDLE_LIMIT_4H     = 250
CANDLE_LIMIT_1D     = 300      # EMA200 + WF 검증 충분히
WF_CANDLE_LIMIT     = 500      # Walk-Forward 전용: IS 75% + OOS 25% 통계적 유의성 확보

# ─────────────────────────────────────────────────────────────
# Module C — 15m 브레이크아웃 모멘텀
# ─────────────────────────────────────────────────────────────
# [과최적화 방지 원칙]
#   ATR 기간(14), 볼륨 룩백(20) → A/B와 동일 파라미터 재사용
#   RR 2.0 고정 (수수료 손익분기 확보, 튜닝 대상 아님)
#   심볼별 신호 파라미터 미분리 (레버리지/리스크만 변동성 차등)
V2_MC_SYMBOLS     = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
V2_MC_ATR_PERIOD  = 14      # A/B와 동일 (신규 하이퍼파라미터 없음)
V2_MC_ATR_SL      = 1.0     # SL = ATR × 1.0
V2_MC_RR          = 2.0     # 고정 RR — 변경 금지 (수수료 드래그 임계값)
V2_MC_VOL_LB      = 20      # 거래량 평균 룩백
V2_MC_VOL_MULT    = 1.5     # 거래량 스파이크 배수
V2_MC_HTF_EMA     = 21      # 1H EMA 방향 필터 (단일 파라미터)
V2_MC_CHASE_LIMIT = 0.003   # 브레이크아웃 레벨 대비 최대 추격 거리 0.3% (슬리피지 제한)
V2_MC_COOLDOWN    = 3       # 청산 후 재진입 금지 봉수 (×15분 = 45분)
V2_MC_LIMIT_SEC   = 30      # 지정가 타임아웃 — 시장가 폴백 없음 (슬리피지 제어)

# 심볼별 레버리지/리스크 (변동성 차등: SOL 고변동 → 레버리지 축소)
V2_MC_LEVERAGE = {'BTCUSDT': 5, 'ETHUSDT': 5, 'SOLUSDT': 3}
V2_MC_RISK_PCT = {'BTCUSDT': 0.010, 'ETHUSDT': 0.010, 'SOLUSDT': 0.008}

CANDLE_LIMIT_15M  = 100     # 15분봉 캔들 수 (ATR14 + VOL20 + 여유분)

# Module D — 펀딩비 차익 (Funding Rate Harvest)
# ─────────────────────────────────────────────────────────────
# 펀딩비 > 임계값 → SHORT 진입, 8h마다 펀딩비 수령
# 정상화(<0.01%) 또는 48h 초과 시 청산. 레짐 무관 (CRISIS만 제외)
# 지정가 전용 (Maker 0.02%) — 수수료 최소화 핵심
V2_MD_SYMBOLS        = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
V2_MD_FUNDING_ENTER  = 0.0005   # 진입 임계값: +0.05%/8h 이상
V2_MD_FUNDING_EXIT   = 0.0001   # 청산 임계값: +0.01%/8h 이하로 정상화
V2_MD_LEVERAGE       = 2        # 레버리지 (방향성 리스크 최소화)
V2_MD_RISK_PCT       = 0.010    # 거래당 리스크 1.0%
V2_MD_ATR_SL         = 3.0      # SL = ATR × 3.0 (넓게 — 펀딩 수집이 목적)
V2_MD_ATR_PERIOD     = 14
V2_MD_MAX_HOURS      = 48       # 최대 보유 48h (6회 수집 후 강제 청산)
V2_MD_LIMIT_SEC      = 30       # 지정가 타임아웃
V2_MD_CHECK_SEC      = 300      # 펀딩비 확인 주기 (5분)

# ─────────────────────────────────────────────────────────────
# ATLAS v2 심볼 배분
# ─────────────────────────────────────────────────────────────
V2_MA_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']  # Module A: 추세추종 4H
V2_MR_SYMBOLS = ['ETHUSDT', 'BNBUSDT']                         # Module B: 평균회귀 1D

# ─────────────────────────────────────────────────────────────
# Module A: Trend Follow 4H
# ─────────────────────────────────────────────────────────────
# EMA(20) 교차 EMA(50) 신호 + close vs EMA200 필터
# 레짐: TRENDING_UP → LONG, TRENDING_DOWN → SHORT, RANGING → 스킵
# SL: ATR×2.5 / Trailing: BEP(1R) 후 peak/trough - ATR×2.5
V2_MA_EMA_FAST   = 20      # 진입 신호 EMA (fast)
V2_MA_EMA_SLOW   = 50      # 진입 신호 EMA (slow)
V2_MA_ATR_PERIOD = 14      # ATR 기간
V2_MA_ATR_SL     = 2.5     # SL/Trailing 배수
V2_MA_TRAIL_R    = 1.0     # BEP 트리거: 1R 이익 시 SL → 진입가
V2_MA_RISK_PCT   = 0.018   # 거래당 리스크 1.8% (1.0% → 1.8% / MDD ~12% 허용 범위 내 수익 최대화)
V2_MA_LEVERAGE   = 5       # 레버리지 (리스크 계산용)
V2_MA_COOLDOWN   = 2       # 청산 후 쿨다운 봉수 (4H 기준 → 8H)
V2_MA_ADX_MIN    = 20      # EMA 크로스 진입 시 ADX 최소값 (방향성 모멘텀 확인)

# ─────────────────────────────────────────────────────────────
# Module B: Mean Reversion 1D
# ─────────────────────────────────────────────────────────────
# RSI < 30 + 종가 ≤ BB_lower → LONG / RSI > 70 + 종가 ≥ BB_upper → SHORT
# 레짐: RANGING 전용 / TP: BB_mid(SMA20) / 최대보유: 15봉
V2_MR_RSI_PERIOD = 14
V2_MR_RSI_LONG   = 35      # RSI < 35 = 과매도 구간 (회복 크로스 후 LONG)
V2_MR_RSI_SHORT  = 65      # RSI > 65 = 과매수 구간 (하락 크로스 후 SHORT)
V2_MR_BB_PERIOD  = 20      # BB / SMA 기간
V2_MR_BB_SIGMA   = 2.0     # BB 표준편차 배수
V2_MR_ATR_PERIOD = 14      # ATR 기간
V2_MR_ATR_SL     = 1.5     # SL 배수
V2_MR_MAX_BARS   = 15      # 시간청산: 최대 15봉 (15일)
V2_MR_RISK_PCT     = 0.008   # 거래당 리스크 0.8% (ETH 기준)
V2_MR_RISK_PCT_BNB = 0.004   # BNB 한정 0.4% — IS 7건으로 통계 불충분, 데이터 축적 후 상향
V2_MR_LEVERAGE     = 2       # 레버리지

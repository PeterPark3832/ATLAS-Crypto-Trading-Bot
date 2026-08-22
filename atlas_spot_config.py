"""
ATLAS Spot Trading Bot — 설정
==============================
현물(Spot) 전용 파라미터. 선물 config(atlas_config.py)와 완전히 분리.

전략 목록 (구현 8개 — 라이브 진입은 아래 레짐 맵에 배정된 4개뿐):
  S1  : Buy & Hold (벤치마크)
  S2  : SMA Golden Cross       1D
  S3  : EMA Trend Follow       4H  (v2 Module A 롱 전용 변환)
  S4  : RSI Mean Reversion     1D  (v2 Module B 롱 전용 변환)
  S5  : Bollinger Band Bounce  1D
  S6  : Donchian Breakout      1D  (터틀 트레이딩)
  S7  : MACD Momentum          4H
  S7V4: MACD Momentum(강화)    4H  (b470232에서 벤치 — 레짐 미배정)

레짐별 활성 전략 (REGIME_STRATEGY_MAP 이 단일 출처):
  TRENDING_UP  : S6            (돌파)
  RANGING      : S4, S5        (평균회귀)
  WEAK_TREND   : S3, S5, S6    (추세 형성 초기, 리스크 50%)
  TRENDING_DOWN: S4            (과매도 반등 한정 — S5는 실전 0승으로 제외, 리스크 30%)
  MICRO_RANGING: 전면 차단     (1D는 추세인데 4H ADX<20 — 실제로는 횡보)
  CRISIS       : 전면 차단
  UNKNOWN      : 전면 차단
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / '.env'
load_dotenv(_env_path if _env_path.exists() else None)


def _opt(key: str, default: str = '') -> str:
    return os.getenv(key, default).strip()


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
# 사이징의 두 축: 거래당 리스크(SPOT_BASE_RISK_PCT)와 집중도 제한
# (SPOT_MAX_ALLOC_PCT). 둘의 관계를 모르면 설정을 잘못 읽는다.
#
#   주문 명목가 = min( 자본 × 리스크 / SL거리 ,  자본 × 배분상한 )
#
# 상한이 걸리는 조건은 **SL거리 < 리스크 ÷ 배분상한 = 0.020/0.15 = 13.3%**
# 이다. 전형 SL은 5% 안팎이므로 **거의 모든 거래가 상한 쪽에서 결정**된다.
#
#   SL  2% → 실효 0.30%   ← 배분상한
#   SL  5% → 실효 0.75%   ← 배분상한 (전형 구간)
#   SL 10% → 실효 1.50%   ← 배분상한
#   SL 13.3% 이상 → 실효 2.00% (= 설정값. 여기서부터 상한이 안 걸린다)
#
# 읽는 법: SPOT_BASE_RISK_PCT는 **거래당 리스크의 상한(ceiling)** 이고,
# 상한이 걸리는 구간에서는 실효 리스크가 `배분상한 × SL거리`가 된다.
# 즉 SL이 넓을수록(=불확실할수록) 리스크가 커지는 방향이다 — 리스크 기반
# 사이징의 취지(SL과 무관하게 일정)와는 반대다. 이 사실 자체는
# `_report_risk_profile`이 기동 시마다 로그로 남긴다.
#
# ── 그런데 왜 값을 "정직하게" 0.0075로 안 내리는가 ──────────────
# ① 포트폴리오 관점에서는 지금이 오히려 맞다.
#    가용자본 90%를 종목당 15%로 나누면 동시 보유는 **약 6종목**이고,
#    6종목이 동시에 손절당하면 손실은 90% × 5%(전형 SL) = **4.5%**다.
#    일간 손실 한도(-4%)와 거의 같은 눈금이다.
#    반면 2%가 문자 그대로 먹히면 6 × 2% = **12%** — 일간 한도의 3배다.
#    즉 배분 상한이 조용히 **옳은 포트폴리오 리스크를 만들어 주고 있다.**
#
# ② 값을 내리면 소액 구간에서 거래가 끊긴다. 배분 상한은 Kelly에 곱해지지
#    않으므로, Kelly가 1.0일 땐 주문 크기가 같지만 **연패로 Kelly가 내려간
#    구간**에서만 주문이 작아진다:
#        Kelly 1.00 →  $45.00 → $45.00  (동일)
#        Kelly 0.15 →  $18.00 →  $6.75  (−62%)
#    그 결과 "연패 중에도 전 조합이 멈추지 않는 최소 자본"이
#    **$556 → $1,481**로 오른다. 회복이 가장 필요한 순간에 주문이 $5
#    NOTIONAL 아래로 떨어져 조합이 죽는다 — 이 저장소에서 가장 비쌌던
#    버그가 정확히 그것이었다(dc7fb24: 하락장 전략이 통째로 미체결).
#
# 판단 기준: **자본 $1,500 이상**이 되면 0.0075로 내려 설정과 실제를
# 일치시키는 편이 낫다. 그 아래에서는 현행(0.020)이 낫다.
# 바꾸기 전에 `python capital_plan.py --equity <자산>`로 문턱을 확인할 것.
# 상한 자체를 올리는 방향은 답이 아니다 — 상한을 40%까지 올려야 2%가
# 살아나는데, 그건 6종목 포트폴리오를 2~3종목으로 몰아넣는 것과 같다.
SPOT_MAX_POSITIONS     = 6       # 최대 동시 보유 종목 수 (15 → 6, 2026-08).
                                 # 자본이 지탱하는 종목 수는 **Kelly에 따라
                                 # 변한다**. 포지션 하나의 크기가
                                 #   min(자본×리스크×스케일/SL거리, 자본×15%)
                                 # 이므로, Kelly가 1.0이면 종목당 15%(=6개면
                                 # 가용자본 90%를 다 쓴다), Kelly가 하한
                                 # (0.15)이면 종목당 약 4.6%라 19개까지 들어간다.
                                 # ⚠️ 그래서 이 값은 장식이 아니다. 실제로
                                 # 2026-08 라이브에서 Kelly가 하한에 붙어
                                 # **15개가 전부 채워진 상태**였다(합계 $407 /
                                 # 자산 $580). 이전 주석이 "실제 한계는 6개라
                                 # 값을 낮춰도 동작이 같다"고 했던 것은 Kelly=1.0
                                 # 만 가정한 오류다.
                                 # 6으로 두는 의미: 종목 수를 집중도 기준으로
                                 # 고정한다. Kelly가 회복되면 6종목이 곧 만기
                                 # 투입이고, Kelly가 낮은 구간에서는 현금이
                                 # 남는다 — 연패 중 노출을 줄이는 쪽을 택한 것.
SPOT_BASE_RISK_PCT     = 0.020   # 거래당 리스크 **상한**.
                                 # ⚠️ SL거리 13.3%(=0.020/0.15) 미만이면
                                 # 배분상한이 먼저 걸려 실효 리스크는
                                 # `배분상한 × SL거리`가 된다(전형 SL 5%에서
                                 # 0.75%). 위 주석 블록을 반드시 읽을 것.
SPOT_MAX_ALLOC_PCT     = 0.15    # 단일 종목 최대 배분 15%.
                                 # 백스톱이 아니라 **사실상의 주 사이징 축**이다
                                 # (전형 SL 구간에서 이쪽이 먼저 걸린다).
SPOT_MIN_ALLOC_PCT     = 0.02    # 단일 종목 최소 배분 2%
SPOT_RESERVE_PCT       = 0.10    # USDT 최소 예비금 10%
SPOT_DAILY_LOSS_LIMIT  = -0.04   # 일간 손실 한도 -4%
SPOT_MIN_ORDER_USDT    = 5.0     # 최소 주문 금액 (Binance NOTIONAL 실제 기준 $5)
SPOT_MAX_SL_PCT        = 0.20    # SL 거리 상한 20% (초과 시 포지션이 너무 작아 차단)

# 거래소 측 스탑 주문 (봇 다운 중에도 SL 집행 — 소프트웨어 SL은 백업)
SPOT_EXCHANGE_STOP     = True    # STOP_LOSS_LIMIT 주문 사용 여부
SPOT_STOP_LIMIT_GAP    = 0.005   # 지정가 = 트리거가 × (1 - 0.5%) — 급락 시 미체결 방지 버퍼
SPOT_EXCHANGE_OCO      = True    # SL+TP를 OCO로 함께 등록 (동적 TP 전략 S5는 자동 제외)

# ─────────────────────────────────────────────────────────────
# 리스크 사이징 자기교정 (2026-07 매매 로직 점검)
# ─────────────────────────────────────────────────────────────
SPOT_KELLY_FRACTION    = 0.5     # half-Kelly — raw Kelly는 승률 추정오차에 과베팅
                                 # (표본 10건 수준에서 분산이 커 기하성장·파산확률 모두 악화)
SPOT_EQUITY_PER_SLOT   = 20.0    # 동시 포지션 1개당 최소 자본 $20
                                 # (소액 계좌에서 포지션 과분할 → NOTIONAL 턱걸이 + 수수료 드래그 방지)
SPOT_HEALTH_MIN_TRADES = 20      # 전략 건강도 판정 최소 표본 (미만이면 개입 없음)
# ─────────────────────────────────────────────────────────────
# 비용 대비 엣지 (거래 1건의 기대값 하한)
# ─────────────────────────────────────────────────────────────
# 포지션 명목가 = 리스크금액 / SL거리% 이므로,
#   왕복비용(명목가 기준) = 리스크금액 × (비용률 / SL거리%)
#   ∴ 순기대값 = 리스크금액 × [ avg_r − 비용률/SL거리% ]
# 즉 **SL이 좁게 잡힌 신호일수록 비용이 R을 크게 잠식**한다.
# 예) 왕복 0.3%(수수료+스프레드+슬리피지) 기준
#     SL 5% → 0.06R,  SL 3% → 0.10R,  SL 1.5% → 0.20R
# ─────────────────────────────────────────────────────────────
# 트레일링 스탑 (기본 OFF — 검증 전에는 켜지 말 것)
# ─────────────────────────────────────────────────────────────
# peak_price는 이미 기록되고 있었으나 손절을 따라 올리는 데 쓰이지 않아
# 효과가 0이었다. 추세추종에서 청산은 수익의 상당 부분을 좌우하므로
# 기능을 살리되, **검증되지 않은 매매 변경**이므로 기본값은 OFF다.
#
# 동작: 가격이 +ACTIVATE_R 만큼 유리하게 움직이면 그때부터
#       SL = max(기존SL, 최고가 − SL거리 × TRAIL_MULT) 로 따라 올린다.
#       (내리지 않는다. ACTIVATE_R=TRAIL_MULT=1.0이면 활성화 시점이
#        곧 본전 이동이 되고 이후 1R 폭으로 추적한다)
# SL 거리를 기준으로 하므로 심볼·변동성에 무관하게 스케일이 맞는다.
#
# 켜기 전 검증: reoptimize.py / 월간 WFO로 IS→OOS 개선을 확인할 것.
# 추세추종(S3·S6)에는 대체로 유리하고 평균회귀(S4·S5)에는 불리한 경향이
# 있으므로, 전 전략 일괄 적용 전에 전략별로 확인하는 편이 안전하다.
SPOT_TRAIL_ENABLED     = False
SPOT_TRAIL_ACTIVATE_R  = 1.0     # 이 R배수만큼 이익이 난 뒤부터 추적 시작
SPOT_TRAIL_MULT        = 1.0     # 최고가에서 (SL거리 × 이 값)만큼 아래로 추적
SPOT_TRAIL_REARM_FRAC  = 0.25    # 거래소 스탑 재등록 기준: SL이 SL거리의 이
                                 # 비율 이상 움직였을 때만 (주문 과다 방지)

SPOT_MAX_COST_PER_R    = 0.20    # 왕복비용이 1R의 이 비율을 넘으면 진입 차단
SPOT_EDGE_MIN_TRADES   = 30      # 실현 avg_r로 차단 판정할 최소 표본
                                 # (20건 남짓의 avg_r은 표준오차가 커 오차단 위험)
SPOT_ASSUMED_SLIP_PCT  = 0.0005  # 시장가 체결의 편도 슬리피지 가정 (호가 이탈분)
SPOT_DEFAULT_SPREAD_PCT = 0.0010 # 호가 조회 실패 시 보수적 스프레드 가정

SPOT_CAPITAL_FLOW_PCT  = 0.05    # 포지션이 없을 때 실현손익으로 설명되지 않는
                                 # 자산 변동이 이 비율을 넘으면 입출금으로 보고
                                 # 드로다운 기준(피크)을 재조정한다.

SPOT_LOG_MAX_BYTES     = 50 * 1024 * 1024   # 로그 파일 1개 최대 50MB (로테이션)
SPOT_LOG_BACKUPS       = 5                  # 보관 개수 → 최대 300MB에서 고정

SPOT_HEALTH_WINDOW_DAYS = 45     # 판정 대상 기간. 차단되면 신규 거래가 없어 표본이
                                 # 고정되므로, 시간 창이 없으면 영구 차단이 된다.
                                 # 창이 지나면 표본이 최소치 미만이 되어 자동 해제.
SPOT_HEALTH_PF_SOFT    = 1.0     # 실계좌 net PF < 1.0 → 해당 전략 리스크 50% 감봉
SPOT_HEALTH_SOFT_SCALE = 0.5
SPOT_HEALTH_PF_HARD    = 0.7     # 실계좌 net PF < 0.7 → 해당 전략 신규 진입 차단

# ── 자기주도 학습 (atlas_learning) ────────────────────────────────
# 켜면 Kelly·건강도를 **대체**한다(곱하지 않는다). 스케일을 곱할수록
# 주문이 작아져 거래소 최소액 아래로 내려가는 문제가 있었고, 학습기는
# Kelly가 하려던 일(성과 기반 배분)을 레짐까지 나눠서 더 정교하게 한다.
#
# ⚠️ 기본 OFF. 켜기 전에 반드시 확인할 것:
#    python capital_plan.py --equity <자산>   # 최소주문 문턱
#    python atlas_learning.py                 # 현재 이력으로 어떤 배분이 나오는지
#    이력이 없으면 전 조합이 '미검증'이라 배분이 SPOT_LEARN_UNPROVEN_SCALE로
#    시작한다. 소액 계좌에서는 이것만으로 주문이 $5 미만이 될 수 있다.
SPOT_LEARN_ENABLED       = False
SPOT_LEARN_HALF_LIFE_DAYS = 45.0  # 증거 반감기 — 시장 비정상성 대응
SPOT_LEARN_MIN_INFO      = 8.0    # 이보다 정보량이 적으면 개입하지 않는다
SPOT_LEARN_UNPROVEN_SCALE = 0.25  # 미검증 조합 배분(증명 전엔 작게)
SPOT_LEARN_FLOOR         = 0.25   # 배분 하한 — 흡수상태 방지
SPOT_LEARN_MAX_SCALE     = 1.50   # 배분 상한
SPOT_LEARN_GAIN          = 0.50   # 팔 간 차이 1σ당 배분 변화폭
SPOT_LEARN_COST_PER_R    = 0.04   # 수수료가 1R에서 차지하는 비중.
                                  # 슬리피지는 넣지 말 것 — pnl_r이 실제
                                  # 체결가 기반이라 이미 반영돼 있다
# ── BNB 수수료 잔고 감시 ──────────────────────────────────────────
# "BNB로 수수료 지불"이 켜져 있어도 **BNB 잔고가 0이면 조용히 기초자산
# 차감으로 되돌아간다.** 그러면 두 가지가 동시에 일어난다:
#   ① 25% 할인을 못 받는다 (0.075% → 0.1%)
#   ② 체결량 > 보유량이 되어 보호주문이 -2010으로 거부된다
#      (실제로 ADA·RIF·FET 3종목에서 발생했다 — 3fa6074)
# 잔고가 바닥나기 **전에** 알려야 의미가 있다.
# 임계값은 **달러가 아니라 소모일수**로 잡는다.
# 고정 $5는 자본이 커지면 무의미해진다 — 자본 $10,000이면 $5가 하루치라
# 경고를 받아도 반응할 시간이 없다. 자본이 10배가 되면 명목가도 10배가 되고
# 수수료도 10배가 되므로, 기준은 자본이 아니라 **실제 소모 속도**를 따라가야 한다.
SPOT_BNB_MIN_DAYS      = 7.0      # 남은 소모 여유가 이보다 적으면 경고
SPOT_BNB_MIN_USD       = 5.0      # 소모일수 기준의 **하한**(소액 계좌용).
                                  # 이력이 없거나 거래가 뜸할 때의 폴백.
SPOT_BNB_ALERT_HOURS   = 24       # 같은 경고 재발송 간격(시간)
SPOT_BNB_PRICE_TTL_SEC = 300      # BNB 시세 캐시. 잔고폴러가 60초마다 도는데
                                  # 매번 티커를 치면 하루 1,440회를 낭비한다.

# 실제 수수료율 재확인 주기(초). 예전에는 프로세스당 1회만 조회하고 영원히
# 캐시해서, BNB를 채워 할인이 살아나도 **재시작 전까지 인식하지 못했다**.
# 반대로 BNB가 비어 할인이 사라져도 계속 싸다고 믿는다. 그 값으로 진입
# 가드가 판정하므로 현실과 벌어지면 곧바로 오판이 된다.
SPOT_FEE_RECHECK_SEC   = 6 * 3600  # 6시간

# ── BNB 자동 충전 ────────────────────────────────────────────────
# ⚠️ 이건 봇이 **당신 돈으로 스스로 매수**하는 유일한 기능이다.
#    매매와 무관한 지출이므로 기본 OFF이며, 켜더라도 아래 상한들이 모두
#    동시에 걸린다. 하나라도 위반하면 매수하지 않는다.
#
#    ① 1회 매수 상한        (SPOT_BNB_MAX_BUY_USD)
#    ② 재매수 최소 간격      (SPOT_BNB_REFILL_COOLDOWN_H)
#    ③ USDT 예비금 침범 금지 (SPOT_RESERVE_PCT 아래로 못 내려간다)
#    ④ 드라이런에서는 절대 실행하지 않는다
#
# 필요량은 **실제 거래 이력**에서 추정한다(최근 30일 명목가 합 × 수수료율
# × 2회). 이력이 없으면 보수적 기본값으로 최소 금액만 산다.
SPOT_BNB_AUTO_REFILL   = False    # 자동 충전 사용 여부
SPOT_BNB_TARGET_MONTHS = 2.0      # 몇 개월치를 채울 것인가
SPOT_BNB_MAX_BUY_USD   = 30.0     # 1회 매수 상한의 **하한**(소액 계좌용).
                                  # 실제 상한은 소모량에 맞춰 커지되,
                                  # 아래 SPOT_BNB_MAX_BUY_PCT를 넘지 않는다.
SPOT_BNB_MAX_BUY_PCT   = 0.01     # 1회 매수는 총자산의 이 비율을 넘지 않는다.
                                  # 소모량 추정이 틀려도 자산의 1% 이상은
                                  # 한 번에 BNB로 묶이지 않게 하는 최종 방어선.
SPOT_BNB_MIN_BUY_USD   = 12.0     # 이보다 적으면 사지 않는다
                                  # (NOTIONAL 턱걸이 + 수수료 낭비 방지)
SPOT_BNB_REFILL_COOLDOWN_H = 72   # 재매수 최소 간격(시간)

SPOT_LEARN_REFRESH_MIN   = 30     # 학습 결과 재계산 주기(분). 진입마다
                                  # DB 전체를 훑으면 루프가 느려진다

# Kelly / Ratchet
SPOT_KELLY_MIN_TRADES  = 10      # 최소 거래수 (20→10: Kelly 조기 활성화)
SPOT_KELLY_SCALE_MIN   = 0.15
# ↑ 하한은 **SPOT_KELLY_FRACTION과 정합**해야 한다.
#   원래 0.30은 full Kelly 기준으로 잡힌 값이라, half-Kelly(×0.5) 도입 후에는
#   현실적인 모든 성과 구간을 하한이 흡수해 Kelly가 상수 0.30이 됐다:
#     WR45%/RR1.5 → half 0.04,  WR60%/RR2.5 → half 0.22  (둘 다 0.30으로 클램프)
#   즉 좋은 전략에 자본을 더 배분하는 기능 자체가 죽어 있었다.
#   0.15로 낮추면 차등이 복원된다 (약한 전략 0.15 / 우수 전략 0.22~0.27).
#   ※ 절대 리스크를 키우고 싶다면 이 값이 아니라 SPOT_BASE_RISK_PCT를 조정할 것.
#     하한을 올리면 Kelly가 다시 상수가 되어 성과 기반 배분이 사라진다.
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

# RS Gate 차단 기준: 모멘텀 순위 백분위가 이 값을 넘으면 진입 차단.
#
# ⚠️ 이 상수는 원래 코드에 `MOMENTUM_TOP_TIER_PCT * 3`(= 0.99)로 인라인돼
#    있었다. 주석은 "모멘텀 하위 67% 심볼 진입 차단"이라 되어 있었지만,
#    rank_pct의 최댓값은 (n-1)/n 이므로 유니버스가 100개여도 0.99를 넘지
#    못한다 — **어떤 심볼도 차단된 적이 없다.** 필터가 죽어 있었다.
#
#    주석의 의도대로라면 0.33이어야 한다(상위 33%만 통과).
#    다만 그렇게 바꾸면 S6 진입의 약 2/3가 사라지는 큰 매매 변경이므로,
#    **기존 동작(사실상 무차단)을 그대로 두고** 값만 밖으로 꺼냈다.
#    reoptimize.py 그리드에 넣어 두었으니 WFO가 OOS로 판단하게 한다.
MOMENTUM_RS_GATE_PCT    = 0.99

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
BT_SPOT_SLIPPAGE = 0.0003   # (미사용) 슬리피지는 BT_SLIPPAGE_BY_TIER 로 결정된다.
                            # '티어 미매칭 시 폴백'으로 적혀 있었으나
                            # _get_slippage 의 tier3 가 catch-all 이라 이 값에
                            # 도달하는 경로가 없다. 존재하지 않는 폴백을
                            # 설명하던 주석이라 사실대로 고친다.
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
#   RANGING      : S4, S5 (평균회귀 — ADX<20 박스권, RSI 과매도 + BB 반등)
#   WEAK_TREND   : S3, S5, S6 (추세 형성 초기 혼합)
#   TRENDING_DOWN: S4 only (평균회귀 — RSI 과매도 반등만 보수적으로)
#       · S5(BB반등) 제외: 실전 10전 0승 실증 — 강한 하락추세(ADX35+)에서 BB 이탈은 추가하락 신호
#       · S7V4 제외: 종가>EMA200 요구하는 상승 모멘텀 전략이라 하락장에 논리적으로 부적합(진입 불가)
#   CRISIS       : 전면 차단
# ※ classify_regime()이 반환할 수 있는 **모든** 레짐을 여기에 명시할 것.
#   빠뜨리면 .get(regime, [])의 기본값 때문에 그 구간 전체가 조용히 거래
#   정지가 된다 — 로그에도 남지 않아 운영자는 알 수 없다.
#   (tests/test_regime_parity.py 가 누락을 잡는다)
REGIME_STRATEGY_MAP = {
    'TRENDING_UP':   ['S6'],
    'RANGING':       ['S4', 'S5'],
    'WEAK_TREND':    ['S3', 'S5', 'S6'],
    'TRENDING_DOWN': ['S4'],
    'CRISIS':        [],      # 변동성 폭발 — 의도적으로 전 전략 정지
    # 1D는 상승 추세인데 4H가 눌린 구간. 지금까지 맵에 없어 .get() 기본값으로
    # 전 전략이 차단돼 왔다(라이브에서만 발생 — 백테스트는 adx_4h를 안 넘겨
    # 이 레짐이 나온 적이 없다). **기존 동작을 그대로 보존**해 명시만 한다.
    #   검토 사항: classify_regime의 주석은 "Module C 차단"이라 특정 모듈만
    #   막으려던 것으로 읽히고, 이름 그대로 '횡보' 구간이라면 횡보 전략
    #   ['S4','S5']를 돌리는 편이 자연스럽다. 바꾸기 전에 WFO로 확인할 것.
    'MICRO_RANGING': [],
    'UNKNOWN':       [],      # 레짐 판별 실패(데이터 오류 등) — 안전하게 정지
}

# 기본 활성 전략 (main.py --strategies 기본값)
# REGIME_STRATEGY_MAP의 어느 레짐에든 배정된 전략만 나열할 것 —
# 맵에 없는 전략은 레짐 라우팅에서 항상 차단되어 지표 계산만 낭비한다.
# (과거 기본값의 S7V4가 어떤 레짐에도 배정되지 않아 영구 진입불가 상태였고,
#  TRENDING_DOWN 담당 S4는 기본값에 빠져 있어 하락장 마비가 지속됐음)
DEFAULT_ACTIVE_STRATEGIES = ['S3', 'S4', 'S5', 'S6']

# 라이브 라우팅되는 전략 = REGIME_STRATEGY_MAP에 배정된 전략의 합집합.
# 여기 없는 전략(S1/S2/S7/S7V4)은 백테스트·연구 전용이며 라이브에서
# 신호가 나와도 진입하지 않는다. (atlas_spot_backtest.py는 전 전략 실행 가능)
#   · S1 : Buy&Hold 벤치마크 (백테스트 비교 기준)
#   · S2 : SMA 골든크로스 — 1D 장기추세, 현재 라우팅 제외
#   · S7 : MACD 모멘텀 — S7V4(강화판)로 대체된 구버전
#   · S7V4: MACD 강화판 — 상승 모멘텀 전략이라 하락장 부적합해 벤치(b470232)
LIVE_STRATEGIES = sorted({s for strats in REGIME_STRATEGY_MAP.values() for s in strats})

# ─────────────────────────────────────────────────────────────
# 전략/레짐 표시 메타데이터 (대시보드·리포트 공용 단일 출처)
# ─────────────────────────────────────────────────────────────
STRATEGY_NAMES_KR = {
    'S1': 'Buy & Hold',   'S2': 'SMA 골든크로스', 'S3': 'EMA 추세추종',
    'S4': 'RSI 평균회귀',  'S5': 'BB 밴드반등',    'S6': 'Donchian 돌파',
    'S7': 'MACD 모멘텀',   'S7V4': 'MACD 모멘텀(강화)',
}

# 전략별 진입 조건 요약 (아래 상수 변경 시 문구도 함께 갱신)
STRATEGY_CONDITIONS = {
    'S3': 'EMA20이 EMA50을 상향 돌파 + 종가 > EMA200',
    'S4': 'RSI(14)가 30을 상향 돌파 + 종가 ≤ BB하단×1.03',
    'S5': '종가 < BB하단 + RSI(14) < 30',
    'S6': '20일 신고가 돌파 + 거래량 2.0배 스파이크',
    'S7': 'MACD 히스토그램 골든크로스 + 종가 > EMA200',
    'S7V4': 'MACD 히스토그램 골든크로스 + ADX≥25 + 거래량 1.3배 + 종가 > EMA200',
}

# 레짐 한글명 + 한 줄 성격 설명
REGIME_DESCRIPTIONS = {
    'TRENDING_UP':   ('상승 추세',   'BTC가 EMA200 위 + 강한 추세 — 돌파 전략으로 신고가 코인 추격'),
    'TRENDING_DOWN': ('하락 추세',   'BTC가 EMA200 아래 + 강한 추세 — 신규 진입에 보수적'),
    'RANGING':       ('박스권 횡보', '추세 약함(ADX 낮음) — 박스 하단 반등만 제한적으로 노림'),
    'WEAK_TREND':    ('약한 추세',   '추세 방향 모호 — 추세·반등 전략 혼합 운용'),
    'CRISIS':        ('변동성 위기', 'ATR 급등 — 모든 신규 진입 차단, 자본 방어 모드'),
}

# WEAK_TREND 리스크 스케일 (50% — 약추세에서 리스크 절반)
WEAK_TREND_RISK_SCALE    = 0.50
TRENDING_DOWN_RISK_SCALE = 0.30  # TRENDING_DOWN 구간 리스크 30% (백테스트 검증)

# ─────────────────────────────────────────────────────────────
# S5 리스크 제어 — 전문가 회의 결론 반영 (2026-06-04)
# ─────────────────────────────────────────────────────────────
# SL 후 같은 종목 재진입 금지 기간 (bars = 1D 봉 단위)
S5_SL_COOLDOWN_BARS = 2   # 2 bar = 48h. SL 직후 재진입 루프 방지

# BTC 고상관 종목 — 동시 S5 포지션 1개 한도
# (BTC 하락 시 전부 동반 하락 → 사실상 BTC 집중 베팅 방지)
S5_BTC_CORR_SYMBOLS = frozenset({
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'LINKUSDT', 'AVAXUSDT', 'DOTUSDT', 'BNBUSDT', 'LTCUSDT',
})
S5_CORR_MAX_POS = 1  # 위 그룹에서 동시 S5 최대 포지션 수

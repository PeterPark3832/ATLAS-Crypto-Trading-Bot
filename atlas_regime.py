"""
ATLAS — 장세 분류기 + 전략 라우터
===================================
BTC 1D 기준 ADX + EMA200 + ATR/Price 로 4가지 장세 분류 후
각 심볼·전략의 활성화 여부를 반환.

장세 정의:
  TRENDING_UP   ADX ≥ 25 & BTC > EMA200    상승 추세
  TRENDING_DOWN ADX ≥ 25 & BTC < EMA200    하락 추세
  RANGING       ADX < 20                   횡보
  WEAK_TREND    20 ≤ ADX < 25              약추세 (혼합 허용)
  CRISIS        ATR/Price ≥ 7%             위기 (전면 차단)

전략 활성화 매핑:
  TRENDING_UP   → Equinox LONG(ETH) ✓  Phoenix M1 LONG(BTC/SOL/BNB) ✓  SHORT ✗
  TRENDING_DOWN → Phoenix M1/M2 SHORT(BTC/SOL/BNB) ✓  Equinox LONG ✗
  RANGING       → Phoenix M2 양방향 ✓  Bounce ETH ✓  Equinox ✗
  WEAK_TREND    → 모두 허용 (리스크 70% 축소)
  CRISIS        → 전면 차단
"""

import logging
import threading
import time
import pandas as pd
import ccxt
from dataclasses import dataclass
from atlas_config import (
    REGIME_ADX_TREND, REGIME_ADX_WEAK, REGIME_CRISIS_ATR, REGIME_BTC_LOOKBACK,
    BINANCE_API_KEY, BINANCE_API_SECRET,
)
from atlas_indicators import calc_adx, _ohlcv_to_df


def _make_regime_ex() -> ccxt.binance:
    """
    regime_loop 스레드 전용 CCXT 인스턴스.
    atlas_main.py의 _get_ex()와 독립적으로 동작 → Nonce 충돌 없음.
    markets는 lazy-load (첫 API 호출 시 자동 로드).
    """
    return ccxt.binance({
        'apiKey':  BINANCE_API_KEY,
        'secret':  BINANCE_API_SECRET,
        'options': {'defaultType': 'future'},
    })

# ──────────────────────────────────────────────────────────────
REGIME_UNKNOWN      = 'UNKNOWN'
REGIME_TRENDING_UP  = 'TRENDING_UP'
REGIME_TRENDING_DOWN= 'TRENDING_DOWN'
REGIME_RANGING      = 'RANGING'
REGIME_WEAK_TREND   = 'WEAK_TREND'
REGIME_CRISIS       = 'CRISIS'


@dataclass
class RegimeState:
    regime:     str   = REGIME_UNKNOWN
    adx:        float = 0.0
    btc_price:  float = 0.0
    ema200:     float = 0.0
    atr_pct:    float = 0.0
    updated_at: float = 0.0   # time.time()

    @property
    def age_min(self) -> float:
        return (time.time() - self.updated_at) / 60

    def summary(self) -> str:
        emoji = {
            REGIME_TRENDING_UP:   '🟢',
            REGIME_TRENDING_DOWN: '🔴',
            REGIME_RANGING:       '🟡',
            REGIME_WEAK_TREND:    '🟠',
            REGIME_CRISIS:        '🚨',
            REGIME_UNKNOWN:       '❓',
        }.get(self.regime, '❓')
        return (
            f"{emoji} 장세: {self.regime}\n"
            f"  ADX={self.adx:.1f}  ATR%={self.atr_pct*100:.2f}%\n"
            f"  BTC={self.btc_price:,.0f}  EMA200={self.ema200:,.0f}\n"
            f"  갱신: {self.age_min:.0f}분 전"
        )


# ──────────────────────────────────────────────────────────────
# 전역 레짐 캐시 (스레드 안전)
# ──────────────────────────────────────────────────────────────
_regime_lock  = threading.Lock()
_regime_cache = RegimeState()
_REGIME_TTL   = 3600   # 1시간마다 갱신


def get_cached_regime() -> RegimeState:
    with _regime_lock:
        return _regime_cache


def classify_regime(adx: float, btc_price: float,
                    ema200: float, atr_pct: float) -> str:
    """순수 분류 함수 (테스트 가능)."""
    if atr_pct >= REGIME_CRISIS_ATR:
        return REGIME_CRISIS
    if adx >= REGIME_ADX_TREND:
        return REGIME_TRENDING_UP if btc_price >= ema200 else REGIME_TRENDING_DOWN
    if adx < REGIME_ADX_WEAK:
        return REGIME_RANGING
    return REGIME_WEAK_TREND


def update_regime(ex, candle_cache=None) -> RegimeState:
    """
    BTC 1D OHLCV를 조회해 레짐을 갱신한다.
    candle_cache: CandleCache 인스턴스 (있으면 캐시 사용)
    """
    try:
        if candle_cache is not None:
            ohlcv = candle_cache.get(ex, 'BTC/USDT', '1d', REGIME_BTC_LOOKBACK + 20)
        else:
            ohlcv = ex.fetch_ohlcv('BTC/USDT', '1d', limit=REGIME_BTC_LOOKBACK + 20)

        if not ohlcv or len(ohlcv) < 30:
            return get_cached_regime()

        df = _ohlcv_to_df(ohlcv)
        closes = df['close']
        ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
        btc_price = float(closes.iloc[-1])

        # ATR%
        pc  = closes.shift(1)
        tr  = pd.concat([
            df['high'] - df['low'],
            (df['high'] - pc).abs(),
            (df['low']  - pc).abs(),
        ], axis=1).max(axis=1)
        atr     = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / btc_price if btc_price else 0

        adx    = calc_adx(ohlcv[-REGIME_BTC_LOOKBACK:], period=14)
        regime = classify_regime(adx, btc_price, ema200, atr_pct)

        state = RegimeState(
            regime     = regime,
            adx        = adx,
            btc_price  = btc_price,
            ema200     = ema200,
            atr_pct    = atr_pct,
            updated_at = time.time(),
        )
        with _regime_lock:
            global _regime_cache
            _regime_cache = state
        return state

    except Exception as e:
        logging.getLogger('atlas').warning(f'[Regime] 갱신 실패: {e}')
        return get_cached_regime()


def regime_loop(ex=None, candle_cache=None):
    """
    백그라운드 스레드: 1시간마다 레짐 갱신.
    ex 파라미터는 하위 호환용으로만 유지 — 실제로는 스레드 전용
    인스턴스(_make_regime_ex())를 사용하여 Nonce 충돌을 방지한다.
    """
    log    = logging.getLogger('atlas')
    _ex    = _make_regime_ex()   # 이 스레드만의 전용 CCXT 인스턴스
    while True:
        try:
            state = update_regime(_ex, candle_cache)
            log.info(f'[Regime] {state.summary()}')
        except Exception as e:
            log.warning(f'[Regime] loop 오류: {e}')
        time.sleep(_REGIME_TTL)


# ══════════════════════════════════════════════════════════════
#  전략 라우터 — 장세별 전략 활성화 판단
# ══════════════════════════════════════════════════════════════

def is_phoenix_long_allowed(sym: str) -> bool:
    """Phoenix 4H LONG 진입 허용 여부."""
    regime = get_cached_regime().regime
    if regime == REGIME_CRISIS:
        return False
    if regime == REGIME_TRENDING_DOWN:
        return False   # 하락추세에서 4H 롱 차단
    return True        # 상승추세, 횡보, 약추세 모두 허용


def is_phoenix_short_allowed(sym: str) -> bool:
    """Phoenix 4H SHORT 진입 허용 여부."""
    regime = get_cached_regime().regime
    if regime == REGIME_CRISIS:
        return False
    if regime == REGIME_TRENDING_UP:
        return False   # 상승추세에서 4H 숏 차단
    return True        # 하락추세, 횡보, 약추세 모두 허용


def is_equinox_long_allowed(sym: str) -> bool:
    """Equinox 1D LONG 진입 허용 여부."""
    regime = get_cached_regime().regime
    if regime in (REGIME_CRISIS, REGIME_TRENDING_DOWN, REGIME_RANGING):
        return False
    return True   # 상승추세, 약추세 허용


def is_bounce_allowed(sym: str) -> bool:
    """Bounce 1D LONG 진입 허용 여부 (횡보·하락 초입 반등)."""
    regime = get_cached_regime().regime
    if regime == REGIME_CRISIS:
        return False
    if regime == REGIME_TRENDING_DOWN:
        return False   # 하락추세 지속 중 반등 노리기 금지
    return True


def is_crisis() -> bool:
    return get_cached_regime().regime == REGIME_CRISIS


def get_risk_scale() -> float:
    """
    장세별 리스크 배율.
    WEAK_TREND → 0.7배 (불확실 구간)
    그 외       → 1.0배
    """
    regime = get_cached_regime().regime
    if regime == REGIME_WEAK_TREND:
        return 0.70
    return 1.00

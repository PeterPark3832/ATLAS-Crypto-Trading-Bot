"""
ATLAS — 장세 분류기
====================
BTC 1D 기준 ADX + EMA200 + ATR/Price 로 5가지 장세 분류.

장세 정의:
  TRENDING_UP   ADX ≥ 25 & BTC > EMA200    상승 추세
  TRENDING_DOWN ADX ≥ 25 & BTC < EMA200    하락 추세
  RANGING       ADX < 20                   횡보
  WEAK_TREND    20 ≤ ADX < 25              약추세 (혼합 허용)
  CRISIS        ATR/Price ≥ 7%             위기 (전면 차단)
"""

import logging
from typing import Any
import threading
import time
import pandas as pd
import ccxt
from dataclasses import dataclass
from atlas_spot_config import BINANCE_API_KEY, BINANCE_API_SECRET
from atlas_indicators import adx_min_bars, calc_adx, _ohlcv_to_df

# ── 레짐 분류 임계값 ─────────────────────────────────────────
REGIME_ADX_TREND    = 25      # ADX >= 25 → 추세 구간
REGIME_ADX_WEAK     = 20      # 20 <= ADX < 25 → 약추세
REGIME_CRISIS_ATR   = 0.07    # ATR/Price >= 7% → 위기
REGIME_BTC_LOOKBACK = 50      # ADX 계산에 사용할 1D 캔들 수


def _make_regime_ex() -> ccxt.binance:
    """regime_loop 스레드 전용 CCXT 인스턴스 (Nonce 충돌 방지)."""
    return ccxt.binance({
        'apiKey':  BINANCE_API_KEY,
        'secret':  BINANCE_API_SECRET,
        'options': {'defaultType': 'spot'},
    })

# ──────────────────────────────────────────────────────────────
REGIME_UNKNOWN       = 'UNKNOWN'
REGIME_TRENDING_UP   = 'TRENDING_UP'
REGIME_TRENDING_DOWN = 'TRENDING_DOWN'
REGIME_RANGING       = 'RANGING'
REGIME_WEAK_TREND    = 'WEAK_TREND'
REGIME_CRISIS        = 'CRISIS'
REGIME_MICRO_RANGING = 'MICRO_RANGING'  # TRENDING_UP + 4H ADX<20: 단기 횡보 삽입


@dataclass
class RegimeState:
    regime:     str   = REGIME_UNKNOWN
    adx:        float = 0.0
    adx_4h:     float = 0.0   # BTC 4H ADX (마이크로 레짐 판별용)
    adx_slope:  float = 0.0   # ADX 3일 기울기 (추세 소멸 감지)
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
            REGIME_MICRO_RANGING: '🔵',
            REGIME_CRISIS:        '🚨',
            REGIME_UNKNOWN:       '❓',
        }.get(self.regime, '❓')
        slope_arrow = '↗' if self.adx_slope > 0 else ('↘' if self.adx_slope < 0 else '→')
        return (
            f"{emoji} 장세: {self.regime}\n"
            f"  ADX(1D)={self.adx:.1f}{slope_arrow}  ADX(4H)={self.adx_4h:.1f}  ATR%={self.atr_pct*100:.2f}%\n"
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
                    ema200: float, atr_pct: float,
                    adx_4h: float = 0.0,
                    adx_slope: float = 0.0) -> str:
    """
    순수 분류 함수 (테스트 가능).
    adx_4h: BTC 4H ADX — MICRO_RANGING 판별
    adx_slope: 1D ADX 최근 3봉 기울기 — 추세 소멸 조기 감지
    """
    if atr_pct >= REGIME_CRISIS_ATR:
        return REGIME_CRISIS
    if adx >= REGIME_ADX_TREND:
        base = REGIME_TRENDING_UP if btc_price >= ema200 else REGIME_TRENDING_DOWN
        # ADX 기울기 하락 중 (3봉 연속 감소) + ADX < 30 → WEAK_TREND 강제 다운그레이드
        if adx_slope < 0 and adx < 30:
            return REGIME_WEAK_TREND
        # 1D TRENDING_UP이지만 4H ADX < 20 → 단기 횡보 삽입, Module C 차단
        if base == REGIME_TRENDING_UP and adx_4h > 0 and adx_4h < 20:
            return REGIME_MICRO_RANGING
        return base
    if adx < REGIME_ADX_WEAK:
        return REGIME_RANGING
    return REGIME_WEAK_TREND


def _drop_forming_bar(ohlcv: list | None) -> list:
    """마지막(형성 중인) 봉을 제거한다.

    거래소 OHLCV의 마지막 원소는 **아직 마감되지 않은** 현재 봉이다.
    지표에 그대로 넣으면 하루 종일 값이 흔들리고, 완성봉만 보는 백테스트와
    서로 다른 경로를 걷게 된다.

    봉이 2개 미만이면 자를 것이 없으므로 그대로 돌려준다 — 호출부의
    최소 봉수 가드가 이어서 처리한다.
    """
    if not ohlcv or len(ohlcv) < 2:
        return list(ohlcv or [])
    return list(ohlcv[:-1])


def update_regime(ex: Any, candle_cache: Any = None) -> RegimeState:
    """
    BTC 1D OHLCV를 조회해 레짐을 갱신한다.
    candle_cache: CandleCache 인스턴스 (있으면 캐시 사용)
    """
    try:
        if candle_cache is not None:
            ohlcv = candle_cache.get(ex, 'BTC/USDT', '1d', REGIME_BTC_LOOKBACK + 20)
        else:
            ohlcv = ex.fetch_ohlcv('BTC/USDT', '1d', limit=REGIME_BTC_LOOKBACK + 20)

        # 형성 중인 봉을 버린다. 거래소는 **아직 끝나지 않은** 현재 봉을
        # 마지막에 붙여 주는데, 그 값은 하루 종일 변한다.
        #
        # 예전에는 이걸 그대로 써서 두 가지 문제가 있었다:
        #   ① 내부 불일치 — ADX '레벨'은 미완성 봉을 포함하는데 ADX '기울기'는
        #      제외했다(기울기는 `[-(LB+k):-k]` 슬라이스라 끝을 잘라낸다).
        #      같은 함수가 같은 지표를 두 기준으로 읽고 있었다.
        #   ② 백테스트와 괴리 — 백테스트는 완성된 일봉만 본다. 즉 검증한
        #      레짐 경로를 실계좌가 그대로 따라가지 않았다. 합성 데이터로
        #      재어 보니 **약 15%**에서 레짐이 달라졌고, 레짐은 어떤 전략이
        #      거래할지를 정하므로 그대로 성과 차이가 된다.
        #
        # 완성봉만 쓰면 반응이 최대 하루 늦지만, 미완성 봉으로 레짐이
        # 하루에도 몇 번씩 뒤집히는 것보다 낫다 — 그 진동은 백테스트가
        # 한 번도 본 적 없는 동작이다.
        ohlcv = _drop_forming_bar(ohlcv)

        # 가드는 **ADX가 실제로 계산 가능한 봉 수**에 맞춰야 한다.
        # 30봉만 요구하면 30~41봉 구간에서 calc_adx가 0.0을 돌려주고,
        # classify_regime은 그걸 '추세 없음'으로 읽어 RANGING으로 분류한다.
        # 즉 데이터 문제가 정상 레짐으로 위장돼 S4·S5가 그대로 진입한다.
        _min_bars = adx_min_bars(14)
        if not ohlcv or len(ohlcv) < _min_bars:
            log = logging.getLogger('atlas')
            log.warning(f'[Regime] BTC 1D 봉 부족({len(ohlcv) if ohlcv else 0}/'
                        f'{_min_bars}) — 레짐 갱신 보류, 직전 값 유지')
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

        adx = calc_adx(ohlcv[-REGIME_BTC_LOOKBACK:], period=14)

        # ADX 기울기: 직전 3봉 ADX 시계열로 추세 소멸 조기 감지
        # 기울기의 **끝점은 레벨과 같은 봉**이어야 한다.
        #
        # 예전에는 k가 3,2,1이라 끝점이 t-1이었고, 위의 `adx`(레벨)는 t였다.
        # 같은 함수가 같은 지표를 한 봉 어긋난 기준으로 읽은 셈이다.
        # 게다가 백테스트는 `adx(t) − adx(t−2)`로 계산하므로 라이브만
        # `adx(t−1) − adx(t−3)`이 되어 검증한 경로와 달랐다.
        #
        # k=0은 '끝까지'를 뜻하므로 슬라이스 끝을 None으로 둬야 한다.
        # (`-0`은 0이라 `[-LB:0]` = 빈 리스트가 된다 — 예전 코드가
        #  `if k else None`을 쓴 이유이고, k가 0이 될 일이 없어 죽어 있었다)
        adx_series_raw = [
            calc_adx(ohlcv[-(REGIME_BTC_LOOKBACK + k):(-k if k else None)],
                     period=14)
            for k in range(2, -1, -1)
        ]
        adx_slope = (adx_series_raw[-1] - adx_series_raw[0]) if len(adx_series_raw) == 3 else 0.0

        # 4H BTC ADX (마이크로 레짐 판별)
        adx_4h = 0.0
        try:
            ohlcv_4h = ex.fetch_ohlcv('BTC/USDT', '4h', limit=60)
            if ohlcv_4h and len(ohlcv_4h) >= 42:
                adx_4h = calc_adx(ohlcv_4h[-50:], period=14)
        except Exception:
            pass  # 4H 조회 실패 시 MICRO_RANGING 분류 건너뜀 (adx_4h=0 → 조건 불충족)

        regime = classify_regime(adx, btc_price, ema200, atr_pct, adx_4h, adx_slope)

        state = RegimeState(
            regime     = regime,
            adx        = adx,
            adx_4h     = adx_4h,
            adx_slope  = adx_slope,
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


def regime_loop(stop_event: Any = None, candle_cache: Any = None) -> None:
    """
    백그라운드 스레드: 1시간마다 레짐 갱신.
    stop_event: threading.Event — set되면 루프 종료 (graceful shutdown).
    CCXT는 스레드 전용 인스턴스(_make_regime_ex())를 생성해 Nonce 충돌 방지.
    """
    log    = logging.getLogger('atlas')
    _ex    = _make_regime_ex()   # 이 스레드만의 전용 CCXT 인스턴스
    while stop_event is None or not stop_event.is_set():
        try:
            state = update_regime(_ex, candle_cache)
            log.info(f'[Regime] {state.summary()}')
        except Exception as e:
            log.warning(f'[Regime] loop 오류: {e}')
        if stop_event is not None:
            stop_event.wait(_REGIME_TTL)
        else:
            time.sleep(_REGIME_TTL)


def is_crisis() -> bool:
    return get_cached_regime().regime == REGIME_CRISIS


def get_risk_scale() -> float:
    """
    장세별 리스크 배율.
    MICRO_RANGING → 0.50배 (1D 상승추세 + 4H 횡보 삽입)
    WEAK_TREND    → 0.70배 (불확실 구간)
    그 외          → 1.00배
    """
    regime = get_cached_regime().regime
    if regime == REGIME_MICRO_RANGING:
        return 0.50
    if regime == REGIME_WEAK_TREND:
        return 0.70
    return 1.00

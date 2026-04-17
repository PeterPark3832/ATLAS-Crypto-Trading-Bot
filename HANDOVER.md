# ATLAS v2 인수인계서

> 작성일: 2026-04-11  
> 목적: 새 세션에서 컨텍스트 없이도 즉시 이어서 작업 가능하도록 정리

---

## 1. 프로젝트 개요

**ATLAS v2** — 바이낸스 선물 자동매매 봇  
Vultr 서버에서 현재 **실거래 가동 중**

### 전략 구조
| 모듈 | 유형 | 심볼 | 타임프레임 | 방향 |
|------|------|------|-----------|------|
| Module A | 추세추종 (EMA 크로스오버) | BTC/ETH/SOL/BNB | 4H | 롱+숏 |
| Module B | 평균회귀 (RSI 크로스오버) | ETH/BNB | 1D | 롱+숏 |

### 레짐 라우팅
- `TRENDING_UP` → Module A LONG 전용
- `TRENDING_DOWN` → Module A SHORT 전용
- `RANGING` → Module B 전용 (Module A 스킵)
- `WEAK_TREND` → Module A 양방향 (EMA200 필터 적용)
- `CRISIS` → 전면 차단

---

## 2. 파일 구조

```
C:\Users\쩡이\Downloads\Vibe Code\ATLAS\
├── atlas_config.py          ← 모든 파라미터 (V1 + V2)
├── atlas_indicators.py      ← 지표 계산 유틸 (수정 없음)
├── atlas_regime.py          ← 레짐 분류기 (수정 없음)
├── atlas_main.py            ← V1 라이브 엔진 (현재 미사용)
├── atlas_v2_backtest.py     ← V2 백테스트 프레임워크
├── atlas_v2_main.py         ← V2 라이브 엔진 (현재 가동 중)
├── atlas_backtest.py        ← V1 백테스트 (기존)
├── state/
│   └── atlas_v2.db          ← SQLite DB (포지션/거래 히스토리)
└── logs/
    └── atlas_v2_YYYYMMDD_HHMMSS.log
```

---

## 3. 현재 파라미터 (atlas_config.py 기준)

### Module A
```python
V2_MA_SYMBOLS    = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
V2_MA_EMA_FAST   = 20
V2_MA_EMA_SLOW   = 50
V2_MA_ADX_MIN    = 20        # ADX 필터
V2_MA_ATR_SL     = 2.5       # SL 배수
V2_MA_RISK_PCT   = 0.018     # 1.8% (1.0% → 1.8%로 조정됨 — 연수익 6~7% 목표)
V2_MA_LEVERAGE   = 5
V2_MA_COOLDOWN   = 2         # 청산 후 4H 캔들 2개 재진입 금지
```

### Module B
```python
V2_MR_SYMBOLS      = ['ETHUSDT', 'BNBUSDT']
V2_MR_RSI_LONG     = 35      # RSI < 35 → 35 크로스오버 시 LONG
V2_MR_RSI_SHORT    = 65      # RSI > 65 → 65 크로스오버 시 SHORT
V2_MR_ATR_SL       = 1.5
V2_MR_RISK_PCT     = 0.008   # ETH 0.8%
V2_MR_RISK_PCT_BNB = 0.004   # BNB 0.4% ← IS 7건으로 통계 불충분, 의도적 절감
V2_MR_LEVERAGE     = 2
V2_MR_MAX_BARS     = 15      # 15일 초과 시 시간청산
```

### 포트폴리오 리스크
```python
MAX_OPEN_POSITIONS = 5
PORTFOLIO_VAR_CAP  = 0.10    # 총 리스크 10% 초과 시 진입 차단
DAILY_LOSS_LIMIT   = -0.03   # 일일 -3% 손실 시 자동 정지
```

---

## 4. V2 백테스트 결과 (Walk-Forward)

### IS (2022-01-01 ~ 2024-06-30)
| 항목 | 수치 |
|------|------|
| PF | 1.19 |
| 수익률 | +10.72% |
| Sharpe | 0.77 |
| MDD | 9.98% |

### OOS (2024-07-01 ~ 현재) ← 실전 유효 지표
| 항목 | 수치 |
|------|------|
| PF | 1.16 |
| 수익률 | +6.06% |
| Sharpe | 0.59 |
| MDD | 6.82% |

### 모듈별 핵심 포인트
- **Module A**: OOS PF(1.18) > IS PF(1.07) → 과최적화 없음, 강한 범용성
- **Module B ETH**: IS 1.60 → OOS 1.64 → 안정적 PASS
- **Module B BNB**: IS 3.64 → OOS 0.66 → IS 7건 통계 불충분, 리스크 절반 적용 중

---

## 5. V1 vs V2 비교

| 항목 | V1 | V2 |
|------|----|----|
| OOS 수익률 | +7.54% | +6.06% |
| OOS Sharpe | 0.70 | 0.59 |
| OOS MDD | ~12% (추정) | **6.82%** |
| RANGING 대응 | 사실상 없음 (Equinox OOS 3건) | Module B |
| 과최적화 위험 | 중간 | 낮음 (단순 파라미터) |

**핵심**: V1은 수치상 앞서지만 Equinox가 OOS에서 실질적으로 작동하지 않아 단일 전략 봇이나 다름없었음. V2는 전 레짐 커버 + 낮은 MDD로 장기 우상향에 구조적 우위.

---

## 6. 운용자 프로파일 (인터뷰 결과)

| 항목 | 내용 |
|------|------|
| 투입 자본 | $1,000 이하 (전체 자산의 5%) |
| 목표 수익 | 연 20% 이상 |
| 허용 MDD | 10~15% |
| 운용 스타일 | 완전 위임형 (개입 최소화) |
| 모니터링 | 수시 가능, 외부에서 수동 청산 가능 |
| 인내 기간 | 1~2개월 수익 없어도 유지 가능 |
| 이전 봇 종료 이유 | 단일 추세장 전략 + 숏 연속 손절 |

### 수익 목표 갭 분석
```
현재 파라미터 기준 예상 연수익:  6~7%  (리스크 1.8% 기준)
목표:                           20%
갭 해소 방법:
  ① 지금은 불가 (리스크를 더 올리면 MDD 15% 초과)
  ② 3~6개월 실전 데이터 축적 후 Sharpe 확인
  ③ Sharpe 유지 확인되면 자본 증액으로 금액 목표 달성
```

---

## 7. 지금까지 적용된 코드 수정 사항

| 수정 내용 | 파일 | 이유 |
|---------|------|------|
| V2 파라미터 추가 | atlas_config.py | 신규 전략 |
| `V2_MR_RISK_PCT_BNB = 0.004` 추가 | atlas_config.py | BNB Module B 통계 불충분 |
| `V2_MA_RISK_PCT` 1.0% → 1.8% | atlas_config.py | 연수익 목표 상향 |
| `tg()` 언더바 치환 | atlas_v2_main.py | 텔레그램 Markdown 파싱 오류 방지 |
| `exit_position()` 이모지 복구 | atlas_v2_main.py | 청산 알람 이모지 누락 |
| BNB 리스크 분기 | atlas_v2_main.py | `V2_MR_RISK_PCT_BNB` 적용 |

---

## 8. Telegram 명령어

| 명령어 | 기능 |
|--------|------|
| `/status` | 현재 열린 포지션 목록 |
| `/stats` | 전략별 누적 통계 (건수, WR, PnL, avgR) |
| `/regime` | 현재 BTC 레짐 상태 |
| `/equity` | 현재 잔고 |
| `/ratchet` | Ratchet 리스크 배율 상태 |
| `/pause` | 신규 진입 일시정지 |
| `/resume` | 신규 진입 재개 |
| `/stop` | Kill Switch (봇 종료) |

---

## 9. 향후 액션 아이템

### 즉시 (지금)
- [x] BNB Module B 리스크 0.4%로 절감
- [x] Module A 리스크 1.8%로 상향
- [ ] 서버에서 봇 재시작하여 파라미터 반영

### 단기 (1~3개월)
- [ ] 실전 승률 vs 백테스트 37% 비교 모니터링
- [ ] Kelly 스케일링 자동 작동 시작 (거래 20건 이상 누적 후)
- [ ] BNB Module B 실전 거래 20건 달성 시 리스크 0.8%로 상향 검토

### 중기 (3~6개월)
- [ ] 실전 Sharpe 0.4 이상 유지 확인
- [ ] OIS PF 1.0 이상 유지 확인
- [ ] 자본 증액 or 리스크 조정 재검토

### 보류 (데이터 축적 후 검토)
- OI(미결제약정) 필터 추가 — 백테스트 먼저
- Walk-Forward 최적화 루프 추가 — 실적 붕괴 감지 시 검토
- 전략 Sharpe 개선 (진입 조건 강화, 부분 TP 등)

---

## 10. 다음 세션에서 이어갈 수 있는 작업

1. **성과 리포트 스크립트** (`atlas_v2_report.py`)
   - DB에서 일별/전략별 PnL, 누적 수익곡선, MDD 자동 출력

2. **모니터링 자동화**
   - 매일 아침 Telegram으로 전날 성과 + 현재 레짐 + 열린 포지션 브리핑

3. **실전 데이터 기반 전략 개선** (3개월 후)
   - 승률/Sharpe 실측치 확인 후 파라미터 재검토

---

## 11. 핵심 판단 기준 (봇 개입 시점)

봇을 멈추거나 파라미터를 수정해야 하는 신호:

```
경고 수준
  실전 승률 < 30%   (백테스트 37% 대비 유의미한 하락)
  실전 PF  < 1.0    (손실 구간 진입)
  MDD      > 15%    (허용치 초과)

즉시 중단 수준
  MDD      > 20%
  일일 손실 > 3%    (DAILY_LOSS_LIMIT 자동 작동)
  거래소 API 오류 반복
```

---

*이 문서는 ATLAS v2 최초 실거래 가동 시점(2026-04-11) 기준으로 작성됨*

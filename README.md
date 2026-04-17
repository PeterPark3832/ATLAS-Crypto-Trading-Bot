# ATLAS — Binance Futures Automated Trading Bot

Binance 선물 자동매매 봇. 다중 전략 + 레짐 라우팅 + 통합 리스크 관리.

---

## 전략 구조

| 모듈 | 유형 | 심볼 | 타임프레임 | 방향 |
|------|------|------|-----------|------|
| Module A | 추세추종 (EMA 크로스오버) | BTC/ETH/SOL/BNB | 4H | 롱+숏 |
| Module B | 평균회귀 (RSI 크로스오버) | ETH/BNB | 1D | 롱+숏 |
| Module C | 브레이크아웃 모멘텀 | BTC/ETH/SOL | 15m | 롱+숏 |

### 레짐 라우팅

| 레짐 | Module A | Module B | Module C |
|------|----------|----------|----------|
| TRENDING_UP | LONG | ✗ | LONG |
| TRENDING_DOWN | SHORT | ✗ | SHORT |
| RANGING | ✗ | LONG+SHORT | ✗ |
| WEAK_TREND | 양방향 | ✗ | 양방향 |
| CRISIS | ✗ | ✗ | ✗ |

---

## 리스크 관리

- **Kelly 스케일링** — 거래 누적 후 승률 기반 자동 포지션 조절
- **Drawdown Ratchet** — 낙폭 5% → 리스크 70%, 7% → 40%
- **VaR 한도** — 포트폴리오 총 리스크 10% 초과 시 진입 차단
- **상관계수 보정** — 동일 방향 포지션 집중 시 리스크 축소
- **변동성 패리티** — 고변동 심볼 자동 리스크 감소
- **일일 손실 한도** — -3% 도달 시 당일 자동 정지

---

## 파일 구조

```
ATLAS/
├── atlas_config.py        # 모든 파라미터
├── atlas_indicators.py    # 지표 계산 (4H/1D/15m)
├── atlas_regime.py        # 레짐 분류기 (ADX 기반)
├── atlas_v2_main.py       # 라이브 트레이딩 엔진 (v2 현재 가동)
├── atlas_v2_backtest.py   # Walk-Forward 백테스트
├── atlas_mcp_server.py    # MCP 서버 (원격 모니터링)
├── atlas_main.py          # v1 레거시 엔진
├── atlas_backtest.py      # v1 백테스트
├── tests/                 # 단위 테스트
├── .env.example           # 환경변수 템플릿
└── requirements.txt       # 의존성
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 실제 API 키 입력
```

`.env` 파일에 입력할 항목:

```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TG_TOKEN=your_telegram_bot_token
TG_CHAT_ID=your_telegram_chat_id
```

> ⚠️ `.env` 파일은 절대 공개 저장소에 올리지 마세요.

### 3. 봇 실행

```bash
python atlas_v2_main.py
```

---

## Telegram 명령어

| 명령어 | 기능 |
|--------|------|
| `/status` | 현재 열린 포지션 |
| `/stats` | 전략별 누적 통계 |
| `/regime` | 현재 BTC 레짐 |
| `/equity` | 현재 잔고 |
| `/ratchet` | Ratchet 리스크 배율 |
| `/pause` | 신규 진입 일시정지 |
| `/resume` | 신규 진입 재개 |
| `/stop` | Kill Switch (봇 종료) |

---

## 백테스트 결과 (Walk-Forward OOS)

| 항목 | Module A | Module B (ETH) | 합산 |
|------|----------|----------------|------|
| OOS 수익률 | +6.06% | 안정적 | +6.06% |
| OOS Sharpe | 0.59 | — | 0.59 |
| OOS MDD | 6.82% | — | 6.82% |
| PF | 1.16 | 1.64 | — |

> OOS 기간: 2024-07-01 ~ 현재  
> Module C는 라이브 트랙레코드 수집 중

---

## 주의사항

- Binance Futures 전용 (현물 미지원)
- API 키에 **IP 제한** 설정 권장
- 소액($500~$2,000)으로 시작 후 트랙레코드 확인 권장
- 과거 수익이 미래 수익을 보장하지 않습니다

---

## License

MIT License

# ATLAS — 암호화폐 현물 자동매매 봇

Binance 현물 시장에서 7가지 전략을 동시에 운용하는 Python 트레이딩 봇.  
레짐 기반 전략 라우팅, Kelly 포지션 사이징, 동적 리스크 관리를 통해 장세에 맞는 전략만 자동으로 활성화합니다.

---

## 전략 구성 (S1~S7)

| ID | 이름 | 타임프레임 | 특징 |
|----|------|-----------|------|
| S1 | Buy & Hold | — | 벤치마크 (성과 비교용) |
| S2 | SMA Golden Cross | 1D | 50/200 데드크로스·골든크로스 |
| S3 | EMA Trend Follow | 4H | EMA20/50 크로스 + ADX 필터 + 동적 RR |
| S4 | RSI Mean Reversion | 1D | RSI<30 과매도 + BB 하단 반등 |
| S5 | Bollinger Band Bounce | 1D | BB 하단 이탈 + RSI 확증 |
| S6 | Donchian Breakout | 1D | 20일 신고가 돌파 + 거래량 스파이크 + VWAP 확증 |
| S7 | MACD Momentum | 4H | MACD 히스토그램 방향 전환 + EMA200 필터 |

---

## 레짐 라우팅

BTC 1D ADX + EMA200 기반으로 5가지 시장 상태를 자동 분류, 각 레짐에 맞는 전략만 활성화합니다.

| 레짐 | 조건 | 활성 전략 | 리스크 |
|------|------|----------|--------|
| TRENDING_UP | ADX ≥ 25 & BTC > EMA200 | S3, S6, S7 (추세추종) | 100% |
| RANGING | ADX < 20 | S4, S5 (평균회귀) | 100% |
| WEAK_TREND | 20 ≤ ADX < 25 | 전체 허용 | 50% |
| TRENDING_DOWN | ADX ≥ 25 & BTC < EMA200 | S4, S5 (과매도 반등) | 30% |
| CRISIS | ATR/가격 ≥ 7% | 전면 차단 | 0% |

---

## 리스크 관리

### 포지션 사이징
- **기본 리스크**: 자본의 2%/거래 (`SPOT_BASE_RISK_PCT`)
- **Kelly 스케일**: 전략별 실전 승률·PF 기반 자동 조정 (최소 0.3배, 최대 2.0배)
- **Kelly 최소 거래수**: 10건 이후 활성화 (초기 30% 고정)
- **단일 종목 상한**: 자본의 15% (`SPOT_MAX_ALLOC_PCT`)
- **최대 동시 포지션**: 15개 (`SPOT_MAX_POSITIONS`)

### Drawdown Ratchet
| MDD | 리스크 배율 |
|-----|------------|
| < 5% | 100% (정상) |
| 5~8% | 70% |
| > 8% | 40% |
| 회복 +15% | 원상 복구 |

### 모멘텀 집중 베팅 (RS Gate)
- 유니버스 상위 33% 심볼에만 S6/S7 진입 허용
- 주도주 티어 리스크 30% 상향 부스트

### 선물 펀딩비 필터
- 펀딩비 > +0.05%/8h → S3/S6/S7 진입 차단 (롱 과밀)
- 펀딩비 < -0.01%/8h → 리스크 스케일 +20% (숏 스퀴즈 기대)

---

## 심볼 유니버스

- Binance 현물 24h 거래량 기준 상위 50개 USDT 페어 자동 선별
- 스테이블코인·레버리지 토큰 자동 제외
- 45일 모멘텀 랭킹으로 주도주 순서 정렬
- **4시간 주기** 자동 갱신

---

## 설치

```bash
git clone https://github.com/PeterPark3832/ATLAS-Crypto-Trading-Bot.git
cd ATLAS-Crypto-Trading-Bot
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 입력
```

> **remote URL에 토큰을 넣지 말 것.** 이 저장소는 공개라 `git pull`에
> 인증이 필요 없다. `https://<token>@github.com/...` 형태로 clone 하면
> 토큰이 `.git/config`에 **평문으로** 남아, 서버를 백업하거나 이미지를
> 뜨는 순간 그대로 새어 나간다. 실제로 운영 서버가 그 상태였다.
> 이미 그렇게 설정했다면 아래로 정리한다.
>
> ```bash
> git remote set-url origin https://github.com/PeterPark3832/ATLAS-Crypto-Trading-Bot.git
> grep -c github_pat_ .git/config   # 0 이어야 한다
> ```
>
> 운영 서버는 pull만 하면 되므로 push가 막히는 편이 오히려 안전하다.

### 환경변수 (.env)

```env
BINANCE_API_KEY=...          # Binance Spot API 키
BINANCE_API_SECRET=...       # Binance Spot API 시크릿
TG_TOKEN=...                 # Telegram 봇 토큰
TG_CHAT_ID=...               # Telegram 채팅 ID
INITIAL_CAPITAL=1000         # 초기 자본 ($)
DASH_PASSWORD=atlas2026      # 대시보드 접속 비밀번호
DB_FILE=/path/to/atlas_spot.db
```

---

## 실행

```bash
# 봇 실행
python atlas_spot_main.py

# 대시보드 실행 (별도 터미널)
uvicorn atlas_web_dashboard:app --host 0.0.0.0 --port 8080

# 헬스체크 crontab 등록
# */5 * * * * /path/to/healthcheck.sh
```

---

## 백테스트

```bash
# 전체 전략 Walk-Forward
python atlas_spot_backtest.py --wf

# 단일 전략
python atlas_spot_backtest.py --strategy S3

# 특정 심볼
python atlas_spot_backtest.py --symbol BTCUSDT --strategy S6

# 기간 지정
python atlas_spot_backtest.py --start 2022-01-01 --end 2024-01-01

# IS/OOS 분리 결과 저장
python atlas_spot_backtest.py --wf --save
```

### 백테스트 비용 모델
| 티어 | 심볼 | 수수료 | 슬리피지 |
|------|------|--------|---------|
| Tier 1 | BTC, ETH | 0.1% | 0.03% |
| Tier 2 | BNB, SOL, XRP... | 0.1% | 0.08% |
| Tier 3 | 그 외 | 0.1% | 0.15% |

---

## 대시보드

`http://서버IP:8080` 접속 후 `DASH_PASSWORD`로 인증.

| 섹션 | 내용 |
|------|------|
| 자산 현황 | 실잔고(Binance API), 누적 PnL, 현재 레짐 |
| 수익 곡선 | 누적 Equity Curve |
| 전략별 성과 | WR, PF, avgR, 거래수 |
| Kelly 통계 | 전략별 Full/Half Kelly 계산 |
| 열린 포지션 | 현재 보유 중인 포지션 목록 |
| 최근 거래 | 최근 50건 거래 내역 |
| 드로다운 곡선 | MDD 추이 |
| 월별 PnL | 히트맵 |

> **추가 입금 자동 반영**: 실잔고 API(Binance Spot)를 60초마다 조회하여 입금 즉시 자산에 반영됩니다.

---

## Telegram 명령어

봇 실행 중 Telegram에서 직접 제어:

| 명령 | 기능 |
|------|------|
| `/status` | 현재 레짐, 포지션 수, 일일 PnL |
| `/pause` | 신규 진입 일시 중단 |
| `/resume` | 진입 재개 |
| `/positions` | 열린 포지션 목록 |
| `/stop` | 봇 안전 종료 (열린 포지션 유지) |

---

## MCP 서버 (Claude Code 연동)

```bash
# 등록
claude mcp add --transport stdio atlas -- python atlas_mcp_server.py
```

| 도구 | 기능 |
|------|------|
| `get_trade_history(days)` | 최근 N일 거래 내역 |
| `get_pnl_by_strategy()` | 전략별 성과 비교 |
| `check_alert_thresholds()` | MDD/WR/PF 경고 판단 |
| `get_error_logs(n)` | 최근 ERROR 로그 |

---

## 파일 구조

```
ATLAS-Crypto-Trading-Bot/
├── atlas_spot_main.py          ← 메인 봇 엔진 (라이브)
├── atlas_spot_backtest.py      ← Walk-Forward 백테스트
├── atlas_spot_strategies.py    ← 7개 전략 구현 (S1~S7)
├── atlas_spot_universe.py      ← 동적 심볼 유니버스
├── atlas_spot_config.py        ← 전략/리스크 파라미터
├── atlas_indicators.py         ← 기술 지표 라이브러리 (공유)
├── atlas_regime.py             ← 레짐 분류기 (공유)
├── atlas_web_dashboard.py      ← FastAPI 대시보드 백엔드
├── dashboard_ui.html           ← 대시보드 HTML UI
├── atlas_mcp_server.py         ← Claude Code MCP 연동
├── weekly_report.py            ← 주간 성과 Telegram 리포트
├── healthcheck.sh              ← 프로세스 감시 + 자동 재시작
├── requirements.txt
├── .env.example
└── tests/
    ├── test_indicators.py
    └── test_regime.py
```

---

## 테스트

```bash
pytest tests/ -v
```

---

## 운영 가이드

### 처음 시작
1. `.env` 파일 작성 (API 키, TG 토큰, 초기 자본)
2. `python atlas_spot_main.py` 실행
3. `uvicorn atlas_web_dashboard:app --port 8080` 대시보드 실행
4. crontab에 `healthcheck.sh` 등록 (`*/5 * * * *`)

### 모니터링 기준
| 지표 | 경고 | 중단 |
|------|------|------|
| MDD | > 15% | > 20% |
| 승률 | < 35% | — |
| PF | < 1.0 | — |

### 수동 종료
```bash
touch /tmp/ATLAS_SPOT_STOP   # Kill Switch 파일 생성 → 안전 종료
```

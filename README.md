# ATLAS — 암호화폐 현물 자동매매 봇

Binance 현물 시장에서 돌아가는 Python 트레이딩 봇. BTC 장세(레짐)를 판정해 그 장세에 맞는 전략만 활성화하고, Kelly·드로다운 래칫·전략 건강도로 베팅 크기를 자동 조절합니다.

- **라이브 운용 전략: 4개** (S3 / S4 / S5 / S6) — 나머지는 백테스트·연구 전용
- 레버리지 없음, 롱 전용, 거래소 측 보호주문(SL/TP) + 소프트웨어 SL 이중화
- 모든 매매 파라미터는 `atlas_spot_config.py` 한 곳에 모여 있고, CI가 미검증 기능의 기본 ON 병합을 차단합니다

---

## 전략 구성

구현된 전략은 8개지만, **레짐 라우팅에 배정된 4개만 라이브에서 진입합니다.** 나머지는 신호가 나와도 진입하지 않습니다 (`DEFAULT_ACTIVE_STRATEGIES = ['S3','S4','S5','S6']`).

| ID | 이름 | TF | 진입 조건 | 라이브 |
|----|------|----|----------|:---:|
| S1 | Buy & Hold | 1D | 벤치마크 (성과 비교용) | — |
| S2 | SMA 골든크로스 | 1D | SMA 50/200 크로스 | — |
| S3 | EMA 추세추종 | 4H | EMA20이 EMA50 상향 돌파 + 종가 > EMA200 | ✅ |
| S4 | RSI 평균회귀 | 1D | RSI < 30 과매도 + BB 하단 반등 | ✅ |
| S5 | BB 밴드반등 | 1D | BB 하단 이탈 후 복귀 + RSI 확증 | ✅ |
| S6 | Donchian 돌파 | 1D | 20일 신고가 돌파 + 거래량 스파이크 + VWAP 확증 | ✅ |
| S7 | MACD 모멘텀 | 4H | MACD 히스토그램 방향 전환 + EMA200 필터 | — |
| S7V4 | MACD 모멘텀(강화) | 4H | S7 + 추가 필터 (벤치 상태) | — |

---

## 레짐 분류 & 라우팅

BTC 1D 캔들의 **ADX + EMA200 + ATR/가격**으로 장세를 판정합니다 (`atlas_regime.py`). 판정에는 **형성 중인 봉을 제외한 완성봉만** 사용합니다 — 봉 중간에 값이 흔들려 백테스트가 검증한 적 없는 경로로 새는 것을 막기 위함입니다.

### 판정 순서

1. `ATR/가격 ≥ 7%` → **CRISIS**
2. `ADX ≥ 25` → BTC ≥ EMA200이면 TRENDING_UP, 아니면 TRENDING_DOWN
   - 단, **ADX 기울기가 하락 중이고 ADX < 30이면 WEAK_TREND로 강등** (추세 소멸 구간)
   - 단, TRENDING_UP인데 **4H ADX가 0~20이면 MICRO_RANGING** (일봉만 보면 추세, 실제로는 횡보)
3. `ADX < 20` → **RANGING**
4. 그 외(20 ≤ ADX < 25) → **WEAK_TREND**

### 레짐별 활성 전략

| 레짐 | 활성 전략 | 리스크 배율 |
|------|----------|------------|
| TRENDING_UP | S6 (돌파) | 100% |
| RANGING | S4, S5 (평균회귀) | 100% |
| WEAK_TREND | S3, S5, S6 | **50%** |
| TRENDING_DOWN | S4 (과매도 반등만) | **30%** |
| MICRO_RANGING | 전면 차단 | — |
| CRISIS | 전면 차단 | — |
| UNKNOWN | 전면 차단 | — |

> S5는 하락장 실전 10전 0승 이력으로 TRENDING_DOWN에서 제외돼 있습니다.

---

## 리스크 관리

### 포지션 사이징

```
실효 리스크 = 기본리스크 × Kelly × 래칫 × 레짐 × 펀딩 × RS × 건강도
주문 명목가 = min( 자본 × 실효리스크 / SL거리 ,  자본 × 배분상한 )
```

배분 상한에 걸리면 수량을 줄이고 **실효 리스크를 역산해서 기록**합니다. 상한 때문에 실제로 건 위험이 줄었는데 의도값을 그대로 저장하면 Kelly·건강도·학습기 통계가 전부 부풀기 때문입니다.

| 상수 | 값 | 의미 |
|------|-----|------|
| `SPOT_BASE_RISK_PCT` | 2.0% | 거래당 기본 리스크 |
| `SPOT_MAX_ALLOC_PCT` | 15% | 단일 종목 최대 배분 |
| `SPOT_MIN_ALLOC_PCT` | 2% | 단일 종목 최소 배분 |
| `SPOT_MAX_POSITIONS` | 15 | 동시 포지션 상한 |
| `SPOT_EQUITY_PER_SLOT` | $20 | 슬롯 1개당 최소 자본 (자본 연동) |
| `SPOT_RESERVE_PCT` | 10% | USDT 최소 예비금 |
| `SPOT_MIN_ORDER_USDT` | $5 | Binance NOTIONAL 하한 |
| `SPOT_MAX_SL_PCT` | 20% | SL 거리 상한 (초과 시 진입 차단) |
| `SPOT_DAILY_LOSS_LIMIT` | −4% | 일간 손실 한도 (초과 시 당일 신규 진입 차단) |

> **실제 동시 포지션 수는 자본에 연동됩니다**: `min(15, 자본 ÷ $20)`. 소액 계좌에서 과분할되면 주문이 NOTIONAL 턱걸이가 되고 수수료 드래그만 커지기 때문입니다.

### Kelly 스케일

전략별 최근 200건(실거래만)의 승률·PF로 **half-Kelly**를 계산합니다.

- 최소 표본 **10건** — 미만이면 하한 고정
- 하한 **0.15**, 상한은 조건부: `승률 ≥ 55% AND PF ≥ 1.5`면 **2.00**, 아니면 **1.50**
- 전패 → 하한, 전승 → 1.0

### 드로다운 래칫

| MDD | 리스크 배율 |
|-----|------------|
| < 5% | 100% |
| 5~8% | 70% |
| ≥ 8% | 40% |

하드 구간(≥8%)에서 **바닥 대비 +15% 회복** 시 100%가 아니라 **70% 중간 단계로만 복원**합니다.

### 전략 건강도 자기교정

최근 45일 창에서 **수수료 차감 net PF**로 전략을 감시합니다 (최소 표본 20건).

| net PF | 조치 |
|--------|------|
| < 1.0 | 리스크 **50% 감봉** |
| < 0.7 | **신규 진입 차단** (회복 시 자동 해제, 텔레그램 알림) |

### 비용 대비 엣지 게이트

SL이 좁을수록 명목가가 커져 왕복비용이 R을 잠식합니다. 왕복비용이 **1R의 20%를 넘으면 진입을 차단**합니다 (`SPOT_MAX_COST_PER_R = 0.20`). 승률이 좋아도 비용이 avg_r을 넘으면 그 거래는 마이너스 기대값입니다.

### 모멘텀 RS Gate

- 유니버스 상위 33%(주도주 티어)에 **리스크 ×1.30 부스트**
- 차단 임계값 `MOMENTUM_RS_GATE_PCT = 0.99` — **현재 이 값으로는 어떤 심볼도 차단되지 않습니다.** rank_pct의 최댓값이 `(n−1)/n`이라 0.99를 넘을 수 없기 때문입니다. 기존 동작 보존을 위해 값만 밖으로 꺼낸 상태이며, 실제 차단 임계값은 WFO 검증 후 결정합니다.

### 펀딩비 필터 (S3 / S6)

- 펀딩비 ≥ **+0.05%/8h** → 진입 차단 (롱 과밀)
- 펀딩비 ≤ **−0.01%/8h** → 리스크 **+20%** (숏 스퀴즈 기대)
- 퍼프 마켓이 없는 심볼은 필터 미적용(통과)

### 기본 OFF 기능 (검증 전 잠금)

| 기능 | 상수 | 상태 | 켜기 전 조건 |
|------|------|------|-------------|
| 추적 손절 | `SPOT_TRAIL_ENABLED` | **False** | reoptimize/월간 WFO로 IS→OOS 개선 확인. 추세추종(S3·S6)에 유리, 평균회귀(S4·S5)에 불리 경향 |
| 자기주도 학습기 | `SPOT_LEARN_ENABLED` | **False** | 켜면 (전략×레짐) 배분이 **Kelly와 건강도를 곱하지 않고 대체**합니다 — 지금 S4·S5를 조이는 건강도 감봉이 꺼집니다. 조합당 8건 이상 쌓인 뒤 재검토 |

> CI의 `guard` job이 이 두 상수가 `False`인지 정규식으로 검사합니다. 검증 전 기능이 기본 ON으로 병합되는 것을 막기 위한 래칫입니다.

---

## 주문 실행 & 포지션 보호

- **거래소 측 보호주문**: 진입 직후 STOP_LOSS_LIMIT(+OCO로 TP 동봉) 등록 → **봇이 죽어 있어도 거래소가 손절을 집행**합니다. S5는 TP가 매 봉 BB 상단으로 갱신되는 동적 목표라 OCO에서 제외(스탑 단독).
- **소프트웨어 SL**: 60초 폴링으로 백업 판정. 거래소 주문 등록에 실패하면 이쪽이 유일한 방어선이 됩니다.
- **보호주문 자가복구**: OCO 한 레그가 취소되면 전체가 사멸하므로, 5분 주기로 상태를 확인해 재등록합니다.
- **검증(reconcile) 루프**: 10분마다 DB ↔ 거래소 잔고를 대조합니다. 잔고 0이면 먼저 **보호주문 체결 여부를 확인**해 실제 체결가·정확한 사유(SL/TP)로 기록하고, 확인되지 않을 때만 수동매도로 정리합니다. dry-run에서는 이 루프가 동작하지 않습니다(가상 포지션이 전부 삭제되는 것을 방지).
- **실수령 수량 보정**: 매수 체결량에서 기초자산 수수료를 뺀 실수령량으로 매도 주문을 겁니다. gross로 걸면 보유량 초과(-2010)로 SL/TP 등록이 통째로 실패합니다.

---

## 심볼 유니버스

- Binance 현물 24h 거래량 **$10M 이상**, 상위 **50개** USDT 페어 자동 선별
- **4시간 주기** 갱신 (주도주 전환을 빠르게 반영)
- 제외 규칙 3종:
  1. 스테이블·래핑 코인 기초자산 목록 (USDT/USDC/DAI/FDUSD/USDE, FBTC/LBTC, PAXG/XAUT, EURS 등)
  2. 레버리지·인버스 토큰 키워드 (`3L 3S 2L 2S UP DOWN BULL BEAR`)
  3. **가격 기반 자동 감지** — 현재가가 $0.97~$1.03이면 스테이블로 간주해 제외
- 45일 모멘텀 랭킹으로 정렬 (RS Gate 입력)
- 유니버스에서 빠져도 **보유 중인 심볼은 계속 관리**합니다 (관리 전용, 신규 진입은 금지)

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
# 필수
BINANCE_API_KEY=...          # Binance Spot API 키 (거래 권한, IP 제한 권장)
BINANCE_API_SECRET=...
TG_TOKEN=...                 # Telegram 봇 토큰
TG_CHAT_ID=...               # Telegram 채팅 ID (이 ID 외의 명령은 무시)

# 대시보드
DASH_PASSWORD=...            # 반드시 변경 (미설정 시 랜덤 생성 → 로그인 불가)
DASH_REFRESH_SEC=60
BOT_START_ARGS=--strategies S3,S4,S5,S6   # ⚠️ 아래 주의 참조

# 성과 계산 / 경로
INITIAL_CAPITAL=1000
DB_FILE=/root/atlas_spot/state/atlas_spot.db

# MCP 서버 (Claude Code 연동 시에만)
VULTR_HOST=... VULTR_PORT=22 VULTR_USER=root VULTR_KEY_PATH=...
REMOTE_DIR=/root/atlas_spot
MCP_SSH_TRUST_NEW=            # '1'일 때만 미등록 호스트 신뢰 (기본 거부)

# 월간 WFO 자동화
WFO_AUTO_REOPT=1              # WFO 판정 미달 시 재최적화(제안) 트리거
```

> **`BOT_START_ARGS` 주의**: 대시보드의 `start` 버튼은 이 인자로 봇을 재기동합니다. **미설정이면 인자 없이(= 라이브 모드, 기본 전략) 기동**되므로, dry-run으로 운용 중이었다면 실주문 모드로 바뀝니다. 운용 구성을 반드시 명시하세요.

> **`DB_FILE` 주의**: 이 값은 **파일이 실제로 존재할 때만** 우선 적용됩니다. 존재하지 않으면 경고를 남기고 정식 경로로 폴백합니다 — 과거에 오타난 경로가 리포트를 조용히 빈 데이터로 만든 사고가 있었습니다.

---

## 실행

### systemd (운영 권장)

`deploy/` 디렉터리에 유닛 파일이 들어 있습니다.

| 유닛 | 역할 | 스케줄 |
|------|------|--------|
| `atlas-spot.service` | 라이브 봇 | 상시 (`Restart=on-failure`) |
| `atlas-dash.service` | 대시보드 (:8080) | 상시 (`Restart=always`) |
| `atlas-weekly-report.timer` | 주간 성과 리포트 | 매주 월 00:00 UTC |
| `atlas-wfo-report.timer` | 월간 Walk-Forward 재검증 | 매월 1일 03:00 UTC |
| `atlas-wfo-reopt.service` | 파라미터 재최적화(제안) | WFO 판정 미달 시 조건부 |
| `atlas-logrotate.conf` | 로그 로테이션 | — |

배치 잡에는 **`MemoryMax`/`CPUQuota`/`Nice` 제한이 걸려 있습니다** — 서버 RAM이 951MB인데 라이브 봇 혼자 ~700MB를 쓰므로, 백테스트가 봇을 밀어내지 않도록 cgroup으로 격리합니다.

```bash
sudo cp deploy/*.service deploy/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now atlas-spot atlas-dash
sudo systemctl enable --now atlas-weekly-report.timer atlas-wfo-report.timer
```

### 직접 실행

```bash
python atlas_spot_main.py                          # 라이브
python atlas_spot_main.py --dry-run                # 가상 실행 (주문 없음)
python atlas_spot_main.py --strategies S3,S6       # 전략 지정
uvicorn atlas_web_dashboard:app --host 0.0.0.0 --port 8080
```

`atlas_spot_main.py`의 CLI 인자는 `--dry-run`, `--strategies` 두 개뿐입니다.

---

## 백테스트 & 검증

```bash
# Walk-Forward (IS 2021~2023 / OOS 2024~현재)
python atlas_spot_backtest.py --wf

# 롤링 윈도우 4단계 추가
python atlas_spot_backtest.py --wf --rolling

# 단일 전략 / 심볼 / 기간
python atlas_spot_backtest.py --strategy S3
python atlas_spot_backtest.py --symbol BTCUSDT --strategy S6
python atlas_spot_backtest.py --start 2022-01-01 --end 2024-01-01

# 로컬 CSV 캐시 사용 (API 재다운로드 방지)
python atlas_spot_backtest.py --wf --data-dir data/
```

전체 플래그: `--start --end --strategy --symbol --top --extended --wf --rolling --data-dir --save --risk`
(`--save`는 항상 True로 고정돼 있어 지정 여부와 무관하게 결과가 저장됩니다.)

### 비용 모델

| 티어 | 심볼 | 수수료(편도) | 슬리피지(편도) |
|------|------|-------------|---------------|
| Tier 1 | BTC, ETH | 0.1% | 0.03% |
| Tier 2 | BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT, MATIC, UNI | 0.1% | 0.08% |
| Tier 3 | 그 외 | 0.1% | 0.15% |

초기 자본 $10,000 기준. BNB로 수수료를 결제하면 25% 할인(0.075%)됩니다. 왕복비용이 1R의 20%를 넘는 신호는 백테스트에서도 진입이 차단됩니다(라이브와 동일 게이트).

### 라이브 ↔ 백테스트 패리티

두 경로가 갈라지면 검증한 적 없는 봇이 됩니다. 이를 막는 장치:

- **공유 규칙 모듈** `atlas_rules.py` — `trailing_sl`을 라이브·백테스트가 **같은 함수 객체**로 사용 (테스트가 `is` 동일성으로 고정)
- **특성화 스냅샷** `tests/data/*.json` — 백테스트 수치 결과를 잠금. 행동 보존 리팩토링에서 이 스냅샷이 깨지면 리팩토링이 틀린 것입니다
- **레짐 패리티 테스트** — 백테스트가 라이브와 같은 입력(4H ADX 포함)으로 레짐을 판정하는지 검증

### 월간 자동 검증 & 재최적화

매월 1일 03:00 UTC에 WFO를 재실행하고 텔레그램으로 결과를 보냅니다. OOS 판정이 미달인 전략이 있으면 재최적화를 트리거합니다.

> **재최적화는 제안 전용입니다.** config를 자동으로 고치지 않고 JSON + 텔레그램 알림만 냅니다. 반영은 사람이 판단합니다.

```bash
python monthly_wfo_report.py --rolling      # 수동 실행
python reoptimize.py --strategies S3,S6     # 재최적화 제안
python capital_plan.py --equity 565         # 자본 대비 수익성 설계 점검
python check_protection.py                  # 보호주문 점검 (읽기전용, --fix로 DB 수량 교정)
```

---

## 대시보드

`http://서버IP:8080` 접속 후 `DASH_PASSWORD`로 인증.

| 섹션 | 내용 |
|------|------|
| 자산 현황 | 실잔고(Binance API), 누적 PnL, 현재 레짐 |
| 수익 곡선 / 드로다운 | 누적 Equity Curve, MDD 추이 |
| 전략별 성과 | WR, PF, avgR, 거래수 |
| Kelly 통계 | 전략별 Full/Half Kelly |
| 열린 포지션 / 최근 거래 | 보유 포지션, 최근 50건 |
| 월별 PnL | 히트맵 |
| 로그 뷰어 | tail + 키워드 필터 |
| 제어 | 봇 시작/정지, **패닉 청산**(전 포지션 시장가 매도) |

**보안**:
- 토큰 인증 (`secrets.token_hex(32)`, **TTL 24시간**), 비밀번호는 상수시간 비교
- **IP당 15분에 5회 실패 시 429** 차단
- 토큰이 쿼리스트링으로 오므로 access log에서 `token=<redacted>`로 마스킹
- OpenAPI/Swagger 비활성, CSV 다운로드는 인젝션 방어(`= + - @` 이스케이프)
- 패닉 청산은 `confirm="PANIC"` 명시 필요

> **추가 입금 자동 반영**: 실잔고를 60초마다 조회하므로 입금이 즉시 자산에 반영됩니다.

---

## Telegram 명령어

발신자가 `TG_CHAT_ID`와 다르면 무시합니다.

| 명령 | 기능 |
|------|------|
| `/status` | 열린 포지션 전체 — 전략/심볼/진입가→현재가/손익% |
| `/equity` | 총자산 + USDT 잔고 |
| `/regime` | 현재 레짐 + ADX |
| `/pause` | 신규 진입 일시 중단 |
| `/resume` | 진입 재개 |
| `/stop` | 킬 스위치 생성 → 봇 안전 종료 (포지션 유지) |

---

## MCP 서버 (Claude Code 연동)

```bash
claude mcp add --transport stdio atlas -- python atlas_mcp_server.py
```

| 도구 | 기능 |
|------|------|
| `get_trade_history(days)` | 최근 N일 청산 거래 (최대 25건) |
| `get_pnl_by_strategy()` | 전략별 누적 성과 (WR/PF/avgR) |
| `check_alert_thresholds()` | 경고 판단 — WR<35%, PF<1.0, MDD>15% / 즉시중단 MDD>20% |
| `get_error_logs(n)` | 최근 WARNING/ERROR/CRITICAL 로그 |

SSH로 원격 DB를 조회합니다. `MCP_SSH_TRUST_NEW=1`이 아니면 미등록 호스트를 자동 신뢰하지 않습니다(MITM 방어).

---

## 파일 구조

```
ATLAS-Crypto-Trading-Bot/
├── atlas_spot_main.py          ← 라이브 트레이딩 엔진
├── atlas_spot_backtest.py      ← 백테스트 + Walk-Forward
├── atlas_spot_strategies.py    ← 전략 구현 (S1~S7V4)
├── atlas_spot_universe.py      ← 동적 심볼 유니버스
├── atlas_spot_config.py        ← 전략/리스크 파라미터 (단일 출처)
│
├── atlas_rules.py              ← 라이브·백테스트 공유 규칙 (leaf)
├── atlas_indicators.py         ← 기술 지표 라이브러리
├── atlas_regime.py             ← 레짐 분류기
├── atlas_learning.py           ← 자기주도 학습 배분 (기본 OFF)
│
├── atlas_bootstrap.py          ← 스크립트 공용 dotenv/자격증명 부트스트랩
├── atlas_db.py                 ← 조회 전용 DB 접속 + 경로 해석
├── atlas_notify.py             ← 배치용 동기 텔레그램 전송
│
├── atlas_web_dashboard.py      ← FastAPI 대시보드 백엔드
├── dashboard_ui.html           ← 대시보드 UI
├── atlas_mcp_server.py         ← Claude Code MCP 연동
│
├── weekly_report.py            ← 주간 성과 리포트
├── monthly_wfo_report.py       ← 월간 WFO 재검증 + 조건부 재최적화 트리거
├── reoptimize.py               ← 파라미터 재최적화 (제안 전용)
├── capital_plan.py             ← 수익성 설계 계산기
├── check_protection.py         ← 보호주문 점검·교정
│
├── deploy/                     ← systemd 유닛 + logrotate
├── .github/workflows/ci.yml
├── requirements.txt / pyproject.toml / .env.example
└── tests/                      ← 54개 모듈, 1,200개+ 테스트
    ├── conftest.py             ← 공용 부트스트랩
    └── data/                   ← 특성화 스냅샷
```

---

## 테스트 & CI

```bash
pytest tests/ -q                # 전체
python -m ruff check .          # 린트
```

CI(`.github/workflows/ci.yml`)가 강제하는 게이트:

| 게이트 | 기준 |
|--------|------|
| `ruff check .` | 실패 시 중단 (스타일이 아니라 버그 탐지기로 취급) |
| `mypy` | `atlas_indicators`, `atlas_learning`, `atlas_regime`, `capital_plan` (점진 도입) |
| `bandit -lll` | HIGH 심각도만 차단 |
| `pytest --cov-fail-under=78` | 전체 커버리지 ≥ 78% |
| 진입 경로 커버리지 | `atlas_spot_main / learning / indicators / regime / rules` ≥ **88%** |
| `guard` job | `SPOT_LEARN_ENABLED`·`SPOT_TRAIL_ENABLED`가 `False`인지 검사 |

`DeprecationWarning`은 `atlas_*` 모듈에서 발생하면 **하드 실패**로 처리합니다.

---

## 보안

실계좌를 다루므로 **접근 통제가 전략보다 먼저**다. 2026-08 점검 기준 현황.

### 거래소 API 키 (가장 중요한 방어선)

키가 유출돼도 자산을 빼갈 수 없게 만드는 것이 핵심이다. Binance API 관리에서 확인:

| 항목 | 설정 | 이유 |
|------|------|------|
| 출금(Withdrawals) | **해제** | 키가 새도 출금 불가 — 이 한 줄이 최악을 막는다 |
| IP 제한 | **서버 IP만 허용** | 키만으로는 다른 곳에서 못 쓴다 |
| 현물/마진 거래 | 현물만 허용 | 마진은 레버리지 손실 위험 |
| 선물(Futures) | **해제** | 봇은 펀딩비를 **공개 endpoint**로만 읽는다 (인증 불필요) |
| 내부이체·유니버설전송 | 해제 | 계정 간 자산 이동 차단 |

### 대시보드

봇 중지·재시작과 **전량 시장가 매도(패닉)** 를 할 수 있으므로 노출 범위를 좁게 유지한다.

- **접속 IP 제한** — ufw에서 특정 IP만 8080 허용:
  ```bash
  ufw allow from <내-IP> to any port 8080 proto tcp
  ```
  집 IP가 바뀌면 대시보드만 안 열린다. SSH로 들어가 이전 규칙을 지우고 새 IP로 다시 걸면 된다:
  ```bash
  ufw status numbered          # 기존 8080 규칙 번호 확인
  ufw delete <번호>
  ufw allow from <새-IP> to any port 8080 proto tcp
  ```
- 토큰 인증(TTL 24h), 비밀번호 상수시간 비교, 로그인 IP당 15분 5회 실패 시 429
- 접근 로그의 토큰 마스킹, CSV 인젝션 이스케이프, Swagger 비활성
- 보안 헤더: CSP(`frame-ancestors 'none'`으로 클릭재킹 차단, `connect-src 'self'`로 토큰 외부 전송 차단), `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`
- 외부 스크립트(chart.js)는 **SRI 해시로 고정** — 버전을 올릴 때 `integrity` 값도 반드시 함께 갱신할 것

### 시크릿

- `.env`는 커밋하지 않는다(`.gitignore`가 `.env.*`까지 차단). 권한은 `600`.
- remote URL에 토큰을 넣지 말 것 — 아래 설치 항목의 경고 참조

### 아직 열려 있는 항목

| 항목 | 상태 | 비고 |
|------|------|------|
| 대시보드 TLS | 미적용 | IP 제한으로 무차별 스캔은 막았으나 **통신은 평문**이다. 같은 네트워크(카페 WiFi 등)에서의 도청은 못 막는다. 도메인 + Caddy 자동 TLS가 다음 단계 |
| SSH 하드닝 | 미적용 | `PermitRootLogin yes` + `PasswordAuthentication yes`. 적용 절차: 키 접속 확인 → `PasswordAuthentication no`, `PermitRootLogin prohibit-password` → **현재 세션을 열어둔 채** 다른 세션으로 접속 검증 → `systemctl reload sshd`. fail2ban도 함께 |
| 서버 동거 | 구조적 | 같은 서버에서 다른 프로젝트가 root로 공개 포트를 열고 있다. 그쪽이 뚫리면 이 봇의 `.env`도 함께 노출된다 |

---

## 운영 가이드

### 모니터링 기준

| 지표 | 경고 | 즉시 중단 |
|------|------|----------|
| MDD | > 15% | > 20% |
| 승률 | < 35% | — |
| PF | < 1.0 | — |

거래 10건 미만이면 통계 신뢰도가 낮으므로 판단을 보류합니다.

### 백업

봇이 **6시간마다** DB 스냅샷을 `state/backups/`에 저장하고 **최근 28개(7일치)** 만 보관합니다. 배포로 봇을 재시작하기 전에는 별도로 백업해 두는 편이 안전합니다.

### 수동 종료

```bash
touch /tmp/ATLAS_SPOT_STOP   # 킬 스위치 → 안전 종료 (포지션 유지)
```

킬 스위치가 있으면 봇은 기동 즉시 정상 종료(exit 0)하므로 `Restart=on-failure` 재기동 루프에 빠지지 않습니다. 대시보드의 `start`는 이 파일을 자동으로 지웁니다.

### 거래소에 직접 걸어둔 주문

봇은 자기가 등록한 보호주문(DB에 ID가 있는 것)만 관리합니다. 사용자가 거래소에서 직접 건 주문은 봇의 관리 대상이 아닙니다.

#!/bin/bash
# ATLAS v2 헬스체크 — crontab */5 * * * * 로 실행
# 봇 프로세스 감지 → 죽으면 TG 알림 + 자동 재시작

ATLAS_DIR="/root/atlas_bot"
ENV_FILE="$ATLAS_DIR/.env"
LOG_DIR="$ATLAS_DIR/logs"

TG_TOKEN=$(grep "^TG_TOKEN=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '[:space:]')
TG_CHAT_ID=$(grep "^TG_CHAT_ID=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '[:space:]')

tg_send() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d "chat_id=${TG_CHAT_ID}" \
        --data-urlencode "text=$1" > /dev/null
}

if pgrep -f "atlas_v2_main.py" > /dev/null; then
    exit 0
fi

tg_send "🚨 ATLAS v2 프로세스 종료 감지 — 자동 재시작 중..."

cd "$ATLAS_DIR" || exit 1
LOG_FILE="$LOG_DIR/atlas_v2_$(date +%Y%m%d_%H%M%S).log"
nohup python3 atlas_v2_main.py > "$LOG_FILE" 2>&1 &

sleep 15

if pgrep -f "atlas_v2_main.py" > /dev/null; then
    tg_send "✅ ATLAS v2 재시작 완료 ($(date '+%Y-%m-%d %H:%M UTC'))"
else
    tg_send "❌ ATLAS v2 재시작 실패 — 수동 확인 필요
로그: $LOG_FILE"
fi

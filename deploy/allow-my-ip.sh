#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 대시보드(:8080) 접속 허용 IP 갱신
#
# 가정용 인터넷은 IP가 주기적으로 바뀐다. 바뀌면 ufw 규칙에 걸려
# 대시보드만 안 열린다(SSH는 계속 열려 있으므로 복구는 항상 가능).
# 이 스크립트를 SSH 로 접속한 상태에서 실행하면 **지금 접속한 IP**로
# 규칙을 갈아끼운다.
#
#   ssh root@<서버> 'bash /root/atlas_spot/deploy/allow-my-ip.sh'
#
# IP를 직접 지정할 수도 있다:
#   bash deploy/allow-my-ip.sh 1.2.3.4
#
# 이전 IP 규칙은 **남기지 않고 지운다** — 동적 IP는 통신사가 다른
# 사람에게 재할당하므로, 남겨두면 모르는 사람에게 대시보드를 열어두는
# 셈이 된다(이 대시보드에는 전량 시장가 매도 버튼이 있다).
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8080
F2B_JAIL=/etc/fail2ban/jail.d/atlas-sshd.local

IP="${1:-${SSH_CLIENT%% *}}"

if [ -z "${IP:-}" ]; then
    echo "접속 IP를 알 수 없다." >&2
    echo "SSH로 접속한 상태에서 실행하거나, IP를 인자로 넘길 것:" >&2
    echo "  bash deploy/allow-my-ip.sh 1.2.3.4" >&2
    exit 1
fi

if ! printf '%s' "$IP" | grep -qE '^[0-9]{1,3}(\.[0-9]{1,3}){3}$'; then
    echo "IPv4 형식이 아니다: $IP" >&2
    exit 1
fi

echo "허용할 IP: $IP"

# 새 규칙을 **먼저** 넣는다. 지우고 넣으면 그 사이에 접속이 끊긴다.
ufw allow from "$IP" to any port "$PORT" proto tcp \
    comment "ATLAS dashboard - home IP" >/dev/null

# 이 IP가 아닌 기존 8080 규칙을 전부 제거.
# 번호는 지울 때마다 밀리므로 **역순**으로 지운다.
for n in $(ufw status numbered \
           | grep -E "(^|[^0-9])${PORT}(/tcp)?[[:space:]]" \
           | grep -v "$IP" \
           | grep -oE '^\[[[:space:]]*[0-9]+' \
           | grep -oE '[0-9]+' \
           | sort -rn); do
    yes | ufw delete "$n" >/dev/null 2>&1 || true
done

# fail2ban 화이트리스트도 같이 맞춘다 (키 인증이라 밴될 일은 거의 없지만,
# 두 곳이 어긋나 있으면 나중에 원인을 찾기 어렵다).
if [ -f "$F2B_JAIL" ]; then
    sed -i "s|^ignoreip .*|ignoreip = 127.0.0.1/8 ::1 $IP|" "$F2B_JAIL"
    systemctl reload fail2ban 2>/dev/null || systemctl restart fail2ban || true
fi

echo
echo "── 적용 결과 ──"
ufw status | grep -E "^${PORT}" || echo "  (8080 규칙 없음 — 확인 필요)"
[ -f "$F2B_JAIL" ] && grep '^ignoreip' "$F2B_JAIL"
printf '  대시보드 로컬 응답: %s\n' \
    "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || echo '실패')"

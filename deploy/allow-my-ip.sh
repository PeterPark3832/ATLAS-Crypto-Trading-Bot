#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 대시보드(:8080) 접속 허용 IP 관리
#
# 가정용·모바일 인터넷은 IP가 주기적으로 바뀐다. 바뀌면 ufw 규칙에 걸려
# 대시보드만 안 열린다(SSH는 계속 열려 있으므로 복구는 항상 가능).
#
# 사용법:
#   bash allow-my-ip.sh                 # 지금 SSH 접속한 IP 하나로 교체
#   bash allow-my-ip.sh --add 1.2.3.4   # 기존은 두고 하나 추가 (폰 등)
#   bash allow-my-ip.sh 1.2.3.4 5.6.7.8 # 목록을 정확히 이 IP들로 교체
#   bash allow-my-ip.sh --list          # 현재 허용 목록만 보기
#   bash allow-my-ip.sh --blocked       # 최근 차단된 IP (내 폰 IP 찾을 때)
#
# ⚠️ 이전 IP를 습관적으로 남겨두지 말 것. 동적 IP는 통신사가 다른 사람에게
#    재할당하므로, 남겨두면 모르는 사람에게 대시보드를 열어두는 셈이 된다
#    — 이 화면에는 **전량 시장가 매도** 버튼이 있다.
#    특히 모바일(LTE/5G)은 IP가 자주 바뀌니 되도록 그때그때 교체할 것.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8080
F2B_JAIL=/etc/fail2ban/jail.d/atlas-sshd.local

_valid_ip() {
    printf '%s' "$1" | grep -qE '^[0-9]{1,3}(\.[0-9]{1,3}){3}$'
}

# 현재 8080 허용 IP 목록
_current_ips() {
    ufw status | grep -E "^${PORT}(/tcp)?[[:space:]]" \
        | grep -oE '[0-9]{1,3}(\.[0-9]{1,3}){3}' | sort -u
}

case "${1:-}" in
    --list)
        echo "현재 ${PORT} 허용 IP:"
        _current_ips | sed 's/^/  /' || echo "  (없음)"
        exit 0
        ;;
    --blocked)
        echo "최근 ${PORT} 차단 IP (횟수 많은 순 — 폰에서 대시보드를 한 번"
        echo "열어본 뒤 실행하면 그 IP가 목록에 뜬다):"
        grep -h "DPT=${PORT}" /var/log/ufw.log /var/log/kern.log 2>/dev/null \
            | tail -500 | grep -oE 'SRC=[0-9.]+' | cut -d= -f2 \
            | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /'
        exit 0
        ;;
esac

# ── 목표 IP 목록 결정 ────────────────────────────────────────
TARGETS=()
if [ "${1:-}" = "--add" ]; then
    [ -n "${2:-}" ] || { echo "--add 뒤에 IP를 넣을 것" >&2; exit 1; }
    mapfile -t TARGETS < <(_current_ips)
    TARGETS+=("$2")
elif [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${SSH_CLIENT%% *}")
fi

if [ ${#TARGETS[@]} -eq 0 ] || [ -z "${TARGETS[0]:-}" ]; then
    echo "허용할 IP를 알 수 없다. SSH로 접속한 상태에서 실행하거나 IP를 넘길 것." >&2
    exit 1
fi

# 중복 제거 + 형식 검사
mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | sort -u)
for ip in "${TARGETS[@]}"; do
    _valid_ip "$ip" || { echo "IPv4 형식이 아니다: $ip" >&2; exit 1; }
done

echo "허용할 IP: ${TARGETS[*]}"

# 새 규칙을 **먼저** 넣는다. 지우고 넣으면 그 사이에 접속이 끊긴다.
for ip in "${TARGETS[@]}"; do
    ufw allow from "$ip" to any port "$PORT" proto tcp \
        comment "ATLAS dashboard" >/dev/null
done

# 목표에 없는 기존 규칙 제거. 번호는 지울 때마다 밀리므로 **역순**으로.
keep=$(printf '%s\n' "${TARGETS[@]}" | paste -sd'|')
for n in $(ufw status numbered \
           | grep -E "(^|[^0-9])${PORT}(/tcp)?[[:space:]]" \
           | grep -vE "$keep" \
           | grep -oE '^\[[[:space:]]*[0-9]+' | grep -oE '[0-9]+' \
           | sort -rn); do
    yes | ufw delete "$n" >/dev/null 2>&1 || true
done

# fail2ban 화이트리스트도 같이 맞춘다 (두 곳이 어긋나면 원인 추적이 어렵다).
if [ -f "$F2B_JAIL" ]; then
    sed -i "s|^ignoreip .*|ignoreip = 127.0.0.1/8 ::1 ${TARGETS[*]}|" "$F2B_JAIL"
    systemctl reload fail2ban 2>/dev/null || systemctl restart fail2ban || true
fi

echo
echo "── 적용 결과 ──"
ufw status | grep -E "^${PORT}" | sed 's/^/  /' || echo "  (8080 규칙 없음 — 확인 필요)"
[ -f "$F2B_JAIL" ] && grep '^ignoreip' "$F2B_JAIL" | sed 's/^/  /'
printf '  대시보드 로컬 응답: %s\n' \
    "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || echo '실패')"

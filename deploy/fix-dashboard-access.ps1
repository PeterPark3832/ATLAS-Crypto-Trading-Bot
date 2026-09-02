# ─────────────────────────────────────────────────────────────
# 대시보드 접속 복구 (Windows 원클릭)
#
# 가정용 IP가 바뀌면 ufw 규칙에 걸려 대시보드만 안 열린다.
# 이 스크립트를 실행하면 SSH로 서버에 붙어 지금 IP로 규칙을 갈아끼우고,
# 성공하면 브라우저까지 열어준다.
#
# 사용법:
#   .\fix-dashboard-access.ps1                    # ATLAS_SERVER 환경변수 사용
#   .\fix-dashboard-access.ps1 -Server 1.2.3.4    # 직접 지정
#
# 서버 주소를 한 번만 등록해두면 그 뒤로는 인자 없이 실행하면 된다:
#   setx ATLAS_SERVER "<서버IP>"
# ─────────────────────────────────────────────────────────────
param(
    [string]$Server = $env:ATLAS_SERVER,
    [string]$User   = 'root',
    [int]$Port      = 8080,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Server)) {
    Write-Host '서버 주소가 없습니다.' -ForegroundColor Red
    Write-Host '  한 번만 등록:  setx ATLAS_SERVER "<서버IP>"'
    Write-Host '  또는 직접:     .\fix-dashboard-access.ps1 -Server <서버IP>'
    exit 1
}

Write-Host "서버 $Server 에 접속해 현재 IP로 방화벽 규칙을 갱신합니다..." -ForegroundColor Cyan

# 서버의 allow-my-ip.sh 가 $SSH_CLIENT 를 읽어 지금 접속한 IP를 그대로 쓴다.
$out = & ssh.exe "$User@$Server" 'bash /root/atlas_spot/deploy/allow-my-ip.sh' 2>&1
$out | ForEach-Object { Write-Host "  $_" }

if ($LASTEXITCODE -ne 0) {
    Write-Host '실패했습니다. SSH 키 접속이 되는지 먼저 확인하세요.' -ForegroundColor Red
    Read-Host '엔터를 누르면 닫힙니다'
    exit 1
}

# 로컬 응답 200 확인은 서버 스크립트가 이미 출력한다. 여기서는 실제
# 바깥에서 열리는지(=규칙이 먹었는지)를 브라우저로 확인한다.
if (-not $NoBrowser) {
    Start-Process "http://${Server}:${Port}/"
    Write-Host '브라우저를 열었습니다.' -ForegroundColor Green
}

Write-Host '완료. 그래도 안 열리면 대시보드 서비스 상태를 확인하세요:' -ForegroundColor Green
Write-Host "  ssh $User@$Server 'systemctl status atlas-dash'"
if ($Host.Name -eq 'ConsoleHost' -and -not $NoBrowser) {
    Start-Sleep -Seconds 2
}

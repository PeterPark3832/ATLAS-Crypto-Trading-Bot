@echo off
REM 대시보드가 안 열릴 때 더블클릭하세요.
REM 지금 접속 IP로 서버 방화벽 규칙을 갱신하고 브라우저를 엽니다.
REM 서버 주소는 한 번만 등록해두면 됩니다:  setx ATLAS_SERVER "<서버IP>"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fix-dashboard-access.ps1" %*
pause

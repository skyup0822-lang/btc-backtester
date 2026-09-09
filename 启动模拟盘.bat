@echo off
cd /d "%~dp0"
set PYW=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe
set PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
curl -s -o nul -m 2 http://127.0.0.1:8765/api/status
if not errorlevel 1 (
  start "" http://127.0.0.1:8765
  exit /b 0
)
start "" "%PYW%" web_paper.py
timeout /t 4 /nobreak >nul
set OK=0
for /L %%P in (8765,1,8770) do (
  curl -s -o nul -m 3 http://127.0.0.1:%%P/api/status && set OK=1
)
if "%OK%"=="0" (
  echo.
  echo [ERROR] console failed to start. Opening debug window - send a screenshot of red errors to Codex.
  "%PY%" web_paper.py
  pause
)
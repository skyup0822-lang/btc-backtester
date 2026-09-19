@echo off
cd /d "%~dp0"
set "PYW=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
set "PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

rem If a console is already running, just open the page and exit.
curl -s -o nul -m 2 http://127.0.0.1:8765/api/status
if not errorlevel 1 (
  start "" http://127.0.0.1:8765
  exit /b 0
)

rem Supervisor loop: relaunch the console whenever it asks to restart (exit 42) or crashes.
:loop
if defined PM_RELAUNCH set "PM_NO_BROWSER=1"
"%PYW%" web_paper.py
set "RC=%ERRORLEVEL%"
if "%RC%"=="42" (
  set "PM_RELAUNCH=1"
  timeout /t 1 /nobreak >nul
  goto loop
)
if not "%RC%"=="0" (
  echo.
  echo [WARN] console exited with code %RC%, auto-restarting in 3s... (close this window to stop)
  set "PM_RELAUNCH=1"
  timeout /t 3 /nobreak >nul
  goto loop
)
exit /b 0

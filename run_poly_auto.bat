@echo off
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul || set "PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

echo ============================================
echo  Polymarket BTC-above BATCH edge backtest
echo ============================================
echo  Running: %PY% work\poly_batch.py %*
echo --------------------------------------------
"%PY%" work\poly_batch.py %*
echo.
echo  DONE. Press any key to close.
pause >nul

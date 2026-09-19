@echo off
cd /d "%~dp0"
set "PY=python"
where python >nul 2>nul || set "PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
echo ============================================
echo  Data source reachability test (no VPN)
echo ============================================
"%PY%" work\test_sources.py
echo.
echo  DONE. Press any key to close.
pause >nul

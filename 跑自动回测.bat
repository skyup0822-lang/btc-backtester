@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
where python >nul 2>nul || set "PY=C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

echo ============================================
echo  Polymarket BTC 历史 edge 自动回测
echo ============================================
echo  正在运行: %PY% work\poly_auto.py %*
echo  (若报错会显示在下面, 窗口不会自动关闭)
echo --------------------------------------------

"%PY%" work\poly_auto.py %*

echo.
echo ============================================
echo  运行结束。按任意键关闭窗口。
echo ============================================
pause >nul

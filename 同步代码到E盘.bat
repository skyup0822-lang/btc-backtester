@echo off
setlocal
set SRC=C:\Users\Administrator\Documents\Codex\2026-09-09\x20\btc-backtester
set DST=E:\btc-backtester

echo ==============================================
echo   把 C盘开发目录的最新代码同步到 E盘
echo   状态、数据、成交记录保留E盘版本, 不会被覆盖
echo ==============================================
echo.

if not exist "%SRC%\web_paper.py" (
  echo [错误] 找不到开发目录 %SRC% 。
  echo        如果之前已删除C盘目录, 请联系Codex重新生成开发目录。
  pause
  exit /b 1
)
if not exist "%DST%\web_paper.py" (
  echo [错误] 找不到 E 盘运行目录, 请先运行 搬家到E盘.bat 。
  pause
  exit /b 1
)
curl -s -o nul -m 2 http://127.0.0.1:8765/api/status
if not errorlevel 1 (
  echo [未完成] 模拟盘还在运行! 请先停止再同步: 网页点 3停止 或关掉黑色窗口。
  pause
  exit /b 1
)

robocopy "%SRC%" "%DST%" *.py *.bat /S /R:1 /W:1 /NFL /NDL /NP /XD data results __pycache__ /XF paper_state_okx.json paper_trades_okx.csv pattern_history.json paper_config_okx.json
if %ERRORLEVEL% GEQ 8 (
  echo [错误] 同步失败, 错误码 %ERRORLEVEL% , 可直接重试。
  pause
  exit /b 1
)

echo.
echo 同步完成。以后启动: E:\btc-backtester\启动模拟盘.bat
pause

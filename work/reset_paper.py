# -*- coding: utf-8 -*-
"""把模拟盘重置为干净起点: 100 USDT、空仓、从现在开始(不回放历史K线)。
旧文件全部备份为 <名>.reset-bak-<时间戳>。
用法: python work/reset_paper.py
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = time.strftime("%Y%m%d-%H%M%S")
BACKUP = ["paper_state_okx.json", "paper_trades_okx.csv", "daily_report.csv",
          "decision_log.csv", "cash_audit.csv"]

for name in BACKUP:
    p = ROOT / name
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".reset-bak-" + TS))

# 干净状态: 100 USDT / 空仓 / last_bar_ms=现在(这样不会把历史K线当成"新K线"来回放)
state = {
    "cash": 100.0, "units": 0.0, "entry_price": 0.0, "entry_equity": 0.0,
    "entry_cost": 0.0, "stop_price": 0.0, "high_watermark": 0.0,
    "signal": 0.0, "reentry_bar_ms": 0, "last_bar_ms": int(time.time() * 1000),
}
(ROOT / "paper_state_okx.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
(ROOT / "paper_trades_okx.csv").write_text(
    "time_utc,side,reason,expected_px,fill_px,slippage_pct,notional,units,pnl_acct_pct,balance\n",
    encoding="utf-8")
(ROOT / "daily_report.csv").write_text(
    "date,equity_usdt,pnl_usdt,equity_cny,pnl_cny,btc_price,units\n", encoding="utf-8")
(ROOT / "decision_log.csv").write_text("", encoding="utf-8")
(ROOT / "cash_audit.csv").write_text(
    "time,reason,cash_before,cash_after,delta,units_before,units_after\n", encoding="utf-8")

print("已重置为 100 USDT / 空仓。备份后缀: .reset-bak-" + TS)
print("新状态:", json.dumps(state, ensure_ascii=False))

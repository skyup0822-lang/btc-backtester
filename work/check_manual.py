"""自检: 手动账户的下单、记账、落盘, 以及"手动 vs 自动"对比口径。

离线跑, 价格是假的, 不联网。手动账户的文件指到临时目录, 不碰真实账本。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import web_paper as w

tmp = Path(tempfile.mkdtemp(prefix="manual_check_"))
w.MANUAL_STATE = tmp / "paper_state_manual.json"
w.MANUAL_TRADES = tmp / "paper_trades_manual.csv"

core = w.Core()
core.log = lambda m: None
core._record_decision = lambda *a, **k: None
core.live = {"price": 78000.0}

# 1) 空仓时买入 50% 现金: 现金按比例扣, 到手的币要扣掉 0.1% 手续费
ok, msg = core.manual_order("buy", 0.5)
assert ok, msg
t = core.manual
assert abs(t.cash - 50.0) < 1e-9, f"现金应剩50, 实际 {t.cash}"
assert t.units > 0
assert abs(t.units * 78000.0 - 50.0 * 0.999) < 1e-6, f"手续费没扣对: {t.units * 78000.0}"

# 2) 已持仓再买 -> 拒绝(和自动盘一样只做单向)
ok2, msg2 = core.manual_order("buy", 0.5)
assert not ok2 and "已持仓" in msg2, msg2

# 3) 涨 10% 后全部卖出
core.live = {"price": 78000.0 * 1.1}
ok3, msg3 = core.manual_order("sell", 0)
assert ok3, msg3
assert t.units == 0.0, "卖完必须清零"
assert t.cash > 50.0, f"涨了10%应该赚钱, 实际现金 {t.cash}"
assert len(t.trades) == 1
assert t.trades[0]["pnl"] > 0

# 4) 空仓再卖 -> 拒绝
ok4, msg4 = core.manual_order("sell", 0)
assert not ok4 and "空仓" in msg4, msg4

# 5) 对比: 手动赚了, 自动(没接盘)还是 100 起点
man, auto = core.compare(78000.0)
assert man["ret"] > 0, man
assert abs(auto["ret"]) < 1e-9, auto
assert man["n_trades"] == 1 and auto["n_trades"] == 0
assert man["win_rate"] == 1.0

# 6) 落盘后重新载入, 账不能变
core._save_manual()
saved = json.loads(w.MANUAL_STATE.read_text(encoding="utf-8"))
assert abs(saved["cash"] - t.cash) < 1e-9
core2 = w.Core()
core2.log = lambda m: None
t2 = core2.manual_trader()
assert abs(t2.cash - t.cash) < 1e-9, (t2.cash, t.cash)

# 6b) 成交日志里的"余额"列: 卖出后必须是真现金, 不能把已清零的仓位再加一遍
import csv as _csv
rows = list(_csv.reader(w.MANUAL_TRADES.open(encoding="utf-8")))
hdr, body = rows[0], rows[1:]
assert hdr[-1] == "balance", hdr
sell_rows = [r for r in body if r[1] == "SELL"]
assert sell_rows, "没有卖出记录"
logged = float(sell_rows[-1][-1])
assert abs(logged - t.cash) < 0.02, f"卖出后日志余额 {logged} 与真实现金 {t.cash:.2f} 不符"
assert logged < 105, f"卖出后余额被算高了: {logged}"

# 6c) 刚启动、一单都还没下时, 面板必须显示磁盘上的真实手动账户(不能默认成 100)
core4 = w.Core()
core4.log = lambda m: None
man4, _ = core4.compare(78000.0)
assert abs(man4["cash"] - t.cash) < 1e-9, f"重启后面板显示 {man4['cash']}, 磁盘上是 {t.cash}"

# 7) 没有实时报价时必须拒绝下单, 不能拿 0 价成交
core3 = w.Core()
core3.log = lambda m: None
core3._record_decision = lambda *a, **k: None
core3.live = {}
ok5, msg5 = core3.manual_order("buy", 0.5)
assert not ok5 and "报价" in msg5, msg5

# 8) 自动盘的老口径没被改坏: 不传 notional 仍按 风险/止损 尺子
p = w.PaperTrader(cash=100.0, risk_pct=0.02, stop_pct=0.15, log_path=None)
p.buy(78000.0, "signal", None)
assert abs(p.cash - (100.0 - 100.0 * 0.02 / 0.15)) < 1e-9, p.cash
# 手动指定金额时不许花超现金
p2 = w.PaperTrader(cash=100.0, log_path=None)
p2.buy(78000.0, "manual", None, notional=1e9)
assert p2.cash >= -1e-9, f"花超了现金: {p2.cash}"

print("手动交易自检通过:")
print("  买入按比例扣现金 / 扣 0.1% 手续费")
print("  持仓时再买、空仓时卖 -> 拒绝")
print("  卖出清零并记一笔盈亏, 涨10%为正")
print("  无报价时拒绝下单")
print("  落盘重载账不变")
print("  自动盘的风险尺子没被改坏")

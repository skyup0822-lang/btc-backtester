"""自检: 多品种子账户(Sleeve)的下单、记账、止损、落盘。

离线跑, 价格和K线都是假的, 账本文件指到临时目录。验的是逻辑, 不联网络。

重点验两件容易被忽略的事:
  1. 正常触发止损 -> 亏损应当接近"名义风险"(仓位13% × 止损8% ≈ 1%)
  2. 跳空穿越止损 -> 成交价是市价, 亏损会远超名义风险(止损不保证成交价)
"""
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import data_source as ds
import web_paper as w

# ---- 造一段K线: 先跌后涨, 保证会触发一次金叉 ----
N = 2000
close = np.concatenate([np.linspace(3000, 1800, 1200), np.linspace(1800, 4200, 800)])
last_close_ms = (int(time.time() * 1000) // 3_600_000) * 3_600_000
open_ms = last_close_ms - N * 3_600_000 + np.arange(N) * 3_600_000
bars = pd.DataFrame({
    "open_time_ms": open_ms, "close_time_ms": open_ms + 3_600_000,
    "open": close, "high": close * 1.01, "low": close * 0.99,
    "close": close, "volume": np.full(N, 1000.0),
})

PRICE = {"v": float(bars["close"].iloc[-1])}
ds.binance_candles = lambda n, symbol=None: bars.tail(n).reset_index(drop=True)
ds.binance_ticker = lambda symbol=None: {"last": PRICE["v"]}


def new_sleeve(tag):
    d = Path(tempfile.mkdtemp(prefix=f"sleeve_{tag}_"))
    sv = w.Sleeve("ETH", "ETHUSDT", "ETH_USDT", d / "state.json", d / "trades.csv")
    sv.cache = bars.tail(2000).reset_index(drop=True)
    return sv, d


def buy(sv, logs):
    for _ in range(2):
        sv._step(logs.append)
    assert sv.t.units > 0, f"金叉出现了却没买: signal={sv.t.signal}"
    return sv.t.entry_price


# ================= 情形1: 正常触发止损 =================
PRICE["v"] = float(bars["close"].iloc[-1])
sv, d1 = new_sleeve("normal")
logs = []
buy_px = buy(sv, logs)

PRICE["v"] = buy_px * 0.92 * 0.999  # 刚跌破止损线, 没有跳空
sv._step(logs.append)
assert sv.t.units == 0.0, "跌破止损线没卖出"
assert any("止损卖出" in m for m in logs), logs
loss1 = 1 - sv.t.cash / w.START_EQUITY
assert 0.005 < loss1 < 0.02, f"正常止损的亏损不合理: {loss1:.2%}"
assert abs(sv.t.cash - 98.93) < 0.1, f"止损后现金异常: {sv.t.cash:.4f}"

# ================= 情形2: 跳空穿越止损 =================
PRICE["v"] = float(bars["close"].iloc[-1])
sv2, d2 = new_sleeve("gap")
logs2 = []
buy_px2 = buy(sv2, logs2)

PRICE["v"] = buy_px2 * 0.5  # 直接腰斩, 跳空穿过 8% 止损线
sv2._step(logs2.append)
assert sv2.t.units == 0.0, "跳空后没卖出"
loss2 = 1 - sv2.t.cash / w.START_EQUITY
assert loss2 > 0.05, f"跳空应当亏得远超名义风险, 实际 {loss2:.2%}"
assert loss2 > loss1 * 3, "跳空亏损和正常止损没拉开差距, 成交价逻辑可能有问题"

# ================= 情形3: 落盘重载 =================
saved = json.loads((d1 / "state.json").read_text(encoding="utf-8"))
assert abs(saved["cash"] - sv.t.cash) < 1e-9
sv3 = w.Sleeve("ETH", "ETHUSDT", "ETH_USDT", d1 / "state.json", d1 / "trades.csv")
assert abs(sv3.t.cash - sv.t.cash) < 1e-9, (sv3.t.cash, sv.t.cash)

# ================= 情形4: stats 口径 =================
st = sv.stats()
assert st["label"] == "ETH"
assert abs(st["equity"] - sv.t.cash) < 1e-9, "空仓时净值应当等于现金"
assert st["n_trades"] == 1 and st["win_rate"] == 0.0

# ================= 情形5: 首次启动绝不回溯历史建仓 =================
# 这是实际踩到的 bug: last_bar_ms=0 时循环把 2000 根历史当"新K线"重放,
# 于是在两个月前的价位建了仓, 账面多出一笔根本不存在的浮盈。
PRICE["v"] = float(bars["close"].iloc[-1])
sv4, d4 = new_sleeve("fresh")
assert sv4.t.last_bar_ms == 0
logs4 = []
_cl4 = sv4.cache[sv4.cache["close_time_ms"] <= int(time.time() * 1000)]
assert w._skip_history(sv4.t, _cl4, logs4.append, "ETH") is True
assert sv4.t.last_bar_ms == int(_cl4.iloc[-1]["close_time_ms"]), "游标没定到最后一根已收盘K线"
assert any("不回溯历史" in m for m in logs4), logs4
assert w._skip_history(sv4.t, _cl4, logs4.append, "ETH") is False, "第二次应当什么都不做"
sv4._step(lambda m: None)
assert sv4.t.units == 0.0, "首次启动就回溯历史建了仓"

# ================= 情形6: 之后来了新K线, 该出手还是要出手 =================
sv5, d5 = new_sleeve("nextbar")
logs5 = []
_cl5 = sv5.cache[sv5.cache["close_time_ms"] <= int(time.time() * 1000)]
# 把游标退回到倒数第二根 -> 最后一根已收盘K线就成了"待处理的新K线"
sv5.t.last_bar_ms = int(_cl5.iloc[-2]["close_time_ms"])
sv5._step(logs5.append)
assert sv5.t.units > 0, "新K线收盘且信号为1, 却因为跳过历史把这次也跳过了"
assert any("信号买入" in m for m in logs5), logs5

# ================= 情形7: BTC 那条路径没被改坏 =================
assert "pair=None" in str(inspect.signature(ds.gate_candles))
assert "symbol=None" in str(inspect.signature(ds.binance_candles))
assert "symbol=None" in str(inspect.signature(ds.binance_ticker))
core = w.Core()
core.log = lambda m: None
_c, _a = core.compare(0.0)
assert abs(_a["ret"]) < 1e-9, f"没有自动仓位时, BTC 自动盘应当正好是起始资金: {_a}"
for k in ("equity", "ret", "n_trades", "win_rate", "units"):
    assert k in _c and k in _a, k

print("多品种子账户自检通过:")
print(f"  1 正常触发止损 @ {buy_px:,.2f} -> 亏 {loss1:.2%} (名义风险约1%)")
print(f"  2 跳空腰斩       @ {buy_px2:,.2f} -> 亏 {loss2:.2%} (止损不保证成交价)")
print(f"  3 落盘重载一致: 现金 {sv.t.cash:.4f}")
print(f"  4 stats: 净值 {st['equity']:.2f} | 已平仓 {st['n_trades']} 笔")
print(f"  5 首次启动不回溯历史(游标定在最后一根已收盘K线)")
print(f"  6 之后的新K线照常出手")
print(f"  7 BTC 主盘与数据源默认参数未受影响")

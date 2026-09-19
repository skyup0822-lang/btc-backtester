"""自检: 三类请求(价格/K线/成交明细)的往返耗时是否被正确测量并暴露给前端。

离线跑, 用假的耗时请求, 不联网。同时静态检查前端确实按 5 位小数显示。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import web_paper as w

core = w.Core()
core.src = "binance"
core.log = lambda m: None

# 1) 报价请求: 记录真实往返耗时
def slow_ticker():
    time.sleep(0.05)
    return {"last": 78000.0}


w.ds.binance_ticker = slow_ticker
core._load_ticker("binance")
assert 0.045 <= core.lat["price"] < 0.5, core.lat

# 2) K线请求: 同样记账
def slow_candles(n):
    time.sleep(0.08)
    import pandas as pd
    return pd.DataFrame([{"close_time_ms": 1, "close": 1.0, "volume": 1.0}])


w.ds.binance_candles = slow_candles
core._load_candles("binance", initial=False)
assert 0.075 <= core.lat["kline"] < 0.5, core.lat

# 3) 请求失败也要记账(try/finally), 否则界面上会一直停在旧数字
def boom(*a, **k):
    raise RuntimeError("offline")


w.ds.binance_ticker = boom
core.lat["price"] = 999.0  # 哨兵: 只要被刷新过就不再是它
try:
    core._load_ticker("binance")
except RuntimeError:
    pass
assert core.lat["price"] != 999.0, "失败的请求也必须刷新耗时"

# 4) 必须经由 /api/live 暴露给前端
snap = core.live_snapshot()
for k in ("lat_price", "lat_kline", "lat_trades"):
    assert k in snap, f"{k} 没暴露给前端"
    assert isinstance(snap[k], float), (k, type(snap[k]))

# 5) 前端按 5 位小数显示
html = w.HTML
assert "toFixed(5)" in html, "前端没有 5 位小数格式化"
assert "setLat($('lag_px')" in html and "setLat($('lag_candle')" in html and "setLat($('lag_tr')" in html
assert "setLag(" not in html, "旧的 setLag 还有残留调用"

# 5b) 每个 $('id') 引用的元素必须真的存在 —— 这条能挡住"id 写错导致某个框永远显示 -"
import re
ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
refs = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", html))
missing = sorted(refs - ids)
assert not missing, f"JS 引用了不存在的元素 id: {missing}"
for el in ("lag_px", "lag_candle", "lag_tr", "lag_clock"):
    assert el in refs, f"{el} 没被任何 JS 更新, 会一直显示 -"

# 5c) 反向验证: 故意把 id 写错, 上面这条检查必须抓得到(否则等于摆设)
_bad = html.replace("setLat($('lag_candle')", "setLat($('lag_kline')")
_bad_missing = sorted(set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", _bad))
                      - set(re.findall(r'id="([A-Za-z0-9_]+)"', _bad)))
assert _bad_missing == ["lag_kline"], f"id 检查抓不到写错的引用: {_bad_missing}"

print("延迟监测自检通过:")
print(f"  价格往返   {core.lat['price']:.5f}秒")
print(f"  K线往返    {core.lat['kline']:.5f}秒")
print(f"  成交明细   {core.lat['trades']:.5f}秒 (未跑, 由主循环每4秒刷新)")
print("  失败请求也记账 / 经 /api/live 暴露 / 前端 5 位小数")

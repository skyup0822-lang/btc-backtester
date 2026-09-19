# -*- coding: utf-8 -*-
"""自检: OKX 分页是否已越过 /market/candles 的最近1440根上限。

离线跑(不需要网络), 只检查 fetch_candles 的分页逻辑:
第一页必须是 /market/candles(带未收盘K线), 之后必须切到 /market/history-candles。
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paper_trader_okx as m

src = inspect.getsource(m.fetch_candles)
paths = [l.strip() for l in src.splitlines() if l.strip().startswith("path = ")]

assert len(paths) == 2, f"path 赋值应恰好两处, 实际 {paths}"
assert "/api/v5/market/candles" in paths[0], paths[0]
assert "/api/v5/market/history-candles" in paths[1], paths[1]

# 分页循环必须真的会翻页(有 after 游标), 否则2000根永远拿不满
assert 'q["after"] = str(after)' in src
assert "after = int(d[-1][0])" in src

print("OKX 分页自检通过:")
print("  第一页 ->", paths[0])
print("  后续页 ->", paths[1])
sys.exit(0)

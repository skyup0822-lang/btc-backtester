# -*- coding: utf-8 -*-
"""看一眼历史基准率算出来的数, 顺便当自检。

离线跑。除了打印, 还断言几条硬规则:
  - 每个格子的样本数 > 0
  - 样本门槛以下的形态不给结论
  - 中位数不能离谱到超过 +/-50%(真出现了说明代码有问题)
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import web_paper as w

t0 = time.perf_counter()
h = w._history_stats()
dt = time.perf_counter() - t0
assert h, "历史统计算不出来"
print(f"耗时 {dt:.2f}s | 区间 {h['range'][0]} ~ {h['range'][1]} | {h['n_bars']:,} 根1小时K线")
print()

print("基准(任意时点买入):")
for k, v in sorted(h["base"].items(), key=lambda x: int(x[0])):
    print(f"  未来{k:>3}h   中位 {v['median']*100:+.3f}%   上涨 {v['up']*100:5.1f}%   样本 {v['n']:,}")
print("策略金叉触发后:")
for k, v in sorted(h["trigger"].items(), key=lambda x: int(x[0])):
    print(f"  未来{k:>3}h   中位 {v['median']*100:+.3f}%   上涨 {v['up']*100:5.1f}%   样本 {v['n']:,}")
print("各形态(过样本门槛的):")
for f, d in h["flags"].items():
    v = d.get("24")
    if v:
        print(f"  {w.PAT_CN[f]:<10} 24h 中位 {v['median']*100:+.3f}%   上涨 {v['up']*100:5.1f}%   样本 {v['n']:,}")

# --- 硬规则 ---
for grp in ("base", "trigger"):
    assert set(h[grp]) == {"24", "72", "168"}, (grp, list(h[grp]))
    for k, v in h[grp].items():
        assert v["n"] > 0, (grp, k)
        assert 0.0 <= v["up"] <= 1.0, (grp, k, v["up"])
        assert abs(v["median"]) < 0.5, (grp, k, v["median"])
assert h["trigger"]["24"]["n"] < h["base"]["24"]["n"], "触发次数必须少于总样本"
for f, d in h["flags"].items():
    for k, v in d.items():
        assert v["n"] >= w.ADVICE_MIN_N, f"{f} {k} 样本 {v['n']} 低于门槛却给了结论"

# 形态筛选确实起过滤作用: 拉盘/盘整这类常见形态样本应远少于基准
if "range" in h["flags"]:
    assert h["flags"]["range"]["24"]["n"] < h["base"]["24"]["n"] * 0.2

print()
print(f"自检通过: 门槛 {w.ADVICE_MIN_N} 次, 触发样本 {h['trigger']['24']['n']:,} 条")

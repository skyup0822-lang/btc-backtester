# -*- coding: utf-8 -*-
"""自检: 数据源优先级与降级链。

离线跑(不联网), 只校验顺序和"每个源都有取数/报价函数"这个契约。
改 SRC_ORDER 后跑一次, 顺序错了会直接报错。
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_source as ds
import web_paper as w

# 1) 优先级顺序: 用户要求币安第一
assert w.SRC_ORDER[0] == "binance", f"币安必须是首位, 实际 {w.SRC_ORDER}"
assert w.SRC_ORDER == ("binance", "okx", "gate"), w.SRC_ORDER

# 2) 每个源都要有标签, 且探测顺序就是 SRC_ORDER
for name in w.SRC_ORDER:
    assert name in w.SRC_LABEL, f"{name} 缺标签"
assert "for name in SRC_ORDER" in inspect.getsource(w.Core._probe_source)

# 3) 降级链必须覆盖到 SRC_ORDER 的每一环, 并且标签查得到(漏一个就是运行时 KeyError)
src = inspect.getsource(w.Core._run)
m = re.search(r"_next = (\{[^}]*\})", src)
assert m, "找不到降级链"
chain = eval(m.group(1))
seen = set()
cur = w.SRC_ORDER[0]
while cur in chain:
    seen.add(cur)
    cur = chain[cur]
seen.add(cur)
assert seen == set(w.SRC_ORDER), f"降级链 {chain} 走不到 {set(w.SRC_ORDER) - seen}"
for target in chain.values():
    assert target in w.SRC_LABEL, f"降级目标 {target} 没有标签 -> 运行时会 KeyError"

# 4) 每个源都要有真实的取数/报价实现(不能只在链里挂个名字)
assert callable(ds.binance_candles) and callable(ds.binance_ticker)
assert callable(ds.gate_candles) and callable(ds.gate_ticker)
assert callable(w.okx_fetch_candles)

# 5) 只有 OKX 支持密钥真实挂单, 币安/ Gate 是纯行情
run_src = inspect.getsource(w.Core._run)
assert 'real_orders = has_keys and src == "okx"' in run_src

print("数据源自检通过:")
print("  优先级  :", " -> ".join(w.SRC_ORDER))
cur, path = w.SRC_ORDER[0], [w.SRC_ORDER[0]]
while cur in chain:
    cur = chain[cur]
    path.append(cur)
print("  降级链  :", " -> ".join(path))
print("  真实挂单: 仅 OKX(需填密钥), 其余源本地虚拟成交")
sys.exit(0)

"""自检: 三源交叉校验 _verify_bar 的判定逻辑(离线, 用假的源, 不联网)。

覆盖四种情况:
  1. 三源一致            -> 放行
  2. 主源插针, 别的源正常 -> 拦下(这根K线不可信)
  3. 只有一个源能用       -> 放行(没法验证, 但要记日志)
  4. 某个源取数老失败     -> 冷却, 不再反复重试拖慢主循环
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import web_paper as w

CT = 1789034400000  # 任意一根已收盘K线


def mk(close):
    return pd.DataFrame([{"close_time_ms": CT, "close": close, "volume": 100.0}])


def new_core():
    core = w.Core()
    core.log = lambda m: None
    core._record_decision = lambda *a, **k: None
    return core


logs = []
core = new_core()
core.log = logs.append

# 1) 三源一致 -> 放行
core._load_candles = lambda name, initial: mk({"binance": 78000.0, "okx": 78010.0, "gate": 78005.0}[name])
assert core._verify_bar("binance", {"close_time_ms": CT, "close": 78000.0}) is True, "一致时应放行"
assert any("通过" in m for m in logs), logs

# 2) 主源插针(99999), 另外两个源都是 78000 附近 -> 必须拦下
logs.clear()
core._load_candles = lambda name, initial: mk({"binance": 99999.0, "okx": 78010.0, "gate": 78005.0}[name])
assert core._verify_bar("binance", {"close_time_ms": CT, "close": 99999.0}) is False, "插针必须被拦下"
assert any("警告" in m and "假K线" in m for m in logs), logs

# 3) 只有一个源能用 -> 放行, 但不能假装验证过
logs.clear()
core._load_candles = lambda name, initial: (_ for _ in ()).throw(RuntimeError("offline"))
assert core._verify_bar("binance", {"close_time_ms": CT, "close": 78000.0}) is True, "单源应放行"
assert any("只有单一数据源" in m for m in logs), logs

# 4) 冷却: okx/gate 都失败过一次后, 第二次不应该再调它们
calls = []


def failing(name, initial):
    calls.append(name)
    raise RuntimeError("offline")


core2 = new_core()
core2._load_candles = failing
core2._verify_bar("binance", {"close_time_ms": CT, "close": 78000.0})
first = len(calls)
core2._verify_bar("binance", {"close_time_ms": CT, "close": 78000.0})
assert first == 2, f"第一次应尝试另外两个源, 实际 {calls}"
assert len(calls) == first, f"冷却期内不该再重试, 实际又调了 {calls[first:]}"
assert set(core2.src_cooldown) == {"okx", "gate"}, core2.src_cooldown

print("三源交叉校验自检通过:")
print("  1 三源一致        -> 放行")
print("  2 主源插针        -> 拦下本根信号决策")
print("  3 只有单源        -> 放行并记日志")
print("  4 取数失败        -> 冷却30分钟, 不反复重试")

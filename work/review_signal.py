# -*- coding: utf-8 -*-
"""复盘: 这几天为什么一单没开。

核心问题不是"亏没亏", 是"程序到底有没有在正常判断"。做法:
  1. 读控制台自己归档的 bars_history.csv(每根K线带它当时算出的 signal)
  2. 用回测同一套公式独立重算一遍, 跟归档值逐根对比
  3. 若两者一致 -> 程序没坏, 是真的没信号
  4. 若不一致 -> 程序有 bug

顺带输出: 这几天信号的分布、以及为什么没触发(均线谁在上面)。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SHORT, LONG, GATE = 48, 336, 1440

df = pd.read_csv(ROOT / "bars_history.csv")
print(f"归档文件原始行数: {len(df):,}")
# 每次重启都会把整段缓存重新写一遍, 会有大量重复
df = df.drop_duplicates("open_time_ms", keep="last").sort_values("open_time_ms").reset_index(drop=True)
print(f"去重后唯一K线: {len(df):,}")

df["t"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
print(f"覆盖区间: {df['t'].iloc[0]} ~ {df['t'].iloc[-1]}")
print()

# --- 用回测同一套公式重算 ---
close = df["close"]
sma_s = close.rolling(SHORT).mean()
sma_l = close.rolling(LONG).mean()
sma_g = close.rolling(GATE).mean()
recomputed = ((sma_s > sma_l) & (close > sma_g)).astype(float).fillna(0.0)

archived = pd.to_numeric(df["signal"], errors="coerce")  # 空=当时数据不够, 不算数
diff = (recomputed != archived) & archived.notna()
print("=== 程序算的 signal vs 独立重算 ===")
print(f"  有归档信号的K线: {int(archived.notna().sum())} / {len(df)}"
      f" (其余 {int(archived.isna().sum())} 根当时历史不足, 留空)")
print(f"  不一致的K线: {int(diff.sum())}")
if diff.sum():
    bad = df[diff].tail(10)[["t", "close", "signal"]].copy()
    bad["recomputed"] = recomputed[diff].tail(10).values
    print(bad.to_string(index=False))
else:
    print("  完全一致 -> 程序的信号计算没有问题")
print()

# --- 这几天信号到底是什么 ---
since = df["t"] > pd.Timestamp("2026-09-10 12:00", tz="UTC")
recent = df[since]
print(f"=== 模拟盘重置之后({recent['t'].iloc[0]} 起, {len(recent)} 根K线) ===")
print(f"  信号=1 的K线: {int((recent['signal'] > 0).sum())}")
print(f"  信号=0 的K线: {int((recent['signal'] == 0).sum())}")
print()

# --- 为什么没触发: 两个条件分别看 ---
gate_ok = (close > sma_g)
cross_ok = (sma_s > sma_l)
r = pd.DataFrame({"t": df["t"], "close": close, "sma48": sma_s, "sma336": sma_l,
                  "sma1440": sma_g, "闸门(收盘>1440)": gate_ok, "金叉(48>336)": cross_ok})
rr = r[since]
print("=== 两个条件各自满足了多少根 ===")
print(f"  闸门 收盘>SMA1440 满足: {int(rr['闸门(收盘>1440)'].sum())} / {len(rr)}")
print(f"  金叉 SMA48>SMA336 满足: {int(rr['金叉(48>336)'].sum())} / {len(rr)}")
print()
print("=== 最近 12 根 ===")
tail = rr.tail(12).copy()
tail["t"] = tail["t"].dt.strftime("%m-%d %H:%M")
tail["距离1440"] = (tail["close"] / tail["sma1440"] - 1).map(lambda v: f"{v:+.2%}")
tail["48-336差"] = (tail["sma48"] / tail["sma336"] - 1).map(lambda v: f"{v:+.2%}")
for c in ("close", "sma48", "sma336", "sma1440"):
    tail[c] = tail[c].map(lambda v: f"{v:,.0f}")
print(tail[["t", "close", "sma48", "sma336", "sma1440", "距离1440", "48-336差",
            "闸门(收盘>1440)", "金叉(48>336)"]].to_string(index=False))

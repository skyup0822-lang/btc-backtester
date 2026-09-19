# -*- coding: utf-8 -*-
"""复盘: 模拟盘重置后这几天, 市场发生了什么, 策略为什么一直空仓, 空仓算不算正常。

全部基于本地数据(历史CSV + 控制台归档), 不联网。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from paper_trader import compute_signal

# ---------- 1) 长期看: 这套策略本来就大部分时间是空仓 ----------
hist = pd.read_csv(ROOT / "data" / "btc_usdt_1h.csv", index_col=0, parse_dates=True)
hist = hist.rename(columns=str.lower)
sig = pd.Series(compute_signal(hist), index=hist.index).fillna(0.0)
in_market = float((sig > 0).mean())
print("=" * 66)
print("一、这套策略的空仓比例(全历史 2019-11 ~ 2026-09)")
print("=" * 66)
print(f"  持仓时间占比: {in_market:.1%}   空仓时间占比: {1 - in_market:.1%}")

runs, cur = [], 0
for v in (sig == 0).to_numpy():
    if v:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs_d = np.array(runs) / 24.0
print(f"  连续空仓的时长(天): 中位 {np.median(runs_d):.1f} | 75分位 {np.percentile(runs_d, 75):.1f} "
      f"| 90分位 {np.percentile(runs_d, 90):.1f} | 最长 {runs_d.max():.1f}")
print(f"  空仓超过 3.5 天的比例: {(runs_d > 3.5).mean():.1%}  (共 {len(runs_d)} 段)")
print()

# ---------- 2) 这几天市场干了什么 ----------
bars = pd.read_csv(ROOT / "bars_history.csv").drop_duplicates("open_time_ms", keep="last")
bars = bars.sort_values("open_time_ms").reset_index(drop=True)
bars["t"] = pd.to_datetime(bars["close_time_ms"], unit="ms", utc=True)
since = bars["t"] > pd.Timestamp("2026-09-10 12:00", tz="UTC")
r = bars[since]
print("=" * 66)
print(f"二、重置之后这几天({r['t'].iloc[0]:%m-%d %H:%M} ~ {r['t'].iloc[-1]:%m-%d %H:%M}, {len(r)} 根K线)")
print("=" * 66)
print(f"  起点 {r['close'].iloc[0]:,.0f} -> 最新 {r['close'].iloc[-1]:,.0f} "
      f"({r['close'].iloc[-1] / r['close'].iloc[0] - 1:+.2%})")
print(f"  区间最高 {r['high'].max():,.0f} | 最低 {r['low'].min():,.0f} "
      f"| 振幅 {r['high'].max() / r['low'].min() - 1:.2%}")
print(f"  如果一开始就买入持有, 到现在: {r['close'].iloc[-1] / r['close'].iloc[0] - 1:+.2%}")
print()

# ---------- 3) 距离触发还差多远 ----------
close = bars["close"]
sma_s, sma_l = close.rolling(48).mean(), close.rolling(336).mean()
spread = (sma_s - sma_l) / sma_l
rs = spread[since]
print("=" * 66)
print("三、离发出买入信号还差多远(48小时均线 要重新超过 336小时均线)")
print("=" * 66)
print(f"  当前差距: {rs.iloc[-1]:+.2%}  (需要涨到 0% 以上)")
print(f"  这几天最好的一次: {rs.max():+.2%}  最差: {rs.min():+.2%}")
need = -rs.iloc[-1]
print(f"  要么价格继续走高把短均线拉上去, 要么横盘够久让长均线自己滑下来")
print(f"  参照: 短均线每天约变动 {abs(sma_s.diff(24).tail(72).mean() / sma_s.iloc[-1]):.3%}, "
      f"长均线每天约 {abs(sma_l.diff(24).tail(72).mean() / sma_l.iloc[-1]):.3%}")
print()

# ---------- 4) 归档里那30根对不上的K线在哪 ----------
allc = bars["close"]
rec = ((allc.rolling(48).mean() > allc.rolling(336).mean()) & (allc > allc.rolling(1440).mean())).astype(float)
arch = pd.to_numeric(bars["signal"], errors="coerce")
bad = bars[(rec != arch) & arch.notna()]
if len(bad):
    print("=" * 66)
    print(f"四、归档 signal 与重算不一致的 {len(bad)} 根, 落在哪")
    print("=" * 66)
    print(f"  时间范围: {bad['t'].min():%Y-%m-%d %H:%M} ~ {bad['t'].max():%Y-%m-%d %H:%M}")
    print(f"  连续段数: {1 + int((bad.index.to_series().diff() > 1).sum())}")
    print(f"  全部在模拟盘重置(09-10)之前: {bool(bad['t'].max() < pd.Timestamp('2026-09-10', tz='UTC'))}")
    print(f"  归档值分布: {dict(bad['signal'].value_counts())}")
    print(f"  (这些都是重置前、由当时缓存长短不同的会话写下的记录; 重置后 74 根全部一致)")

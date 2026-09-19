# -*- coding: utf-8 -*-
"""整理 bars_history.csv: 去重 + 把"缓存不足导致的假信号0"清成空。

为什么会有一堆重复: 控制台的 _arch_ms 以前每次重启都从 0 开始, 于是每次重启都把
整段 2000 根缓存重新追加一遍。62152 行里只有 2176 根是唯一的。

为什么有的行 signal 是假的 0: sig 是在 2000 根缓存上算的, 前 1440 根还没有 SMA1440,
闸门必然为假 -> 信号记成 0。那不代表"当时判断为空仓", 只代表"当时数据不够"。

处理: 同一个 bar 保留最后一次写入的值; 保留的那份如果落在"数据不够"的位置, 就清空。
原文件先备份。
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

SRC = ROOT / "bars_history.csv"
BAK = ROOT / f"bars_history.csv.dup-bak-{time.strftime('%Y%m%d-%H%M%S')}"

raw = pd.read_csv(SRC)
before = len(raw)
GATE_BARS = 1440

shutil.copy2(SRC, BAK)
print(f"原文件 {before:,} 行 -> 备份到 {BAK.name}")

df = raw.drop_duplicates("open_time_ms", keep="last").sort_values("open_time_ms").reset_index(drop=True)
print(f"去重后 {len(df):,} 行 (删掉 {before - len(df):,} 行重复)")

# 用完整的K线序列重算闸门, 判断每一根的 signal 是不是可信的
df["t"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
sma_g = df["close"].rolling(GATE_BARS).mean()
valid = sma_g.notna()  # 有60日线才算得出来
df["signal"] = df["signal"].astype(object)  # 要先转 object, 否则塞不进空字符串
n_bad = int((~valid).sum())
df.loc[~valid, "signal"] = ""
print(f"其中 {n_bad:,} 根在开头(没有60日线), signal 已清空")

out = df.drop(columns=["t"])
out["signal"] = out["signal"].apply(lambda v: "" if v == "" else f"{float(v):.1f}")
out.to_csv(SRC, index=False)
print(f"已写回: {SRC.name}, {len(out):,} 行")

sig_vals = out["signal"].astype(str)
print(f"  有信号的K线: {int(sig_vals.isin(['0.0', '1.0']).sum()):,}"
      f" | 其中 signal=1: {int((sig_vals == '1.0').sum()):,}"
      f" | 空(数据不够): {int((sig_vals == '').sum()):,}")

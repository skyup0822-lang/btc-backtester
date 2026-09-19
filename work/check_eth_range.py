# -*- coding: utf-8 -*-
"""核对 BTC/ETH 两个数据文件的样本外区间是否可比(不然组合那行不成立)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

START = pd.Timestamp("2024-12-31", tz="UTC")
for f in ("btc_usdt_1h.csv", "eth_usdt_1h.csv"):
    d = pd.read_csv(ROOT / "data" / f, index_col=0, parse_dates=True)
    t = d[d.index > START]
    chg = t["Close"].iloc[-1] / t["Close"].iloc[0] - 1
    print(f"{f:<18} 全区间 {d.index[0].date()} ~ {d.index[-1].date()} | 样本外 {len(t)} 根")
    print(f"{'':<18} 样本外涨跌 {chg:+.2%} | 最高 {t['High'].max():,.0f} 最低 {t['Low'].min():,.0f}")

b = pd.read_csv(ROOT / "data" / "btc_usdt_1h.csv", index_col=0, parse_dates=True)
e = pd.read_csv(ROOT / "data" / "eth_usdt_1h.csv", index_col=0, parse_dates=True)
inter = b.index.intersection(e.index)
inter = inter[inter > START]
print(f"\n两个文件在样本外共同覆盖的K线: {len(inter)}")
print(f"ETH 缺失的 BTC 时点: {len(b[b.index > START].index.difference(e.index))}")

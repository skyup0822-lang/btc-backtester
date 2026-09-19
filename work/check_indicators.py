"""自检: 建议卡的指标必须只用"已收盘"的K线。

背景: 缓存最后一行是本小时还在走的K线, 才走了几分钟, 用它算 ATR 和成交量会严重偏低
(看着像死水行情, 其实只是没攒够数据)。这里造一根极端的"未收盘"K线塞到缓存末尾,
如果指标被它影响, 断言就会炸。

离线跑, 用本地历史CSV, 不联网。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import web_paper as w

raw = pd.read_csv(ROOT / "data" / "btc_usdt_1h.csv", index_col=0, parse_dates=True)
df = raw.rename(columns=str.lower).reset_index(drop=True)
df["open_time_ms"] = raw.index.astype("int64") // 10 ** 6
df["close_time_ms"] = df["open_time_ms"] + 3_600_000

# 造一根"正在走"的K线: 收盘时间在未来 -> 不该被算进指标
now_ms = int(time.time() * 1000)
last = df.iloc[-1]
fake = {
    "open_time_ms": now_ms - 1_800_000, "close_time_ms": now_ms + 1_800_000,
    "open": last["close"] * 1.5, "high": last["close"] * 1.5001,
    "low": last["close"] * 1.4999, "close": last["close"] * 1.5, "volume": 0.0001,
}
cache = pd.concat([df, pd.DataFrame([fake])], ignore_index=True)

core = w.Core()
core.log = lambda m: None
core.cache = cache
ind = core._indicators()
assert ind is not None, "指标算不出来"

# 1) 收盘价必须取"最后一根已收盘"的, 不是那根 1.5 倍的假K线
assert abs(ind["last"] - last["close"]) < 1e-9, \
    f"用了未收盘K线: 拿到 {ind['last']}, 应该是 {last['close']}"

# 2) ATR / 量比必须与"只用已收盘K线"独立算出来的一致
close = df["close"]
prev = close.shift(1)
tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                (df["low"] - prev).abs()], axis=1).max(axis=1)
exp_atr = float(tr.rolling(14).mean().iloc[-1]) / float(close.iloc[-1])
exp_vr = float(df["volume"].iloc[-1]) / float(df["volume"].tail(w.VOL_WINDOW).median())
assert abs(ind["atr_pct"] - exp_atr) < 1e-12, (ind["atr_pct"], exp_atr)
assert abs(ind["vol_ratio"] - exp_vr) < 1e-12, (ind["vol_ratio"], exp_vr)

# 3) 72小时高低点也不能被那根假K线拉高
assert abs(ind["high72"] - float(df["high"].tail(72).max())) < 1e-9, ind["high72"]
assert ind["high72"] < last["close"] * 1.5, "72小时高点被未收盘K线污染了"

# 4) 顺手看一眼当前 ATR 在历史上处于什么水平, 判断"低"是不是真的低
atr_hist = (tr / close).tail(5000)
pct_rank = float((atr_hist < exp_atr).mean())
print("建议卡指标自检通过:")
print(f"  已收盘K线 {pd.to_datetime(df['open_time_ms'].iloc[-1], unit='ms', utc=True)}")
print(f"  收盘 {ind['last']:.2f} | ATR14 {ind['atr_pct']*100:.3f}% | 量比 {ind['vol_ratio']:.2f}")
print(f"  未收盘的那根假K线(1.5倍价/几乎没有量)已被完全排除")
print(f"  参考: 最近5000根的 ATR 中位 {atr_hist.median()*100:.3f}%, "
      f"当前值处于第 {pct_rank*100:.1f} 百分位")

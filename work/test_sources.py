# -*- coding: utf-8 -*-
"""行情源可达性测试: 关闭VPN后跑一次, 看哪些数据源能用。

只请求公开接口, 不需要密钥。用法: python work/test_sources.py
"""
import time
import urllib.request

TARGETS = [
    ("OKX (实时行情)", "https://www.okx.com/api/v5/public/time"),
    ("Gate.io (实时行情)", "https://api.gateio.ws/api/v4/spot/tickers?currency_pair=BTC_USDT"),
    ("Binance api (实时)", "https://api.binance.com/api/v3/time"),
    ("Binance 存档(data.binance.vision)", "https://data.binance.vision/"),
]

print("=" * 70)
print("行情源可达性测试 (当前网络)")
print("=" * 70)
ok = []
for name, url in TARGETS:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read(200)
        dt = time.time() - t0
        print(f"  [通]  {name:<34} {dt:.2f}s")
        ok.append(name)
    except Exception as e:
        print(f"  [断]  {name:<34} {type(e).__name__}: {str(e)[:60]}")

# 真正走一遍控制台用的取数函数(不只是ping网址), 确认K线列和根数没问题
print("-" * 70)
print("实际取数自检(控制台走的就是这几个函数)")
print("-" * 70)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import data_source as ds
except Exception as e:
    print(f"  导入 data_source 失败: {e}")
    raise SystemExit(1)

good = []
for label, fn in (("OKX", None), ("Gate.io", ds.gate_candles), ("Binance", ds.binance_candles)):
    t0 = time.time()
    try:
        if fn is None:
            from paper_trader_okx import fetch_candles
            fn = fetch_candles
        df = fn(2000)
        dt = time.time() - t0
        need = {"open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"}
        missing = need - set(df.columns)
        last = df.iloc[-1]
        print(f"  [{len(df)}根] {label:<9} {dt:.2f}s  最后一根收盘 {last['close']:.2f}"
              f"  close_time={int(last['close_time_ms'])}")
        if missing:
            print(f"          !! 缺少列: {missing}")
        elif len(df) < 1441:
            print(f"          !! 不足1440根, 信号算不出来")
        else:
            good.append(label)
    except Exception as e:
        print(f"  [取数失败] {label:<9} {type(e).__name__}: {str(e)[:70]}")

print("=" * 70)
print("可用的源:", ", ".join(ok) if ok else "(全都不通)")
print("能直接喂给控制台的源:", ", ".join(good) if good else "(全都不行)")
print("提示: 控制台依次试 OKX -> Gate.io -> Binance, 用第一个连得上的。")

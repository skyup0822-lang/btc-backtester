# -*- coding: utf-8 -*-
"""多周期对比回测: 实盘策略(48/336/1440 趋势系统 + 15%止损 + 每笔1%风险)
分别在 5m/15m/30m/1h/4h 上回测, 同一时间段(2024-06-01起), 同样手续费/滑点。

方案A 同参数: 48/336/1440 根, 即"面板上只把周期换掉"会发生的事。
方案B 时间等比: 窗口按小时数等比缩放(48h/336h/1440h 的经济含义一致)。
数据源: data.binance.vision 官方存档(免密钥, 月度zip+当月按日zip)。
输出: results/tf_sweep.csv 与 results/tf_sweep_curves.csv(净值曲线)
"""
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics, compute_metrics
from config import DEFAULT_CONFIG

INTERVALS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}
BARS_PER_HOUR = {k: 3600.0 / v for k, v in INTERVALS.items()}
WINDOW_START = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp())
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
BINANCE_COLUMNS = ["open_time", "Open", "High", "Low", "Close", "Volume",
                   "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def _get_bytes(url, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def _open_time_ms(series):
    vals = series.to_numpy(dtype="int64")
    m = int(np.abs(vals).max())
    if m > 1e17:
        return vals // 1_000_000
    if m > 1e14:
        return vals // 1_000
    return vals


def binance_vision_candles(interval: str, start_ts: int) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    start_month = pd.Timestamp(start_ts, unit="s", tz="UTC").replace(day=1)
    month_first = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    months = pd.date_range(start_month, month_first, freq="MS")
    frames = []
    monthly_base = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/{interval}/"
    for m in months:
        name = f"BTCUSDT-{interval}-{m:%Y-%m}.zip"
        try:
            raw = _get_bytes(monthly_base + name)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                fr = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), header=None, names=BINANCE_COLUMNS)
            fr["open_time"] = _open_time_ms(fr["open_time"])
            frames.append(fr)
            print(f"    {m:%Y-%m} ok ({len(fr)}根)", flush=True)
        except Exception:
            print(f"    {m:%Y-%m} 缺失, 跳过", flush=True)
        time.sleep(0.05)
    daily_base = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/{interval}/"
    for d in pd.date_range(month_first, now.normalize(), freq="D"):
        name = f"BTCUSDT-{interval}-{d:%Y-%m-%d}.zip"
        try:
            raw = _get_bytes(daily_base + name)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                fr = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), header=None, names=BINANCE_COLUMNS)
            fr["open_time"] = _open_time_ms(fr["open_time"])
            frames.append(fr)
        except Exception:
            pass
        time.sleep(0.03)
    if not frames:
        raise RuntimeError("Binance Vision 未获取到数据")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["open_time"] >= start_ts * 1000]
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    step = INTERVALS[interval]
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]].rename(columns={"open_time": "open_time_ms"})
    df["close_time_ms"] = df["open_time_ms"] + step * 1000
    return df


def signal_sma_gate(df, short, long, gate):
    close = df["Close"]
    fast = close.rolling(short).mean()
    slow = close.rolling(long).mean()
    gate_s = close > close.rolling(gate).mean()
    return ((fast > slow) & gate_s).astype(float).fillna(0.0)


def run(interval, df, sig_series, pppy):
    sig = sig_series.reindex(df.index).fillna(0.0)
    eq_arr, trades, _ = run_backtest(df, sig, DEFAULT_CONFIG, stop_pct=0.15, risk_pct=0.01)
    equity = pd.Series(eq_arr, index=df.index)
    m = compute_metrics(equity, trades, pppy)
    bench, _ = buy_hold_metrics(df, DEFAULT_CONFIG)
    m["benchmark_return"] = bench["total_return"]
    m["avg_hold_hours"] = (m.get("avg_holding_bars") or 0) * INTERVALS[interval] / 3600
    m["final_equity"] = equity.iloc[-1]
    return m, equity


def main():
    RES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    results, curves = [], {}
    for interval, step in INTERVALS.items():
        bph = BARS_PER_HOUR[interval]
        warmup_bars = max(1440, 1440 * bph)
        start_ts = WINDOW_START - warmup_bars * step
        csv_path = DATA_DIR / f"tf_{interval}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"[{interval}] 用本地缓存 {len(df)} 根", flush=True)
        else:
            print(f"[{interval}] 下载 Binance 存档 K线(起始 {pd.to_datetime(start_ts, unit='s', utc=True):%Y-%m-%d}) ...", flush=True)
            df = binance_vision_candles(interval, start_ts)
            df.to_csv(csv_path, index=False)
            print(f"[{interval}] 下载完成, 共 {len(df)} 根", flush=True)
        if "Close" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.set_index(pd.to_datetime(df["open_time_ms"], unit="ms", utc=True))
        win = df[df.index >= pd.Timestamp(WINDOW_START, unit="s", tz="UTC")]
        pppy = 365 * 24 * bph
        variants = {
            "A_同参数(48/336/1440根)": (48, 336, 1440),
            "B_时间等比(48h/336h/1440h)": (max(4, int(48 * bph)), int(336 * bph), int(1440 * bph)),
        }
        for vname, (s_, l_, g_) in variants.items():
            sig = signal_sma_gate(df, s_, l_, g_)
            try:
                m, equity = run(interval, win, sig, pppy)
            except Exception as e:
                print(f"[{interval}][{vname}] 回测失败: {e}", flush=True)
                continue
            results.append({"interval": interval, "variant": vname, "n_bars": len(win), **m})
            curves[f"{interval}_{vname}"] = equity
            print(f"[{interval}][{vname}] 净收益 {m['total_return']:+.2%} | 回撤 {m['max_drawdown']:.2%} | "
                  f"交易 {m['n_trades']} | 胜率 {m['win_rate']:.1%} | 单笔期望 {m['expectancy_pct']:+.3%}", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(RES_DIR / "tf_sweep.csv", index=False, encoding="utf-8-sig")
    if curves:
        pd.DataFrame(curves).to_csv(RES_DIR / "tf_sweep_curves.csv", encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("多周期对比成绩单(2024-06-01 至今, 手续费0.1%/边+滑点0.05%/边, 每笔风险1%, 止损15%)")
    print("=" * 100)
    cols = {
        "interval": "周期", "variant": "方案", "n_bars": "K线数",
        "total_return": "净收益", "cagr": "年化", "max_drawdown": "最大回撤",
        "n_trades": "交易次数", "win_rate": "胜率", "profit_factor": "盈亏比",
        "expectancy_pct": "单笔期望", "avg_hold_hours": "平均持仓h",
        "benchmark_return": "买入持有", "final_equity": "期末权益",
    }
    show = res.rename(columns=cols)[list(cols.values())].copy()
    for c in ["净收益", "年化", "最大回撤", "胜率", "单笔期望", "买入持有"]:
        show[c] = show[c].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
    show["盈亏比"] = show["盈亏比"].apply(lambda v: f"{v:.2f}" if pd.notna(v) and np.isfinite(v) else "-")
    for c in ["平均持仓h", "期末权益"]:
        show[c] = show[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    show["交易次数"] = show["交易次数"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    show["K线数"] = show["K线数"].apply(lambda v: f"{v:.0f}")
    print(show.to_string(index=False))
    print("\n结果已保存: results/tf_sweep.csv, results/tf_sweep_curves.csv")


if __name__ == "__main__":
    main()

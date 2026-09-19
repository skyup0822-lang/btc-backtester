# -*- coding: utf-8 -*-
"""新旧引擎行为对比: 定位样本外收益从 +5.76% 变 +11.51% 的原因"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, BacktestConfig
from strategies.trend_system import sma_gate


def old_run_backtest(df, signals, config=BacktestConfig(), stop_pct=None, trail_pct=None):
    sig = signals.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    open_px = df["Open"].to_numpy(dtype=float)
    close_px = df["Close"].to_numpy(dtype=float)
    low_px = df["Low"].to_numpy(dtype=float) if "Low" in df else close_px
    index = df.index
    n = len(df)
    cash = config.initial_capital
    units = 0.0
    equity = np.empty(n)
    trades = []
    prev_target = 0.0
    entry_time = None
    entry_bar = 0
    entry_cost = 0.0
    stop_price = np.inf
    peak_close = 0.0
    for i in range(n):
        target = sig[i - 1] if i > 0 else 0.0
        stop_exited = False
        if units > 0:
            if trail_pct is not None:
                if close_px[i - 1] > peak_close:
                    peak_close = close_px[i - 1]
                trail_level = peak_close * (1 - trail_pct)
                if trail_level > stop_price:
                    stop_price = trail_level
            if np.isfinite(stop_price) and low_px[i] <= stop_price:
                exec_price = min(open_px[i], stop_price)
                cash = units * exec_price * (1 - config.fee_rate)
                trades.append((index[i], cash - entry_cost, "stop"))
                units = 0.0
                stop_price = np.inf
                stop_exited = True
        if not stop_exited and target != prev_target:
            px = open_px[i]
            if target > prev_target:
                exec_price = px * (1 + config.slippage)
                units = (cash / exec_price) * (1 - config.fee_rate)
                cash = 0.0
                entry_time = index[i]
                entry_bar = i
                entry_cost = units * exec_price
                if stop_pct is not None:
                    stop_price = exec_price * (1 - stop_pct)
                elif trail_pct is not None:
                    stop_price = exec_price * (1 - trail_pct)
                else:
                    stop_price = np.inf
                peak_close = exec_price if trail_pct is not None else 0.0
            else:
                exec_price = px * (1 - config.slippage)
                cash = units * exec_price * (1 - config.fee_rate)
                trades.append((index[i], cash - entry_cost, "signal"))
                units = 0.0
                stop_price = np.inf
            prev_target = target
        elif stop_exited:
            prev_target = 0.0
        equity[i] = cash + units * close_px[i]
    return pd.Series(equity, index=index), trades


from backtest.engine import run_backtest as new_run_backtest

df = pd.read_csv(ROOT / "data" / "btc_usdt_1h.csv", index_col=0, parse_dates=True)
df = df[~df.index.duplicated(keep="first")].sort_index()
signals = sma_gate(df, short=48, long=336, gate=1440)
test = df[df.index > pd.Timestamp("2024-12-31", tz="UTC")]

eq_old, tr_old = old_run_backtest(test, signals.reindex(test.index), DEFAULT_CONFIG, stop_pct=0.15)
eq_new, tr_new, _ = new_run_backtest(test, signals.reindex(test.index), DEFAULT_CONFIG, stop_pct=0.15)

print(f"old: n={len(tr_old)} return={eq_old.iloc[-1]/eq_old.iloc[0]-1:+.4%}")
print(f"new: n={len(tr_new)} return={eq_new.iloc[-1]/eq_new.iloc[0]-1:+.4%}")
print("first 6 old:", [(t[0].strftime('%m-%d %H:%M'), f'{t[1]:+.1f}', t[2]) for t in tr_old[:6]])
print("first 6 new:", [(t.entry_time.strftime('%m-%d %H:%M'), f'{t.pnl:+.1f}', t.exit_reason) for t in tr_new[:6]])
d = (eq_old - eq_new).abs()
print("max |eq_old - eq_new| =", d.max())
idx = d.idxmax()
print("at", idx, "old=", eq_old.loc[idx], "new=", eq_new.loc[idx], "ratio=", eq_new.loc[idx]/eq_old.loc[idx])
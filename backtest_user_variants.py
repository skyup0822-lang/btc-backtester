# -*- coding: utf-8 -*-
"""变体实验: 固定 14天低点+15%突破买入 / -10%止损, 只改变止盈回撤宽度。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(r'C:\Users\Administrator\Documents\Codex\2026-09-09\x20\btc-backtester')
FEE, SLIP = 0.001, 0.0005
TRAIN_END = pd.Timestamp('2024-12-31')
df = pd.read_csv(ROOT / 'data' / 'btc_usdt_1h.csv', index_col=0, parse_dates=True)
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
if TRAIN_END.tz is None:
    TRAIN_END = TRAIN_END.tz_localize(df.index.tz)

def run(sub, N=336, stop_pct=0.10, trail_act=0.15, trail_pct=0.10, trail_from_start=False):
    close = sub['Close'].to_numpy(dtype=float)
    open_ = sub['Open'].to_numpy(dtype=float)
    low = sub['Low'].to_numpy(dtype=float)
    roll = pd.Series(close, index=sub.index).rolling(N, min_periods=N).min().to_numpy()
    n = len(sub)
    cash, units = 10000.0, 0.0
    eq = np.empty(n)
    entry_px = entry_cost = 0.0
    stop = np.inf
    peak = 0.0
    activated = False
    trades = []
    for i in range(n):
        if units > 0:
            if close[i-1] > peak:
                peak = close[i-1]
            if trail_from_start or (not activated and peak >= entry_px * (1 + trail_act)):
                activated = True
            if activated:
                lvl = peak * (1 - trail_pct)
                if lvl > stop:
                    stop = lvl
            if low[i] <= stop:
                exec_px = min(open_[i], stop)
                proceeds = units * exec_px * (1 - FEE)
                cash += proceeds
                trades.append({'pnl': proceeds - entry_cost})
                units = 0.0
                stop = np.inf
                peak = 0.0
                activated = False
        if units == 0 and i > 0 and np.isfinite(roll[i-1]) and close[i-1] >= roll[i-1] * 1.15:
            entry_px = open_[i] * (1 + SLIP)
            units = cash / entry_px
            entry_cost = units * entry_px * (1 + FEE)
            cash -= entry_cost
            stop = entry_px * (1 - stop_pct)
            peak = entry_px
            activated = False
        eq[i] = cash + units * close[i]
    return pd.Series(eq, index=sub.index), trades

def m(eq, trades):
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    return dict(ret=eq.iloc[-1]/eq.iloc[0]-1, dd=(eq/eq.cummax()-1).min(),
                n=len(trades), wr=len(wins)/len(trades) if trades else 0,
                pf=gp/gl if gl > 0 else float('inf'))

print('=== 全周期(2019-11~2026-09) 变体对比 ===')
for name, kw in [
    ('原版: 盈15%后, 高点回撤10%止盈', dict()),
    ('变体A: 回撤止盈放宽到15%', dict(trail_pct=0.15)),
    ('变体B: 回撤止盈放宽到20%', dict(trail_pct=0.20)),
    ('变体C: 从买入起就跟踪(不等+15%)', dict(trail_from_start=True)),
    ('变体D: 从买入起跟踪+20%回撤', dict(trail_from_start=True, trail_pct=0.20)),
]:
    eq, tr = run(df, **kw)
    r = m(eq, tr)
    print(f'{name:36s} 收益 {r["ret"]*100:+7.1f}% | 回撤 {r["dd"]*100:6.1f}% | 交易 {r["n"]:3d} | 胜率 {r["wr"]*100:4.1f}% | 盈亏比 {r["pf"]:.2f}')
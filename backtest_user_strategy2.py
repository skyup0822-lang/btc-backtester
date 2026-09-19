# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\Administrator\Documents\Codex\2026-09-09\x20\btc-backtester')
FEE, SLIP = 0.001, 0.0005
df = pd.read_csv(ROOT / 'data' / 'btc_usdt_1h.csv', index_col=0, parse_dates=True)
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

def backtest(sub, N, stop_pct=0.10, trail_act=0.15, trail_pct=0.10, always=False):
    close = sub['Close'].to_numpy(dtype=float)
    open_ = sub['Open'].to_numpy(dtype=float)
    low = sub['Low'].to_numpy(dtype=float)
    idx = sub.index
    roll = pd.Series(close, index=idx).rolling(N, min_periods=N).min().to_numpy()
    n = len(sub)
    cash = 10000.0
    units = 0.0
    eq = np.empty(n)
    entry_px = entry_cost = 0.0
    stop = np.inf
    peak = 0.0
    activated = False
    entry_i = 0
    trades = []
    for i in range(n):
        if units > 0:
            if close[i-1] > peak:
                peak = close[i-1]
            if not activated and peak >= entry_px * (1 + trail_act):
                activated = True
            if activated:
                lvl = peak * (1 - trail_pct)
                if lvl > stop:
                    stop = lvl
            if low[i] <= stop:
                exec_px = min(open_[i], stop)
                proceeds = units * exec_px * (1 - FEE)
                cash += proceeds
                pnl = proceeds - entry_cost
                trades.append({'pnl': pnl, 'pnl_pct': pnl/entry_cost if entry_cost else 0.0,
                               'bars': i-entry_i, 'reason': 'trail' if activated else 'stop'})
                units = 0.0
                stop = np.inf
                peak = 0.0
                activated = False
        if units == 0 and i > 0 and (always or (np.isfinite(roll[i-1]) and close[i-1] >= roll[i-1]*1.15)):
            entry_px = open_[i] * (1 + SLIP)
            units = cash / entry_px
            entry_cost = units * entry_px * (1 + FEE)
            cash -= entry_cost
            stop = entry_px * (1 - stop_pct)
            peak = entry_px
            activated = False
            entry_i = i
        eq[i] = cash + units * close[i]
    return pd.Series(eq, index=idx), trades

# sanity: always-in, stop 99% never hit -> must approximate buy&hold
eq, tr = backtest(df, 336, stop_pct=0.99, always=True)
print('SANITY always-in: 收益 %.1f%% (买入持有应为 +766%%, 扣费后略低)' % ((eq.iloc[-1]/eq.iloc[0]-1)*100))

# daily resample variant
daily = df.resample('1D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
print('daily bars:', len(daily), daily.index[0], daily.index[-1])
for N, label in [(20, '20天低点'), (60, '60天低点'), (120, '120天低点')]:
    eq, trades = backtest(daily, N)
    m = dict(total_return=eq.iloc[-1]/eq.iloc[0]-1,
             max_drawdown=(eq/eq.cummax()-1).min(),
             n=len(trades),
             wr=sum(1 for t in trades if t['pnl']>0)/len(trades) if trades else 0,
             pf=(sum(t['pnl'] for t in trades if t['pnl']>0)/abs(sum(t['pnl'] for t in trades if t['pnl']<=0))) if any(t['pnl']<=0 for t in trades) else float('inf'))
    print(f'{label} 日线: 收益 {m["total_return"]*100:+.1f}% | 最大回撤 {m["max_drawdown"]*100:.1f}% | 交易 {m["n"]} | 胜率 {m["wr"]*100:.1f}% | 盈亏比 {m["pf"]:.2f}')
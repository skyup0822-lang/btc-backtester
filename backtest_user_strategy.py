# -*- coding: utf-8 -*-
"""用户策略回测: 阶段低点+15%突破买入, -10%止损, +15%后激活高点回撤10%止盈。"""
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
print(f'数据: {df.index[0]} ~ {df.index[-1]}, {len(df)} 根1hK线')

def backtest(sub, N, stop_pct=0.10, trail_act=0.15, trail_pct=0.10, risk_pct=None):
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
                trades.append({
                    'entry': idx[entry_i], 'exit': idx[i],
                    'entry_px': entry_px, 'exit_px': exec_px,
                    'pnl': pnl, 'pnl_pct': pnl / entry_cost if entry_cost else 0.0,
                    'bars': i - entry_i,
                    'reason': 'trail' if activated else 'stop',
                })
                units = 0.0
                stop = np.inf
                peak = 0.0
                activated = False
        if units == 0 and i > 0 and np.isfinite(roll[i-1]) and close[i-1] >= roll[i-1] * 1.15:
            entry_px = open_[i] * (1 + SLIP)
            if risk_pct is not None:
                risk_usd = cash * risk_pct
                units = risk_usd / (entry_px * stop_pct)
            else:
                units = cash / entry_px
            entry_cost = units * entry_px * (1 + FEE)
            cash -= entry_cost
            stop = entry_px * (1 - stop_pct)
            peak = entry_px
            activated = False
            entry_i = i
        eq[i] = cash + units * close[i]
    return pd.Series(eq, index=idx), trades

def metrics(eq, trades):
    tr = eq.iloc[-1] / eq.iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).total_seconds() / 31557600.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else float('nan')
    dd = (eq / eq.cummax() - 1).min()
    n = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / n if n else float('nan')
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gp / gl if gl > 0 else float('inf')
    exp = (sum(t['pnl'] for t in trades) / n / 10000.0) if n else float('nan')
    avg_bars = sum(t['bars'] for t in trades) / n if n else float('nan')
    return dict(total_return=tr, cagr=cagr, max_drawdown=dd, n_trades=n,
                win_rate=wr, profit_factor=pf, expectancy_pct=exp, avg_holding_h=avg_bars)

rows = []
for N, label in [(336, '14天低点'), (720, '30天低点'), (1440, '60天低点')]:
    for split, part in [('train', df[df.index <= TRAIN_END]), ('test', df[df.index > TRAIN_END]), ('full', df)]:
        eq, trades = backtest(part, N)
        m = metrics(eq, trades)
        rows.append({'策略': label, '区间': split, **m})
        if split == 'full':
            print(f'{label} 全周期: 收益 {m["total_return"]*100:+.1f}% | 年化 {m["cagr"]*100:.1f}% | 最大回撤 {m["max_drawdown"]*100:.1f}% | 交易 {m["n_trades"]} | 胜率 {m["win_rate"]*100:.1f}% | 盈亏比 {m["profit_factor"]:.2f} | 单笔期望 {m["expectancy_pct"]*100:+.2f}% | 平均持仓 {m["avg_holding_h"]:.0f}h')
print()
print('===== 对比表 =====')
t = pd.DataFrame(rows)
for split in ['train', 'test', 'full']:
    sub = t[t['区间'] == split]
    print(f'--- {split} ---')
    print(sub[['策略','total_return','cagr','max_drawdown','n_trades','win_rate','profit_factor','expectancy_pct']].to_string(index=False))
t.to_csv(ROOT / 'results' / 'user_strategy_backtest.csv', index=False)
print('saved: results/user_strategy_backtest.csv')
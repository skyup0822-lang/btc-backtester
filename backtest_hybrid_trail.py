# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
sys.path.insert(0, r'C:\Users\Administrator\Documents\Codex\2026-09-09\x20\btc-backtester')
import pandas as pd
from config import DATA_FILE, DEFAULT_CONFIG
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from strategies import STRATEGIES, STRATEGY_EXTRA

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
name = 'trend_system_48_336_1440'
sig = STRATEGIES[name][1](df)
extra = dict(STRATEGY_EXTRA.get(name, {}))

print('=== 杂交策略 + 不同跟踪止盈 (全周期, 项目引擎) ===')
for label, kw in [
    ('杂交原版(仅-15%固定止损)', dict(extra)),
    ('杂交+高点回撤20%止盈(跟踪)', dict(extra, trail_pct=0.20)),
    ('杂交+高点回撤15%止盈(跟踪)', dict(extra, trail_pct=0.15)),
]:
    equity, trades, _ = run_backtest(df, sig, DEFAULT_CONFIG, **kw)
    m = compute_metrics(equity, trades, 365*24)
    print(f'{label:30s} 收益 {m["total_return"]*100:+8.1f}% | 最大回撤 {m["max_drawdown"]*100:6.1f}% | 交易 {m["n_trades"]:3d} | 胜率 {m["win_rate"]*100:4.1f}% | 盈亏比 {m["profit_factor"]:.2f} | 年化 {m["cagr"]*100:.1f}%')
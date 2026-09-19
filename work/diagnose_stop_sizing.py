# -*- coding: utf-8 -*-
"""诚实性诊断: 15%止损在趋势系统回测里到底触发过没有, 以及当前回测用的什么仓位模式。

回答三个问题:
  1. 用 run_backtest.py 同款配置(只传 stop_pct=0.15, 不传 risk_pct)跑, 退出原因是 data还是 stop?
  2. 仓位模式: 不传 risk_pct => 全仓; 传 risk_pct=0.01 => 每笔1%风险(实盘用这个)。
     对比两种模式的 总收益/最大回撤/成交数, 看报表口径和实盘是否一致。
  3. 若止损从未触发 => 说明"固定止损"在当前信号下是死代码, 结论要据此修正。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config import DATA_FILE, TRAIN_END
from strategies.trend_system import sma_gate

SHORT, LONG, GATE = 48, 336, 1440


def summarize(name, equity, trades, test):
    oos = [t for t in trades if t.entry_time > test]
    oos_pnls = np.array([t.pnl_pct for t in oos], dtype=float)
    all_pnls = np.array([t.pnl_pct for t in trades], dtype=float)
    t_eq = equity[equity.index > test]
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    sigl = sum(1 for t in trades if t.exit_reason == "signal")
    print(f"\n【{name}】")
    print(f"  全部交易 {len(trades)} 笔 | 信号离场 {sigl} | 止损离场 {stops} | "
          f"全周期单笔期望 {all_pnls.mean():+.3%}")
    print(f"  样本外({test:%Y-%m-%d}起) 交易 {len(oos)} 笔 | "
          f"单笔期望 {oos_pnls.mean():+.3%} | 胜率 {(oos_pnls > 0).mean():.1%}")
    print(f"  样本外 总收益 {t_eq.iloc[-1]/t_eq.iloc[0]-1:+.2%} | 最大回撤 {(t_eq/t_eq.cummax()-1).min():.2%}")


def main():
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    test = pd.Timestamp(TRAIN_END, tz="UTC")
    sig = sma_gate(df, short=SHORT, long=LONG, gate=GATE)

    # 全仓(报表口径: 只传 stop_pct)
    eq_a, tr_a, _ = run_backtest(df, sig, stop_pct=0.15)
    summarize("全仓(只传 stop_pct=0.15, run_backtest.py 同款)", eq_a, tr_a, test)

    # 1%风险(实盘口径: stop_pct + risk_pct)
    eq_b, tr_b, _ = run_backtest(df, sig, stop_pct=0.15, risk_pct=0.01)
    summarize("每笔1%风险(stop_pct=0.15 + risk_pct=0.01, 实盘同款)", eq_b, tr_b, test)

    print("\n说明:")
    print("  - 若『全仓』的止损离场数=0, 说明固定止损在信号驱动下从未触发(信号先离场)。")
    print("  - 若『全仓』与『1%风险』结果差异大, 说明报表口径(全仓)与实盘(1%风险)不一致。")


if __name__ == "__main__":
    main()

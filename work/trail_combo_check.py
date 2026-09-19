# -*- coding: utf-8 -*-
"""选定参数前的最后校验: 把 trail8 和 2%风险 组合起来, 看真实样本外结果。

三种组合(入场信号都一样=趋势系统, 只差退出/风险):
  A  trail8 + 每笔2%风险, 仓位尺子=8%   (notional = 2%/8% = 25%权益)
  B  trail8 + 每笔2%风险, 仓位尺子=15%  (notional = 2%/15% = 13.3%权益, 与之前报告一致)
  C  固定15%止损(死)+ 每笔2%风险         (对照, 无trail)
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
from config import DEFAULT_CONFIG, TRAIN_END
from strategies.trend_system import sma_gate

SHORT, LONG, GATE = 48, 336, 1440
TEST = pd.Timestamp(TRAIN_END, tz="UTC")
COMBO = {
    "A trail8+2%风险(尺=8%)":  dict(stop_pct=0.08, trail_pct=0.08, risk_pct=0.02),
    "B trail8+2%风险(尺=15%)": dict(stop_pct=0.15, trail_pct=0.08, risk_pct=0.02),
    "C 固定15%+2%风险":        dict(stop_pct=0.15, risk_pct=0.02),
}
ASSETS = {"BTC": ROOT / "data" / "btc_usdt_1h.csv",
          "ETH": ROOT / "data" / "eth_usdt_1h.csv"}


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC"))
    df["Volume"] = df["Volume"].fillna(0.0)
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].ffill()
    return df


def main():
    for asset, path in ASSETS.items():
        df = load(path)
        sig = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
        bench = df[df.index > TEST]["Close"]
        bench_ret = bench.iloc[-1] / bench.iloc[0] - 1
        print(f"\n【{asset}】 样本外 2025-01 起 | 买入持有 {bench_ret:+.2%}")
        print(f"  {'组合':<26} {'交易':>3} {'止损离场':>5} {'单笔期望':>9} {'区间收益':>9} {'最大回撤':>9}")
        for label, kw in COMBO.items():
            equity, trades, _ = run_backtest(df, sig, DEFAULT_CONFIG, **kw)
            t_eq = equity[equity.index > TEST]
            oos = [t for t in trades if t.entry_time > TEST]
            pnls = np.array([t.pnl_pct for t in oos], dtype=float)
            stops = sum(1 for t in trades if t.exit_reason == "stop")
            print(f"  {label:<26} {len(oos):>3} {stops:>5} {pnls.mean() if len(pnls) else float('nan'):>+9.3%} "
                  f"{t_eq.iloc[-1]/t_eq.iloc[0]-1:>+9.2%} {(t_eq/t_eq.cummax()-1).min():>9.2%}")


if __name__ == "__main__":
    main()

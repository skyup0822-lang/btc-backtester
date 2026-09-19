# -*- coding: utf-8 -*-
"""跟踪止损(trailing stop)与固定止损的诚实对比 + "小资金赚更多"的风险阶梯。

三层结论都只基于样本外(2025-01 起):
  ① 隔离止损类型: 全仓跑, 只看"退出规则"对 期望/收益/回撤 的影响(固定15% vs 各种trail%)。
  ② 看 trail 是否真的触发(有没有变成"没用的死代码")。
  ③ 赚钱阶梯: 在选定的退出规则上, 把每笔风险从 1% 提到 2/3/5/8%, 看 收益/回撤 怎么线性放大——
     这就是"小资金想赚更多"的代价量化, 绝无免费午餐。
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
ASSETS = {
    "BTC": ROOT / "data" / "btc_usdt_1h.csv",
    "ETH": ROOT / "data" / "eth_usdt_1h.csv",
}
# 退出规则变体: 只定"离场", 入场信号完全一致(趋势系统 48/336/1440)
#   fixed15 = 固定15%止损(当前实盘, 但回测里它从不触发 → 实际=纯信号离场)
#   trailX  = X% 跟踪止损(只上移), 这是在趋势里锁利润的候选
VARIANTS = [
    ("fixed15(信号离场)", dict(stop_pct=0.15)),
    ("trail5",            dict(trail_pct=0.05)),
    ("trail8",            dict(trail_pct=0.08)),
    ("trail10",           dict(trail_pct=0.10)),
    ("trail15",           dict(trail_pct=0.15)),
    ("trail20",           dict(trail_pct=0.20)),
]

TEST_START = pd.Timestamp(TRAIN_END, tz="UTC")


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    df = df.reindex(full)
    df["Volume"] = df["Volume"].fillna(0.0)
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].ffill()
    return df


def run(df, kwargs):
    sig = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
    equity, trades, _ = run_backtest(df, sig, DEFAULT_CONFIG, **kwargs)
    return equity, trades


def oos_stats(equity, trades):
    t_eq = equity[equity.index > TEST_START]
    oos = [t for t in trades if t.entry_time > TEST_START]
    pnls = np.array([t.pnl_pct for t in oos], dtype=float)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    return {
        "all_trades": len(trades), "stop_exits": stops,
        "n": len(pnls),
        "exp": pnls.mean() if len(pnls) else np.nan,
        "win": (pnls > 0).mean() if len(pnls) else np.nan,
        "ret": t_eq.iloc[-1] / t_eq.iloc[0] - 1,
        "dd": (t_eq / t_eq.cummax() - 1).min(),
    }


def main():
    print("=" * 108)
    print("① 退出规则对比(全仓, 样本外 2025-01 起) —— 隔离『止损变体』对 edge 的影响")
    print("=" * 108)
    oos_rows = []
    for asset, path in ASSETS.items():
        df = load(path)
        print(f"\n【{asset}】")
        print(f"  {'变体':<18} {'交易':>3} {'止损离场':>5} {'单笔期望':>9} {'胜率':>7} {'区间收益':>9} {'最大回撤':>9}")
        for label, kw in VARIANTS:
            equity, trades = run(df, kw)
            s = oos_stats(equity, trades)
            oos_rows.append({"asset": asset, "variant": label, **s})
            print(f"  {label:<18} {s['n']:>3} {s['stop_exits']:>5} {s['exp']:>+9.3%} "
                  f"{s['win']:>7.1%} {s['ret']:>+9.2%} {s['dd']:>9.2%}")

    print()
    print("=" * 108)
    print("② 赚钱阶梯 (每笔风险 r%, 固定15%止损作为仓位尺子, 样本外 2025-01 起)")
    print("   说明: 期望/胜率与风险无关; 收益/回撤随 r% 线性放大。这就是'小资金赚更多'的价格。")
    print("=" * 108)
    RISKS = [0.01, 0.02, 0.03, 0.05, 0.08]
    for asset, path in ASSETS.items():
        df = load(path)
        sig = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
        print(f"\n【{asset}】 每笔风险从 1% 往上加")
        print(f"  {'每笔风险':>8} {'名义仓位/笔':>10} {'区间收益':>9} {'最大回撤':>9} {'夏普' if False else '年化收益':>10}")
        for r in RISKS:
            equity, trades, _ = run_backtest(df, sig, DEFAULT_CONFIG, stop_pct=0.15, risk_pct=r)
            t_eq = equity[equity.index > TEST_START]
            ret = t_eq.iloc[-1] / t_eq.iloc[0] - 1
            dd = (t_eq / t_eq.cummax() - 1).min()
            years = len(t_eq) / (365 * 24)
            cagr = (t_eq.iloc[-1] / t_eq.iloc[0]) ** (1 / years) - 1
            notional = r / 0.15
            print(f"  {r:>7.1%} {notional:>9.1%} {ret:>+9.2%} {dd:>9.2%} {cagr:>10.2%}")

    print()
    print("(" + "=" * 108 + ")")
    print("诚实前提: '小资金赚更多' = 主动接受更大回撤, 没有任何免费午餐。")
    print("跟踪止损的价值只有一条: 若它在样本外把回撤压下来、又不砍右尾赢家, 你才有资格把 r% 提高。")


if __name__ == "__main__":
    main()

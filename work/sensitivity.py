"""参数敏感性检验:对选定参数做 +/-20% 扰动,观察样本外(2025-01 起)绩效是否翻转。

判定标准:单笔期望在样本外仍为正、回撤没有显著恶化,才算稳健;
若某个参数微调就导致期望变负,说明该参数存在过拟合风险。
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
from backtest.metrics import buy_hold_metrics
from config import DATA_FILE, DEFAULT_CONFIG, TRAIN_END
from strategies.trend_system import sma_gate

VARIANTS = [
    ("基线 48/336/1440, 止损15%", dict(short=48, long=336, gate=1440, stop=0.15)),
    ("短均线 38 (-20%)",           dict(short=38, long=336, gate=1440, stop=0.15)),
    ("短均线 58 (+20%)",           dict(short=58, long=336, gate=1440, stop=0.15)),
    ("长均线 269 (-20%)",          dict(short=48, long=269, gate=1440, stop=0.15)),
    ("长均线 403 (+20%)",          dict(short=48, long=403, gate=1440, stop=0.15)),
    ("闸门 1152 (-20%)",           dict(short=48, long=336, gate=1152, stop=0.15)),
    ("闸门 1728 (+20%)",           dict(short=48, long=336, gate=1728, stop=0.15)),
    ("止损 12%",                   dict(short=48, long=336, gate=1440, stop=0.12)),
    ("止损 18%",                   dict(short=48, long=336, gate=1440, stop=0.18)),
    ("去掉止损",                   dict(short=48, long=336, gate=1440, stop=None)),
]


def fmt(v, suffix="%"):
    if pd.isna(v):
        return "-"
    return f"{v * 100:+.2f}{suffix}"


def main() -> None:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    test_start = pd.Timestamp(TRAIN_END, tz="UTC")
    test = df[df.index > test_start]
    print(f"样本外区间: {test.index[0]} ~ {test.index[-1]}, {len(test)} 根 1h K 线")
    print()

    bench, _ = buy_hold_metrics(test, DEFAULT_CONFIG)
    rows = []
    for label, p in VARIANTS:
        signals = sma_gate(df, short=p["short"], long=p["long"], gate=p["gate"])
        equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, stop_pct=p["stop"])
        test_trades = [t for t in trades if t.entry_time > test_start]
        pnls = np.array([t.pnl_pct for t in test_trades], dtype=float)
        t_eq = equity[equity.index > test_start]
        rows.append({
            "variant": label,
            "n_trades": len(pnls),
            "expectancy_pct": pnls.mean() if len(pnls) else np.nan,
            "win_rate": (pnls > 0).mean() if len(pnls) else np.nan,
            "test_return": t_eq.iloc[-1] / t_eq.iloc[0] - 1,
            "max_drawdown": (t_eq / t_eq.cummax() - 1).min(),
        })
    table = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "sensitivity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print("===== 参数敏感性(样本外 2025-01-01 ~ 2026-09-07) =====")
    print(f"{'参数组合':<30} {'交易':>4} {'单笔期望':>9} {'胜率':>8} {'区间收益':>9} {'最大回撤':>9}")
    for _, r in table.iterrows():
        print(f"{r['variant']:<30} {r['n_trades']:>4.0f} {fmt(r['expectancy_pct']):>9} "
              f"{fmt(r['win_rate']):>8} {fmt(r['test_return']):>9} {fmt(r['max_drawdown']):>9}")
    print()
    print(f"同期买入持有收益: {bench['total_return']:+.2%} | 最大回撤: {bench['max_drawdown']:.2%}")
    print(f"结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
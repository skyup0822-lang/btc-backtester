"""成交量开仓门检验:在趋势系统入场规则上加"成交量确认",看样本外是否真的更好。

问题背景:核心信号(sma_gate)只看收盘价,SMA 交叉可能发生在成交量极低的死水里,
那种突破没有资金参与,容易被假K线/插针骗进去。这里测的是:
    入场触发当根 Volume >= vol_mult * 过去 vol_window 根成交量中位数
不满足就放弃这次入场,继续空仓等下一次触发(离场规则完全不变)。

口径与 sensitivity.py 一致:信号在全周期计算,回测跑全周期,再按开仓时间切片到样本外。
注意成交量用的是"相对自己滚动中位数的倍数",不是绝对值 —— 三家交易所的绝对
成交量差好几倍(BTC 那一小时 Binance 605 个 / Gate 123 个 / OKX 89 个),
只有相对量才跨交易所可比。
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
from strategies.trend_system import sma_gate, sma_gate_filtered

# 与实盘固化参数一致: 8% 跟踪止损 + 每笔 2% 风险
LIVE_EXTRA = {"stop_pct": 0.15, "trail_pct": 0.08, "risk_pct": 0.02}
VOL_WINDOW = 336  # 14 天

VARIANTS = [
    ("基线(无成交量门)", dict(vol_mult=0.0, atr_mult=0.0)),
    ("量门 1.0x", dict(vol_mult=1.0, atr_mult=0.0)),
    ("量门 1.2x", dict(vol_mult=1.2, atr_mult=0.0)),
    ("量门 1.5x", dict(vol_mult=1.5, atr_mult=0.0)),
    ("量门 2.0x", dict(vol_mult=2.0, atr_mult=0.0)),
    ("量门 3.0x", dict(vol_mult=3.0, atr_mult=0.0)),
    ("量门1.5x + ATR门", dict(vol_mult=1.5, atr_mult=1.0)),
]


def fmt(v):
    if pd.isna(v):
        return "-"
    return f"{v * 100:+.2f}%"


def main() -> None:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    test_start = pd.Timestamp(TRAIN_END, tz="UTC")
    test = df[df.index > test_start]
    print(f"数据: {df.index[0]} ~ {df.index[-1]}, {len(df)} 根 1h K 线")
    print(f"样本外区间: {test.index[0]} ~ {test.index[-1]}, {len(test)} 根")
    print(f"成交量中位数窗口: {VOL_WINDOW} 根({VOL_WINDOW // 24} 天)")
    print()

    base = sma_gate(df, short=48, long=336, gate=1440)
    n_trig = int((base.diff() > 0).sum())

    rows = []
    for label, p in VARIANTS:
        if p["vol_mult"] == 0.0 and p["atr_mult"] == 0.0:
            signals = base
        elif p["atr_mult"] == 0.0:
            signals = sma_gate_filtered(df, short=48, long=336, gate=1440,
                                        vol_window=VOL_WINDOW, vol_mult=p["vol_mult"],
                                        atr_mult=0.0)
        else:
            signals = sma_gate_filtered(df, short=48, long=336, gate=1440,
                                        vol_window=VOL_WINDOW, vol_mult=p["vol_mult"],
                                        atr_mult=p["atr_mult"])
        n_kept = int((signals.diff() > 0).sum())
        equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, **LIVE_EXTRA)
        test_trades = [t for t in trades if t.entry_time > test_start]
        pnls = np.array([t.pnl_pct for t in test_trades], dtype=float)
        t_eq = equity[equity.index > test_start]
        rows.append({
            "variant": label,
            "n_trades": len(pnls),
            "blocked": n_trig - n_kept,
            "expectancy_pct": pnls.mean() if len(pnls) else np.nan,
            "median_pct": float(np.median(pnls)) if len(pnls) else np.nan,
            "win_rate": (pnls > 0).mean() if len(pnls) else np.nan,
            "test_return": t_eq.iloc[-1] / t_eq.iloc[0] - 1,
            "max_drawdown": (t_eq / t_eq.cummax() - 1).min(),
        })

    table = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "volume_gate.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    bench, _ = buy_hold_metrics(test, DEFAULT_CONFIG)
    print("===== 成交量开仓门(样本外) =====")
    print(f"{'方案':<22} {'交易':>4} {'拦掉':>4} {'单笔期望':>9} {'中位数':>9} "
          f"{'胜率':>8} {'区间收益':>9} {'最大回撤':>9}")
    for _, r in table.iterrows():
        print(f"{r['variant']:<22} {r['n_trades']:>4.0f} {r['blocked']:>4.0f} "
              f"{fmt(r['expectancy_pct']):>9} {fmt(r['median_pct']):>9} "
              f"{fmt(r['win_rate']):>8} {fmt(r['test_return']):>9} {fmt(r['max_drawdown']):>9}")
    print()
    print(f"同期买入持有收益: {bench['total_return']:+.2%} | 最大回撤: {bench['max_drawdown']:.2%}")
    print(f"结果已保存: {out_csv}")


if __name__ == "__main__":
    main()

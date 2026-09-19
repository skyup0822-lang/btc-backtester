# -*- coding: utf-8 -*-
"""跨品种验证:用与 BTC 完全相同的趋势系统参数与成本假设,在 ETH 上重复
滚动窗口(walk-forward)验证,并把两品种的交易样本合并,重估每笔数学期望。

规则: SMA48 上穿 SMA336 且收盘价 > SMA1440 才做多; 固定 15% 止损;
      手续费 0.10%/边 + 滑点 0.05%/边。一个参数都不调。

输出:
  results/walk_forward_eth.csv        ETH 分段明细
  results/pooled_stats.csv            合并统计(BTC / ETH / 合并)
  results/cross_validate_equity.png   两品种样本外净值对比图
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mplconfig"))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics, compute_metrics
from config import DEFAULT_CONFIG, PERIODS_PER_YEAR
from strategies.trend_system import sma_gate

SHORT, LONG, GATE = 48, 336, 1440
STOP_PCT = 0.15
SEGMENT_DAYS = 182
PRIME_DAYS = 180
BOOTSTRAP_N = 5000
SEED = 42

SYMBOLS = {
    "BTC": ROOT / "data" / "btc_usdt_1h.csv",
    "ETH": ROOT / "data" / "eth_usdt_1h.csv",
}


def fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v * 100:+.2f}%"


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    df = df.reindex(full)
    df["Volume"] = df["Volume"].fillna(0.0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df[cols] = df[cols].ffill()
    return df


def walk_forward(df: pd.DataFrame):
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = df.index[-1]
    rows, all_trades = [], []
    seg_start, seg_id = start, 1
    while seg_start < end:
        seg_end = min(seg_start + pd.Timedelta(days=SEGMENT_DAYS), end)
        prime_start = seg_start - pd.Timedelta(days=PRIME_DAYS)
        window = df[(df.index >= prime_start) & (df.index <= seg_end)]
        if len(window) < 300:
            break
        signals = sma_gate(window, short=SHORT, long=LONG, gate=GATE)
        equity, trades, _ = run_backtest(window, signals, DEFAULT_CONFIG, stop_pct=STOP_PCT)
        seg_trades = [t for t in trades if seg_start <= t.entry_time < seg_end]
        seg_eq = equity[equity.index >= seg_start]
        bench = window["Close"][window.index >= seg_start]
        pnls = np.array([t.pnl_pct for t in seg_trades], dtype=float)
        if len(pnls) > 0:
            expectancy = pnls.mean()
            win_rate = (pnls > 0).mean()
            wins = pnls[pnls > 0].sum()
            losses = -pnls[pnls < 0].sum()
            pf = wins / losses if losses > 0 else np.inf
        else:
            expectancy = win_rate = pf = np.nan
        rows.append({
            "segment": seg_id,
            "start": seg_start,
            "end": seg_end,
            "n_trades": len(pnls),
            "expectancy_pct": expectancy,
            "win_rate": win_rate,
            "profit_factor": pf,
            "strat_return": seg_eq.iloc[-1] / seg_eq.iloc[0] - 1,
            "bench_return": bench.iloc[-1] / bench.iloc[0] - 1,
            "max_drawdown": (seg_eq / seg_eq.cummax() - 1).min(),
        })
        all_trades.extend(seg_trades)
        seg_start = seg_end
        seg_id += 1
    return pd.DataFrame(rows), all_trades


def oos_summary(df: pd.DataFrame):
    """样本外(2025-01-01 至今)整体成绩单,口径与之前 BTC 敏感性分析一致。"""
    signals = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
    test = df[df.index > pd.Timestamp("2024-12-31", tz="UTC")]
    equity, trades, _ = run_backtest(test, signals.reindex(test.index), DEFAULT_CONFIG, stop_pct=STOP_PCT)
    metrics = compute_metrics(equity, trades, PERIODS_PER_YEAR)
    bench, bench_equity = buy_hold_metrics(test, DEFAULT_CONFIG)
    return {
        "n_trades": metrics["n_trades"],
        "expectancy_pct": metrics["expectancy_pct"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "strat_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "bench_return": bench["total_return"],
        "bench_drawdown": bench["max_drawdown"],
    }, equity, bench_equity


def bootstrap_ci(pnls: np.ndarray):
    rng = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(pnls, size=len(pnls), replace=True).mean()
        for _ in range(BOOTSTRAP_N)
    ])
    return np.percentile(boot, [2.5, 97.5])


def main() -> None:
    frames = {name: load_clean(p) for name, p in SYMBOLS.items()}
    common_start = max(f.index[0] for f in frames.values())
    common_end = min(f.index[-1] for f in frames.values())
    frames = {
        name: f[(f.index >= common_start) & (f.index <= common_end)]
        for name, f in frames.items()
    }
    print(f"对齐后共同时间范围: {common_start} ~ {common_end} (UTC)")
    print(f"成本假设: 手续费 {DEFAULT_CONFIG.fee_rate:.2%}/边, 滑点 {DEFAULT_CONFIG.slippage:.2%}/边\n")

    per_trade = {}
    positive_seg = {}
    for name, df in frames.items():
        table, trades = walk_forward(df)
        per_trade[name] = np.array([t.pnl_pct for t in trades], dtype=float)
        positive_seg[name] = int(table["expectancy_pct"].gt(0).sum())
        if name == "ETH":
            out = ROOT / "results" / "walk_forward_eth.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out, index=False)
        print(f"===== {name} 滚动窗口验证 (12 段 x 半年, 2021-01 ~ {df.index[-1]:%Y-%m-%d}) =====")
        for _, r in table.iterrows():
            pf_str = "inf" if (isinstance(r["profit_factor"], float) and np.isinf(r["profit_factor"])) else (
                "-" if pd.isna(r["profit_factor"]) else f"{r['profit_factor']:.2f}")
            print(
                f"第{r['segment']:>2}段 {r['start']:%Y-%m-%d} ~ {r['end']:%Y-%m-%d}"
                f" | 交易 {r['n_trades']:>2} 笔 | 单笔期望 {fmt_pct(r['expectancy_pct']):>8}"
                f" | 胜率 {fmt_pct(r['win_rate']):>7} | 盈亏比 {pf_str:>5}"
                f" | 策略 {fmt_pct(r['strat_return']):>8} | 买入持有 {fmt_pct(r['bench_return']):>8}"
                f" | 最大回撤 {fmt_pct(r['max_drawdown']):>8}"
            )
        pnls = per_trade[name]
        if len(pnls) > 0:
            lo, hi = bootstrap_ci(pnls)
            wins = pnls[pnls > 0].sum()
            losses = -pnls[pnls < 0].sum()
            print(f"【{name} 汇总】交易 {len(pnls)} 笔 | 单笔期望 {pnls.mean():+.3%} "
                  f"(bootstrap 95% CI: {lo:+.3%} ~ {hi:+.3%})")
            print(f"  中位数 {np.median(pnls):+.3%} | 标准差 {pnls.std():.3%} | 胜率 {(pnls > 0).mean():.1%} "
                  f"| 盈亏比 {wins / losses if losses > 0 else float('inf'):.2f}")
            print(f"  段落稳定性: {positive_seg[name]}/{int(table['expectancy_pct'].notna().sum())} 段期望为正")
        else:
            print(f"【{name} 汇总】没有产生交易")
        print()

    pooled = np.concatenate([per_trade["BTC"], per_trade["ETH"]])
    lo, hi = bootstrap_ci(pooled)
    wins = pooled[pooled > 0].sum()
    losses = -pooled[pooled < 0].sum()
    print("=" * 92)
    print("【两品种合并统计】")
    print(f"  交易总数        : {len(pooled)} (BTC {len(per_trade['BTC'])} + ETH {len(per_trade['ETH'])})")
    print(f"  单笔数学期望    : {pooled.mean():+.3%}  (bootstrap 95% CI: {lo:+.3%} ~ {hi:+.3%})")
    print(f"  单笔期望中位数  : {np.median(pooled):+.3%} | 标准差 {pooled.std():.3%}")
    print(f"  胜率            : {(pooled > 0).mean():.1%} | 盈亏比 {wins / losses if losses > 0 else float('inf'):.2f}")
    print(f"  段落稳定性      : {positive_seg['BTC'] + positive_seg['ETH']}/{12 * 2} 段期望为正")
    print("=" * 92)

    oos_rows, equities = [], {}
    for name, df in frames.items():
        s, eq, bench_eq = oos_summary(df)
        oos_rows.append({"symbol": name, **s})
        equities[name] = (eq, bench_eq)
    oos = pd.DataFrame(oos_rows)
    print()
    print("【样本外成绩单对比 (2025-01-01 ~ 2026-09-07)】")
    for _, r in oos.iterrows():
        pf_str = "inf" if np.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        print(f"{r['symbol']}: 交易 {r['n_trades']:>3} 笔 | 单笔期望 {r['expectancy_pct']:+.2%} "
              f"| 胜率 {r['win_rate']:.1%} | 盈亏比 {pf_str} | 策略收益 {r['strat_return']:+.2%} "
              f"| 最大回撤 {r['max_drawdown']:.2%} | 买入持有 {r['bench_return']:+.2%} "
              f"(回撤 {r['bench_drawdown']:.2%})")

    stats = pd.DataFrame({
        "symbol": ["BTC", "ETH", "BTC+ETH"],
        "n_trades": [len(per_trade["BTC"]), len(per_trade["ETH"]), len(pooled)],
        "expectancy_pct": [per_trade["BTC"].mean(), per_trade["ETH"].mean(), pooled.mean()],
        "ci_low": [bootstrap_ci(per_trade["BTC"])[0], bootstrap_ci(per_trade["ETH"])[0], lo],
        "ci_high": [bootstrap_ci(per_trade["BTC"])[1], bootstrap_ci(per_trade["ETH"])[1], hi],
        "median_pct": [np.median(per_trade["BTC"]), np.median(per_trade["ETH"]), np.median(pooled)],
        "win_rate": [(per_trade["BTC"] > 0).mean(), (per_trade["ETH"] > 0).mean(), (pooled > 0).mean()],
        "positive_segments": [positive_seg["BTC"], positive_seg["ETH"], positive_seg["BTC"] + positive_seg["ETH"]],
    })
    out_csv = ROOT / "results" / "pooled_stats.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_csv, index=False)
    print(f"\n已保存: {out_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for font_name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        fp = windir / "Fonts" / font_name
        if fp.exists():
            try:
                font_manager.fontManager.addfont(str(fp))
                plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (name, (eq, bench_eq)) in zip(axes, equities.items()):
        ax.plot(eq.index, eq / eq.iloc[0], label="趋势系统", linewidth=1.4, color="#1f77b4")
        ax.plot(bench_eq.index, bench_eq / bench_eq.iloc[0], label="买入持有", linewidth=1.2,
                color="black", linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"{name} 样本外净值 (2025-01 起, log 坐标)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("同一套策略参数在两个品种上的样本外表现", fontsize=13)
    fig.tight_layout()
    png = ROOT / "results" / "cross_validate_equity.png"
    fig.savefig(png, dpi=150)
    print(f"已保存: {png}")


if __name__ == "__main__":
    main()
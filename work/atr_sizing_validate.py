# -*- coding: utf-8 -*-
"""ATR 止损 + 波动率反向仓位验证

三个方案同台对比(离场/入场规则只有止损和仓位不同):
  方案1 base   : 原趋势系统 + 固定 15% 止损 + 全仓
  方案2 atrstop: 原趋势系统 + k*ATR14 止损 + 全仓     (只换"尺子")
  方案3 sized  : 原趋势系统 + k*ATR14 止损 + 每笔风险 r% (波动大自动下小注)
指标统一用"每笔盈亏 / 入场时账户权益"做期望, 三种仓位方案才能公平比较。
验证口径不变: BTC/ETH 各 12 段滚动回测(2021-01~2026-09) + 5000 bootstrap
+ 样本外成绩单(2025-01+) + 参数稳健性网格。
输出:
  results/atr_sizing_segments.csv            三方案x两品种分段明细
  results/atr_sizing_summary.csv             三方案汇总对比(含置信区间)
  results/atr_sizing_grid.csv                参数网格
  results/atr_sizing_expectancy.png          期望与置信区间三方案对比图
  results/atr_sizing_oos_equity.png          样本外净值对比图
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_LIB = ROOT / "work" / "pylibs"
if _LIB.exists() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mplconfig"))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config import DEFAULT_CONFIG
from strategies.trend_system import sma_gate

SHORT, LONG, GATE = 48, 336, 1440
ATR_WINDOW = 14
K_DEFAULT = 3.0
RISK_DEFAULT = 0.01
SEGMENT_DAYS = 182
PRIME_DAYS = 180
BOOTSTRAP_N = 5000
SEED = 42

SYMBOLS = {
    "BTC": ROOT / "data" / "btc_usdt_1h.csv",
    "ETH": ROOT / "data" / "eth_usdt_1h.csv",
}

SYSTEMS = {
    "base": "原策略(15%固定止损)",
    "atrstop": f"ATR止损({K_DEFAULT:g}xATR)",
    "sized": f"ATR止损+仓位({K_DEFAULT:g}xATR, 每笔风险{RISK_DEFAULT:.0%})",
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


def atr_series(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def run_system(df, kind, k=K_DEFAULT, r=RISK_DEFAULT):
    signals = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
    if kind == "base":
        return run_backtest(df, signals, DEFAULT_CONFIG, stop_pct=0.15)
    atr = atr_series(df)
    if kind == "atrstop":
        return run_backtest(df, signals, DEFAULT_CONFIG, atr=atr, atr_stop_mult=k)
    if kind == "sized":
        return run_backtest(df, signals, DEFAULT_CONFIG, atr=atr, atr_stop_mult=k, risk_pct=r)
    raise ValueError(kind)


def walk_forward(df, kind, k=K_DEFAULT, r=RISK_DEFAULT):
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
        equity, trades, _ = run_system(window, kind, k, r)
        seg_trades = [t for t in trades if seg_start <= t.entry_time < seg_end]
        seg_eq = equity[equity.index >= seg_start]
        bench = window["Close"][window.index >= seg_start]
        pnls = np.array([t.pnl_acct_pct for t in seg_trades], dtype=float)
        if len(pnls) > 0:
            expectancy = pnls.mean()
            win_rate = (pnls > 0).mean()
            wins = np.array([t.pnl for t in seg_trades]).clip(min=0).sum()
            losses = -np.array([t.pnl for t in seg_trades]).clip(max=0).sum()
            pf = wins / losses if losses > 0 else np.inf
        else:
            expectancy = win_rate = pf = np.nan
        rows.append({
            "segment": seg_id,
            "start": seg_start,
            "end": seg_end,
            "n_trades": len(pnls),
            "expectancy_acct_pct": expectancy,
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


def oos_summary(df, kind, k=K_DEFAULT, r=RISK_DEFAULT):
    signals = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
    atr = atr_series(df) if kind != "base" else None
    test = df[df.index > pd.Timestamp("2024-12-31", tz="UTC")]
    if kind == "base":
        equity, trades, _ = run_backtest(
            test, signals.reindex(test.index), DEFAULT_CONFIG, stop_pct=0.15
        )
    elif kind == "atrstop":
        equity, trades, _ = run_backtest(
            test, signals.reindex(test.index), DEFAULT_CONFIG,
            atr=atr.reindex(test.index), atr_stop_mult=k,
        )
    else:
        equity, trades, _ = run_backtest(
            test, signals.reindex(test.index), DEFAULT_CONFIG,
            atr=atr.reindex(test.index), atr_stop_mult=k, risk_pct=r,
        )
    bench_equity = test["Close"] / test["Close"].iloc[0] * DEFAULT_CONFIG.initial_capital
    bench_equity = bench_equity * (1 - DEFAULT_CONFIG.fee_rate - DEFAULT_CONFIG.slippage) ** 2
    pnls = np.array([t.pnl_acct_pct for t in trades], dtype=float)
    wins = np.array([t.pnl for t in trades]).clip(min=0).sum()
    losses = -np.array([t.pnl for t in trades]).clip(max=0).sum()
    return {
        "n_trades": len(pnls),
        "expectancy_acct_pct": pnls.mean() if len(pnls) else np.nan,
        "win_rate": (pnls > 0).mean() if len(pnls) else np.nan,
        "profit_factor": wins / losses if losses > 0 else np.inf,
        "strat_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "bench_return": bench_equity.iloc[-1] / bench_equity.iloc[0] - 1,
        "bench_drawdown": (bench_equity / bench_equity.cummax() - 1).min(),
    }, equity, bench_equity


def bootstrap_ci(pnls):
    rng = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(pnls, size=len(pnls), replace=True).mean()
        for _ in range(BOOTSTRAP_N)
    ])
    return np.percentile(boot, [2.5, 97.5])


def trade_stats(name, trades, positive_seg, segments):
    pnls = np.array([t.pnl_acct_pct for t in trades], dtype=float)
    row = {
        "symbol": name,
        "n_trades": len(pnls),
        "positive_segments": positive_seg,
        "segments": segments,
    }
    if len(pnls) == 0:
        row.update({
            "expectancy_acct_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "median_acct_pct": np.nan, "win_rate": np.nan, "profit_factor": np.nan,
        })
        return row
    lo, hi = bootstrap_ci(pnls)
    wins = np.array([t.pnl for t in trades]).clip(min=0).sum()
    losses = -np.array([t.pnl for t in trades]).clip(max=0).sum()
    row.update({
        "expectancy_acct_pct": pnls.mean(),
        "ci_low": lo,
        "ci_high": hi,
        "median_acct_pct": np.median(pnls),
        "win_rate": (pnls > 0).mean(),
        "profit_factor": wins / losses if losses > 0 else np.inf,
    })
    return row


def setup_fonts():
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
    return plt


def print_segments(name, table):
    print(f"\n【{name} · {SYSTEMS['sized']} 分段明细】")
    for _, r in table.iterrows():
        pf_str = "inf" if np.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        print(
            f"  段{r['segment']:>2} {r['start'].strftime('%Y-%m')}~{r['end'].strftime('%Y-%m')} "
            f"| 交易 {r['n_trades']:>3} 笔 | 期望 {fmt_pct(r['expectancy_acct_pct']):>8} | 胜率 {fmt_pct(r['win_rate']):>7} "
            f"| 盈亏比 {pf_str:>5} | 策略 {fmt_pct(r['strat_return']):>8} | 买入持有 {fmt_pct(r['bench_return']):>8} "
            f"| 回撤 {fmt_pct(r['max_drawdown']):>8}"
        )


def main():
    frames = {name: load_clean(p) for name, p in SYMBOLS.items()}

    print("=" * 100)
    print("第一步 · 三方案滚动回测 (2021-01 ~ 2026-09, 每品种 12 段)")
    tables = {}
    per_trade = {}
    summaries = {}
    for name, df in frames.items():
        print(f"\n【{name}】")
        tables[name] = {}
        per_trade[name] = {}
        summaries[name] = {}
        for kind in ("base", "atrstop", "sized"):
            table, trades = walk_forward(df, kind)
            tables[name][kind] = table.assign(symbol=name, system=kind)
            per_trade[name][kind] = trades
            summaries[name][kind] = trade_stats(
                name, trades,
                int((table["expectancy_acct_pct"] > 0).sum()),
                int(table["expectancy_acct_pct"].notna().sum()),
            )
            s = summaries[name][kind]
            print(f"  {SYSTEMS[kind]:<28}: 交易 {s['n_trades']:>3} 笔 "
                  f"| 每笔期望(账户%) {s['expectancy_acct_pct']:+.3%} "
                  f"| 95%CI [{s['ci_low']:+.3%}, {s['ci_high']:+.3%}] "
                  f"| 胜率 {s['win_rate']:.1%} | 正段 {s['positive_segments']}/{s['segments']}")
        print_segments(name, tables[name]["sized"])

    print()
    print("=" * 100)
    print("第二步 · 三方案合并统计 (BTC+ETH, bootstrap 95% 置信区间)")
    pooled_stats = {}
    summary_rows = []
    for kind in ("base", "atrstop", "sized"):
        trades = per_trade["BTC"][kind] + per_trade["ETH"][kind]
        pos = summaries["BTC"][kind]["positive_segments"] + summaries["ETH"][kind]["positive_segments"]
        seg = summaries["BTC"][kind]["segments"] + summaries["ETH"][kind]["segments"]
        s = trade_stats("BTC+ETH", trades, pos, seg)
        pooled_stats[kind] = s
        summary_rows.append({"system": kind, **s})
        print(f"  {SYSTEMS[kind]:<28}: 交易 {s['n_trades']:>3} 笔 "
              f"| 每笔期望(账户%) {s['expectancy_acct_pct']:+.3%} "
              f"| 95%CI [{s['ci_low']:+.3%}, {s['ci_high']:+.3%}] "
              f"| 中位 {s['median_acct_pct']:+.3%} | 胜率 {s['win_rate']:.1%} | 正段 {s['positive_segments']}/{s['segments']}")
    for kind in ("base", "atrstop", "sized"):
        for name in ("BTC", "ETH"):
            s = summaries[name][kind]
            summary_rows.append({"system": kind, **s})

    print()
    print("=" * 100)
    print("第三步 · 样本外成绩单 (2025-01-01 ~ 2026-09-07)")
    oos_eq = {}
    oos_rows = []
    for name, df in frames.items():
        oos_eq[name] = {}
        for kind in ("base", "atrstop", "sized"):
            s, eq, bench_eq = oos_summary(df, kind)
            oos_eq[name][kind] = (eq, bench_eq)
            oos_rows.append({"symbol": name, "system": kind, **s})
            print(f"  {name} {SYSTEMS[kind]:<28}: 交易 {s['n_trades']:>3} 笔 "
                  f"| 每笔期望 {s['expectancy_acct_pct']:+.3%} | 策略收益 {s['strat_return']:+.2%} "
                  f"| 最大回撤 {s['max_drawdown']:.2%} | 买入持有 {s['bench_return']:+.2%} "
                  f"(回撤 {s['bench_drawdown']:.2%})")

    print()
    print("=" * 100)
    print("第四步 · 参数稳健性网格 (BTC+ETH 合并, 完整滚动12段)")
    grid_rows = []
    for kind, combos in (
        ("atrstop", [(k, None) for k in (2.0, 3.0, 4.0)]),
        ("sized", [(k, r) for k in (2.0, 3.0, 4.0) for r in (0.005, 0.01, 0.02)]),
    ):
        for k, r in combos:
            trades = []
            pos = seg = 0
            for df in frames.values():
                table, ts = walk_forward(df, kind, k, r or RISK_DEFAULT)
                trades.extend(ts)
                pos += int((table["expectancy_acct_pct"] > 0).sum())
                seg += int(table["expectancy_acct_pct"].notna().sum())
            pnls = np.array([t.pnl_acct_pct for t in trades], dtype=float)
            if len(pnls) == 0:
                continue
            lo, hi = bootstrap_ci(pnls)
            wins = np.array([t.pnl for t in trades]).clip(min=0).sum()
            losses = -np.array([t.pnl for t in trades]).clip(max=0).sum()
            grid_rows.append({
                "system": kind, "k": k, "risk_pct": r,
                "n_trades": len(pnls), "expectancy_acct_pct": pnls.mean(),
                "ci_low": lo, "ci_high": hi, "median_acct_pct": np.median(pnls),
                "win_rate": (pnls > 0).mean(),
                "profit_factor": wins / losses if losses > 0 else np.inf,
                "positive_segments": pos, "segments": seg,
            })
            label = f"{kind} k={k:g}" + (f" r={r:.1%}" if r else "")
            print(f"  {label:<22}: 交易 {len(pnls):>3} 笔 | 期望 {pnls.mean():+.3%} "
                  f"(95%CI {lo:+.3%}~{hi:+.3%}) | 中位 {np.median(pnls):+.3%} "
                  f"| 胜率 {(pnls > 0).mean():.0%} | 正段 {pos}/{seg}")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    seg_df = pd.concat(
        [tables[n][k] for n in frames for k in ("base", "atrstop", "sized")],
        ignore_index=True,
    )
    seg_df.to_csv(results_dir / "atr_sizing_segments.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(results_dir / "atr_sizing_summary.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(results_dir / "atr_sizing_grid.csv", index=False)
    print(f"\n已保存: results/atr_sizing_segments.csv | atr_sizing_summary.csv | atr_sizing_grid.csv")

    plt = setup_fonts()
    labels = ["BTC", "ETH", "BTC+ETH"]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, (kind, color) in enumerate((
        ("base", "#7f7f7f"), ("atrstop", "#2ca02c"), ("sized", "#ff7f0e"),
    )):
        rows = []
        for label in labels:
            if label == "BTC+ETH":
                rows.append(pooled_stats[kind])
            else:
                rows.append(summaries[label][kind])
        exp = np.array([r["expectancy_acct_pct"] for r in rows]) * 100
        yerr = np.array([
            [(r["expectancy_acct_pct"] - r["ci_low"]) * 100 for r in rows],
            [(r["ci_high"] - r["expectancy_acct_pct"]) * 100 for r in rows],
        ])
        ax.bar(x + (j - 1) * width, exp, width, yerr=yerr, capsize=3, color=color, label=SYSTEMS[kind])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("每笔对账户的数学期望 (%)")
    ax.set_title("三方案单笔期望对比(误差线 = 95% bootstrap 置信区间)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png1 = results_dir / "atr_sizing_expectancy.png"
    fig.savefig(png1, dpi=150)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (name, d) in zip(axes, oos_eq.items()):
        for kind, color, ls in (("base", "#7f7f7f", "--"), ("atrstop", "#2ca02c", "-."), ("sized", "#ff7f0e", "-")):
            eq, _ = d[kind]
            ax.plot(eq.index, eq / eq.iloc[0], label=SYSTEMS[kind], linewidth=1.3, color=color, linestyle=ls)
        _, bench_eq = d["base"]
        ax.plot(bench_eq.index, bench_eq / bench_eq.iloc[0], label="买入持有", linewidth=1.1, color="black", linestyle=":")
        ax.set_yscale("log")
        ax.set_title(f"{name} 样本外净值 (2025-01 起, log)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig2.suptitle("样本外表现: 三方案 vs 买入持有", fontsize=13)
    fig2.tight_layout()
    png2 = results_dir / "atr_sizing_oos_equity.png"
    fig2.savefig(png2, dpi=150)
    print(f"已保存: {png1.name} | {png2.name}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""成交量 + ATR 波动率开仓过滤验证

问题: 原趋势系统在横盘震荡里频繁产生假入场, 开仓即止损, 手续费损耗大。
方案: 只在"入场触发当根"加两道过滤门(离场规则完全不变, 便于公平对比):
  1) 成交量门: Volume >= vol_mult * 过去336根成交量中位数
  2) 波动率门: ATR14/Close >= atr_mult * 过去1440根该比率中位数(不做"死水"行情)
验证口径与之前跨品种验证完全一致:
  - BTC / ETH 各 12 段滚动回测(2021-01 ~ 2026-09), 每段前置 180 天预热;
  - 单笔期望的 95% 置信区间用 5000 次 bootstrap 估计(种子 42);
  - 样本外成绩单(2025-01-01 之后)与参数稳健性网格(3x3)作为辅助证据。
输出:
  results/vol_atr_segments.csv            两策略x两品种分段明细
  results/vol_atr_summary.csv             过滤前后汇总对比(含置信区间)
  results/vol_atr_sensitivity.csv         参数网格
  results/vol_atr_expectancy_compare.png  单笔期望前后对比(带误差线)
  results/vol_atr_oos_equity.png          样本外净值对比
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _lib in (ROOT / "work" / "pylibs", ROOT / "lib"):
    if _lib.exists() and str(_lib) not in sys.path:
        sys.path.insert(0, str(_lib))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mplconfig"))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics, compute_metrics
from config import DEFAULT_CONFIG, PERIODS_PER_YEAR
from strategies.trend_system import sma_gate, sma_gate_filtered

SHORT, LONG, GATE = 48, 336, 1440
STOP_PCT = 0.15
SEGMENT_DAYS = 182
PRIME_DAYS = 180
BOOTSTRAP_N = 5000
SEED = 42
VOL_MULT_DEFAULT = 1.5
ATR_MULT_DEFAULT = 1.0

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


def make_signals(df, filtered, vol_mult=VOL_MULT_DEFAULT, atr_mult=ATR_MULT_DEFAULT):
    if filtered:
        return sma_gate_filtered(
            df, short=SHORT, long=LONG, gate=GATE,
            vol_mult=vol_mult, atr_mult=atr_mult,
        )
    return sma_gate(df, short=SHORT, long=LONG, gate=GATE)


def walk_forward(df, filtered, vol_mult=VOL_MULT_DEFAULT, atr_mult=ATR_MULT_DEFAULT):
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
        signals = make_signals(window, filtered, vol_mult, atr_mult)
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


def oos_summary(df, filtered, vol_mult=VOL_MULT_DEFAULT, atr_mult=ATR_MULT_DEFAULT):
    signals = make_signals(df, filtered, vol_mult, atr_mult)
    test = df[df.index > pd.Timestamp("2024-12-31", tz="UTC")]
    equity, trades, _ = run_backtest(
        test, signals.reindex(test.index), DEFAULT_CONFIG, stop_pct=STOP_PCT
    )
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


def bootstrap_ci(pnls):
    rng = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(pnls, size=len(pnls), replace=True).mean()
        for _ in range(BOOTSTRAP_N)
    ])
    return np.percentile(boot, [2.5, 97.5])


def trade_stats(name, pnls, positive_seg, segments):
    row = {
        "symbol": name,
        "n_trades": len(pnls),
        "positive_segments": positive_seg,
        "segments": segments,
    }
    if len(pnls) == 0:
        row.update({
            "expectancy_pct": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "median_pct": np.nan, "win_rate": np.nan, "profit_factor": np.nan,
        })
        return row
    lo, hi = bootstrap_ci(pnls)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    row.update({
        "expectancy_pct": pnls.mean(),
        "ci_low": lo,
        "ci_high": hi,
        "median_pct": np.median(pnls),
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


def explore(df, name):
    hold = sma_gate(df, short=SHORT, long=LONG, gate=GATE)
    trig = (hold.diff() > 0).fillna(False)
    vol_med = df["Volume"].rolling(336).median()
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = true_range.rolling(14).mean() / df["Close"]
    atr_rel = atr_pct / atr_pct.rolling(1440).median()
    vol_ratio = df["Volume"] / vol_med
    vr = vol_ratio[trig].dropna()
    ar = atr_rel[trig].dropna()
    print(f"  {name}: 历史入场触发共 {int(trig.sum())} 次")
    print(f"    触发时 成交量/近14天中位: 中位 {vr.median():.2f} | 25% {vr.quantile(0.25):.2f} | 75% {vr.quantile(0.75):.2f}")
    print(f"    触发时 ATR%/近60天中位:   中位 {ar.median():.2f} | 25% {ar.quantile(0.25):.2f} | 75% {ar.quantile(0.75):.2f}")
    pv = (vol_ratio[trig] >= 1.5).mean()
    pa = (atr_rel[trig] >= 1.0).mean()
    pb = ((vol_ratio[trig] >= 1.5) & (atr_rel[trig] >= 1.0)).mean()
    print(f"    通过量门(>=1.5x) {pv:.0%} | 通过波动门(>=1.0x) {pa:.0%} | 两门同时通过 {pb:.0%}")


def print_segments(name, table):
    print(f"\n【{name} · 加过滤后 分段明细】")
    for _, r in table.iterrows():
        pf_str = "inf" if np.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        print(
            f"  段{r['segment']:>2} {r['start'].strftime('%Y-%m')}~{r['end'].strftime('%Y-%m')} "
            f"| 交易 {r['n_trades']:>3} 笔 | 期望 {fmt_pct(r['expectancy_pct']):>8} | 胜率 {fmt_pct(r['win_rate']):>7} "
            f"| 盈亏比 {pf_str:>5} | 策略 {fmt_pct(r['strat_return']):>8} | 买入持有 {fmt_pct(r['bench_return']):>8} "
            f"| 回撤 {fmt_pct(r['max_drawdown']):>8}"
        )


def main():
    frames = {name: load_clean(p) for name, p in SYMBOLS.items()}

    print("=" * 100)
    print("第一步 · 摸底: 过去所有入场触发当根, 成交量和波动率处于什么水平")
    for name, df in frames.items():
        explore(df, name)

    print()
    print("=" * 100)
    print("第二步 · 主对比: 原策略 vs 加过滤 (滚动12段 x 两品种, 同成本假设)")
    tables = {}
    per_trade = {}
    summaries = {}
    for name, df in frames.items():
        base_table, base_trades = walk_forward(df, False)
        filt_table, filt_trades = walk_forward(df, True)
        bpnls = np.array([t.pnl_pct for t in base_trades], dtype=float)
        fpnls = np.array([t.pnl_pct for t in filt_trades], dtype=float)
        print(f"\n【{name} · 原策略】(与之前同口径重算) 交易 {len(bpnls)} 笔, "
              f"合并期望 {bpnls.mean():+.3%}")
        print_segments(name, filt_table)
        tables[name] = {
            "baseline": base_table.assign(symbol=name, strategy="baseline"),
            "filtered": filt_table.assign(symbol=name, strategy="filtered"),
        }
        per_trade[name] = {"baseline": bpnls, "filtered": fpnls}
        summaries[name] = {
            "baseline": trade_stats(
                name, bpnls,
                int((base_table["expectancy_pct"] > 0).sum()),
                int(base_table["expectancy_pct"].notna().sum()),
            ),
            "filtered": trade_stats(
                name, fpnls,
                int((filt_table["expectancy_pct"] > 0).sum()),
                int(filt_table["expectancy_pct"].notna().sum()),
            ),
        }

    print()
    print("=" * 100)
    print("第三步 · 过滤前后统计对比 (bootstrap 95% 置信区间, 种子42)")
    summary_by_strategy = {}
    summary_rows = []
    for label in ("BTC", "ETH", "BTC+ETH"):
        if label == "BTC+ETH":
            bpnls = np.concatenate([per_trade[s]["baseline"] for s in frames])
            fpnls = np.concatenate([per_trade[s]["filtered"] for s in frames])
            b = trade_stats(label, bpnls,
                            sum(summaries[s]["baseline"]["positive_segments"] for s in frames),
                            sum(summaries[s]["baseline"]["segments"] for s in frames))
            f = trade_stats(label, fpnls,
                            sum(summaries[s]["filtered"]["positive_segments"] for s in frames),
                            sum(summaries[s]["filtered"]["segments"] for s in frames))
        else:
            b = summaries[label]["baseline"]
            f = summaries[label]["filtered"]
        summary_by_strategy[label] = {"baseline": b, "filtered": f}
        for key, row in (("baseline", b), ("filtered", f)):
            summary_rows.append({"strategy": key, **row})
        print(f"  {label:>7}: 交易 {b['n_trades']:>3} -> {f['n_trades']:<3} 笔 "
              f"| 单笔期望 {b['expectancy_pct']:+.2%} -> {f['expectancy_pct']:+.2%}")
        print(f"          95%CI [{b['ci_low']:+.2%}, {b['ci_high']:+.2%}] -> "
              f"[{f['ci_low']:+.2%}, {f['ci_high']:+.2%}] | 中位 {b['median_pct']:+.2%} -> {f['median_pct']:+.2%} "
              f"| 胜率 {b['win_rate']:.1%} -> {f['win_rate']:.1%} "
              f"| 正段 {b['positive_segments']} -> {f['positive_segments']}/{f['segments']}")

    print()
    print("=" * 100)
    print("第四步 · 样本外成绩单 (2025-01-01 ~ 2026-09-07, 该时段参数从未参与选择)")
    oos_eq = {}
    oos_rows = []
    for name, df in frames.items():
        oos_eq[name] = {}
        for key, flag in (("baseline", False), ("filtered", True)):
            s, eq, bench_eq = oos_summary(df, flag)
            oos_eq[name][key] = (eq, bench_eq)
            oos_rows.append({"symbol": name, "strategy": key, **s})
            pf_str = "inf" if np.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
            print(f"  {name} {key:>8}: 交易 {s['n_trades']:>3} 笔 | 单笔期望 {s['expectancy_pct']:+.2%} "
                  f"| 策略收益 {s['strat_return']:+.2%} | 最大回撤 {s['max_drawdown']:.2%} "
                  f"| 买入持有 {s['bench_return']:+.2%} (回撤 {s['bench_drawdown']:.2%})")

    print()
    print("=" * 100)
    print("第五步 · 参数稳健性网格 (BTC+ETH 合并, 9 组合 x 完整滚动12段)")
    grid_rows = []
    for vol_mult in (1.25, 1.5, 2.0):
        for atr_mult in (1.0, 1.25, 1.5):
            all_pnls = []
            pos_seg = seg_cnt = 0
            for df in frames.values():
                table, trades = walk_forward(df, True, vol_mult, atr_mult)
                all_pnls.extend(t.pnl_pct for t in trades)
                pos_seg += int((table["expectancy_pct"] > 0).sum())
                seg_cnt += int(table["expectancy_pct"].notna().sum())
            pnls = np.array(all_pnls, dtype=float)
            if len(pnls) == 0:
                continue
            lo, hi = bootstrap_ci(pnls)
            wins = pnls[pnls > 0].sum()
            losses = -pnls[pnls < 0].sum()
            grid_rows.append({
                "vol_mult": vol_mult, "atr_mult": atr_mult,
                "n_trades": len(pnls), "expectancy_pct": pnls.mean(),
                "ci_low": lo, "ci_high": hi, "median_pct": np.median(pnls),
                "win_rate": (pnls > 0).mean(),
                "profit_factor": wins / losses if losses > 0 else np.inf,
                "positive_segments": pos_seg, "segments": seg_cnt,
            })
            print(f"  量门>={vol_mult:.2f}x, 波动门>={atr_mult:.2f}x: 交易 {len(pnls):>3} 笔 "
                  f"| 期望 {pnls.mean():+.2%} (95%CI {lo:+.2%}~{hi:+.2%}) | 中位 {np.median(pnls):+.2%} "
                  f"| 胜率 {(pnls > 0).mean():.0%} | 正段 {pos_seg}/{seg_cnt}")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    seg_df = pd.concat(
        [tables[n][k] for n in frames for k in ("baseline", "filtered")],
        ignore_index=True,
    )
    seg_df.to_csv(results_dir / "vol_atr_segments.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(results_dir / "vol_atr_summary.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(results_dir / "vol_atr_sensitivity.csv", index=False)
    print(f"\n已保存: results/vol_atr_segments.csv | vol_atr_summary.csv | vol_atr_sensitivity.csv")

    plt = setup_fonts()
    labels = ["BTC", "ETH", "BTC+ETH"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for off, key, color, label_cn in (
        (-width / 2, "baseline", "#7f7f7f", "原策略"),
        (width / 2, "filtered", "#ff7f0e", "加量+波动过滤"),
    ):
        rows = [summary_by_strategy[l][key] for l in labels]
        exp = np.array([r["expectancy_pct"] for r in rows]) * 100
        yerr = np.array([
            [(r["expectancy_pct"] - r["ci_low"]) * 100 for r in rows],
            [(r["ci_high"] - r["expectancy_pct"]) * 100 for r in rows],
        ])
        ax.bar(x + off, exp, width, yerr=yerr, capsize=4, color=color, label=label_cn)
        for xi, r in zip(x + off, rows):
            ax.annotate(f"{r['n_trades']}笔", (xi, exp[int((xi - off) / 1.0)]), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("单笔数学期望 (%)")
    ax.set_title("过滤前后单笔数学期望对比(误差线 = 95% bootstrap 置信区间)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png1 = results_dir / "vol_atr_expectancy_compare.png"
    fig.savefig(png1, dpi=150)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (name, d) in zip(axes, oos_eq.items()):
        eq, bench_eq = d["baseline"]
        eqf, _ = d["filtered"]
        ax.plot(eq.index, eq / eq.iloc[0], label="原策略", linewidth=1.2, color="#7f7f7f", linestyle="--")
        ax.plot(eqf.index, eqf / eqf.iloc[0], label="加过滤", linewidth=1.4, color="#ff7f0e")
        ax.plot(bench_eq.index, bench_eq / bench_eq.iloc[0], label="买入持有", linewidth=1.2, color="black", linestyle=":")
        ax.set_yscale("log")
        ax.set_title(f"{name} 样本外净值 (2025-01 起, log)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig2.suptitle("样本外表现: 原策略 vs 加过滤 vs 买入持有", fontsize=13)
    fig2.tight_layout()
    png2 = results_dir / "vol_atr_oos_equity.png"
    fig2.savefig(png2, dpi=150)
    print(f"已保存: {png1.name} | {png2.name}")


if __name__ == "__main__":
    main()
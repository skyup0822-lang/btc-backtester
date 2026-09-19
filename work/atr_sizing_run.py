# -*- coding: utf-8 -*-
"""ATR 止损 + 波动率反向仓位验证 (纯 numpy/pandas 版, 图表用自绘 HTML/SVG)

三个方案同台对比(离场/入场规则只有止损和仓位不同):
  方案1 base   : 原趋势系统 + 固定 15% 止损 + 全仓
  方案2 atrstop: 原趋势系统 + k*ATR14 止损 + 全仓     (只换"尺子")
  方案3 sized  : 原趋势系统 + k*ATR14 止损 + 每笔风险 r% (波动大自动下小注)
指标统一用"每笔盈亏 / 入场时账户权益"做期望, 三种仓位方案才能公平比较。
验证口径: BTC/ETH 各 12 段滚动回测(2021-01~2026-09) + 5000 bootstrap
+ 样本外成绩单(2025-01+) + 参数稳健性网格。
输出:
  results/atr_sizing_segments.csv   三方案x两品种分段明细
  results/atr_sizing_summary.csv    三方案汇总对比(含置信区间)
  results/atr_sizing_grid.csv       参数网格
  results/atr_sizing_report.html    自绘 SVG 报告(双击浏览器打开)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    "atrstop": "ATR止损(3xATR)",
    "sized": "ATR止损+仓位(3xATR, 每笔风险1%)",
}
COLORS = {"base": "#7f7f7f", "atrstop": "#2ca02c", "sized": "#ff7f0e"}


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


# ---------------- 自绘 SVG 图表 ----------------

def svg_bar_chart(data, width=880, height=420):
    """data: list of (label, [(system, exp%, ci_low%, ci_high%), ...])"""
    systems = [SYSTEMS[k] for k in ("base", "atrstop", "sized")]
    lo_all = min(v for _, _, lo, _ in [item for grp in data for item in grp[1]] for v in [lo])
    hi_all = max(v for _, _, _, hi in [item for grp in data for item in grp[1]] for v in [hi])
    ymin = min(lo_all, -0.1)
    ymax = max(hi_all, 0.1)
    rng = ymax - ymin
    step = 0.5 if rng <= 3.5 else 1.0
    ymin = np.floor(ymin / step) * step
    ymax = np.ceil(ymax / step) * step
    left, top, right, bottom = 60, 40, width - 20, height - 60
    plot_w, plot_h = right - left, bottom - top

    def y(v):
        return top + (ymax - v) / (ymax - ymin) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;font-family:Microsoft YaHei,SimHei,sans-serif;">']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ccc"/>')
    for tick in np.arange(ymin, ymax + 1e-9, step):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e0e0e0"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" font-size="12" fill="#555" text-anchor="end">{tick:g}%</text>')
    yy0 = y(0)
    parts.append(f'<line x1="{left}" y1="{yy0:.1f}" x2="{right}" y2="{yy0:.1f}" stroke="#333" stroke-width="1.2"/>')
    n_groups = len(data)
    group_w = plot_w / n_groups
    bar_w = min(group_w * 0.22, 52)
    for gi, (label, items) in enumerate(data):
        cx = left + group_w * (gi + 0.5)
        parts.append(f'<text x="{cx:.1f}" y="{bottom + 24}" font-size="13" fill="#333" text-anchor="middle">{label}</text>')
        for si, (name, exp, lo, hi) in enumerate(items):
            bx = cx + (si - 1) * bar_w
            yy_exp = y(exp)
            ytop = min(yy_exp, yy0)
            hgt = abs(yy_exp - yy0)
            parts.append(f'<rect x="{bx:.1f}" y="{ytop:.1f}" width="{bar_w * 0.8:.1f}" height="{hgt:.1f}" '
                         f'fill="{COLORS[{"base": "base", "atrstop": "atrstop", "sized": "sized"}[name]] if False else ["#7f7f7f","#2ca02c","#ff7f0e"][si]}" '
                         f'stroke="none"/>')
            mx = bx + bar_w * 0.4
            parts.append(f'<line x1="{mx:.1f}" y1="{y(lo):.1f}" x2="{mx:.1f}" y2="{y(hi):.1f}" stroke="#333" stroke-width="1.5"/>')
            parts.append(f'<line x1="{mx - 4:.1f}" y1="{y(lo):.1f}" x2="{mx + 4:.1f}" y2="{y(lo):.1f}" stroke="#333" stroke-width="1.5"/>')
            parts.append(f'<line x1="{mx - 4:.1f}" y1="{y(hi):.1f}" x2="{mx + 4:.1f}" y2="{y(hi):.1f}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" font-size="13" fill="#333" text-anchor="middle">'
                 '每笔对账户的数学期望(误差线 = 95% bootstrap 置信区间)</text>')
    lx = left + 12
    for si, name in enumerate(("base", "atrstop", "sized")):
        parts.append(f'<rect x="{lx}" y="{height - 34}" width="14" height="14" fill="{["#7f7f7f","#2ca02c","#ff7f0e"][si]}"/>')
        parts.append(f'<text x="{lx + 20}" y="{height - 22}" font-size="12" fill="#333">{systems[si]}</text>')
        lx += 20 + 12 * len(systems[si]) + 28
    parts.append("</svg>")
    return "".join(parts)


def svg_line_chart(series_list, title, width=860, height=380):
    """series_list: list of (name, color, dash, daily index, values)"""
    vals = np.concatenate([v for *_, v in series_list])
    vmin, vmax = vals.min(), vals.max()
    vmin = min(vmin, 0.75)
    vmax = max(vmax, 1.25)
    pad = 0.05
    lo, hi = vmin / (1 + pad * 0), vmax
    lo = min(lo, 0.7)
    ticks = [t for t in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6) if lo <= t <= hi * 1.01]
    if not ticks:
        ticks = [round(lo + (hi - lo) * i / 4, 2) for i in range(5)]
    left, top, right, bottom = 56, 36, width - 16, height - 52
    plot_w, plot_h = right - left, bottom - top

    def y(v):
        return top + (np.log10(hi) - np.log10(v)) / (np.log10(hi) - np.log10(lo)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;font-family:Microsoft YaHei,SimHei,sans-serif;">']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ccc"/>')
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e0e0e0"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" font-size="11" fill="#555" text-anchor="end">{tick:g}</text>')
    all_days = series_list[0][3]
    for si, (name, color, dash, days, values) in enumerate(series_list):
        xs = left + np.arange(len(days)) / max(len(days) - 1, 1) * plot_w
        pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{1.6 if si == 2 else 1.3}"{dash_attr}/>')
    n_ticks = 5
    for i in range(n_ticks + 1):
        idx = int(round(len(all_days) - 1) * i / n_ticks)
        xx = left + plot_w * i / n_ticks
        parts.append(f'<line x1="{xx:.1f}" y1="{bottom}" x2="{xx:.1f}" y2="{bottom + 5}" stroke="#999"/>')
        parts.append(f'<text x="{xx:.1f}" y="{bottom + 20}" font-size="11" fill="#555" text-anchor="middle">'
                     f'{all_days[idx].strftime("%Y-%m")}</text>')
    lx = left + 10
    for si, (name, color, dash, days, values) in enumerate(series_list):
        parts.append(f'<line x1="{lx}" y1="{bottom + 40}" x2="{lx + 22}" y2="{bottom + 40}" stroke="{color}" '
                     f'stroke-width="2" stroke-dasharray="{dash or ""}"/>')
        parts.append(f'<text x="{lx + 28}" y="{bottom + 44}" font-size="12" fill="#333">{name}</text>')
        lx += 28 + 12 * len(name) + 30
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" font-size="13" fill="#333" text-anchor="middle">{title}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_html(pooled_stats, summaries, oos_eq, oos_rows, grid_rows):
    sys_names = [SYSTEMS[k] for k in ("base", "atrstop", "sized")]
    rows1 = []
    for k in ("base", "atrstop", "sized"):
        s = pooled_stats[k]
        rows1.append(
            f"<tr><td>{SYSTEMS[k]}</td><td>{s['n_trades']}</td>"
            f"<td>{s['expectancy_acct_pct'] * 100:+.3f}%</td>"
            f"<td>[{s['ci_low'] * 100:+.3f}%, {s['ci_high'] * 100:+.3f}%]</td>"
            f"<td>{s['median_acct_pct'] * 100:+.3f}%</td>"
            f"<td>{s['win_rate']:.1%}</td><td>{s['positive_segments']}/{s['segments']}</td></tr>"
        )
    rows2 = []
    for _, r in oos_rows.iterrows():
        rows2.append(
            f"<tr><td>{r['symbol']}</td><td>{SYSTEMS[r['system']]}</td><td>{r['n_trades']}</td>"
            f"<td>{r['strat_return'] * 100:+.2f}%</td><td>{r['max_drawdown'] * 100:.2f}%</td>"
            f"<td>{r['bench_return'] * 100:+.2f}%</td></tr>"
        )
    rows3 = []
    for _, g in grid_rows.iterrows():
        label = f"{'ATR止损' if g['system'] == 'atrstop' else '止损+仓位'} k={g['k']:g}"
        if pd.notna(g["risk_pct"]):
            label += f" r={g['risk_pct']:.1%}"
        rows3.append(
            f"<tr><td>{label}</td><td>{g['n_trades']}</td>"
            f"<td>{g['expectancy_acct_pct'] * 100:+.3f}%</td>"
            f"<td>[{g['ci_low'] * 100:+.3f}%, {g['ci_high'] * 100:+.3f}%]</td>"
            f"<td>{g['median_acct_pct'] * 100:+.3f}%</td><td>{g['win_rate']:.0%}</td>"
            f"<td>{g['positive_segments']}/{g['segments']}</td></tr>"
        )
    chart1_data = []
    for label in ("BTC", "ETH", "BTC+ETH"):
        items = []
        for k in ("base", "atrstop", "sized"):
            s = pooled_stats[k] if label == "BTC+ETH" else summaries[label][k]
            items.append((k, s["expectancy_acct_pct"] * 100, s["ci_low"] * 100, s["ci_high"] * 100))
        chart1_data.append((label, items))
    charts = [svg_bar_chart(chart1_data)]
    for name in ("BTC", "ETH"):
        series = []
        for k in ("base", "atrstop", "sized"):
            eq, _ = oos_eq[name][k]
            daily = eq.resample("1D").last().dropna()
            vals = (daily / daily.iloc[0]).to_numpy()
            series.append((SYSTEMS[k], COLORS[k], None, daily.index, vals))
        _, bench_eq = oos_eq[name]["base"]
        daily_b = bench_eq.resample("1D").last().dropna()
        series.append(("买入持有", "#000000", "3,3", daily_b.index, (daily_b / daily_b.iloc[0]).to_numpy()))
        charts.append(svg_line_chart(series, f"{name} 样本外净值 (2025-01 起, log 坐标)"))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>ATR止损+波动率仓位验证报告</title>
<style>
body{{font-family:"Microsoft YaHei",SimHei,sans-serif;max-width:960px;margin:0 auto;padding:24px;color:#222;background:#fff}}
h1{{font-size:22px;border-bottom:3px solid #ff7f0e;padding-bottom:8px}}
h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border:1px solid #ccc;padding:6px 8px;text-align:right}}
th{{background:#f4f4f4}}
td:first-child,th:first-child{{text-align:left}}
.card{{border:1px solid #e0e0e0;border-radius:8px;padding:14px;margin:14px 0;background:#fafafa}}
.note{{font-size:12px;color:#666;line-height:1.7}}
.good{{color:#1a7f37;font-weight:bold}}
.bad{{color:#c62828;font-weight:bold}}
</style></head><body>
<h1>ATR 止损 + 波动率反向仓位 · 验证报告</h1>
<div class="card"><b>口径:</b> BTC/ETH 1小时K线, 2021-01 ~ 2026-09 各12段滚动回测(每段前置180天预热),
成本 0.10%/边手续费 + 0.05%/边滑点, 期望统一按"每笔盈亏/入场时账户权益"计算,
置信区间为 5000 次 bootstrap(种子42)。</div>
<h2>一、三方案全历史统计 (BTC+ETH 合并)</h2>
<table><tr><th>方案</th><th>交易数</th><th>每笔期望</th><th>95% 置信区间</th><th>中位数</th><th>胜率</th><th>正期望段数</th></tr>
{''.join(rows1)}</table>
{charts[0]}
<h2>二、样本外成绩单 (2025-01-01 ~ 2026-09-07)</h2>
<table><tr><th>品种</th><th>方案</th><th>交易数</th><th>策略收益</th><th>最大回撤</th><th>买入持有收益</th></tr>
{''.join(rows2)}</table>
{charts[1]}
{charts[2]}
<h2>三、参数稳健性网格 (BTC+ETH 合并)</h2>
<table><tr><th>参数</th><th>交易数</th><th>每笔期望</th><th>95% 置信区间</th><th>中位数</th><th>胜率</th><th>正期望段数</th></tr>
{''.join(rows3)}</table>
<div class="note"><b>读图提示:</b> 置信区间全部横跨 0 说明"该设置下不能从统计上排除亏钱可能";
区间下界越大越可靠。"正期望段数"是 24 个半年段中该方案期望为正的段数。<br>
<b>本报告仅为历史回测结果, 不构成投资建议; 历史表现不代表未来收益。</b></div>
</body></html>"""
    return html


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
            print(f"  {SYSTEMS[kind]:<26}: 交易 {s['n_trades']:>3} 笔 "
                  f"| 每笔期望(账户%) {s['expectancy_acct_pct']:+.3%} "
                  f"| 95%CI [{s['ci_low']:+.3%}, {s['ci_high']:+.3%}] "
                  f"| 胜率 {s['win_rate']:.1%} | 正段 {s['positive_segments']}/{s['segments']}")
        print(f"\n【{name} · 方案3({SYSTEMS['sized']})分段明细】")
        for _, r in tables[name]["sized"].iterrows():
            pf_str = "inf" if np.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
            print(f"  段{r['segment']:>2} {r['start'].strftime('%Y-%m')}~{r['end'].strftime('%Y-%m')} "
                  f"| 交易 {r['n_trades']:>3} 笔 | 期望 {fmt_pct(r['expectancy_acct_pct']):>8} | 胜率 {fmt_pct(r['win_rate']):>7} "
                  f"| 盈亏比 {pf_str:>5} | 策略 {fmt_pct(r['strat_return']):>8} | 买入持有 {fmt_pct(r['bench_return']):>8} "
                  f"| 回撤 {fmt_pct(r['max_drawdown']):>8}")

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
        print(f"  {SYSTEMS[kind]:<26}: 交易 {s['n_trades']:>3} 笔 "
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
            print(f"  {name} {SYSTEMS[kind]:<26}: 交易 {s['n_trades']:>3} 笔 "
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
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(results_dir / "atr_sizing_grid.csv", index=False)
    print(f"\n已保存: results/atr_sizing_segments.csv | atr_sizing_summary.csv | atr_sizing_grid.csv")

    html = build_html(pooled_stats, summaries, oos_eq, pd.DataFrame(oos_rows), grid_df)
    html_path = results_dir / "atr_sizing_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"已保存: {html_path.name}")


if __name__ == "__main__":
    main()
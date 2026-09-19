"""滚动窗口(walk-forward)验证。

目的:检验"趋势系统 48/336 + 60日闸门 + 15%止损"这套固定参数,
在历史上不同时间段的表现是否稳定(重点看单笔数学期望的稳定性)。

方法:从 2021-01-01 起,把数据切成每段约 6 个月,每段向前多取 180 天做指标预热,
逐段独立回测,汇总全部段落的单笔期望,并用 bootstrap 估计 95% 置信区间。
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
from config import DATA_FILE, DEFAULT_CONFIG
from strategies.trend_system import sma_gate

SHORT, LONG, GATE = 48, 336, 1440
STOP_PCT = 0.15
SEGMENT_DAYS = 182
PRIME_DAYS = 180
BOOTSTRAP_N = 5000
SEED = 42


def fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v * 100:+.2f}%"


def main() -> None:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = df.index[-1]

    rows = []
    all_trades = []
    seg_start = start
    seg_id = 1
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

    table = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "walk_forward.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print("=" * 92)
    print("滚动窗口验证: 趋势系统 48/336 + 60日闸门 + 15%止损 | 每段约 6 个月 | 预热 180 天")
    print("=" * 92)
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
    print()

    pnls = np.array([t.pnl_pct for t in all_trades], dtype=float)
    if len(pnls) > 0:
        rng = np.random.default_rng(SEED)
        boot = np.array([
            rng.choice(pnls, size=len(pnls), replace=True).mean()
            for _ in range(BOOTSTRAP_N)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pos_seg = int(table["expectancy_pct"].gt(0).sum())
        total_seg = int(table["expectancy_pct"].notna().sum())
        wins = pnls[pnls > 0].sum()
        losses = -pnls[pnls < 0].sum()
        print("【汇总统计(全部段落,共 5.5 年)】")
        print(f"  交易总数        : {len(pnls)}")
        print(f"  单笔数学期望    : {pnls.mean():+.3%}  (bootstrap 95% 置信区间: {lo:+.3%} ~ {hi:+.3%})")
        print(f"  单笔期望中位数  : {np.median(pnls):+.3%} | 标准差 {pnls.std():.3%}")
        print(f"  胜率            : {(pnls > 0).mean():.1%} | 盈亏比 {wins / losses if losses > 0 else float('inf'):.2f}")
        print(f"  最差单笔        : {pnls.min():+.3%} | 最好单笔 {pnls.max():+.3%}")
        print(f"  段落稳定性      : {pos_seg}/{total_seg} 个段落单笔期望为正")
        print()
        print(f"结果已保存: {out_csv}")
    else:
        print("没有产生交易。")


if __name__ == "__main__":
    main()
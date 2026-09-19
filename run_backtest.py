"""主入口:加载数据 -> 训练/测试/全周期切分 -> 回测所有策略 -> 输出绩效表与净值曲线图。

用法(在项目根目录 btc-backtester 下):
    python run_backtest.py
"""
import os
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mplconfig"))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics, compute_metrics
from config import DATA_FILE, DEFAULT_CONFIG, PERIODS_PER_YEAR, RESULTS_DIR, TRAIN_END
from strategies import STRATEGIES, STRATEGY_EXTRA

METRIC_LABELS = {
    "total_return": "总收益",
    "cagr": "年化收益",
    "max_drawdown": "最大回撤",
    "sharpe": "夏普比率",
    "n_trades": "交易次数",
    "win_rate": "胜率",
    "profit_factor": "盈亏比",
    "expectancy_pct": "单笔期望",
    "avg_holding_bars": "平均持仓(h)",
}

PCT_KEYS = {"total_return", "cagr", "max_drawdown", "win_rate", "expectancy_pct", "benchmark_return"}


def pct(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: "-" if pd.isna(v) else f"{v * 100:.2f}%")


def num(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: "-" if pd.isna(v) else f"{v:.2f}")


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        sys.exit(f"未找到数据文件 {DATA_FILE},请先运行: python fetch_data.py")
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    return df


def evaluate(df: pd.DataFrame, signals: pd.Series, extra: dict):
    equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, **extra)
    metrics = compute_metrics(equity, trades, PERIODS_PER_YEAR)
    bench, bench_equity = buy_hold_metrics(df, DEFAULT_CONFIG)
    metrics["benchmark_return"] = bench["total_return"]
    return metrics, equity, bench_equity


def main() -> None:
    df = load_data()
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    train = df[df.index <= train_end]
    test = df[df.index > train_end]
    print(f"数据: {df.index[0]} ~ {df.index[-1]},共 {len(df)} 根 1h K 线")
    print(f"训练期(样本内): {train.index[0]} ~ {train.index[-1]}, {len(train)} 根")
    print(f"测试期(样本外): {test.index[0]} ~ {test.index[-1]}, {len(test)} 根")
    print(f"成本假设: 手续费 {DEFAULT_CONFIG.fee_rate:.2%}/边, 滑点 {DEFAULT_CONFIG.slippage:.2%}/边, 初始资金 {DEFAULT_CONFIG.initial_capital:,.0f} USDT")
    print(f"测试期 BTC: 最高 {test['High'].max():,.0f}, 最低 {test['Low'].min():,.0f}, 期末 {test['Close'].iloc[-1]:,.0f}\n")

    results = []
    curves = {}
    bench_curve = None
    for name, (label, fn) in STRATEGIES.items():
        signals = fn(df)
        extra = STRATEGY_EXTRA.get(name, {})
        for split_name, part in (("train", train), ("test", test), ("full", df)):
            metrics, equity, bench_equity = evaluate(part, signals.reindex(part.index), extra)
            results.append({"strategy": name, "label": label, "split": split_name, **metrics})
            if split_name == "test":
                curves[name] = (label, equity)
                bench_curve = bench_equity
    table = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "btc_backtest_results.csv", index=False)

    for split_name, title in (
        ("train", "样本内(训练期,用于调参)"),
        ("test", "样本外(测试期,只用于验证)"),
        ("full", "全周期"),
    ):
        sub = table[table["split"] == split_name].set_index("label")
        cols = {}
        for key, label in METRIC_LABELS.items():
            if key in PCT_KEYS:
                cols[label] = pct(sub[key])
            elif key == "n_trades":
                cols[label] = sub[key].round(0).astype("Int64").apply(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
            else:
                cols[label] = num(sub[key])
        cols["同期基准收益"] = pct(sub["benchmark_return"])
        print(f"===== {title} =====")
        print(pd.DataFrame(cols).to_string())
        print()

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

    fig, ax = plt.subplots(figsize=(12, 7))
    for name, (label, equity) in curves.items():
        ax.plot(equity.index, equity / equity.iloc[0], label=label, linewidth=1.2)
    if bench_curve is not None:
        ax.plot(bench_curve.index, bench_curve / bench_curve.iloc[0],
                label="买入持有(基准)", color="black", linestyle="--", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_title("样本外(测试期)净值曲线对比(log 坐标,初始资金归一化为 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curves.png", dpi=150)
    print(f"已保存: {RESULTS_DIR / 'btc_backtest_results.csv'}")
    print(f"已保存: {RESULTS_DIR / 'equity_curves.png'}")


if __name__ == "__main__":
    main()
"""回测绩效统计指标。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 365


def compute_metrics(equity: pd.Series, trades: list, periods_per_year: int) -> dict:
    """由净值曲线与交易记录计算常见绩效指标。"""
    metrics = {}
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    metrics["total_return"] = total_return
    years = len(equity) / periods_per_year
    metrics["cagr"] = (
        (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    metrics["max_drawdown"] = drawdown.min()
    daily = equity.resample("1D").last().dropna().pct_change().dropna()
    if len(daily) > 30 and daily.std() > 0:
        metrics["sharpe"] = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
        metrics["volatility"] = float(daily.std() * np.sqrt(TRADING_DAYS))
    else:
        metrics["sharpe"] = np.nan
        metrics["volatility"] = np.nan
    pnls = np.array([t.pnl for t in trades], dtype=float)
    pnl_pcts = np.array([t.pnl_pct for t in trades], dtype=float)
    metrics["n_trades"] = len(pnls)
    if len(pnls) > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        metrics["win_rate"] = len(wins) / len(pnls)
        metrics["profit_factor"] = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
        metrics["expectancy_pct"] = pnl_pcts.mean()  # 单笔数学期望(占投入本金的百分比)
        metrics["avg_holding_bars"] = np.mean([t.holding_bars for t in trades])
    else:
        metrics["win_rate"] = np.nan
        metrics["profit_factor"] = np.nan
        metrics["expectancy_pct"] = np.nan
        metrics["avg_holding_bars"] = np.nan
    return metrics


def buy_hold_metrics(df: pd.DataFrame, config) -> tuple:
    """买入持有基准:期初全仓买入、期末卖出,计入一次完整往返成本。"""
    equity = df["Close"] / df["Close"].iloc[0] * config.initial_capital
    equity = equity * (1 - config.fee_rate - config.slippage) ** 2
    return compute_metrics(equity, [], 365 * 24), equity
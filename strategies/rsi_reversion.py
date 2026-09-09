"""RSI 均值回归策略:超卖买入,超买卖出(只做多)。"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def rsi_reversion(
    df: pd.DataFrame,
    period: int = 14,
    buy_below: float = 30.0,
    sell_above: float = 65.0,
) -> pd.Series:
    rsi = _rsi(df["Close"], period).to_numpy()
    pos = np.zeros(len(df))
    current = 0.0
    for i, value in enumerate(rsi):
        if np.isnan(value):
            pos[i] = 0.0
            current = 0.0
            continue
        if current == 0.0 and value < buy_below:
            current = 1.0
        elif current == 1.0 and value > sell_above:
            current = 0.0
        pos[i] = current
    return pd.Series(pos, index=df.index)
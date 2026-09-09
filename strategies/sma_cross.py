"""双均线趋势策略:快线上穿慢线后持有,下穿后空仓。"""
import pandas as pd


def sma_cross(df: pd.DataFrame, short: int = 24, long: int = 72) -> pd.Series:
    fast = df["Close"].rolling(short).mean()
    slow = df["Close"].rolling(long).mean()
    signal = (fast > slow).astype(float)
    return signal.fillna(0.0)
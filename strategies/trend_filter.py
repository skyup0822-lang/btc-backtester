"""趋势过滤策略:收盘价在 SMA 之上持有,之下空仓。"""
import pandas as pd


def trend_filter(df: pd.DataFrame, window: int = 168) -> pd.Series:
    sma = df["Close"].rolling(window).mean()
    signal = (df["Close"] > sma).astype(float)
    return signal.fillna(0.0)
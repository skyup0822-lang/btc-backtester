"""基准策略:始终满仓持有。"""
import pandas as pd


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)
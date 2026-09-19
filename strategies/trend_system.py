"""趋势系统策略:双均线交叉 + 长周期趋势闸门,两者同时满足才持仓。

短周期均线决定进出场时机,长周期闸门过滤掉大级别下跌行情中的反复震荡。

sma_gate_filtered 在 sma_gate 的入场规则上增加两道"开仓过滤门",离场规则完全不变:
  1) 成交量门: 入场触发当根 Volume >= vol_mult * 过去 vol_window 根成交量中位数;
  2) 波动率门: 入场触发当根 ATR(atr_window)/Close >= atr_mult * 过去 atr_base 根
     该比率的中位数,即"当前波动必须达到近期常态水平以上,死水行情不开仓"。
若触发当根任一扇门未通过,该次入场作废,继续空仓等待下一次触发。
"""
import numpy as np
import pandas as pd

from .sma_cross import sma_cross
from .trend_filter import trend_filter


def sma_gate(df: pd.DataFrame, short: int = 48, long: int = 336, gate: int = 1440) -> pd.Series:
    base = sma_cross(df, short, long)
    gate_sig = trend_filter(df, gate)
    return (base * gate_sig).astype(float).fillna(0.0)


def sma_gate_filtered(
    df: pd.DataFrame,
    short: int = 48,
    long: int = 336,
    gate: int = 1440,
    vol_window: int = 336,
    vol_mult: float = 1.5,
    atr_window: int = 14,
    atr_base: int = 1440,
    atr_mult: float = 1.0,
) -> pd.Series:
    """带成交量 + ATR 波动率开仓过滤的趋势系统信号(0=空仓, 1=持仓)。"""
    hold = sma_gate(df, short, long, gate)
    trigger = (hold.diff() > 0).fillna(False)

    vol_med = df["Volume"].rolling(vol_window).median()
    vol_ok = (df["Volume"] >= vol_mult * vol_med).fillna(False)

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = true_range.rolling(atr_window).mean() / df["Close"]
    atr_ref = atr_pct.rolling(atr_base).median()
    atr_ok = (atr_pct >= atr_mult * atr_ref).fillna(False)

    entry = (trigger & vol_ok & atr_ok).to_numpy()
    hold_np = hold.to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0.0
    for i in range(len(df)):
        if hold_np[i] == 0.0:
            state = 0.0
        elif entry[i]:
            state = 1.0
        out[i] = state
    return pd.Series(out, index=df.index)
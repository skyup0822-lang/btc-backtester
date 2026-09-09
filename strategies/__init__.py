"""策略注册表:每个策略接收 OHLCV DataFrame,返回 0/1 信号序列(1=持有,0=空仓)。

参数基于 2019-11 ~ 2024-12 训练期选定,测试期(2025-01 起)仅作验证。
"""
from .buy_hold import buy_hold
from .sma_cross import sma_cross
from .trend_filter import trend_filter
from .trend_system import sma_gate

STRATEGIES = {
    "buy_hold": ("买入持有(基准)", lambda df: buy_hold(df)),
    "sma_48_336": ("双均线 48/336", lambda df: sma_cross(df, short=48, long=336)),
    "trend_1440": ("趋势过滤 收盘>SMA1440(60日)", lambda df: trend_filter(df, window=1440)),
    "trend_system_48_336_1440": ("趋势系统 48/336 + 60日闸门", lambda df: sma_gate(df, short=48, long=336, gate=1440)),
}

# 各策略的回测附加参数(止损等),同样在训练期选定
STRATEGY_EXTRA = {
    "buy_hold": {},
    "sma_48_336": {"stop_pct": 0.15},
    "trend_1440": {"stop_pct": 0.15},
    "trend_system_48_336_1440": {"stop_pct": 0.15},
}
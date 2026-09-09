"""逐根 K 线撮合的回测引擎(现货、只做多)。

撮合规则(避免"未来函数",即用尚未发生的信息交易):
- 策略在第 t 根 K 线收盘后产生信号;
- 引擎在第 t+1 根 K 线开盘价成交;
- 每笔交易计入手续费与滑点;
- 止损方式三选一:
  * stop_pct      : 固定止损(相对买入价的百分比);
  * trail_pct     : 跟踪止损(只做多:止损位只上移不下移);
  * atr_stop_mult : 入场时用信号当根的 ATR 定止损距离(止损价 = 买价 - k*ATR);
- risk_pct 为 None 时全仓进出; 传入 risk_pct 时启用"风险仓位":
  每笔投入 = 账户 * risk_pct / 止损距离, 无论止损距离多大, 止损触发最多亏账户的 risk_pct;
  止损距离由 atr_stop_mult(动态ATR) 或 stop_pct(固定百分比) 提供;
- 止损触发按本根最低价判断, 成交价取 min(开盘价, 止损价), 跳空按开盘价成交;
- 账户权益按每根 K 线收盘价记账。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, BacktestConfig


@dataclass
class TradeRecord:
    """一笔完整交易(买入->卖出)的记录。"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float      # 含滑点与手续费后的实际成本价
    exit_price: float       # 含滑点后的实际成交价
    pnl: float              # 净盈亏(USDT)
    pnl_pct: float          # 盈亏 / 投入本金
    holding_bars: int       # 持仓 K 线根数
    exit_reason: str = "signal"  # signal=信号离场, stop=止损离场
    pnl_acct_pct: float = 0.0    # 盈亏 / 入场时账户权益(不同仓位方案公平比较用)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig = DEFAULT_CONFIG,
    stop_pct: float | None = None,
    trail_pct: float | None = None,
    atr: pd.Series | None = None,
    atr_stop_mult: float | None = None,
    risk_pct: float | None = None,
):
    """运行回测。

    df: 索引为 UTC 时间,含 Open/Close 列(Low 用于止损判断,缺省时用 Close)
    signals: 与 df 对齐的 0/1 序列,1=持有,0=空仓
    stop_pct: 固定止损比例(相对买入价),None 表示不启用
    trail_pct: 跟踪止损比例(相对持仓期间最高收盘价),None 表示不启用
    atr: 与 df 对齐的 ATR 序列(atr_stop_mult 模式使用)
    atr_stop_mult: 止损距离 = k * 信号当根的 ATR
    risk_pct: 每笔承担账户比例(0.01=1%); 需同时提供 stop_pct 或 (atr + atr_stop_mult)
    返回: (equity 净值序列, 已平仓交易列表, 实际持仓序列)
    """
    if atr_stop_mult is not None and atr is None:
        raise ValueError("atr_stop_mult 模式必须提供 atr 序列")
    if risk_pct is not None and not (
        (atr is not None and atr_stop_mult is not None) or stop_pct is not None
    ):
        raise ValueError("risk_pct 模式必须提供 stop_pct 或 (atr + atr_stop_mult)")
    if "Open" not in df or "Close" not in df:
        raise ValueError("数据必须包含 Open 与 Close 列")
    sig = signals.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    open_px = df["Open"].to_numpy(dtype=float)
    close_px = df["Close"].to_numpy(dtype=float)
    low_px = df["Low"].to_numpy(dtype=float) if "Low" in df else close_px
    atr_arr = atr.reindex(df.index).to_numpy(dtype=float) if atr is not None else None
    index = df.index
    n = len(df)

    cash = config.initial_capital
    units = 0.0
    equity = np.empty(n)
    actual_pos = np.zeros(n)
    trades = []

    prev_target = 0.0
    entry_time = None
    entry_bar = 0
    entry_cost = 0.0
    entry_capital = 0.0
    stop_price = np.inf
    peak_close = 0.0

    for i in range(n):
        target = sig[i - 1] if i > 0 else 0.0  # 上根收盘的信号,本根开盘执行
        stop_exited = False

        if units > 0:
            # 用上一根收盘价更新跟踪止损(只上移)
            if trail_pct is not None:
                if close_px[i - 1] > peak_close:
                    peak_close = close_px[i - 1]
                trail_level = peak_close * (1 - trail_pct)
                if trail_level > stop_price:
                    stop_price = trail_level
            # 本根最低价触发止损(未配置止损时 stop_price 为 inf,不触发)
            if np.isfinite(stop_price) and low_px[i] <= stop_price:
                exec_price = min(open_px[i], stop_price)
                proceeds = units * exec_price * (1 - config.fee_rate)
                cash += proceeds
                pnl = proceeds - entry_cost
                trades.append(
                    TradeRecord(
                        entry_time=entry_time,
                        exit_time=index[i],
                        entry_price=entry_cost / units if units > 0 else float("nan"),
                        exit_price=exec_price,
                        pnl=pnl,
                        pnl_pct=pnl / entry_cost if entry_cost > 0 else 0.0,
                        holding_bars=i - entry_bar,
                        exit_reason="stop",
                        pnl_acct_pct=pnl / entry_capital if entry_capital > 0 else 0.0,
                    )
                )
                units = 0.0
                stop_price = np.inf
                stop_exited = True

        if not stop_exited and target != prev_target:
            px = open_px[i]
            if target > prev_target:  # 买入
                exec_price = px * (1 + config.slippage)
                entry_capital = cash
                use_atr_stop = atr_stop_mult is not None
                if risk_pct is not None:
                    if use_atr_stop:
                        stop_dist = atr_stop_mult * atr_arr[i - 1]
                    else:
                        stop_dist = exec_price * stop_pct
                elif use_atr_stop:
                    stop_dist = atr_stop_mult * atr_arr[i - 1]
                else:
                    stop_dist = None
                if stop_dist is None:
                    units = (cash / exec_price) * (1 - config.fee_rate)
                    if stop_pct is not None:
                        stop_price = exec_price * (1 - stop_pct)
                    elif trail_pct is not None:
                        stop_price = exec_price * (1 - trail_pct)
                    else:
                        stop_price = np.inf
                else:
                    if not (np.isfinite(stop_dist) and stop_dist > 0 and exec_price - stop_dist > 0):
                        prev_target = target  # 数据未就绪,跳过本次触发
                        equity[i] = cash + units * close_px[i]
                        actual_pos[i] = 1.0 if units > 0 else 0.0
                        continue
                    stop_price = exec_price - stop_dist
                    if risk_pct is not None:
                        units = (cash * risk_pct / stop_dist) * (1 - config.fee_rate)
                    else:
                        units = (cash / exec_price) * (1 - config.fee_rate)
                cash -= units * exec_price
                entry_time = index[i]
                entry_bar = i
                entry_cost = units * exec_price
                peak_close = exec_price if trail_pct is not None else 0.0
            else:  # 卖出:全部币换回资金
                exec_price = px * (1 - config.slippage)
                proceeds = units * exec_price * (1 - config.fee_rate)
                cash += proceeds
                pnl = proceeds - entry_cost
                trades.append(
                    TradeRecord(
                        entry_time=entry_time,
                        exit_time=index[i],
                        entry_price=entry_cost / units if units > 0 else float("nan"),
                        exit_price=exec_price,
                        pnl=pnl,
                        pnl_pct=pnl / entry_cost if entry_cost > 0 else 0.0,
                        holding_bars=i - entry_bar,
                        exit_reason="signal",
                        pnl_acct_pct=pnl / entry_capital if entry_capital > 0 else 0.0,
                    )
                )
                units = 0.0
                stop_price = np.inf
            prev_target = target
        elif stop_exited:
            prev_target = 0.0  # 止损当根不按信号重新入场,下根再判断

        equity[i] = cash + units * close_px[i]
        actual_pos[i] = 1.0 if units > 0 else 0.0

    return pd.Series(equity, index=index), trades, pd.Series(actual_pos, index=index)
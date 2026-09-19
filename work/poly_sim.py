# -*- coding: utf-8 -*-
"""Polymarket 加密"价格阈值"策略的模拟引擎 (虚拟本金)。

用途: 输入一批(市场, 模型概率, 市场价, 结果), 用 Kelly 定仓, 算虚拟 P&L。
这是"模型 vs 市场价"策略的可运行骨架——一旦有市场数据(历史或实时快照)就喂进来。

口径:
  - 每个市场是二元 YES/NO, 买 YES 价 = p (隐含概率)。
  - edge = 模型概率 p_model - 市场价 p_market。
  - 若 p_model > p_market  => 买 YES (市场低估); 若 < 0 则买 NO (0.5/p 对称处理)。
  - Kelly: f = (p_model * b - q) / b, b = 赔率 = (1 / p_market - 1)。封顶 max_kelly 防高估。
  - 结算: 结果 YES -> 每份 +$1, NO -> 0; 计入 taker 费(可选, 模拟吃单成本)或 maker 返佣(可选)。
虚拟本金默认 140 USDC (≈1000 RMB), 可改。

用法: python work/poly_sim.py
"""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "btc_usdt_1h.csv"


@dataclasses.dataclass
class SimConfig:
    bankroll: float = 140.0     # 虚拟本金 USDC (≈1000 RMB)
    max_kelly: float = 0.25     # 仓位封顶 = 24% 建议
    taker_fee_rate: float = 0.0  # 若当前 taker 费率(模拟吃单); 当 maker 则设 0 或负(返佣)
    min_edge: float = 0.05      # 最小 edge 才下单
    min_abs_prob: float = 0.03  # 概率太极端不下单(避免 1% 极端)


def realized_vol(close: pd.Series, lookback_days: int = 60) -> float:
    """近 lookback 天日对数收益年化波动率(只用过去数据, 无未来函数)。"""
    daily = close.resample("1D").last().dropna().pct_change().dropna()
    sd = daily.tail(lookback_days).std()
    return sd * np.sqrt(365) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def model_prob_above(S0: float, K: float, T_days: int, vol: float, drift: float = 0.0) -> float:
    """对数正态 P(S_T > K), 用于"涨破阈值"市场的真实概率。"""
    if not np.isfinite(vol) or vol <= 0 or S0 <= 0 or K <= 0:
        return float("nan")
    T = T_days / 365.0
    d = (np.log(S0 / K) + (drift - 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return _norm_cdf(d)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def kelly_fraction(p_model: float, p_market: float, max_kelly: float) -> float:
    """给定模型概率与市场价, 返回 Kelly 仓位(占bankroll比例, 已封顶)。"""
    # 买 YES: payoff b = (1/p_market - 1); f = (p_model*b - (1-p_model)) / b
    if p_market <= 0 or p_market >= 1:
        return 0.0
    b = 1.0 / p_market - 1.0
    q = 1.0 - p_model
    f = (p_model * b - q) / b
    return float(np.clip(f, 0.0, max_kelly))


def simulate(markets: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """markets 列: question, T_days, S0, K, market_prob, outcome(0/1)。
    返回每笔的 方向/仓位/成本/结果/盈亏; 累计权益。
    """
    cfg = cfg or SimConfig()
    rows = []
    equity = cfg.bankroll
    for _, m in markets.iterrows():
        p_model = m["p_model"]
        p_market = m["market_prob"]
        edge = p_model - p_market
        if abs(edge) < cfg.min_edge or p_model < cfg.min_abs_prob or p_model > 1 - cfg.min_abs_prob:
            rows.append({**m, "bet_frac": 0.0, "bet_usd": 0.0, "side": "skip",
                         "cost": 0.0, "pnl": 0.0, "equity": equity})
            continue
        side = "YES" if edge > 0 else "NO"
        # 对 NO: 市场价看作 P(not), 模型 prob 对称
        p_use = p_model if side == "YES" else (1.0 - p_model)
        price_use = p_market if side == "YES" else (1.0 - p_market)
        frac = kelly_fraction(p_use, price_use, cfg.max_kelly)
        # 若很小, 放弃
        bet_usd = equity * frac
        cost = bet_usd  # 投入
        # taker 费(模拟吃单): 按买入成本收; 当 maker 可设负数(返佣)
        fee = cost * cfg.taker_fee_rate if cfg.taker_fee_rate > 0 else cost * cfg.taker_fee_rate
        # 结果: 买的side是否兑现
        got = (m["outcome"] == (1 if side == "YES" else 0))
        payout = bet_usd / price_use if got else 0.0  # 每份价 price_use, 中奖每份$1
        pnl = payout - cost - fee
        equity += pnl
        rows.append({**m, "edge": edge, "side": side, "bet_frac": frac, "bet_usd": bet_usd,
                     "cost": cost, "fee": fee, "payout": payout, "pnl": pnl, "equity": equity})
    return pd.DataFrame(rows)


def main():
    # 一个超小型演示: 用最近 BTC 价格 + 历史波动率, 算一批"阈值市场"的模型概率,
    # 市场价用"随机给一个与模型近似的价"来示范 P&L 记账。真实运行时把 market_prob 换成真实报价。
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    close = df["Close"]
    S0 = float(close.iloc[-1])
    vol = realized_vol(close)
    # 演示市场: 未来 N 天, 阈值 K(=S0*x%), 市场价给一个"可能被高估/低估"的值
    demo = []
    for T_days, off, mkt in [(7, 0.05, 0.40), (7, 0.10, 0.30), (30, 0.10, 0.35),
                             (30, 0.20, 0.15), (90, 0.20, 0.25), (90, 0.30, 0.10)]:
        K = S0 * (1 + off)
        p_model = model_prob_above(S0, K, T_days, vol)
        # 随机"真结果"(用当前模型作为真实世界概率的代理, 仅作演示)
        outcome = int(np.random.rand() < p_model)
        demo.append({"question": f"BTC>{K:.0f} in {T_days}d", "T_days": T_days, "S0": S0,
                     "K": K, "p_model": p_model, "market_prob": mkt, "outcome": outcome})
    markets = pd.DataFrame(demo)
    res = simulate(markets, SimConfig(bankroll=140.0, max_kelly=0.25, taker_fee_rate=0.0))
    print(f"模拟本金 140 USDC (≈1000 RMB) | 波动率 {vol:.1%}")
    print(f"  最新价 {S0:.0f}")
    print(res[["question", "p_model", "market_prob", "edge", "side", "bet_usd",
               "pnl", "equity"]].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\n期末虚拟权益: {res['equity'].iloc[-1]:.2f} USDC | "
          f"净盈亏 {res['pnl'].sum():+.2f} USDC | 累计收益 {res['equity'].iloc[-1]/140.0-1:+.2%}")


if __name__ == "__main__":
    main()

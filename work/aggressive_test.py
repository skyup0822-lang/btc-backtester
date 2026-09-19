# -*- coding: utf-8 -*-
"""小资金要"更激进": 把每一条路都拿样本外数据测一遍, 而不是凭感觉。

要区分清楚两件完全不同的事:
  甲、加大每笔仓位(风险%) —— 这只是杠杆, 按比例放大收益和回撤, 不改变策略本身的优势;
  乙、提高出手频率(更快的均线/去掉闸门/多品种/更小周期) —— 这才可能改变盈利结构。

口径与实盘一致: 8% 跟踪止损 + 每笔风险 2%(可调); 手续费 0.1%/边 + 滑点 0.05%/边。
样本外 = 2025-01-01 之后, 训练期只用来选参数。

额外揭露一个实盘口径问题: PaperTrader 建仓时 stop_price 取 max(-15%, -8%) = -8%,
但仓位是按 15% 的止损距离算的 -> 实盘真实风险只有标称的 53%。下面"口径修正"行就是把
止损距离改成 8%, 让标称风险变成真风险(同样的 2% 风险, 仓位会翻倍)。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics
from config import DATA_FILE, DEFAULT_CONFIG, TRAIN_END
from strategies.sma_cross import sma_cross
from strategies.trend_system import sma_gate

TRAIL = 0.08

# (显示名, 信号函数, 回测附加参数)
VARIANTS = [
    ("甲1 现状: 风险2% / 止损15%", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02)),
    ("甲2 风险5%", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.05)),
    ("甲3 风险10%", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.10)),
    ("甲4 风险20%", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.20)),
    ("甲5 口径修正: 止损8% / 风险2%(真)", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.08, trail_pct=TRAIL, risk_pct=0.02)),
    ("甲6 口径修正 + 风险5%", lambda d: sma_gate(d, 48, 336, 1440),
     dict(stop_pct=0.08, trail_pct=TRAIL, risk_pct=0.05)),
    ("乙1 快信号 24/168 + 闸门", lambda d: sma_gate(d, 24, 168, 1440),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02)),
    ("乙2 快信号 24/168 去闸门", lambda d: sma_cross(d, 24, 168),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02)),
    ("乙3 去闸门 48/336", lambda d: sma_cross(d, 48, 336),
     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02)),
]


def fmt(v, pct=True, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v * 100:+.{nd}f}%" if pct else f"{v:,.0f}"


def worst_streak(trades):
    """最长连亏笔数 + 那一段累计亏掉多少(账户%)。"""
    ts = sorted(trades, key=lambda t: t.exit_time)
    best = cur = 0
    best_sum = cur_sum = 0.0
    for t in ts:
        if t.pnl < 0:
            cur += 1
            cur_sum += t.pnl_acct_pct
            if cur > best:
                best, best_sum = cur, cur_sum
        else:
            cur, cur_sum = 0, 0.0
    return best, best_sum


def evaluate(df, signals, extra, test_start):
    equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, **extra)
    test_trades = [t for t in trades if t.entry_time > test_start]
    acct = np.array([t.pnl_acct_pct for t in test_trades], dtype=float)
    t_eq = equity[equity.index > test_start]
    streak, streak_sum = worst_streak(test_trades)
    return {
        "n_trades": len(test_trades),
        "expectancy": float(acct.mean()) if len(acct) else np.nan,
        "win_rate": float((acct > 0).mean()) if len(acct) else np.nan,
        "ret": float(t_eq.iloc[-1] / t_eq.iloc[0] - 1) if len(t_eq) > 1 else np.nan,
        "maxdd": float((t_eq / t_eq.cummax() - 1).min()) if len(t_eq) > 1 else np.nan,
        "streak": streak,
        "streak_sum": streak_sum,
    }


def portfolio(pairs, extra, test_start):
    """两个品种各分一半资金(独立子账户)的合并效果。

    单品种引擎跑不了真正的多品种组合, 这里用"两条净值曲线各归一化后取平均"近似 ——
    等价于资金对半分、每个品种各自持仓, 是这套单资产引擎能给的最接近的估计。
    """
    curves, all_trades = [], []
    for df, signals in pairs:
        equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, **extra)
        ts = [t for t in trades if t.entry_time > test_start]
        all_trades += ts
        ce = equity[equity.index > test_start]
        if len(ce) > 1:
            curves.append(ce / ce.iloc[0])
    if not curves:
        return None
    comb = pd.concat(curves, axis=1).ffill().dropna().mean(axis=1)
    acct = np.array([t.pnl_acct_pct for t in all_trades], dtype=float)
    streak, streak_sum = worst_streak(all_trades)
    return {
        "n_trades": len(all_trades),
        "expectancy": float(acct.mean()) if len(acct) else np.nan,
        "win_rate": float((acct > 0).mean()) if len(acct) else np.nan,
        "ret": float(comb.iloc[-1] / comb.iloc[0] - 1),
        "maxdd": float((comb / comb.cummax() - 1).min()),
        "streak": streak,
        "streak_sum": streak_sum,
    }


def main():
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    test_start = pd.Timestamp(TRAIN_END, tz="UTC")
    test = df[df.index > test_start]
    eth_path = ROOT / "data" / "eth_usdt_1h.csv"
    eth = pd.read_csv(eth_path, index_col=0, parse_dates=True) if eth_path.exists() else None

    bench, _ = buy_hold_metrics(test, DEFAULT_CONFIG)
    print(f"数据 {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} | "
          f"样本外 {test.index[0]:%Y-%m-%d} ~ {test.index[-1]:%Y-%m-%d} ({len(test)} 根1h)")
    print(f"同期买入持有 BTC: {bench['total_return']:+.2%} | 最大回撤 {bench['max_drawdown']:.2%}")
    print()

    rows = []
    for name, sigfn, extra in VARIANTS:
        r = evaluate(df, sigfn(df), extra, test_start)
        r["variant"] = name
        rows.append(r)
    if eth is not None:
        r = evaluate(eth, sma_gate(eth, 48, 336, 1440),
                     dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02), test_start)
        r["variant"] = "乙4 ETH 同策略(独立100U)"
        rows.append(r)
        r = portfolio(
            [(df, sma_gate(df, 48, 336, 1440)), (eth, sma_gate(eth, 48, 336, 1440))],
            dict(stop_pct=0.15, trail_pct=TRAIL, risk_pct=0.02), test_start)
        if r:
            r["variant"] = "乙5 BTC+ETH 各一半资金"
            rows.append(r)

    tbl = pd.DataFrame(rows)[["variant", "n_trades", "expectancy", "win_rate",
                              "ret", "maxdd", "streak", "streak_sum"]]
    out = ROOT / "results" / "aggressive.csv"
    tbl.to_csv(out, index=False)

    print(f"{'方案':<30}{'交易':>5}{'单笔期望':>10}{'胜率':>8}{'区间收益':>10}"
          f"{'最大回撤':>10}{'最长连亏':>9}{'连亏累计':>10}")
    for _, r in tbl.iterrows():
        print(f"{r['variant']:<30}{int(r['n_trades']):>5}{fmt(r['expectancy']):>10}"
              f"{fmt(r['win_rate'], nd=1):>8}{fmt(r['ret']):>10}{fmt(r['maxdd']):>10}"
              f"{int(r['streak']):>7}笔{fmt(r['streak_sum']):>10}")
    print()
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()

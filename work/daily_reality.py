# -*- coding: utf-8 -*-
"""「不能每天盈利就不是好策略」—— 先用数据把这件事的可行性算清楚。

回答三个问题:
  1. BTC 自己每天上涨的概率是多少? (再好的策略也顶不破这个上限)
  2. 现有策略盈利的交易日占多少? 最长连亏多少天?
  3. 高频变体是不是"盈利天数更多"? (如果多, 那才是换策略的真正理由)

全部离线, 用的是本地数据和已经跑好的净值曲线。
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
from config import DATA_FILE, DEFAULT_CONFIG
from strategies.trend_system import sma_gate

LIVE = dict(stop_pct=0.15, trail_pct=0.08, risk_pct=0.02)


def streak(flags):
    """最长连续 True 的长度。"""
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def daily_stats(eq, label):
    """把小时净值折成日净值, 拆成 上涨/持平/下跌 三类。

    注意"持平"要单独算: 趋势策略大部分时间空仓, 那些天账户是 0.00% 不动,
    既不是赚也不是亏 —— 混进"未盈利"里会把策略说得比实际惨。
    """
    d = eq.resample("1D").last().dropna()
    if len(d) < 30:
        return None
    r = d.pct_change().dropna()
    up = r > 1e-9
    dn = r < -1e-9
    flat = ~up & ~dn
    return {
        "label": label,
        "days": len(r),
        "up_days": float(up.mean()),
        "flat_days": float(flat.mean()),
        "down_days": float(dn.mean()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "avg_day": float(r.mean()),
        "daily_vol": float(r.std()),
        "lose_streak": streak(dn.to_numpy()),
        "flat_streak": streak(flat.to_numpy()),
        "total": float(d.iloc[-1] / d.iloc[0] - 1),
        "maxdd": float((d / d.cummax() - 1).min()),
    }


def show(rows):
    print(f"{'方案':<26}{'上涨':>7}{'持平':>7}{'下跌':>7}{'日均':>9}{'日波动':>8}"
          f"{'最长连跌':>9}{'总收益':>10}{'最大回撤':>10}")
    for r in rows:
        print(f"{r['label']:<26}{r['up_days'] * 100:>6.1f}%{r['flat_days'] * 100:>6.1f}%"
              f"{r['down_days'] * 100:>6.1f}%{r['avg_day'] * 100:>8.3f}%"
              f"{r['daily_vol'] * 100:>7.2f}%{r['lose_streak']:>7}天"
              f"{r['total'] * 100:>9.1f}%{r['maxdd'] * 100:>9.1f}%")


print("=" * 88)
print("一、BTC 自己: 日线上涨的概率(这是所有做多策略的天花板)")
print("=" * 88)
raw = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
dc = raw["Close"].resample("1D").last().dropna()
dr = dc.pct_change().dropna()
up = dr > 0
print(f"  区间 {dc.index[0]:%Y-%m-%d} ~ {dc.index[-1]:%Y-%m-%d}, 共 {len(dr)} 个交易日")
print(f"  BTC 上涨的天数: {up.sum()} ({up.mean():.1%})   下跌: {(~up).sum()} ({(~up).mean():.1%})")
print(f"  最长连续下跌: {streak(~up)} 天   最长连续上涨: {streak(up)} 天")
print(f"  日均涨跌 {dr.mean():+.4%}   日波动 {dr.std():.2%}")
print()

rows = []
eq, _, _ = run_backtest(raw, sma_gate(raw, 48, 336, 1440), DEFAULT_CONFIG, **LIVE)
r = daily_stats(eq, "现状: 趋势系统(48/336/1440)")
if r:
    rows.append(r)
r = daily_stats(raw["Close"], "买入持有 BTC")
if r:
    rows.append(r)

print("=" * 88)
print("二、现有策略盈利的交易日占多少")
print("=" * 88)
show(rows)
print()

print("=" * 88)
print("三、提高频率是不是「盈利天数更多」? (2024-06 起的曲线)")
print("=" * 88)
cur = pd.read_csv(ROOT / "results" / "tf_sweep_curves.csv", index_col=0, parse_dates=True)
hf = []
for col in cur.columns:
    r = daily_stats(cur[col].dropna(), col)
    if r:
        hf.append(r)
show(hf)
print()

print("=" * 88)
print("四、算一下「每天都赚」到底要求什么")
print("=" * 88)
for target, vol in ((0.90, 0.02), (0.95, 0.02), (0.80, 0.02)):
    from statistics import NormalDist
    edge = NormalDist().inv_cdf(target) * vol
    print(f"  日波动 {vol:.0%} 时, 想 {target:.0%} 的交易日盈利 -> 每天平均要赚 {edge:.3%}, "
          f"一年 {((1 + edge) ** 365 - 1):,.0f} 倍")
print()
print(f"  参照: 现有策略日均 {rows[0]['avg_day']:+.4%}, 一年约 {((1 + rows[0]['avg_day']) ** 365 - 1):+.1%}")

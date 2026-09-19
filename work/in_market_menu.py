# -*- coding: utf-8 -*-
"""「少空仓」到底要拿什么换? 把"在场时间 -> 上涨日占比 / 回撤"的整条曲线测出来。

你说的痛点其实是: 55.8% 的交易日账户一动不动(空仓), 看起来像"策略没用"。
那就把"在场时间"当成一个旋钮, 看每一个档位要付出多少回撤。

全部离线, 同一套成本(0.1%/边 + 0.05%滑点), 同一个区间。
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
from strategies.buy_hold import buy_hold
from strategies.sma_cross import sma_cross
from strategies.trend_filter import trend_filter
from strategies.trend_system import sma_gate

LIVE = dict(stop_pct=0.15, trail_pct=0.08, risk_pct=0.02)

VARIANTS = [
    ("① 始终在场(买入持有)", lambda d: buy_hold(d), {}),
    ("② 收盘>SMA1440 才在场(慢闸门)", lambda d: trend_filter(d, 1440), LIVE),
    ("③ 收盘>SMA480 才在场(快闸门)", lambda d: trend_filter(d, 480), LIVE),
    ("④ 48/336 无闸门", lambda d: sma_cross(d, 48, 336), LIVE),
    ("⑤ 现状 48/336 + 1440闸门", lambda d: sma_gate(d, 48, 336, 1440), LIVE),
    ("⑥ 24/168 + 1440闸门(更快)", lambda d: sma_gate(d, 24, 168, 1440), LIVE),
    ("⑦ 96/672 + 1440闸门(更慢更严)", lambda d: sma_gate(d, 96, 672, 1440), LIVE),
]


def streak(flags):
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    raw = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    oos_start = pd.Timestamp("2024-12-31", tz="UTC")
    for tag, frame in (("全周期 2019-11 ~ 2026-09", raw),
                       ("样本外 2025-01 ~ 2026-09", raw[raw.index > oos_start])):
        rows = []
        for label, sigfn, extra in VARIANTS:
            signals = sigfn(frame)
            eq, trades, pos = run_backtest(frame, signals, DEFAULT_CONFIG, **extra)
            d = eq.resample("1D").last().dropna()
            r = d.pct_change().dropna()
            up, dn = r > 1e-9, r < -1e-9
            in_mkt = float(pd.Series(pos, index=frame.index).resample("1D").last().mean())
            rows.append({
                "label": label,
                "in_mkt": in_mkt,
                "up": float(up.mean()), "flat": float((~up & ~dn).mean()), "dn": float(dn.mean()),
                "streak": streak(dn.to_numpy()),
                "total": float(d.iloc[-1] / d.iloc[0] - 1),
                "maxdd": float((d / d.cummax() - 1).min()),
                "avg": float(r.mean()),
                "vol": float(r.std()),
            })

        print("=" * 96)
        print(f"空仓时间 -> 上涨日占比 / 回撤 的菜单  [{tag}]")
        print("=" * 96)
        print(f"{'方案':<30}{'在场':>7}{'上涨':>7}{'持平':>7}{'下跌':>7}{'最长连跌':>9}"
              f"{'总收益':>11}{'最大回撤':>10}{'收益/回撤':>10}")
        for x in rows:
            rr = abs(x['total'] / x['maxdd']) if x['maxdd'] else float('nan')
            print(f"{x['label']:<30}{x['in_mkt'] * 100:>6.1f}%{x['up'] * 100:>6.1f}%"
                  f"{x['flat'] * 100:>6.1f}%{x['dn'] * 100:>6.1f}%{x['streak']:>7}天"
                  f"{x['total'] * 100:>10.1f}%{x['maxdd'] * 100:>9.1f}%{rr:>10.2f}")
        print()
        out = ROOT / "results" / ("in_market_menu.csv" if frame is raw else "in_market_menu_oos.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"已保存: {out}")
        print()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""以「最大回撤」为约束, 反解每个方案在预算内能给出多少收益。

做法: 先把每个方案按 2% 风险跑一遍, 看它自然产生多大回撤;
      再用 risk = 2% x 预算/自然回撤 把仓位调到刚好用满预算, 重跑一遍取真实数字。
      回撤和收益对风险近似线性(前面 aggressive_test 已经验证过), 所以一次标定就够。

这样比出来的才是"同等风险下谁赚得多", 而不是"谁胆子大"。
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

BASE_RISK = 0.02
STOP, TRAIL = 0.15, 0.08
BUDGETS = [0.05, 0.10, 0.20]

# 现货不能加杠杆: 名义仓位 = cash * risk / stop_pct, 所以 risk 超过 stop_pct(15%)
# 就等于"每次全仓"。超过这个点再想放大, 只能上杠杆, 那已经不是现货模拟盘了。
FULL_POS_RISK = STOP

STRATS = [
    ("现状 48/336+1440闸门", lambda d: sma_gate(d, 48, 336, 1440)),
    ("更快 24/168+1440闸门", lambda d: sma_gate(d, 24, 168, 1440)),
    ("只看 收盘>SMA1440", lambda d: trend_filter(d, 1440)),
    ("48/336 无闸门", lambda d: sma_cross(d, 48, 336)),
]


def run(frame, signals, risk=None):
    extra = dict(stop_pct=STOP, trail_pct=TRAIL)
    if risk is not None:
        extra["risk_pct"] = risk
    eq, trades, pos = run_backtest(frame, signals, DEFAULT_CONFIG, **extra)
    d = eq.resample("1D").last().dropna()
    r = d.pct_change().dropna()
    up, dn = r > 1e-9, r < -1e-9
    posd = pd.Series(pos, index=frame.index).resample("1D").last()
    flat = (posd < 0.5).astype(int)
    # 最长连续空仓天数
    best = cur = 0
    for v in flat.to_numpy():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {
        "ret": float(d.iloc[-1] / d.iloc[0] - 1),
        "dd": float((d / d.cummax() - 1).min()),
        "up": float(up.mean()),
        "trades": len(trades),
        "flat_streak": best,
        "in_mkt": float(posd.mean()),
    }


def main():
    oos = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    oos = oos[oos.index > pd.Timestamp("2024-12-31", tz="UTC")]
    months = (oos.index[-1] - oos.index[0]).days / 30.44
    print(f"样本外 {oos.index[0]:%Y-%m-%d} ~ {oos.index[-1]:%Y-%m-%d} ({months:.1f} 个月), "
          f"起始 100 USDT")
    print()

    rows = []
    for name, fn in STRATS:
        sig = fn(oos)
        base = run(oos, sig, BASE_RISK)
        for b in BUDGETS:
            if base["dd"] >= 0:
                continue
            risk = BASE_RISK * b / abs(base["dd"])
            if risk > 0.60:
                continue
            r = run(oos, sig, risk)
            rows.append({
                "预算": b, "方案": name, "风险": risk,
                "回撤": r["dd"], "收益": r["ret"],
                "上涨日": r["up"], "交易": r["trades"],
                "最长空仓": r["flat_streak"], "在场": r["in_mkt"],
                "月均": r["ret"] / months,
                "100U月均": r["ret"] / months * 100,
            })

    df = pd.DataFrame(rows)
    for b in BUDGETS:
        sub = df[df["预算"] == b].sort_values("收益", ascending=False)
        if not len(sub):
            continue
        print("=" * 100)
        print(f"回撤预算 {b:.0%}")
        print("=" * 100)
        print(f"{'方案':<24}{'用到的风险':>10}{'实际回撤':>10}{'收益':>9}{'月均':>9}"
              f"{'上涨日':>8}{'交易':>6}{'最长空仓':>9}")
        for _, x in sub.iterrows():
            print(f"{x['方案']:<24}{x['风险'] * 100:>9.1f}%{x['回撤'] * 100:>9.1f}%"
                  f"{x['收益'] * 100:>8.1f}%{x['月均'] * 100:>8.2f}%{x['上涨日'] * 100:>7.1f}%"
                  f"{x['交易']:>6.0f}{x['最长空仓']:>7.0f}天")
        best = sub.iloc[0]
        print(f"  -> 最优: {best['方案']}, 收益 {best['收益'] * 100:+.1f}% "
              f"= 每月 {best['月均'] * 100:+.2f}% = 100 USDT 上每月 {best['100U月均']:+.2f} USDT")
        print()

    df.to_csv(ROOT / "results" / "drawdown_budget.csv", index=False)

    # ---- 关键: 全周期尾巴检验 + 仓位顶满的位置 ----
    full_raw = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    print("=" * 100)
    print("尾巴检验: 20% 回撤预算在样本外标定出来的风险, 放到全周期(2019-11起)会跌多深")
    print("=" * 100)
    print(f"{'方案':<24}{'标定风险':>9}{'名义仓位':>9}{'样本外回撤':>11}{'全周期回撤':>11}{'全周期收益':>11}")
    for name, fn in STRATS:
        base_oos = run(oos, fn(oos), BASE_RISK)
        if base_oos["dd"] >= 0:
            continue
        risk = BASE_RISK * 0.20 / abs(base_oos["dd"])
        rf = run(full_raw, fn(full_raw), risk)
        notional = min(risk / STOP, 1.0)
        print(f"{name:<24}{risk * 100:>8.1f}%{notional * 100:>8.0f}%"
              f"{base_oos['dd'] * 100:>10.1f}%{rf['dd'] * 100:>10.1f}%{rf['ret'] * 100:>10.1f}%")
    print()
    print(f"  现货的尽头: risk = {FULL_POS_RISK:.0%} 时名义仓位正好是 100%(每次全仓)。")
    print(f"  再往上要放大, 只能上杠杆 —— 那已经不是这个现货模拟盘能做的事了。")

    print()
    print("=" * 100)
    print("按「20% 是硬上限」标定: 找全周期回撤真的不超过 20% 的最大风险")
    print("=" * 100)
    print(f"{'方案':<24}{'风险':>8}{'全周期回撤':>12}{'全周期收益':>12}{'样本外回撤':>12}"
          f"{'样本外收益':>12}{'月均':>9}")
    for name, fn in STRATS:
        cand = []
        for risk in np.arange(0.02, 0.161, 0.005):
            rf = run(full_raw, fn(full_raw), float(risk))
            cand.append((float(risk), rf))
        ok = [c for c in cand if abs(c[1]["dd"]) <= 0.20]
        if not ok:
            continue
        risk, rf = max(ok, key=lambda c: c[0])
        ro = run(oos, fn(oos), risk)
        months = (oos.index[-1] - oos.index[0]).days / 30.44
        print(f"{name:<24}{risk * 100:>7.1f}%{rf['dd'] * 100:>11.1f}%{rf['ret'] * 100:>11.1f}%"
              f"{ro['dd'] * 100:>11.1f}%{ro['ret'] * 100:>11.1f}%"
              f"{ro['ret'] / months * 100:>8.2f}%")
    print()
    print("  注: 全周期回撤 <= 20% 才算真的守住。上一张表用样本外标定的 15.3%,")
    print("      全周期会跌到 -45.8% —— 那等于没守。")

    print()
    print("参考(无仓位调节, 现货满仓): ", end="")
    bh = run(oos, buy_hold(oos))
    print(f"买入持有 回撤 {bh['dd'] * 100:.1f}% 收益 {bh['ret'] * 100:+.1f}% 上涨日 {bh['up'] * 100:.1f}%")
    print(f"已保存: results/drawdown_budget.csv")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""加密"价格阈值"市场概率模型校准 (用现有 BTC/ETH 历史数据, 不依赖拿不到的市场价)。

目标: 检验"用波动率算 P(价在 T 天后 > K)"这个模型, 历史上是否真的校准。
校准的模型才可能赚钱; 校准就是: 模型说 30% 的概率, 历史上其实是 30% 发生。

方法:
  选取一系列"定价日" t0, 对未来 T 天后的阈值 K(=t0 价的 ±x%) 用 波动率模型 算 概率 p,
  记录真实结果(是否 >K)。把预测概率分桶, 比较 桶内平均预测概率 vs 实际命中率(可靠性图),
  并报告 Brier 分数 (越接近 0 越好, 0.25=纯随机)。
  波动率只用 t0 之前的数据(无未来函数), 无漂移/含历史漂移 两种口径都测。

用法: python work/pred_market_prob.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import math

import numpy as np
import pandas as pd


def norm_cdf(x):
    """标准正态累积分布, 用 math.erf 实现(无需 scipy)。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

ASSETS = {
    "BTC": ROOT / "data" / "btc_usdt_1h.csv",
    "ETH": ROOT / "data" / "eth_usdt_1h.csv",
}
EXPIRIES = [7, 30, 90]                     # 未来天数
KEYS = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]  # 阈值相对 t0 价的偏移
VOL_WINDOW_DAYS = 60                       # 用近 60 天日收益估波动率
SAMPLE = "1D"                              # 定价日采样频率(每周)=校准样本量
RISK_FREE = 0.0                            # PM 是"真实世界概率"口径, 无风险利率取0


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def realized_vol(close, lookback_days):
    """用近 lookback_days 天的日对数收益标准差 -> 年化波动率(只用过去数据)。"""
    daily = close.resample("1D").last().dropna().pct_change().dropna()
    if len(daily) < 20:
        return np.nan, np.nan
    r = daily.tail(lookback_days)
    sd = r.std()
    if not np.isfinite(sd) or sd <= 0:
        return np.nan, np.nan
    mean_drift = r.mean()
    # 年化: 按 365 天
    vol = sd * np.sqrt(365)
    # 历史日均漂移 -> 年化(用于"含漂移"口径)
    drift = mean_drift * 365
    return vol, drift


def prob_above(S0, K, T_days, vol, drift=0.0):
    """对数正态: P(S_T > K)。dv=漂移, sigma=年化波动率, T 单位年。"""
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    T = T_days / 365.0
    d2 = (np.log(S0 / K) + (drift - 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return norm_cdf(d2)


def main():
    results = []
    for asset, path in ASSETS.items():
        df = load(path)
        close = df["Close"]
        open_ms = df.index
        print(f"【{asset}】 {close.index[0]:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}")

        # 定价日: 从数据中段开始(确保有足够历史估波动率), 每周取一次
        start = close.index[0] + pd.Timedelta(days=VOL_WINDOW_DAYS + 10)
        end = close.index[-1] - pd.Timedelta(days=max(EXPIRIES) + 10)
        dates = pd.date_range(start, end, freq="7D", tz="UTC")

        rows = []
        for t0 in dates:
            if t0 not in close.index:
                continue
            # 用 t0 之前的数据被严格到 close[:t0]
            hist = close[close.index <= t0]
            if len(hist) < VOL_WINDOW_DAYS + 5:
                continue
            S0 = hist.iloc[-1]
            vol, drift = realized_vol(hist, VOL_WINDOW_DAYS)
            if not np.isfinite(vol):
                continue
            for T_days in EXPIRIES:
                tT = t0 + pd.Timedelta(days=T_days)
                if tT not in close.index:
                    continue
                ST = float(close.loc[tT])
                for kk in KEYS:
                    K = S0 * (1 + kk)
                    p0 = prob_above(S0, K, T_days, vol, 0.0)
                    p_d = prob_above(S0, K, T_days, vol, drift)
                    outcome = float(ST > K)
                    rows.append({
                        "asset": asset, "t0": t0, "T": T_days, "K": K, "S0": S0,
                        "outcome": outcome, "p_zero": p0, "p_drift": p_d,
                    })
        res = pd.DataFrame(rows).dropna(subset=["p_zero"])

        def report(prob_col, label):
            sub = res[["outcome", prob_col]].copy()
            sub["prob"] = sub[prob_col]
            # brier
            brier = float(((sub["prob"] - sub["outcome"]) ** 2).mean())
            print(f"\n  --- 概率口径: {label} | 样本 {len(sub)} | Brier {brier:.4f} (0.25=纯随机) ---")
            # 分桶可靠性
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
            sub["bin"] = pd.cut(sub["prob"], bins)
            for b, g in sub.groupby("bin", observed=True):
                if len(g) < 20:
                    continue
                pred = g["prob"].mean()
                act = g["outcome"].mean()
                print(f"    预测{pred:>6.1%}|{b.right:>5.0%} 实际命中 {act:>6.1%}  n={len(g)}")
            # 简单校准: 分半
            med = sub["prob"].median()
            lo = sub[sub["prob"] <= med]
            hi = sub[sub["prob"] > med]
            print(f"  [低概率半] 平均预测 {lo['prob'].mean():.1%} 实际 {lo['outcome'].mean():.1%}")
            print(f"  [高概率半] 平均预测 {hi['prob'].mean():.1%} 实际 {hi['outcome'].mean():.1%}")
        report("p_zero", "无漂移")
        report("p_drift", "含历史漂移")
        results.append(res)

    # BTC+ETH 合并, 用无漂移口径汇总
    merged = pd.concat([r for r in results if len(r)])
    print("\n" + "=" * 70)
    print("BTC+ETH 合并, 无漂移口径: 校准是否可信(分桶)")
    print("=" * 70)
    sub = merged[["outcome", "p_zero"]].copy()
    sub["prob"] = sub["p_zero"]
    brier = float(((sub["prob"] - sub["outcome"]) ** 2).mean())
    print(f"  Brier {brier:.4f}")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    sub["bin"] = pd.cut(sub["prob"], bins)
    for b, g in sub.groupby("bin", observed=True):
        if len(g) < 15:
            continue
        print(f"    预测{g['prob'].mean():>6.1%}|{b.right:>5.0%} 实际 {g['outcome'].mean():>6.1%}  n={len(g)}")


if __name__ == "__main__":
    main()

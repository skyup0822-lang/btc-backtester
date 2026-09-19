# -*- coding: utf-8 -*-
"""短周期 BTC 涨跌 预测能力 / 期望测试。

问题: Polymarket 上有 5m/15m/... 的 "BTC 涨/跌" 二元市场。要在里面赚钱,
需要"能预测下根涨跌准确率明显 >50%, 且扣掉成本后仍为正"。
本测试用项目已有的 5m/15m/30m/1h/4h 数据, 检验最简单信号(动量/反转/基准率)
的预测准确率, 并给出净期望。

成本口径(保守给两档): taker 费 + 点差, 按每笔占投入比例估。5m 短期点差通常更大。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STEPS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}
HORIZONS = [1, 3, 6]   # 预测未来 N 根(1根=该周期; 5m的3根=15m...)
COSTS = [0.005, 0.01, 0.02]  # 每笔成本(占投入): 0.5%/1%/2%


def load_interval(name):
    df = pd.read_csv(DATA / f"tf_{name}.csv")
    col = "close" if "close" in df.columns else "Close" if "Close" in df.columns else None
    if col is None:
        raise ValueError(f"{name} 无 close 列")
    df["dt"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    return df[col].astype(float)


def test(name, close):
    fwd = {}
    for L in HORIZONS:
        fwd[L] = (close.shift(-L) / close - 1.0).dropna()

    # 信号(仅用过去数据)
    ret1 = close.pct_change()          # 上一根收益
    ret3 = close.pct_change(3)         # 近3根累计
    ret6 = close.pct_change(6)

    def hit(sig, L, pred_up):
        """给定信号, 计算预测方向命中率(相对 fwd>0)。"""
        s = sig.copy()
        # 对齐: 与 fwd 同索引
        f = fwd[L]
        idx = s.index.intersection(f.index)
        actual_up = (f.loc[idx] > 0).astype(float)
        if pred_up == "momentum":
            pred = (s.loc[idx] > 0)          # 前一根涨 -> 预测涨
        elif pred_up == "reversion":
            pred = (s.loc[idx] < 0)          # 前一根跌 -> 预测涨(反转)
        else:
            pred = pd.Series(1, index=idx)   # 恒预测涨(基准率)
        return float((pred == (actual_up > 0)).mean())

    out = []
    for L in HORIZONS:
        rows = {"interval": name, "horizon_bars": L,
                "horizon_min": L * STEPS[name] // 60}
        for tag, sig, kind in [
            ("动量(1根)", ret1, "momentum"),
            ("反转(1根)", ret1, "reversion"),
            ("动量(3根)", ret3, "momentum"),
            ("反转(3根)", ret3, "reversion"),
            ("动量(6根)", ret6, "momentum"),
            ("基准率(恒涨)", None, "base"),
        ]:
            acc = hit(sig, L, kind) if sig is not None else float((fwd[L] > 0).mean())
            rows[tag] = round(acc, 4)
        out.append(rows)
    return out


def main():
    print("=" * 110)
    print("短周期 BTC 涨跌 预测能力 | 准确率 = 预测下根方向正确的比例 (50%=纯随机, 越高越有edge)")
    print("=" * 110)
    all_rows = []
    for name in STEPS:
        close = load_interval(name)
        table = test(name, close)
        all_rows.extend(table)
        for r in table:
            print(f"\n【{name}】 预测未来 {r['horizon_bars']} 根 ({r['horizon_min']} 分钟):")
            for tag in ["动量(1根)", "反转(1根)", "动量(3根)", "反转(3根)", "动量(6根)", "基准率(恒涨)"]:
                print(f"    {tag:<8} 准确率 {r[tag]:.2%}")

    print("\n" + "=" * 110)
    print("净期望估算: 若你在『公平价 50%』买入预测的那一侧, 每笔 = 准确率 - 0.50 - 成本")
    print("占投入比例。只看『基准率』已=50%, 若动量/反转不到 52%, 则扣成本后必为负。")
    print("=" * 110)
    df = pd.DataFrame(all_rows)
    for name in STEPS:
        print(f"\n【{name}】(未来1根) 各信号净期望@成本1%:")
        r = df[(df["interval"] == name) & (df["horizon_bars"] == 1)].iloc[0]
        for tag in ["动量(1根)", "反转(1根)", "基准率(恒涨)"]:
            acc = r[tag]
            net = acc - 0.5 - 0.01
            print(f"    {tag:<8} acc {acc:.2%} | 净期望 {net:+.2%}  ({'正' if net>0 else '负(亏)'})")


if __name__ == "__main__":
    main()

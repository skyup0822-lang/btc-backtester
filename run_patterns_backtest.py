# -*- coding: utf-8 -*-
"""形态规则回测: "及时收手"到底能不能提高期望?

对比方案(全部基于杂交策略: SMA48/336 + 收盘>SMA1440, 15%止损, 每笔风险1%):
  base         杂交策略(不加形态规则)
  v_dump       瀑布/砸盘出现当根收盘离场 + 24小时禁止再入场
  v_dump_range v_dump + 盘整时禁止新开仓
  v_dump_pump  v_dump + 拉盘当根禁止新开仓(不追高)
  v_all        以上全部

评估方法(与之前选型一致): BTC+ETH 各12段滚动验证(训练12个月->测试6个月),
汇总所有测试段交易, 5000次自助抽样给出单笔期望的95%置信区间; 另报 2025+ 样本外表现。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config import DEFAULT_CONFIG, RESULTS_DIR
from patterns import detect, forward_stats
from strategies.trend_system import sma_gate

RNG = np.random.default_rng(42)
ASSETS = {"BTC": "data/btc_usdt_1h.csv", "ETH": "data/eth_usdt_1h.csv"}
ORIGINS = pd.date_range("2021-07-01", periods=12, freq="150D", tz="UTC")
TRAIN_MO, TEST_MO = 12, 6
COOLDOWN = 24

VARIANTS = {
    "base": {"dump_exit": False},
    "v_dump": {"dump_exit": True},
    "v_dump_range": {"dump_exit": True, "block_range": True},
    "v_dump_pump": {"dump_exit": True, "block_pump": True},
    "v_all": {"dump_exit": True, "block_range": True, "block_pump": True},
}
LABELS = {
    "base": "杂交策略(不加形态)",
    "v_dump": "瀑布/砸盘离场+24h冷却",
    "v_dump_range": "+盘整不开仓",
    "v_dump_pump": "+拉盘不追高",
    "v_all": "全部形态规则",
}


def effective_signal(sig, dump, range_, pump, dump_exit, block_range=False, block_pump=False):
    """在策略信号上叠加形态规则, 输出逐根目标仓位(只用当根及过去信息)。"""
    n = len(sig)
    s = sig.to_numpy(dtype=float)
    d = dump.fillna(False).to_numpy()
    r = range_.fillna(False).to_numpy()
    p = pump.fillna(False).to_numpy()
    out = np.zeros(n)
    cool = 0
    for i in range(n):
        if cool > 0:
            out[i] = 0.0
            cool -= 1
            continue
        if s[i] == 0.0:
            out[i] = 0.0
            continue
        held = i > 0 and out[i - 1] == 1.0
        if not held:
            if block_range and r[i]:
                out[i] = 0.0
                continue
            if block_pump and p[i]:
                out[i] = 0.0
                continue
        out[i] = 1.0
        if dump_exit and d[i]:
            out[i] = 0.0
            cool = COOLDOWN
    return pd.Series(out, index=sig.index)


def backtest_window(df, sig_full, pat, start, end, variant_kwargs):
    part = df[(df.index >= start) & (df.index <= end)]
    if len(part) < 100:
        return None
    sig = sig_full.reindex(part.index)
    sig_eff = effective_signal(sig, pat["dump"], pat["range"], pat["pump"], **variant_kwargs)
    equity, trades, _ = run_backtest(part, sig_eff, DEFAULT_CONFIG, stop_pct=0.15, risk_pct=0.01)
    return equity, trades


def main():
    print("=" * 100)
    print("形态规则回测: 瀑布/砸盘离场、盘整/拉盘过滤, 能否及时收手?")
    print("=" * 100)
    summary_rows = []
    seg_rows = []
    all_seg_trades = {v: [] for v in VARIANTS}

    for asset, path in ASSETS.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        sig = sma_gate(df)
        pat = detect(df)
        for origin in ORIGINS:
            tr_start = origin - pd.DateOffset(months=TRAIN_MO)
            tr_end = origin - pd.Timedelta(hours=1)
            te_start = origin
            te_end = min(origin + pd.DateOffset(months=TEST_MO), df.index[-1])
            if te_start > df.index[-1] or tr_start < df.index[0]:
                continue
            for v, kw in VARIANTS.items():
                res = backtest_window(df, sig, pat, te_start, te_end, kw)
                if res is None:
                    continue
                equity, trades = res
                pnls = np.array([t.pnl_acct_pct for t in trades], dtype=float)
                exp = pnls.mean() if len(pnls) else float("nan")
                seg_rows.append({
                    "asset": asset, "origin": origin, "variant": v,
                    "n_trades": len(pnls), "expectancy": exp,
                    "test_return": equity.iloc[-1] / equity.iloc[0] - 1,
                    "max_dd": float((equity / equity.cummax() - 1).min()),
                })
                if len(pnls):
                    all_seg_trades[v].extend(pnls.tolist())
    seg = pd.DataFrame(seg_rows)

    print("\n===== 单笔期望(12段滚动验证, BTC+ETH 合并) =====")
    for v in VARIANTS:
        pnls = np.array(all_seg_trades[v])
        n = len(pnls)
        exp = pnls.mean()
        boots = [RNG.choice(pnls, size=n, replace=True).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sub = seg[seg["variant"] == v]
        pos = int((sub["expectancy"] > 0).sum())
        tot = int(sub["expectancy"].notna().sum())
        summary_rows.append({
            "variant": v, "label": LABELS[v], "n_trades": n, "expectancy": exp,
            "ci_lo": lo, "ci_hi": hi, "pos_segments": f"{pos}/{tot}",
        })
        print(f"  {LABELS[v]:<20s} 交易{n:>4d}笔 | 单笔期望 {exp:+.4%} "
              f"(95%CI {lo:+.4%} ~ {hi:+.4%}) | 正收益段 {pos}/{tot}")
    print("\n结论方向: 若加规则后期望的95%CI 反而更低/更宽, 说明及时收手规则在历史上帮倒忙。")

    print("\n===== 样本外 2025+ 表现 =====")
    oos_rows = []
    for asset, path in ASSETS.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        sig = sma_gate(df)
        pat = detect(df)
        oos = df[df.index > "2024-12-31"]
        for v, kw in VARIANTS.items():
            sig_eff = effective_signal(sig, pat["dump"], pat["range"], pat["pump"], **kw)
            sig_eff_oos = sig_eff.reindex(oos.index)
            equity, trades, _ = run_backtest(oos, sig_eff_oos, DEFAULT_CONFIG,
                                             stop_pct=0.15, risk_pct=0.01)
            ret = equity.iloc[-1] / equity.iloc[0] - 1
            mdd = float((equity / equity.cummax() - 1).min())
            blocks = int(((sig_eff_oos < sig.reindex(oos.index)).sum()))
            oos_rows.append({"asset": asset, "variant": v, "return": ret, "max_dd": mdd,
                             "n_trades": len(trades), "blocks": blocks})
    oos = pd.DataFrame(oos_rows)
    pivot = oos.pivot_table(index="asset", columns="variant", values="return", aggfunc="first")
    print(pivot.map(lambda x: f"{x:+.2%}").to_string())
    mdd_p = oos.pivot_table(index="asset", columns="variant", values="max_dd", aggfunc="first")
    print("\n最大回撤:")
    print(mdd_p.map(lambda x: f"{x:.2%}").to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "pattern_backtest_summary.csv", index=False)
    seg.to_csv(RESULTS_DIR / "pattern_backtest_segments.csv", index=False)
    oos.to_csv(RESULTS_DIR / "pattern_backtest_oos.csv", index=False)

    print("\n===== 形态出现后的历史统计(离线, 用了未来数据仅供参考) =====")
    stats_all = []
    for asset, path in ASSETS.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        s = forward_stats(df)
        s["币种"] = asset
        stats_all.append(s)
    stats = pd.concat(stats_all)
    stats.to_csv(RESULTS_DIR / "pattern_stats.csv", index=False)
    for h in (24, 72):
        print(f"\n未来{h}小时:")
        sub = stats[(stats["未来小时"] == h) & (stats["形态"].isin(
            ["全部时间(基准)", "瀑布/砸盘", "疑似洗盘", "拉盘", "盘整", "回调"]))]
        sub = sub.groupby("形态").agg(
            次数=("次数", "sum"), 平均=("平均涨跌", "mean"), 上涨概率=("上涨概率", "mean"))
        print(sub.map(lambda x: f"{x:.2f}" if isinstance(x, float) else f"{x:.0f}").to_string())
    print(f"\n已保存: pattern_backtest_summary/segments/oos.csv, pattern_stats.csv")


if __name__ == "__main__":
    main()
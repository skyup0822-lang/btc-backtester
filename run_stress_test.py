# -*- coding: utf-8 -*-
"""压力测试: 历史瀑布/暴跌与"拉盘-砸盘"割韭菜事件中, 策略能否及时撤离。

对比两个版本:
  base     杂交策略(15%固定止损, 每笔风险1%)
  v_dump   加"瀑布/砸盘自动离场+24h冷却"
输出: 每个事件的峰值/谷底/深度, 策略当时是否持仓, 撤离时间与伤害, 对比"傻拿"的伤害。
"""
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config import DEFAULT_CONFIG, RESULTS_DIR
from strategies.trend_system import sma_gate
from run_patterns_backtest import effective_signal
from patterns import detect

MIN_DD = 0.10          # 从峰值回撤10%以上才叫"瀑布事件"
TRAP_PUMP = 0.10       # 72小时内涨10%以上
TRAP_DD = 0.15         # 随后从局部峰值回撤15%以上 = 割韭菜


def crash_events(close):
    peak = close.cummax()
    dd = close / peak - 1
    events, active = [], False
    start_ts = trough_ts = peak_ts = peak_px = None
    trough_px = None
    for ts, c in close.items():
        d = dd[ts]
        if not active:
            if d <= -MIN_DD:
                sub = dd.loc[:ts]
                zeros = sub[sub == 0].index
                peak_ts = zeros[-1] if len(zeros) else close.index[0]
                peak_px = float(close[peak_ts])
                active = True
                trough_ts, trough_px = ts, c
        else:
            if c < trough_px:
                trough_ts, trough_px = ts, c
            if c >= peak_px or (ts - trough_ts) > pd.Timedelta(days=45):
                events.append({
                    "peak_ts": peak_ts, "peak_px": peak_px,
                    "trough_ts": trough_ts, "trough_px": trough_px,
                    "depth": trough_px / peak_px - 1,
                    "end_ts": ts,
                })
                active = False
    if active:
        events.append({
            "peak_ts": peak_ts, "peak_px": peak_px,
            "trough_ts": trough_ts, "trough_px": trough_px,
            "depth": trough_px / peak_px - 1, "end_ts": close.index[-1],
        })
    return events


def trap_events(df):
    close = df["Close"]
    r72 = close.pct_change(72)
    events = []
    for i, ts in enumerate(close.index):
        if r72.iloc[i] >= TRAP_PUMP:
            fut = close.iloc[i:i + 168]
            peak_f = fut.cummax()
            dd_f = fut / peak_f - 1
            worst = dd_f.min()
            if worst <= -TRAP_DD:
                j = dd_f.idxmin()
                events.append({
                    "pump_end_ts": ts, "pump_px": float(close.iloc[i]),
                    "dump_ts": j, "dump_px": float(close[j]),
                    "depth": worst,
                })
                # 避免同一波行情重复计
                r72.iloc[i + 1:i + 168] = np.nan
    return events


def analyze(asset, df, sig_eff, label):
    equity, trades, _ = run_backtest(df, sig_eff, DEFAULT_CONFIG, stop_pct=0.15, risk_pct=0.01)
    out = []
    for ev in crash_events(df["Close"]):
        held = False
        ex_ts = ex_px = reason = None
        for t in trades:
            if t.entry_time <= ev["peak_ts"] < t.exit_time:
                held = True
                ex_ts, ex_px, reason = t.exit_time, t.exit_price, t.exit_reason
                break
        if held and ex_px and ex_px > 0:
            dmg = ex_px / ev["peak_px"] - 1
            avoid = dmg / ev["depth"]
            out.append({
                "asset": asset, "variant": label,
                "peak": str(ev["peak_ts"])[:10], "depth": ev["depth"],
                "held": "是", "exit": str(ex_ts)[:10],
                "dmg": dmg, "avoided_pct": avoid, "reason": reason,
            })
        else:
            out.append({
                "asset": asset, "variant": label,
                "peak": str(ev["peak_ts"])[:10], "depth": ev["depth"],
                "held": "否", "exit": "-", "dmg": 0.0,
                "avoided_pct": 1.0, "reason": "场外",
            })
    # 割韭菜事件
    for ev in trap_events(df):
        held = False
        ex_ts = ex_px = reason = None
        entry_during_pump = None
        for t in trades:
            if t.entry_time <= ev["pump_end_ts"] < t.exit_time:
                held = True
                entry_during_pump = ev["pump_end_ts"] - t.entry_time <= pd.Timedelta(hours=72)
                ex_ts, ex_px, reason = t.exit_time, t.exit_price, t.exit_reason
                break
        out.append({
            "asset": asset, "variant": label,
            "peak": "割韭菜:" + str(ev["pump_end_ts"])[:10], "depth": ev["depth"],
            "held": "是(追高)" if held and entry_during_pump else ("是" if held else "否"),
            "exit": str(ex_ts)[:10] if ex_ts else "-",
            "dmg": (ex_px / ev["pump_px"] - 1) if held and ex_px and ex_px > 0 else 0.0,
            "avoided_pct": ((ex_px / ev["pump_px"] - 1) / ev["depth"]) if held and ex_px and ex_px > 0 else 1.0,
            "reason": reason or ("场外" if not held else ""),
        })
    return pd.DataFrame(out)


def dedupe(df):
    """同一场瀑布(相同峰值时间)只保留最深的一次。"""
    df = df.copy()
    df["_depth"] = df["depth"]
    return (df.sort_values("_depth")
            .drop_duplicates(subset=["asset", "variant", "peak"], keep="first")
            .drop(columns="_depth"))


def main():
    print("=" * 110)
    print("压力测试: 历史瀑布与割韭菜事件中的撤离表现(回撤>=10%算瀑布)")
    print("=" * 110)
    frames = []
    for asset, path in [("BTC", "data/btc_usdt_1h.csv"), ("ETH", "data/eth_usdt_1h.csv")]:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        sig = sma_gate(df)
        pat = detect(df)
        base = effective_signal(sig, pat["dump"], pat["range"], pat["pump"], dump_exit=False)
        vdump = effective_signal(sig, pat["dump"], pat["range"], pat["pump"], dump_exit=True)
        for label, s in [("杂交策略", base), ("+瀑布自动离场", vdump)]:
            frames.append(analyze(asset, df, s, label))
    out = pd.concat(frames, ignore_index=True)
    trap = out[out["peak"].str.startswith("割韭菜")].copy()
    out = out[~out["peak"].str.startswith("割韭菜")]
    out = dedupe(out)
    out = pd.concat([out, trap], ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "stress_test_events.csv", index=False)

    # 汇总: 每个变体, 场内时的平均伤害 vs 瀑布平均深度
    for v in ["杂交策略", "+瀑布自动离场"]:
        sub = out[out["variant"] == v]
        held = sub[sub["held"].str.startswith("是") | (sub["held"] == "是")]
        avoided = sub[sub["held"] == "否"]
        print(f"\n【{v}】共 {len(sub)} 个事件: 场外躲过 {len(avoided)} 个, 场内经历 {len(held)} 个")
        if len(held):
            print(f"  场内事件平均瀑布深度 {held['depth'].mean():.1%}, 策略实际平均伤害 {held['dmg'].mean():.1%}, 平均躲过跌幅的 {held['avoided_pct'].mean():.0%}")
            worst = held.loc[held["dmg"].idxmin()]
            print(f"  最惨一次: {worst['asset']} {worst['peak']} 瀑布 {worst['depth']:.1%}, 策略伤害 {worst['dmg']:.1%}, 原因 {worst['reason']}")

    print("\n===== 最深瀑布事件明细 =====")
    show = out[~out["peak"].str.startswith("割韭菜")].sort_values("depth").head(16)
    print(show.to_string(index=False, formatters={
        "depth": lambda x: f"{x:.1%}", "dmg": lambda x: f"{x:.1%}",
        "avoided_pct": lambda x: f"{x:.0%}",
    }))
    print("\n===== 割韭菜事件(先拉盘再砸盘) =====")
    t = trap[trap["variant"] == "杂交策略"].sort_values("depth")
    if len(t) == 0:
        print("  历史数据里没有满足条件的大规模拉盘-砸盘事件")
    else:
        print(t.to_string(index=False, formatters={
            "depth": lambda x: f"{x:.1%}", "dmg": lambda x: f"{x:.1%}",
            "avoided_pct": lambda x: f"{x:.0%}",
        }))
    print(f"\n已保存: results/stress_test_events.csv")


if __name__ == "__main__":
    main()
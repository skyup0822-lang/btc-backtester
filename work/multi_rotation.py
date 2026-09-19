# -*- coding: utf-8 -*-
"""多币种动量轮动回测(C+ 思路的忠实复刻 + 去幸存者偏差的币池)。

规则照搬对方 research/five_year_ab_backtest.py 的 prepare_rotation_frames:
  动量 = 0.30*30日 + 0.50*90日 + 0.20*180日 涨幅
  评分 = 动量 / (60日年化波动率, 下限0.05)
  币池 = 固定名单里按 30日中位成交额 排前 15
  趋势 = 收盘 > SMA100
  大盘 = BTC收盘>SMA200 且 BTC SMA50>SMA200 且 广度>=50%
  入场 = 排名前 3; 持有容忍到前 5
  止损 = ATR14*4, 夹在 8%~25%

关键差别: 币池是「2021-09-09 当天按成交额排出来的前 25 名」, 只看当天之前的数据,
里面天然包含后来归零的币(SRM/FTT/LUNC 等), 不是拿今天的榜单回头测。

成交约定(防未来函数): 当天收盘后才知道信号, 用**第二天开盘价**成交。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "work" / "multi_data"

import numpy as np
import pandas as pd

FEE = 0.001
SLIP = 0.0005
START = pd.Timestamp("2021-09-09", tz="UTC")
OOS = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-09-17", tz="UTC")
INIT = 10000.0


def load_panel():
    """读币池数据。必须防"代码被换标的"这种拼接:

    LUNAUSDT 在 2022-05-13 收 $0.00005, 2022-05-31 变成 $8.87 —— 币安把旧 LUNA(已归零)
    换成了 Terra 2.0 的新 LUNA, 单日"涨幅" +177399 倍。不截断的话回测会凭空多出上千万倍。
    这里用"单日暴涨超过 10 倍"作为拼接信号, 把序列截在断点之前(等于该币退市)。
    """
    basket = json.loads((DATA / "_basket.json").read_text(encoding="utf-8"))
    frames, truncated = {}, []
    for s in basket:
        f = DATA / f"{s}_1d.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["t"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
        d = d.set_index("t").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if len(d) > 2:
            jump = d["close"].pct_change()
            bad = jump[jump > 9.0]          # 单日超过 +900%
            if len(bad):
                cut = bad.index[0]
                truncated.append((s, cut, float(jump.loc[cut])))
                d = d[d.index < cut]
        frames[s] = d
    if truncated:
        print("!! 检测到代码换标的(已截断, 按退市处理):")
        for s, t, j in truncated:
            print(f"   {s:<10} 截于 {t:%Y-%m-%d}, 那天原始涨幅 {j:,.0f} 倍")
        print()
    return basket, frames


def build_panel(frames, universe_size=15):
    close = pd.concat({s: f["close"] for s, f in frames.items()}, axis=1)
    op = pd.concat({s: f["open"] for s, f in frames.items()}, axis=1)
    high = pd.concat({s: f["high"] for s, f in frames.items()}, axis=1)
    low = pd.concat({s: f["low"] for s, f in frames.items()}, axis=1)
    qv = pd.concat({s: f["quote_volume"] for s, f in frames.items()}, axis=1)

    mom = (0.30 * (close / close.shift(30) - 1)
           + 0.50 * (close / close.shift(90) - 1)
           + 0.20 * (close / close.shift(180) - 1))
    vol = close.pct_change().rolling(60).std() * math.sqrt(365)
    score = mom / vol.clip(lower=0.05)

    prev = close.shift(1)
    # 逐币算 TR 再拼: 直接按列拼会产生重复列名
    trs = {}
    for s in frames:
        h, l, c = high[s], low[s], close[s].shift(1)
        trs[s] = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    tr = pd.concat(trs, axis=1)
    atr_pct = tr.rolling(14).mean() / close

    sma100 = close.rolling(100).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    dvol = qv.rolling(30).median()

    liq_rank = dvol.rank(axis=1, ascending=False, method="first")
    in_universe = liq_rank <= min(universe_size, len(frames))
    above = close > sma100
    breadth = (above & in_universe).sum(axis=1) / in_universe.sum(axis=1).replace(0, np.nan)
    bt = "BTCUSDT" if "BTCUSDT" in frames else list(frames)[0]
    regime = (close[bt] > sma200[bt]) & (sma50[bt] > sma200[bt]) & (breadth >= 0.50)

    eligible = score.where(in_universe & above & score.gt(0))
    ranks = eligible.rank(axis=1, ascending=False, method="first")
    risk_signal = pd.DataFrame({s: (regime & in_universe[s] & above[s]) for s in frames})

    return {
        "open": op, "close": close, "high": high, "low": low,
        "rank": ranks, "risk_signal": risk_signal,
        "stop_pct": (atr_pct * 4).clip(lower=0.08, upper=0.25),
        "regime": regime,
    }


def simulate(panel, start, end, risk=0.0075, max_pos=3, max_expo=0.6,
             entry_rank=3, hold_rank=5, rebalance_days=3):
    """注意: 对方还有一个"当日盘中亏 4% 就停手"的风控, 那需要小时级数据才复刻得出。
    这里跑的是日线, 复刻不了 —— 少这一层只会让它更难看, 不会更好看, 所以偏保守。"""
    dates = panel["open"].index
    dates = dates[(dates >= start) & (dates <= end)]
    cash = INIT
    pos = {}
    rows, trades = [], []
    last_close = {}
    rebal_i = 0

    for i, d in enumerate(dates):
        o, c, lo = panel["open"].loc[d], panel["close"].loc[d], panel["low"].loc[d]

        # 用"昨天收盘后才知道的"信号
        if i == 0:
            rows.append({"t": d, "equity": cash})
            continue
        pd_ = dates[i - 1]
        rank_prev, rs_prev = panel["rank"].loc[pd_], panel["risk_signal"].loc[pd_]
        stop_prev = panel["stop_pct"].loc[pd_]

        # --- 1) 止损(当天盘中触发) ---
        for s in list(pos):
            p = pos[s]
            ov, lv = o.get(s, np.nan), lo.get(s, np.nan)
            hit = None
            if np.isfinite(ov) and ov <= p["stop"]:
                hit = ov                      # 跳空 -> 按开盘价, 更差
            elif np.isfinite(lv) and lv <= p["stop"]:
                hit = p["stop"]
            if hit is not None:
                proceeds = p["units"] * hit * (1 - FEE)
                cash += proceeds
                trades.append({"sym": s, "entry": p["entry"], "exit": hit,
                               "pnl": proceeds - p["cost"], "reason": "stop",
                               "t_exit": d})
                del pos[s]

        # --- 2) 数据断了(退市/合并) -> 按最后已知价离场 ---
        for s in list(pos):
            if not np.isfinite(o.get(s, np.nan)) and not np.isfinite(c.get(s, np.nan)):
                last = last_close.get(s, pos[s]["entry"])
                proceeds = pos[s]["units"] * last * (1 - FEE)
                cash += proceeds
                trades.append({"sym": s, "entry": pos[s]["entry"], "exit": last,
                               "pnl": proceeds - pos[s]["cost"], "reason": "delisted",
                               "t_exit": d})
                del pos[s]

        # --- 3) 大盘转空 -> 全部离场(风控) ---
        if not bool(panel["regime"].loc[pd_]):
            for s in list(pos):
                px = o.get(s, c.get(s, np.nan))
                if not np.isfinite(px):
                    continue
                fill = px * (1 - SLIP)
                proceeds = pos[s]["units"] * fill * (1 - FEE)
                cash += proceeds
                trades.append({"sym": s, "entry": pos[s]["entry"], "exit": fill,
                               "pnl": proceeds - pos[s]["cost"], "reason": "regime",
                               "t_exit": d})
                del pos[s]

        # --- 4) 定期再平衡: 掉出前 hold_rank 的换掉, 补进前 entry_rank ---
        if i % rebalance_days == 0:
            for s in list(pos):
                r = rank_prev.get(s, np.nan)
                if not (np.isfinite(r) and r <= hold_rank) or not bool(rs_prev.get(s, False)):
                    px = o.get(s, c.get(s, np.nan))
                    if not np.isfinite(px):
                        continue
                    fill = px * (1 - SLIP)
                    proceeds = pos[s]["units"] * fill * (1 - FEE)
                    cash += proceeds
                    trades.append({"sym": s, "entry": pos[s]["entry"], "exit": fill,
                                   "pnl": proceeds - pos[s]["cost"], "reason": "rank",
                                   "t_exit": d})
                    del pos[s]

            eq = cash + sum(p["units"] * (o.get(s, p["entry"]) if np.isfinite(o.get(s, np.nan)) else p["entry"])
                            for s, p in pos.items())
            expo = sum(p["units"] * (o.get(s, p["entry"]) if np.isfinite(o.get(s, np.nan)) else p["entry"])
                       for s, p in pos.items())
            room = max(0.0, eq * max_expo - expo)
            cands = [s for s in rank_prev.index
                     if np.isfinite(rank_prev.get(s, np.nan)) and rank_prev[s] <= entry_rank
                     and bool(rs_prev.get(s, False)) and s not in pos]
            cands.sort(key=lambda s: rank_prev[s])
            for s in cands:
                if len(pos) >= max_pos or room <= 0:
                    break
                px = o.get(s, np.nan)
                if not np.isfinite(px) or px <= 0:
                    continue
                sp = float(stop_prev.get(s, 0.10))
                if not (0 < sp < 1):
                    continue
                budget = min(eq * risk / sp, room, cash)
                if budget <= INIT * 0.001:
                    continue
                fill = px * (1 + SLIP)
                units = budget / (fill * (1 + FEE))
                cost = budget
                cash -= cost
                pos[s] = {"units": units, "entry": fill, "stop": fill * (1 - sp), "cost": cost}
                room -= cost
                trades.append({"sym": s, "entry": fill, "exit": None, "pnl": None,
                               "reason": "open", "t_exit": d})
            for s, p in pos.items():
                last_close[s] = c.get(s, last_close.get(s, p["entry"]))

        # --- 净值 ---
        eq = cash
        for s, p in pos.items():
            px = c.get(s, np.nan)
            if not np.isfinite(px):
                px = last_close.get(s, p["entry"])
            eq += p["units"] * px
        rows.append({"t": d, "equity": eq, "expo": (eq - cash) / eq if eq > 0 else 0.0})

    eqc = pd.DataFrame(rows).set_index("t")["equity"]
    expo = pd.DataFrame(rows).set_index("t")["expo"]
    done = [t for t in trades if t["pnl"] is not None]
    return eqc, done, expo


def stats(eqc, trades, expo, label):
    if len(eqc) < 30:
        return None
    r = eqc.pct_change().dropna()
    up, dn = (r > 1e-9), (r < -1e-9)
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    return {
        "label": label,
        "days": len(r),
        "total": float(eqc.iloc[-1] / eqc.iloc[0] - 1),
        "dd": float((eqc / eqc.cummax() - 1).min()),
        "up_days": float(up.mean()),
        "flat_days": float((~up & ~dn).mean()),
        "down_days": float(dn.mean()),
        "in_mkt": float((expo > 0.01).mean()),
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else np.nan,
        "pf": (sum(wins) / abs(sum(losses))) if losses else np.inf,
        "cagr": float((eqc.iloc[-1] / eqc.iloc[0]) ** (365 / max(len(eqc), 1)) - 1),
    }


def main():
    basket, frames = load_panel()
    print(f"币池 {len(basket)} 个: {', '.join(basket)}")
    missing = [s for s in basket if s not in frames]
    if missing:
        print(f"没抓到数据的: {missing}")
    panel = build_panel(frames)
    print(f"数据区间: {panel['close'].index[0]:%Y-%m-%d} ~ {panel['close'].index[-1]:%Y-%m-%d}")
    print()

    configs = [
        ("C+ 原版(风险0.75%/仓位上限60%)", dict(risk=0.0075, max_expo=0.6)),
        ("C+ 激进版(风险2%/仓位上限90%)", dict(risk=0.02, max_expo=0.9)),
        ("C+ 原版 每周才轮动一次", dict(risk=0.0075, max_expo=0.6, rebalance_days=7)),
    ]
    out_rows = []
    for tag, period in (("样本外 2025-01 起", (OOS, END)),
                        ("全区间 2021-09 起", (START, END)),
                        ("前段 2021-09~2024-12", (START, OOS))):
        print("=" * 92)
        print(tag)
        print("=" * 92)
        print(f"{'方案':<32}{'天数':>6}{'总收益':>10}{'年化':>8}{'最大回撤':>9}"
              f"{'在场':>7}{'上涨':>7}{'持平':>7}{'下跌':>7}{'交易':>6}{'胜率':>7}{'盈利因子':>9}")
        for label, kw in configs:
            eqc, tr, expo = simulate(panel, period[0], period[1], **kw)
            st = stats(eqc, tr, expo, label)
            if st:
                out_rows.append({**st, "period": tag})
                print(f"{label:<32}{st['days']:>6}{st['total'] * 100:>9.1f}%"
                      f"{st['cagr'] * 100:>7.1f}%{st['dd'] * 100:>8.1f}%"
                      f"{st['in_mkt'] * 100:>6.1f}%{st['up_days'] * 100:>6.1f}%"
                      f"{st['flat_days'] * 100:>6.1f}%{st['down_days'] * 100:>6.1f}%"
                      f"{st['trades']:>6}{st['win_rate'] * 100:>6.1f}%{st['pf']:>9.2f}")
        # BTC 买入持有基准
        btc = panel["close"]["BTCUSDT"].loc[period[0]:period[1]].dropna()
        if len(btc) > 30:
            br = btc.pct_change().dropna()
            tot = btc.iloc[-1] / btc.iloc[0] - 1
            dd = (btc / btc.cummax() - 1).min()
            cagr = (1 + tot) ** (365 / len(btc)) - 1
            print(f"{'BTC 买入持有(基准)':<32}{len(br):>6}{tot * 100:>9.1f}%"
                  f"{cagr * 100:>7.1f}%{dd * 100:>8.1f}%{'100%':>6}"
                  f"{(br > 0).mean() * 100:>6.1f}%{0.0:>6.1f}%{(br < 0).mean() * 100:>6.1f}%"
                  f"{'—':>6}{'—':>6}{'—':>9}")
        print()

    pd.DataFrame(out_rows).to_csv(ROOT / "results" / "multi_rotation.csv", index=False)
    print("已保存: results/multi_rotation.csv")


if __name__ == "__main__":
    main()

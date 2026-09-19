# -*- coding: utf-8 -*-
"""Polymarket「BTC Multi Strikes」事件 —— 遍历所有档位做批量 edge 回测。

一个事件(如 bitcoin-above-on-july-29-2026)内含多个"涨破 $K"档位子市场。
脚本: 对每个档位拉 token 历史价 + 用 vol 模型算 P(价>K), 得 edge;
      并按"edge≥min_edge 买入持有到结算"算真实盈亏(已知结算结果)。

重点: 档位里既有"接近当时价(不确定, 近五五开)"又有"远低于/远高于(近稳赢)", 
      正好检验 edge 是只在近稳赢(模型高估)为正, 还是在不确定(真 edge)也为正。

用法: python work/poly_batch.py  <event_slug> ...   (默认 bitcoin-above-on-july-29-2026)
"""
import json
import re
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

import numpy as np
import pandas as pd

from poly_sim import model_prob_above, realized_vol
from poly_auto import http_json, price_history, btc_series

GAMMA = "https://gamma-api.polymarket.com"
MIN_EDGE = 0.05
COST = 0.01


def event_strikes(slug):
    """事件 -> 每个档位 {token,K,end_dt,outcome} 列表。"""
    try:
        data = http_json(GAMMA + "/events?" + urllib.parse.urlencode({"slug": slug}))
    except Exception:
        return []
    if not data:
        return []
    strikes = []
    for mk in data[0].get("markets", []):
        tokens = json.loads(mk.get("clobTokenIds") or "[]")
        if not tokens:
            continue
        q = mk.get("question", "") or mk.get("groupItemTitle", "")
        m = re.search(r"above\s*\$?\s*([\d,]+)", q.lower())
        K = float(m.group(1).replace(",", "")) if m else None
        op = json.loads(mk.get("outcomePrices") or "[]")
        outcome = 1.0 if (op and float(op[0]) >= 0.5) else 0.0
        end = mk.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except Exception:
            end_dt = None
        strikes.append({"token": tokens[0], "K": K, "end_dt": end_dt, "outcome": outcome})
    return strikes


def run_strike(st):
    token, K, end_dt, outcome = st["token"], st["K"], st["end_dt"], st["outcome"]
    if not token or K is None or end_dt is None:
        return None
    start_ts = int((end_dt - timedelta(days=7)).timestamp())  # 只需最近一周贴近现价
    try:
        hist = price_history(token, start_ts)
    except Exception:
        return None
    if not hist:
        return None
    _, close = btc_series()
    rows = []
    for t, p_market in hist:
        dt = pd.Timestamp(t, unit="s", tz="UTC")
        h = close[close.index <= dt]
        if len(h) < 90:
            continue
        S0 = float(h.iloc[-1])
        vol = realized_vol(h, 60)
        if not np.isfinite(vol):
            continue
        T = max(1, (end_dt - dt).days)
        p_model = model_prob_above(S0, K, T, vol)
        rows.append({"p_model": p_model, "p_market": p_market, "S0": S0,
                     "edge": p_model - p_market, "dist": (S0 - K) / K if K > 0 else np.nan})
    df = pd.DataFrame(rows).dropna(subset=["edge"])
    if df.empty:
        return None
    trig = df[df["edge"] >= MIN_EDGE]
    if len(trig):
        p_entry = trig.iloc[0]["p_market"]
        pnl = (outcome - p_entry) - COST * p_entry
        ret = (pnl / p_entry) if p_entry > 0 else 0.0
    else:
        pnl = ret = np.nan
    return {"K": K, "outcome": outcome, "n": len(df), "edge_mean": df["edge"].mean(),
            "dist_med": float(df["dist"].median()), "triggered": len(trig),
            "pnl": pnl, "ret": ret}


def main():
    slugs = sys.argv[1:] or ["bitcoin-above-on-july-29-2026"]
    all_rows = []
    for slug in slugs:
        print("\n" + "=" * 96)
        print(f"【事件】{slug}")
        print("=" * 96)
        strikes = event_strikes(slug)
        print(f"  档位数: {len(strikes)}")
        rows = []
        for st in strikes:
            r = run_strike(st)
            if r:
                rows.append(r)
        if not rows:
            print("  都无法回测")
            continue
        df = pd.DataFrame(rows).sort_values("K")
        all_rows.append(df)
        print(df[["K", "outcome", "n", "edge_mean", "dist_med", "triggered", "pnl", "ret"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if not all_rows:
        print("没有结果")
        return
    big = pd.concat(all_rows, ignore_index=True)
    print("\n" + "=" * 96)
    print("汇总 + 关键分档")
    print("=" * 96)
    print(f"  总档位 {len(big)} | 平均edge {big['edge_mean'].mean():+.2%}"
          f" | 触发买入 {int((big['triggered']>0).sum())} | 平均净期望 {big['pnl'].mean():+.4f}")
    big["bucket"] = pd.cut(big["dist_med"], [-1, 0.6, 0.75, 0.92, 1.08, 1.25, 1e9],
                           labels=["价远低于K(暴跌才中)", "~-25%", "~-8%", "≈现价(不确定)", "+25%", "远高于K(近稳赢)"])
    for b, g in big.groupby("bucket", observed=True):
        if len(g) == 0:
            continue
        print(f"  {b:<18} n={len(g)} | 平均edge {g['edge_mean'].mean():+.2%}"
              f" | 触发 {int((g['triggered']>0).sum())} | 平均pnl {g['pnl'].mean():+.4f}")

    out = ROOT / "results" / "poly_batch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(out, index=False)
    print(f"\n已保存: {out.name}")


if __name__ == "__main__":
    main()

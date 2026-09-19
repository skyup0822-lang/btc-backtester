# -*- coding: utf-8 -*-
"""Polymarket 加密价格市场 —— 自动收集历史数据 + 模型 vs 市场 edge 回测。

全自动(在你联网机器上跑, 一次完成):
  1. 按市场 slug 从 Gamma API 拿 tokenID + 阈值 + 到期日;
  2. 从 clob 拉该 YES token 的历史价 (p=市场隐含概率);
  3. 从 OKX/Gate 拉 BTC K线, 算历史波动率 + 现价;
  4. 在每个历史时点用模型算 P(价>K 到到期日), 得 edge = 模型 - 市场;
  5. 统计 edge 分布 + 一个"买入当 edge 超阈值"的净期望(扣成本)。

用法: python work/poly_auto.py  <slug1> [slug2 ...]
例:   python work/poly_auto.py  bitcoin-above-on-july-29-2026  bitcoin-above-on-august-8-2026
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import numpy as np
import pandas as pd

from poly_sim import model_prob_above, realized_vol  # 已测模型
import data_source as ds
from paper_trader_okx import fetch_candles as okx_candles

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KM = 60 * 60 * 1000  # 1h ms


def http_json(url, timeout=30, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    raise last


def get_market(slug):
    """从 Gamma API 拿一个市场: 返回 (token_id, question, end_dt)。先 market 后 event。"""
    for path in (f"{GAMMA}/markets?{urllib.parse.urlencode({'slug': slug})}",
                 f"{GAMMA}/events?{urllib.parse.urlencode({'slug': slug})}"):
        try:
            data = http_json(path)
        except Exception:
            continue
        if not data:
            continue
        # 可能 events 嵌套 markets
        mk = data[0] if isinstance(data, list) else data
        if "markets" in mk:  # event 对象
            cands = mk["markets"]
        else:
            cands = [mk]
        pick = cands[0]
        tokens = json.loads(pick.get("clobTokenIds") or "[]")
        end = pick.get("endDate") or ""
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return tokens[0], pick.get("question", "") or pick.get("groupItemTitle", ""), end_dt
    return None, slug, None


def price_history(token_id, start_ts):
    url = (CLOB + "/prices-history?" + urllib.parse.urlencode(
        {"startTs": int(start_ts), "market": token_id, "fidelity": 60}))
    d = http_json(url)
    return [(int(x["t"]), float(x["p"])) for x in d.get("history", [])]


def btc_series():
    """返回 (last_close, close_series)。先 OKX, 失败切 Gate。"""
    try:
        df = okx_candles(2000)
        close = df.set_index(pd.to_datetime(df["close_time_ms"], unit="ms", utc=True))["close"].astype(float)
        return float(close.iloc[-1]), close
    except Exception:
        pass
    df = ds.gate_candles(2000)
    close = df.set_index(pd.to_datetime(df["close_time_ms"], unit="ms", utc=True))["close"].astype(float)
    return float(close.iloc[-1]), close


def run(slug, cost=0.01, min_edge=0.05):
    print(f"\n{'='*90}\n【市场】{slug}\n{'='*90}")
    token, question, end_dt = get_market(slug)
    if not token:
        print("  拿不到市场数据(检查 slug 或网络)")
        return
    print(f"  token(前12): {token[:12]}… | 问题: {question} | 到期: {end_dt:%Y-%m-%d}")

    # 用市场开始前一段时间作 startTs
    start_ts = int((end_dt - timedelta(days=45)).timestamp())
    hist = price_history(token, start_ts)
    if not hist:
        print("  历史价为空")
        return
    print(f"  历史价点数: {len(hist)} ({hist[0][0]} ~ {hist[-1][0]})")

    _, close = btc_series()
    # 提取阈值 K 与方向
    m = re.search(r"above\s*\$?\s*([\d,]+)", (question or "").lower())
    K = float(m.group(1).replace(",", "")) if m else None
    print(f"  阈值 K = {K if K else '未知(解析失败)'}")

    rows = []
    for t, p_market in hist:
        dt = pd.Timestamp(t, unit="s", tz="UTC")
        hist_btc = close[close.index <= dt]
        if K is None or len(hist_btc) < 90:
            continue
        S0 = float(hist_btc.iloc[-1])
        vol = realized_vol(hist_btc, 60)
        if not np.isfinite(vol):
            continue
        T_days = max(1, (end_dt - dt).days)
        p_model = model_prob_above(S0, K, T_days, vol) if m else float("nan")
        edge = p_model - p_market
        rows.append({"t": dt, "S0": S0, "vol": vol, "p_model": p_model,
                     "p_market": p_market, "edge": edge})

    if not rows:
        print("  无法计算(数据不足)")
        return
    df = pd.DataFrame(rows).dropna(subset=["edge"])
    print(f"\n  样本点 {len(df)} | edge 均值 {df['edge'].mean():+.2%} | 中位 {df['edge'].median():+.2%}"
          f" | 标准差 {df['edge'].std():.2%}")
    print(f"  edge>0 占比 {(df['edge']>0).mean():.1%} | 最大edge {df['edge'].max():+.2%} | 最小 {df['edge'].min():+.2%}")

    # 简单策略: 当 edge > min_edge 买入持有到结算, 扣成本后净期望(每份 $1)
    # 每笔: 以当时市场价买YES, edge>0 的样本结算为 1 的概率 = p_model(近似真实), 收益 = p_model - p_market (期望)
    trig = df[df["edge"] > min_edge]
    if len(trig):
        net = (trig["p_model"] - trig["p_market"]).mean() - cost
        print(f"  [策略] edge>{min_edge:.0%} 触发 {len(trig)} 次 | 平均净期望(扣{cost:.0%}成本) {net:+.2%}"
              f" | {'正(有edge)' if net>0 else '≤0(无edge/被成本吃掉)'}")
    else:
        print(f"  [策略] edge>{min_edge:.0%} 从未触发, 无交易")

    out = ROOT / "results" / f"poly_backtest_{slug}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  结果已存: {out.name}")


if __name__ == "__main__":
    slugs = sys.argv[1:] or ["bitcoin-above-on-july-29-2026"]
    for s in slugs:
        run(s)

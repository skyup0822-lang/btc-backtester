# -*- coding: utf-8 -*-
"""Polymarket 加密价格阈值 — 向前模拟盘(在你机器上联网运行, 虚拟本金)。

每 POLL 分钟做一轮:
  1. 拉 BTC 1h K线(先 OKX, 不通切 Gate.io, 复用项目已有 data_source) -> 算历史波动率(只用过去) + 现价;
  2. 拉 Polymarket 加密阈值市场(Gamma API, 文本过滤 question 含 Bitcoin/BTC 且 above/below $K);
  3. 对每个市场: 用波动率模型算 P(价>K) 的"真实概率", 与市场价(隐含概率)比较 => edge;
  4. edge 超过 min_edge -> 记录"计划单"(挂 maker 价=你的公平价), 用 Kelly 定虚拟仓位;
  5. 查询已结算市场的结果, 更新虚拟 P&L, 写 CSV(poly_paper_pnl.csv)。

注意: 这是"向前模拟盘", 只记录计划单与虚拟 P&L, 不真实下单。
真实下单需要: Polymarket 私钥 + Polygon USDC(下一步再做)。运行前确保 python 联网。

用法: python work/poly_paper.py    (Ctrl+C 停止)
"""
import dataclasses
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import numpy as np
import pandas as pd

from poly_sim import model_prob_above, realized_vol  # 复用已测的模型/波动率
import data_source as ds  # Gate.io 备用源
from paper_trader_okx import fetch_candles as okx_candles  # OKX 1h K线

GAMMA = "https://gamma-api.polymarket.com"
PNL_CSV = ROOT / "results" / "poly_paper_pnl.csv"


@dataclasses.dataclass
class Cfg:
    bankroll: float = 140.0          # 虚拟本金 USDC (≈1000 RMB)
    poll_seconds: int = 300          # 每 5 分钟一轮
    max_kelly: float = 0.25
    min_edge: float = 0.05           # 最小 edge 才记单
    min_abs_prob: float = 0.03       # 概率太极端不下
    vol_lookback_days: int = 60
    watch_days_max: int = 90         # 只看未来 <=90 天到期的市场
    market_slugs: list = None        # 手工指定市场 slug(最可靠); None=走自动发现


def http_get(url, timeout=25, retries=3):
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


def fetch_btc():
    """返回 (current_price, close_series)。先 OKX, 失败切 Gate。"""
    try:
        df = okx_candles(300)
        close = df.set_index(pd.to_datetime(df["close_time_ms"], unit="ms", utc=True))["close"]
        px = float(close.iloc[-1])
        return px, close
    except Exception:
        pass
    df = ds.gate_candles(2000)
    if len(df) == 0:
        raise RuntimeError("OKX 与 Gate 都拿不到 BTC K线")
    close = df.set_index(pd.to_datetime(df["close_time_ms"], unit="ms", utc=True))["close"]
    px = float(close.iloc[-1])
    return px, close


def parse_threshold(question):
    """从 "Will Bitcoin be above $100,000 by Dec 31?" 里抓 (金额, 方向 above/below)。"""
    q = question.lower()
    m = re.search(r"(above|over|higher than|exceed|greater than)\s*\$?\s*([\d,]+)", q)
    if m:
        return float(m.group(2).replace(",", "")), "above" if m.group(1) in ("above", "over", "exceed", "higher than", "greater than") else "below"
    m = re.search(r"(below|under|lower than|less than)\s*\$?\s*([\d,]+)", q)
    if m:
        return float(m.group(2).replace(",", "")), "below"
    return None, None


def fetch_pm_markets(cfg):
    """拉 Polymarket 市场, 返回可解析的列表。支持两种模式:
       - cfg.market_slugs 非空: 逐个按 slug 精确拉取(最可靠, 从网站复制 slug 填入);
       - 否则: 拉成交量前100, 文本过滤"加密价格阈值"(自动发现, 但加密市场可能不出现)。
    """
    if cfg.market_slugs:
        data = []
        for slug in cfg.market_slugs:
            try:
                found = http_get(GAMMA + "/markets?" + urllib.parse.urlencode({"slug": slug}))
                if isinstance(found, list):
                    data.extend(found)
            except Exception:
                continue
        if not data:
            print(f"  (警告: 按 slug {cfg.market_slugs} 拉取失败, 回退自动发现)")
            data = http_get(GAMMA + "/markets?order=volume24hr&ascending=false&limit=100"
                            "&closed=false&active=true")
    else:
        data = http_get(GAMMA + "/markets?order=volume24hr&ascending=false&limit=100"
                        "&closed=false&active=true")
    out = []
    for mk in data:
        q = mk.get("question", "") or ""
        if not (("bitcoin" in q.lower()) or ("btc" in q.lower())):
            continue
        th, direction = parse_threshold(q)
        if th is None:
            continue
        end = mk.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except Exception:
            continue
        # 只看未来 <= watch_days_max 天
        if (end_dt - datetime.now(timezone.utc)).days > cfg.watch_days_max:
            continue
        try:
            prices = json.loads(mk.get("outcomePrices") or "[]")
        except Exception:
            continue
        yes_price = float(prices[0]) if prices else None
        if yes_price is None or not (0 <= yes_price <= 1):
            continue
        out.append({
            "id": mk.get("id"), "question": q, "threshold": th,
            "direction": direction, "end": str(end_dt)[:10],
            "yes_price": yes_price, "end_dt": end_dt,
        })
    return out


def compute_and_log(cfg, btc_px, close, market):
    """对单个市场算 edge + Kelly, 返回一条记录(dict)。"""
    th = market["threshold"]
    T_days = max(1, (market["end_dt"] - datetime.now(timezone.utc)).days)
    vol = realized_vol(close, cfg.vol_lookback_days)
    # 模型概率: P(价>T) 或 P(价<T)
    if market["direction"] == "above":
        p_touch = model_prob_above(btc_px, th, T_days, vol)
    else:
        p_touch = 1.0 - model_prob_above(btc_px, th, T_days, vol)
    # 若市场是"BTC below K", 其隐含YES概率=p(below)
    p_model = p_touch
    p_market = market["yes_price"]
    edge = p_model - p_market
    if abs(edge) < cfg.min_edge or p_model < cfg.min_abs_prob or p_model > 1 - cfg.min_abs_prob:
        return {**market, "btc": btc_px, "vol": vol, "p_model": p_model,
                "edge": edge, "side": "-", "bet_frac": 0.0, "bet_usd": 0.0}
    side = "YES" if edge > 0 else "NO"
    p_use = p_model if side == "YES" else 1 - p_model
    price_use = p_market if side == "YES" else 1 - p_market
    # Kelly
    if price_use <= 0 or price_use >= 1:
        frac = 0.0
    else:
        b = 1.0 / price_use - 1.0
        f = (p_use * b - (1 - p_use)) / b
        frac = float(np.clip(f, 0.0, cfg.max_kelly))
    return {**market, "btc": btc_px, "vol": vol, "p_model": p_model,
            "edge": edge, "side": side, "bet_frac": frac,
            "bet_usd": round(cfg.bankroll * frac, 2)}


def main():
    cfg = Cfg()
    print("=" * 96)
    print(f"Polymarket 加密价格阈值 · 向前模拟盘 | 虚拟本金 {cfg.bankroll} USDC (≈1000 RMB)"
          f" | 每 {cfg.poll_seconds}s 一轮 | Ctrl+C 停止")
    print("=" * 96)
    # 尝试初始化 CSV 头
    if not PNL_CSV.exists():
        PNL_CSV.parent.mkdir(parents=True, exist_ok=True)
        PNL_CSV.write_text("time,id,question,btc,vol,p_model,yes_price,edge,side,bet_usd\n",
                           encoding="utf-8")
    while True:
        try:
            btc_px, close = fetch_btc()
            vol = realized_vol(close, cfg.vol_lookback_days)
            print(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}] BTC {btc_px:,.0f} | 波动率 {vol:.1%}")
            markets = fetch_pm_markets(cfg)
            print(f"  匹配到 {len(markets)} 个加密价格阈值市场(未来{cfg.watch_days_max}天)")
            rows = []
            for mk in markets:
                r = compute_and_log(cfg, btc_px, close, mk)
                rows.append(r)
                print(f"    {mk['end']} {'涨破' if mk['direction']=='above' else '跌破'} "
                      f"${mk['threshold']:,.0f} | 模型 {r['p_model']:.1%} vs 市价 {mk['yes_price']:.1%} "
                      f"| edge {r['edge']:+.1%} | {r['side']} 仓位 {r['bet_usd']:.2f} USDC")
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(PNL_CSV, mode="a", header=False, index=False,
                          encoding="utf-8")  # 简化: 追加
            print("  (模拟盘: 只记录计划单, 不真实下单; 结算P&L后续版本加)")
        except Exception as e:
            print(f"  本轮异常({type(e).__name__}): {e}, {cfg.poll_seconds}s后重试")
        for _ in range(cfg.poll_seconds):
            time.sleep(1)


if __name__ == "__main__":
    main()

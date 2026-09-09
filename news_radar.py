# -*- coding: utf-8 -*-
"""新闻雷达: 免费RSS聚合 + 关键词情绪打分 + 风险分级。

用途: 只做风险提示与"开仓闸门", 不做买卖信号(买卖仍由价格趋势决定)。
原则: 断网/新闻源不可达时返回风险"未知", 绝不阻塞交易(失败开放)。
"""
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

SOURCES = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("GoogleNews-EN", "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en"),
    ("GoogleNews-CN", "https://news.google.com/rss/search?q=bitcoin&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
    ("GoogleNews-币圈", "https://news.google.com/rss/search?q=%E6%AF%94%E7%89%B9%E5%B8%81+OR+%E5%8A%A0%E5%AF%86%E8%B4%A7%E5%B8%81&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
]

NEG_CN = ["暴跌", "崩盘", "瀑布", "闪崩", "大跌", "跳水", "黑客", "被盗", "被黑", "攻击", "漏洞",
          "监管", "禁令", "禁止", "取缔", "起诉", "处罚", "罚款", "破产", "爆雷", "挤兑", "清算",
          "抛售", "砸盘", "做空", "危机", "恐慌", "失守", "跌破", "下架", "关闭", "冻结", "跑路",
          "归零", "退市", "骗局", "收割", "爆仓", "违约", "停摆", "熔断", "警告"]
POS_CN = ["获批", "批准", "通过", "减半", "储备", "买入", "增持", "上市", "合作", "降息", "放水",
          "利好", "反弹", "新高", "突破", "上涨", "大涨", "资金流入", "涌入", "入局", "支持",
          "合法", "落地", "试点", "采纳", "接受", "支付", "增发", "护盘"]

NEG_EN = ["crash", "crashes", "hack", "hacked", "exploit", "theft", "stolen", "ban", "banned",
          "crackdown", "lawsuit", "sues", "sec", "bankrupt", "bankruptcy", "liquidation",
          "liquidated", "dump", "plunge", "plunges", "selloff", "sell-off", "freeze", "frozen",
          "delist", "scam", "fraud", "panic", "fear", "collapse", "bleed", "insolvent",
          "default", "meltdown", "warning", "outage"]
POS_EN = ["approval", "approved", "approves", "etf", "halving", "reserve", "reserves", "buys",
          "purchase", "adoption", "adopts", "partnership", "listing", "rate cut", "bullish",
          "rally", "rallies", "surge", "soars", "record high", "breakout", "inflows",
          "accumulat", "legalize", "launch", "endorse", "endorsement"]


def _fetch(url, timeout=8):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BTCNewsRadar/1.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(400000)


def _parse(data):
    root = ET.fromstring(data)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title or len(title) < 8:
            continue
        ts = None
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate") or "")
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.astimezone(timezone.utc)
        except Exception:
            pass
        out.append({"title": title, "ts": ts})
    return out


def fetch_items(limit=12):
    items = []
    for src, url in SOURCES:
        try:
            for it in _parse(_fetch(url))[:limit]:
                it["src"] = src
                items.append(it)
        except Exception:
            continue
    return items


def word_score(title):
    s = 0
    for w in NEG_CN:
        if w in title:
            s -= 1
    for w in POS_CN:
        if w in title:
            s += 1
    tl = title.lower()
    for w in NEG_EN:
        if re.search(r"\b" + re.escape(w) + r"\b", tl):
            s -= 1
    for w in POS_EN:
        if re.search(r"\b" + re.escape(w) + r"\b", tl):
            s += 1
    return s


def assess(items=None, hours=6):
    if items is None:
        items = fetch_items()
    if not items:
        return {"risk": "未知", "score": 0, "items": [],
                "msg": "所有新闻源都连不上, 雷达离线(不影响交易)"}
    now = datetime.now(timezone.utc)
    recent, seen = [], set()
    for it in items:
        if it.get("ts") is None:
            continue
        age_h = (now - it["ts"]).total_seconds() / 3600.0
        if age_h < 0 or age_h > hours:
            continue
        key = re.sub(r"\s+", "", it["title"])[:24]
        if key in seen:
            continue
        seen.add(key)
        sc = word_score(it["title"])
        tag = "利好" if sc > 0 else ("利空" if sc < 0 else "中性")
        age = f"{int(age_h * 60)}分前" if age_h < 1 else f"{int(age_h)}小时前"
        recent.append({"title": it["title"], "src": it["src"], "tag": tag,
                       "score": sc, "age": age})
    neg_n = sum(1 for x in recent if x["score"] < 0)
    pos_n = sum(1 for x in recent if x["score"] > 0)
    net = sum(x["score"] for x in recent)
    if neg_n >= 3 or net <= -4:
        risk = "高"
    elif neg_n >= 1 or net < 0:
        risk = "中"
    else:
        risk = "低"
    msg = f"近{hours}小时抓到{len(recent)}条: 利好{pos_n} 利空{neg_n} 情绪净值{net:+d}"
    recent.sort(key=lambda x: x["score"])
    return {"risk": risk, "score": net, "items": recent[:8], "msg": msg}


if __name__ == "__main__":
    import time as _t
    t0 = _t.time()
    rep = assess()
    print(f"耗时 {_t.time()-t0:.1f}s | 风险={rep['risk']} | {rep['msg']}")
    for x in rep["items"]:
        print(f"  [{x['tag']}] {x['src']} {x['age']} {x['title'][:80]}")
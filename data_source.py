# -*- coding: utf-8 -*-
"""备用行情源: Gate.io 公开接口(无需密钥, 国内可直连, 不用VPN)。

统一返回与 OKX 版相同的 DataFrame 列: open_time_ms/open/high/low/close/volume/close_time_ms
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

GATE_BASE = "https://api.gateio.ws"
GATE_PAIR = "BTC_USDT"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _get(url, timeout=40, retries=3):
    """带重试的 GET: 网络抖动/限流时自动退避重试。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    raise last


def _page(to):
    q = {"currency_pair": GATE_PAIR, "interval": "1h", "limit": 1000}
    if to is not None:
        q["to"] = str(to)
    return _get(GATE_BASE + "/api/v4/spot/candlesticks?" + urllib.parse.urlencode(q))


def gate_candles(limit=2000):
    """分页拉取 Gate 1小时K线, 整理为升序 DataFrame。短页自动重试。"""
    rows = {}
    to = None
    while len(rows) < limit:
        d = _page(to)
        if not d:
            break
        # 短页(不足1000根)常见于限流截断, 重试同页最多2次
        if len(d) < 1000 and len(rows) < limit:
            for _ in range(2):
                time.sleep(3)
                d2 = _page(to)
                if len(d2) > len(d):
                    d = d2
                    break
        before = len(rows)
        oldest = None
        for c in d:
            ts = int(c[0])
            rows[ts] = {
                "open_time_ms": ts * 1000, "open": float(c[5]), "high": float(c[3]),
                "low": float(c[4]), "close": float(c[2]), "volume": float(c[6]),
                "close_time_ms": (ts + 3600) * 1000,
            }
            oldest = ts if oldest is None else min(oldest, ts)
        if len(rows) <= before:
            break
        if oldest is None:
            break
        if len(d) < 1000:
            break
        to = oldest - 1
    df = pd.DataFrame(list(rows.values()))
    if len(df) == 0:
        return df
    return df.sort_values("open_time_ms").reset_index(drop=True)


def gate_ticker():
    """最新成交价, 返回 {'last': float}。实时报价用短超时, 断网快速失败不卡死循环。"""
    d = _get(GATE_BASE + "/api/v4/spot/tickers?currency_pair=" + GATE_PAIR,
             timeout=8, retries=2)
    return {"last": float(d[0]["last"])}


def gate_recent_trades(limit=300):
    """最近成交(秒级): 返回 [[时间戳ms, 价格], ...] 按时间升序。"""
    d = _get(GATE_BASE + "/api/v4/spot/trades?currency_pair=" + GATE_PAIR
             + "&limit=" + str(limit), timeout=10, retries=2)
    out = []
    for t in d:
        try:
            out.append([int(float(t["create_time_ms"])), float(t["price"])])
        except Exception:
            continue
    out.sort()
    return out


if __name__ == "__main__":
    df = gate_candles(2000)
    print(f"gate 1H K线: {len(df)} 根")
    if len(df):
        print("最新:", df.iloc[-1].to_dict())
    print("ticker:", gate_ticker())
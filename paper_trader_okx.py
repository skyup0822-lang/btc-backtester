# -*- coding: utf-8 -*-
"""OKX 模拟盘(模拟交易) 趋势交易脚本 —— 杂交方案实盘逻辑版

与 paper_trader.py(币安测试网版) 同策略: SMA48/336 + 收盘>SMA1440 做多;
15% 固定止损; 每笔风险1%仓位。OKX 模拟盘 = 真实行情撮合 + 虚拟资金,
测滑点比币安测试网更接近真实。

用法:
  --status            只读: 连 OKX 公共接口, 打印当前信号(不需要密钥)
  --live [config]     模拟盘循环: 读 paper_config_okx.json;
                      有密钥=在模拟盘真实挂单, 无密钥=本地虚拟成交(DRY)
文件:
  paper_config_okx.json   密钥与参数(OKX API: key/secret/passphrase)
  paper_state_okx.json    持仓状态(重启恢复)
  paper_trades_okx.csv    每笔成交记录(含滑点测算)
"""
import base64
import hashlib
import hmac
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from paper_trader import PaperTrader, compute_signal

BASE = "https://www.okx.com"
INST = "BTC-USDT"
RISK_DEFAULT = 0.01
STOP_DEFAULT = 0.15
GATE_BARS = 1440


def okx_request(method, path, key="", secret="", passphrase="", body=None, query=None, simulated=True):
    url = BASE + path
    qs = ""
    if query:
        qs = "?" + urllib.parse.urlencode(query)
    body_str = json.dumps(body) if body is not None else ""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Content-Type": "application/json"}
    if secret:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"
        sign = base64.b64encode(hmac.new(
            secret.encode(), f"{ts}{method}{path}{qs}{body_str}".encode(), hashlib.sha256
        ).digest()).decode()
        headers.update({
            "OK-ACCESS-KEY": key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
        })
    if key or simulated:
        headers["x-simulated-trading"] = "1" if simulated else "0"
    req = urllib.request.Request(
        url + qs, data=body_str.encode() if body_str else None, headers=headers, method=method
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_candles(limit=2000):
    """分页拉取 1H K线(每页最多300, 新->旧), 整理为升序 DataFrame。"""
    pages = []
    after = None
    while sum(len(p) for p in pages) < limit:
        q = {"instId": INST, "bar": "1H", "limit": 300}
        if after is not None:
            q["after"] = str(after)
        d = okx_request("GET", "/api/v5/market/candles", query=q)["data"]
        if not d:
            break
        pages.append(d)
        if len(d) < 300:
            break
        after = int(d[-1][0])
    rows = []
    for page in pages:
        for c in page:
            rows.append({
                "open_time_ms": int(c[0]), "open": float(c[1]), "high": float(c[2]),
                "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
                "close_time_ms": int(c[0]) + 3_600_000,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)


def lot_size():
    d = okx_request("GET", "/api/v5/public/instruments", query={"instType": "SPOT", "instId": INST})
    return float(d["data"][0]["lotSz"])


def status_check():
    t = okx_request("GET", "/api/v5/public/time")
    df = fetch_candles(2000)
    now = int(time.time() * 1000)
    closed = df[df["close_time_ms"] <= now]
    s = float(compute_signal(df)[len(closed) - 1]) if len(closed) > 0 else 0.0
    last = closed.iloc[-1]
    print("=" * 80)
    print("【OKX 模拟盘连通性检查】")
    print(f"  服务器时间: {t['data'][0]['ts']} | 本机: {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z")
    print(f"  已取 1H K线: {len(df)} 根 (需要 >{GATE_BARS} 根计算SMA1440)")
    print(f"  最新收盘K线: {pd.to_datetime(last['close_time_ms'], unit='ms', utc=True)} 收盘价 {last['close']:.2f}")
    print(f"  当前杂交信号: {'持仓(做多)' if s == 1 else '空仓'} (SMA48/336 + 收盘>SMA1440)")
    print("  连通性: OK")


def load_config(path):
    p = Path(path)
    if not p.exists():
        return {"api_key": "", "api_secret": "", "api_passphrase": "",
                "risk_pct": RISK_DEFAULT, "stop_pct": STOP_DEFAULT, "poll_seconds": 60}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def okx_market_order(side, sz, key, secret, passphrase, simulated):
    body = {"instId": INST, "tdMode": "cash", "side": side, "ordType": "market", "sz": sz}
    resp = okx_request("POST", "/api/v5/trade/order", key, secret, passphrase, body=body, simulated=simulated)
    if resp.get("code") != "0":
        raise RuntimeError(f"下单失败: {resp.get('code')} {resp.get('msg')}")
    d = resp["data"][0]
    avg = float(d.get("avgPx") or 0)
    fill = float(d.get("fillSz") or 0)
    return avg, fill, d["ordId"]


def usdt_balance(key, secret, passphrase, simulated):
    d = okx_request("GET", "/api/v5/account/balance", key, secret, passphrase,
                    query={"ccy": "USDT"}, simulated=simulated)
    details = d["data"][0].get("details", [])
    if not details:
        return 0.0
    return float(details[0]["availBal"])


def live_loop(cfg):
    key = cfg.get("api_key", "")
    secret = cfg.get("api_secret", "")
    passphrase = cfg.get("api_passphrase", "")
    simulated = bool(cfg.get("simulated", True))
    risk = float(cfg.get("risk_pct", RISK_DEFAULT))
    stop = float(cfg.get("stop_pct", STOP_DEFAULT))
    poll = float(cfg.get("poll_seconds", 60))
    has_keys = bool(key and secret and passphrase)
    lot = lot_size() if has_keys else 0.00001
    state_path = Path("paper_state_okx.json")
    t = PaperTrader(cash=10000.0, risk_pct=risk, stop_pct=stop, log_path=Path("paper_trades_okx.csv"))
    if state_path.exists():
        with state_path.open(encoding="utf-8") as f:
            st = json.load(f)
        t.cash, t.units, t.entry_price, t.entry_equity = (
            st["cash"], st["units"], st["entry_price"], st["entry_equity"]
        )
        t.stop_price = st.get("stop_price", np.inf)
        t.signal = st.get("signal", 0.0)
        t.reentry_bar_ms = st.get("reentry_bar_ms", 0)
        t.last_bar_ms = st.get("last_bar_ms", 0)
        print(f"恢复状态: 现金 {t.cash:.2f} | 持仓 {t.units:.8f} | 止损 {t.stop_price:.2f}")
    if has_keys:
        try:
            t.cash = usdt_balance(key, secret, passphrase, simulated)
            print(f"读取模拟盘余额: {t.cash:.2f} USDT (虚拟资金)")
        except Exception as e:
            print(f"读余额失败({e}), 使用本地记录 {t.cash:.2f}")
    print(f"模式: {'OKX模拟盘真实挂单' if has_keys else 'DRY 本地虚拟成交'} | 风险 {risk:.1%} | "
          f"止损 {stop:.0%} | 轮询 {poll:g}s | 品种 {INST}")
    print("循环已启动, Ctrl+C 退出。")
    cache = fetch_candles(2000)
    while True:
        try:
            now = int(time.time() * 1000)
            fresh = fetch_candles(300)
            cache = pd.concat([cache, fresh]).drop_duplicates("open_time_ms").sort_values("open_time_ms")
            if len(cache) > 3000:
                cache = cache.iloc[-3000:].reset_index(drop=True)
            closed = cache[cache["close_time_ms"] <= now]
            sig = compute_signal(cache)
            new_bars = closed[closed["close_time_ms"] > t.last_bar_ms]
            for _, bar in new_bars.iterrows():
                pos = int((closed["close_time_ms"] <= bar["close_time_ms"]).sum()) - 1
                s = float(sig[pos])
                t.last_close = bar["close"]
                t.last_bar_ms = int(bar["close_time_ms"])
                if t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms:
                    t.signal = 1.0
                    if has_keys:
                        notional = t.cash * risk / stop
                        avg, fill, oid = okx_market_order("buy", f"{notional:.2f}", key, secret, passphrase, simulated)
                        t.cash = usdt_balance(key, secret, passphrase, simulated)
                        t.buy(avg, "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    else:
                        t.buy(bar["close"], "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    print(f"[{pd.to_datetime(bar['close_time_ms'], unit='ms', utc=True)}] 信号买入 @ {bar['close']:.2f}, 止损 {t.stop_price:.2f}")
                elif t.units > 0 and s == 0:
                    t.signal = 0.0
                    if has_keys:
                        qty = np.floor(t.units / lot) * lot
                        okx_market_order("sell", f"{qty:.8f}", key, secret, passphrase, simulated)
                        t.cash = usdt_balance(key, secret, passphrase, simulated)
                        t.sell(bar["close"], "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    else:
                        t.sell(bar["close"], "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    print(f"[{pd.to_datetime(bar['close_time_ms'], unit='ms', utc=True)}] 信号卖出 @ {bar['close']:.2f}")
            if t.units > 0:
                tick = okx_request("GET", "/api/v5/market/ticker", query={"instId": INST})
                px = float(tick["data"][0]["last"])
                if px <= t.stop_price:
                    t.signal = 0.0
                    t.reentry_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                    if has_keys:
                        qty = np.floor(t.units / lot) * lot
                        okx_market_order("sell", f"{qty:.8f}", key, secret, passphrase, simulated)
                        t.cash = usdt_balance(key, secret, passphrase, simulated)
                        t.sell(min(px, t.stop_price), "stop", pd.to_datetime(now, unit="ms", utc=True))
                    else:
                        t.sell(min(px, t.stop_price), "stop", pd.to_datetime(now, unit="ms", utc=True))
                    print(f"[{pd.to_datetime(now, unit='ms', utc=True)}] 止损卖出 @ {px:.2f}")
            with state_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                    "entry_equity": t.entry_equity, "stop_price": t.stop_price,
                    "signal": t.signal, "reentry_bar_ms": t.reentry_bar_ms,
                    "last_bar_ms": t.last_bar_ms,
                }, f)
            time.sleep(poll)
        except KeyboardInterrupt:
            print("已退出, 状态已保存到 paper_state_okx.json")
            break
        except Exception as e:
            print(f"循环异常({type(e).__name__}): {e}, 60秒后重试")
            time.sleep(60)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("用法: python paper_trader_okx.py --status | --live [config.json]")
        return
    if args[0] == "--status":
        status_check()
    elif args[0] == "--live":
        cfg = load_config(args[1] if len(args) > 1 else "paper_config_okx.json")
        live_loop(cfg)
    else:
        print("未知参数。用法见文档字符串。")


if __name__ == "__main__":
    main()
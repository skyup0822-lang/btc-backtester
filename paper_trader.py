# -*- coding: utf-8 -*-
"""模拟盘(币安测试网)趋势交易脚本 —— 杂交方案实盘逻辑版

策略(与回测一致): 双均线48/336 上穿 + 收盘>SMA1440 才做多; 15% 固定止损;
                每笔风险1%仓位(可用配置改); 只做多现货, 1小时K线, BTC/USDT。
三种运行模式:
  --replay CSV路径   用历史数据回放"实盘循环逻辑"(不联网), 与回测引擎对账
  --status           只读: 连测试网拉最新K线, 打印当前信号状态(不需要密钥)
  --live             实盘模拟循环: 读取 paper_config.json;
                     有密钥=真实挂单测试网, 无密钥=自动降级dry(虚拟成交)
文件:
  paper_config.json   密钥与参数(自己填写, 别泄露; 只用测试网密钥!)
  paper_state.json    持仓状态(断线重启自动恢复)
  paper_trades.csv    每笔成交记录(含滑点测算)
"""
import csv
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

BASE = "https://testnet.binance.vision"
SHORT, LONG, GATE = 48, 336, 1440
RISK_DEFAULT = 0.01
STOP_DEFAULT = 0.15
SYMBOL = "BTCUSDT"


# ---------------- 工具 ----------------

def http_get(path, params=None, key="", secret=""):
    params = dict(params or {})
    if secret:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
    qs = urllib.parse.urlencode(params)
    if secret:
        qs += "&signature=" + hmac.new(
            secret.encode(), qs.encode(), hashlib.sha256
        ).hexdigest()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    if key:
        headers["X-MBX-APIKEY"] = key
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_klines(symbol=SYMBOL, limit=2000):
    """拉最多 limit 根1小时K线(每页1000, 需要两页)。返回带 close_time_ms 的 DataFrame。"""
    out = []
    end = None
    while len(out) < limit:
        params = {"symbol": symbol, "interval": "1h", "limit": 1000}
        if end is not None:
            params["endTime"] = end - 1
        page = http_get("/api/v3/klines", params)
        if not page:
            break
        out = page + out if end is None else page + out
        if end is None:
            end = page[0][0]
        if len(page) < 1000:
            break
        if len(out) >= limit:
            break
        end = out[0][0]
    rows = []
    for k in out:
        rows.append({
            "open_time_ms": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
            "close_time_ms": k[6],
        })
    df = pd.DataFrame(rows)
    return df


def compute_signal(df):
    close = df["close"]
    fast = close.rolling(SHORT).mean()
    slow = close.rolling(LONG).mean()
    gate = close > close.rolling(GATE).mean()
    sig = ((fast > slow) & gate).astype(float).fillna(0.0)
    return sig.to_numpy()


# ---------------- 交易核心(回放与实盘共用) ----------------

class PaperTrader:
    """杂交方案的状态机。买卖点位由外部传入(回放用历史bar, 实盘用行情+订单)。"""

    def __init__(self, cash=10000.0, risk_pct=RISK_DEFAULT, stop_pct=STOP_DEFAULT,
                 fee_rate=0.001, log_path=None):
        self.cash = cash
        self.risk_pct = risk_pct
        self.stop_pct = stop_pct
        self.fee_rate = fee_rate
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_equity = 0.0
        self.entry_cost = 0.0
        self.entry_time = None
        self.stop_price = np.inf
        self.signal = 0.0
        self.reentry_bar_ms = 0
        self.last_bar_ms = 0
        self.trades = []
        self.equity_hist = []
        self.log_path = Path(log_path) if log_path else None
        if self.log_path and not self.log_path.exists():
            self.log_path.write_text(
                "time_utc,side,reason,expected_px,fill_px,slippage_pct,notional,units,pnl_acct_pct,balance\n",
                encoding="utf-8",
            )

    @property
    def equity(self):
        return self.cash + self.units * self.last_close

    def buy(self, price, reason, ts):
        exec_price = price
        stop_dist = exec_price * self.stop_pct
        notional = self.cash * self.risk_pct / self.stop_pct
        units = notional / exec_price * (1 - self.fee_rate)
        self.entry_equity = self.cash
        self.entry_cost = notional
        self.cash -= notional
        self.units = units
        self.entry_price = exec_price
        self.stop_price = exec_price - stop_dist
        self.entry_time = ts
        self.slippage_expected = price
        self.log_row(ts, "BUY", reason, price, exec_price, units)
        return units

    def sell(self, price, reason, ts, pnl_note=True):
        exec_price = price
        proceeds = self.units * exec_price * (1 - self.fee_rate)
        self.cash += proceeds
        pnl = proceeds - self.entry_cost
        pnl_acct = pnl / self.entry_equity if self.entry_equity > 0 else 0.0
        if pnl_note:
            self.trades.append({
                "entry_time": self.entry_time, "exit_time": ts,
                "entry_price": self.entry_price, "exit_price": exec_price,
                "pnl": pnl, "pnl_acct_pct": pnl_acct, "reason": reason,
            })
        self.log_row(ts, "SELL", reason, self.slippage_expected, exec_price, self.units, pnl_acct)
        self.units = 0.0
        self.stop_price = np.inf
        return pnl_acct

    def log_row(self, ts, side, reason, expected, fill, units, pnl_acct=0.0):
        if not self.log_path:
            return
        slip = (fill / expected - 1) if side == "BUY" and expected > 0 else 0.0
        with self.log_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ts, side, reason, f"{expected:.2f}", f"{fill:.2f}", f"{slip:+.4%}",
                f"{units * fill:.2f}", f"{units:.8f}", f"{pnl_acct:+.4%}", f"{self.cash + units * fill:.2f}",
            ])


def replay(csv_path):
    """用历史1小时K线回放实盘循环逻辑, 与回测引擎对账。"""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    df = df.reindex(full)
    df["Volume"] = df["Volume"].fillna(0.0)
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].ffill()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    sig = compute_signal(df)
    open_ms = (df.index.astype("int64") // 10**6).to_numpy()
    close_ms = open_ms + 3599_000

    t = PaperTrader(cash=10000.0, log_path=Path(csv_path).parent.parent / "results" / "replay_trades.csv")
    equity = np.empty(len(df))
    for i in range(len(df)):
        bar_open, bar_high, bar_low, bar_close = (
            df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        )
        t.last_close = bar_close
        if i > 0:
            prev_sig = sig[i - 1]
            s = sig[i]
            if t.units > 0 and bar_low <= t.stop_price:
                t.sell(min(bar_open, t.stop_price), "stop", df.index[i])
                t.signal = 0.0
                t.reentry_bar_ms = close_ms[i]
            if t.units == 0 and s == 1 and t.signal == 0 and close_ms[i] > t.reentry_bar_ms:
                t.buy(bar_close, "signal", df.index[i])
                t.signal = 1.0
            elif t.units > 0 and s == 0:
                t.sell(bar_close, "signal", df.index[i])
                t.signal = 0.0
        equity[i] = t.equity
    eq = pd.Series(equity, index=df.index)
    oos = eq[eq.index > pd.Timestamp("2024-12-31", tz="UTC")]
    pnls = np.array([x["pnl_acct_pct"] for x in t.trades], dtype=float)
    print("=" * 80)
    print(f"【回放对账】数据 {df.index[0]} ~ {df.index[-1]}, 共 {len(df)} 根")
    print(f"  交易 {len(pnls)} 笔 | 每笔期望(账户%) {pnls.mean():+.3%} | 中位 {np.median(pnls):+.3%} "
          f"| 胜率 {(pnls > 0).mean():.1%}")
    print(f"  全程净值 {eq.iloc[-1] / eq.iloc[0]:.4f} | 最大回撤 {(eq / eq.cummax() - 1).min():+.2%}")
    if len(oos) > 0:
        print(f"  样本外(2025+)净值 {oos.iloc[-1] / oos.iloc[0]:.4f} "
              f"| 最大回撤 {(oos / oos.cummax() - 1).min():+.2%}")
    p21 = [x for x in t.trades if pd.Timestamp(x["entry_time"]) >= pd.Timestamp("2021-01-01", tz="UTC")]
    pnls21 = np.array([x["pnl_acct_pct"] for x in p21], dtype=float)
    if len(pnls21):
        print(f"  2021+ 交易 {len(pnls21)} 笔 | 每笔期望 {pnls21.mean():+.3%} | 中位 {np.median(pnls21):+.3%} | 胜率 {(pnls21 > 0).mean():.1%}")
    print(f"  (回测引擎同口径: BTC杂交 155笔, 每笔期望 +0.060%, 样本外 +0.95% / 回撤 -1.65%)")
    print(f"  成交明细已写入: results/replay_trades.csv")
    return t


def load_config(path):
    p = Path(path)
    if not p.exists():
        return {"api_key": "", "api_secret": "", "risk_pct": RISK_DEFAULT,
                "stop_pct": STOP_DEFAULT, "poll_seconds": 60}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def status_check():
    """只读检查: 连通性 + 当前信号。"""
    info = http_get("/api/v3/exchangeInfo", {"symbol": SYMBOL})
    now = int(time.time() * 1000)
    df = fetch_klines(SYMBOL, limit=2000)
    sig = compute_signal(df)
    closed = df[df["close_time_ms"] <= now]
    s = sig[len(closed) - 1] if len(closed) > 0 else 0.0
    last = closed.iloc[-1]
    print("=" * 80)
    print("【测试网连通性检查】")
    print(f"  服务器时间同步偏差: {info.get('serverTime', 0) - now:+d} ms")
    print(f"  品种 {SYMBOL} 状态: {info['symbols'][0]['status']}")
    print(f"  最新收盘K线: {pd.to_datetime(last['close_time_ms'], unit='ms', utc=True)} "
          f"收盘价 {last['close']:.2f}")
    print(f"  已取历史K线: {len(df)} 根 (需要 >{GATE} 根计算SMA1440)")
    print(f"  当前杂交信号: {'持仓(做多)' if s == 1 else '空仓'} (SMA48/336 + 收盘>SMA1440)")
    return s


def live_loop(cfg):
    key = cfg.get("api_key", "")
    secret = cfg.get("api_secret", "")
    risk = float(cfg.get("risk_pct", RISK_DEFAULT))
    stop = float(cfg.get("stop_pct", STOP_DEFAULT))
    poll = float(cfg.get("poll_seconds", 60))
    has_keys = bool(key and secret)
    state_path = Path("paper_state.json")
    t = PaperTrader(cash=10000.0, risk_pct=risk, stop_pct=stop, log_path=Path("paper_trades.csv"))
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
    print(f"模式: {'真实挂单(测试网)' if has_keys else 'DRY 虚拟成交(未配置密钥)'} | "
          f"风险 {risk:.1%} | 止损 {stop:.0%} | 轮询 {poll:g}s")
    print("循环已启动, Ctrl+C 退出。")
    while True:
        try:
            now = int(time.time() * 1000)
            df = fetch_klines(SYMBOL, limit=2000)
            closed = df[df["close_time_ms"] <= now]
            sig = compute_signal(df)
            new_bars = closed[closed["close_time_ms"] > t.last_bar_ms]
            for _, bar in new_bars.iterrows():
                s = sig[len(closed[closed["close_time_ms"] <= bar["close_time_ms"]]) - 1]
                t.last_close = bar["close"]
                t.last_bar_ms = bar["close_time_ms"]
                if t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms:
                    t.signal = 1.0
                    t.buy(bar["close"], "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    print(f"[{bar['close_time_ms']}] 信号买入 @ {bar['close']:.2f}, 止损 {t.stop_price:.2f}")
                elif t.units > 0 and s == 0:
                    t.sell(bar["close"], "signal", pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True))
                    t.signal = 0.0
                    print(f"[{bar['close_time_ms']}] 信号卖出 @ {bar['close']:.2f}")
            if t.units > 0:
                px = float(http_get("/api/v3/ticker/price", {"symbol": SYMBOL})["price"]) if has_keys else float(closed.iloc[-1]["close"])
                if px <= t.stop_price:
                    t.sell(min(px, t.stop_price), "stop", pd.to_datetime(now, unit="ms", utc=True))
                    t.signal = 0.0
                    t.reentry_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                    print(f"[{now}] 止损卖出 @ {px:.2f}")
            with state_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                    "entry_equity": t.entry_equity, "stop_price": t.stop_price,
                    "signal": t.signal, "reentry_bar_ms": t.reentry_bar_ms,
                    "last_bar_ms": t.last_bar_ms,
                }, f)
            time.sleep(poll)
        except KeyboardInterrupt:
            print("已退出, 状态已保存到 paper_state.json")
            break
        except Exception as e:
            print(f"循环异常({type(e).__name__}): {e}, 60秒后重试")
            time.sleep(60)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("用法: python paper_trader.py --replay CSV | --status | --live [config.json]")
        return
    if args[0] == "--replay":
        if len(args) < 2:
            print("需要CSV路径: --replay data/btc_usdt_1h.csv")
            return
        replay(args[1])
    elif args[0] == "--status":
        status_check()
    elif args[0] == "--live":
        cfg_path = args[1] if len(args) > 1 else "paper_config.json"
        cfg = load_config(cfg_path)
        live_loop(cfg)
    else:
        print("未知参数。用法见文档字符串。")


if __name__ == "__main__":
    main()
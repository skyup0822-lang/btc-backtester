# -*- coding: utf-8 -*-
"""傻瓜式模拟盘控制台(网页版, 多数据源 + 杂交趋势策略)

数据源自动选择: 依次试 Binance -> OKX -> Gate.io, 用第一个连得上的(都不需要VPN/密钥)。
双击"启动模拟盘.bat"后自动打开浏览器控制面板:
  [检查连接]   测试 Binance / OKX / Gate.io, 自动选择可用的数据源
  [保存并启动] 保存密钥与参数, 每小时检查信号、自动买卖、盯止损
  [停止]       停止循环(状态自动保存, 下次启动自动恢复)
  [退出程序]   关闭控制台程序
没填密钥也能跑(本地虚拟成交, 用于熟悉流程)。
"""
import csv as _csv
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import data_source as ds
import news_radar as nr
from patterns import detect as detect_patterns
from paper_trader import PaperTrader, compute_signal
from paper_trader_okx import (
    INST, fetch_candles as okx_fetch_candles, lot_size, okx_market_order,
    okx_request, usdt_balance,
)

ROOT = Path(__file__).resolve().parent

# 数据源优先级: 依次试, 用第一个连得上的。改这里就改了全局顺序。
SRC_ORDER = ("binance", "okx", "gate")
SRC_LABEL = {"binance": "Binance(直连)", "okx": "OKX", "gate": "Gate.io(直连)"}
XCHECK_TOL = 0.005  # 三源交叉校验允许的收盘价偏差(0.5%); 超过就认为这根K线可疑
VOL_WINDOW = 336    # 相对成交量的中位数窗口(14天), 与 work/volume_gate_test.py 一致
GATE_BARS = 1440    # 收盘>SMA1440 这道闸门需要的历史根数; 不足的话信号必然是0(假信号)


def _atomic_write(path, obj, indent=None):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    tmp.replace(path)


def _fetch_cny_rate(timeout=5):
    """免费汇率接口获取 USD/CNY, 失败返回0(回退用用户填的汇率)。"""
    for url in ("https://open.er-api.com/v6/latest/USD",
                "https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            rate = float((d.get("rates") or {}).get("CNY") or 0.0)
            if rate > 0:
                return rate
        except Exception:
            continue
    return 0.0
CFG = ROOT / "paper_config_okx.json"
STATE = ROOT / "paper_state_okx.json"
TRADES = ROOT / "paper_trades_okx.csv"
PAT_HIST = ROOT / "pattern_history.json"
DAILY = ROOT / "daily_report.csv"
DEC = ROOT / "decision_log.csv"
AUDIT = ROOT / "cash_audit.csv"
BARS_HIST = ROOT / "bars_history.csv"
# 手动账户: 跟自动盘完全分开记账, 否则钱混在一起没法对比谁做得好
MANUAL_STATE = ROOT / "paper_state_manual.json"
MANUAL_TRADES = ROOT / "paper_trades_manual.csv"
START_EQUITY = 100.0  # 自动/手动两个账户的共同起始资金, 对比的基准
# 多品种子账户: 同一套策略跑在别的品种上, 各自独立 100 USDT, 和 BTC 主盘分开记账
ETH_SYMBOL = "ETHUSDT"
ETH_PAIR = "ETH_USDT"
ETH_STATE = ROOT / "paper_state_eth.json"
ETH_TRADES = ROOT / "paper_trades_eth.csv"
PORT = 8765

# 图上只标记这些重要形态(回调/盘整/吸筹太常见, 画上去会太乱)
SIGNIFICANT = {"瀑布", "砸盘", "拉盘", "疑似洗盘", "反弹", "大资金异动(疑似)"}

# 建议卡里的"历史同类情况"用这几个形态标记索引
PAT_FLAGS = ["dump", "pump", "bounce", "pullback", "range", "shake", "whale", "accum"]
PAT_CN = {"dump": "瀑布/砸盘", "pump": "拉盘", "bounce": "反弹", "pullback": "回调",
          "range": "盘整", "shake": "疑似洗盘", "whale": "大资金异动", "accum": "疑似吸筹"}
ADVICE_MIN_N = 20  # 样本少于这个数就不给结论 —— 十几次的结果不是规律


def _stat(s):
    return {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
            "up": float((s > 0).mean())}


def _history_stats():
    """历史基准率: 过去出现"某种情况"之后, 未来 24/72/168 小时涨跌的分布。

    算的时候用了未来数据, 所以它讲的是"过去发生过什么", 不是预测。
    每个格子都带样本数; 样本太少的直接不给。
    """
    path = ROOT / "data" / "btc_usdt_1h.csv"
    if not path.exists():
        return {}
    # 回测CSV是大写列名, 实盘逻辑要小写 —— 统一成小写再往下走
    df = pd.read_csv(path, index_col=0, parse_dates=True).rename(columns=str.lower)
    close = df["close"]
    pat = detect_patterns(df)
    sig = pd.Series(compute_signal(df), index=df.index)
    trig = (sig.diff() > 0).fillna(False)
    out = {"range": [str(df.index[0])[:10], str(df.index[-1])[:10]],
           "n_bars": int(len(df)), "base": {}, "trigger": {}, "flags": {}}
    for h in (24, 72, 168):
        fwd = close.shift(-h) / close - 1
        ok = fwd.notna()
        out["base"][str(h)] = _stat(fwd[ok])
        out["trigger"][str(h)] = _stat(fwd[trig & ok])
        for f in PAT_FLAGS:
            sel = fwd[pat[f].fillna(False) & ok]
            if len(sel) >= ADVICE_MIN_N:
                out["flags"].setdefault(f, {})[str(h)] = _stat(sel)
    return out

# 形态出现后的历史统计(BTC+ETH 小时线, 2019-11~2026-09, 离线统计, 非预测)
BASE_RATE = {
    "瀑布": "历史上瀑布/砸盘后24小时平均+3.4%(上涨概率67%), 72小时+4.3%(69%)",
    "砸盘": "历史上瀑布/砸盘后24小时平均+3.4%(上涨概率67%), 72小时+4.3%(69%)",
    "疑似洗盘": "历史上疑似洗盘后24小时平均+3.0%(上涨概率69%), 72小时+4.0%(69%)",
    "拉盘": "历史上拉盘后24小时平均+2.0%(上涨概率59%), 72小时+2.0%(56%)",
    "回调": "历史上回调后24小时平均持平(上涨概率55%), 72小时+1.0%(56%)",
    "盘整": "历史上盘整后24/72小时接近整体平均(上涨概率52~54%), 方向不明",
}


_TF10_BARS = None


def ten_min_bars():
    """10分钟K线: 由本地 data/tf_5m.csv(5分钟) 两两合成, 惰性加载一次。"""
    global _TF10_BARS
    if _TF10_BARS is not None:
        return _TF10_BARS
    try:
        df = pd.read_csv(ROOT / "data" / "tf_5m.csv")
        df["bucket"] = df["open_time_ms"] // 600_000 * 600_000
        g = df.groupby("bucket")["close"].last()
        _TF10_BARS = [[int(k), float(v)] for k, v in g.items()]
    except Exception:
        _TF10_BARS = []
    return _TF10_BARS


def _book_stats(cash, units, entry_price, entry_cost, equity_start, trades, px):
    """把一个账户折算成对比用的几个数。Core 和 Sleeve 共用。"""
    eq = cash + units * (px or entry_price or 0.0)
    pnls = [x.get("pnl", 0.0) for x in trades]
    return {
        "equity": eq,
        "ret": eq / equity_start - 1.0 if equity_start > 0 else 0.0,
        "units": units,
        "entry": entry_price,
        "cash": cash,
        "n_trades": len(pnls),
        "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None,
        "upnl": (px - entry_price) * units if (units > 0 and px > 0 and entry_price > 0) else 0.0,
    }


def _skip_history(t, closed, log=None, label=""):
    """首次启动(没有历史状态)时, 把游标定在最后一根已收盘K线上。

    否则 last_bar_ms 是 0, 循环会把缓存里 2000 根历史全当成"新K线"逐根重放,
    结果是在两个月前的价位上建仓, 账面立刻多出一笔根本不存在的浮盈。
    """
    if t.last_bar_ms or closed is None or not len(closed):
        return False
    t.last_bar_ms = int(closed.iloc[-1]["close_time_ms"])
    if log:
        _ts = pd.to_datetime(t.last_bar_ms, unit="ms", utc=True)
        log(f"[{label}] 首次启动: 只从 {_ts:%m-%d %H:%M} 之后的新K线开始判断, 不回溯历史")
    return True


class Sleeve:
    """一个品种的独立子账户, 用来验证"多品种"是不是真的比单做 BTC 好。

    策略参数与 BTC 主盘完全一致(信号/止损/跟踪/风险都同一套), 刻意不带形态解读、
    新闻雷达、人民币折算、现金审计 —— 只保留影响结果的信号、止损、账本三件事,
    两个品种口径一致才比得出来。

    数据走币安, 挂了自动退到 Gate.io; 不复用主盘的数据源选择, 免得互相干扰。
    """

    def __init__(self, label, symbol, pair, state_path, trades_path):
        self.label, self.symbol, self.pair = label, symbol, pair
        self.state_path, self.trades_path = state_path, trades_path
        self.t = PaperTrader(cash=START_EQUITY, risk_pct=0.02, stop_pct=0.15,
                             log_path=trades_path, trail_pct=0.08)
        self.price = 0.0
        self.source = "-"
        self.note = "未启动"
        self.cache = None
        self._err = 0
        self.load()

    # ---- 状态持久化 ----
    def load(self):
        if not self.state_path.exists():
            return
        try:
            with self.state_path.open(encoding="utf-8") as f:
                s = json.load(f)
            t = self.t
            t.cash = float(s.get("cash", START_EQUITY))
            t.units = float(s.get("units", 0.0))
            t.entry_price = float(s.get("entry_price", 0.0))
            t.entry_cost = float(s.get("entry_cost", 0.0))
            t.entry_equity = float(s.get("entry_equity", 0.0))
            t.stop_price = float(s.get("stop_price", float("inf"))) or float("inf")
            t.high_watermark = float(s.get("high_watermark", 0.0))
            t.signal = float(s.get("signal", 0.0))
            t.reentry_bar_ms = int(s.get("reentry_bar_ms", 0))
            t.last_bar_ms = int(s.get("last_bar_ms", 0))
            t.last_close = float(s.get("last_close", 0.0))
        except Exception:
            pass

    def save(self):
        t = self.t
        try:
            _atomic_write(self.state_path, {
                "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                "entry_cost": t.entry_cost, "entry_equity": t.entry_equity,
                "stop_price": t.stop_price if np.isfinite(t.stop_price) else 0.0,
                "high_watermark": t.high_watermark, "signal": t.signal,
                "reentry_bar_ms": t.reentry_bar_ms, "last_bar_ms": t.last_bar_ms,
                "last_close": getattr(t, "last_close", 0.0),
            })
        except Exception:
            pass

    # ---- 取数 ----
    def _candles(self, initial):
        n = 2000 if initial else 300
        try:
            df = ds.binance_candles(n, self.symbol)
            if len(df) >= 60:
                self.source = "Binance"
                return df
        except Exception:
            pass
        df = ds.gate_candles(2000 if initial else 60, self.pair)
        if len(df) < 60:
            raise RuntimeError(f"{self.label} K线不足({len(df)}根)")
        self.source = "Gate.io"
        return df

    def _ticker(self):
        try:
            return float(ds.binance_ticker(self.symbol)["last"])
        except Exception:
            return float(ds.gate_ticker(self.pair)["last"])

    # ---- 主循环 ----
    def run(self, stop_event, log, poll=60, risk=0.02, stop=0.15, trail=0.08):
        t = self.t
        t.risk_pct, t.stop_pct, t.trail_pct = risk, stop, (trail or None)
        try:
            self.cache = self._candles(True)
        except Exception as e:
            self.note = f"首次取数失败({type(e).__name__})"
            log(f"[{self.label}] 启动失败: {e}")
            return
        self.note = "运行中"
        log(f"[{self.label}] 启动: 取到 {len(self.cache)} 根历史K线({self.source}) | "
            f"风险 {risk:.1%} | 止损 {stop:.0%} | 跟踪 {trail:.0%}")
        _skip_history(t, self.cache[self.cache["close_time_ms"] <= int(time.time() * 1000)],
                      log, self.label)
        self.save()
        while not stop_event.is_set():
            try:
                self._step(log)
                self._err = 0
                self.note = "运行中"
            except Exception as e:
                self._err += 1
                self.note = f"网络抖动({type(e).__name__})"
                if self._err == 1 or self._err % 5 == 0:
                    log(f"[{self.label}] 本轮失败({type(e).__name__}), 继续重试(连续{self._err}次)")
            for _ in range(int(poll)):
                if stop_event.is_set():
                    break
                time.sleep(1)
        self.save()
        self.note = "已停止"
        log(f"[{self.label}] 已停止, 状态已保存")

    def _step(self, log):
        t = self.t
        now = int(time.time() * 1000)
        self.price = self._ticker()
        fresh = self._candles(False)
        cache = pd.concat([self.cache, fresh]).drop_duplicates("open_time_ms") \
            .sort_values("open_time_ms").reset_index(drop=True)
        self.cache = cache
        closed = cache[cache["close_time_ms"] <= now]
        if len(closed) < GATE_BARS + 1:
            return
        sig = pd.Series(compute_signal(cache), index=cache.index)
        for _, bar in closed[closed["close_time_ms"] > t.last_bar_ms].iterrows():
            pos = int((closed["close_time_ms"] <= bar["close_time_ms"]).sum()) - 1
            s = float(sig.iloc[pos]) if 0 <= pos < len(sig) else 0.0
            if np.isnan(s):
                continue
            t.last_close = float(bar["close"])
            t.last_bar_ms = int(bar["close_time_ms"])
            ts = pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True)
            if t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms:
                t.buy(float(bar["close"]), "signal", ts)
                t.signal = 1.0
                log(f"[{self.label}] 信号买入 @ {bar['close']:,.2f}, 止损 {t.stop_price:,.2f}")
            elif t.units > 0 and s == 0:
                pnl = t.sell(float(bar["close"]), "signal", ts)
                t.signal = 0.0
                t.reentry_bar_ms = int(bar["close_time_ms"]) + 3_600_000
                log(f"[{self.label}] 信号卖出 @ {bar['close']:,.2f}, 本笔 {pnl:+.2%}")
        if t.units > 0 and self.price > 0:
            t.update_stop(self.price)
            if self.price <= t.stop_price:
                fill = min(self.price, t.stop_price)
                pnl = t.sell(fill, "stop", pd.to_datetime(now, unit="ms", utc=True))
                t.signal = 0.0
                t.reentry_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                log(f"[{self.label}] 止损卖出 @ {fill:,.2f}, 本笔 {pnl:+.2%}")
        self.save()

    def stats(self):
        t = self.t
        px = self.price or t.entry_price or 0.0
        s = _book_stats(t.cash, t.units, t.entry_price, t.entry_cost,
                        START_EQUITY, t.trades, px)
        s["label"] = self.label
        s["note"] = self.note
        s["source"] = self.source
        s["price"] = px
        s["signal"] = t.signal
        return s


class Core:
    """交易核心: 状态 + 日志 + 后台循环。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.cache = None
        self.src = None
        self.live = None
        self.src_cooldown = {}  # 交叉验证取数失败的源 -> 冷却到什么时候(避免拖慢主循环)
        self.lat = {"price": 0.0, "kline": 0.0, "trades": 0.0}  # 三类请求的往返耗时(秒)
        self.manual = None  # 手动交易账户(PaperTrader), 第一次点买入/卖出时才建
        self.sleeves = []   # 多品种子账户(ETH), 和 BTC 主盘各自独立记账
        self.sleeve_threads = []
        self._arch_ms = self._last_arch_ms()  # 从归档文件续写, 别每次重启重写一遍
        self.hist = {}      # 历史基准率(后台算, 约1秒)
        self.hist_msg = "历史统计还在算..."
        self.trader_ref = None
        self.pattern_history = []
        if PAT_HIST.exists():
            try:
                with PAT_HIST.open(encoding="utf-8") as f:
                    self.pattern_history = json.load(f)
            except Exception:
                self.pattern_history = []
        self.fx_rate = 7.2
        self._fx_date = ""
        self.daily_rows = []
        if DAILY.exists():
            try:
                with DAILY.open(encoding="utf-8") as f:
                    self.daily_rows = list(_csv.DictReader(f))
            except Exception:
                self.daily_rows = []
        self.dec_lock = threading.Lock()
        self.decision_rows = []
        if DEC.exists():
            try:
                with DEC.open(encoding="utf-8") as f:
                    for _line in f:
                        _line = _line.rstrip("\n").rstrip("\r")
                        if not _line:
                            continue
                        _parts = _line.split("|")
                        if len(_parts) >= 4:
                            self.decision_rows.append({
                                "t": _parts[0], "action": _parts[1], "reason": _parts[2],
                                "px": _parts[3], "sig": _parts[4] if len(_parts) > 4 else "",
                            })
                self.decision_rows = self.decision_rows[-5000:]
            except Exception:
                self.decision_rows = []
        self.live_thread = None
        self.news = {"risk": "未知", "items": [], "msg": ""}
        self.last_news_ms = 0
        self.last_news_risk = "未知"
        self.status = {
            "net": "unknown", "net_msg": "尚未检查",
            "source": "-", "signal": 0.0, "balance": 100.0, "units": 0.0,
            "entry": 0.0, "stop": 0.0, "last_close": 0.0,
            "pattern": "无", "pattern_msg": "启动后自动分析",
            "pattern_time": "", "pattern_next": "",
            "candle_time": "", "candle_ts_ms": 0,
            "ticker_time": "", "ticker_ts_ms": 0,
            "server_ts_ms": 0,
            "decision_time": "",
            "news_risk": "未知", "news_time": "", "news_msg": "", "news_items": [],
            "news_gate": True,
            "pattern_base": "", "auto_dump": False, "auto_filter": False,
            "mode": "未启动", "running": False, "has_keys": False,
        }
        self.logs = []

    def log(self, text):
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")
            self.logs = self.logs[-300:]

    def snapshot(self):
        with self.lock:
            st = dict(self.status)
            st["pattern_history"] = list(self.pattern_history[-50:])
            st["server_ts_ms"] = int(time.time() * 1000)
            st["decisions"] = list(reversed(self.decision_rows[-400:]))
            st["fx_rate"] = self.fx_rate
            st["daily"] = {"rows": list(self.daily_rows[-60:])}
            _px = float((self.live or {}).get("price", 0.0) or 0.0)
            _man, _auto = self.compare(_px)
            st["manual"], st["auto"] = _man, _auto
            st["start_equity"] = START_EQUITY
            _sl = [sv.stats() for sv in self.sleeves]
            st["sleeves"] = _sl
            if _sl:
                # 组合口径: BTC 主盘 + 各子账户, 资金各 100, 合起来就是等权组合
                _books = [_auto] + _sl
                _eq = sum(b["equity"] for b in _books)
                _base = START_EQUITY * len(_books)
                _w = [(b["n_trades"], b["win_rate"]) for b in _books
                      if b["n_trades"] and b["win_rate"] is not None]
                _n = sum(a for a, _ in _w)
                st["portfolio"] = {
                    "label": "自动-组合", "equity": _eq, "ret": _eq / _base - 1,
                    "n_trades": sum(b["n_trades"] for b in _books),
                    "units": sum(b["units"] for b in _books),
                    "win_rate": (sum(a * b for a, b in _w) / _n) if _n else None,
                }
            st["advice"] = self.build_advice(_px)
            return st, list(self.logs[-60:])

    def _record_daily(self, equity_usdt, px, units):
        today = time.strftime("%Y-%m-%d")
        if today != self._fx_date:
            fresh = _fetch_cny_rate()
            if fresh > 0:
                self.fx_rate = fresh
            self._fx_date = today
        rate = self.fx_rate or 7.2
        rows = list(self.daily_rows)
        prev = None
        for r in reversed(rows):
            if r.get("date", "") < today:
                prev = r
                break
        prev_eq = float(prev["equity_usdt"]) if prev is not None else 100.0
        pnl = equity_usdt - prev_eq
        row = {
            "date": today,
            "equity_usdt": f"{equity_usdt:.2f}",
            "pnl_usdt": f"{pnl:.2f}",
            "equity_cny": f"{equity_usdt * rate:.2f}",
            "pnl_cny": f"{pnl * rate:.2f}",
            "btc_price": f"{px:.2f}",
            "units": f"{units:.8f}",
        }
        rows = [r for r in rows if r.get("date") != today] + [row]
        rows.sort(key=lambda r: r.get("date", ""))
        self.daily_rows = rows[-400:]
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["date", "equity_usdt", "pnl_usdt", "equity_cny", "pnl_cny", "btc_price", "units"])
        for r in self.daily_rows:
            w.writerow([r["date"], r["equity_usdt"], r["pnl_usdt"], r["equity_cny"], r["pnl_cny"], r["btc_price"], r["units"]])
        try:
            _atomic_write_text(DAILY, buf.getvalue())
        except Exception:
            pass

    def _record_decision(self, action, reason, px, sig):
        if not hasattr(self, "dec_appends"):
            self.dec_appends = 0
        # 降噪: 重要的买卖决策永远记; 其余"动作+原因"相同则每 5 分钟才记一次
        # (信号一变原因就变 -> 立刻记, 所以仍能看出实时判断)
        _nowms = int(time.time() * 1000)
        if action not in ("买", "卖", "手动买", "手动卖") and self.decision_rows:
            _last = self.decision_rows[-1]
            if (_last.get("action") == action and _last.get("reason") == reason
                    and _nowms - getattr(self, "_last_dec_ms", 0) < 5 * 60 * 1000):
                return
        self._last_dec_ms = _nowms
        row = {
            "t": time.strftime("%m-%d %H:%M"),
            "action": action,
            "reason": reason,
            "px": f"{px:.2f}" if px else "-",
            "sig": str(sig),
        }
        self.decision_rows.append(row)
        self.decision_rows = self.decision_rows[-5000:]
        self.status["decision_time"] = time.strftime("%H:%M:%S")
        self.dec_appends += 1
        try:
            with self.dec_lock:
                if self.dec_appends % 720 == 0:
                    buf = io.StringIO()
                    for r in self.decision_rows:
                        buf.write("|".join([r["t"], r["action"], r["reason"], r["px"], r["sig"]]) + "\n")
                    _atomic_write_text(DEC, buf.getvalue())
                else:
                    with DEC.open("a", encoding="utf-8") as f:
                        f.write("|".join([row["t"], row["action"], row["reason"], row["px"], row["sig"]]) + "\n")
        except Exception:
            pass

    def _audit_cash(self, reason, cash0, cash1, units0, units1):
        """审计: 记录每次现金变动及原因; 无交易却变动则告警并写 cash_audit.csv。"""
        try:
            new = not AUDIT.exists()
            with AUDIT.open("a", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                if new:
                    w.writerow(["time", "reason", "cash_before", "cash_after", "delta",
                                "units_before", "units_after"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), reason,
                            f"{cash0:.6f}", f"{cash1:.6f}", f"{cash1 - cash0:+.6f}",
                            f"{units0:.10f}", f"{units1:.10f}"])
        except Exception:
            pass
        if reason.startswith("UNEXPLAINED"):
            self.log(f"[账本告警] cash 无交易却变动 {cash1 - cash0:+.6f} "
                     f"({cash0:.4f} -> {cash1:.4f}), 已记 cash_audit.csv")

    # ---------------- 数据源 ----------------

    def _probe_source(self):
        probes = {
            "binance": ds.binance_ticker,
            "okx": lambda: okx_request("GET", "/api/v5/public/time"),
            "gate": ds.gate_ticker,
        }
        for name in SRC_ORDER:
            try:
                probes[name]()
                return name
            except Exception:
                continue
        return None

    def _load_candles(self, src, initial):
        _t0 = time.perf_counter()
        try:
            if src == "gate":
                return ds.gate_candles(2000 if initial else 50)
            if src == "binance":
                return ds.binance_candles(2000 if initial else 300)
            return okx_fetch_candles(2000 if initial else 300)
        finally:
            self.lat["kline"] = time.perf_counter() - _t0

    def _load_ticker(self, src):
        _t0 = time.perf_counter()
        try:
            if src == "gate":
                return ds.gate_ticker()
            if src == "binance":
                return ds.binance_ticker()
            d = okx_request("GET", "/api/v5/market/ticker", query={"instId": INST})
            x = d["data"][0]
            out = {"last": float(x["last"])}
            if x.get("ts"):
                out["ts"] = int(x["ts"])
            return out
        finally:
            self.lat["price"] = time.perf_counter() - _t0

    # ---------------- 三源交叉校验 ----------------

    def _bar_from(self, name, close_time_ms):
        """从指定源取同一根已收盘K线, 返回 {'close','volume'}；取不到返回 None。

        失败的源冷却 30 分钟: 否则每次校验都要等它超时重试, 会把主循环拖死。
        """
        if time.time() < self.src_cooldown.get(name, 0.0):
            return None
        try:
            df = self._load_candles(name, initial=False)
        except Exception as e:
            self.src_cooldown[name] = time.time() + 1800
            self.log(f"[校验] {SRC_LABEL.get(name, name)} 取数失败({type(e).__name__}), 30分钟内不再拿它交叉验证")
            return None
        hit = df[df["close_time_ms"] == close_time_ms]
        if len(hit) == 0:
            return None
        r = hit.iloc[0]
        return {"close": float(r["close"]), "volume": float(r["volume"])}

    def _verify_bar(self, src, bar):
        """同一根已收盘K线, 别的源给出的收盘价必须对得上。

        K线可以伪造/插针, 单一数据源说了不算。对不上就跳过这根的信号决策;
        盯止损不受影响 —— 它每分钟看实时报价, 走的是另一条路。
        """
        ct = int(bar["close_time_ms"])
        close = float(bar["close"])
        got = {}
        for name in SRC_ORDER:
            if name == src:
                continue
            r = self._bar_from(name, ct)
            if r:
                got[name] = r
        if not got:
            self.log(f"[校验] {close:,.1f} 这根K线只有单一数据源, 无法交叉验证")
            return True
        dev = {n: r["close"] / close - 1 for n, r in got.items()}
        detail = " | ".join(f"{SRC_LABEL.get(n, n)} {got[n]['close']:,.1f}({d:+.2%})"
                            for n, d in dev.items())
        if any(abs(d) <= XCHECK_TOL for d in dev.values()):
            self.log(f"[校验] {SRC_LABEL.get(src, src)} {close:,.1f} 通过 -> {detail}")
            return True
        self.log(f"[校验][警告] {SRC_LABEL.get(src, src)} 这根K线 {close:,.1f} 与其他所有源都对不上, "
                 f"疑似假K线/插针, 已跳过本根的信号决策 -> {detail}")
        self._record_decision("跳过", "K线三源对不上, 疑似假K线, 本根不决策", close, 0)
        return False

    # ---------------- 手动交易(独立账户) ----------------

    def manual_trader(self):
        """手动账户: 起始资金、手续费口径与自动盘一致, 但不开跟踪止损(全靠自己点)。

        手续费走 PaperTrader 默认的 0.1%/边, 与回测口径相同; 这样两边差异只来自"谁在什么时候买卖"。
        """
        if self.manual is None:
            t = PaperTrader(cash=START_EQUITY, risk_pct=0.02, stop_pct=0.15,
                            log_path=MANUAL_TRADES, trail_pct=None)
            if MANUAL_STATE.exists():
                try:
                    with MANUAL_STATE.open(encoding="utf-8") as f:
                        s = json.load(f)
                    t.cash = float(s.get("cash", START_EQUITY))
                    t.units = float(s.get("units", 0.0))
                    t.entry_price = float(s.get("entry_price", 0.0))
                    t.entry_cost = float(s.get("entry_cost", 0.0))
                    t.entry_equity = float(s.get("entry_equity", 0.0))
                except Exception:
                    pass
            self.manual = t
        return self.manual

    def _save_manual(self):
        t = self.manual
        if t is None:
            return
        _atomic_write(MANUAL_STATE, {
            "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
            "entry_cost": t.entry_cost, "entry_equity": t.entry_equity,
            "start_equity": START_EQUITY,
        })

    def manual_order(self, side, pct):
        """手动买/卖。买只在空仓时允许, 卖就是全部卖出 —— 与自动盘同口径才好比。"""
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = 0.133
        t = self.manual_trader()
        with self.lock:
            px = float((self.live or {}).get("price", 0.0) or 0.0)
        if px <= 0:
            return False, "还没有实时报价, 等一两秒再点"
        ts = pd.to_datetime(int(time.time() * 1000), unit="ms", utc=True)
        if side == "buy":
            if t.units > 0:
                return False, "手动账户已持仓, 先卖出才能再买(和自动盘一样只做单向)"
            pct = min(max(pct, 0.01), 1.0)
            t.buy(px, "manual", ts, notional=t.cash * pct)
            self.log(f"[手动] 买入 @ {px:,.2f}, 投入现金 {pct:.1%}, 持仓 {t.units:.8f} BTC")
            self._record_decision("手动买", f"手动下单, 投入现金 {pct:.0%}", px, 1)
        else:
            if t.units <= 0:
                return False, "手动账户是空仓, 没有可卖的"
            pnl_pct = t.sell(px, "manual", ts)
            self.log(f"[手动] 卖出 @ {px:,.2f}, 本笔账户收益 {pnl_pct:+.2%}")
            self._record_decision("手动卖", f"手动平仓, 本笔 {pnl_pct:+.2%}", px, 0)
        self._save_manual()
        return True, ""

    def _book_stats(self, cash, units, entry_price, entry_cost, equity_start, trades, px):
        """兼容旧调用: 实际算账的是模块级 _book_stats。"""
        return _book_stats(cash, units, entry_price, entry_cost, equity_start, trades, px)

    def compare(self, px):
        """手动 vs 自动 的实时对比。两边起始都是 START_EQUITY, 手续费口径一致。"""
        mt = self.manual_trader()  # 必须在这里取, 不能用 self.manual: 重启后它还是 None
        if mt is None:
            man = self._book_stats(START_EQUITY, 0.0, 0.0, 0.0, START_EQUITY, [], px)
        else:
            man = self._book_stats(mt.cash, mt.units, mt.entry_price, mt.entry_cost,
                                   START_EQUITY, mt.trades, px)
        t = self.trader_ref
        if t is None:
            auto = self._book_stats(START_EQUITY, 0.0, 0.0, 0.0, START_EQUITY, [], px)
        else:
            auto = self._book_stats(t.cash, t.units, t.entry_price, t.entry_cost,
                                    START_EQUITY, t.trades, px)
        man["label"] = "手动"
        auto["label"] = "自动"
        return man, auto

    # ---------------- 实时建议(手动模式的参考) ----------------

    def _last_arch_ms(self):
        """归档文件里最后一根K线的时间。

        必须从文件里读回来: 这个值默认 0 的话, 每次重启都会把整段缓存(2000根)
        重新追加一遍 —— 同一个 bar 在文件里会有好几份, signal 还可能来自不同会话,
        复盘时就分不清哪份是当时真正用的信号。
        """
        try:
            last = BARS_HIST.read_text(encoding="utf-8").strip().splitlines()[-1]
            return int(last.split(",", 1)[0])
        except Exception:
            return 0

    def start_history_stats(self):
        """后台算历史基准率。约1秒, 不占主循环。"""
        def work():
            try:
                self.hist = _history_stats()
                r = self.hist.get("range")
                self.hist_msg = (f"统计区间 {r[0]} ~ {r[1]}, 共 {self.hist.get('n_bars', 0):,} 根1小时K线"
                                 if r else "没找到本地历史数据, 少了这一块")
            except Exception as e:
                self.hist_msg = f"历史统计算不出来({type(e).__name__}), 少了这一块"
            self.log(f"[建议] {self.hist_msg}")
        threading.Thread(target=work, daemon=True).start()

    def _indicators(self):
        """建议卡要用的指标, 每根收盘K线只算一次。

        必须用**已收盘**的K线: 缓存最后一行是本小时还在走的K线, 拿它算 ATR 和成交量
        会严重偏低(一根才走了几分钟), 那种数字看着像"死水", 其实是没攒够。
        """
        cache = self.cache
        if cache is None or len(cache) < 60:
            return None
        df = cache[cache["close_time_ms"] <= int(time.time() * 1000)]
        if len(df) < 60:
            return None
        key = int(df.iloc[-1]["close_time_ms"])
        if getattr(self, "_ind_key", None) == key:
            return self._ind
        close = df["close"]
        last = float(close.iloc[-1])

        def _ma(n):
            return float(close.rolling(n).mean().iloc[-1]) if len(df) >= n else None

        prev = close.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                        (df["low"] - prev).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        vmed = float(df["volume"].tail(VOL_WINDOW).median())
        ind = {
            "last": last,
            "sma48": _ma(48), "sma336": _ma(336), "sma1440": _ma(1440),
            "atr_pct": (atr14 / last) if last else None,
            "vol_ratio": (float(df["volume"].iloc[-1]) / vmed) if vmed > 0 else None,
            "high72": float(df["high"].tail(72).max()),
            "low72": float(df["low"].tail(72).min()),
            "chg24h": (last / float(close.iloc[-25]) - 1) if len(df) > 25 else None,
        }
        self._ind, self._ind_key = ind, key
        return ind

    def _cur_flags(self):
        """当前这根收盘K线命中了哪些形态标记。"""
        pdf = getattr(self, "pat_df", None)
        if pdf is None or not len(pdf):
            return []
        row = pdf.iloc[-1]
        return [f for f in PAT_FLAGS if f in pdf.columns and bool(row.get(f))]

    def build_advice(self, px):
        """手动模式的参考面板: 现在动手会承担什么, 历史上同类情况是什么结果。"""
        out = {"hist_msg": self.hist_msg, "range": (self.hist or {}).get("range"),
               "min_n": ADVICE_MIN_N, "pat_cn": PAT_CN}
        ind = self._indicators()
        if ind is None:
            out["ready"] = False
            return out
        out["ready"] = True
        ent = px if px > 0 else ind["last"]

        # 现在买: 按自动盘同款口径(跟踪止损)会承担多少
        t = self.trader_ref
        stop_pct = float(getattr(t, "trail_pct", None) or 0.08)
        out["entry"] = {
            "px": ent, "stop": ent * (1 - stop_pct), "stop_pct": stop_pct,
            "breakeven": 0.002,                      # 双边手续费 0.1%×2
            "target2r": ent * (1 + 2 * stop_pct),    # 2:1 盈亏比对应的目标价
        }

        # 现在卖: 手动持仓时锁定多少
        mt = self.manual_trader()
        if mt is not None and mt.units > 0 and mt.entry_price > 0:
            pnl = (ent - mt.entry_price) * mt.units
            held = None
            try:
                held = (time.time() - pd.Timestamp(mt.entry_time).timestamp()) / 3600.0
            except Exception:
                pass
            out["exit"] = {"pnl": pnl, "pnl_pct": pnl / START_EQUITY,
                           "ret_on_pos": ent / mt.entry_price - 1, "held_h": held}

        # 历史同类情况
        hist = self.hist or {}
        out["base"] = hist.get("base", {})
        out["trigger"] = hist.get("trigger", {})
        flags = self._cur_flags()
        out["flags"] = [[f, PAT_CN.get(f, f), hist.get("flags", {}).get(f, {})] for f in flags]

        # 情报
        g = lambda k: ind.get(k)
        out["intel"] = {
            "vol_ratio": g("vol_ratio"), "atr_pct": g("atr_pct"),
            "vs1440": (ent / g("sma1440") - 1) if g("sma1440") else None,
            "ma_spread": (g("sma48") / g("sma336") - 1) if (g("sma48") and g("sma336")) else None,
            "dd72": ent / g("high72") - 1, "chg24h": g("chg24h"),
            "news_risk": self.status.get("news_risk"),
            "news_msg": (self.status.get("news_msg") or "")[:120],
            "pattern": self.status.get("pattern"),
            "signal": float(self.status.get("signal") or 0.0),
        }

        # 给个明确判断, 别两边都说
        sig = out["intel"]["signal"]
        holding = mt is not None and mt.units > 0
        auto_holding = t is not None and t.units > 0
        if sig >= 1 and not holding:
            out["verdict"] = "策略信号=做多。但这套趋势信号的胜率历来只有两成上下, 靠盈亏比赚钱, 不是靠猜对方向。"
            out["verdict_color"] = "ok"
        elif holding and sig >= 1:
            out["verdict"] = "策略信号=继续持有, 你也在场, 可以盯着自动盘什么时候走。"
            out["verdict_color"] = "ok"
        elif holding and sig < 1:
            out["verdict"] = "策略信号已经转空仓, 而你还拿着 —— 历史上信号转空后继续持有的结果见下面统计。"
            out["verdict_color"] = "warn"
        else:
            out["verdict"] = "策略信号=空仓等待。现在入场属于逆着信号做, 先看下面的历史基准率再决定。"
            out["verdict_color"] = "gray"
        out["auto"] = ("自动盘持仓中" if auto_holding else "自动盘空仓")
        return out

    # ---------------- 控制面板动作 ----------------

    def check_conn(self):
        def work():
            self.log("开始检查网络: 先试 Binance, 不通再试 OKX / Gate.io(最长约90秒)...")
            self.status["net"] = "unknown"
            src = self._probe_source()
            self.src = src
            self.ensure_live()
            if src == "binance":
                self.status.update({
                    "net": "ok", "source": SRC_LABEL["binance"],
                    "net_msg": "Binance 连接正常(无需VPN/密钥, 本地虚拟成交)",
                })
                self.log("Binance 连接正常!(本地虚拟成交)")
            elif src == "okx":
                self.status.update({
                    "net": "ok", "source": SRC_LABEL["okx"],
                    "net_msg": "Binance不通, 已自动用 OKX 行情",
                })
                self.log("Binance 连不上, 已自动切换到 OKX。")
            elif src == "gate":
                self.status.update({
                    "net": "ok", "source": SRC_LABEL["gate"],
                    "net_msg": "Binance/OKX不通, 已自动用 Gate.io 行情",
                })
                self.log("Binance 和 OKX 都连不上, 已自动切换到 Gate.io(无需VPN/密钥, 本地虚拟成交)。")
            else:
                self.status.update({
                    "net": "fail", "source": "离线",
                    "net_msg": "都连不上, 请检查网络",
                })
                self.log("Binance / OKX / Gate.io 都连不上, 请检查网络后重试。")

        threading.Thread(target=work, daemon=True).start()

    def start(self, cfg):
        if self.worker and self.worker.is_alive():
            self.log("已经在运行中")
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.worker.start()
        # 多品种子账户: 与主盘同一套策略参数, 方便直接对比单品种和多品种
        if cfg.get("eth_enabled", True):
            # 每次都从磁盘重建: 账本被手动清过/改过时, 内存里的旧持仓不该继续沿用
            self.sleeves = [Sleeve("ETH", ETH_SYMBOL, ETH_PAIR, ETH_STATE, ETH_TRADES)]
            self.sleeve_threads = []
            for sv in self.sleeves:
                th = threading.Thread(
                    target=sv.run, args=(self.stop_event, self.log),
                    kwargs={"poll": cfg.get("poll_seconds", 60),
                            "risk": cfg.get("risk_pct", 0.02),
                            "stop": cfg.get("stop_pct", 0.15),
                            "trail": cfg.get("trail_pct", 0.08)},
                    daemon=True)
                th.start()
                self.sleeve_threads.append(th)
        else:
            self.log("多品种子账户: 已关闭")

    def stop(self):
        self.stop_event.set()
        self.log("正在停止(最多60秒内生效)...")

    def _trade_marks(self):
        trades = []
        if TRADES.exists():
            try:
                rows = []
                with TRADES.open(encoding="utf-8") as f:
                    for r in _csv.DictReader(f):
                        rows.append(r)
                for r in rows[-200:]:
                    try:
                        ts = r.get("time_utc", "")
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        trades.append({
                            "t": int(dt.timestamp() * 1000),
                            "px": float(r["fill_px"]),
                            "side": r["side"],
                            "time": ts[:16],
                        })
                    except Exception:
                        continue
            except Exception:
                pass
        return trades

    def chart_data(self, scale="1h", hours=24):
        with self.lock:
            cache = self.cache.copy() if self.cache is not None and len(self.cache) else None
            live_trades = [list(x) for x in (self.live or {}).get("trades", [])]
        trades = self._trade_marks()
        ts_now = int(time.time() * 1000)
        candle_time = self.status.get("candle_time", "")
        if scale in ("30s", "1m"):
            step_ms = 30_000 if scale == "30s" else 60_000
            buckets = {}
            for t, p in live_trades:
                b = int(t) // step_ms * step_ms
                buckets[b] = float(p)
            bars = [[b, p] for b, p in sorted(buckets.items())]
            return {"bars": bars, "trades": trades, "pats": [], "scale": scale,
                    "ts_ms": ts_now, "candle_time": candle_time}
        if scale == "10m":
            bars = ten_min_bars()[-300:]
            return {"bars": bars, "trades": trades, "pats": [], "scale": scale,
                    "ts_ms": ts_now, "candle_time": candle_time}
        if cache is None:
            return {"bars": [], "trades": trades, "pats": [], "scale": "1h",
                    "ts_ms": ts_now, "candle_time": candle_time}
        now = int(time.time() * 1000)
        _n = int(hours) if hours and int(hours) > 0 else 1200
        closed = cache[cache["close_time_ms"] <= now].tail(_n)
        bars = [[int(r["open_time_ms"]), float(r["close"])] for _, r in closed.iterrows()]
        pats = []
        try:
            pat = getattr(self, "pat_df", None)
            if pat is None or len(pat) != len(cache) or pat.index.max() != cache.index.max():
                pat = detect_patterns(cache)
                self.pat_df = pat
            for i, r in closed.iterrows():
                try:
                    pr = pat.loc[i]
                except Exception:
                    continue
                lbl = str(pr["label"])
                if lbl in SIGNIFICANT:
                    pats.append({"t": int(r["open_time_ms"]), "label": lbl,
                                 "strength": str(pr["strength"])})
        except Exception:
            pass
        return {"bars": bars, "trades": trades, "pats": pats, "scale": "1h",
                "ts_ms": ts_now, "candle_time": candle_time}

    # ---------------- 秒级行情 ----------------

    def ensure_live(self):
        """确保有一个后台线程在取实时价(无密钥也能用)。"""
        src = self.src
        if not src:
            return
        if self.live_thread and self.live_thread.is_alive():
            return
        self.live_thread = threading.Thread(target=self._live_loop, args=(src,), daemon=True)
        self.live_thread.start()

    def _live_loop(self, src):
        n = 0
        while not self.stop_event.is_set():
            cur = self.src or src
            try:
                tick = self._load_ticker(cur)
                px = float(tick["last"])
                trades = None
                if n % 4 == 0:
                    _t0 = time.perf_counter()
                    if cur == "gate":
                        trades = ds.gate_recent_trades(1000)
                    elif cur == "binance":
                        trades = ds.binance_recent_trades(1000)
                    else:
                        d = okx_request("GET", "/api/v5/market/trades", query={"instId": INST, "limit": 100})
                        trades = sorted([[int(x["ts"]), float(x["px"])] for x in d["data"]])
                    self.lat["trades"] = time.perf_counter() - _t0
                with self.lock:
                    live = dict(self.live or {"price": 0.0, "trades": [], "ts": 0.0})
                    live["price"] = px
                    live["ts"] = time.time()
                    live["exch_ts"] = tick.get("ts") or 0
                    if trades is not None:
                        live["trades"] = trades
                        live["trades_ts"] = time.time()
                    self.live = live
                    self.status["ticker_time"] = time.strftime("%H:%M:%S")
                    self.status["ticker_ts_ms"] = int(time.time() * 1000)
            except Exception:
                pass
            n += 1
            for _ in range(2):
                if self.stop_event.is_set():
                    return
                time.sleep(0.5)

    def live_snapshot(self):
        with self.lock:
            live = dict(self.live) if self.live else {}
            t = self.trader_ref
        px = float(live.get("price", 0.0) or 0.0)
        now_ms = int(time.time() * 1000)
        elapsed = now_ms % 3_600_000
        out = {
            "price": px,
            "trades": live.get("trades", []) or [],
            "price_ts": int(live.get("ts", 0) * 1000) or 0,
            "exch_ts": int(live.get("exch_ts") or 0),
            "trades_ts": int(live.get("trades_ts", 0) * 1000) or 0,
            "server_ts_ms": now_ms,
            "bar_min": int(elapsed // 60_000),
            "sec_to_next": int((3_600_000 - elapsed) // 1000),
            "upnl": None,
            "entry": 0.0, "stop": 0.0, "units": 0.0,
            "lat_price": round(self.lat["price"], 6),
            "lat_kline": round(self.lat["kline"], 6),
            "lat_trades": round(self.lat["trades"], 6),
        }
        if t is not None:
            out["units"] = t.units
            out["entry"] = t.entry_price or 0.0
            out["stop"] = t.stop_price if np.isfinite(t.stop_price) else 0.0
            if t.units > 0 and t.entry_price > 0 and px > 0:
                out["upnl"] = round((px - t.entry_price) * t.units, 2)
        return out

    # ---------------- 主循环 ----------------

    def _run(self, cfg):
        st = self.status
        st["running"] = True
        key = cfg.get("api_key", "")
        secret = cfg.get("api_secret", "")
        passphrase = cfg.get("api_passphrase", "")
        simulated = bool(cfg.get("simulated", True))
        risk = float(cfg.get("risk_pct", 0.02))
        stop = float(cfg.get("stop_pct", 0.15))
        trail = float(cfg.get("trail_pct", 0.08) or 0)
        poll = float(cfg.get("poll_seconds", 60))
        auto_dump = bool(cfg.get("auto_dump", False))
        auto_filter = bool(cfg.get("auto_filter", False))
        news_gate = bool(cfg.get("news_gate", True))
        st["auto_dump"], st["auto_filter"] = auto_dump, auto_filter
        st["news_gate"] = news_gate
        st["eth_enabled"] = bool(cfg.get("eth_enabled", True))
        # 面板上那三个参数框要能显示"正在跑的"配置, 否则一按[保存并启动]就被表单默认值覆盖
        st["risk_pct"], st["stop_pct"], st["trail_pct"] = risk, stop, trail
        has_keys = bool(key and secret and passphrase)
        st["has_keys"] = has_keys
        cny_default = float(cfg.get("cny_rate", 7.2) or 7.2)
        fresh_rate = _fetch_cny_rate()
        self.fx_rate = fresh_rate if fresh_rate > 0 else cny_default
        self.log(f"人民币汇率: {self.fx_rate:.4f} {'(自动获取)' if fresh_rate > 0 else '(用面板填的汇率)'}")
        t = PaperTrader(cash=100.0, risk_pct=risk, stop_pct=stop, log_path=TRADES, trail_pct=trail)
        if STATE.exists():
            try:
                with STATE.open(encoding="utf-8") as f:
                    prev = json.load(f)
                t.cash, t.units, t.entry_price, t.entry_equity = (
                    prev["cash"], prev["units"], prev["entry_price"], prev["entry_equity"]
                )
                # entry_cost 也必须恢复, 否则卖出盈亏会被算成"全额收入"(旧状态无此字段时按名义成本补)
                t.entry_cost = prev.get(
                    "entry_cost",
                    (t.units * t.entry_price / (1.0 - t.fee_rate)) if t.units > 0 else 0.0,
                )
                t.stop_price = prev.get("stop_price", float("inf"))
                t.high_watermark = prev.get("high_watermark", t.entry_price or 0.0)
                if t.trail_pct and t.units > 0 and t.high_watermark > 0:
                    t.stop_price = max(t.stop_price, t.high_watermark * (1 - t.trail_pct))
                t.signal = prev.get("signal", 0.0)
                t.reentry_bar_ms = prev.get("reentry_bar_ms", 0)
                t.last_bar_ms = prev.get("last_bar_ms", 0)
                # 账本一致性自检(启动自愈): 持仓中 cash 应 = entry_equity - units*entry/(1-手续费)
                if t.units > 0 and t.entry_equity > 0:
                    _expect = t.entry_equity - t.units * t.entry_price / (1.0 - t.fee_rate)
                    if abs(t.cash - _expect) > 0.01:
                        self._audit_cash("state_inconsistent(启动自愈)", t.cash, _expect, t.units, t.units)
                        self.log(f"[账本告警] 启动发现 cash 不一致: {t.cash:.4f} -> 应为 {_expect:.4f}, 已自动修正")
                        t.cash = _expect
                self.log(f"已恢复上次状态: 现金 {t.cash:.2f} | 持仓 {t.units:.8f}")
            except Exception as e:
                self.log(f"恢复状态失败({e}), 从头开始")
        src = self._probe_source()
        self.src = src
        if src is None:
            st.update({"source": "离线", "mode": "未启动"})
            self.log("启动失败: OKX 和 Gate.io 都连不上。请检查网络后点[检查连接]。")
            st["running"] = False
            return
        real_orders = has_keys and src == "okx"
        lot = 0.00001
        if has_keys and src == "okx":
            try:
                lot = lot_size()
                t.cash = usdt_balance(key, secret, passphrase, simulated)
                self.log(f"OKX模拟盘余额(虚拟资金): {t.cash:,.2f} USDT")
            except Exception as e:
                self.log(f"读取OKX余额失败({e}), 使用本地记录")
        if has_keys and src != "okx":
            self.log("警告: 当前数据源不是OKX(币安优先), 密钥暂不可用, 自动改为本地虚拟成交。")
        st["source"] = SRC_LABEL.get(src, src)
        st["mode"] = "OKX模拟盘真实挂单" if real_orders else "本地虚拟成交"
        self.log(f"模式: {st['mode']} | 数据源: {st['source']} | 风险 {risk:.1%} | 止损 {stop:.0%} | 形态离场 {'开' if auto_dump else '关'} | 开仓过滤 {'开' if auto_filter else '关'}")
        try:
            self.log("正在拉取历史1小时K线(第一次约10~60秒)...")
            cache = self._load_candles(src, initial=True)
            if len(cache) < 1441:
                self.log("历史K线不足1440根, 自动用 Gate.io 补足历史数据...")
                try:
                    extra = ds.gate_candles(2000)
                    cache = pd.concat([cache, extra]).drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
                except Exception as e2:
                    self.log(f"补数失败({e2}), 等新K线收盘后自动补齐")
            if len(cache) < 1441:
                self.log(f"警告: 历史K线只有 {len(cache)} 根, 信号暂时无法计算(需要1440根), 每小时会自动补齐")
            else:
                self.log(f"已取到 {len(cache)} 根历史K线, 信号正常")
        except Exception as e:
            self.log(f"启动失败, 拉不到行情: {e}。请先点[检查连接]。")
            st["running"] = False
            return
        self.trader_ref = t
        self.ensure_live()
        try:
            rep_news = nr.assess(nr.fetch_items(), hours=6)
            self.news = rep_news
            st["news_risk"] = rep_news["risk"]
            st["news_msg"] = rep_news.get("msg", "")
            st["news_items"] = rep_news["items"][:6]
            st["news_time"] = time.strftime("%H:%M:%S")
            self.last_news_ms = int(time.time() * 1000)
            self.last_news_risk = rep_news["risk"]
            self.log(f"[新闻雷达] 首轮扫描完成: 风险={rep_news['risk']} | {rep_news.get('msg', '')}")
        except Exception as e:
            self.log(f"[新闻雷达] 首轮扫描失败({type(e).__name__}), 10分钟后重试")
        fail_streak = 0
        while not self.stop_event.is_set():
            try:
                _cash0, _units0 = t.cash, t.units  # 审计: 本轮现金/持仓快照
                now = int(time.time() * 1000)
                fresh = self._load_candles(src, initial=False)
                fail_streak = 0
                st["candle_time"] = time.strftime("%H:%M:%S")
                st["candle_ts_ms"] = now
                cache = pd.concat([cache, fresh]).drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)
                if len(cache) > 3000:
                    cache = cache.iloc[-3000:]
                with self.lock:
                    self.cache = cache.copy()
                closed = cache[cache["close_time_ms"] <= now]
                _skip_history(t, closed)
                sig = compute_signal(cache)
                valid_idx = np.flatnonzero(~np.isnan(sig))
                st["signal"] = float(sig[valid_idx[-1]]) if len(valid_idx) else 0.0
                st["last_close"] = float(closed.iloc[-1]["close"]) if len(closed) else 0.0
                # 诊断: 实时信号 vs 最后一根已收盘K线的信号(后者才决定卖不卖) + 已处理到哪根
                if len(closed):
                    _cp = int((cache["close_time_ms"] <= closed.iloc[-1]["close_time_ms"]).sum()) - 1
                    _cs = float(sig[_cp]) if 0 <= _cp < len(sig) and not np.isnan(sig[_cp]) else -1.0
                else:
                    _cs = -1.0
                st["closed_signal"] = _cs
                st["last_bar_t"] = (pd.to_datetime(t.last_bar_ms, unit="ms", utc=True)
                                    .strftime("%m-%d %H:%M") if t.last_bar_ms else "-")
                # 存档: 把新收盘的K线 + 其信号 追加到 bars_history.csv(供日后精确回溯/复算)
                try:
                    _am = getattr(self, "_arch_ms", 0)
                    _new = closed[closed["open_time_ms"] > _am] if len(closed) else closed
                    if len(_new):
                        _c = cache["close_time_ms"].to_numpy()
                        _rows = []
                        for _, _r in _new.iterrows():
                            _p = int((_c <= _r["close_time_ms"]).sum()) - 1
                            # 缓存前 1440 根还没有60日线, 那时的信号必然是0 —— 写空字符串,
                            # 别写 0 冒充"当时判断为空仓", 复盘时会把人带沟里
                            _s = ("" if _p < GATE_BARS else
                                  (float(sig[_p]) if 0 <= _p < len(sig) and not np.isnan(sig[_p]) else ""))
                            _rows.append([int(_r["open_time_ms"]), int(_r["close_time_ms"]),
                                          float(_r["open"]), float(_r["high"]), float(_r["low"]),
                                          float(_r["close"]), float(_r["volume"]), _s])
                        _isnew = not BARS_HIST.exists()
                        with BARS_HIST.open("a", encoding="utf-8", newline="") as _f:
                            _w = _csv.writer(_f)
                            if _isnew:
                                _w.writerow(["open_time_ms", "close_time_ms", "open", "high",
                                             "low", "close", "volume", "signal"])
                            _w.writerows(_rows)
                        self._arch_ms = int(_new["open_time_ms"].max())
                except Exception:
                    pass
                need_pat = (now - getattr(self, "last_pat_ms", 0) >= 10 * 60 * 1000) or (
                    len(closed) > 0 and int(closed.iloc[-1]["close_time_ms"]) > getattr(self, "last_pat_bar_ms", 0)
                )
                if need_pat:
                    try:
                        pat = detect_patterns(cache)
                    except Exception:
                        pat = getattr(self, "pat_df", None)
                    self.pat_df = pat
                    self.last_pat_ms = now
                    if len(closed):
                        self.last_pat_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                    st["pattern_time"] = time.strftime("%H:%M:%S")
                    st["pattern_next"] = time.strftime("%H:%M", time.localtime(now / 1000 + 600))
                    if now - self.last_news_ms >= 10 * 60 * 1000:
                        self.last_news_ms = now
                        try:
                            rep_news = nr.assess(nr.fetch_items(), hours=6)
                        except Exception as e:
                            rep_news = {"risk": "未知", "score": 0, "items": [],
                                        "msg": f"新闻雷达异常: {type(e).__name__}"}
                        self.news = rep_news
                        st["news_risk"] = rep_news["risk"]
                        st["news_msg"] = rep_news.get("msg", "")
                        st["news_items"] = rep_news["items"][:6]
                        st["news_time"] = time.strftime("%H:%M:%S")
                        self.log(f"[新闻雷达] 风险={rep_news['risk']} | {rep_news.get('msg', '')}")
                        if t.units > 0 and rep_news["risk"] == "高" and self.last_news_risk != "高":
                            self._record_decision("不卖", "新闻面利空(风险高), 继续持有, 请盯紧止损线", st.get("last_close", 0.0), 1)
                        self.last_news_risk = rep_news["risk"]
                    if len(closed) and pat is not None and closed.index[-1] in pat.index:
                        prow = pat.loc[closed.index[-1]]
                        label = str(prow["label"])
                        st["pattern"] = label if label == "无" else f"{label}({str(prow['strength'])})"
                        st["pattern_msg"] = str(prow["msg"])
                        st["pattern_base"] = BASE_RATE.get(label, "该形态历史统计与整体平均接近, 无明显方向性")
                        bar_ms = int(closed.iloc[-1]["close_time_ms"])
                        if (not self.pattern_history) or self.pattern_history[-1].get("bar_ms") != bar_ms:
                            self.pattern_history.append({
                                "t": datetime.fromtimestamp(bar_ms / 1000).strftime("%m-%d %H:%M"),
                                "bar_ms": bar_ms,
                                "label": label,
                                "strength": str(prow["strength"]),
                                "msg": str(prow["msg"]),
                            })
                            self.pattern_history = self.pattern_history[-500:]
                            try:
                                _atomic_write(PAT_HIST, self.pattern_history)
                            except Exception:
                                pass
                else:
                    pat = getattr(self, "pat_df", None)
                dec_before = len(self.decision_rows)
                new_bars = closed[closed["close_time_ms"] > t.last_bar_ms]
                for _, bar in new_bars.iterrows():
                    pos = int((closed["close_time_ms"] <= bar["close_time_ms"]).sum()) - 1
                    s = float(sig[pos])
                    if np.isnan(s):
                        continue
                    if pat is None or bar.name not in pat.index:
                        continue
                    prow = pat.loc[bar.name]
                    t.last_close = bar["close"]
                    t.last_bar_ms = int(bar["close_time_ms"])
                    ts = pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True)
                    if not self._verify_bar(src, bar):
                        continue
                    _vw = closed["volume"].tail(VOL_WINDOW)
                    _vm = float(_vw.median()) if len(_vw) else 0.0
                    vr = float(bar["volume"]) / _vm if _vm > 0 else 0.0
                    vol_note = f"成交量{vr:.1f}倍常态" if vr > 0 else "成交量未知"
                    filter_entry = auto_filter and (bool(prow["range"]) or bool(prow["pump"]))
                    if t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms and not filter_entry and not (news_gate and self.news.get("risk") == "高"):
                        if real_orders:
                            try:
                                notional = t.cash * risk / stop
                                avg, fill, oid = okx_market_order("buy", f"{notional:.2f}", key, secret, passphrase, simulated)
                                t.cash = usdt_balance(key, secret, passphrase, simulated)
                                t.buy(avg, "signal", ts)
                                t.signal = 1.0
                                self.log(f"[{ts:%m-%d %H:%M}] 信号买入 {fill:.8f} BTC @ {avg:.2f}, 止损 {t.stop_price:.2f}")
                                self._record_decision("买", f"做多信号确认, 买入 {fill:.8f} BTC", avg, 1)
                            except Exception as e:
                                t.reentry_bar_ms = int(bar["close_time_ms"]) + 3_600_000
                                self.log(f"[{ts:%m-%d %H:%M}] 信号买入下单失败({type(e).__name__}): {e}, 1小时后重试")
                                self._record_decision("不买", "做多信号, 下单失败, 1小时后重试", bar["close"], 1)
                        else:
                            t.buy(bar["close"], "signal", ts)
                            t.signal = 1.0
                            self.log(f"[{ts:%m-%d %H:%M}] 信号买入(虚拟) @ {bar['close']:.2f}, 止损 {t.stop_price:.2f}, {vol_note}")
                            self._record_decision("买", f"做多信号确认({vol_note}), 虚拟买入 @{bar['close']:.2f}", bar["close"], 1)
                    elif t.units > 0 and (s == 0 or (auto_dump and bool(prow["dump"]))):
                        reason = "signal" if s == 0 else "pattern_dump"
                        if reason == "pattern_dump":
                            t.reentry_bar_ms = int(bar["close_time_ms"]) + 24 * 3_600_000
                        if real_orders:
                            try:
                                qty = np.floor(t.units / lot) * lot
                                okx_market_order("sell", f"{qty:.8f}", key, secret, passphrase, simulated)
                                t.cash = usdt_balance(key, secret, passphrase, simulated)
                                t.sell(bar["close"], reason, ts)
                                t.signal = 0.0
                                self._record_decision("卖", ("信号转空" if reason == "signal" else "瀑布/砸盘自动离场") + ", 卖出", bar["close"], 0)
                            except Exception as e:
                                self.log(f"[{ts:%m-%d %H:%M}] 信号卖出下单失败({type(e).__name__}): {e}, 下根K线继续尝试")
                                self._record_decision("不卖", "卖出下单失败, 下根K线重试", bar["close"], 0)
                        else:
                            t.sell(bar["close"], reason, ts)
                            t.signal = 0.0
                            self._record_decision("卖", ("信号转空" if reason == "signal" else "瀑布/砸盘自动离场") + ", 卖出", bar["close"], 0)
                        label = "信号" if reason == "signal" else "形态(瀑布/砸盘)"
                        self.log(f"[{ts:%m-%d %H:%M}] {label}卖出 @ {bar['close']:.2f}")
                    elif t.units == 0 and s == 0:
                        self._record_decision("不买", "无做多信号, 空仓观望", bar["close"], 0)
                    elif t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] <= t.reentry_bar_ms:
                        self._record_decision("不买", "出现做多信号, 但离场冷却期内, 跳过", bar["close"], 1)
                    elif t.units == 0 and s == 1 and t.signal == 0 and filter_entry:
                        self._record_decision("不买", "做多信号, 但盘整/拉盘过滤开启, 暂停开仓", bar["close"], 1)
                    elif t.units > 0:
                        if bool(prow["dump"]):
                            self._record_decision("不卖", "检测到瀑布/砸盘, 但自动离场已关, 继续持有", bar["close"], 1)
                        else:
                            self._record_decision("不卖", "信号仍做多, 继续持有", bar["close"], 1)
                    elif t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms and not filter_entry and news_gate and self.news.get("risk") == "高":
                        self._record_decision("不买", "新闻面利空(近6小时风险高), 暂不开仓", bar["close"], 1)
                # 安全网: 仍持仓 + "最后一根已收盘K线"看空 => 立即补卖
                # (防止 last_bar_ms 已吞掉该K线, 导致永久漏卖)
                if t.units > 0 and len(closed) and st.get("closed_signal") == 0.0:
                    _sx = float(closed.iloc[-1]["close"])
                    _ts = pd.to_datetime(now, unit="ms", utc=True)
                    try:
                        if real_orders:
                            _qty = np.floor(t.units / lot) * lot
                            okx_market_order("sell", f"{_qty:.8f}", key, secret, passphrase, simulated)
                            t.cash = usdt_balance(key, secret, passphrase, simulated)
                        t.sell(_sx, "signal", _ts)
                        t.signal = 0.0
                        t.reentry_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                        self.log(f"[安全网] 收盘信号=0 仍持仓, 已补卖出 @ {_sx:.2f}")
                        self._record_decision("卖", "收盘信号转空, 补卖出(安全网)", _sx, 0)
                    except Exception as e:
                        self.log(f"[安全网] 补卖失败({type(e).__name__}): {e}")
                if t.units > 0:
                    try:
                        px = float(self._load_ticker(src)["last"])
                        t.update_stop(px)
                        if px <= t.stop_price:
                            if len(closed):
                                t.reentry_bar_ms = int(closed.iloc[-1]["close_time_ms"])
                            ts_now = pd.to_datetime(now, unit="ms", utc=True)
                            if real_orders:
                                try:
                                    qty = np.floor(t.units / lot) * lot
                                    okx_market_order("sell", f"{qty:.8f}", key, secret, passphrase, simulated)
                                    t.cash = usdt_balance(key, secret, passphrase, simulated)
                                    t.sell(min(px, t.stop_price), "stop", ts_now)
                                    t.signal = 0.0
                                    self.log(f"[{ts_now:%m-%d %H:%M}] 止损卖出 @ {px:.2f}")
                                    self._record_decision("卖", "价格跌破止损线, 止损卖出", px, 0)
                                except Exception as e:
                                    self.log(f"[{ts_now:%m-%d %H:%M}] 止损卖出下单失败({type(e).__name__}): {e}, 下轮重试")
                                    self._record_decision("不卖", "止损下单失败, 下轮重试", px, 0)
                            else:
                                t.sell(min(px, t.stop_price), "stop", ts_now)
                                t.signal = 0.0
                                self.log(f"[{ts_now:%m-%d %H:%M}] 止损卖出 @ {px:.2f}")
                                self._record_decision("卖", "价格跌破止损线, 止损卖出", px, 0)
                    except Exception as e:
                        self.log(f"读取实时价失败({type(e).__name__}), 本次跳过止损检查")
                if len(self.decision_rows) == dec_before:
                    cur_sig = float(st.get("signal", 0.0) or 0.0)
                    last_px = float(st.get("last_close", 0.0) or 0.0)
                    if t.units > 0:
                        _dbg = f"[实时={cur_sig:.0f} 收盘={st.get('closed_signal')} 已处理至{st.get('last_bar_t')}]"
                        if cur_sig == 1.0:
                            self._record_decision("不卖", f"持仓中, 信号仍做多, 继续持有 {_dbg}", last_px, cur_sig)
                        else:
                            self._record_decision("不卖", f"实时信号已转空, 等本小时K线收盘确认后卖出 {_dbg}", last_px, cur_sig)
                    else:
                        if cur_sig == 1.0:
                            if t.reentry_bar_ms and now < t.reentry_bar_ms:
                                self._record_decision("不买", "出现做多信号, 但离场冷却期内, 不买", last_px, cur_sig)
                            else:
                                self._record_decision("不买", "出现做多信号, 等K线收盘确认后买入", last_px, cur_sig)
                        else:
                            self._record_decision("不买", "无做多信号, 空仓观望", last_px, cur_sig)
                st["balance"], st["units"], st["entry"] = t.cash, t.units, t.entry_price
                st["stop"] = t.stop_price if np.isfinite(t.stop_price) else 0.0
                if t.cash != _cash0 or t.units != _units0:
                    _reason = "trade" if t.units != _units0 else "UNEXPLAINED(无交易却变)"
                    self._audit_cash(_reason, _cash0, t.cash, _units0, t.units)
                _atomic_write(STATE, {
                    "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                    "entry_equity": t.entry_equity, "entry_cost": t.entry_cost,
                    "stop_price": t.stop_price,
                    "high_watermark": t.high_watermark,
                    "signal": t.signal, "reentry_bar_ms": t.reentry_bar_ms,
                    "last_bar_ms": t.last_bar_ms,
                })
                with self.lock:
                    live_px = (self.live or {}).get("price", 0.0) or 0.0
                eq_px = live_px if live_px > 0 else (t.last_close or 0.0)
                if eq_px > 0:
                    self._record_daily(t.cash + t.units * eq_px, eq_px, t.units)
            except Exception as e:
                fail_streak += 1
                _next = {"binance": "okx", "okx": "gate"}.get(src)
                if _next and fail_streak >= 3:
                    _old = src
                    src = _next
                    self.src = src
                    fail_streak = 0
                    st["source"] = SRC_LABEL[src]
                    st["net_msg"] = f"{_old}不稳, 已自动切换到 {st['source']}"
                    self.log(f"{_old} 连续断线3次, 已自动切换到 {st['source']}。状态不变, 继续运行; 想切回请重启控制台。")
                elif fail_streak == 1 or fail_streak % 5 == 0:
                    self.log(f"网络抖动({type(e).__name__}), 本轮跳过, 每60秒自动重试(已连续失败{fail_streak}次)")
            for _ in range(int(poll)):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        st["running"] = False
        self.log("已停止。状态已保存, 下次启动自动恢复。")


_SERVER = None  # main() 里赋值, 自重启时用来释放端口


def _code_snapshot():
    import glob
    fs = glob.glob(str(ROOT / "*.py")) + glob.glob(str(ROOT / "strategies" / "*.py"))
    return {f: os.path.getmtime(f) for f in fs if os.path.exists(f)}


def _code_ok():
    for f in _code_snapshot():
        try:
            compile(open(f, encoding="utf-8").read(), f, "exec")
        except Exception:
            return False
    return True


def restart_self(delay=0.8, log=None, reopen=False):
    """重启: 保存状态后以退出码 42 退出, 由"启动模拟盘.bat"的循环监工自动重新拉起。
    (不再自己 spawn 新进程——那在 Windows 上不可靠, 会把控制台弄死。)
    """
    def _do():
        time.sleep(delay)
        try:
            if log:
                log("正在重启, 即将退出并由启动脚本自动拉起...")
            CORE.stop_event.set()
            try:
                if _SERVER:
                    _SERVER.shutdown()
                    _SERVER.server_close()
            except Exception:
                pass
            time.sleep(0.6)
        finally:
            os._exit(42)  # 42 = 请求重启; 启动脚本据此重新拉起
    threading.Thread(target=_do, daemon=True).start()


def watch_code():
    """后台监视: 代码文件一变(且全部能编译) -> 自动重启重载。"""
    base = _code_snapshot()
    while True:
        time.sleep(15)
        cur = _code_snapshot()
        if cur != base:
            if _code_ok():
                CORE.log("检测到代码更新, 自动重启重载...")
                restart_self(log=CORE.log, reopen=False)
                return
            base = cur  # 代码当前有错, 等修好再看


CORE = Core()

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>比特币模拟盘控制台</title>
<style>
body{font-family:"Microsoft YaHei",SimHei,sans-serif;max-width:1280px;margin:0 auto;padding:20px;background:#0d1117;color:#e6edf3}
h1{font-size:22px;border-bottom:3px solid #ff7f0e;padding-bottom:8px;margin-bottom:4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:14px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0 14px;align-items:start}
.col{min-width:0}
.col.a{order:2}
.col.b{order:1}
@media(max-width:1000px){.cols{grid-template-columns:1fr}.col.a,.col.b{order:0}}
.item{border:1px solid #30363d;border-radius:8px;padding:10px;background:#0d1117}
.item .k{color:#8b949e;font-size:12px}
.item .v{font-size:18px;font-weight:bold;margin-top:2px}
.ok{color:#3fb950}.fail{color:#f85149}.gray{color:#8b949e}
label{display:block;font-size:13px;margin:8px 0 2px;color:#c9d1d9}
input[type=text],input[type=password]{width:100%;padding:8px;border:1px solid #30363d;border-radius:6px;font-size:13px;box-sizing:border-box;background:#0d1117;color:#e6edf3}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row>div{flex:1;min-width:140px}
button{border:none;border-radius:6px;padding:10px 18px;font-size:14px;cursor:pointer;margin-right:8px;font-family:inherit;color:#e6edf3}
.btn-check{background:#0c2d6b}.btn-check:hover{background:#123f8f}
.btn-start{background:#0f3d2e}.btn-start:hover{background:#155843}
.btn-stop{background:#5b1a1a}.btn-stop:hover{background:#7a2323}
.btn-quit{background:#30363d}.btn-quit:hover{background:#3d444d}
button:disabled{opacity:.45;cursor:not-allowed}
#log{background:#010409;color:#9ce39c;font-family:Consolas,monospace;font-size:12px;border-radius:8px;border:1px solid #30363d;padding:10px;height:220px;overflow-y:auto;white-space:pre-wrap}
.hint{font-size:12px;color:#8b949e;margin-top:8px}
</style></head><body>
<h1>比特币模拟盘控制台</h1>
<div class="sub">杂交趋势策略: SMA48/336 + 收盘&gt;SMA1440 做多 | 15% 止损 | 每笔风险 1% | 现货只做多 | 1小时K线</div>
<div id="offline" style="display:none;background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:10px;margin:10px 0;color:#7a5c00;font-size:14px">与程序失去连接(程序已退出或没启动)。请重新双击文件夹里的「启动模拟盘.bat」。</div>

<div class="card">
<div style="font-weight:bold;margin-bottom:6px">实时行情 <span style="color:#888;font-size:12px">(约1秒刷新, 行情自动选Binance/OKX/Gate.io)</span> <span id="tick_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span>
<span style="float:right;font-weight:normal;font-size:12px;color:#8b949e">本图尺度
<select id="scale_live_sel" style="font-size:12px;padding:1px 4px">
<option value="30" selected>30秒</option>
<option value="60">1分</option>
<option value="600">10分</option>
<option value="3600">1小时</option>
</select></span></div>
<div class="row" style="align-items:baseline">
<div><div class="k" style="color:#888;font-size:12px">最新价</div><div id="live_px" style="font-size:26px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">持仓浮盈亏</div><div id="upnl" style="font-size:20px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">下次信号判断</div><div id="countdown" style="font-size:20px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">本小时K线进度</div><div id="bar_prog" style="font-size:20px;font-weight:bold;margin-top:2px">-</div></div>
</div>
<svg id="livechart" viewBox="0 0 900 140" preserveAspectRatio="none" style="width:100%;height:170px;background:#0d1117;border:1px solid #30363d;border-radius:8px;margin-top:10px;display:block"></svg>
</div>

<div class="cols"><div class="col a">
<div class="card">
<div class="grid">
<div class="item"><div class="k">网络</div><div class="v" id="net" style="font-size:12px">-</div></div>
<div class="item"><div class="k">数据源</div><div class="v" id="src" style="font-size:13px">-</div></div>
<div class="item"><div class="k">当前信号</div><div class="v" id="signal">-</div></div>
<div class="item"><div class="k">模拟余额(USDT)</div><div class="v" id="balance">-</div></div>
<div class="item"><div class="k">持仓(BTC)</div><div class="v" id="units">-</div></div>
<div class="item"><div class="k">开仓价</div><div class="v" id="entry">-</div></div>
<div class="item"><div class="k">止损价</div><div class="v" id="stop">-</div></div>
<div class="item"><div class="k">最新收盘价</div><div class="v" id="last">-</div></div>
<div class="item"><div class="k">当前形态</div><div class="v" id="pattern" style="font-size:15px">-</div></div>
<div class="item"><div class="k">运行模式</div><div class="v" id="mode" style="font-size:13px">-</div></div>
</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:4px">形态解读 <span style="font-weight:normal;font-size:12px;color:#888">(每10分钟解读一次, 每条带时间, 最多留500条)</span> <span id="pat_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span></div><div id="pattern_msg" style="font-size:13px;color:#c9d1d9">启动后自动分析最近K线形态</div><div id="pattern_base" style="font-size:12px;color:#888;margin-top:4px"></div><div id="pattern_hist" style="margin-top:8px;max-height:160px;overflow-y:auto;font-size:12px;color:#8b949e;border-top:1px solid #30363d;padding-top:4px"></div></div>

<div class="card"><div style="font-weight:bold;margin-bottom:4px">决策记录 <span style="font-weight:normal;font-size:12px;color:#888">(每次信号判断都记录, 新的在上, 存档 decision_log.csv)</span> <span id="dec_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span></div>
<div id="dec_list" style="margin-top:6px;max-height:200px;overflow-y:auto;font-size:12px;color:#8b949e;border-top:1px solid #30363d;padding-top:4px">暂无记录, 点启动后每次信号判断都会记到这里</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:4px">每日盈亏汇报 <span style="font-weight:normal;font-size:12px;color:#888">(人民币按汇率折算, 每天自动存档 daily_report.csv)</span> <span id="daily_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span></div>
<div class="row" style="align-items:baseline">
<div><div class="k" style="color:#888;font-size:12px">今日盈亏(¥)</div><div id="dpnl_cny" style="font-size:20px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">今日盈亏(USDT)</div><div id="dpnl_usd" style="font-size:16px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">当前权益(¥)</div><div id="deq_cny" style="font-size:20px;font-weight:bold;margin-top:2px">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">当前汇率</div><div id="dfx" style="font-size:14px;font-weight:bold;margin-top:2px">-</div></div>
</div>
<svg id="daily_chart" viewBox="0 0 900 160" preserveAspectRatio="none" style="width:100%;height:130px;background:#0d1117;border:1px solid #30363d;border-radius:8px;margin-top:8px;display:block"></svg>
<div id="daily_table" style="margin-top:8px;font-size:12px;color:#8b949e"></div>
</div>
</div><div class="col b">

<div class="card"><div style="font-weight:bold;margin-bottom:6px">手动交易 <span style="font-weight:normal;font-size:12px;color:#888">(独立账户, 起始 100 USDT, 和自动盘分开记账; 手动没有跟踪止损, 全靠自己点卖出)</span></div>
<div class="row" style="align-items:center">
<div><div class="k" style="color:#888;font-size:12px">投入现金比例</div>
<input id="m_pct" type="number" value="13.3" min="1" max="100" step="0.1" style="width:80px;margin-top:3px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:5px;padding:4px"> <span style="font-size:12px;color:#888">%</span></div>
<div><button class="btn-start" onclick="manualOrder('buy')">买入</button></div>
<div><button class="btn-stop" onclick="manualOrder('sell')">卖出</button></div>
<div id="m_msg" style="font-size:13px;color:#8b949e"></div>
</div>
<div class="row" style="align-items:baseline;margin-top:6px">
<div><div class="k" style="color:#888;font-size:12px">手动持仓</div><div id="m_pos" style="font-size:16px;font-weight:bold">空仓</div></div>
<div><div class="k" style="color:#888;font-size:12px">手动净值</div><div id="m_eq" style="font-size:16px;font-weight:bold">100.00 USDT</div></div>
<div><div class="k" style="color:#888;font-size:12px">手动浮盈亏</div><div id="m_upnl" style="font-size:16px;font-weight:bold">-</div></div>
</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">实时建议 <span style="font-weight:normal;font-size:12px;color:#888">(每5秒刷新; 历史统计是"过去发生过什么", 不是预测)</span></div>
<div id="adv_verdict" style="font-size:15px;font-weight:bold;margin-bottom:8px">加载中...</div>

<div class="cols" style="gap:0 16px">
<div class="col">
<div style="font-weight:bold;font-size:13px;margin-bottom:4px;color:#58a6ff">如果现在买入</div>
<div id="adv_entry" style="font-size:13px;line-height:1.75;color:#c9d1d9"></div>
</div>
<div class="col">
<div style="font-weight:bold;font-size:13px;margin-bottom:4px;color:#f0883e">现在的情况</div>
<div id="adv_intel" style="font-size:13px;line-height:1.75;color:#c9d1d9"></div>
</div>
</div>

<div style="font-weight:bold;font-size:13px;margin:12px 0 4px;color:#3fb950">历史同类情况(未来涨跌分布)</div>
<table style="width:100%;border-collapse:collapse;font-size:13px">
<tr style="color:#888;font-size:12px"><th style="text-align:left;padding:3px 6px">当时的状态</th><th style="text-align:right;padding:3px 6px">未来24h中位</th><th style="text-align:right;padding:3px 6px">未来72h中位</th><th style="text-align:right;padding:3px 6px">未来168h中位</th><th style="text-align:right;padding:3px 6px">24h上涨概率</th><th style="text-align:right;padding:3px 6px">样本数</th></tr>
<tbody id="adv_hist"></tbody>
</table>
<div id="adv_hist_tip" style="font-size:12px;color:#8b949e;margin-top:5px"></div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">手动 vs 自动 <span style="font-weight:normal;font-size:12px;color:#888">(同一个起始资金、同一套手续费, 差别只在"什么时候买卖")</span></div>
<table style="width:100%;border-collapse:collapse;font-size:14px">
<tr style="color:#888;font-size:12px"><th style="text-align:left;padding:3px 6px">账户</th><th style="text-align:right;padding:3px 6px">净值(USDT)</th><th style="text-align:right;padding:3px 6px">收益率</th><th style="text-align:right;padding:3px 6px">已平仓</th><th style="text-align:right;padding:3px 6px">胜率</th><th style="text-align:right;padding:3px 6px">当前持仓</th></tr>
<tr><td style="padding:4px 6px">手动</td><td id="cmp_m_eq" style="text-align:right;padding:4px 6px">-</td><td id="cmp_m_ret" style="text-align:right;padding:4px 6px">-</td><td id="cmp_m_n" style="text-align:right;padding:4px 6px">-</td><td id="cmp_m_wr" style="text-align:right;padding:4px 6px">-</td><td id="cmp_m_hold" style="text-align:right;padding:4px 6px">-</td></tr>
<tr><td style="padding:4px 6px">自动-BTC</td><td id="cmp_a_eq" style="text-align:right;padding:4px 6px">-</td><td id="cmp_a_ret" style="text-align:right;padding:4px 6px">-</td><td id="cmp_a_n" style="text-align:right;padding:4px 6px">-</td><td id="cmp_a_wr" style="text-align:right;padding:4px 6px">-</td><td id="cmp_a_hold" style="text-align:right;padding:4px 6px">-</td></tr>
<tbody id="cmp_extra"></tbody>
</table>
<div id="cmp_tip" style="font-size:12px;color:#8b949e;margin-top:6px">手动账户还没开始, 点上面的买入即可开张。</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">数据延迟监测 <span style="font-weight:normal;font-size:12px;color:#888">(每次请求往返耗时, 精确到小数点后5位; 绿=好 黄=注意 红=异常)</span></div>
<div class="row" style="align-items:baseline">
<div><div class="k" style="color:#888;font-size:12px">价格延迟</div><div id="lag_px" style="font-size:20px;font-weight:bold">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">K线延迟</div><div id="lag_candle" style="font-size:20px;font-weight:bold">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">成交明细延迟</div><div id="lag_tr" style="font-size:20px;font-weight:bold">-</div></div>
<div><div class="k" style="color:#888;font-size:12px">本机时钟偏差</div><div id="lag_clock" style="font-size:20px;font-weight:bold">-</div></div>
</div>
<div id="lag_tip" style="font-size:12px;color:#8b949e;margin-top:5px">等待数据...</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">新闻雷达 <span style="font-weight:normal;font-size:12px;color:#888">(每10分钟扫描, 只影响开仓, 不自动卖出)</span> <span id="news_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span></div>
<div id="news_risk" style="font-size:20px;font-weight:bold">未知</div>
<div id="news_msg" style="font-size:12px;color:#8b949e;margin-top:2px">等待首次扫描...</div>
<div id="news_list" style="margin-top:8px;max-height:170px;overflow-y:auto;font-size:12px;color:#8b949e;border-top:1px solid #30363d;padding-top:4px"></div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">风险温度计 <label style="float:right;font-weight:normal;font-size:12px;color:#8b949e"><input type="checkbox" id="risk_sound" checked> 接近止损时声音提醒</label></div>
<div id="thermo_box" style="height:18px;border-radius:9px;background:#21262d;border:1px solid #30363d;overflow:hidden"><div id="thermo_fill" style="height:100%;width:0%;background:#30363d;transition:width .5s"></div></div>
<div id="thermo_txt" style="font-size:12px;color:#8b949e;margin-top:5px">空仓, 无风险敞口</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">最近走势与买卖点 <span style="color:#888;font-size:12px">(橙线=价格, 红点=买, 蓝点=卖, 彩色三角=形态)</span> <span id="chart_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span>
<span style="float:right;font-weight:normal;font-size:12px;color:#8b949e">范围
<select id="range_chart_sel" style="font-size:12px;padding:1px 4px">
<option value="24" selected>24小时</option>
<option value="72">3天</option>
<option value="168">7天</option>
<option value="720">30天</option>
<option value="0">全部</option>
</select> 本图尺度
<select id="scale_chart_sel" style="font-size:12px;padding:1px 4px">
<option value="30">30秒</option>
<option value="60">1分</option>
<option value="600">10分</option>
<option value="3600" selected>1小时</option>
</select></span></div>
<svg id="chart" viewBox="0 0 900 260" preserveAspectRatio="none" style="width:100%;height:240px;background:#0d1117;border:1px solid #30363d;border-radius:8px;display:block"></svg>
</div>
</div></div>

<div class="card">
<div class="row">
<div><label>OKX API Key(没有先不填)</label><input type="text" id="api_key" placeholder="先留空, 用虚拟模式"></div>
<div><label>Secret</label><input type="password" id="api_secret"></div>
<div><label>Passphrase</label><input type="password" id="api_passphrase"></div>
</div>
<div class="row">
<div><label>每笔风险 %(默认2)</label><input type="text" id="risk" value="2"></div>
<div><label>跟踪止损 %(默认8, 0=关)</label><input type="text" id="trail_pct" value="8"></div>
<div><label>止损 %(默认15, 作仓位尺子)</label><input type="text" id="stop_pct" value="15"></div>
<div><label>人民币汇率(默认7.2, 自动获取失败时使用)</label><input type="text" id="cny_rate" value="7.2"></div>
</div>
<div style="margin-top:8px;font-size:13px">
<label style="display:inline-block;margin-right:18px"><input type="checkbox" id="auto_dump"> 瀑布/砸盘自动离场(回测: 历史上无益, 默认关)</label>
<label style="display:inline-block"><input type="checkbox" id="auto_filter"> 盘整/拉盘时暂停开仓(默认关)</label>
<label style="display:inline-block"><input type="checkbox" id="news_gate" checked> 新闻面利空时暂停开仓(默认开)</label>
<label style="display:inline-block"><input type="checkbox" id="eth_enabled" checked> 同时跑 ETH 子账户(同一策略, 独立100U, 默认开)</label>
</div>
<div style="margin-top:12px">
<button class="btn-check" id="b_check" onclick="act('check')">① 检查连接</button>
<button class="btn-start" id="b_start" onclick="act('start')">② 保存并启动</button>
<button class="btn-stop" id="b_stop" onclick="act('stop')" disabled>③ 停止</button>
<button class="btn-quit" onclick="act('quit')">④ 退出程序</button>
<button class="btn-check" onclick="act('restart')">⑤ 重启程序</button>
</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">运行日志</div><div id="log">等待操作...</div></div>
<div class="hint">提示: ①先点[检查连接], 程序会自动选 Binance / OKX / Gate.io(国内直连); ②再点[保存并启动]。
没填密钥=本地虚拟成交(推荐先用); 填了密钥且OKX可达=模拟盘真实挂单(虚拟资金)。
密钥只保存在本机 paper_config_okx.json, 不会外传。历史成交明细在 paper_trades_okx.csv。</div>

<script>
function $(id){return document.getElementById(id)}
function fmtEq(v){ return Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}); }
function paintBook(prefix, b){
  $('cmp_'+prefix+'_eq').textContent = fmtEq(b.equity);
  const ret=Number(b.ret||0);
  const el=$('cmp_'+prefix+'_ret');
  el.textContent=(ret>=0?'+':'')+(ret*100).toFixed(2)+'%';
  el.style.color=ret>0?'#3fb950':(ret<0?'#f85149':'#e6edf3');
  $('cmp_'+prefix+'_n').textContent = b.n_trades ? b.n_trades : '0';
  $('cmp_'+prefix+'_wr').textContent = (b.win_rate===null||b.win_rate===undefined) ? '-' : (b.win_rate*100).toFixed(0)+'%';
  $('cmp_'+prefix+'_hold').textContent = Number(b.units)>0 ? (Number(b.units).toFixed(8)+' BTC') : '空仓';
}
function bookCells(b){
  const ret=Number(b.ret||0);
  const col=ret>0?'#3fb950':(ret<0?'#f85149':'#e6edf3');
  const wr=(b.win_rate===null||b.win_rate===undefined)?'-':(b.win_rate*100).toFixed(0)+'%';
  const hold=Number(b.units)>0?(Number(b.units).toFixed(6)+' 仓'):'空仓';
  return '<td style="text-align:right;padding:4px 6px">'+fmtEq(b.equity)+'</td>'
    +'<td style="text-align:right;padding:4px 6px;color:'+col+'">'+(ret>=0?'+':'')+(ret*100).toFixed(2)+'%</td>'
    +'<td style="text-align:right;padding:4px 6px">'+(b.n_trades||0)+'</td>'
    +'<td style="text-align:right;padding:4px 6px">'+wr+'</td>'
    +'<td style="text-align:right;padding:4px 6px">'+hold+'</td>';
}
function paintExtra(s){
  let h='';
  (s.sleeves||[]).forEach(b=>{
    const warn=(b.note&&b.note.indexOf('运行中')<0)?(' <span style="color:#f9a825;font-size:11px">'+b.note+'</span>'):'';
    h+='<tr><td style="padding:4px 6px">'+b.label
      +' <span style="color:#8b949e;font-size:11px">'+(b.source||'')+'</span>'+warn+'</td>'+bookCells(b)+'</tr>';
  });
  if(s.portfolio){
    h+='<tr style="border-top:1px solid #30363d"><td style="padding:4px 6px"><b>'+s.portfolio.label
      +'</b> <span style="color:#8b949e;font-size:11px">BTC+全部子账户等权</span></td>'+bookCells(s.portfolio)+'</tr>';
  }
  $('cmp_extra').innerHTML=h;
}
function paintManual(s){
  const m=s.manual||{}, a=s.auto||{};
  paintBook('m', m); paintBook('a', a); paintExtra(s);
  $('m_pos').textContent = Number(m.units)>0 ? ('持仓 '+Number(m.units).toFixed(8)+' BTC @ '+Number(m.entry).toFixed(2)) : '空仓';
  $('m_eq').textContent = fmtEq(m.equity)+' USDT';
  const u=Number(m.upnl||0);
  const eu=$('m_upnl');
  if(Number(m.units)>0){ eu.textContent=(u>=0?'+':'')+u.toFixed(2)+' USDT'; eu.style.color=u>=0?'#3fb950':'#f85149'; }
  else { eu.textContent='-'; eu.style.color='#e6edf3'; }
  const gap=(Number(m.ret||0)-Number(a.ret||0))*100;
  $('cmp_tip').textContent = (Number(m.n_trades)===0 && Number(a.n_trades)===0)
    ? '两边都还没平过仓, 现在比的是浮盈浮亏, 参考价值有限 —— 等各走完一两笔再看。'
    : ('手动 - 自动 = '+(gap>=0?'+':'')+gap.toFixed(2)+' 个百分点 (已平仓 '+m.n_trades+' vs '+a.n_trades+' 笔)');
}
function pctTxt(v,d){ return (v===null||v===undefined)?'-':((v>=0?'+':'')+(v*100).toFixed(d===undefined?2:d)+'%'); }
function histCells(b){
  // b: {"24":{n,mean,median,up}, "72":..., "168":...}
  const cell=(h)=>{ const x=b&&b[h]; return x? pctTxt(x.median) : '-'; };
  const n24=(b&&b['24'])?b['24'].n:0;
  let ns=[]; ['24','72','168'].forEach(h=>{ if(b&&b[h]) ns.push(b[h].n); });
  const up=(b&&b['24'])?pctTxt(b['24'].up,1):'-';
  const col=(h)=>{ const x=b&&b[h]; if(!x) return '#8b949e'; return x.median>0?'#3fb950':(x.median<0?'#f85149':'#e6edf3'); };
  return {v24:cell('24'),v72:cell('72'),v168:cell('168'),up:up,n:ns.length?Math.min.apply(null,ns):0,
          c24:col('24'),c72:col('72'),c168:col('168'),cup:(b&&b['24']&&b['24'].up>0.5)?'#3fb950':'#f85149'};
}
function histRow(name, b){
  const c=histCells(b);
  return '<tr><td style="padding:3px 6px">'+name+'</td>'
    +'<td style="text-align:right;padding:3px 6px;color:'+c.c24+'">'+c.v24+'</td>'
    +'<td style="text-align:right;padding:3px 6px;color:'+c.c72+'">'+c.v72+'</td>'
    +'<td style="text-align:right;padding:3px 6px;color:'+c.c168+'">'+c.v168+'</td>'
    +'<td style="text-align:right;padding:3px 6px;color:'+c.cup+'">'+c.up+'</td>'
    +'<td style="text-align:right;padding:3px 6px;color:'+(c.n<100?'#f9a825':'#8b949e')+'"'
      +(c.n<100?' title="样本偏少, 别当规律看"':'')+'>'+c.n+(c.n<100?' ⚠':'')+'</td></tr>';
}
function paintAdvice(s){
  const a=(s.advice||{});
  const v=$('adv_verdict');
  v.textContent=a.verdict|| '正在等第一根K线和历史统计...';
  v.style.color=a.verdict_color==='ok'?'#3fb950':(a.verdict_color==='warn'?'#f9a825':'#8b949e');
  if(!a.ready){ $('adv_entry').textContent='-'; $('adv_intel').textContent='-'; return; }

  // 用你填的仓位比例把"最大亏损"折算成钱
  const pct=(Number($('m_pct').value)||13.3)/100;
  const mEq=Number((s.manual||{}).equity||100);
  const e=a.entry;
  const riskUsd=mEq*pct*e.stop_pct;
  $('adv_entry').innerHTML=
     '现价 <b>'+Number(e.px).toFixed(2)+'</b><br>'
    +'建议止损(跟踪'+((e.stop_pct*100).toFixed(0))+'%) <b>'+Number(e.stop).toFixed(2)+'</b><br>'
    +'按你填的 '+(pct*100).toFixed(1)+'% 仓位: 最大亏损 <b style="color:#f85149">'+riskUsd.toFixed(2)+' USDT</b>'
      +' (账户的 '+(riskUsd/mEq*100).toFixed(2)+'%)<br>'
    +'手续费双边 0.20%, 保本要涨 <b>'+(e.breakeven*100).toFixed(2)+'%</b><br>'
    +'盈亏比 2:1 的目标价 <b style="color:#3fb950">'+Number(e.target2r).toFixed(2)+'</b>';

  const it=a.intel||{};
  const line=(k,val)=>'<div><span style="color:#8b949e">'+k+'</span> '+val+'</div>';
  let h='';
  h+=line('自动盘', a.auto||'-');
  h+=line('当前形态', (it.pattern||'无')+(it.vol_ratio?' (成交量 '+Number(it.vol_ratio).toFixed(2)+' 倍常态)':''));
  h+=line('波动率 ATR14', it.atr_pct!==null&&it.atr_pct!==undefined?((it.atr_pct*100).toFixed(2)+'%'):'-');
  h+=line('距60日线(1440)', pctTxt(it.vs1440));
  h+=line('48/336 均线差', pctTxt(it.ma_spread));
  h+=line('24小时涨跌', pctTxt(it.chg24h));
  h+=line('距72小时高点', pctTxt(it.dd72));
  h+=line('新闻风险', (it.news_risk||'未知'));
  if(a.exit){
    const x=a.exit;
    h+='<div style="margin-top:5px;color:'+(x.pnl>=0?'#3fb950':'#f85149')+'">你手上的仓位现在卖: '
      +((x.pnl>=0?'+':'')+Number(x.pnl).toFixed(2))+' USDT ('+pctTxt(x.pnl_pct)+')'
      +(x.held_h!==null&&x.held_h!==undefined?(' 已持有 '+Number(x.held_h).toFixed(1)+' 小时'):'')+'</div>';
  }
  $('adv_intel').innerHTML=h;

  let rows='';
  rows+=histRow('<b>任意时点买入(基准)</b>', a.base);
  if(a.trigger && a.trigger['24']) rows+=histRow('<b style="color:#58a6ff">策略金叉触发后</b>', a.trigger);
  (a.flags||[]).forEach(f=>{ rows+=histRow(f[1]+' 之后', f[2]); });
  $('adv_hist').innerHTML=rows;
  $('adv_hist_tip').textContent=(a.hist_msg||'')+' | 只显示样本数 ≥ '+a.min_n+' 次的情况; 中位数比平均值更抗极端值, 所以看中位数。';
}
async function manualOrder(side){
  const pct=(Number($('m_pct').value)||13.3)/100;
  const msg=$('m_msg');
  msg.textContent='提交中...'; msg.style.color='#8b949e';
  try{
    const r=await fetch('/api/manual',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({side:side,pct:pct})});
    const d=await r.json();
    msg.textContent=d.ok?(side==='buy'?'已买入':'已卖出'):(d.msg||'下单失败');
    msg.style.color=d.ok?'#3fb950':'#f85149';
    if(d.ok) setTimeout(()=>{ msg.textContent=''; },4000);
  }catch(e){ msg.textContent='请求失败, 看看控制台还在不在'; msg.style.color='#f85149'; }
  poll();
}
async function act(kind){
  const body = kind==='start' ? {
    api_key:$('api_key').value.trim(), api_secret:$('api_secret').value.trim(),
    api_passphrase:$('api_passphrase').value.trim(),
    risk_pct:$('risk').value, stop_pct:$('stop_pct').value,
    trail_pct:$('trail_pct').value,
    cny_rate:$('cny_rate').value,
    auto_dump:$('auto_dump').checked, auto_filter:$('auto_filter').checked, news_gate:$('news_gate').checked,
    eth_enabled:$('eth_enabled').checked
  } : {};
  try{
    const r = await fetch('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json();
    if(!j.ok){alert(j.msg||'操作失败'); return;}
    if(kind==='restart'){
      const _o=document.createElement('div');
      _o.style.cssText='position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(13,17,23,.88);color:#e6edf3;display:flex;align-items:center;justify-content:center;font-size:20px;z-index:99';
      _o.textContent='正在重启，几秒后自动重连…（窗口不用关）';
      document.body.appendChild(_o);
      setTimeout(function(){ _o.remove(); }, 8000);
    }
    if(kind==='quit'){
      document.body.innerHTML='<div style="font-family:Microsoft YaHei,SimHei,sans-serif;color:#e6edf3;background:#0d1117;height:100vh;margin:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:20px"><div>程序已关闭</div><div style="font-size:14px;color:#8b949e;margin-top:10px">可以关闭本页了</div></div>';
      try{ window.open('','_self',''); window.close(); }catch(e){}
      setTimeout(function(){ try{ window.close(); }catch(e){} }, 400);
    }
  }catch(e){alert('请求失败: '+e); $('offline').style.display='block';}
}
async function poll(){
  try{
    const r = await fetch('/api/status'); const s = await r.json();
    $('net').textContent = s.net_msg; $('net').className = 'v ' + (s.net==='ok'?'ok':s.net==='fail'?'fail':'gray');
    $('src').textContent = s.source;
    $('signal').textContent = s.signal===1 ? (Number(s.units)>0 ? '持仓(做多)' : '做多信号(下根K线买入)') : '空仓/无信号';
    $('signal').className = 'v ' + (s.signal===1?'ok':'gray');
    $('balance').textContent = Number(s.balance).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    $('units').textContent = Number(s.units).toFixed(8);
    $('entry').textContent = s.entry?Number(s.entry).toFixed(2):'-';
    $('stop').textContent = s.stop?Number(s.stop).toFixed(2):'-';
    $('last').textContent = s.last_close?Number(s.last_close).toFixed(2):'-';
    $('pattern').textContent = s.pattern||'-';
    $('pattern').className = 'v ' + (s.pattern && s.pattern!=='无' ? 'ok':'gray');
    $('pattern_msg').textContent = s.pattern_msg||'';
    $('pattern_base').textContent = s.pattern_base||'';
    $('pat_time').textContent = s.pattern_time ? ('解读于 '+s.pattern_time+(s.pattern_next?(' | 下轮 '+s.pattern_next):'')) : '等待首次解读(每10分钟)';
    $('daily_time').textContent = s.candle_time ? ('数据 '+s.candle_time) : '-';
    paintManual(s);
    paintAdvice(s);
    const clk0=Math.abs(Date.now()-Number(s.server_ts_ms||0));
    $('lag_clock').textContent = clk0>0 ? clk0.toFixed(0)+'毫秒' : '-';
    $('lag_clock').style.color = clk0<=2000?'#3fb950':(clk0<=10000?'#f9a825':'#f85149');
    const hist=(s.pattern_history||[]).slice().reverse().slice(0,40);
    $('pattern_hist').innerHTML = hist.map(h=>{
      const esc=v=>String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
      const m=h.msg||''; const short=m.length>44?m.slice(0,44)+'…':m;
      return '<div title="'+esc(m)+'"><b>'+esc(h.t)+'</b> '+esc(h.label)+'('+esc(h.strength)+') '+esc(short)+'</div>';
    }).join('');
    const decs=(s.decisions||[]);
    $('dec_time').textContent = s.decision_time ? ('更新于 '+s.decision_time) : '';
    const esc2=v=>String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const _dsig = decs.length+'|'+(decs[0]?decs[0].t+'|'+decs[0].reason:'');
    if($('dec_list').dataset.sig!==_dsig){
      $('dec_list').dataset.sig=_dsig;
      $('dec_list').innerHTML = decs.length ? decs.map(x=>{
        const col = x.action==='买'?'#3fb950':(x.action==='卖'?'#f85149':'#8b949e');
        return '<div style="padding:2px 0"><b>'+esc2(x.t)+'</b> <span style="color:'+col+';font-weight:bold">'+esc2(x.action)+'</span> '+esc2(x.reason)+' <span style="color:#6e7681">@'+esc2(x.px)+'</span></div>';
      }).join('') : '暂无记录, 点启动后每次信号判断都会记到这里';
    }
    renderDaily(s.daily, s.fx_rate);
    $('auto_dump').checked = !!s.auto_dump; $('auto_filter').checked = !!s.auto_filter; $('news_gate').checked = !!s.news_gate;
    $('eth_enabled').checked = (s.eth_enabled===undefined) ? true : !!s.eth_enabled;
    // 风险/止损/跟踪三个框: 只在首次加载时用"正在运行的配置"填一次。
    // 每次轮询都填会把用户正在输入的内容冲掉。
    if(!window._cfgFilled && s.risk_pct!==undefined){
      window._cfgFilled = true;
      const p=v=>String(+(v*100).toFixed(4));
      $('risk').value=p(s.risk_pct); $('stop_pct').value=p(s.stop_pct); $('trail_pct').value=p(s.trail_pct);
    }
    const nrisk=s.news_risk||'未知';
    $('news_risk').textContent = '新闻面风险: '+nrisk;
    $('news_risk').style.color = nrisk==='高'?'#f85149':(nrisk==='中'?'#f9a825':(nrisk==='低'?'#3fb950':'#8b949e'));
    $('news_time').textContent = s.news_time ? ('更新于 '+s.news_time) : '';
    $('news_msg').textContent = s.news_msg||'';
    const nitems=(s.news_items||[]);
    $('news_list').innerHTML = nitems.length ? nitems.map(x=>{
      const nc = x.tag==='利空'?'#f85149':(x.tag==='利好'?'#3fb950':'#8b949e');
      return '<div style="padding:2px 0"><span style="color:'+nc+'">['+esc2(x.tag)+']</span> '+esc2(x.title)+' <span style="color:#6e7681">'+esc2(x.src)+' '+esc2(x.age)+'</span></div>';
    }).join('') : '暂无新闻数据';
    $('mode').textContent = s.mode + (s.running?' (运行中)':' (未运行)');
    $('b_start').disabled = s.running; $('b_stop').disabled = !s.running;
    $('log').textContent = (s.logs||[]).join('\\n') || '等待操作...';
    $('log').scrollTop = $('log').scrollHeight;
    $('offline').style.display='none';
  }catch(e){$('offline').style.display='block';}
}
const PCOL={'瀑布':'#d62728','砸盘':'#e53935','拉盘':'#2ca02c','疑似洗盘':'#9467bd','反弹':'#ff7f0e','回调':'#1f77b4','盘整':'#999999','大资金异动(疑似)':'#17becf','疑似吸筹':'#bcbd22'};
function renderDaily(daily, fx){
  const c=v=>Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  $('dfx').textContent = fx?Number(fx).toFixed(4):'-';
  const rows=(daily&&daily.rows)?daily.rows:[];
  drawDailyChart(rows);
  if(!rows.length){
    $('daily_table').innerHTML='暂无记录, 启动后每天自动归档';
    $('dpnl_cny').textContent='-'; $('dpnl_usd').textContent='-'; $('deq_cny').textContent='-';
    return;
  }
  const t=rows[rows.length-1];
  const pn=Number(t.pnl_cny), pu=Number(t.pnl_usdt);
  $('dpnl_cny').textContent=(pn>=0?'+':'')+c(pn);
  $('dpnl_cny').style.color=pn>=0?'#3fb950':'#f85149';
  $('dpnl_usd').textContent=(pu>=0?'+':'')+c(pu);
  $('dpnl_usd').style.color=pu>=0?'#3fb950':'#f85149';
  $('deq_cny').textContent=c(Number(t.equity_cny));
  const show=rows.slice(-7).reverse();
  $('daily_table').innerHTML='<table style="border-collapse:collapse;width:100%"><tr style="color:#888"><td>日期</td><td>盈亏USDT</td><td>盈亏¥</td><td>权益¥</td><td>BTC价</td></tr>'
    +show.map(r=>{const p=Number(r.pnl_cny);return '<tr style="border-top:1px solid #30363d"><td>'+r.date+'</td><td>'+c(Number(r.pnl_usdt))+'</td><td style="color:'+(p>=0?'#3fb950':'#f85149')+'">'+(p>=0?'+':'')+c(p)+'</td><td>'+c(Number(r.equity_cny))+'</td><td>'+c(Number(r.btc_price))+'</td></tr>';}).join('')+'</table>';
}
function drawDailyChart(rows){
  const svg=$('daily_chart');
  if(!svg) return;
  if(!rows || !rows.length){
    svg.innerHTML='<text x="6" y="16" font-size="12" fill="#999">暂无净值数据</text>';
    return;
  }
  if(rows.length<2){
    svg.innerHTML='<text x="6" y="16" font-size="12" fill="#999">满2天数据后自动画净值曲线</text>';
    return;
  }
  const W=900,H=160,padL=8,padR=52,padT=10,padB=44;
  const eq=rows.map(r=>Number(r.equity_cny));
  const pn=rows.map(r=>Number(r.pnl_cny));
  let min=Math.min.apply(null,eq), max=Math.max.apply(null,eq);
  if(max-min<1){max+=1;min-=1;}
  const yw=H-padT-padB, xw=W-padL-padR;
  const X=i=>padL+xw*i/(rows.length-1);
  const Y=v=>padT+yw*(1-(v-min)/(max-min));
  let pts='';
  rows.forEach((r,i)=>{pts+=(i?' ':'')+X(i).toFixed(1)+','+Y(eq[i]).toFixed(1);});
  let html='<polyline points="'+pts+'" fill="none" stroke="#1f77b4" stroke-width="1.6"/>';
  for(let i=0;i<=3;i++){
    const v=min+(max-min)*i/3, y=Y(v);
    html+='<line x1="'+padL+'" y1="'+y.toFixed(1)+'" x2="'+(W-padR)+'" y2="'+y.toFixed(1)+'" stroke="#21262d"/>';
    html+='<text x="'+(W-padR+4)+'" y="'+(y+3).toFixed(1)+'" font-size="9" fill="#888">'+Math.round(v)+'</text>';
  }
  const pnMax=Math.max.apply(null,pn.map(Math.abs).concat([1]));
  const base=H-padB, barW=Math.max(3, xw/rows.length*0.7);
  rows.forEach((r,i)=>{
    const p=pn[i];
    if(!p) return;
    if(p>0){
      const h=Math.max(2,(p/pnMax)*(yw*0.25));
      html+='<rect x="'+(X(i)-barW/2).toFixed(1)+'" y="'+(base-h).toFixed(1)+'" width="'+barW.toFixed(1)+'" height="'+h.toFixed(1)+'" fill="#3fb950" opacity="0.6"/>';
    }else{
      const h=Math.max(2,(Math.abs(p)/pnMax)*(padB-6));
      html+='<rect x="'+(X(i)-barW/2).toFixed(1)+'" y="'+base.toFixed(1)+'" width="'+barW.toFixed(1)+'" height="'+h.toFixed(1)+'" fill="#f85149" opacity="0.6"/>';
    }
  });
  html+='<text x="'+padL+'" y="'+(H-4)+'" font-size="9" fill="#999">'+rows[0].date+' (蓝线=权益¥, 绿柱=当日赚, 红柱=当日亏)</text>';
  html+='<text x="'+(W-padR-72)+'" y="'+(H-4)+'" font-size="9" fill="#999">'+rows[rows.length-1].date+'</text>';
  svg.innerHTML=html;
}
let lastBeep=0;
function beep(){
  try{
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    const o=ac.createOscillator(), g=ac.createGain();
    o.type='sine'; o.frequency.value=880;
    g.gain.setValueAtTime(0.2, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime+0.4);
    o.connect(g); g.connect(ac.destination);
    o.start(); o.stop(ac.currentTime+0.4);
  }catch(e){}
}
function drawThermo(px, entry, stop, units){
  const fill=$('thermo_fill'), txt=$('thermo_txt'), box=$('thermo_box');
  if(!(Number(units)>0) || !(Number(entry)>0) || !(Number(stop)>0) || !(Number(px)>0)){
    fill.style.width='0%'; fill.style.background='#30363d';
    box.style.background='#21262d';
    txt.textContent='空仓, 无风险敞口';
    return;
  }
  const span=entry-stop;
  const dist=span>0?(px-stop)/span:1;
  const pct=Math.max(0, Math.min(1, dist))*100;
  fill.style.width=pct+'%';
  let col, zone;
  const leftPct=(px/stop-1)*100;
  if(dist<0.15){col='#f85149'; zone='危险: 离止损只剩 '+(leftPct>0?leftPct.toFixed(1):'0.0')+'%';}
  else if(dist<0.4){col='#ef6c00'; zone='警戒: 正在逼近止损线';}
  else if(dist<0.75){col='#f9a825'; zone='注意: 回撤较多, 别放松';}
  else{col='#3fb950'; zone='安全: 离止损还有 '+(leftPct>0?leftPct.toFixed(1):'0.0')+'%';}
  fill.style.background=col;
  box.style.background=col+'22';
  txt.textContent=zone+' | 开仓 '+Number(entry).toFixed(1)+' | 止损 '+Number(stop).toFixed(1)+' | 现价 '+Number(px).toFixed(1);
  if(dist<0.15 && $('risk_sound').checked){
    const now=Date.now();
    if(now-lastBeep>15000){lastBeep=now; beep();}
  }
}
function drawChart(bars, trades, pats, scale){
  const svg = $('chart');
  if(!bars || !bars.length){svg.innerHTML='<text x="6" y="16" font-size="12" fill="#999">该尺度暂无历史数据</text>'; return;}
  const W=900,H=260,padL=6,padR=6,padT=14,padB=20;
  let min=Infinity,max=-Infinity;
  for(const b of bars){if(b[1]<min)min=b[1]; if(b[1]>max)max=b[1];}
  if(!isFinite(min)){min=0;max=1;}
  const yw=H-padT-padB, xw=W-padL-padR;
  const X=i=>padL+xw*i/(bars.length-1||1);
  const Y=p=>padT+yw*(1-(p-min)/(max-min||1));
  let pts='';
  for(let i=0;i<bars.length;i++){pts+=(i?' ':'')+X(i).toFixed(1)+','+Y(bars[i][1]).toFixed(1);}
  let html='<polyline points="'+pts+'" fill="none" stroke="#ff7f0e" stroke-width="1.6"/>';
  const tms=bars.map(b=>b[0]);
  for(const p of pats||[]){
    if(p.t<tms[0]||p.t>tms[tms.length-1])continue;
    let idx=tms.findIndex(x=>x>p.t); if(idx<0)idx=bars.length-1;
    const col=PCOL[p.label]||'#888';
    const cx=X(idx).toFixed(1), cy=(Y(bars[idx][1])-12).toFixed(1);
    html+='<polygon points="'+cx+','+cy+' '+(Number(cx)-5).toFixed(1)+','+(Number(cy)-9).toFixed(1)+' '+(Number(cx)+5).toFixed(1)+','+(Number(cy)-9).toFixed(1)+'" fill="'+col+'">';
    html+='<title>'+p.label+'('+p.strength+')</title></polygon>';
  }
  for(const t of trades||[]){
    if(t.t<tms[0])continue;
    let idx=tms.findIndex(x=>x>t.t); if(idx<0)idx=bars.length-1;
    const col=(t.side==='BUY')?'#d62728':'#1f77b4';
    html+='<circle cx="'+X(idx).toFixed(1)+'" cy="'+Y(t.px).toFixed(1)+'" r="4" fill="'+col+'">';
    html+='<title>'+t.side+' '+t.time+' @ '+Number(t.px).toFixed(0)+'</title></circle>';
  }
  html+='<text x="2" y="'+(Y(max)+4).toFixed(1)+'" font-size="10" fill="#999">'+max.toFixed(0)+'</text>';
  html+='<text x="2" y="'+(Y(min)-2).toFixed(1)+'" font-size="10" fill="#999">'+min.toFixed(0)+'</text>';
  const fine=(Number(scale)||3600)<3600;
  const tt=ts=>fine?new Date(ts).toTimeString().slice(0,8):new Date(ts).toISOString().slice(0,10);
  html+='<text x="'+padL+'" y="'+(H-4)+'" font-size="10" fill="#999">'+tt(tms[0])+'</text>';
  html+='<text x="'+(W-padR-60)+'" y="'+(H-4)+'" font-size="10" fill="#999">'+tt(tms[tms.length-1])+'</text>';
  svg.innerHTML=html;
}
let lastPx = null;
function drawLive(trades, scale){
  const svg=$('livechart');
  if(!trades||trades.length<2){return;}
  const step=(Number(scale)||30)*1000;
  const buckets={};
  for(const x of trades){const b=Math.floor(x[0]/step)*step; buckets[b]=x[1];}
  let sel=Object.keys(buckets).sort((a,b)=>a-b).map(k=>[Number(k),buckets[k]]);
  if(sel.length>2000){sel=sel.slice(-2000);}
  if(sel.length<1){return;}
  const SNAME={30:'30秒',60:'1分',600:'10分',3600:'1小时'}[Number(scale)||30]||'30秒';
  if(sel.length===1){
    svg.innerHTML='<circle cx="450" cy="70" r="3" fill="#1f77b4"/>'
      +'<text x="462" y="74" font-size="10" fill="#8b949e">唯一成交桶 '+sel[0][1].toFixed(1)+'</text>'
      +'<text x="4" y="10" font-size="10" fill="#999">每笔成交价('+SNAME+'桶, 当前只有1个桶)</text>';
    return;
  }
  const W=900,H=140,padL=8,padR=52,padT=12,padB=14;
  const t0=sel[0][0], t1=sel[sel.length-1][0];
  const span=Math.max(t1-t0,60000);
  let min=Infinity,max=-Infinity;
  for(const x of sel){if(x[1]<min)min=x[1];if(x[1]>max)max=x[1];}
  const xw=W-padL-padR, yw=H-padT-padB;
  const X=t=>padL+xw*Math.min(Math.max((t-t0)/span,0),1);
  const Y=p=>padT+yw*(1-(p-min)/(max-min||1));
  let pts='';
  for(const x of sel){pts+=(pts?' ':'')+X(x[0]).toFixed(1)+','+Y(x[1]).toFixed(1);}
  let html='';
  const divs=4;
  for(let i=0;i<=divs;i++){
    const p=min+(max-min)*i/divs, y=Y(p);
    html+='<line x1="'+padL+'" y1="'+y.toFixed(1)+'" x2="'+(W-padR)+'" y2="'+y.toFixed(1)+'" stroke="#21262d" stroke-width="0.6"/>';
    html+='<text x="'+(W-padR+6)+'" y="'+(y+3).toFixed(1)+'" font-size="9" fill="#888">'+p.toFixed(1)+'</text>';
  }
  html+='<polyline points="'+pts+'" fill="none" stroke="#1f77b4" stroke-width="1.4"/>';
  const labelAll = sel.length<=60;
  for(let i=0;i<sel.length;i++){
    if(labelAll || i===sel.length-1){
      html+='<text x="'+X(sel[i][0]).toFixed(1)+'" y="'+(Y(sel[i][1])-3).toFixed(1)+'" font-size="8" fill="#8b949e" text-anchor="middle">'+sel[i][1].toFixed(1)+'</text>';
    }
  }
  const f=ts=>new Date(ts).toTimeString().slice(0,8);
  html+='<text x="4" y="10" font-size="10" fill="#999">每笔成交价('+SNAME+'桶,自动铺满,右侧=价格刻度'+(labelAll?'':',点数多仅标最新')+')</text>';
  html+='<text x="'+padL+'" y="'+(H-3)+'" font-size="9" fill="#999">'+f(t0)+'</text>';
  html+='<text x="'+(W-padR-58)+'" y="'+(H-3)+'" font-size="9" fill="#999">'+f(t1)+'</text>';
  svg.innerHTML=html;
}

function fmtHMS(ms){const d=new Date(ms);return d.toTimeString().slice(0,8);}
function setLat(el, sec, good, warn){
  if(!el) return;
  const s=Number(sec||0);
  if(!(s>0)){el.textContent='-';el.style.color='#8b949e';return;}
  el.textContent=s.toFixed(5)+'秒';
  el.style.color=s<=good?'#3fb950':(s<=warn?'#f9a825':'#f85149');
}
async function pollLive(){
  try{
    const r=await fetch('/api/live'); const d=await r.json();
    const px=Number(d.price||0);
    const el=$('live_px');
    if(px>0){
      el.textContent=px.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
      el.style.color = lastPx===null ? '#e6edf3' : (px>lastPx?'#3fb950':(px<lastPx?'#f85149':'#e6edf3'));
      lastPx=px;
    }
    const u=d.upnl;
    $('upnl').textContent = (u===null||u===undefined) ? '-' : ((u>=0?'+':'')+Number(u).toFixed(2)+' USDT');
    $('upnl').style.color = (u===null||u===undefined)?'#e6edf3':(u>=0?'#3fb950':'#f85149');
    const s=Math.max(0,Number(d.sec_to_next||0));
    $('countdown').textContent = Math.floor(s/60)+'分'+(s%60)+'秒';
    $('bar_prog').textContent = (d.bar_min||0)+'/60 分钟';
    drawLive(d.trades||[], liveScaleSec());
    drawThermo(px, d.entry, d.stop, d.units);
    const nowMs=Date.now();
    $('tick_time').textContent = d.price_ts ? ('价格 '+fmtHMS(d.price_ts)) : '-';
    // 三个框显示"这次请求实际花了多久"(往返耗时), 5位小数; 数据陈旧度放在下面提示行
    setLat($('lag_px'), d.lat_price, 0.5, 2);
    setLat($('lag_candle'), d.lat_kline, 2, 8);
    setLat($('lag_tr'), d.lat_trades, 0.5, 2);
    const clk1=Math.abs(nowMs-Number(d.server_ts_ms||0));
    $('lag_clock').textContent = clk1>0 ? clk1.toFixed(0)+'毫秒' : '-';
    $('lag_clock').style.color = clk1<=2000?'#3fb950':(clk1<=10000?'#f9a825':'#f85149');
    const tips=[];
    if(d.price_ts) tips.push('价格已收到 '+((nowMs-d.price_ts)/1000).toFixed(5)+'秒');
    if(d.trades_ts) tips.push('成交明细已收到 '+((nowMs-d.trades_ts)/1000).toFixed(5)+'秒');
    if(d.exch_ts>0) tips.push('交易所报价时间戳 '+fmtHMS(d.exch_ts));
    if(d.price_ts && nowMs-d.price_ts>5000) tips.push('价格已'+((nowMs-d.price_ts)/1000).toFixed(5)+'秒未更新, 疑似断网');
    $('lag_tip').textContent = tips.join(' | ') || '正常: 本机每秒直连行情源';
  }catch(e){}
}
const SMAP={30:'30s',60:'1m',600:'10m',3600:'1h'};
function liveScaleSec(){return Number($('scale_live_sel').value)||30;}
function chartScaleSec(){return Number($('scale_chart_sel').value)||3600;}
function chartRangeHours(){return Number($('range_chart_sel').value)||0;}
async function pollChart(){
  try{
    const r=await fetch('/api/chart?scale='+(SMAP[chartScaleSec()]||'1h')+'&hours='+chartRangeHours()); const c=await r.json();
    drawChart(c.bars,c.trades,c.pats, chartScaleSec());
    $('chart_time').textContent = c.candle_time ? ('数据 '+c.candle_time) : (c.ts_ms ? ('数据 '+fmtHMS(c.ts_ms)) : '-');
  }catch(e){}
}
poll(); setInterval(poll,2000);
pollChart(); setInterval(pollChart,15000);
pollLive(); setInterval(pollLive,1000);
$('scale_live_sel').addEventListener('change', pollLive);
$('scale_chart_sel').addEventListener('change', pollChart);
$('range_chart_sel').addEventListener('change', pollChart);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            st, logs = CORE.snapshot()
            self._json({"ok": True, **st, "logs": logs})
        elif path == "/api/chart":
            try:
                _q = parse_qs(urlparse(self.path).query)
                scale = _q.get("scale", ["1h"])[0]
                hours = int(_q.get("hours", ["24"])[0] or 24)
            except Exception:
                scale, hours = "1h", 24
            self._json(CORE.chart_data(scale, hours))
        elif path == "/api/live":
            self._json(CORE.live_snapshot())
        else:
            self._json({"ok": False, "msg": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if path == "/api/check":
            CORE.check_conn()
            self._json({"ok": True})
        elif path == "/api/start":
            try:
                risk = float(data.get("risk_pct", "2")) / 100.0
                stop = float(data.get("stop_pct", "15")) / 100.0
                trail = float(data.get("trail_pct", "8")) / 100.0
                if not (0 < risk <= 0.1 and 0.02 <= stop <= 0.6 and 0 <= trail <= 0.5):
                    raise ValueError("范围不对")
            except ValueError:
                self._json({"ok": False, "msg": "风险请填 0.1~10, 止损请填 2~60, 跟踪止损请填 0~50(0=关)"})
                return
            try:
                cny_rate = float(data.get("cny_rate", "7.2") or 7.2)
                if not (3 <= cny_rate <= 10):
                    cny_rate = 7.2
            except ValueError:
                cny_rate = 7.2
            cfg = {
                "api_key": str(data.get("api_key", "")).strip(),
                "api_secret": str(data.get("api_secret", "")).strip(),
                "api_passphrase": str(data.get("api_passphrase", "")).strip(),
                "simulated": True, "risk_pct": risk, "stop_pct": stop, "trail_pct": trail,
                "cny_rate": cny_rate, "poll_seconds": 60,
                "auto_dump": bool(data.get("auto_dump", False)),
                "auto_filter": bool(data.get("auto_filter", False)),
                "news_gate": bool(data.get("news_gate", True)),
                "eth_enabled": bool(data.get("eth_enabled", True)),
            }
            _atomic_write(CFG, cfg, indent=2)
            CORE.start(cfg)
            self._json({"ok": True})
        elif path == "/api/manual":
            side = str(data.get("side", "")).strip().lower()
            if side not in ("buy", "sell"):
                self._json({"ok": False, "msg": "side 只能是 buy 或 sell"})
                return
            ok, msg = CORE.manual_order(side, data.get("pct", 0.133))
            self._json({"ok": ok, "msg": msg})
        elif path == "/api/stop":
            CORE.stop()
            self._json({"ok": True})
        elif path == "/api/restart":
            restart_self(log=CORE.log)
            self._json({"ok": True, "msg": "正在重启, 页面会自动重连"})
        elif path == "/api/quit":
            def _shutdown():
                time.sleep(0.4)
                threading.Thread(target=self.server.shutdown, daemon=True).start()

            threading.Thread(target=_shutdown, daemon=True).start()
            self._json({"ok": True, "msg": "控制台即将关闭, 可以关闭本页面了"})
        else:
            self._json({"ok": False, "msg": "not found"}, 404)

    def log_message(self, *args):
        pass


class ExclusiveServer(ThreadingHTTPServer):
    """端口独占: 已有一个控制台在跑时, 新实例自动换下一个端口。"""
    allow_reuse_address = False


def main():
    port = PORT
    while port < PORT + 6:
        try:
            server = ExclusiveServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print("没有可用端口, 退出")
        return
    global _SERVER
    _SERVER = server
    CORE.start_history_stats()
    _url = f"http://127.0.0.1:{port}"
    print(f"控制台已启动: {_url}")
    if not os.environ.get("PM_NO_BROWSER"):
        # 优先用"应用模式窗口"打开: 这种窗口才允许页面在退出时自动关闭; 失败则回退默认浏览器
        _opened = False
        for _exe in ("msedge", "chrome",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
            try:
                subprocess.Popen([_exe, f"--app={_url}"], shell=False,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _opened = True
                break
            except Exception:
                continue
        if not _opened:
            webbrowser.open(_url)
    threading.Thread(target=watch_code, daemon=True).start()  # 代码一变就自动重启
    if os.environ.get("PM_RELAUNCH"):
        # 是"重启拉起"(非手动打开): 自动沿用上次配置直接开始运行, 无需再点[保存并启动]
        def _auto_start():
            time.sleep(2.0)
            try:
                cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
                CORE.log("检测到自动重启: 已沿用上次配置, 自动开始运行。")
                CORE.start(cfg)
            except Exception as e:
                CORE.log(f"自动恢复运行失败({type(e).__name__}): {e}, 请手动点[保存并启动]")
        threading.Thread(target=_auto_start, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("已退出")


if __name__ == "__main__":
    main()
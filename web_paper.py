# -*- coding: utf-8 -*-
"""傻瓜式模拟盘控制台(网页版, 多数据源 + 杂交趋势策略)

数据源自动选择: 先试 OKX, 连不上自动切 Gate.io(国内可直连, 无需VPN/密钥)。
双击"启动模拟盘.bat"后自动打开浏览器控制面板:
  [检查连接]   测试 OKX / Gate.io, 自动选择可用的数据源
  [保存并启动] 保存密钥与参数, 每小时检查信号、自动买卖、盯止损
  [停止]       停止循环(状态自动保存, 下次启动自动恢复)
  [退出程序]   关闭控制台程序
没填密钥也能跑(本地虚拟成交, 用于熟悉流程)。
"""
import csv as _csv
import io
import json
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
PORT = 8765

# 图上只标记这些重要形态(回调/盘整/吸筹太常见, 画上去会太乱)
SIGNIFICANT = {"瀑布", "砸盘", "拉盘", "疑似洗盘", "反弹", "大资金异动(疑似)"}

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


class Core:
    """交易核心: 状态 + 日志 + 后台循环。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.cache = None
        self.src = None
        self.live = None
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
            st["decisions"] = list(reversed(self.decision_rows[-40:]))
            st["fx_rate"] = self.fx_rate
            st["daily"] = {"rows": list(self.daily_rows[-60:])}
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

    # ---------------- 数据源 ----------------

    def _probe_source(self):
        try:
            okx_request("GET", "/api/v5/public/time")
            return "okx"
        except Exception:
            pass
        try:
            ds.gate_ticker()
            return "gate"
        except Exception:
            return None

    def _load_candles(self, src, initial):
        if src == "gate":
            return ds.gate_candles(2000 if initial else 50)
        return okx_fetch_candles(2000 if initial else 300)

    def _load_ticker(self, src):
        if src == "gate":
            return ds.gate_ticker()
        d = okx_request("GET", "/api/v5/market/ticker", query={"instId": INST})
        x = d["data"][0]
        out = {"last": float(x["last"])}
        if x.get("ts"):
            out["ts"] = int(x["ts"])
        return out

    # ---------------- 控制面板动作 ----------------

    def check_conn(self):
        def work():
            self.log("开始检查网络: 先试 OKX, 不通再试 Gate.io(最长约90秒)...")
            self.status["net"] = "unknown"
            src = self._probe_source()
            self.src = src
            self.ensure_live()
            if src == "okx":
                self.status.update({
                    "net": "ok", "source": "OKX",
                    "net_msg": "OKX 连接正常(可填密钥真实挂单)",
                })
                self.log("OKX 连接正常!")
            elif src == "gate":
                self.status.update({
                    "net": "ok", "source": "Gate.io(直连)",
                    "net_msg": "OKX不通, 已自动用 Gate.io 行情",
                })
                self.log("OKX 连不上, 已自动切换到 Gate.io(无需VPN/密钥, 本地虚拟成交)。")
            else:
                self.status.update({
                    "net": "fail", "source": "离线",
                    "net_msg": "都连不上, 请检查网络",
                })
                self.log("OKX 和 Gate.io 都连不上, 请检查网络后重试。")

        threading.Thread(target=work, daemon=True).start()

    def start(self, cfg):
        if self.worker and self.worker.is_alive():
            self.log("已经在运行中")
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.worker.start()

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

    def chart_data(self, scale="1h"):
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
        closed = cache[cache["close_time_ms"] <= now].tail(400)
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
                    if cur == "gate":
                        trades = ds.gate_recent_trades(1000)
                    else:
                        d = okx_request("GET", "/api/v5/market/trades", query={"instId": INST, "limit": 100})
                        trades = sorted([[int(x["ts"]), float(x["px"])] for x in d["data"]])
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
        risk = float(cfg.get("risk_pct", 0.01))
        stop = float(cfg.get("stop_pct", 0.15))
        poll = float(cfg.get("poll_seconds", 60))
        auto_dump = bool(cfg.get("auto_dump", False))
        auto_filter = bool(cfg.get("auto_filter", False))
        news_gate = bool(cfg.get("news_gate", True))
        st["auto_dump"], st["auto_filter"] = auto_dump, auto_filter
        st["news_gate"] = news_gate
        has_keys = bool(key and secret and passphrase)
        st["has_keys"] = has_keys
        cny_default = float(cfg.get("cny_rate", 7.2) or 7.2)
        fresh_rate = _fetch_cny_rate()
        self.fx_rate = fresh_rate if fresh_rate > 0 else cny_default
        self.log(f"人民币汇率: {self.fx_rate:.4f} {'(自动获取)' if fresh_rate > 0 else '(用面板填的汇率)'}")
        t = PaperTrader(cash=100.0, risk_pct=risk, stop_pct=stop, log_path=TRADES)
        if STATE.exists():
            try:
                with STATE.open(encoding="utf-8") as f:
                    prev = json.load(f)
                t.cash, t.units, t.entry_price, t.entry_equity = (
                    prev["cash"], prev["units"], prev["entry_price"], prev["entry_equity"]
                )
                t.stop_price = prev.get("stop_price", float("inf"))
                t.signal = prev.get("signal", 0.0)
                t.reentry_bar_ms = prev.get("reentry_bar_ms", 0)
                t.last_bar_ms = prev.get("last_bar_ms", 0)
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
            self.log("警告: OKX连不上, 密钥暂不可用, 自动改为本地虚拟成交(行情用Gate.io)。")
        st["source"] = "OKX" if src == "okx" else "Gate.io(直连)"
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
                sig = compute_signal(cache)
                valid_idx = np.flatnonzero(~np.isnan(sig))
                st["signal"] = float(sig[valid_idx[-1]]) if len(valid_idx) else 0.0
                st["last_close"] = float(closed.iloc[-1]["close"]) if len(closed) else 0.0
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
                            self.log(f"[{ts:%m-%d %H:%M}] 信号买入(虚拟) @ {bar['close']:.2f}, 止损 {t.stop_price:.2f}")
                            self._record_decision("买", f"做多信号确认, 虚拟买入 @{bar['close']:.2f}", bar["close"], 1)
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
                if t.units > 0:
                    try:
                        px = float(self._load_ticker(src)["last"])
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
                        if cur_sig == 1.0:
                            self._record_decision("不卖", "持仓中, 信号仍做多, 继续持有", last_px, cur_sig)
                        else:
                            self._record_decision("不卖", "信号已转空, 等本小时K线收盘确认后卖出", last_px, cur_sig)
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
                _atomic_write(STATE, {
                    "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                    "entry_equity": t.entry_equity, "stop_price": t.stop_price,
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
                if src == "okx" and fail_streak >= 3:
                    src = "gate"
                    self.src = src
                    fail_streak = 0
                    st["source"] = "Gate.io(直连)"
                    st["net_msg"] = "OKX不稳, 已自动切到 Gate.io(不用VPN)"
                    self.log("OKX 连续断线3次, 已自动切换到 Gate.io(国内直连)。状态不变, 继续运行; 想切回OKX请重启控制台。")
                elif fail_streak == 1 or fail_streak % 5 == 0:
                    self.log(f"网络抖动({type(e).__name__}), 本轮跳过, 每60秒自动重试(已连续失败{fail_streak}次)")
            for _ in range(int(poll)):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        st["running"] = False
        self.log("已停止。状态已保存, 下次启动自动恢复。")


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

<div class="card">
<div style="font-weight:bold;margin-bottom:6px">实时行情 <span style="color:#888;font-size:12px">(约1秒刷新, 行情自动选OKX或Gate.io)</span> <span id="tick_time" style="font-weight:normal;font-size:12px;color:#8b949e">--:--:--</span>
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
<svg id="livechart" viewBox="0 0 900 140" preserveAspectRatio="none" style="width:100%;height:120px;background:#0d1117;border:1px solid #30363d;border-radius:8px;margin-top:10px;display:block"></svg>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">数据延迟监测 <span style="font-weight:normal;font-size:12px;color:#888">(越小=越新鲜, 绿=好 黄=注意 红=异常)</span></div>
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
<span style="float:right;font-weight:normal;font-size:12px;color:#8b949e">本图尺度
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
<div><label>每笔风险 %(默认1)</label><input type="text" id="risk" value="1"></div>
<div><label>止损 %(默认15)</label><input type="text" id="stop_pct" value="15"></div>
<div><label>人民币汇率(默认7.2, 自动获取失败时使用)</label><input type="text" id="cny_rate" value="7.2"></div>
</div>
<div style="margin-top:8px;font-size:13px">
<label style="display:inline-block;margin-right:18px"><input type="checkbox" id="auto_dump"> 瀑布/砸盘自动离场(回测: 历史上无益, 默认关)</label>
<label style="display:inline-block"><input type="checkbox" id="auto_filter"> 盘整/拉盘时暂停开仓(默认关)</label>
<label style="display:inline-block"><input type="checkbox" id="news_gate" checked> 新闻面利空时暂停开仓(默认开)</label>
</div>
<div style="margin-top:12px">
<button class="btn-check" id="b_check" onclick="act('check')">① 检查连接</button>
<button class="btn-start" id="b_start" onclick="act('start')">② 保存并启动</button>
<button class="btn-stop" id="b_stop" onclick="act('stop')" disabled>③ 停止</button>
<button class="btn-quit" onclick="act('quit')">④ 退出程序</button>
</div>
</div>

<div class="card"><div style="font-weight:bold;margin-bottom:6px">运行日志</div><div id="log">等待操作...</div></div>
<div class="hint">提示: ①先点[检查连接], 程序会自动选 OKX 或 Gate.io(国内直连); ②再点[保存并启动]。
没填密钥=本地虚拟成交(推荐先用); 填了密钥且OKX可达=模拟盘真实挂单(虚拟资金)。
密钥只保存在本机 paper_config_okx.json, 不会外传。历史成交明细在 paper_trades_okx.csv。</div>

<script>
function $(id){return document.getElementById(id)}
async function act(kind){
  const body = kind==='start' ? {
    api_key:$('api_key').value.trim(), api_secret:$('api_secret').value.trim(),
    api_passphrase:$('api_passphrase').value.trim(),
    risk_pct:$('risk').value, stop_pct:$('stop_pct').value,
    cny_rate:$('cny_rate').value,
    auto_dump:$('auto_dump').checked, auto_filter:$('auto_filter').checked, news_gate:$('news_gate').checked
  } : {};
  try{
    const r = await fetch('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json();
    if(!j.ok){alert(j.msg||'操作失败')}
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
    setLag($('lag_candle'), Date.now()-Number(s.candle_ts_ms||0), 90, 300);
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
    $('dec_list').innerHTML = decs.length ? decs.map(x=>{
      const col = x.action==='买'?'#3fb950':(x.action==='卖'?'#f85149':'#8b949e');
      return '<div style="padding:2px 0"><b>'+esc2(x.t)+'</b> <span style="color:'+col+';font-weight:bold">'+esc2(x.action)+'</span> '+esc2(x.reason)+' <span style="color:#6e7681">@'+esc2(x.px)+'</span></div>';
    }).join('') : '暂无记录, 点启动后每次信号判断都会记到这里';
    renderDaily(s.daily, s.fx_rate);
    $('auto_dump').checked = !!s.auto_dump; $('auto_filter').checked = !!s.auto_filter; $('news_gate').checked = !!s.news_gate;
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
    if(t.t<tms[0]||t.t>tms[tms.length-1])continue;
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
function setLag(el, ms, good, warn){
  if(!el) return;
  if(!ms || ms<0){el.textContent='-';el.style.color='#8b949e';return;}
  const s=ms/1000;
  el.textContent=(s>=60?(s/60).toFixed(1)+'分':s.toFixed(1)+'秒');
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
    setLag($('lag_px'), nowMs-d.price_ts, 2, 10);
    setLag($('lag_tr'), nowMs-d.trades_ts, 2, 10);
    const clk1=Math.abs(nowMs-Number(d.server_ts_ms||0));
    $('lag_clock').textContent = clk1>0 ? clk1.toFixed(0)+'毫秒' : '-';
    $('lag_clock').style.color = clk1<=2000?'#3fb950':(clk1<=10000?'#f9a825':'#f85149');
    const tips=[];
    if(d.exch_ts>0) tips.push('交易所报价时间戳 '+fmtHMS(d.exch_ts));
    if(d.price_ts && nowMs-d.price_ts>5000) tips.push('价格已'+((nowMs-d.price_ts)/1000).toFixed(0)+'秒未更新, 疑似断网');
    $('lag_tip').textContent = tips.join(' | ') || '正常: 本机每秒直连行情源';
  }catch(e){}
}
const SMAP={30:'30s',60:'1m',600:'10m',3600:'1h'};
function liveScaleSec(){return Number($('scale_live_sel').value)||30;}
function chartScaleSec(){return Number($('scale_chart_sel').value)||3600;}
async function pollChart(){
  try{
    const r=await fetch('/api/chart?scale='+(SMAP[chartScaleSec()]||'1h')); const c=await r.json();
    drawChart(c.bars,c.trades,c.pats, chartScaleSec());
    $('chart_time').textContent = c.candle_time ? ('数据 '+c.candle_time) : (c.ts_ms ? ('数据 '+fmtHMS(c.ts_ms)) : '-');
  }catch(e){}
}
poll(); setInterval(poll,2000);
pollChart(); setInterval(pollChart,15000);
pollLive(); setInterval(pollLive,1000);
$('scale_live_sel').addEventListener('change', pollLive);
$('scale_chart_sel').addEventListener('change', pollChart);
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
                scale = parse_qs(urlparse(self.path).query).get("scale", ["1h"])[0]
            except Exception:
                scale = "1h"
            self._json(CORE.chart_data(scale))
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
                risk = float(data.get("risk_pct", "1")) / 100.0
                stop = float(data.get("stop_pct", "15")) / 100.0
                if not (0 < risk <= 0.1 and 0.02 <= stop <= 0.6):
                    raise ValueError("范围不对")
            except ValueError:
                self._json({"ok": False, "msg": "风险请填 0.1~10, 止损请填 2~60"})
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
                "simulated": True, "risk_pct": risk, "stop_pct": stop,
                "cny_rate": cny_rate, "poll_seconds": 60,
                "auto_dump": bool(data.get("auto_dump", False)),
                "auto_filter": bool(data.get("auto_filter", False)),
                "news_gate": bool(data.get("news_gate", True)),
            }
            _atomic_write(CFG, cfg, indent=2)
            CORE.start(cfg)
            self._json({"ok": True})
        elif path == "/api/stop":
            CORE.stop()
            self._json({"ok": True})
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
    print(f"控制台已启动: http://127.0.0.1:{port}")
    webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("已退出")


if __name__ == "__main__":
    main()
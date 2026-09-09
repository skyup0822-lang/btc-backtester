# -*- coding: utf-8 -*-
"""傻瓜式模拟盘窗口程序(OKX 模拟盘, 杂交趋势策略)

双击"启动模拟盘.bat"即可打开窗口:
  [检查连接]   测试本机到 OKX 的网络是否通畅(不用密钥)
  [保存配置]   把窗口里填的密钥和参数写进 paper_config_okx.json
  [启动模拟盘] 开始每小时检查信号、自动买卖、盯止损(填了密钥=模拟盘真实挂单, 没填=本地虚拟)
  [停止]       停止循环(状态自动保存, 下次启动自动恢复)
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

from paper_trader import PaperTrader, compute_signal
from paper_trader_okx import (
    INST, fetch_candles, lot_size, okx_market_order, okx_request, usdt_balance,
)

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "paper_config_okx.json"
STATE = ROOT / "paper_state_okx.json"
FONT = ("Microsoft YaHei", 11)
FONT_BIG = ("Microsoft YaHei", 13)
GREEN, RED, GRAY = "#1a7f37", "#c62828", "#555555"


def load_cfg():
    if not CFG.exists():
        return {"api_key": "", "api_secret": "", "api_passphrase": "",
                "risk_pct": 0.01, "stop_pct": 0.15, "poll_seconds": 60}
    with CFG.open(encoding="utf-8") as f:
        return json.load(f)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("比特币模拟盘控制台(杂交策略)")
        self.geometry("800x700")
        self.minsize(760, 640)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.log_q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.status = {
            "net": "unknown", "net_msg": "尚未检查",
            "signal": 0.0, "balance": 10000.0, "units": 0.0,
            "entry": 0.0, "stop": 0.0, "last_close": 0.0,
            "mode": "未启动", "running": False,
        }
        self._build()
        self.after(400, self._poll)

    # ---------- 界面 ----------

    def _build(self):
        head = ttk.Frame(self, padding=12)
        head.pack(fill="x")
        ttk.Label(head, text="比特币模拟盘控制台", font=("Microsoft YaHei", 16, "bold")).pack(anchor="w")
        ttk.Label(head, text="策略: SMA48/336 + 收盘>SMA1440 做多 | 15% 止损 | 每笔风险 1%",
                  foreground="#666666", font=FONT).pack(anchor="w")

        self.net_lbl = tk.Label(self, text="网络: 尚未检查", font=FONT_BIG, fg=GRAY, anchor="w", padx=12)
        self.net_lbl.pack(fill="x")
        self.sig_lbl = tk.Label(self, text="当前信号: -", font=FONT_BIG, fg=GRAY, anchor="w", padx=12)
        self.sig_lbl.pack(fill="x")

        info = ttk.Frame(self, padding=12)
        info.pack(fill="x")
        for r, (k, v) in enumerate([
            ("模拟余额(USDT)", "balance"), ("持仓数量(BTC)", "units"),
            ("开仓价", "entry"), ("止损价", "stop"), ("最新收盘价", "last_close"),
        ]):
            ttk.Label(info, text=k + ":", font=FONT).grid(row=r // 3, column=(r % 3) * 2, sticky="e", padx=(0, 4), pady=3)
            lbl = tk.Label(info, text="-", font=FONT_BIG, anchor="w", width=16)
            lbl.grid(row=r // 3, column=(r % 3) * 2 + 1, sticky="w", pady=3)
            setattr(self, f"lbl_{v}", lbl)

        keyf = ttk.LabelFrame(self, text=" OKX 模拟盘 API 密钥(没有就先不填, 照样能跑虚拟模式) ", padding=10)
        keyf.pack(fill="x", padx=12, pady=4)
        cfg = load_cfg()
        self.var_key = tk.StringVar(value=cfg.get("api_key", ""))
        self.var_secret = tk.StringVar(value=cfg.get("api_secret", ""))
        self.var_pass = tk.StringVar(value=cfg.get("api_passphrase", ""))
        for r, (name, var, show) in enumerate([
            ("API Key", self.var_key, None),
            ("Secret", self.var_secret, "*"),
            ("Passphrase", self.var_pass, "*"),
        ]):
            ttk.Label(keyf, text=name, font=FONT).grid(row=r, column=0, sticky="e", padx=(0, 6), pady=3)
            ttk.Entry(keyf, textvariable=var, width=64, show=show or "", font=FONT).grid(row=r, column=1, pady=3)

        parf = ttk.LabelFrame(self, text=" 参数(建议保持默认) ", padding=10)
        parf.pack(fill="x", padx=12, pady=4)
        self.var_risk = tk.StringVar(value=f"{cfg.get('risk_pct', 0.01) * 100:g}")
        self.var_stop = tk.StringVar(value=f"{cfg.get('stop_pct', 0.15) * 100:g}")
        ttk.Label(parf, text="每笔风险 %(默认1):", font=FONT).grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(parf, textvariable=self.var_risk, width=8, font=FONT).grid(row=0, column=1, sticky="w")
        ttk.Label(parf, text="止损 %(默认15):", font=FONT).grid(row=0, column=2, sticky="e", padx=(18, 6))
        ttk.Entry(parf, textvariable=self.var_stop, width=8, font=FONT).grid(row=0, column=3, sticky="w")

        btns = ttk.Frame(self, padding=12)
        btns.pack(fill="x")
        self.btn_check = tk.Button(btns, text="① 检查连接", font=FONT_BIG, width=14, bg="#e3f2fd",
                                   activebackground="#bbdefb", command=self.check_conn)
        self.btn_start = tk.Button(btns, text="③ 启动模拟盘", font=FONT_BIG, width=14, bg="#c8e6c9",
                                   activebackground="#a5d6a7", command=self.start_worker)
        self.btn_stop = tk.Button(btns, text="④ 停止", font=FONT_BIG, width=10, bg="#ffcdd2",
                                  activebackground="#ef9a9a", command=self.stop_worker, state="disabled")
        self.btn_save = tk.Button(btns, text="② 保存配置", font=FONT_BIG, width=12, command=self.save_cfg)
        self.btn_check.pack(side="left", padx=4)
        self.btn_save.pack(side="left", padx=4)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)

        logf = ttk.LabelFrame(self, text=" 运行日志 ", padding=8)
        logf.pack(fill="both", expand=True, padx=12, pady=(2, 8))
        self.logbox = tk.Text(logf, height=12, font=("Consolas", 10), wrap="word", state="disabled")
        self.logbox.pack(fill="both", expand=True)
        ttk.Label(self, text="提示: 点①测试网络; 连不上就关掉VPN再试。密钥只保存在本机文件, 不会外传。",
                  foreground="#666666", font=("Microsoft YaHei", 9)).pack(pady=(0, 6))

    def log(self, text):
        self.log_q.put(text)

    def _poll(self):
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            self.logbox.configure(state="normal")
            self.logbox.insert("end", line + "\n")
            self.logbox.see("end")
            self.logbox.configure(state="disabled")
        st = self.status
        net_map = {"unknown": ("网络: 尚未检查", GRAY), "ok": ("网络: OKX 连接正常", GREEN),
                   "fail": ("网络: 连不上 OKX, 请关闭VPN后点①重试", RED)}
        self.net_lbl.configure(text=net_map[st["net"]][0], fg=net_map[st["net"]][1])
        sig_txt = "当前信号: 持仓(做多)" if st["signal"] == 1 else "当前信号: 空仓"
        self.sig_lbl.configure(text=sig_txt, fg=GREEN if st["signal"] == 1 else GRAY)
        self.lbl_balance.configure(text=f"{st['balance']:,.2f}")
        self.lbl_units.configure(text=f"{st['units']:.8f}")
        self.lbl_entry.configure(text=f"{st['entry']:.2f}" if st["entry"] else "-")
        self.lbl_stop.configure(text=f"{st['stop']:.2f}" if st["stop"] else "-")
        self.lbl_last_close.configure(text=f"{st['last_close']:.2f}" if st["last_close"] else "-")
        if st["running"]:
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
        else:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        self.after(400, self._poll)

    # ---------- 动作 ----------

    def save_cfg(self):
        try:
            risk = float(self.var_risk.get()) / 100.0
            stop = float(self.var_stop.get()) / 100.0
            assert 0 < risk <= 0.1 and 0.02 <= stop <= 0.6
        except Exception:
            messagebox.showwarning("参数不对", "风险请填 0.1~10, 止损请填 2~60")
            return
        cfg = {
            "api_key": self.var_key.get().strip(),
            "api_secret": self.var_secret.get().strip(),
            "api_passphrase": self.var_pass.get().strip(),
            "simulated": True,
            "risk_pct": risk,
            "stop_pct": stop,
            "poll_seconds": 60,
        }
        with CFG.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self.log("配置已保存到 paper_config_okx.json")

    def check_conn(self):
        self.btn_check.configure(state="disabled")
        self.status["net"] = "unknown"
        self.status["net_msg"] = "检查中..."
        self.log("开始检查与 OKX 的连接(最长约20秒)...")

        def work():
            try:
                t = okx_request("GET", "/api/v5/public/time")
                df = fetch_candles(300)
                self.status.update({"net": "ok", "net_msg": f"服务器时间 {t['data'][0]['ts']}"})
                self.log(f"连接正常! 已取到 {len(df)} 根小时K线")
            except Exception as e:
                self.status.update({"net": "fail", "net_msg": str(e)[:80]})
                self.log(f"连接失败: {type(e).__name__}: {e}")
            finally:
                self.btn_check.configure(state="normal")

        threading.Thread(target=work, daemon=True).start()

    def start_worker(self):
        if self.worker and self.worker.is_alive():
            return
        self.save_cfg()
        cfg = load_cfg()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.worker.start()

    def stop_worker(self):
        self.stop_event.set()
        self.log("正在停止(最多60秒内生效)...")

    def on_close(self):
        self.stop_event.set()
        self.destroy()

    # ---------- 交易循环(后台线程) ----------

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
        has_keys = bool(key and secret and passphrase)
        lot = 0.00001
        t = PaperTrader(cash=10000.0, risk_pct=risk, stop_pct=stop,
                        log_path=ROOT / "paper_trades_okx.csv")
        if STATE.exists():
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
        if has_keys:
            try:
                lot = lot_size()
                t.cash = usdt_balance(key, secret, passphrase, simulated)
                self.log(f"模拟盘余额(虚拟资金): {t.cash:,.2f} USDT")
            except Exception as e:
                self.log(f"读取余额失败({e}), 使用本地记录")
        mode = "OKX模拟盘真实挂单" if has_keys else "DRY本地虚拟成交(未填密钥)"
        self.log(f"模式: {mode} | 风险 {risk:.1%} | 止损 {stop:.0%} | 品种 {INST}")
        try:
            cache = fetch_candles(2000)
        except Exception as e:
            self.log(f"启动失败, 拉不到行情: {e}. 请点①检查连接。")
            st["running"] = False
            return
        while not self.stop_event.is_set():
            try:
                now = int(time.time() * 1000)
                fresh = fetch_candles(300)
                cache = pd.concat([cache, fresh]).drop_duplicates("open_time_ms").sort_values("open_time_ms")
                if len(cache) > 3000:
                    cache = cache.iloc[-3000:].reset_index(drop=True)
                closed = cache[cache["close_time_ms"] <= now]
                sig = compute_signal(cache)
                st["signal"] = float(sig[len(closed) - 1]) if len(closed) else 0.0
                st["last_close"] = float(closed.iloc[-1]["close"]) if len(closed) else 0.0
                new_bars = closed[closed["close_time_ms"] > t.last_bar_ms]
                for _, bar in new_bars.iterrows():
                    pos = int((closed["close_time_ms"] <= bar["close_time_ms"]).sum()) - 1
                    s = float(sig[pos])
                    t.last_close = bar["close"]
                    t.last_bar_ms = int(bar["close_time_ms"])
                    ts = pd.to_datetime(bar["close_time_ms"], unit="ms", utc=True)
                    if t.units == 0 and s == 1 and t.signal == 0 and bar["close_time_ms"] > t.reentry_bar_ms:
                        t.signal = 1.0
                        if has_keys:
                            notional = t.cash * risk / stop
                            avg, fill, oid = okx_market_order("buy", f"{notional:.2f}", key, secret, passphrase, simulated)
                            t.cash = usdt_balance(key, secret, passphrase, simulated)
                            t.buy(avg, "signal", ts)
                            self.log(f"[{ts:%Y-%m-%d %H:%M}] 信号买入 {fill:.8f} BTC @ {avg:.2f}, 止损 {t.stop_price:.2f}")
                        else:
                            t.buy(bar["close"], "signal", ts)
                            self.log(f"[{ts:%Y-%m-%d %H:%M}] 信号买入(虚拟) @ {bar['close']:.2f}, 止损 {t.stop_price:.2f}")
                    elif t.units > 0 and s == 0:
                        t.signal = 0.0
                        if has_keys:
                            qty = np.floor(t.units / lot) * lot
                            okx_market_order("sell", f"{qty:.8f}", key, secret, passphrase, simulated)
                            t.cash = usdt_balance(key, secret, passphrase, simulated)
                            t.sell(bar["close"], "signal", ts)
                        else:
                            t.sell(bar["close"], "signal", ts)
                        self.log(f"[{ts:%Y-%m-%d %H:%M}] 信号卖出 @ {bar['close']:.2f}")
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
                        self.log(f"[{pd.to_datetime(now, unit='ms', utc=True):%Y-%m-%d %H:%M}] 止损卖出 @ {px:.2f}")
                st["balance"], st["units"], st["entry"], st["stop"] = t.cash, t.units, t.entry_price, t.stop_price
                with STATE.open("w", encoding="utf-8") as f:
                    json.dump({
                        "cash": t.cash, "units": t.units, "entry_price": t.entry_price,
                        "entry_equity": t.entry_equity, "stop_price": t.stop_price,
                        "signal": t.signal, "reentry_bar_ms": t.reentry_bar_ms,
                        "last_bar_ms": t.last_bar_ms,
                    }, f)
            except Exception as e:
                self.log(f"循环异常({type(e).__name__}): {e}")
            for _ in range(int(poll)):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        st["running"] = False
        self.log("已停止。状态已保存, 下次启动自动恢复。")


if __name__ == "__main__":
    app = App()
    if "--selftest" in sys.argv:
        app.after(1200, app.destroy)
        app.mainloop()
        print("GUI selftest OK")
    else:
        app.mainloop()
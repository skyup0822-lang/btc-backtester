"""生成交互式交易复盘看板(HTML)与信号预览图(PNG)。

- HTML: 样本外(2025-01-01 起)1h K 线 + 买卖标记(悬停可见理由/仓位/止损/新闻上下文)
         + 重大新闻事件标注 + 策略/买入持有净值曲线 + 交易明细表 + 最新信号面板
- PNG: 静态预览版(快速查看,交互请打开 HTML)

用法: python build_dashboard.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIB = ROOT / "lib"
if LIB.exists() and str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mplconfig"))

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.engine import run_backtest
from backtest.metrics import buy_hold_metrics
from config import DATA_FILE, DEFAULT_CONFIG
from strategies.trend_system import sma_gate
from news_events import events_df

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
STOP_PCT = 0.15
OUT_DIR = ROOT.parent / "outputs"
SEED = 42
TAG_COLOR = {"利好": "#00e676", "利空": "#ff5252", "中性": "#b0bec5"}


def pct(v):
    return "-" if pd.isna(v) else f"{v * 100:+.2f}%"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df = df.copy()
    for w in (48, 336, 1440):
        df[f"sma{w}"] = df["Close"].rolling(w).mean()
    df["rv720"] = df["Close"].pct_change().rolling(720).std() * np.sqrt(365 * 24)
    return df


def vol_pctile(df: pd.DataFrame, at: pd.Timestamp) -> float:
    series = df["rv720"].dropna()
    cur = df["rv720"].asof(at)
    if pd.isna(cur) or len(series) == 0:
        return np.nan
    return float((series <= cur).mean())


def news_near(events: pd.DataFrame, t: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = t - pd.Timedelta(days=days)
    hi = t + pd.Timedelta(days=days)
    return events[(events["time"] >= lo) & (events["time"] <= hi)]


def news_impact(events: pd.DataFrame, daily: pd.Series) -> list:
    out = []
    for _, e in events.iterrows():
        t = e["time"]
        before = daily[daily.index <= t]
        after3 = daily[daily.index >= t + pd.Timedelta(days=3)]
        after30 = daily[daily.index >= t + pd.Timedelta(days=30)]
        if len(before) == 0:
            continue
        base = float(before.iloc[-1])
        i3 = float(after3.iloc[0]) / base - 1 if len(after3) else np.nan
        i30 = float(after30.iloc[0]) / base - 1 if len(after30) else np.nan
        out.append((t, e["title"], e["desc"], e["tag"], i3, i30))
    return out


def buy_hover(t: pd.Timestamp, px: float, df: pd.DataFrame, events: pd.DataFrame) -> str:
    s48 = df["sma48"].asof(t)
    s336 = df["sma336"].asof(t)
    s1440 = df["sma1440"].asof(t)
    vp = vol_pctile(df, t)
    lines = [
        f"<b>买入信号</b>  {t:%Y-%m-%d %H:%M} UTC",
        f"成交价(含滑点): {px:,.0f} USD",
        "理由: SMA48 上穿 SMA336(中期转多),且收盘价 > SMA1440(60日线,长期多头闸门)",
        f"技术位: SMA48 {s48:,.0f} / SMA336 {s336:,.0f} / SMA1440 {s1440:,.0f}",
    ]
    if not pd.isna(vp):
        lines.append(f"波动环境: 30日波动率处于历史约 {vp:.0%} 分位")
    near = news_near(events, t, 7)
    if len(near):
        names = "、".join(near["title"].tolist())
        lines.append(f"事件窗口(±7天): {names} —— 建议降低仓位或等事件落地再进场")
    lines.append(f"仓位建议: 单笔风险 ≤ 总资金 1%,止损 -15% ⇒ 仓位 ≈ 6.7%(保守)或更低")
    lines.append(f"止损位: 入场价 x 0.85 = {px * 0.85:,.0f} USD")
    return "<br>".join(lines)


def sell_hover(t, df: pd.DataFrame, events: pd.DataFrame) -> str:
    reason = ("信号离场: SMA48 下穿 SMA336 或收盘跌破 60 日线(趋势转弱)"
              if t.exit_reason == "signal"
              else "止损离场: 价格触及 -15% 止损线(保护本金,截断大亏)")
    vp = vol_pctile(df, t.exit_time)
    lines = [
        f"<b>卖出</b>  {t.exit_time:%Y-%m-%d %H:%M} UTC",
        f"成交价(含滑点): {t.exit_price:,.0f} USD",
        f"理由: {reason}",
        f"本笔盈亏: {t.pnl_pct:+.2%} (持有 {t.holding_bars / 24:.1f} 天)",
    ]
    if not pd.isna(vp):
        lines.append(f"波动环境: 30日波动率处于历史约 {vp:.0%} 分位")
    near = news_near(events, t.exit_time, 7)
    if len(near):
        lines.append(f"事件窗口(±7天): {'、'.join(near['title'].tolist())}")
    lines.append("后续建议: 空仓等待下一次 SMA48 上穿 SMA336 且收盘站上 60 日线")
    return "<br>".join(lines)


def current_state(df: pd.DataFrame, signals: pd.Series, events: pd.DataFrame) -> tuple:
    """返回 (是否持仓, 最近入场价, 最新状态说明行列表)。"""
    t0 = df.index[-1]
    close = float(df["Close"].iloc[-1])
    s48 = float(df["sma48"].iloc[-1])
    s336 = float(df["sma336"].iloc[-1])
    s1440 = float(df["sma1440"].iloc[-1])
    vp = vol_pctile(df, t0)
    try:
        local = f"{t0.tz_convert('Asia/Shanghai'):%Y-%m-%d %H:%M}"
    except Exception:
        local = f"{t0 + pd.Timedelta(hours=8):%Y-%m-%d %H:%M}"
    lines = [f"数据截至: {t0:%Y-%m-%d %H:%M} UTC(北京时间 {local})",
             f"最新收盘: {close:,.0f} USD | SMA48 {s48:,.0f} | SMA336 {s336:,.0f} | SMA1440(60日线) {s1440:,.0f}"]
    if not pd.isna(vp):
        lines.append(f"波动环境: 30 日波动率处于历史约 {vp:.0%} 分位")
    up = (signals == 1) & (signals.shift(1) == 0)
    down = (signals == 0) & (signals.shift(1) == 1)
    last_up = up[up].index[-1] if up.any() else None
    last_down = down[down].index[-1] if down.any() else None
    in_pos = (last_up is not None) and (last_down is None or last_up > last_down)
    entry_px = np.nan
    entry_time = None
    if in_pos:
        i = df.index.get_loc(last_up) + 1
        if i < len(df):
            entry_px = float(df["Open"].iloc[i]) * (1 + DEFAULT_CONFIG.slippage)
            entry_time = df.index[i]
            stop = entry_px * (1 - STOP_PCT)
            fp = close / entry_px - 1 - DEFAULT_CONFIG.fee_rate - DEFAULT_CONFIG.slippage
            lines.append(f"当前状态: <b>持仓中</b>(最近买入信号 {last_up:%Y-%m-%d %H:%M} UTC,近似入场价 {entry_px:,.0f} USD)")
            lines.append(f"浮盈(近似): {fp:+.2%} | 止损位: {stop:,.0f} USD(-15%)")
            lines.append("建议: 继续持有;若收盘跌破 60 日线或 SMA48 下穿 SMA336 则离场")
        else:
            lines.append("当前状态: 刚发出买入信号,下一根 K 线开盘成交(近似)")
    else:
        lines.append("当前状态: <b>空仓</b>")
        lines.append("建议: 等待 SMA48 上穿 SMA336 且收盘价重新站上 SMA1440 再入场;信号出现前不交易")
    near = news_near(events, t0, 14)
    if len(near):
        lines.append("近 14 天已收录事件: " + "、".join(near["title"].tolist()))
    else:
        lines.append("近 14 天: 无已收录事件(2026 年新闻尚未收录,接入实时新闻源后可补齐)")
    lines.append("风险提醒: 本策略只做多、无杠杆;以上为程序化信号演示,不构成投资建议")
    return in_pos, entry_px, entry_time, lines


def oos_stats(df: pd.DataFrame, test: pd.DataFrame, equity: pd.Series, trades: list,
              bench_metrics: dict) -> tuple:
    eq_test = equity[equity.index >= TEST_START]
    ret = eq_test.iloc[-1] / eq_test.iloc[0] - 1
    dd = (eq_test / eq_test.cummax() - 1).min()
    oos = [t for t in trades if t.entry_time >= TEST_START]
    pnls = np.array([t.pnl_pct for t in oos], dtype=float)
    s = {"n": len(pnls), "ret": ret, "dd": dd,
         "bench_ret": bench_metrics["total_return"], "bench_dd": bench_metrics["max_drawdown"]}
    if len(pnls):
        s["exp"] = pnls.mean()
        s["win"] = (pnls > 0).mean()
        wins = pnls[pnls > 0].sum()
        losses = -pnls[pnls < 0].sum()
        s["pf"] = wins / losses if losses > 0 else np.inf
        s["worst"] = pnls.min()
        s["best"] = pnls.max()
        s["hold_days"] = float(np.mean([t.holding_bars for t in oos]) / 24)
        rng = np.random.default_rng(SEED)
        boot = np.array([rng.choice(pnls, len(pnls), replace=True).mean() for _ in range(5000)])
        s["ci_lo"], s["ci_hi"] = np.percentile(boot, [2.5, 97.5])
    else:
        for k in ("exp", "win", "pf", "worst", "best", "hold_days", "ci_lo", "ci_hi"):
            s[k] = np.nan
    return s, eq_test


def build_figure(df: pd.DataFrame, test: pd.DataFrame, eq_test: pd.Series, bench_eq: pd.Series,
                 oos_trades: list, events: pd.DataFrame, in_pos: bool, entry_px: float, entry_time=None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(
        x=test.index, open=test["Open"], high=test["High"], low=test["Low"], close=test["Close"],
        name="BTC/USDT 1h", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=test.index, y=test["sma48"], mode="lines", name="SMA48",
                             line=dict(color="#40c4ff", width=1), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=test.index, y=test["sma1440"], mode="lines", name="SMA1440(60日线)",
                             line=dict(color="#ffd54f", width=1.2), opacity=0.9), row=1, col=1)
    buy_x = [t.entry_time for t in oos_trades]
    buy_y = [t.entry_price for t in oos_trades]
    buy_text = [buy_hover(t.entry_time, t.entry_price, df, events) for t in oos_trades]
    sell_x = [t.exit_time for t in oos_trades]
    sell_y = [t.exit_price for t in oos_trades]
    sell_text = [sell_hover(t, df, events) for t in oos_trades]
    sell_colors = ["#ff5252" if t.exit_reason == "signal" else "#ff9800" for t in oos_trades]
    fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="买入",
                             marker=dict(symbol="triangle-up", size=11, color="#00e676",
                                         line=dict(color="black", width=1)),
                             hovertext=buy_text, hoverinfo="text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="卖出",
                             marker=dict(symbol="triangle-down", size=11, color=sell_colors,
                                         line=dict(color="black", width=1)),
                             hovertext=sell_text, hoverinfo="text"), row=1, col=1)
    link_x, link_y = [], []
    for t in oos_trades:
        link_x += [t.entry_time, t.exit_time, None]
        link_y += [t.entry_price, t.exit_price, None]
    fig.add_trace(go.Scatter(x=link_x, y=link_y, mode="lines", name="持仓区间",
                             line=dict(color="rgba(255,255,255,0.22)", width=1),
                             hoverinfo="skip"), row=1, col=1)
    if in_pos and not np.isnan(entry_px):
        last_entry = entry_time
        fig.add_trace(go.Scatter(x=[last_entry], y=[entry_px], mode="markers",
                                 name="当前持仓(未平仓)",
                                 marker=dict(symbol="triangle-up", size=14, color="#40c4ff",
                                             line=dict(color="black", width=1)),
                                 hovertext=f"当前仍在持仓中(尚未平仓,未计入统计)<br>入场价 {entry_px:,.0f} USD",
                                 hoverinfo="text"), row=1, col=1)
    for e in events.itertuples():
        if TEST_START <= e.time <= test.index[-1]:
            fig.add_vline(x=e.time, row=1, col=1, opacity=0.55,
                          line=dict(color=TAG_COLOR[e.tag], width=1, dash="dot"))
            fig.add_annotation(x=e.time, yref="y domain", y=0.99, row=1, col=1,
                               text=e.title, showarrow=False, xanchor="left",
                               font=dict(size=9, color=TAG_COLOR[e.tag]))
    fig.add_trace(go.Scatter(x=eq_test.index, y=eq_test / eq_test.iloc[0], name="趋势系统(策略)",
                             line=dict(color="#40c4ff", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq / bench_eq.iloc[0], name="买入持有(基准)",
                             line=dict(color="#b0bec5", width=1.4, dash="dash")), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=860,
        title=dict(text="BTC/USDT 趋势系统交易复盘(样本外 2025-01-01 起)· 悬停 ▲/▼ 查看每笔买卖建议与新闻上下文",
                   font=dict(size=15)),
        hovermode="closest", margin=dict(l=60, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.03, x=0))
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1, rangeselector=dict(buttons=[
        dict(count=1, label="1月", step="month", stepmode="backward"),
        dict(count=3, label="3月", step="month", stepmode="backward"),
        dict(count=6, label="6月", step="month", stepmode="backward"),
        dict(count=1, label="1年", step="year", stepmode="backward"),
        dict(step="all", label="全部")]))
    fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="净值(期初=1)", row=2, col=1)
    return fig


def card(label: str, value: str, cls: str = "neutral") -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v {cls}">{value}</div></div>'


def render_page(fig_html: str, s: dict, oos_trades: list, events: pd.DataFrame,
                impacts: list, state_lines: list, data_end: pd.Timestamp) -> str:
    cls_ret = "pos" if s["ret"] >= 0 else "neg"
    cls_exp = "pos" if (not pd.isna(s.get("exp", np.nan)) and s["exp"] >= 0) else "neg"
    cards = [
        card("策略区间收益", pct(s["ret"]), cls_ret),
        card("买入持有收益", pct(s["bench_ret"]), "pos" if s["bench_ret"] >= 0 else "neg"),
        card("单笔数学期望", pct(s.get("exp", np.nan)), cls_exp),
        card("期望 95% 置信区间", f"{pct(s.get('ci_lo', np.nan))} ~ {pct(s.get('ci_hi', np.nan))}"),
        card("胜率", f"{s.get('win', np.nan):.1%}" if not pd.isna(s.get("win", np.nan)) else "-"),
        card("盈亏比", f"{s.get('pf', np.nan):.2f}" if np.isfinite(s.get("pf", np.nan)) else "-"),
        card("策略最大回撤", pct(s["dd"]), "neg"),
        card("买入持有最大回撤", pct(s["bench_dd"]), "neg"),
        card("交易笔数", f"{s['n']:.0f}"),
        card("平均持仓", f"{s.get('hold_days', np.nan):.1f} 天" if not pd.isna(s.get("hold_days", np.nan)) else "-"),
        card("最好 / 最差单笔", f"{pct(s.get('best', np.nan))} / {pct(s.get('worst', np.nan))}"),
    ]
    rows = []
    for i, t in enumerate(oos_trades, 1):
        near = news_near(events, t.entry_time, 7)
        near2 = news_near(events, t.exit_time, 7)
        names = sorted(set(near["title"].tolist() + near2["title"].tolist()))
        news_cell = "、".join(names) if names else "—"
        cls = "win" if t.pnl_pct >= 0 else "loss"
        reason = ("止损 -15% 触发" if t.exit_reason == "stop" else "均线死叉/跌破 60 日线")
        advice = ("止损离场,保护本金;等待下次金叉" if t.exit_reason == "stop"
                  else "趋势转弱离场;空仓等待金叉+站上 60 日线")
        rows.append(
            f'<tr class="{cls}"><td>{i}</td>'
            f'<td>{t.entry_time:%Y-%m-%d %H:%M}</td><td>{t.entry_price:,.0f}</td>'
            f'<td>{t.exit_time:%Y-%m-%d %H:%M}</td><td>{t.exit_price:,.0f}</td>'
            f'<td class="{"pos" if t.pnl_pct >= 0 else "neg"}">{t.pnl_pct:+.2%}</td>'
            f'<td>{t.holding_bars / 24:.1f}</td><td>{reason}</td><td>{news_cell}</td><td>{advice}</td></tr>')
    news_rows = []
    for t, title, desc, tag, i3, i30 in impacts:
        news_rows.append(
            f'<tr><td>{t:%Y-%m-%d}</td><td>{title} <span class="badge b-good" style="visibility:hidden">x</span></td>'
            f'<td>{desc}</td><td class="{"pos" if tag == "利好" else ("neg" if tag == "利空" else "neutral")}">{tag}</td>'
            f'<td>{pct(i3)}</td><td>{pct(i30)}</td></tr>')
    state = "".join(f"<div>{l}</div>" for l in state_lines)
    notes = (
        "<li>回测假设: 手续费 0.10%/边、滑点 0.05%/边,信号在下一根 K 线开盘价成交(无未来函数),只做多、无杠杆。</li>"
        "<li>样本外区间 2025-01-01 ~ 2026-09-07 为下跌市(最高 126,200 / 最低 57,800),仅 52 笔交易,统计证据有限,"
        "期望为正不等于未来一定盈利。</li>"
        "<li>新闻清单为手工整理的 2020-2025 年重大公开事件,不完整、非实时;2026 年事件未收录。"
        "后续可接入实时新闻源(如加密新闻 API)自动补充。</li>"
        "<li>本页面仅用于策略研究与教学演示,不构成投资建议;实盘前必须经过更长时间样本外验证与模拟盘测试。</li>")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC 趋势系统交易复盘看板</title>
<style>
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#0e1117;color:#e6edf3;margin:0 auto;padding:24px;max-width:1420px}}
h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:16px;margin-top:30px;border-left:4px solid #40c4ff;padding-left:10px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-top:14px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px}}
.card .k{{font-size:12px;color:#8b949e}}.card .v{{font-size:19px;margin-top:6px;font-weight:600}}
.pos{{color:#00e676}}.neg{{color:#ff5252}}.neutral{{color:#e6edf3}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:12px}}
th,td{{border:1px solid #30363d;padding:6px 8px;text-align:left}}
th{{background:#1c2430;position:sticky;top:0;z-index:2}}
tr.win td{{background:rgba(0,230,118,0.05)}}tr.loss td{{background:rgba(255,82,82,0.06)}}
.scroll{{max-height:520px;overflow:auto;border:1px solid #30363d;border-radius:10px}}
.note{{color:#8b949e;font-size:12.5px;line-height:1.75}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px;line-height:1.95;font-size:13.5px}}
.chart{{margin-top:14px}}
</style></head><body>
<h1>BTC 趋势系统 · 交易复盘看板</h1>
<p class="note">样本外回测演示(2025-01-01 ~ {data_end:%Y-%m-%d},1h K线,趋势系统 48/336 + 60日闸门 + 15% 止损)·
 悬停图表上的 ▲/▼ 查看每笔买卖的理由、仓位与新闻上下文 · 非投资建议</p>
<div class="cards">{''.join(cards)}</div>
<div class="chart">{fig_html}</div>
<h2>最新信号与建议</h2><div class="panel">{state}</div>
<h2>交易明细(样本外,共 {s['n']:.0f} 笔)</h2>
<div class="scroll"><table><tr><th>#</th><th>买入时间(UTC)</th><th>买入价</th><th>卖出时间(UTC)</th><th>卖出价</th>
<th>盈亏</th><th>持仓(天)</th><th>离场原因</th><th>新闻上下文(±7天)</th><th>建议</th></tr>{''.join(rows)}</table></div>
<h2>重大新闻事件(手工整理 2020-2025,附事件后实际涨跌)</h2>
<div class="scroll" style="max-height:360px"><table><tr><th>日期</th><th>事件</th><th>说明</th><th>性质</th><th>事件后 3 日</th><th>事件后 30 日</th></tr>{''.join(news_rows)}</table></div>
<h2>重要说明</h2><div class="note"><ul>{''.join(notes)}</ul></div>
</body></html>"""


def make_png(df: pd.DataFrame, test: pd.DataFrame, oos_trades: list, events: pd.DataFrame,
             eq_test: pd.Series, bench_eq: pd.Series, in_pos: bool, entry_px: float, entry_time=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        fp = windir / "Fonts" / fn
        if fp.exists():
            try:
                font_manager.fontManager.addfont(str(fp))
                plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1]})
    ax1.plot(test.index, test["Close"], color="#1f77b4", linewidth=1.0, label="BTC/USDT 收盘价(1h)")
    ax1.plot(test.index, test["sma1440"], color="#e0a800", linewidth=1.0, label="SMA1440(60日线)")
    ax1.scatter([t.entry_time for t in oos_trades], [t.entry_price for t in oos_trades],
                marker="^", color="#00a651", s=46, zorder=5, label="买入信号")
    sig_x = [t.exit_time for t in oos_trades if t.exit_reason == "signal"]
    sig_y = [t.exit_price for t in oos_trades if t.exit_reason == "signal"]
    stp_x = [t.exit_time for t in oos_trades if t.exit_reason == "stop"]
    stp_y = [t.exit_price for t in oos_trades if t.exit_reason == "stop"]
    ax1.scatter(sig_x, sig_y, marker="v", color="#d62728", s=46, zorder=5, label="卖出(信号离场)")
    ax1.scatter(stp_x, stp_y, marker="v", color="#ff7f0e", s=46, zorder=5, label="卖出(止损离场)")
    if in_pos and not np.isnan(entry_px):
        last_entry = entry_time
        ax1.scatter([last_entry], [entry_px], marker="^", color="#00bcd4", s=70, zorder=6,
                    label="当前持仓(未平仓)")
    for e in events.itertuples():
        if TEST_START <= e.time <= test.index[-1]:
            ax1.axvline(e.time, color=TAG_COLOR[e.tag], linestyle=":", linewidth=1.0, alpha=0.9)
            ylim = ax1.get_ylim()
            ax1.annotate(e.title, (e.time, ylim[1]), rotation=90, fontsize=8,
                         color=TAG_COLOR[e.tag], ha="right", va="top")
    ax1.set_title("BTC/USDT 趋势系统交易信号复盘(样本外 2025-01-01 起,叠加重大新闻事件)")
    ax1.set_ylabel("价格 (USD)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.plot(eq_test.index, eq_test / eq_test.iloc[0], color="#40c4ff", linewidth=1.2, label="趋势系统(策略)")
    ax2.plot(bench_eq.index, bench_eq / bench_eq.iloc[0], color="#7f7f7f", linewidth=1.1,
             linestyle="--", label="买入持有(基准)")
    ax2.set_ylabel("净值(期初=1)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "btc_signals_preview.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    test = df[df.index >= TEST_START]
    events = events_df()
    signals = sma_gate(df, short=48, long=336, gate=1440)
    equity, trades, _ = run_backtest(df, signals, DEFAULT_CONFIG, stop_pct=STOP_PCT)
    bench_metrics, bench_eq = buy_hold_metrics(test, DEFAULT_CONFIG)
    s, eq_test = oos_stats(df, test, equity, trades, bench_metrics)
    oos_trades = [t for t in trades if t.entry_time >= TEST_START]
    in_pos, entry_px, entry_time, state_lines = current_state(df, signals, events)
    fig = build_figure(df, test, eq_test, bench_eq, oos_trades, events, in_pos, entry_px, entry_time)
    daily = df["Close"].resample("1D").last()
    impacts = news_impact(events, daily)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="inline")
    page = render_page(fig_html, s, oos_trades, events, impacts, state_lines, df.index[-1])
    html_path = OUT_DIR / "btc_trading_dashboard.html"
    html_path.write_text(page, encoding="utf-8")
    make_png(df, test, oos_trades, events, eq_test, bench_eq, in_pos, entry_px, entry_time)
    print(f"已生成: {html_path}")
    print(f"已生成: {OUT_DIR / 'btc_signals_preview.png'}")
    print()
    print(f"样本外统计: 交易 {s['n']:.0f} 笔 | 单笔期望 {pct(s.get('exp', np.nan))}"
          f" | 区间收益 {pct(s['ret'])} | 最大回撤 {pct(s['dd'])}"
          f" | 买入持有 {pct(s['bench_ret'])} / {pct(s['bench_dd'])}")
    print("当前系统状态: " + ("持仓中" if in_pos else "空仓"))


if __name__ == "__main__":
    main()
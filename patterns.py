# -*- coding: utf-8 -*-
"""形态识别器 —— 只描述"刚刚发生的事", 不预言未来。

所有指标只用当前及过去的数据(无未来函数), 回测和实盘逐根识别通用。
每个形态给出: 标签 / 强度(轻中强) / 白话解释。
"庄家入场"无法直接观测, 只能以"异常放量"作概率性代理, 一律标注"疑似"。
"""
import numpy as np
import pandas as pd

# ---- 阈值(可调) ----
RET_DUMP = 0.04        # 单根跌4%起算瀑布
RET_CRASH = 0.06       # 单根跌6%=强瀑布
SUM_DUMP = 0.10        # 6小时累计跌10%以上=强
RET_PUMP = 0.04        # 单根涨4%起算拉盘
RET_PUMP_STRONG = 0.07
VOL_MULT_DUMP = 2.0    # 量>=2倍中位=砸盘(放量下杀)
VOL_MULT_PUMP = 1.5    # 拉盘需要放量
VOL_MULT_WHALE = 3.0   # 量>=3倍中位+明显波动=大资金异动(疑似)
VOL_MULT_ACCUM = 2.5   # 量>=2.5倍但价格几乎不动=疑似吸筹
BOUNCE_DROP = 0.06     # 48小时内先跌过6%才谈反弹
BOUNCE_UP = 0.025      # 单根涨2.5%或连续2根上涨=反弹
PULLBACK_FROM_HIGH = 0.03   # 距14日高点回撤3%+
RANGE_ATR_MULT = 0.55  # 14小时ATR%低于其90根中位数的55%=盘整
RANGE_RET = 0.025      # 盘整要求单根波动<2.5%
SHAKE_DROP = 0.035     # 趋势内单根急跌3.5%=疑似洗盘
VOL_WINDOW = 20
ATR_WINDOW = 14
ATR_BASE = 90
BAND_WINDOW = 336      # 14日
HORIZONS = (24, 72, 168)


def compute_features(df):
    close = df["Close"]
    vol = df["Volume"]
    ret = close.pct_change()
    prev_close = close.shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW).mean()
    atr_pct = atr / close
    f = {
        "ret": ret,
        "vol_med": vol.rolling(VOL_WINDOW).median(),
        "vol_ratio": vol / vol.rolling(VOL_WINDOW).median(),
        "atr_pct": atr_pct,
        "atr_ref": atr_pct.rolling(ATR_BASE).median(),
        "sma48": close.rolling(48).mean(),
        "sma336": close.rolling(336).mean(),
        "sma1440": close.rolling(1440).mean(),
        "high336": df["High"].rolling(BAND_WINDOW).max(),
        "ret_6h": close.pct_change(6),
        "ret_48h": close.pct_change(48),
    }
    return f


def _normalize(df):
    """实盘缓存用小写列, 回测CSV用大写列; 统一成大写。"""
    if "High" in df.columns:
        return df
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"})


def detect(df):
    """逐根识别形态, 返回与 df 对齐的 DataFrame(布尔列 + 主标签/强度/白话)。"""
    df = _normalize(df)
    f = compute_features(df)
    ret, close = f["ret"], df["Close"]
    vol_ratio = f["vol_ratio"]
    idx = df.index

    # 瀑布: 单根大跌(放量2倍以上改叫砸盘)
    dump = (ret <= -RET_DUMP)
    dump_strong = (ret <= -RET_CRASH) | (f["ret_6h"] <= -SUM_DUMP)
    dump_vol = dump & (vol_ratio >= VOL_MULT_DUMP)
    # 拉盘: 放量上涨
    pump = (ret >= RET_PUMP) & (vol_ratio >= VOL_MULT_PUMP)
    pump_strong = pump & ((ret >= RET_PUMP_STRONG) | (vol_ratio >= VOL_MULT_WHALE))
    # 疑似洗盘: 趋势(48>336 且价在1440上方)内单根急跌, 未跌破1440
    uptrend = (f["sma48"] > f["sma336"]) & (close > f["sma1440"])
    shake = uptrend & (ret <= -SHAKE_DROP) & (close > f["sma1440"])
    # 反弹: 48h先跌过6%, 之后单根涨2.5%或连续2根涨, 且仍在48线下方(下跌中的反抽)
    dropped = f["ret_48h"] <= -BOUNCE_DROP
    up2 = (ret > 0) & (ret.shift(1) > 0)
    bounce = dropped & ((ret >= BOUNCE_UP) | up2) & (close < f["sma48"])
    # 回调: 上升趋势中, 距14日高点回撤3%+, 未跌破1440, 单根未暴跌
    pullback = (
        uptrend & (close < f["high336"] * (1 - PULLBACK_FROM_HIGH))
        & (close > f["sma1440"]) & (ret > -RET_DUMP)
    )
    # 盘整: 波动率收缩 + 单根波动小
    range_ = (
        (f["atr_pct"] < RANGE_ATR_MULT * f["atr_ref"])
        & (ret.abs() < RANGE_RET)
    )
    # 大资金异动(疑似): 3倍量 + 明显波动(若同时是涨/跌, 归入拉盘/砸盘)
    whale = (vol_ratio >= VOL_MULT_WHALE) & (ret.abs() >= 0.025) & ~(pump | dump)
    # 疑似吸筹: 2.5倍量但价格几乎不动
    accum = (vol_ratio >= VOL_MULT_ACCUM) & (ret.abs() < 0.015) & ~range_

    flags = pd.DataFrame({
        "dump": dump, "dump_strong": dump_strong, "dump_vol": dump_vol,
        "pump": pump, "pump_strong": pump_strong, "shake": shake,
        "bounce": bounce, "pullback": pullback, "range": range_,
        "whale": whale, "accum": accum,
    }, index=idx)

    def strength(row):
        if row["dump"]:
            return "强" if row["dump_strong"] else "中"
        if row["pump"]:
            return "强" if row["pump_strong"] else "中"
        return "轻"

    def label(row):
        if row["dump"]:
            return "砸盘" if row["dump_vol"] else "瀑布"
        if row["pump"]:
            return "拉盘"
        if row["shake"]:
            return "疑似洗盘"
        if row["bounce"]:
            return "反弹"
        if row["pullback"]:
            return "回调"
        if row["accum"]:
            return "疑似吸筹"
        if row["whale"]:
            return "大资金异动(疑似)"
        if row["range"]:
            return "盘整"
        return "无"

    out = flags.copy()
    out["strength"] = flags.apply(strength, axis=1)
    out["label"] = flags.apply(label, axis=1)

    def msg(row):
        if row["label"] == "无":
            return "无明显形态"
        r = row["ret"]
        vr = row["vol_ratio"]
        if row["dump"]:
            s = f"1小时急跌{r:.1%}"
            if vr >= VOL_MULT_DUMP:
                s += f", 成交量是平时{vr:.1f}倍(放量下杀)"
            if row["dump_strong"]:
                s += ", 6小时累计跌幅大, 风险高"
            return s
        if row["pump"]:
            s = f"1小时急涨{r:.1%}, 成交量是平时{vr:.1f}倍"
            if row["pump_strong"]:
                s += "(放量猛拉, 谨防追高)"
            return s
        if row["shake"]:
            return f"上升趋势内突然急跌{r:.1%}, 可能是洗盘也可能是趋势反转, 看后续能否收回"
        if row["bounce"]:
            return f"前期大跌后出现反弹, 涨{r:.1%}, 属于下跌中的反抽还是见底需继续观察"
        if row["pullback"]:
            return f"上升趋势中回调, 距近期高点回落{abs(row['dist_high']):.1%}, 趋势未破"
        if row["accum"]:
            return f"成交量是平时{vr:.1f}倍但价格几乎不动, 疑似有大资金悄悄换手"
        if row["whale"]:
            return f"成交量是平时{vr:.1f}倍且价格明显波动, 疑似大资金进场(无法确认)"
        if row["range"]:
            return "波动明显收缩, 处于盘整, 方向不明"
        return ""

    msg_input = pd.DataFrame({
        "ret": ret, "vol_ratio": vol_ratio, "label": out["label"],
        "dump": dump, "dump_vol": dump_vol, "dump_strong": dump_strong,
        "pump": pump, "pump_strong": pump_strong, "shake": shake,
        "bounce": bounce, "pullback": pullback,
        "accum": accum, "whale": whale, "range": range_,
        "dist_high": close / f["high336"] - 1,
    }, index=idx)
    out["msg"] = msg_input.apply(msg, axis=1)
    return out


def forward_stats(df, horizons=HORIZONS):
    """离线统计: 每种形态出现后 24/72/168 小时的涨跌分布(用了未来数据, 仅回测报告用)。"""
    df = _normalize(df)
    pat = detect(df)
    close = df["Close"]
    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1
        # 无条件基准
        rows.append({
            "形态": "全部时间(基准)", "未来小时": h, "次数": int(fwd.notna().sum()),
            "平均涨跌": float(fwd.mean()), "中位涨跌": float(fwd.median()),
            "上涨概率": float((fwd > 0).mean()), "涨5%概率": float((fwd > 0.05).mean()),
        })
        for name in ["dump", "pump", "bounce", "pullback", "range", "shake", "whale", "accum"]:
            sel = fwd[pat[name].fillna(False) & fwd.notna()]
            if len(sel) < 5:
                continue
            rows.append({
                "形态": {"dump": "瀑布/砸盘", "pump": "拉盘", "bounce": "反弹",
                         "pullback": "回调", "range": "盘整", "shake": "疑似洗盘",
                         "whale": "大资金异动", "accum": "疑似吸筹"}[name],
                "未来小时": h, "次数": len(sel),
                "平均涨跌": float(sel.mean()), "中位涨跌": float(sel.median()),
                "上涨概率": float((sel > 0).mean()), "涨5%概率": float((sel > 0.05).mean()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/btc_usdt_1h.csv", index_col=0, parse_dates=True)
    pat = detect(df)
    print("各形态出现次数(全历史):")
    for name in ["dump", "pump", "shake", "bounce", "pullback", "range", "whale", "accum"]:
        print(f"  {name}: {int(pat[name].sum())}")
    print("\n最近5根K线的识别结果:")
    print(pat[["label", "strength", "msg"]].tail(5).to_string())
    print("\n形态出现后的历史统计(未来涨跌):")
    print(forward_stats(df).to_string(index=False))
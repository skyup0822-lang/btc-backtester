"""下载 BTC/USDT 1 小时 K 线历史数据并保存为 CSV(无需 API Key)。

数据源优先级(自动探测可用源):
1. data.binance.vision 官方数据存档(月度 zip + 当月按日补齐)
2. Binance 公开行情接口(api.binance.com/api/v3/klines)
3. OKX 公开行情接口
4. Gate.io 公开行情接口

用法: python fetch_data.py
"""
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_FILE

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START = datetime(2019, 11, 1, tzinfo=timezone.utc)
START_TS = int(START.timestamp() * 1000)

BINANCE_COLUMNS = [
    "open_time", "Open", "High", "Low", "Close", "Volume",
    "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index.name = "Open time"
    return df


def _open_time_ms(series: pd.Series) -> pd.Series:
    """不同月份存档的时间戳单位可能不同(ms/us),按数值量级统一为毫秒。"""
    vals = series.to_numpy(dtype="int64")
    m = int(np.abs(vals).max())
    if m > 1e17:
        return vals // 1_000_000
    if m > 1e14:
        return vals // 1_000
    return vals

def fetch_binance_vision() -> pd.DataFrame:
    """data.binance.vision 官方存档:月度 zip + 当月按日 zip 补齐。"""
    now = pd.Timestamp.now(tz="UTC")
    month_first = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    months = pd.date_range(
        pd.Timestamp("2019-11-01", tz="UTC"),
        month_first,
        freq="MS",
    )
    frames = []
    monthly_base = f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/"
    for month in months:
        name = f"{SYMBOL}-{INTERVAL}-{month:%Y-%m}.zip"
        try:
            raw = _get(monthly_base + name)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                inner = zf.namelist()[0]
                frame = pd.read_csv(io.BytesIO(zf.read(inner)), header=None, names=BINANCE_COLUMNS)
            frame["open_time"] = _open_time_ms(frame["open_time"])
            frames.append(frame)
        except (urllib.error.HTTPError, urllib.error.URLError, zipfile.BadZipFile):
            continue  # 个别月份缺失时跳过
        time.sleep(0.05)
    daily_base = f"https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/{INTERVAL}/"
    for day in pd.date_range(month_first, now.normalize(), freq="D"):
        name = f"{SYMBOL}-{INTERVAL}-{day:%Y-%m-%d}.zip"
        try:
            raw = _get(daily_base + name)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                inner = zf.namelist()[0]
                frame = pd.read_csv(io.BytesIO(zf.read(inner)), header=None, names=BINANCE_COLUMNS)
            frame["open_time"] = _open_time_ms(frame["open_time"])
            frames.append(frame)
        except (urllib.error.HTTPError, urllib.error.URLError, zipfile.BadZipFile):
            continue
        time.sleep(0.05)
    if not frames:
        raise RuntimeError("Binance Vision 未获取到任何数据")
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return _clean(df)


def fetch_binance_api() -> pd.DataFrame:
    """Binance /api/v3/klines,每次最多 1000 根,循环翻页。"""
    rows, start = [], START_TS
    end_ts = int(time.time() * 1000)
    while start < end_ts:
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}"
            f"&interval={INTERVAL}&startTime={start}&limit=1000"
        )
        batch = json.loads(_get(url))
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 1
        time.sleep(0.15)
    if not rows:
        raise RuntimeError("Binance API 返回空数据")
    df = pd.DataFrame(rows, columns=BINANCE_COLUMNS)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return _clean(df)


def fetch_okx() -> pd.DataFrame:
    """OKX 历史 K 线(备选,覆盖最近约 2 年)。"""
    rows, after = [], ""
    url_base = "https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT&bar=1H&limit=100"
    for _ in range(200):
        url = url_base + (f"&after={after}" if after else "")
        candles = json.loads(_get(url)).get("data", [])
        if not candles:
            break
        rows.extend(candles)
        after = candles[-1][0]
        if int(after) <= START_TS:
            break
        time.sleep(0.1)
    if not rows:
        raise RuntimeError("OKX 返回空数据")
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "vol", "volCcy", "volCcyQuote", "confirm"])
    df.index = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    return _clean(df)


def fetch_gateio() -> pd.DataFrame:
    """Gate.io 现货 K 线(备选),每次 1000 根,用 to 参数向前翻页。"""
    rows, to = [], int(time.time())
    url_base = f"https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair=BTC_USDT&interval=1h&limit=1000"
    start_s = START_TS // 1000
    for _ in range(100):
        url = url_base + (f"&to={to}" if to else "")
        batch = json.loads(_get(url))
        if not batch:
            break
        rows.extend(batch)
        to = int(batch[0][0]) - 1
        if to <= start_s:
            break
        time.sleep(0.1)
    if not rows:
        raise RuntimeError("Gate.io 返回空数据")
    df = pd.DataFrame(rows, columns=["ts", "quote_volume", "Close", "High", "Low", "Open", "base_volume", "ignore"])
    df.index = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    return _clean(df)


def main() -> None:
    sources = [
        ("Binance Vision", fetch_binance_vision),
        ("Binance API", fetch_binance_api),
        ("OKX", fetch_okx),
        ("Gate.io", fetch_gateio),
    ]
    df = None
    for name, fn in sources:
        try:
            print(f"尝试数据源: {name} ...")
            df = fn()
            print(f"  成功,共 {len(df)} 根 K 线")
            break
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            print(f"  失败: {exc}")
    if df is None:
        raise SystemExit("所有数据源均失败,请检查网络后重试")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE)
    print(f"已保存: {DATA_FILE}")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
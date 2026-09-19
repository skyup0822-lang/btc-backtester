# -*- coding: utf-8 -*-
"""抓多币种日线, 建一个「没有幸存者偏差」的币池。

核心原则: 币池只能用 2021-09-09 当天能拿到的信息来选。
  1. 从 data.binance.vision 的月度归档列表, 拿到「2021年9月在币安有USDT现货对」的全部币
     —— 这个列表里天然包含后来归零的 LUNA/LUNC、FTT、SRM 等, 因为它们是当时真实存在的
  2. 用 2021-08-10 ~ 2021-09-09 这 30 天的成交额给它们排名(和 C+ 用的筛选规则一致)
  3. 取前 N 名作为固定币池 —— 这一步只看当天之前的数据, 不看谁后来活下来了
  4. 抓这 N 个币从 2021-01 到今天的完整日线

抓下来的数据缓存到 work/multi_data/, 重跑不会重复下载。
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "multi_data"
OUT.mkdir(parents=True, exist_ok=True)

START = "2021-09-09"
RANK_DAYS = 30
TOP_N = int(os.environ.get("TOP_N", "25"))

# 稳定币/法币对杠杆代币不算"币"
EXCLUDE = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT",
    "EURUSDT", "GBPUSDT", "TRYUSDT", "BRLUSDT", "RUBUSDT", "AUDUSDT", "PAXGUSDT",
    "AEURUSDT", "EURIUSDT", "USDTTRY", "USDTBRL", "USTCUSDT", "USTUSDT",
}


def get(url, timeout=25, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # 4xx(除429限流)是"这个请求本身不对", 重试没有意义 —— 比如交易对不存在
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise last


def all_usdt_symbols():
    """从 S3 归档列表拿到「曾经存在于币安现货」的全部 symbol。

    注意: 这个列表接口是分页的(每页 1000 个, 只到 DLTETH 就截断了)。
    不翻页的话 ETH/SOL/XRP 这些排在后面的币根本不会出现 —— 上一版就栽在这里。
    """
    cache = OUT / "_symbols.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    syms, marker = set(), None
    for _ in range(20):
        url = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
               "?delimiter=/&prefix=data/spot/monthly/klines/")
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        body = get(url).decode("utf-8", "ignore")
        syms |= set(re.findall(r"<Prefix>data/spot/monthly/klines/([A-Z0-9]+)/</Prefix>", body))
        if "<IsTruncated>true</IsTruncated>" not in body:
            break
        m = re.search(r"<NextMarker>(.*?)</NextMarker>", body)
        if not m:
            break
        marker = m.group(1)
        time.sleep(0.3)
    out = sorted(syms)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def klines(symbol, start_ms=None, end_ms=None, limit=1000):
    q = f"symbol={symbol}&interval=1d&limit={limit}"
    if start_ms:
        q += f"&startTime={start_ms}"
    if end_ms:
        q += f"&endTime={end_ms}"
    raw = get("https://api.binance.com/api/v3/klines?" + q)
    return json.loads(raw)


def ms(day):
    import datetime as _dt
    return int(_dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)


def main():
    syms = [s for s in all_usdt_symbols() if s.endswith("USDT") and s not in EXCLUDE]
    print(f"币安历史上出现过的 USDT 现货对: {len(syms)} 个")

    # --- 1) 用 2021-08-10 ~ 2021-09-09 的成交额排名, 选出当天真实的大币 ---
    rank_cache = OUT / "_rank_2021_09.json"
    if rank_cache.exists():
        ranked = json.loads(rank_cache.read_text(encoding="utf-8"))
        print(f"排名缓存命中: {len(ranked)} 个")
    else:
        r0, r1 = ms("2021-08-10"), ms(START)
        ranked, failed, not_yet = [], [], 0
        for i, s in enumerate(syms):
            try:
                d = klines(s, r0, r1, limit=40)
                time.sleep(0.10)
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    not_yet += 1        # 2021-08 时这个交易对还不存在 / 已下线
                    continue
                failed.append(f"{s}:HTTP{e.code}")
                continue
            except Exception as e:
                failed.append(f"{s}:{type(e).__name__}")
                continue
            if not isinstance(d, list) or len(d) < 20:
                continue
            qv = sorted(float(x[7]) for x in d)
            ranked.append({"symbol": s, "median_quote_volume": qv[len(qv) // 2],
                           "bars": len(d), "first": d[0][0]})
            if (i + 1) % 100 == 0:
                print(f"  扫描 {i + 1}/{len(syms)} ... 已入选 {len(ranked)}", flush=True)
        ranked.sort(key=lambda x: -x["median_quote_volume"])
        rank_cache.write_text(json.dumps(ranked), encoding="utf-8")
        print(f"2021-09-09 当天在交易、且有30天成交额的币: {len(ranked)} 个"
              f" (当时还不存在的: {not_yet})")
        if failed:
            print(f"!! 真正抓取失败 {len(failed)} 个: {', '.join(failed[:15])}")

    # 硬校验: 这几个当时的大币必须在名单里, 否则说明抓取仍然有问题
    must = ["ETHUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "SOLUSDT", "LINKUSDT"]
    names = {r["symbol"] for r in ranked}
    missing_must = [m for m in must if m not in names]
    if missing_must:
        raise SystemExit(f"名单缺了当时的大币 {missing_must} —— 抓取有问题, 不要用这个币池")

    print(f"\n按 2021-08-10~09-09 成交额排名, 前 {TOP_N} 名(这些就是当时的真实大币):")
    basket = [r["symbol"] for r in ranked[:TOP_N]]
    for r in ranked[:TOP_N]:
        print(f"  {r['symbol']:<12} 30天中位成交额 {r['median_quote_volume'] / 1e6:>10,.1f}M")

    # --- 2) 抓这批币的完整日线 ---
    print(f"\n开始抓 {len(basket)} 个币的完整日线 ...")
    start_ms = ms("2021-01-01")
    for s in basket:
        f = OUT / f"{s}_1d.csv"
        if f.exists() and f.stat().st_size > 1000:
            continue
        rows, cur = [], start_ms
        for _ in range(8):
            d = klines(s, cur, limit=1000)
            if not d:
                break
            rows += d
            if len(d) < 1000:
                break
            cur = int(d[-1][0]) + 86_400_000
            time.sleep(0.3)
        if not rows:
            print(f"  {s:<12} 无数据")
            continue
        with f.open("w", encoding="utf-8") as fh:
            fh.write("open_time,open,high,low,close,volume,quote_volume,trades\n")
            for x in rows:
                fh.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]},{x[7]},{x[8]}\n")
        t0 = time.strftime("%Y-%m-%d", time.gmtime(rows[0][0] / 1000))
        t1 = time.strftime("%Y-%m-%d", time.gmtime(rows[-1][0] / 1000))
        print(f"  {s:<12} {len(rows):>5} 根  {t0} ~ {t1}")
        time.sleep(0.3)

    (OUT / "_basket.json").write_text(json.dumps(basket), encoding="utf-8")
    print(f"\n币池已存: {OUT / '_basket.json'}")


if __name__ == "__main__":
    main()

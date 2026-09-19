#!/usr/bin/env python3
"""
calibration.py -- is Polymarket efficient, and WHERE do longshots get mispriced?

For RESOLVED markets, take the YES price at a fixed horizon before settlement and the
realised outcome, then split the calibration by market CATEGORY x HORIZON.

Why categories: the "carry" trade (buy the near-certain NO, collect time value) has exactly
one source of edge --
        EV = p_market - p_true          (buy NO at q=1-p_market; $1 if the event misses)
so the number that matters is the LONGSHOT bias = mean(p_market) - realised_rate over low-p
markets. >0 means the market overprices the longshot, i.e. selling it (buying NO) has
positive expected value; <0 means the carry is just an under-priced insurance premium.

Concurrent fetches; recent markets only (CLOB price history starts ~2023).
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from runlog import Run

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GAMMA = "https://gamma-api.polymarket.com/events"
PRICES = "https://clob.polymarket.com/prices-history"
DAY = 86400
MIN_TS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
RES = {}

# feeType is Polymarket's own per-category schedule -> a clean category fallback
FEE_CAT = {"crypto_fees_v2": "Crypto", "weather_fees": "Weather", "sports_fees": "Sports",
           "tech_fees": "Tech", "general_fees": "Other", "finance_fees": "Finance",
           "politics_fees": "Politics", "economics_fees": "Economics", "culture_fees": "Culture"}
TAG_CAT = {"crypto": "Crypto", "crypto prices": "Crypto", "bitcoin": "Crypto", "ethereum": "Crypto",
           "token launch": "Crypto", "fdv": "Crypto", "ipos": "Crypto", "public sales": "Crypto",
           "pre-market": "Crypto", "token sales": "Crypto", "memecoins": "Crypto",
           "sports": "Sports", "ufc": "Sports", "nba": "Sports", "nfl": "Sports", "soccer": "Sports",
           "politics": "Politics", "elections": "Politics", "global elections": "Politics", "trump": "Politics",
           "climate & science": "Weather", "weather": "Weather", "climate": "Weather",
           "tech": "Tech", "big tech": "Tech", "ai": "Tech",
           "economics": "Economics", "finance": "Finance", "business": "Business",
           "culture": "Culture", "pop culture": "Culture", "entertainment": "Culture",
           "geopolitics": "Geopolitics", "world": "Geopolitics", "mentions": "Other"}
CAT_ORDER = ["Crypto", "Sports", "Politics", "Weather", "Tech", "Finance", "Economics",
             "Business", "Culture", "Geopolitics", "Other"]


def g(u, retries=3, timeout=30):
    for a in range(retries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}), timeout=timeout).read())
        except Exception:
            time.sleep(0.3 * (a + 1))
    return None


def categorize(e, m):
    hits = set()
    for t in (e.get("tags") or []):
        c = TAG_CAT.get((t.get("label") or "").lower())
        if c:
            hits.add(c)
    for c in CAT_ORDER:
        if c in hits:
            return c
    return FEE_CAT.get(m.get("feeType"), "Other")


def parse_ts(s):
    """Tolerant timestamp parser: gamma mixes '2026-07-29 18:52:59+00' and ISO '...T..:..Z'."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(s.replace("+00", "+0000") if fmt.endswith("%z") else s, fmt)
                       .replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            pass
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def closed_markets(pages=25, min_vol=3000):
    out = []
    for off in range(0, pages * 100, 100):
        evs = g(f"{GAMMA}?closed=true&limit=100&offset={off}&order=endDate&ascending=false") or []
        if not evs:
            break
        for e in evs:
            for m in e.get("markets", []):
                try:
                    if float(m.get("volumeNum") or 0) < min_vol:
                        continue
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                    op = json.loads(m.get("outcomePrices") or "[]")
                    if len(toks) < 2 or len(op) < 2:
                        continue
                    p0 = float(op[0])
                    if not (p0 < 0.01 or p0 > 0.99):
                        continue
                    # the REFERENCE time is when it actually closed, not the scheduled deadline:
                    # markets like "Will X be AG by 2029" resolve early, so endDate sits in the future.
                    ref_ts = parse_ts(m.get("closedTime")) or parse_ts(m.get("endDate"))
                    if ref_ts is None or ref_ts < MIN_TS:
                        continue
                    out.append({"q": (m.get("question") or e.get("title") or "")[:70], "token": toks[0],
                                "outcome": 1.0 if p0 > 0.5 else 0.0, "ref_ts": ref_ts, "cat": categorize(e, m),
                                "vol": float(m.get("volumeNum") or 0)})
                except Exception:
                    continue
        time.sleep(0.15)
    seen, uniq = set(), []
    for m in out:
        if m["token"] not in seen:
            seen.add(m["token"]); uniq.append(m)
    return uniq


def _hist(token, fidelity, start_ts):
    d = g(f"{PRICES}?market={token}&fidelity={fidelity}&startTs={start_ts}")
    return [(int(x["t"]), float(x["p"])) for x in (d or {}).get("history", [])]


def prices(token, ref_ts):
    """Hourly over the last 8d (dense, for T-1d/T-7d) + daily over 70d (for T-30d/T-60d)."""
    pts = _hist(token, 60, ref_ts - 8 * DAY) + _hist(token, 1440, ref_ts - 70 * DAY)
    return sorted({t: p for t, p in pts}.items())


def at(ps, ref_ts, horizon):
    if not ps:
        return None
    tgt = ref_ts - horizon * DAY
    b = min(ps, key=lambda z: abs(z[0] - tgt))
    return b[1] if abs(b[0] - tgt) <= 2 * DAY else None


def stats(data):
    n = len(data)
    if n == 0:
        return None
    mp = sum(p for p, _ in data) / n
    ry = sum(o for _, o in data) / n
    return {"n": n, "mean_p": mp, "realised": ry, "bias": mp - ry,
            "brier": sum((p - o) ** 2 for p, o in data) / n}


def run(horizons=(1, 7, 30, 60), pages=25, min_n=25):
    ms = closed_markets(pages)
    print("resolved markets (vol>=3k, endDate>=2025): %d" % len(ms))
    RES["markets"] = len(ms)
    RES["pages"] = pages
    hist = {}

    def work(m):
        hist[m["token"]] = prices(m["token"], m["ref_ts"])
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, ms))
    print("with price history: %d" % sum(1 for v in hist.values() if v))
    from collections import Counter
    print("category mix: %s\n" % dict(Counter(m["cat"] for m in ms).most_common()))

    rows = []
    for h in horizons:
        grp = defaultdict(list)
        for m in ms:
            p = at(hist.get(m["token"]), m["ref_ts"], h)
            if p is None or p <= 0 or p >= 1:
                continue
            grp[m["cat"]].append((p, m["outcome"]))
        print("=" * 104)
        print("HORIZON T-%dd" % h)
        print("category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)")
        out = []
        for cat in CAT_ORDER:
            d = grp.get(cat)
            if not d or len(d) < min_n:
                continue
            s = stats(d)
            ls = [(p, o) for p, o in d if p < 0.15]
            out.append((cat, s, stats(ls) if len(ls) >= 10 else None))
        for cat, s, l in sorted(out, key=lambda x: -(x[2]["bias"] if x[2] else -9)):
            if l:
                print("%-12s %5d  %6.3f  %6.3f  %+6.3f  %5.3f | %21d  %6.3f  %6.3f    %+7.3f" %
                      (cat, s["n"], s["mean_p"], s["realised"], s["bias"], s["brier"],
                       l["n"], l["mean_p"], l["realised"], l["bias"]))
                rows.append({"horizon_d": h, "category": cat, "n": s["n"], "mean_p": round(s["mean_p"], 4),
                             "realised": round(s["realised"], 4), "bias": round(s["bias"], 4), "brier": round(s["brier"], 4),
                             "ls_n": l["n"], "ls_mean_p": round(l["mean_p"], 4), "ls_realised": round(l["realised"], 4),
                             "ls_edge": round(l["bias"], 4)})
                RES.setdefault("longshot_edge", {})["%s_T%d" % (cat, h)] = round(l["bias"], 4)
            else:
                print("%-12s %5d  %6.3f  %6.3f  %+6.3f  %5.3f | (too few longshots)" %
                      (cat, s["n"], s["mean_p"], s["realised"], s["bias"], s["brier"]))
        print()
    return rows


if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    run_ = Run("calibration_bycat", {"pages": pages, "min_vol": 3000, "horizons_days": [1, 7, 30, 60]})
    run_.source(f"{GAMMA}?closed=true&limit=100&order=endDate&ascending=false")
    with run_.capture("calibration by category x horizon"):
        rows = run(pages=pages)
    for r in rows:
        run_.csv_append(os.path.join(run_.dir, "calib_by_cat.csv"), r)
    run_.finish(RES)

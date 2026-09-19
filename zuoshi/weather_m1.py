#!/usr/bin/env python3
"""
weather_m1.py -- HK daily-temperature market: data layer + market-vs-reality mispricing measurement.

Market: Polymarket `highest-temperature-in-hong-kong-on-<date>` (series hong-kong-daily-weather).
  11 mutually-exclusive 1-degree buckets; resolves on HKO "Absolute Daily Max (deg C)".
  Fee: weather_fees (taker 0.05, maker rebate 0.25).

Reality signal: the market resolves on the **Hong Kong Observatory's official "Daily Maximum
  Temperature"**, truncated to whole degrees (verified 33.1 -> "33C or higher", 30.4 -> "30C",
  29.0 -> "29C").  The Observatory is WMO **45005**, which no longer transmits SYNOP and left
  NOAA ISD in 2018 -- so it is NOT reachable via Ogimet.  `HKO_WMO = 45007` below is instead the
  **Hong Kong International Airport** (verified against VHHH METAR: same days, <1C apart, e.g.
  2026-06-05 34.8 vs 35.0), a different station whose daily max differs from the Observatory by
  -1.0..+2.3C on 2026-06 days -- which is why the SYNOP-based calibration only matched 6/15.
  The correct series comes from HKO's own open-data API through the r.jina.ai proxy (hk.gov.hk
  is network-blocked here): see `fetch_official_daily_max` / the `official` command.

Commands:
  list                       enumerate HK daily-weather events
  calibrate [n]              does the (airport SYNOP) reality signal reproduce the resolved bucket?
  official [n]               does the OFFICIAL Observatory daily max reproduce the resolved bucket?
  measure [n]                trader sim: buy the bucket the HKO running-max is in, at time t
"""
from __future__ import annotations
import csv, json, os, re, sys, time
import urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from runlog import Run

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather")
GAMMA = "https://gamma-api.polymarket.com/events"
PRICES = "https://clob.polymarket.com/prices-history"
OGIMET = "https://www.ogimet.com/cgi-bin/getsynop"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

HK_TZ = timezone(timedelta(hours=8))
HKO_WMO = "45007"          # AIRPORT (Chek Lap Kok / VHHH) -- NOT the resolution station, see docstring
PROXY = "https://r.jina.ai/"           # hk.gov.hk is network-blocked directly
CLMMAXT = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&year=%d&month=%d&station=HKO&rformat=csv&lang=en"
FEE_RATE, REBATE_RATE = 0.05, 0.25   # weather_fees


def http_json(url, retries=4, timeout=45):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pm-weather-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5 * (a + 1))
    raise RuntimeError("GET %s failed: %s" % (url, last))


def http_text(url, retries=4, timeout=60):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pm-weather-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.6 * (a + 1))
    raise RuntimeError("GET %s failed: %s" % (url, last))


def fee(p):
    return FEE_RATE * p * (1 - p)      # taker fee per share at price p


def _cache(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def list_hk_events(force=False):
    p = _cache("hk_events.json")
    if not force and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    evs = http_json(f"{GAMMA}?series_slug=hong-kong-daily-weather&limit=200")
    with open(p, "w") as f:
        json.dump(evs, f)
    return evs


def canonical_events():
    """Canonical (non-'arch-') events with real volume, oldest first."""
    out = []
    for e in list_hk_events():
        if not e.get("markets") or str(e.get("slug", "")).startswith("arch-"):
            continue
        if sum(float(m.get("volumeNum") or 0) for m in e["markets"]) <= 5000:
            continue
        out.append(e)
    out.sort(key=lambda e: str(e.get("endDate")))
    return out


def hk_date_of(ev):
    return datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).astimezone(HK_TZ).strftime("%Y-%m-%d")


def market_of_event(ev):
    out = []
    for m in ev["markets"]:
        tok = json.loads(m["clobTokenIds"])
        out.append({
            "bucket": m.get("groupItemTitle"),
            "token": tok[0],
            "won": float(json.loads(m["outcomePrices"])[0]),
            "vol": float(m.get("volumeNum") or 0),
        })
    return out


# ---- reality signal: HKO official station hourly SYNOP --------------------------
_TEMP_RE = re.compile(r"\b1([01])(\d{3})\s+2[01]\d{3}\b")


def fetch_hko_synop(date_str, force=False):
    """HKO (WMO 45007) hourly air temperature for HK-local day date_str -> [(hk_ts, degC)]."""
    p = _cache("hko_synop_%s.csv" % date_str)
    if not force and os.path.exists(p):
        with open(p) as f:
            return [(int(r["ts"]), float(r["t"])) for r in csv.DictReader(f)]
    d = datetime.strptime(date_str, "%Y-%m-%d")
    begin = (d - timedelta(days=1)).strftime("%Y%m%d1600")
    end = (d + timedelta(days=1)).strftime("%Y%m%d0000")
    txt = http_text(f"{OGIMET}?begin={begin}&end={end}&block={HKO_WMO}")
    rows = []
    for ln in txt.splitlines():
        f = ln.split(",")
        if len(f) < 7 or f[0] != HKO_WMO:
            continue
        m = _TEMP_RE.search(",".join(f[6:]))
        if not m:
            continue
        sign = -1 if m.group(1) == "1" else 1
        ts = datetime(int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), tzinfo=timezone.utc)
        hk = ts.astimezone(HK_TZ)
        if hk.strftime("%Y-%m-%d") == date_str:
            rows.append((int(hk.timestamp()), sign * int(m.group(2)) / 10.0))
    rows.sort()
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ts", "t"]); w.writerows(rows)
    return rows


# ---- official resolution source: HKO Observatory "Daily Maximum Temperature" -------------
def fetch_official_daily_max(year, month, force=False):
    """{YYYY-MM-DD: degC} from HKO CLMMAXT (Observatory), via the r.jina.ai proxy."""
    p = _cache("hko_official_%04d-%02d.csv" % (year, month))
    if not force and os.path.exists(p):
        with open(p) as f:
            return {r["date"]: float(r["t"]) for r in csv.DictReader(f)}
    txt = http_text(PROXY + CLMMAXT % (year, month), retries=3, timeout=120)
    out = {}
    for ln in txt.splitlines():
        f = [x.strip().strip('"') for x in ln.split(",")]
        if len(f) >= 4 and f[0].isdigit() and f[1].isdigit() and f[2].isdigit():
            try:
                out["%04d-%02d-%02d" % (int(f[0]), int(f[1]), int(f[2]))] = float(f[3])
            except ValueError:
                pass
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "t"]); w.writerows(sorted(out.items()))
    return out


def official_max(date_str, force=False):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return fetch_official_daily_max(d.year, d.month, force).get(date_str)


def bucket_for_value(mk, v):
    """Which bucket label contains a daily max value (1-degree buckets, truncated to whole C)."""
    b = int(v)  # truncate: '30C' covers [30,31)
    for m in mk:
        lab = str(m["bucket"]).replace("°C", "C")
        if lab == "%dC" % b:
            return m
        if "or below" in lab and b <= int(re.match(r"(\d+)", lab).group(1)):
            return m
        if "or higher" in lab and b >= int(re.match(r"(\d+)", lab).group(1)):
            return m
    return None


# ---- bucket price history --------------------------------------------------------
def fetch_bucket_prices(token, start_ts, fidelity=10, force=False):
    p = _cache("px_%s_%dm.csv" % (token[:16], fidelity))
    if not force and os.path.exists(p):
        with open(p) as f:
            return [(int(r["ts"]), float(r["p"])) for r in csv.DictReader(f)]
    rows = [(int(h["t"]), float(h["p"])) for h in
            http_json(f"{PRICES}?market={token}&fidelity={fidelity}&startTs={start_ts}").get("history", [])]
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ts", "p"]); w.writerows(rows)
    return rows


def price_at(series, ts):
    """Last price at or before ts (series must be sorted)."""
    lo, hi, best = 0, len(series) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= ts:
            best = series[mid][1]; lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---- commands --------------------------------------------------------------------
def cmd_list():
    evs = canonical_events()
    print("canonical HK daily-weather events (vol>5000): %d" % len(evs))
    for e in evs:
        mk = market_of_event(e)
        w = next((m["bucket"] for m in mk if m["won"] == 1.0), "?")
        print("  %-44s end=%s vol=%8.0f winner=%s" %
              (e["slug"][:44], hk_date_of(e), sum(m["vol"] for m in mk), str(w).replace("°C", "C")))


def cmd_calibrate(n=12, fidelity=10):
    evs = canonical_events()[::3][-n:]
    print("calibration: %d days (HKO SYNOP vs resolved bucket, truncation convention)" % len(evs))
    print("date        hkoMax hkTime  winner        fromMax  match  f  winP(mid)  winP(close)")
    ok = 0
    for ev in evs:
        mk = market_of_event(ev)
        date = hk_date_of(ev)
        winner = next((m for m in mk if m["won"] == 1.0), None)
        try:
            temps = fetch_hko_synop(date)
        except Exception as e:
            print("  %s SYNOP ERR %s" % (date, e)); continue
        if not temps:
            print("  %s  (no SYNOP data)" % date); continue
        mx = max(t for _, t in temps)
        mt = datetime.fromtimestamp(next(ts for ts, t in temps if t == mx), HK_TZ).strftime("%H:%M")
        bm = bucket_for_value(mk, mx)
        match = bm is not None and bm["bucket"] == winner["bucket"]
        ok += match
        try:
            px = fetch_bucket_prices(winner["token"], min(ts for ts, _ in temps) - 6 * 3600, fidelity)
        except Exception as e:
            print("  %s px ERR %s" % (date, e)); continue
        fans = fee(0.5)
        mid = price_at(px, int(datetime.strptime(date + " 12:00", "%Y-%m-%d %H:%M").replace(tzinfo=HK_TZ).timestamp()))
        close = px[-1][1] if px else float("nan")
        print("  %s %6.1f %s  %-13s %-8s %-5s %.2f %8.3f %10.3f" %
              (date, mx, mt, str(winner["bucket"]).replace("°C", "C"),
               str(bm["bucket"]).replace("°C", "C") if bm else "-", "Y" if match else "n",
               fee(0.5), mid if mid is not None else float("nan"), close))
    print("HKO-SYNOP bucket == resolved winner: %d/%d" % (ok, len(evs)))
    return {"days": len(evs), "match": ok, "match_rate": round(ok / len(evs), 4) if evs else None,
            "fidelity_min": fidelity}


def cmd_official(n=20):
    """Does the OFFICIAL Observatory daily max reproduce the resolved bucket?"""
    evs = canonical_events()[::3][-n:]
    print("official-source check: %d days (HKO Observatory Daily Maximum Temperature, CLMMAXT)" % len(evs))
    print("date        official  bucket    winner         match")
    ok = 0
    for ev in evs:
        date = hk_date_of(ev)
        mk = market_of_event(ev)
        winner = next((m for m in mk if m["won"] == 1.0), None)
        try:
            v = official_max(date)
        except Exception as e:
            print("  %s  OFFICIAL ERR %s" % (date, e)); continue
        bm = bucket_for_value(mk, v) if v is not None else None
        match = winner is not None and bm is not None and bm["bucket"] == winner["bucket"]
        ok += match
        print("  %s %8s  %-9s %-14s %s" %
              (date, "%.1f" % v if v is not None else "n/a",
               str(bm["bucket"]).replace("°C", "C") if bm else "-",
               str(winner["bucket"]).replace("°C", "C") if winner else "-", "Y" if match else "n"))
    print("official bucket == resolved winner: %d/%d" % (ok, len(evs)))
    return {"days": len(evs), "match": ok, "match_rate": round(ok / len(evs), 4) if evs else None,
            "source": "HKO CLMMAXT Observatory Daily Maximum Temperature"}


def cmd_measure(n=15, fidelity=10, hours=(13, 15, 16, 17, 18)):
    """Trader sim: at HKT hour h, buy the bucket the HKO running-max is in, hold to settlement."""
    evs = canonical_events()[::2][-n:]
    print("measure: %d days; buy bucket(HKO running max) at HKT hour, settle; fee=%.3f@0.5" % (len(evs), fee(0.5)))
    hdr = "date        hkoFinal  winner    " + "".join("%9s" % ("%dh_edge" % h) for h in hours)
    print(hdr + "   fills")
    agg = {h: [] for h in hours}
    for ev in evs:
        mk = market_of_event(ev)
        date = hk_date_of(ev)
        winner = next((m for m in mk if m["won"] == 1.0), None)
        try:
            temps = fetch_hko_synop(date)
        except Exception:
            continue
        if not temps:
            continue
        # price series per bucket
        ser = {}
        t0 = min(ts for ts, _ in temps) - 6 * 3600
        for m in mk:
            ser[m["bucket"]] = fetch_bucket_prices(m["token"], t0, fidelity)
        row = []
        fills = 0
        for h in hours:
            t = int(datetime.strptime(date + " %02d:00" % h, "%Y-%m-%d %H:%M").replace(tzinfo=HK_TZ).timestamp())
            prior = [x for x in temps if x[0] <= t]
            if not prior:
                row.append(None); continue
            runmax = max(x[1] for x in prior)
            bm = bucket_for_value(mk, runmax)
            if bm is None:
                row.append(None); continue
            p = price_at(ser[bm["bucket"]], t)
            if p is None or p <= 0.001 or p >= 0.999:
                row.append(0.0 if p is not None else None)
                continue
            wins = 1.0 if bm["bucket"] == winner["bucket"] else 0.0
            edge = wins - p - fee(p)        # buy 1 share at p, settle at wins
            row.append(edge)
            fills += 1
            agg[h].append(edge)
        print("  %s %8.1f  %-9s " % (date, max(t for _, t in temps), str(winner["bucket"]).replace("°C", "C")) +
              "".join("%9s" % ("-" if e is None else "%+.3f" % e) for e in row) + "   %d" % fills)
    print("\nmean edge per share by entry hour (n = days with a fill):")
    for h in hours:
        v = agg[h]
        if v:
            print("  %2dh: n=%2d  mean=%+.4f  winrate=%.0f%%" % (h, len(v), sum(v) / len(v), 100 * sum(1 for x in v if x > 0) / len(v)))
    return {"days": len(evs), "fee_at_50c": round(fee(0.5), 4), "hours": hours,
            "hour_edge": {str(h): {"n": len(agg[h]), "mean_edge": round(sum(agg[h]) / len(agg[h]), 5),
                                   "winrate": round(sum(1 for x in agg[h] if x > 0) / len(agg[h]), 3)}
                          for h in hours if agg[h]}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run = Run("weather_%s" % cmd, {"cmd": cmd, "n": arg or (20 if cmd == "official" else (12 if cmd == "calibrate" else 15))})
    rel = os.path.relpath(DATA, os.getcwd())
    with run.capture("weather_%s" % cmd):
        if cmd == "calibrate":
            res = cmd_calibrate(arg or 12)
        elif cmd == "official":
            res = cmd_official(arg or 20)
        elif cmd == "measure":
            res = cmd_measure(arg or 15)
        else:
            res = cmd_list()
    # hash-lock the cached reality signal + price history this run actually read from
    for pat in ("hko_synop_*.csv", "hko_official_*.csv", "px_*.csv", "hk_events.json"):
        run.source_glob(os.path.join(rel, pat))
    run.finish(res)

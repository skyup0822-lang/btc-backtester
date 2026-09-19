#!/usr/bin/env python3
"""Find fee-enabled BTC multi-strikes weekly events (2026+) and classify their BTC regime."""
import json, urllib.request
from datetime import datetime, timezone

def g(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "x"}), timeout=40).read())

def okx_close_from(start_ts, end_ts):
    rows = {}
    after = (end_ts + 2 * 3600) * 1000
    for _ in range(80):
        got = g("https://www.okx.com/api/v5/market/history-candles?instId=BTC-USDT&bar=1H&after=%d&limit=100" % after).get("data", [])
        if not got:
            break
        for c in got:
            rows[int(c[0]) // 1000] = float(c[4])
        earliest = min(int(c[0]) for c in got) // 1000
        if earliest <= start_ts or len(got) < 100:
            break
        after = earliest * 1000
    return sorted((t, c) for t, c in rows.items() if start_ts <= t <= end_ts)

# events deployed mid-2026 onward (fee-enabled era), all regimes
evs = g("https://gamma-api.polymarket.com/events?series_slug=btc-multi-strikes-weekly&closed=true&limit=100&start_date_min=2026-05-01")
print("n(2026+)=%d" % len(evs))
for e in evs:
    m = e["markets"]
    fee = {x.get("feeType") for x in m}
    en = {x.get("feesEnabled") for x in m}
    if not en or True not in en:
        continue
    sd = e["startDate"][:10]; ed = e["endDate"][:10]
    s = int(datetime.fromisoformat(e["startDate"].replace("Z", "+00:00")).timestamp())
    en_ = int(datetime.fromisoformat(e["endDate"].replace("Z", "+00:00")).timestamp())
    b = okx_close_from(s - 3600, en_ + 3600)
    if len(b) < 48:
        continue
    first, last = b[0][1], b[-1][1]
    lo = min(c for _, c in b); hi = max(c for _, c in b)
    drift = (last - first) / first * 100
    rng = (hi - lo) / ((lo + hi) / 2) * 100
    reg = "UP" if drift > 2 else ("DOWN" if drift < -2 else "RANGE")
    print("%-44s %s->%s mkt=%-3d fee=%s  btc %6.0f->%6.0f  %+5.1f%%  range %4.1f%%  %s"
          % (e["slug"], sd, ed, len(m), fee, first, last, drift, rng, reg))

#!/usr/bin/env python3
"""Run the MM paper sim across BTC weekly events of different real regimes.

Same fee model (crypto_fees_v2), same method for every event -> apples-to-apples.
For each event: aggregate over all strikes, plus a "clean" aggregate that drops
strikes flagged by a cross-strike arbitrage-consistency check (P(BTC>K_low) must
be >= P(BTC>K_high); a lower strike printing below a higher one is a data anomaly
that a real MM's M2 fair band would reject).  Reports naive (skew=0) and minimal
M4 (skew=1e-4) side by side.
"""
import importlib.util
import os
from datetime import datetime
from runlog import Run

spec = importlib.util.spec_from_file_location("mm", "pm_mm_sim.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

EVENTS = [
    ("bitcoin-above-on-june-20-2026", "RANGE"),
    ("bitcoin-above-on-july-25-2026", "RANGE"),
    ("bitcoin-above-on-august-10-2026", "RANGE"),
    ("bitcoin-above-on-june-16-2026", "UP"),
    ("bitcoin-above-on-july-15-2026", "UP"),
    ("bitcoin-above-on-august-8-2026", "UP"),
    ("bitcoin-above-on-july-29-2026", "DOWN"),
    ("bitcoin-above-on-august-15-2026", "DOWN"),
]
H, FID, SIZE, REFRESH, MAXPOS = 0.01, 10, 100, 30, 1000


def anomaly_strikes(ms, step=600, tol=0.03):
    """Flag strikes involved in cross-strike monotonicity violations."""
    grid = {}
    for m in ms:
        for ts, p in m["prices"]:
            b = ts // step
            grid.setdefault(b, {})[m["strike"]] = p
    strikes = sorted(m["strike"] for m in ms)
    bad = set()
    for b, d in grid.items():
        have = sorted(d)
        for i, kl in enumerate(have):
            for kh in have[i + 1:]:
                if d[kl] < d[kh] - tol:        # lower strike priced below a higher one
                    bad.add(kl)
                    break
    return bad


RUN = Run("cross_regime", {"events": [s for s, _ in EVENTS], "half_spread": H, "fidelity_min": FID,
                           "size": SIZE, "refresh_min": REFRESH, "max_pos": MAXPOS})
RUN.start_capture()
rows = []
print("regime  endDate     slug                              aggNaive  aggM4  cleanNaive cleanM4  badStrikes        pivot  p0    pivNaive pivM4")
for slug, regime in EVENTS:
    ev = mm.fetch_event(slug)
    ms = mm.market_from_event(ev)
    t0 = int(datetime.fromisoformat(ev["startDate"].replace("Z", "+00:00")).timestamp())
    for m in ms:
        m["prices"] = mm.fetch_prices(m["token"], FID, t0 - 3600)
    bad = anomaly_strikes(ms)

    def run(sk):
        tot = 0.0
        clean = 0.0
        per = {}
        for m in ms:
            r = mm.simulate(m["prices"], m["yes_win"], H, SIZE, REFRESH, MAXPOS, m["tick"], sk, 1.0)
            tot += r["total_pnl"]
            per[m["strike"]] = r["total_pnl"]
            if m["strike"] not in bad:
                clean += r["total_pnl"]
        return tot, clean, per

    tn, cn, pn = run(0.0)
    tm, cm, pm = run(1e-4)
    piv = min(ms, key=lambda m: abs((m["prices"][0][1] if m["prices"] else 1) - 0.5))["strike"]
    p0 = next(m["prices"][0][1] for m in ms if m["strike"] == piv)
    print("%-6s %s  %-32s %8.1f %7.1f %9.1f %8.1f  %-16s %6d %5.3f %8.1f %6.1f"
          % (regime, ev["endDate"][:10], slug, tn, tm, cn, cm,
             ",".join(str(s // 1000) + "k" for s in sorted(bad)) or "-",
             piv, p0, pn.get(piv, 0), pm.get(piv, 0)))
    rows.append({"regime": regime, "slug": slug, "end_date": ev["endDate"][:10],
                 "agg_naive": round(tn, 1), "agg_m4": round(tm, 1),
                 "clean_naive": round(cn, 1), "clean_m4": round(cm, 1),
                 "bad_strikes": sorted(bad), "pivot": piv, "p0": round(p0, 4)})

RUN.stop_capture("cross-regime MM paper sim")
for pat in ("data/event_*.json", "data/prices_*.csv", "data/okx_btc_1h*.csv"):
    RUN.source_glob(pat)
for r in rows:
    RUN.csv_append(os.path.join(RUN.dir, "events.csv"), {**r, "bad_strikes": ",".join(str(s) for s in r["bad_strikes"])})
RUN.finish({"events": rows, "agg_naive_total": round(sum(r["agg_naive"] for r in rows), 1),
            "agg_m4_total": round(sum(r["agg_m4"] for r in rows), 1)})

#!/usr/bin/env python3
"""
carry_scan.py -- treat near-certain, long-dated Polymarket positions as zero-coupon bonds.

Buying NO at price q on a market expiring in D days returns $1 if the event does NOT happen,
i.e. an implied annualised yield = (1/q_effective)^(365/D) - 1, where q_effective includes the
taker fee. Because fee = rate * p * (1-p) -> 0 as p -> 1, near-certain positions carry almost
no fee, so the yield is (almost) pure carry. The RISK is the tail: if the event happens you
lose the whole stake. This scans live markets and ranks carry vs that tail.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from runlog import Run

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GAMMA = "https://gamma-api.polymarket.com/events"
BOOKS = "https://clob.polymarket.com/books"
FEE_RATE = {"crypto_fees_v2": 0.07, "weather_fees": 0.05, "general_fees": 0.05,
            "tech_fees": 0.04, "sports_fees": 0.03, None: 0.0}
DAY = 86400
RF = 0.045          # risk-free benchmark (US T-bills), configurable


def http(url, body=None, retries=3, timeout=30):
    for a in range(retries):
        try:
            if body is None:
                req = urllib.request.Request(url, headers={"User-Agent": "x"})
            else:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "User-Agent": "x"}, method="POST")
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception:
            time.sleep(0.3 * (a + 1))
    return None


def best_ask(book):
    a = [(float(x["price"]), float(x["size"])) for x in (book.get("asks") or [])]
    return min(a)[0] if a else None


def size_at(book, price):
    return sum(float(x["size"]) for x in (book.get("asks") or []) if float(x["price"]) <= price + 1e-9)


def main(min_days=14, min_no=0.85, top=30):
    from datetime import datetime, timezone
    now = time.time()
    evs = http(f"{GAMMA}?active=true&closed=false&limit=400") or []
    cand = []
    for e in evs:
        for m in e.get("markets", []):
            raw = m.get("clobTokenIds")
            if not raw:
                continue
            toks = json.loads(raw)
            if len(toks) < 2:
                continue
            try:
                end_ts = int(datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp())
            except Exception:
                continue
            days = (end_ts - now) / DAY
            if days < min_days:
                continue
            cand.append((e, m, toks[1], days))
    print("active long-dated markets (>=%dd): %d" % (min_days, len(cand)))

    books = {}
    for ch in [cand[i:i + 100] for i in range(0, len(cand), 100)]:
        d = http(BOOKS, [{"token_id": c[2]} for c in ch])
        for b in (d or []):
            books[b["asset_id"]] = b
        time.sleep(0.15)

    rows = []
    for e, m, no_tok, days in cand:
        q = best_ask(books.get(no_tok, {}))
        if q is None or q < min_no:
            continue
        rate = FEE_RATE.get(m.get("feeType"), 0.05)
        fee = rate * 1 * q * (1 - q)
        eff = q + fee                              # effective cost per $1 payout
        if eff >= 1:
            continue
        ann = (1.0 / eff) ** (365.0 / days) - 1
        depth = size_at(books.get(no_tok, {}), q)
        rows.append({"slug": e["slug"], "q": m.get("question", "")[:58], "days": days,
                     "no": q, "fee": fee, "ann": ann, "depth": depth,
                     "p_event": 1 - q, "fee_type": m.get("feeType")})
    rows.sort(key=lambda r: -r["ann"])
    print("\nNO-side carry, annualised (risk-free bench = %.1f%%); p_event = implied chance you lose it all" % (RF * 100))
    print("%-52s %5s %6s %6s %8s %8s %7s" % ("market (question)", "days", "NO", "fee", "ann_yld", "p_event", "depth"))
    for r in rows[:top]:
        flag = "  <-- > RF" if r["ann"] > RF else ""
        print("%-52s %5.0f %6.3f %6.4f %7.1f%% %7.1f%% %7.0f%s" %
              (r["q"], r["days"], r["no"], r["fee"], r["ann"] * 100, r["p_event"] * 100, r["depth"], flag))
    if rows:
        above = [r for r in rows if r["ann"] > RF]
        print("\nmarkets with carry > risk-free: %d / %d" % (len(above), len(rows)))
        if above:
            import statistics
            print("median excess over RF among those: %.1f%% ann" % (statistics.median([(r["ann"] - RF) * 100 for r in above])))
            # break-even: how many periods of carry to recover one tail loss
            print("\ntail math: buying NO at q earns %.1f%% per period; one tail event loses %.0f%% of stake"
                  % ((1 / rows[0]["no"] - 1) * 100, rows[0]["no"] * 100))
            print("  -> need ~%.1f non-events to recover a single loss" % (rows[0]["no"] / (1 - rows[0]["no"])))
    return {
        "min_days": min_days, "min_no": min_no, "risk_free": RF,
        "candidates": len(cand), "priced": len(rows), "above_rf": len([r for r in rows if r["ann"] > RF]),
        "best_ann": round(rows[0]["ann"], 4) if rows else None,
        "best_market": rows[0]["slug"] if rows else None,
        "rows": rows[:top],
    }


if __name__ == "__main__":
    run = Run("carry_scan", {"min_days": 14, "min_no": 0.85, "top": 30, "risk_free": RF})
    run.source(f"{GAMMA}?active=true&closed=false&limit=400")
    with run.capture("NO-side carry scan"):
        res = main()
    snap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "carry_snapshots.csv")
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in res["rows"]:
        run.csv_append(snap, {"ts_utc": ts, "risk_free": RF, **r})
    run.source(snap)
    run.finish({k: v for k, v in res.items() if k != "rows"})

#!/usr/bin/env python3
"""
arb_scan.py -- Polymarket arbitrage & efficiency scanner (executable bid/ask, real fees).

Live CLOB order books (NOT midpoints), batch-fetched. Checks:
  A. binary complement  : buy YES+NO for < $1 (merge -> $1), or sell both for > $1
  B. negRisk multi-outcome: sum of every bucket's ask < $1 (exactly one pays $1); depth-walked
  C. nested strikes      : P(BTC>K) non-increasing in K
Fees are taker fees at the market's rate; min_order_size from the book.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from datetime import datetime, timezone
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from runlog import Run

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GAMMA = "https://gamma-api.polymarket.com/events"
BOOKS = "https://clob.polymarket.com/books"
FEE_RATE = {"crypto_fees_v2": 0.07, "weather_fees": 0.05, "general_fees": 0.05,
            "tech_fees": 0.04, "sports_fees": 0.03, None: 0.0}


def http(url, body=None, retries=3, timeout=35):
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


def lv(book, side):
    out = [(float(x["price"]), float(x["size"])) for x in (book.get(side) or [])]
    return sorted(out, key=lambda z: -z[0]) if side == "bids" else sorted(out, key=lambda z: z[0])


def best(l):
    return l[0][0] if l else None


def cost_buy(asks, size):
    need, cost = size, 0.0
    for p, s in asks:
        t = min(need, s); cost += t * p; need -= t
        if need <= 1e-9:
            return cost
    return None


def size_at(asks, max_price):
    """shares available at or below max_price"""
    return sum(s for p, s in asks if p <= max_price + 1e-9)


def fee(rate, shares, p):
    return rate * shares * p * (1 - p)


def chunk(it, n):
    it = list(it)
    for i in range(0, len(it), n):
        yield it[i:i + n]


def scan(limit_events=60, min_vol=1000, verbose=False):
    evs = http(f"{GAMMA}?active=true&closed=false&limit=300") or []
    evs = [e for e in evs if e.get("markets") and sum(float(m.get("volumeNum") or 0) for m in e["markets"]) >= min_vol]
    evs.sort(key=lambda e: -sum(float(m.get("volumeNum") or 0) for m in e["markets"]))
    evs = evs[:limit_events]

    idx = {}       # token_id -> (event, market_title, side)
    for e in evs:
        for m in e["markets"]:
            raw = m.get("clobTokenIds")
            if not raw:
                continue
            tk = json.loads(raw)
            if len(tk) < 2:
                continue
            idx[tk[0]] = (e, m, "YES")
            idx[tk[1]] = (e, m, "NO")
    print("events=%d  markets=%d  tokens=%d" % (len(evs), len(idx) // 2, len(idx)))

    books = {}
    for ch in chunk(idx.keys(), 100):
        d = http(BOOKS, [{"token_id": t} for t in ch])
        if not d:
            continue
        for b in d:
            books[b["asset_id"]] = b
        time.sleep(0.15)
    print("books fetched=%d" % len(books))

    comp, sums, mono = [], [], []
    comp_spread, comp_net, neg_spread = [], [], []
    for e in evs:
        mk = e["markets"]
        rate = FEE_RATE.get(mk[0].get("feeType"), 0.05)
        rows = []
        for m in mk:
            raw = m.get("clobTokenIds")
            if not raw:
                continue
            tk = json.loads(raw)
            if len(tk) < 2:
                continue
            yb, nb = books.get(tk[0], {}), books.get(tk[1], {})
            ya, ybid = best(lv(yb, "asks")), best(lv(yb, "bids"))
            na, nbid = best(lv(nb, "asks")), best(lv(nb, "bids"))
            rows.append({"t": m.get("groupItemTitle"), "ya": ya, "yb": ybid, "na": na, "nb": nbid,
                         "yasks": lv(yb, "asks"), "ybids": lv(yb, "bids"),
                         "nasks": lv(nb, "asks"), "nbids": lv(nb, "bids"),
                         "vol": float(m.get("volumeNum") or 0)})
            if ya is not None and na is not None:
                gross = ya + na; f = fee(rate, 1, ya) + fee(rate, 1, na)
                comp_spread.append(1 - gross)          # gross edge (pre-fee), >0 = raw dislocation
                comp_net.append(1 - gross - f)         # net edge, >0 = arbitrage
                if 1 - gross - f > 1e-9:
                    comp.append((e["slug"], m.get("groupItemTitle"), "BUY", 1 - gross - f, rate))
            if ybid is not None and nbid is not None:
                gross = ybid + nbid; f = fee(rate, 1, ybid) + fee(rate, 1, nbid)
                comp_spread.append(gross - 1)
                comp_net.append(gross - 1 - f)
                if gross - 1 - f > 1e-9:
                    comp.append((e["slug"], m.get("groupItemTitle"), "SELL", gross - 1 - f, rate))
        # B. negRisk sum
        if e.get("negRisk") and len(rows) > 1 and all(r["ya"] is not None for r in rows):
            s_ask = sum(r["ya"] for r in rows)
            f = sum(fee(rate, 1, r["ya"]) for r in rows)
            neg_spread.append(s_ask - 1 + f)
            if s_ask + f < 1 - 1e-9:
                # depth: how deep does sum(asks) stay < 1 (approx: add equal size per leg)
                size = 0.0
                for step in (5, 25, 100):
                    ok = all(cost_buy(r["yasks"], step) is not None for r in rows)
                    if not ok:
                        break
                    tot = sum(cost_buy(r["yasks"], step) for r in rows)
                    tf = sum(fee(rate, step, cost_buy(r["yasks"], step) / step) for r in rows)
                    if tot + tf < step:
                        size = step
                sums.append((e["slug"], len(rows), "BUYALL", 1 - s_ask - f, s_ask, size))
            if all(r["yb"] is not None for r in rows):
                s_bid = sum(r["yb"] for r in rows)
                if s_bid > 1 + 1e-9:
                    sums.append((e["slug"], len(rows), "SELLALL", s_bid - 1, s_bid, 0))
        # C. monotonicity of nested integer strikes
        pairs = []
        for r in rows:
            try:
                k = int(str(r["t"]).replace(",", "").replace("$", ""))
            except Exception:
                continue
            if r["ya"] is not None and r["yb"] is not None:
                pairs.append((k, (r["ya"] + r["yb"]) / 2, r["yb"], r["ya"]))
        pairs.sort()
        for i in range(len(pairs) - 1):
            (k1, m1, b1, a1), (k2, m2, b2, a2) = pairs[i], pairs[i + 1]
            if m1 < m2 - 0.01:
                mono.append((e["slug"], k1, k2, m1, m2, m2 - m1))
        if verbose:
            print("  %-46s mkt=%-3d negRisk=%-5s" % (e["slug"][:46], len(mk), e.get("negRisk")))

    def pct(v, q):
        if not v:
            return float("nan")
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))]

    print("\nA. BINARY COMPLEMENT net-arb hits: %d  (n=%d priced markets)" % (len(comp), len(comp_spread)))
    if comp_spread:
        print("   GROSS edge (pre-fee):  min=%.4f  p10=%.4f  median=%.4f  max=%.4f  [>0 = raw dislocation]"
              % (min(comp_spread), pct(comp_spread, .10), pct(comp_spread, .50), max(comp_spread)))
        print("   NET   edge (post-fee): min=%.4f  p10=%.4f  median=%.4f  max=%.4f  [>0 = arbitrage]"
              % (min(comp_net), pct(comp_net, .10), pct(comp_net, .50), max(comp_net)))
    for h in comp[:20]:
        print("   %-42s %-12s %-4s edge=%.4f (rate=%.2f)" % (h[0][:42], str(h[1])[:12], h[2], h[3], h[4]))
    print("\nB. negRisk SUM-of-buckets net-arb hits: %d  (n=%d events)" % (len(sums), len(neg_spread)))
    if neg_spread:
        print("   tightness (sum_ask - 1 + fees): min=%.4f  p10=%.4f  median=%.4f  max=%.4f"
              % (min(neg_spread), pct(neg_spread, .10), pct(neg_spread, .50), max(neg_spread)))
    for h in sums[:20]:
        print("   %-42s n=%-3d %-7s edge=%.4f sum=%.4f maxsize=%g" % (h[0][:42], h[1], h[2], h[3], h[4], h[5]))
    print("\nC. NESTED-STRIKE monotonicity violations (>1c): %d" % len(mono))
    for h in mono[:20]:
        print("   %-38s P(K=%d)=%.3f < P(K=%d)=%.3f  viol=%.3f" % (h[0][:38], h[1], h[3], h[2], h[4], h[5]))
    # machine-readable sample result -> run manifest + the long-lived snapshot CSV
    return {
        "events": len(evs), "markets": len(idx) // 2, "books": len(books),
        "priced_markets": len(comp_spread),
        "comp_net_hits": len(comp),
        "comp_gross_median": None if not comp_spread else round(pct(comp_spread, .5), 5),
        "comp_net_best": None if not comp_net else round(max(comp_net), 5),
        "neg_events": len(neg_spread), "neg_hits": len(sums),
        "neg_tightest": None if not neg_spread else round(min(neg_spread), 5),
        "mono_violations": len(mono),
    }


if __name__ == "__main__":
    argv, opts, pos, i = sys.argv[1:], {}, [], 0
    while i < len(argv):                      # --watch/--interval take a value; the rest is positional
        a = argv[i]
        if a in ("--watch", "--interval"):
            opts[a] = argv[i + 1]; i += 2
        elif a.startswith("-"):
            i += 1
        else:
            pos.append(a); i += 1
    n = int(pos[0]) if pos else 60
    verbose = "-v" in argv
    samples, interval = int(opts.get("--watch", 1)), float(opts.get("--interval", 60.0))
    run = Run("arb_scan", {"limit_events": n, "samples": samples, "interval_s": interval, "verbose": verbose})
    run.source(f"{GAMMA}?active=true&closed=false&limit=300")
    snap = os.path.join(DATA, "arb_snapshots.csv")

    rows = []
    for i in range(samples):
        t0 = time.time()
        with run.capture("ARB scan sample %d/%d (limit_events=%d)" % (i + 1, samples, n)):
            stats = scan(limit_events=n, verbose=verbose)
        row = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sample": i + 1,
               "duration_s": round(time.time() - t0, 1), **stats}
        run.csv_append(snap, row)
        rows.append(row)
        if i + 1 < samples:
            time.sleep(interval)

    def best(key, how=max):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(how(vals), 5) if vals else None

    run.source(snap)
    run.finish({
        "samples": len(rows),
        "net_arb_hits_total": sum(r["comp_net_hits"] + r["neg_hits"] for r in rows),
        "best_comp_net_edge": best("comp_net_best"),
        "tightest_neg_basket": best("neg_tightest", min),
        "mono_violations_total": sum(r["mono_violations"] for r in rows),
    })

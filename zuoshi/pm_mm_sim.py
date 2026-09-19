#!/usr/bin/env python3
"""
pm_mm_sim.py -- naive Polymarket crypto-MM paper simulator (M5 + M2 + M3 minimal).

The question:
    Q1  naive two-sided quoting: does (half)spread + maker rebate cover adverse selection?
    Q2  on "near-50/50" strikes specifically, is naive MM profitable?

Docs-verified fee model (Polymarket crypto_fees_v2; a measured activity-replay confirms):
    taker_fee  = feeRate * shares * p * (1-p)   # feeRate=0.07, symmetric around 50%
    maker_reb  = rebateRate * feeRate * shares * p * (1-p)
                 # rebateRate=0.20. It collapses to our own fee_equiv because each taker
                 # fill is matched 1:1 with a maker fill of the same size/price, so the
                 # whole rebate pool is 0.20*taker-fees and our share is my_fee/total.

Pure-passive assumption (ponytail): we only ever fill as a *maker* (our resting limit is
lifted by a taker), so we never pay taker fee and never suffer slippage. Those only enter
once M4 actively de-risks (buying/selling into the book) -- deferred, clearly noted.

P&L decomposition (exact, no double count):
    d(equity) = d(cash) + q*d(fair) + fair*d(q),  d(cash) = -fill_price*d(q)
    spread_income += (requote_fair - fill_price)*d(q)   # quote capture
    inv_mtm       += q * d(fair)                        # adverse selection / carry
    rebate        += maker_reb per fill
    headline total = cash_final + q_final*settle + rebate == spread+inv_mtm+rebate.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
import urllib.request, urllib.parse
from datetime import datetime, timezone
from runlog import Run

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GAMMA = "https://gamma-api.polymarket.com/events"
PRICES = "https://clob.polymarket.com/prices-history"
OKX = "https://www.okx.com/api/v5/market/history-candles"

# ---- fee model (crypto_fees_v2) -------------------------------------------------
FEE_RATE = 0.07
REBATE_RATE = 0.20


def taker_fee(shares, p):
    return FEE_RATE * shares * p * (1 - p)


def maker_rebate(shares, p):
    return REBATE_RATE * FEE_RATE * shares * p * (1 - p)


# ---- tiny HTTP helper -----------------------------------------------------------
def http_json(url, retries=4, timeout=40):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pm-mm-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.6 * (a + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


# ---- M1: fetch + local cache (replayable) ---------------------------------------
def _cache(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def fetch_event(slug, force=False):
    p = _cache(f"event_{slug}.json")
    if not force and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    ev = http_json(f"{GAMMA}?slug={urllib.parse.quote(slug)}")[0]
    with open(p, "w") as f:
        json.dump(ev, f)
    return ev


def fetch_prices(token_id, fidelity, start_ts, force=False):
    p = _cache(f"prices_{token_id[:16]}_{fidelity}m.csv")
    if not force and os.path.exists(p):
        return _read_prices(p)
    rows = _read_prices_from_json(http_json(f"{PRICES}?market={token_id}&fidelity={fidelity}&startTs={start_ts}"))
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "p"])
        w.writerows(rows)
    return rows


def _read_prices(path):
    with open(path) as f:
        return [(int(r["ts"]), float(r["p"])) for r in csv.DictReader(f)]


def _read_prices_from_json(data):
    return [(int(h["t"]), float(h["p"])) for h in data.get("history", [])]


def fetch_okx(start_ts, end_ts, force=False):
    # cache key carries the window: a single shared file silently served the wrong regime context
    p = _cache("okx_btc_1h_%d_%d.csv" % (start_ts, end_ts))
    if not force and os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p) as f:
            return sorted((int(r[0]), float(r[1])) for r in csv.reader(f))
    # OKX `after` returns records OLDER than the ts (newest-first). Page backwards from
    # just past the window end down to the window start. Hard cap on pages as a safety net.
    rows = {}
    after = (end_ts + 2 * 3600) * 1000
    for _ in range(80):
        got = http_json(f"{OKX}?instId=BTC-USDT&bar=1H&after={after}&limit=100").get("data", [])
        if not got:
            break
        for c in got:
            rows[int(c[0]) // 1000] = float(c[4])  # candle [ts,o,h,l,close,...]
        earliest = min(int(c[0]) for c in got) // 1000
        if earliest <= start_ts or len(got) < 100:
            break
        after = earliest * 1000
    rows = sorted((t, c) for t, c in rows.items() if start_ts <= t <= end_ts)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    return rows


# ---- market metadata ------------------------------------------------------------
def market_from_event(ev):
    out = []
    for m in ev["markets"]:
        tok = json.loads(m["clobTokenIds"])
        end = int(datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp())
        out.append({
            "strike": int(m["groupItemTitle"].replace(",", "").replace("$", "")),
            "token": tok[0],
            "end_ts": end,
            "yes_win": float(json.loads(m["outcomePrices"])[0]),  # settlement value of YES
            "tick": float(m.get("orderPriceMinTickSize", 0.001)),
            "min_size": float(m.get("orderMinSize", 5)),
        })
    out.sort(key=lambda x: x["strike"])
    return out


# ---- M3 quote / M5 paper match ---------------------------------------------------
def simulate(prices, yes_win, h, size, refresh_min, max_pos, tick,
             inv_skew=0.0, inv_cap=1.0, trend_block=False, trend_window=12, trend_eps=0.02):
    """Naive two-sided MM + optional M4 inventory control + M2 trend filter.

    inv_skew: quoted-mid shifts by -inv_skew*q (q in shares); long -> quotes move down
              (sell faster / stop buying), short -> up. Soft, spreads across all q.
    inv_cap:  hard inventory bias, fraction of max_pos. Beyond +cap*max_pos we stop
              quoting the bid (no more buys); beyond -cap*max_pos stop the ask.
    trend_block: M2 state filter -- if the price has moved down >= trend_eps over the
              last trend_window intervals, suppress the bid (refuse to buy into a
              falling knife); if up >= trend_eps, suppress the ask.
    All default to pure-naive (skew=0, cap=1.0, trend_block off).
    """
    refresh = refresh_min * 60
    q = cash = rebate = spread_income = 0.0   # spread_income = *intended* half-spread capture
    inv_mtm = 0.0                              # inventory marked-to-market carry
    gap = 0.0                                  # Σ (p_i - req_fair)*dq = move that erased the capture
    bid = ask = req_fair = None
    last_req = None
    daily = {}
    fills = 0
    equity_peak = float("-inf")
    max_dd = 0.0

    def stamp(day_ts, sp, mt, rb):
        d = daily.setdefault(day_ts, {"spread": 0.0, "mtm": 0.0, "rebate": 0.0})
        d["spread"] += sp
        d["mtm"] += mt
        d["rebate"] += rb

    prev_p = prices[0][1]
    for i in range(1, len(prices)):
        ts, p = prices[i]
        carry = q * (p - prev_p)              # inventory MtM during the move [i-1, i]
        inv_mtm += carry
        stamp(ts // 86400 * 86400, 0.0, carry, 0.0)
        prev_p = p
        if last_req is None or (ts - last_req) >= refresh:
            req_fair = p
            shift = -inv_skew * q               # inventory reversion skew (price units)
            bid = _price(req_fair - h + shift)
            ask = _price(req_fair + h + shift)
            if trend_block:
                back = max(0, i - trend_window)
                change = p - prices[back][1]
                if change <= -trend_eps:
                    bid = None                  # falling knife: refuse to buy
                elif change >= trend_eps:
                    ask = None                  # rising spike: refuse to sell
            last_req = ts
        # we are the passive maker: our resting limit is lifted by a taker
        can_bid = q + size <= inv_cap * max_pos
        can_ask = q - size >= -inv_cap * max_pos
        if bid is not None and p <= bid and can_bid:
            reb = maker_rebate(size, bid)
            sp = (req_fair - bid) * size       # intended half-spread capture
            gap += (p - req_fair) * size        # price ran below our requote fair -> adverse
            cash -= size * bid
            q += size
            rebate += reb
            spread_income += sp
            stamp(ts // 86400 * 86400, sp, 0.0, reb)
            fills += 1
            bid = None
        if ask is not None and p >= ask and can_ask:
            reb = maker_rebate(size, ask)
            sp = (ask - req_fair) * size
            gap += (p - req_fair) * (-size)     # price ran above requote fair -> adverse
            cash += size * ask
            q -= size
            rebate += reb
            spread_income += sp
            stamp(ts // 86400 * 86400, sp, 0.0, reb)
            fills += 1
            ask = None
        eq = cash + q * p + rebate
        equity_peak = max(equity_peak, eq)
        max_dd = max(max_dd, equity_peak - eq)

    final_jump = q * (yes_win - prev_p)        # settle remaining inventory
    inv_mtm += final_jump
    stamp(prices[-1][0] // 86400 * 86400, 0.0, final_jump, 0.0)
    cash += q * yes_win
    return {
        "total_pnl": cash + rebate,
        "spread_income": spread_income,
        "adverse_selection": inv_mtm + gap,     # = total - spread - rebate (exact)
        "inv_mtm": inv_mtm,
        "rebate": rebate,
        "fills": fills,
        "max_drawdown": max_dd,
        "daily": daily,
    }


def _price(x):
    return round(round(x / 0.001, 6) * 0.001, 9)  # venue tick = 0.001


# ---- per-strike aggregation -----------------------------------------------------
def strike_stats(prices, res, yes_win):
    ps = [p for _, p in prices]
    daily_tot = sorted((k, d["spread"] + d["mtm"] + d["rebate"]) for k, d in res["daily"].items())
    wins = sum(1 for _, v in daily_tot if v > 0)
    return {
        "avg_daily": sum(v for _, v in daily_tot) / len(daily_tot) if daily_tot else 0.0,
        "win_rate": wins / len(daily_tot) if daily_tot else 0.0,
        "best_daily": max((v for _, v in daily_tot), default=0.0),
        "worst_daily": min((v for _, v in daily_tot), default=0.0),
        "min_p": min(ps), "max_p": max(ps), "final_p": ps[-1],
        "yes_win": yes_win,
    }


def fmt(x, nd=3):
    return f"{x:+,.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="naive polymarket crypto-MM paper sim")
    ap.add_argument("--slug", default="bitcoin-above-on-july-29-2026")
    ap.add_argument("--fidelity", type=int, default=10)
    ap.add_argument("--half-spread", default="0.01", help="comma-list of h")
    ap.add_argument("--size", type=float, default=100)
    ap.add_argument("--refresh", type=int, default=30, help="re-quote interval, min")
    ap.add_argument("--max-pos", type=float, default=1000)
    ap.add_argument("--inv-skew", type=float, default=0.0, help="M4: mid shift per share of inventory (price units)")
    ap.add_argument("--inv-cap", type=float, default=1.0, help="M4: stop quoting the accumulating side beyond this * max_pos")
    ap.add_argument("--trend-block", action="store_true", help="M2: suppress bid on downtrend / ask on uptrend")
    ap.add_argument("--trend-window", type=int, default=12, help="M2: lookback intervals for trend_block")
    ap.add_argument("--trend-eps", type=float, default=0.02, help="M2: move (price units) that counts as a trend")
    ap.add_argument("--fetch", action="store_true", help="force refetch (bypass cache)")
    args = ap.parse_args()

    ev = fetch_event(args.slug, force=args.fetch)
    markets = market_from_event(ev)
    active_min = int(datetime.fromisoformat(ev["startDate"].replace("Z", "+00:00")).timestamp())
    active_max = max(m["end_ts"] for m in markets)
    for m in markets:
        m["prices"] = fetch_prices(m["token"], args.fidelity, active_min - 3600, force=args.fetch)

    hs = [float(x) for x in args.half_spread.split(",")]
    print(f"event={args.slug}  strikes={[m['strike'] for m in markets]}  fidelity={args.fidelity}m")
    print(f"size={args.size} refresh={args.refresh}min max_pos={args.max_pos} fee=0.07 rebate=0.20")

    # BTC context (M1): classify the week's regime + supply for the M2 vol band later
    btc = fetch_okx(active_min - 86400, active_max + 86400, force=args.fetch)
    if btc:
        b0, b1 = btc[0][1], btc[-1][1]
        print(f"btc {b0:.0f} -> {b1:.0f}  chg={b1-b0:+.0f} ({(b1/b0-1)*100:+.1f}%)")

    print("strike yes_win   px_range   fills  avg_daily  win%   best    worst    maxDD   spread    rebate   adverse    total")
    rows = []
    for m in markets:
        for h in hs:
            res = simulate(m["prices"], m["yes_win"], h, args.size, args.refresh,
                           args.max_pos, m["tick"], args.inv_skew, args.inv_cap,
                           args.trend_block, args.trend_window, args.trend_eps)
            st = strike_stats(m["prices"], res, m["yes_win"])
            print(
                f"{m['strike']:>6} {int(m['yes_win']):>3} "
                f"{st['min_p']:.3f}-{st['max_p']:.3f} {res['fills']:>6} "
                f"{fmt(st['avg_daily']):>9} {st['win_rate']*100:>4.0f}% "
                f"{fmt(st['best_daily']):>8} {fmt(st['worst_daily']):>8} "
                f"{(res['max_drawdown']/args.max_pos):>8.3f} "
                f"{fmt(res['spread_income']):>9} {fmt(res['rebate']):>9} {fmt(res['adverse_selection']):>9} "
                f"{fmt(res['total_pnl']):>9}"
            )
            rows.append({
                "strike": m["strike"], "yes_win": m["yes_win"], "half_spread": h, "fills": res["fills"],
                "avg_daily": round(st["avg_daily"], 3), "win_rate": round(st["win_rate"], 3),
                "best_daily": round(st["best_daily"], 3), "worst_daily": round(st["worst_daily"], 3),
                "max_dd_over_maxpos": round(res["max_drawdown"] / args.max_pos, 3),
                "spread_income": round(res["spread_income"], 3), "rebate": round(res["rebate"], 3),
                "adverse_selection": round(res["adverse_selection"], 3), "total_pnl": round(res["total_pnl"], 3),
            })
            if len(hs) > 1:
                print(f"      h={h:g}")

    # aggregate
    all_tot = sum(simulate(m["prices"], m["yes_win"], hs[0], args.size, args.refresh,
                           args.max_pos, m["tick"], args.inv_skew, args.inv_cap,
                           args.trend_block, args.trend_window, args.trend_eps)["total_pnl"]
                  for m in markets)
    print(f"\nsum total_pnl over all strikes (h={hs[0]}, inv_skew={args.inv_skew}, inv_cap={args.inv_cap}): {fmt(all_tot)}")
    return {"event": args.slug, "strikes": rows, "aggregate_total_pnl": round(all_tot, 3),
            "half_spread": hs[0], "size": args.size, "refresh_min": args.refresh,
            "max_pos": args.max_pos, "inv_skew": args.inv_skew, "inv_cap": args.inv_cap,
            "trend_block": args.trend_block, "fee_rate": FEE_RATE, "rebate_rate": REBATE_RATE}


def demo():
    # widen the oscillation so price genuinely crosses our 0.49/0.51 levels -> real fills
    t0 = 1784736000
    prices = [(t0 + i * 600, 0.50 + 0.03 * math.sin(i / 2.5) - 0.0006 * i) for i in range(288)]
    res = simulate(prices, 0.0, 0.01, 100, 30, 1000, 0.001)
    lhs = res["spread_income"] + res["adverse_selection"] + res["rebate"]
    assert res["fills"] > 0, "demo must test a real fill path"
    assert abs(lhs - res["total_pnl"]) < 1e-6, f"decompose mismatch {lhs} vs {res['total_pnl']}"
    assert res["rebate"] >= 0, "maker rebate must be non-negative"
    print(f"demo OK: total={res['total_pnl']:+.3f} spread={res['spread_income']:+.3f} "
          f"adverse={res['adverse_selection']:+.3f} rebate={res['rebate']:+.3f} fills={res['fills']}")


if __name__ == "__main__":
    if "--demo" in sys.argv or os.environ.get("PM_DEMO"):
        demo()
    else:
        run = Run("pm_mm_sim", {"argv": sys.argv[1:]})
        with run.capture("PM MM paper sim"):
            summary = main()
        rel = os.path.relpath(DATA, os.getcwd())
        for pat in ("event_*.json", "prices_*.csv", "okx_btc_1h*.csv"):
            run.source_glob(os.path.join(rel, pat))
        for row in summary["strikes"]:
            run.csv_append(os.path.join(run.dir, "strikes.csv"), row)
        run.finish(summary)

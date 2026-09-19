---
run_id: "20260910T021951Z_arb_scan"
name: "arb_scan"
started_utc: "2026-09-10T02:19:51+00:00"
---

# ARB scan sample 1/2 (limit_events=15)

events=15  markets=609  tokens=1218
books fetched=476

A. BINARY COMPLEMENT net-arb hits: 0  (n=298 priced markets)
   GROSS edge (pre-fee):  min=-0.0400  p10=-0.0080  median=-0.0010  max=-0.0010  [>0 = raw dislocation]
   NET   edge (post-fee): min=-0.0606  p10=-0.0200  median=-0.0017  max=-0.0010  [>0 = arbitrage]

B. negRisk SUM-of-buckets net-arb hits: 0  (n=2 events)
   tightness (sum_ask - 1 + fees): min=0.0255  p10=0.0255  median=0.0489  max=0.0489

C. NESTED-STRIKE monotonicity violations (>1c): 0


# ARB scan sample 2/2 (limit_events=15)

events=15  markets=609  tokens=1218
books fetched=476

A. BINARY COMPLEMENT net-arb hits: 0  (n=298 priced markets)
   GROSS edge (pre-fee):  min=-0.0400  p10=-0.0080  median=-0.0010  max=-0.0010  [>0 = raw dislocation]
   NET   edge (post-fee): min=-0.0606  p10=-0.0200  median=-0.0017  max=-0.0010  [>0 = arbitrage]

B. negRisk SUM-of-buckets net-arb hits: 0  (n=2 events)
   tightness (sum_ask - 1 + fees): min=0.0255  p10=0.0255  median=0.0489  max=0.0489

C. NESTED-STRIKE monotonicity violations (>1c): 0


---
run_id: "20260910T023220Z_calibration_bycat"
name: "calibration_bycat"
started_utc: "2026-09-10T02:32:20+00:00"
---

# calibration by category x horizon

resolved markets (vol>=3k, endDate>=2025): 2780
with price history: 824
category mix: {'Politics': 799, 'Sports': 773, 'Crypto': 764, 'Tech': 213, 'Culture': 102, 'Geopolitics': 77, 'Weather': 22, 'Other': 11, 'Business': 9, 'Economics': 6, 'Finance': 4}

========================================================================================================
HORIZON T-1d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Tech            48   0.285   0.292  -0.007  0.097 |                    26   0.046   0.000     +0.046
Politics        78   0.167   0.218  -0.051  0.070 |                    59   0.029   0.017     +0.012
Sports         130   0.427   0.415  +0.012  0.225 | (too few longshots)

========================================================================================================
HORIZON T-7d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Sports         332   0.471   0.491  -0.020  0.228 |                    21   0.090   0.048     +0.043
Tech            49   0.320   0.367  -0.047  0.070 |                    23   0.036   0.000     +0.036
Politics        66   0.227   0.167  +0.060  0.067 |                    44   0.063   0.045     +0.018
Geopolitics     32   0.419   0.500  -0.081  0.112 |                    17   0.085   0.176     -0.092

========================================================================================================
HORIZON T-30d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Tech            37   0.401   0.378  +0.022  0.139 |                    10   0.072   0.000     +0.072
Sports          29   0.219   0.207  +0.012  0.107 |                    17   0.071   0.059     +0.012
Politics        50   0.313   0.280  +0.033  0.194 |                    18   0.083   0.222     -0.139
Geopolitics     32   0.531   0.719  -0.188  0.165 | (too few longshots)

========================================================================================================
HORIZON T-60d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Politics        31   0.343   0.484  -0.141  0.198 | (too few longshots)



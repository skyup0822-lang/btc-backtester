---
run_id: "20260910T095426Z_calibration_bycat"
name: "calibration_bycat"
started_utc: "2026-09-10T09:54:26+00:00"
---

# calibration by category x horizon

resolved markets (vol>=3k, endDate>=2025): 2931
with price history: 2930
category mix: {'Politics': 857, 'Sports': 842, 'Crypto': 783, 'Tech': 218, 'Culture': 102, 'Geopolitics': 77, 'Weather': 22, 'Other': 11, 'Business': 9, 'Economics': 6, 'Finance': 4}

========================================================================================================
HORIZON T-1d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Culture        102   0.287   0.137  +0.150  0.102 |                    58   0.044   0.000     +0.044
Crypto         783   0.389   0.392  -0.003  0.035 |                   405   0.021   0.005     +0.016
Tech           218   0.465   0.482  -0.017  0.067 |                    86   0.024   0.012     +0.012
Sports         841   0.358   0.377  -0.019  0.159 |                   248   0.016   0.004     +0.012
Politics       856   0.198   0.221  -0.023  0.050 |                   612   0.019   0.018     +0.001
Geopolitics     75   0.532   0.747  -0.215  0.135 |                    24   0.037   0.208     -0.172

========================================================================================================
HORIZON T-7d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Culture        100   0.337   0.130  +0.207  0.152 |                    37   0.069   0.000     +0.069
Crypto         657   0.401   0.393  +0.008  0.100 |                   259   0.040   0.023     +0.017
Sports         302   0.204   0.182  +0.021  0.075 |                   191   0.011   0.010     +0.001
Politics       705   0.180   0.210  -0.030  0.069 |                   497   0.025   0.030     -0.005
Tech           129   0.464   0.519  -0.055  0.135 |                    33   0.043   0.061     -0.018
Geopolitics     60   0.323   0.683  -0.361  0.265 |                    23   0.072   0.261     -0.189

========================================================================================================
HORIZON T-30d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Tech            49   0.404   0.408  -0.004  0.090 |                    16   0.040   0.000     +0.040
Sports         183   0.171   0.126  +0.045  0.057 |                   129   0.027   0.008     +0.020
Politics       418   0.178   0.196  -0.018  0.082 |                   278   0.027   0.025     +0.002
Crypto         351   0.365   0.407  -0.042  0.120 |                   146   0.055   0.089     -0.034

========================================================================================================
HORIZON T-60d
category       n   mean_p realised   bias   Brier | LONGSHOT p<0.15:  n  mean_p realised   EDGE(=bias, >0 => buy NO earns)
Sports         145   0.178   0.117  +0.061  0.062 |                    94   0.037   0.021     +0.016
Politics       296   0.150   0.149  +0.002  0.074 |                   207   0.029   0.014     +0.015
Crypto         191   0.355   0.372  -0.017  0.132 |                    79   0.050   0.101     -0.052
Tech            25   0.383   0.400  -0.017  0.099 | (too few longshots)



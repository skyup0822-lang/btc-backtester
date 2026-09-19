---
run_id: "20260910T022137Z_weather_calibrate"
name: "weather_calibrate"
started_utc: "2026-09-10T02:21:37+00:00"
---

# weather_calibrate

calibration: 3 days (HKO SYNOP vs resolved bucket, truncation convention)
date        hkoMax hkTime  winner        fromMax  match  f  winP(mid)  winP(close)
  2026-06-15   31.1 12:00  29C           31C      n     0.01    0.515      1.000
  2026-06-18   29.5 10:00  29C           29C      Y     0.01    0.295      1.000
  2026-06-21   32.1 13:00  33C           32C      n     0.01    0.355      1.000
HKO-SYNOP bucket == resolved winner: 1/3


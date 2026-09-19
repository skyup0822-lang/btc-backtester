# -*- coding: utf-8 -*-
"""poly_paper 逻辑自测(合成数据): edge+Kelly+记账 是否能跑。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

import numpy as np
import pandas as pd

from poly_sim import realized_vol, model_prob_above
from poly_paper import compute_and_log, Cfg

rng = np.random.default_rng(1)
idx = pd.date_range("2024-01-01", periods=30000, freq="h", tz="UTC")
rets = rng.normal(0.00004, 0.012, len(idx))
close = pd.Series(10_000 * np.exp(np.cumsum(rets)), index=idx)

mk = {
    "id": "1", "question": "Will Bitcoin be above $135000 by 2026-08-01?",
    "threshold": 135000, "direction": "above", "end": "2026-08-01",
    "yes_price": 0.15, "end_dt": idx[-1] + pd.Timedelta(days=14),
}
r = compute_and_log(Cfg(bankroll=140), float(close.iloc[-1]), close, mk)
print("p_model=%.1f%% yes_price=%.1f%% edge=%+.1f%% side=%s bet=%.2f USDC"
      % (r["p_model"] * 100, r["yes_price"] * 100, r["edge"] * 100, r["side"], r["bet_usd"]))
assert abs(r["edge"]) > 0, "应检测到edge"
print("OK: edge/Kelly/记账逻辑可跑")

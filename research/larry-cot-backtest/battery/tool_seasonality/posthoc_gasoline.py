#!/usr/bin/env python3
"""
POST-HOC diagnostic (NOT pre-registered): where does the gasoline-spot seasonal edge come from,
and does it survive on gasoline futures?

The primary gasoline signal (estimated on NY Harbor gasoline spot, L = 15, th = 70, h = 8) is
broken down by calendar month of the decision date. For each month the script reports how many
setup weeks were long and short, and the mean signed 8-week return of those weeks on two series:
  * gas_spot: the series the signal was learned from and evaluated on;
  * psys:gasoline: the roll-adjusted futures index (tradable), over the weeks where both exist.
Writes ROOT/work2/tool_seasonality/posthoc_gasoline.json. Run: python3 -I posthoc_gasoline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))   # seas_data, seasonal
import seas_data as sd  # noqa: E402
import seasonal as se   # noqa: E402

fh = se.fh
HERE = fh.WORK2 / "tool_seasonality"           # outputs: ROOT/work2/tool_seasonality
HERE.mkdir(parents=True, exist_ok=True)
FRI = fh.weekly_decision_dates("1986-01-03", "2026-10-16")
spot = fh.load_series("gas_spot")
fut = sd.tradable_series("gasoline")
sig = se.seasonal_signal(FRI, se.est_table(spot, "gas_spot", (8,)), 8, 15, 70)

ft_s = fh.forward_table(FRI, spot, "price", (8,))
ft_f = fh.forward_table(FRI, fut, "price", (8,))
df = pd.DataFrame(dict(date=FRI, sig=sig, r_spot=ft_s["r8"].to_numpy(),
                       r_fut=ft_f["r8"].to_numpy()))
df = df[df["sig"].notna() & df["r_spot"].notna()]
base_spot = float(df["r_spot"].mean())                 # baseline: all eligible weeks
both = df["r_fut"].notna()
base_fut = float(df.loc[both, "r_fut"].mean())
act = df[df["sig"] != 0].copy()
act["month"] = act["date"].dt.month
act["signed_spot"] = act["sig"] * act["r_spot"]
act["signed_fut"] = act["sig"] * act["r_fut"]

rows = []
for mth, g in act.groupby("month"):
    gb = g[g["r_fut"].notna()]
    rows.append(dict(month=int(mth), n_long=int((g["sig"] > 0).sum()),
                     n_short=int((g["sig"] < 0).sum()),
                     mean_signed_spot=float(g["signed_spot"].mean()),
                     n_with_futures=int(len(gb)),
                     mean_signed_spot_same_weeks=float(gb["signed_spot"].mean()) if len(gb) else None,
                     mean_signed_futures=float(gb["signed_fut"].mean()) if len(gb) else None))
cmp_weeks = act[act["r_fut"].notna()]
out = dict(
    note="POST-HOC, not pre-registered. Primary gasoline signal (learned on gas_spot) by month.",
    baseline_gas_spot_all_eligible_weeks=base_spot,
    baseline_futures_weeks_with_both=base_fut,
    setup_weeks=int(len(act)), setup_weeks_with_futures=int(len(cmp_weeks)),
    mean_signed_spot_on_those_weeks=float(cmp_weeks["signed_spot"].mean()),
    mean_signed_futures_on_those_weeks=float(cmp_weeks["signed_fut"].mean()),
    by_month=rows,
)
(HERE / "posthoc_gasoline.json").write_text(json.dumps(fh.to_jsonable(out), indent=1))
print(json.dumps(fh.to_jsonable(out), indent=1))

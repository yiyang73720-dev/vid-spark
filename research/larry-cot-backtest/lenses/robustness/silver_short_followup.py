#!/usr/bin/env python3
"""
Follow-up on the only recurring positive pocket: SILVER, SHORT side.

Silver-short is where all 8 canonical cells with p < 0.05 sit, and where all 9 BH survivors of
the comm_net/oi variant (f) sit. This script checks whether that pocket is stable in time or
driven by a few years:

  1. the 36 silver-short cells (3 lookbacks x 3 thresholds x 4 horizons) per index variant
     (comm_net = canonical, comm_net/oi = variant f), counted in P1 = 2006-2015 and
     P2 = 2016-2026 (p < 0.05 with positive excess; baseline and permutation within period);
  2. three focal cells (canonical primary N26 80/20 h8; f primary N26 80/20 h8;
     f N26 75/25 h8, the strongest f cell) per sub-period, with exhaustive p;
  3. per-calendar-year mean signed 8w return of the focal f cell and a leave-one-year-out
     range of its full-sample excess.

Outputs: silver_short_followup.json, silver_short_by_year.csv
Run    : python3 -I silver_short_followup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import LENS_DIR, add_oi_normalised_index, lc, subperiod_panel, write_json  # noqa: E402

PERIODS = {"P1_2006_2015": (None, "2015-12-31"), "P2_2016_2026": ("2016-01-01", None),
           "full": (None, None)}
FOCAL = [("canonical", "idx26", 80, 20), ("f_oi", "oiidx26", 80, 20),
         ("f_oi", "oiidx26", 75, 25)]


def main() -> None:
    base = lc.build_panel(markets=("silver",))
    panel = add_oi_normalised_index(base)          # has idx* and oiidx* columns
    out = {"cell_counts": {}, "focal": [], "loyo": {}}

    # 1. silver-short cell counts by period and index variant
    for pname, (a, b) in PERIODS.items():
        f = subperiod_panel(panel, a, b)["silver"]
        for vname, prefix in (("canonical", "idx"), ("f_oi", "oiidx")):
            n_flag = n_pos = 0
            for n in lc.LOOKBACKS:
                for bt, st in lc.THRESHOLDS:
                    for h in lc.HORIZONS_W:
                        s = lc.cell_stats(f, f"{prefix}{n}", bt, st, "short", f"r{h}")
                        n_pos += int(s["excess"] > 0)
                        n_flag += int(s["excess"] > 0 and s["perm_p"] < 0.05)
            out["cell_counts"][f"{vname}|{pname}"] = dict(n_cells=36, n_pos_excess=n_pos,
                                                          n_p_lt_05_pos_excess=n_flag)

    # 2. focal cells by period
    for pname, (a, b) in PERIODS.items():
        f = subperiod_panel(panel, a, b)["silver"]
        for vname, col, bt, st in FOCAL:
            s = lc.cell_stats(f, col, bt, st, "short", "r8", exact_p=True)
            out["focal"].append(dict(period=pname, variant=vname, idx_col=col, buy_th=bt,
                                     sell_th=st, horizon_w=8, **s))

    # 3. per-year + leave-one-year-out for the focal cells (full sample)
    f = panel["silver"]
    yr = f["report_date"].dt.year.to_numpy()
    by_year = []
    for vname, col, bt, st in FOCAL:
        a = lc.cell_arrays(f, col, bt, st, "short", "r8")
        yr_e = yr[a["elig"]]
        sig, signed = a["sig"], a["signed"]
        base_all = float(a["r"].mean())
        for y in np.unique(yr_e):
            m = sig & (yr_e == y)
            by_year.append(dict(variant=vname, idx_col=col, buy_th=bt, year=int(y),
                                n_signal_weeks=int(m.sum()),
                                mean_signed_r8=float(signed[m].mean()) if m.any() else np.nan,
                                year_mean_r8=float(a["r"][yr_e == y].mean())))
        # leave one calendar year out: drop that year's eligible weeks entirely
        loyo = {}
        for y in np.unique(yr_e):
            keep = yr_e != y
            r_k, s_k, sig_k = a["r"][keep], signed[keep], sig[keep]
            loyo[int(y)] = float(s_k[sig_k].mean() + r_k.mean())   # short: excess = mean + base
        vals = np.array(list(loyo.values()))
        out["loyo"][f"{vname}|{col}|{bt}/{st}"] = dict(
            full_excess=float(signed[sig].mean() + base_all),
            loyo_min=float(vals.min()), loyo_min_year=int(list(loyo)[int(vals.argmin())]),
            loyo_max=float(vals.max()), loyo_max_year=int(list(loyo)[int(vals.argmax())]),
            n_years_positive_mean_signed=int(sum(1 for r in by_year
                                                 if r["variant"] == vname and r["idx_col"] == col
                                                 and r["buy_th"] == bt
                                                 and r["n_signal_weeks"] > 0
                                                 and r["mean_signed_r8"] > 0)),
            n_years_with_signals=int(sum(1 for r in by_year
                                         if r["variant"] == vname and r["idx_col"] == col
                                         and r["buy_th"] == bt and r["n_signal_weeks"] > 0)))
    pd.DataFrame(by_year).to_csv(LENS_DIR / "silver_short_by_year.csv", index=False)
    write_json(out, LENS_DIR / "silver_short_followup.json")

    pd.set_option("display.width", 250)
    print(pd.DataFrame(out["cell_counts"]).T.to_string())
    print(pd.DataFrame(out["focal"])[["period", "variant", "idx_col", "buy_th",
                                      "n_signal_weeks", "n_episodes", "mean_ret", "excess",
                                      "hit_rate", "perm_p", "perm_p_exact", "T"]].to_string())
    print(pd.DataFrame(out["loyo"]).T.to_string())
    print(pd.DataFrame(by_year).query("variant=='f_oi' and buy_th==75").to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
(c) Sub-period stability: 2006-2015 vs 2016-2026.

Split by report_date:  P1 = 2006-06-13 .. 2015-12-31,  P2 = 2016-01-01 .. 2026-09-29.
The indices are the full-history ones (rolling 26/52/156 reports, so no look-ahead), the
baseline, excess and circular-shift permutation p are recomputed WITHIN each sub-period
(T = eligible weeks of that sub-period; K=2000, seed 11 via lc.cell_stats / lc.pooled_cell).

Parts
  1. Primary cell (N=26, 80/20, h=8), long and short, per market and pooled, in each half.
  2. Whole-grid out-of-sample consistency: run the 360-cell grid separately in each half and
     ask whether cells that look good in P1 still look good in P2 (rank correlation of excess,
     sign agreement, and the P2 performance of cells "discovered" in P1 at p < 0.05).
  3. Crude calendar-year breakdown of the primary cell, plus the primary crude cell with
     2026 removed (the year of the rally that motivated the question).

Outputs: c_subperiods_primary.csv, c_grid_P1.csv, c_grid_P2.csv, c_crude_by_year.csv,
         c_subperiods.json
Run    : python3 -I c_subperiods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import (KEY_COLS, LENS_DIR, K, SEED, lc, primary_rows, subperiod_panel,  # noqa: E402
                    write_json)

PERIODS = {"P1_2006_2015": (None, "2015-12-31"), "P2_2016_2026": ("2016-01-01", None)}


def crude_by_year(f: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year (of report_date): primary long/short signal weeks and their mean
    signed 8w return, next to that year's unconditional mean 8w return (eligible weeks)."""
    rows = []
    elig = lc.eligible_mask(f, "idx26", "r8")
    lo, sh = lc.signal_masks(f["idx26"].to_numpy(), 80, 20)
    r8 = f["r8"].to_numpy()
    yr = f["report_date"].dt.year.to_numpy()
    for y in np.unique(yr):
        e = elig & (yr == y)
        if not e.any():
            continue
        rows.append(dict(year=int(y), n_eligible=int(e.sum()),
                         year_mean_r8=float(r8[e].mean()),
                         n_long=int((e & lo).sum()),
                         long_mean_r8=float(r8[e & lo].mean()) if (e & lo).any() else np.nan,
                         n_short=int((e & sh).sum()),
                         short_mean_signed_r8=float(-r8[e & sh].mean()) if (e & sh).any()
                         else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    panel = lc.build_panel()

    # ---- 1. primary cell per half ------------------------------------------------------
    rows = []
    for name, (a, b) in PERIODS.items():
        sub = subperiod_panel(panel, a, b)
        rows += primary_rows(sub, label=name)
    rows += primary_rows(panel, label="full_2006_2026")
    prim = pd.DataFrame(rows)
    prim.to_csv(LENS_DIR / "c_subperiods_primary.csv", index=False)

    # ---- 2. grid out-of-sample consistency --------------------------------------------
    grids = {}
    for name, (a, b) in PERIODS.items():
        sub = subperiod_panel(panel, a, b)
        g = lc.canonical_table(sub, k=K, seed=SEED)
        g.to_csv(LENS_DIR / f"c_grid_{name.split('_')[0]}.csv", index=False)
        grids[name] = g
    g1, g2 = grids["P1_2006_2015"], grids["P2_2016_2026"]
    assert (g1[KEY_COLS].to_numpy() == g2[KEY_COLS].to_numpy()).all()
    ok = g1.excess.notna() & g2.excess.notna()
    rho, rho_p = stats.spearmanr(g1.excess[ok], g2.excess[ok])
    # excess per week removes the mechanical horizon scaling before correlating
    rho_w, _ = stats.spearmanr(g1.excess[ok] / g1.horizon_w[ok], g2.excess[ok] / g2.horizon_w[ok])
    sign_agree = float(np.mean(np.sign(g1.excess[ok]) == np.sign(g2.excess[ok])))
    disc = (g1.perm_p < 0.05) & (g1.excess > 0)
    disc_tab = pd.concat([g1.loc[disc, KEY_COLS + ["excess", "perm_p"]].reset_index(drop=True)
                          .rename(columns={"excess": "excess_P1", "perm_p": "perm_p_P1"}),
                          g2.loc[disc, ["excess", "perm_p"]].reset_index(drop=True)
                          .rename(columns={"excess": "excess_P2", "perm_p": "perm_p_P2"})],
                         axis=1)
    disc2 = (g2.perm_p < 0.05) & (g2.excess > 0)
    grid_consistency = dict(
        n_cells_both_defined=int(ok.sum()),
        spearman_excess_P1_vs_P2=float(rho), spearman_p_value_naive=float(rho_p),
        spearman_excess_per_week_P1_vs_P2=float(rho_w),
        sign_agreement_share=sign_agree,
        P1_n_pos_excess=int((g1.excess > 0).sum()), P2_n_pos_excess=int((g2.excess > 0).sum()),
        P1_n_p_lt_05_pos=int(disc.sum()), P2_n_p_lt_05_pos=int(disc2.sum()),
        P1_discoveries_n_still_pos_in_P2=int((disc_tab.excess_P2 > 0).sum()),
        P1_discoveries_n_p_lt_05_in_P2=int(((disc_tab.perm_p_P2 < 0.05)
                                            & (disc_tab.excess_P2 > 0)).sum()),
        P1_discoveries_mean_excess_P1=float(disc_tab.excess_P1.mean()) if len(disc_tab) else None,
        P1_discoveries_mean_excess_P2=float(disc_tab.excess_P2.mean()) if len(disc_tab) else None,
        P1_discoveries=disc_tab.to_dict("records"),
        P2_discoveries=g2.loc[disc2, KEY_COLS + ["excess", "perm_p"]].to_dict("records"),
        per_market_pos_excess={m: dict(P1=int(((g1.market == m) & (g1.excess > 0)).sum()),
                                       P2=int(((g2.market == m) & (g2.excess > 0)).sum()))
                               for m in lc.MARKETS})

    # ---- 3. crude by year, and crude primary without 2026 ------------------------------
    cy = crude_by_year(panel["crude"])
    cy.to_csv(LENS_DIR / "c_crude_by_year.csv", index=False)
    no26 = subperiod_panel(panel, None, "2025-12-31")
    crude_no2026 = {sd: lc.cell_stats(no26["crude"], "idx26", 80, 20, sd, "r8", exact_p=True)
                    for sd in lc.SIDES}
    only26 = subperiod_panel(panel, "2026-01-01", None)["crude"]
    lo26 = lc.cell_arrays(only26, "idx26", 80, 20, "long", "r8")
    f = panel["crude"]
    lo_all, _ = lc.signal_masks(f["idx26"].to_numpy(), 80, 20)
    recent = f.loc[f.report_date >= "2026-01-01",
                   ["report_date", "idx26", "shutdown", "entry_date", "entry_price", "r8"]]
    recent = recent.assign(long_setup=lo_all[recent.index])
    crude_2026 = dict(
        n_long_signal_weeks_with_r8=int(lo26["sig"].sum()),
        mean_r8_long_signal_weeks=float(lo26["r"][lo26["sig"]].mean())
        if lo26["sig"].any() else None,
        mean_r8_all_eligible_2026=float(lo26["r"].mean()) if len(lo26["r"]) else None,
        long_setup_weeks=recent.loc[recent.long_setup, ["report_date", "idx26", "entry_date",
                                                        "entry_price", "r8"]]
        .to_dict("records"),
        latest_report=recent.iloc[-1][["report_date", "idx26"]].to_dict())

    out = dict(periods={k: [a, b] for k, (a, b) in PERIODS.items()},
               primary=prim.to_dict("records"), grid_consistency=grid_consistency,
               crude_primary_excluding_2026=crude_no2026, crude_2026=crude_2026)
    write_json(out, LENS_DIR / "c_subperiods.json")

    pd.set_option("display.width", 250)
    cols = ["variant", "market", "side", "n_signal_weeks", "n_episodes", "mean_ret",
            "excess", "hit_rate", "perm_p", "perm_p_exact", "T"]
    print(prim[cols].to_string())
    print({k: v for k, v in grid_consistency.items()
           if k not in ("P1_discoveries", "P2_discoveries")})
    print(disc_tab.to_string())
    print(pd.DataFrame(grid_consistency["P2_discoveries"]).to_string())
    print(cy.to_string())
    print("crude excl 2026:", crude_no2026)
    print("crude 2026:", crude_2026)


if __name__ == "__main__":
    main()

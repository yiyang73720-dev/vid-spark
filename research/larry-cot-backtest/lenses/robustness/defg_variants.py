#!/usr/bin/env python3
"""
(d)-(g) Specification variants, each changing ONE thing relative to the canonical run:

  canonical          CL continuous prices, Tuesday-after-next entry, shutdown weeks excluded,
                     index on comm_net                                       (engine default)
  d_crude_eia_spot   crude priced off the EIA Cushing WTI spot series (wti-daily.csv) instead
                     of CL continuous (other markets unchanged)
  e_incl_shutdown    report weeks inside the shutdown publication windows are INCLUDED
  f_comm_net_over_oi Larry index computed on comm_net / oi instead of comm_net
  g_monday_entry     entry on the first trading day >= the Monday after the release Friday
                     (report_date - weekday + 7) instead of the Tuesday (+8).
                     Sensitivity only: in holiday weeks the release slips to Monday 15:30 ET,
                     after the close, so this timing has look-ahead in those weeks.

For each variant:
  * primary cell (N26, 80/20, h=8) long & short, per market + pooled (exact and MC p);
  * the full 360-cell grid (MC p, K=2000, seed 11) and its summary: cells with p<0.05 and
    positive excess, BH(q=0.10) survivors, cells with positive excess, the same count in the
    reverse direction (p from negated returns), and how closely the variant's excess tracks
    the canonical excess cell-by-cell (Spearman, sign agreement).

Plus a variant-FAMILY multiple-testing check: BH(q=0.10) over every distinct test run here
(canonical 360 + spot-priced crude 72 + 3 x 360 = 1512 one-sided p-values), because trying
several specifications and keeping the best one is itself data-snooping.

Outputs: variants_primary.csv, variants_grid_summary.csv, variant_grid_<name>.csv,
         variants_family_bh_survivors.csv, variants.json
Run    : python3 -I defg_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import (LENS_DIR, K, SEED, add_oi_normalised_index, bh_reject,  # noqa: E402
                    grid_summary, lc, monday_entry_panel, primary_rows, write_json)


def negate_returns(panel: dict) -> dict:
    """Copy of the panel with every r{h} negated: the engine's one-sided test on this panel is
    the REVERSE test (setup followed by below-baseline returns)."""
    out = {}
    for m, f in panel.items():
        g = f.copy()
        for h in lc.HORIZONS_W:
            g[f"r{h}"] = -g[f"r{h}"]
        out[m] = g
    return out


def run_variant(name: str, panel: dict, idx_prefix: str, include_shutdown: bool,
                canon_ref: pd.DataFrame | None) -> tuple[list, pd.DataFrame, dict]:
    prim = primary_rows(panel, idx_col=f"{idx_prefix}26", include_shutdown=include_shutdown,
                        label=name)
    grid = lc.canonical_table(panel, k=K, seed=SEED, include_shutdown=include_shutdown,
                              idx_prefix=idx_prefix, exact_p=True)
    rev = lc.canonical_table(negate_returns(panel), k=K, seed=SEED,
                             include_shutdown=include_shutdown, idx_prefix=idx_prefix)
    grid["perm_p_reverse"] = rev["perm_p"].to_numpy()
    # T (eligible weeks) per cell and the "+1" version of the exhaustive p:
    #   perm_p_exact = n_ge / n_off  can be exactly 0; (1 + n_ge) / (1 + n_off) cannot.
    grid["T"] = [len(lc.cell_arrays(panel[r.market], f"{idx_prefix}{r.lookback}", r.buy_th,
                                    r.sell_th, r.side, f"r{r.horizon_w}",
                                    include_shutdown)["sig"]) for r in grid.itertuples()]
    n_off = grid["T"] - 2 * lc.MIN_SHIFT + 1
    n_ge = np.round(grid["perm_p_exact"] * n_off)
    grid["perm_p_exact_plus1"] = (1 + n_ge) / (1 + n_off)
    grid.to_csv(LENS_DIR / f"variant_grid_{name}.csv", index=False)

    def summarise(g: pd.DataFrame, ref: pd.DataFrame | None, scope: str) -> dict:
        s = grid_summary(g, "perm_p")
        sx = grid_summary(g, "perm_p_exact")
        s["n_p_exact_lt_05_pos_excess"] = sx["n_p_lt_alpha_and_pos_excess"]
        s["n_bh_survivors_exact"] = sx["n_bh_survivors"]
        s["min_p_exact"] = sx["min_p"]
        s["n_bh_survivors_exact_plus1"] = int(bh_reject(g.perm_p_exact_plus1.to_numpy(),
                                                        0.10).sum())
        s["scope"] = scope
        s["n_reverse_p_lt_05_neg_excess"] = int(((g.perm_p_reverse < 0.05)
                                                 & (g.excess < 0)).sum())
        s["n_reverse_bh_survivors"] = int(bh_reject(g.perm_p_reverse.to_numpy(), 0.10).sum())
        if ref is not None:
            a, b = g.excess.to_numpy(), ref.excess.to_numpy()
            s["spearman_excess_vs_canonical"] = float(stats.spearmanr(a, b)[0])
            s["sign_agreement_vs_canonical"] = float(np.mean(np.sign(a) == np.sign(b)))
        return s

    summ = [summarise(grid, canon_ref, "all_360")]
    cg = grid[grid.market == "crude"]
    summ.append(summarise(cg, None if canon_ref is None else canon_ref[canon_ref.market == "crude"],
                          "crude_72"))
    for s in summ:
        s["variant"] = name
    return prim, grid, summ


def build_variants() -> dict:
    """{name: (panel, idx_prefix, include_shutdown)} for the canonical run and (d)-(g)."""
    base = lc.build_panel()
    prices = lc.load_prices()
    spot = lc.load_wti_spot()
    variants = {}
    # canonical reference (identical to engine canonical.csv)
    variants["canonical"] = (base, "idx", False)
    # (d) crude on EIA spot
    variants["d_crude_eia_spot"] = (lc.build_panel(price_overrides={"crude": spot}), "idx", False)
    # (e) include shutdown weeks
    variants["e_incl_shutdown"] = (base, "idx", True)
    # (f) comm_net / oi
    variants["f_comm_net_over_oi"] = (add_oi_normalised_index(base), "oiidx", False)
    # (g) Monday entry
    variants["g_monday_entry"] = (monday_entry_panel(base, prices), "idx", False)
    return variants


def main() -> None:
    variants = build_variants()
    base = variants["canonical"][0]
    spot = lc.load_wti_spot()

    all_prim, all_summ, canon_grid, grids = [], [], None, {}
    for name, (panel, prefix, incl) in variants.items():
        prim, grid, summ = run_variant(name, panel, prefix, incl, canon_grid)
        if name == "canonical":
            canon_grid = grid
            ref = pd.read_csv(lc.DEFAULT_ROOT / "work" / "engine" / "canonical.csv",
                              float_precision="round_trip")
            assert np.allclose(grid.perm_p, ref.perm_p) and np.allclose(grid.excess, ref.excess)
        grids[name] = grid.assign(variant=name)
        all_prim += prim
        all_summ += summ

    # ---- variant-family BH over all DISTINCT tests -------------------------------------
    fam = pd.concat([grids["canonical"],
                     grids["d_crude_eia_spot"].query("market == 'crude'"),  # others identical
                     grids["e_incl_shutdown"], grids["f_comm_net_over_oi"],
                     grids["g_monday_entry"]], ignore_index=True)
    fam_rej = bh_reject(fam.perm_p.to_numpy(), 0.10)
    fam_rej_ex = bh_reject(fam.perm_p_exact.to_numpy(), 0.10)
    fam_rej_ex1 = bh_reject(fam.perm_p_exact_plus1.to_numpy(), 0.10)
    fam_rev_rej = bh_reject(fam.perm_p_reverse.to_numpy(), 0.10)
    fam.loc[fam_rej | fam_rej_ex].to_csv(LENS_DIR / "variants_family_bh_survivors.csv",
                                         index=False)
    family = dict(n_tests=int(len(fam)), n_bh_survivors=int(fam_rej.sum()),
                  n_bh_survivors_exact_p=int(fam_rej_ex.sum()),
                  n_bh_survivors_exact_p_plus1=int(fam_rej_ex1.sum()),
                  min_p_exact_plus1=float(fam.perm_p_exact_plus1.min()),
                  n_reverse_bh_survivors=int(fam_rev_rej.sum()),
                  n_p_lt_05_pos_excess=int(((fam.perm_p < 0.05) & (fam.excess > 0)).sum()),
                  expected_under_null=0.05 * len(fam),
                  survivors=fam.loc[fam_rej | fam_rej_ex,
                                    ["variant"] + list(lc.CANONICAL_COLUMNS[:6])
                                    + ["excess", "perm_p", "perm_p_exact", "perm_p_exact_plus1",
                                       "T"]].to_dict("records"))

    prim = pd.DataFrame(all_prim)
    summ = pd.DataFrame(all_summ)
    prim.to_csv(LENS_DIR / "variants_primary.csv", index=False)
    summ.to_csv(LENS_DIR / "variants_grid_summary.csv", index=False)

    # ---- diagnostics specific to (d) and (g) ------------------------------------------
    sp = variants["d_crude_eia_spot"][0]["crude"]
    cl = base["crude"]
    both = cl["r8"].notna() & sp["r8"].notna() & cl["idx26"].notna() & ~cl["shutdown"]
    d_diag = dict(
        corr_r8_spot_vs_CL=float(np.corrcoef(cl.loc[both, "r8"], sp.loc[both, "r8"])[0, 1]),
        mean_abs_diff_r8=float(np.mean(np.abs(cl.loc[both, "r8"] - sp.loc[both, "r8"]))),
        n_entry_date_differs=int((cl["entry_date"] != sp["entry_date"]).sum()),
        # the suspicious 2026-09-25 spot print: which eligible r8 observations touch it?
        n_r8_obs_touching_2026_09_25=int(sum(
            1 for i in np.flatnonzero(both.to_numpy())
            if sp["entry_pos"].iat[i] >= 0 and (
                spot.index[sp["entry_pos"].iat[i]] == pd.Timestamp("2026-09-25") or
                (sp["entry_pos"].iat[i] + 40 < len(spot) and
                 spot.index[sp["entry_pos"].iat[i] + 40] == pd.Timestamp("2026-09-25"))))),
    )
    mon = variants["g_monday_entry"][0]
    g_diag = {m: dict(
        entry_gap_days_min=int((mon[m]["entry_date"] - mon[m]["report_date"]).dt.days.min()),
        entry_gap_days_max=int((mon[m]["entry_date"] - mon[m]["report_date"]).dt.days.max()),
        n_reports_monday_entry_same_as_tuesday=int((mon[m]["entry_date"]
                                                     == base[m]["entry_date"]).sum()))
        for m in lc.MARKETS}

    write_json(dict(primary=prim.to_dict("records"), grid_summary=summ.to_dict("records"),
                    variant_family_bh=family,
                    d_spot_diagnostics=d_diag, g_monday_diagnostics=g_diag),
               LENS_DIR / "variants.json")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["variant", "market", "side", "n_signal_weeks", "n_episodes", "mean_ret", "excess",
            "hit_rate", "perm_p", "perm_p_exact"]
    print(prim[cols].to_string())
    print(summ.to_string())
    print("family:", {k: v for k, v in family.items() if k != "survivors"})
    print(pd.DataFrame(family["survivors"]).to_string())
    print("d diag:", d_diag)
    print("g diag:", g_diag)


if __name__ == "__main__":
    main()

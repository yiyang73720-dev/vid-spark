#!/usr/bin/env python3
"""
(b) Heatmap data: excess return of every canonical cell, laid out as
    rows    = lookback x threshold pair  (9 rows: N26 80/20, N26 75/25, N26 90/10, N52 ..., N156 ...)
    columns = horizon_w                  (4 columns: 1, 4, 8, 13 weeks)
for each market (crude, gold, silver, soybean, sugar) and for the 5-market pooled cell,
long and short separately.

Values come straight from the engine outputs (canonical.csv / pooled.csv, seed 11, K=2000).
For the pooled panel "excess" = excess_vs_own_baseline (each signal week minus the signed
baseline of its own market).

Besides the raw `excess` matrix, each panel also carries
  * excess_per_week = excess / horizon_w  (puts horizons on a comparable scale),
  * perm_p, n_signal_weeks, n_episodes (to annotate / grey out thin cells).
A flat `records` list carries the same data in long form.

Output: heatmaps.json      Run: python3 -I b_heatmaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ENGINE_DIR, LENS_DIR, lc, write_json  # noqa: E402

ROWS = [(n, bt, st) for n in lc.LOOKBACKS for bt, st in lc.THRESHOLDS]
ROW_LABELS = [f"N{n} {bt}/{st}" for n, bt, st in ROWS]


def matrices(tab: pd.DataFrame, side: str, excess_col: str) -> dict:
    """9x4 matrices (rows = ROWS, cols = horizons) of several stats for one side."""
    out = {k: [] for k in ["excess", "excess_per_week", "perm_p", "n_signal_weeks",
                           "n_episodes", "mean_ret", "hit_rate"]}
    for n, bt, st in ROWS:
        rowv = {k: [] for k in out}
        for h in lc.HORIZONS_W:
            r = tab[(tab.lookback == n) & (tab.buy_th == bt) & (tab.sell_th == st)
                    & (tab.side == side) & (tab.horizon_w == h)]
            assert len(r) == 1, (n, bt, side, h)
            r = r.iloc[0]
            rowv["excess"].append(float(r[excess_col]))
            rowv["excess_per_week"].append(float(r[excess_col]) / h)
            rowv["perm_p"].append(float(r["perm_p"]))
            rowv["n_signal_weeks"].append(int(r["n_signal_weeks"]))
            rowv["n_episodes"].append(int(r["n_episodes"]))
            rowv["mean_ret"].append(float(r["mean_ret"]))
            rowv["hit_rate"].append(float(r["hit_rate"]))
        for k in out:
            out[k].append(rowv[k])
    ex = np.array(out["excess"])
    out["share_cells_positive_excess"] = float(np.mean(ex > 0))
    out["n_cells_p_lt_05"] = int(np.sum(np.array(out["perm_p"]) < 0.05))
    return out


def main() -> None:
    canon = pd.read_csv(ENGINE_DIR / "canonical.csv", float_precision="round_trip")
    pooled = pd.read_csv(ENGINE_DIR / "pooled.csv", float_precision="round_trip")

    panels, records = {}, []
    for m in lc.MARKETS:
        t = canon[canon.market == m]
        panels[m] = {sd: matrices(t, sd, "excess") for sd in lc.SIDES}
        for _, r in t.iterrows():
            records.append(dict(market=m, side=r.side, lookback=int(r.lookback),
                                threshold=f"{int(r.buy_th)}/{int(r.sell_th)}",
                                horizon_w=int(r.horizon_w), excess=float(r.excess),
                                excess_per_week=float(r.excess) / int(r.horizon_w),
                                perm_p=float(r.perm_p), n_signal_weeks=int(r.n_signal_weeks),
                                n_episodes=int(r.n_episodes), mean_ret=float(r.mean_ret),
                                baseline_mean=float(r.baseline_mean),
                                hit_rate=float(r.hit_rate)))
    panels["pooled5"] = {sd: matrices(pooled, sd, "excess_vs_own_baseline")
                         for sd in lc.SIDES}
    for _, r in pooled.iterrows():
        records.append(dict(market="pooled5", side=r.side, lookback=int(r.lookback),
                            threshold=f"{int(r.buy_th)}/{int(r.sell_th)}",
                            horizon_w=int(r.horizon_w),
                            excess=float(r.excess_vs_own_baseline),
                            excess_per_week=float(r.excess_vs_own_baseline) / int(r.horizon_w),
                            perm_p=float(r.perm_p), n_signal_weeks=int(r.n_signal_weeks),
                            n_episodes=int(r.n_episodes), mean_ret=float(r.mean_ret),
                            hit_rate=float(r.hit_rate)))

    all_ex = np.array([x["excess"] for x in records])
    all_epw = np.array([x["excess_per_week"] for x in records])
    out = dict(
        meta=dict(
            description="Excess return (mean signed forward return minus signed baseline) of "
                        "the Larry Williams commercial COT index setups; rows = lookback x "
                        "threshold, columns = horizon in weeks.",
            units="decimal return over the horizon (0.01 = 1%)",
            source="work/engine/canonical.csv and pooled.csv (seed 11, K=2000)",
            primary_cell="N26 80/20, horizon 8w (row 'N26 80/20', column 8)",
            suggested_diverging_domain_excess=[-float(np.max(np.abs(all_ex))),
                                               float(np.max(np.abs(all_ex)))],
            suggested_diverging_domain_excess_per_week=[-float(np.max(np.abs(all_epw))),
                                                        float(np.max(np.abs(all_epw)))]),
        row_labels=ROW_LABELS,
        rows=[dict(lookback=n, buy_th=bt, sell_th=st) for n, bt, st in ROWS],
        col_labels=[f"{h}w" for h in lc.HORIZONS_W],
        horizons_w=list(lc.HORIZONS_W),
        markets=list(lc.MARKETS) + ["pooled5"],
        panels=panels,
        records=records,
    )
    write_json(out, LENS_DIR / "heatmaps.json")
    for m in out["markets"]:
        for sd in lc.SIDES:
            p = panels[m][sd]
            print(f"{m:8s} {sd:5s} share excess>0 = {p['share_cells_positive_excess']:.3f}"
                  f"  cells p<.05 = {p['n_cells_p_lt_05']}")
    print("records:", len(records), "domain:", out["meta"]["suggested_diverging_domain_excess"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
(h) Rank information coefficient (IC) of the Larry COT index.

IC = Spearman rank correlation between idx26 (as of report_date) and the forward return from
the spec's entry (Tuesday after next, no look-ahead) over h weeks. Primary h = 8; h = 1, 4, 13
are reported as a supplement. Sample = eligible weeks (idx26 defined, outside the shutdown
windows, forward return available), exactly as in the canonical cells.

A positive IC means "higher commercial index -> higher subsequent return" (Larry's claim).

Uncertainty
  * Moving-block bootstrap, block = 13 report weeks, 2000 replications, seed 11. Blocks are
    drawn on the COMMON report-date axis (all 5 markets share the same 1060 report dates) and
    applied to every market at once, so the pooled statistics keep the cross-market
    correlation. Ineligible weeks inside a resampled block are dropped. 95% CI = 2.5% / 97.5%
    percentiles.
  * Circular-shift null (same idea as the spec's permutation test, exhaustive over offsets
    26..T-26): two-sided p = share of shifts with |IC_shift - mean| >= |IC_obs - mean|,
    and one-sided p(IC_shift >= IC_obs).

Pooled versions
  * pooled_mean_ic   : equal-weight average of the 5 per-market ICs
  * pooled_stacked_ic: one Spearman over all stacked market-weeks (raw returns, so markets
                       with larger moves weigh more in the ranking)

Outputs: h_rank_ic.csv, h_rank_ic.json        Run: python3 -I h_rank_ic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import LENS_DIR, SEED, lc, write_json  # noqa: E402

BLOCK, REPS = 13, 2000
HORIZONS = (8, 1, 4, 13)            # 8 = primary


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with average ranks for ties (NaN pairs must already be removed)."""
    if len(x) < 3:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def shift_null(x: np.ndarray, y: np.ndarray) -> dict:
    """Exhaustive circular-shift null of the IC on the T eligible weeks (x rolled vs y)."""
    T = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    obs = float(np.corrcoef(rx, ry)[0, 1])          # Pearson on ranks = Spearman
    offs = np.arange(lc.MIN_SHIFT, T - lc.MIN_SHIFT + 1)
    null = np.array([np.corrcoef(np.roll(rx, k), ry)[0, 1] for k in offs])
    c = null.mean()
    return dict(ic_obs_check=obs, null_mean=float(c), null_sd=float(null.std()),
                p_two_sided=float(np.mean(np.abs(null - c) >= abs(obs - c))),
                p_one_sided_pos=float(np.mean(null >= obs)))


def main() -> None:
    panel = lc.build_panel()
    dates = panel["crude"]["report_date"].to_numpy()
    for m in lc.MARKETS:
        assert (panel[m]["report_date"].to_numpy() == dates).all()
    n = len(dates)

    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BLOCK))
    # one common set of bootstrap index vectors, reused for every horizon and market
    boot_idx = [np.concatenate([np.arange(s, s + BLOCK)
                                for s in rng.integers(0, n - BLOCK + 1, size=n_blocks)])[:n]
                for _ in range(REPS)]

    rows, out = [], {}
    for h in HORIZONS:
        X, Y = {}, {}
        for m in lc.MARKETS:
            f = panel[m]
            e = lc.eligible_mask(f, "idx26", f"r{h}")
            x = np.where(e, f["idx26"].to_numpy(), np.nan)
            y = np.where(e, f[f"r{h}"].to_numpy(), np.nan)
            X[m], Y[m] = x, y

        def stats_for(idx: np.ndarray | None) -> dict:
            """Per-market ICs, their mean and the stacked IC on a (resampled) row index."""
            res, xs, ys = {}, [], []
            for m in lc.MARKETS:
                x = X[m] if idx is None else X[m][idx]
                y = Y[m] if idx is None else Y[m][idx]
                ok = ~np.isnan(x) & ~np.isnan(y)
                res[m] = spearman(x[ok], y[ok])
                xs.append(x[ok])
                ys.append(y[ok])
            res["pooled_mean_ic"] = float(np.mean([res[m] for m in lc.MARKETS]))
            res["pooled_stacked_ic"] = spearman(np.concatenate(xs), np.concatenate(ys))
            return res

        obs = stats_for(None)
        boot = pd.DataFrame([stats_for(ix) for ix in boot_idx])
        res_h = {}
        for key in list(lc.MARKETS) + ["pooled_mean_ic", "pooled_stacked_ic"]:
            b = boot[key].to_numpy()
            d = dict(ic=obs[key], ci95_lo=float(np.quantile(b, 0.025)),
                     ci95_hi=float(np.quantile(b, 0.975)), boot_sd=float(b.std(ddof=1)),
                     share_boot_gt_0=float(np.mean(b > 0)))
            if key in lc.MARKETS:
                ok = ~np.isnan(X[key]) & ~np.isnan(Y[key])
                d["n_weeks"] = int(ok.sum())
                d.update(shift_null(X[key][ok], Y[key][ok]))
            res_h[key] = d
            rows.append(dict(horizon_w=h, series=key, **d))
        out[f"h{h}"] = res_h

    tab = pd.DataFrame(rows)
    tab.to_csv(LENS_DIR / "h_rank_ic.csv", index=False)
    write_json(dict(block_weeks=BLOCK, reps=REPS, seed=SEED, index="idx26",
                    primary_horizon_w=8, results=out), LENS_DIR / "h_rank_ic.json")
    pd.set_option("display.width", 250)
    print(tab.drop(columns=["ic_obs_check"]).to_string())


if __name__ == "__main__":
    main()

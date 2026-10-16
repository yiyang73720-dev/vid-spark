#!/usr/bin/env python3
"""
(a) Multiple testing / data-snooping on the 360-cell canonical grid.

Questions answered
  1. How many of the 360 cells have perm_p < 0.05 AND positive excess, versus the
     0.05 * 360 = 18 expected if the COT index carried no timing information?
  2. Which cells survive Benjamini-Hochberg FDR control at q = 0.10?
  3. Because the 360 cells are strongly dependent (overlapping horizons, nested thresholds,
     overlapping lookbacks, the same weeks reused), a naive Binomial(360, 0.05) reference is
     too narrow. We therefore also build a dependence-preserving JOINT null:
       - every cell is shifted by the SAME circular offset k (one k for all 360 cells, so the
         cross-cell and cross-market dependence is kept intact);
       - for each k we compute every cell's exact null p-value: the rolled signal's mean ranked
         within the rolled signal's OWN exhaustive shift distribution (offsets k + [26, T-26]
         mod T, the same exclusion zone the observed p uses), then the number of cells with
         p < 0.05 and positive excess, the number of BH(0.10) survivors, and the minimum p;
       - k runs over every offset allowed for all cells: [26, T_min - 26].
     A second variant draws an independent offset per market (5000 draws), which keeps the
     within-market dependence only.
  4. The REVERSE direction ("anti-Larry": the setup is followed by returns BELOW baseline,
     i.e. commercials are on the wrong side, consistent with price momentum). The reverse
     one-sided p is obtained from the engine by negating the forward returns (then the
     engine's "signed mean >= observed" test becomes "mean r <= observed"). It is scored
     against the same joint null, together with the number of cells with positive excess.

Inputs : engine canonical.csv (perm_p, Monte Carlo K=2000, seed 11) and the exact p-values
         recomputed here with lc.cell_stats(..., exact_p=True).
Outputs: a_multiple_testing.json, a_flagged_cells.csv, a_bh_survivors.csv, a_grid_with_q.csv,
         a_reverse_flagged_cells.csv
Run    : python3 -I a_multiple_testing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # this lens' code dir (common.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import (ENGINE_DIR, KEY_COLS, LENS_DIR, K, SEED, bh_adjusted, bh_reject,  # noqa: E402
                    grid_summary, lc, write_json)

ALPHA, Q = 0.05, 0.10


def cell_null_matrix(panel: dict, k_common: np.ndarray):
    """For every canonical cell (spec row order) return
         P[k, c]  exact null p-value of cell c when its signal is rolled by k_common[k]
         R[k, c]  exact null REVERSE p-value (share of shifts with mean <= the rolled mean)
         E[k, c]  excess of the rolled signal (rolled mean signed r - signed baseline)
       plus the observed exact p, reverse p and excess of each cell (no shift)."""
    P, R, E, p_obs, r_obs, e_obs, keys = [], [], [], [], [], [], []
    for m in lc.MARKETS:
        f = panel[m]
        for n in lc.LOOKBACKS:
            for bt, st in lc.THRESHOLDS:
                for sd in lc.SIDES:
                    for h in lc.HORIZONS_W:
                        a = lc.cell_arrays(f, f"idx{n}", bt, st, sd, f"r{h}")
                        sig, signed, sgn = a["sig"], a["signed"], a["sgn"]
                        T = len(sig)
                        base = float(a["r"].mean())
                        # mean signed return of np.roll(sig, k) for EVERY circular offset
                        # k = 0..T-1 (k = 0 is the observed signal)
                        M_full = lc._shifted_means(sig, signed, np.arange(T))
                        j = np.arange(lc.MIN_SHIFT, T - lc.MIN_SHIFT + 1)   # allowed offsets
                        # VERIFIER FIX: a null draw (signal rolled by k) must be scored against
                        # ITS OWN shift distribution, i.e. offsets k + [26, T-26] (mod T), the
                        # same exclusion zone around itself that the observed p uses around
                        # k = 0. The earlier version scored every rolled signal against the
                        # ORIGINAL signal's distribution (offsets [26, T-26] relative to 0),
                        # which keeps the rolled signal's own near-identical neighbours in its
                        # reference set, so the null rejection rate was 4.2% instead of 4.9%,
                        # p could never be 0, and the BH / min-p nulls were too optimistic.
                        Mk = M_full[k_common % T]
                        own = M_full[(k_common[:, None] + j[None, :]) % T]
                        P.append((own >= Mk[:, None]).mean(axis=1))   # ties count as >=
                        R.append((own <= Mk[:, None]).mean(axis=1))   # reverse tail
                        E.append(Mk - sgn * base)
                        obs = float(signed[sig].mean())
                        M_all = M_full[j]
                        p_obs.append(float(np.mean(M_all >= obs)))
                        r_obs.append(float(np.mean(M_all <= obs)))
                        e_obs.append(obs - sgn * base)
                        keys.append((m, n, bt, st, sd, h))
    return (np.column_stack(P), np.column_stack(R), np.column_stack(E), np.array(p_obs),
            np.array(r_obs), np.array(e_obs), keys)


def count_stats(P: np.ndarray, R: np.ndarray, E: np.ndarray) -> dict:
    """Per-row (per null draw) statistics over the cells in the columns."""
    return dict(n_flag=np.sum((P < ALPHA) & (E > 0), axis=1),
                n_bh=np.array([bh_reject(row, Q).sum() for row in P]),
                min_p=P.min(axis=1),
                n_rev_flag=np.sum((R < ALPHA) & (E < 0), axis=1),
                n_rev_bh=np.array([bh_reject(row, Q).sum() for row in R]),
                n_pos_excess=np.sum(E > 0, axis=1))


def describe_null(x: np.ndarray, observed: float) -> dict:
    """Summary of a null distribution + the share of null draws at least as extreme."""
    x = np.asarray(x, dtype=float)
    return dict(observed=float(observed), null_mean=float(x.mean()),
                null_median=float(np.median(x)), null_p95=float(np.quantile(x, 0.95)),
                null_max=float(x.max()), n_null=int(len(x)),
                share_null_ge_observed=float(np.mean(x >= observed)))


def main() -> None:
    panel = lc.build_panel()
    canon = pd.read_csv(ENGINE_DIR / "canonical.csv", float_precision="round_trip")
    # exact (exhaustive) permutation p: recomputed here so this script is self-contained
    exact = lc.canonical_table(panel, k=K, seed=SEED, exact_p=True)
    assert np.allclose(exact["perm_p"], canon["perm_p"]), "engine canonical.csv out of date"
    canon["perm_p_exact"] = exact["perm_p_exact"].to_numpy()
    canon["bh_q_mc"] = bh_adjusted(canon["perm_p"].to_numpy())
    canon["bh_q_exact"] = bh_adjusted(canon["perm_p_exact"].to_numpy())

    # ---- 1. naive count vs 5% ------------------------------------------------------------
    summ_mc = grid_summary(canon, "perm_p", ALPHA, Q)
    summ_ex = grid_summary(canon, "perm_p_exact", ALPHA, Q)
    n_obs = summ_mc["n_p_lt_alpha_and_pos_excess"]
    binom_tail = float(stats.binom.sf(n_obs - 1, len(canon), ALPHA))   # P(X >= n_obs)
    flagged = canon[(canon.perm_p < ALPHA) & (canon.excess > 0)].copy()
    flagged.to_csv(LENS_DIR / "a_flagged_cells.csv", index=False)

    # by market / side / lookback breakdown of the flagged cells
    by = {c: flagged.groupby(c).size().to_dict() for c in ["market", "side", "lookback",
                                                           "horizon_w", "buy_th"]}
    per_market_obs = {m: int(((canon.market == m) & (canon.perm_p < ALPHA)
                              & (canon.excess > 0)).sum()) for m in lc.MARKETS}

    # ---- 2. BH survivors -----------------------------------------------------------------
    rej_mc = bh_reject(canon["perm_p"].to_numpy(), Q)
    rej_ex = bh_reject(canon["perm_p_exact"].to_numpy(), Q)
    surv = canon[rej_mc | rej_ex].copy()
    surv["bh_reject_mc"] = rej_mc[rej_mc | rej_ex]
    surv["bh_reject_exact"] = rej_ex[rej_mc | rej_ex]
    surv.to_csv(LENS_DIR / "a_bh_survivors.csv", index=False)

    # ---- 3. dependence-preserving joint null --------------------------------------------
    T_min = min(len(lc.cell_arrays(panel[m], f"idx{n}", 80, 20, "long", f"r{h}")["sig"])
                for m in lc.MARKETS for n in lc.LOOKBACKS for h in lc.HORIZONS_W)
    k_common = np.arange(lc.MIN_SHIFT, T_min - lc.MIN_SHIFT + 1)
    P, R, E, p_obs_ex, r_obs_ex, e_obs, keys = cell_null_matrix(panel, k_common)
    assert np.allclose(p_obs_ex, canon["perm_p_exact"].to_numpy())
    assert np.allclose(e_obs, canon["excess"].to_numpy())
    obs_flag_ex = int(np.sum((p_obs_ex < ALPHA) & (e_obs > 0)))
    obs_bh_ex = int(bh_reject(p_obs_ex, Q).sum())
    obs_rev_flag_ex = int(np.sum((r_obs_ex < ALPHA) & (e_obs < 0)))
    obs_rev_bh_ex = int(bh_reject(r_obs_ex, Q).sum())
    obs_pos = int(np.sum(e_obs > 0))

    # (i) one common offset for all 360 cells
    cs = count_stats(P, R, E)
    joint_common = dict(
        description="same offset k for all 360 cells, k in [26, T_min-26]",
        T_min=int(T_min), n_offsets=int(len(k_common)),
        n_flagged=describe_null(cs["n_flag"], obs_flag_ex),
        n_bh_survivors=describe_null(cs["n_bh"], obs_bh_ex),
        n_reverse_flagged=describe_null(cs["n_rev_flag"], obs_rev_flag_ex),
        n_reverse_bh_survivors=describe_null(cs["n_rev_bh"], obs_rev_bh_ex),
        n_pos_excess=dict(**describe_null(cs["n_pos_excess"], obs_pos),
                          share_null_le_observed=float(np.mean(cs["n_pos_excess"] <= obs_pos))),
        min_p=dict(observed=float(p_obs_ex.min()),
                   share_null_min_p_le_observed=float(np.mean(cs["min_p"] <= p_obs_ex.min())),
                   null_p05_of_min_p=float(np.quantile(cs["min_p"], 0.05))))

    # (ii) independent offset per market (within-market dependence only)
    rng = np.random.default_rng(SEED)
    n_draw = 5000
    mkt_cols = {m: [i for i, kk in enumerate(keys) if kk[0] == m] for m in lc.MARKETS}
    draws = np.column_stack([rng.integers(0, len(k_common), size=n_draw) for _ in lc.MARKETS])
    P_ind = np.empty((n_draw, P.shape[1]))
    R_ind = np.empty_like(P_ind)
    E_ind = np.empty_like(P_ind)
    for j, m in enumerate(lc.MARKETS):
        cols = mkt_cols[m]
        P_ind[:, cols] = P[draws[:, j]][:, cols]
        R_ind[:, cols] = R[draws[:, j]][:, cols]
        E_ind[:, cols] = E[draws[:, j]][:, cols]
    ci = count_stats(P_ind, R_ind, E_ind)
    joint_indep = dict(
        description="independent offset per market, 5000 draws, seed 11",
        n_flagged=describe_null(ci["n_flag"], obs_flag_ex),
        n_bh_survivors=describe_null(ci["n_bh"], obs_bh_ex),
        n_reverse_flagged=describe_null(ci["n_rev_flag"], obs_rev_flag_ex),
        n_reverse_bh_survivors=describe_null(ci["n_rev_bh"], obs_rev_bh_ex),
        n_pos_excess=dict(**describe_null(ci["n_pos_excess"], obs_pos),
                          share_null_le_observed=float(np.mean(ci["n_pos_excess"] <= obs_pos))))

    # per-market null of the 72-cell flagged count (common offset within the market)
    per_market_null = {}
    for m in lc.MARKETS:
        cols = mkt_cols[m]
        nf = np.sum((P[:, cols] < ALPHA) & (E[:, cols] > 0), axis=1)
        obs_m = int(np.sum((p_obs_ex[cols] < ALPHA) & (e_obs[cols] > 0)))
        nr = np.sum((R[:, cols] < ALPHA) & (E[:, cols] < 0), axis=1)
        obs_r = int(np.sum((r_obs_ex[cols] < ALPHA) & (e_obs[cols] < 0)))
        npos = np.sum(E[:, cols] > 0, axis=1)
        obs_pos_m = int(np.sum(e_obs[cols] > 0))
        per_market_null[m] = dict(
            n_flagged=describe_null(nf, obs_m), n_reverse_flagged=describe_null(nr, obs_r),
            n_pos_excess=dict(**describe_null(npos, obs_pos_m),
                              share_null_le_observed=float(np.mean(npos <= obs_pos_m))))

    # ---- 4. reverse-direction p-values from the ENGINE (returns negated) -----------------
    neg = {m: f.copy() for m, f in panel.items()}
    for f in neg.values():
        for h in lc.HORIZONS_W:
            f[f"r{h}"] = -f[f"r{h}"]
    rev = lc.canonical_table(neg, k=K, seed=SEED)
    canon["perm_p_reverse"] = rev["perm_p"].to_numpy()
    canon["perm_p_reverse_exact"] = r_obs_ex
    rev_rej = bh_reject(canon["perm_p_reverse"].to_numpy(), Q)
    canon["bh_reverse_reject"] = rev_rej
    rev_flag = canon[(canon.perm_p_reverse < ALPHA) & (canon.excess < 0)].copy()
    rev_flag.to_csv(LENS_DIR / "a_reverse_flagged_cells.csv", index=False)
    canon.to_csv(LENS_DIR / "a_grid_with_q.csv", index=False)
    reverse_summary = dict(
        n_reverse_p_lt_alpha_and_neg_excess=int(len(rev_flag)),
        n_reverse_bh_survivors_mc=int(rev_rej.sum()),
        n_reverse_bh_survivors_exact=int(bh_reject(r_obs_ex, Q).sum()),
        min_reverse_p=float(canon["perm_p_reverse"].min()),
        reverse_flagged_per_market=rev_flag.groupby("market").size().to_dict(),
        reverse_flagged_per_side=rev_flag.groupby("side").size().to_dict(),
        reverse_bh_survivors=canon.loc[rev_rej, KEY_COLS + ["excess", "perm_p_reverse"]]
        .to_dict("records"))
    pooled_neg = lc.pooled_table(neg, k=K, seed=SEED)

    # ---- pooled 72-cell table (supplement) ---------------------------------------------
    pooled = pd.read_csv(ENGINE_DIR / "pooled.csv", float_precision="round_trip")
    pooled = pooled.rename(columns={"excess_vs_own_baseline": "excess"})
    pooled_summ = grid_summary(pooled, "perm_p", ALPHA, Q)
    pooled["perm_p_reverse"] = pooled_neg["perm_p"].to_numpy()
    pooled_rev_flag = pooled[(pooled.perm_p_reverse < ALPHA) & (pooled.excess < 0)]
    pooled_rev_bh = bh_reject(pooled["perm_p_reverse"].to_numpy(), Q)
    pooled.to_csv(LENS_DIR / "a_pooled_with_reverse_p.csv", index=False)
    pooled_flag = pooled[(pooled.perm_p < ALPHA) & (pooled.excess > 0)]

    head = canon[(canon.market == "crude") & (canon.lookback == 26) & (canon.buy_th == 80)
                 & (canon.horizon_w == 8)]
    out = dict(
        alpha=ALPHA, q=Q, k_perm=K, seed=SEED,
        canonical_mc=summ_mc, canonical_exact=summ_ex,
        naive_binomial_tail_P_X_ge_observed=binom_tail,
        flagged_breakdown=by, flagged_per_market=per_market_obs,
        bh_survivors_mc=canon.loc[rej_mc, KEY_COLS + ["excess", "perm_p"]].to_dict("records"),
        bh_survivors_exact=canon.loc[rej_ex, KEY_COLS + ["excess", "perm_p_exact"]]
        .to_dict("records"),
        min_bh_q_mc=float(canon["bh_q_mc"].min()),
        min_bh_q_exact=float(canon["bh_q_exact"].min()),
        joint_null_common_offset=joint_common,
        joint_null_independent_per_market=joint_indep,
        per_market_joint_null=per_market_null,
        reverse_direction=reverse_summary,
        pooled_72=pooled_summ,
        pooled_reverse=dict(n_reverse_p_lt_alpha_and_neg_excess=int(len(pooled_rev_flag)),
                            n_reverse_bh_survivors=int(pooled_rev_bh.sum()),
                            cells=pooled_rev_flag[["lookback", "buy_th", "sell_th", "side",
                                                   "horizon_w", "excess", "perm_p_reverse"]]
                            .to_dict("records")),
        pooled_flagged=pooled_flag[["lookback", "buy_th", "sell_th", "side", "horizon_w",
                                    "excess", "perm_p"]].to_dict("records"),
        headline_crude_primary=head[KEY_COLS + ["n_signal_weeks", "mean_ret", "excess",
                                                "perm_p", "perm_p_exact"]].to_dict("records"),
    )
    write_json(out, LENS_DIR / "a_multiple_testing.json")

    # console report
    print("MC  :", summ_mc)
    print("EXACT:", summ_ex)
    print("binomial tail P(X>=%d | 360, .05) = %.4g" % (n_obs, binom_tail))
    print("flagged per market:", per_market_obs)
    print("flagged breakdown:", by)
    print(flagged[KEY_COLS + ["n_signal_weeks", "n_episodes", "excess", "perm_p",
                              "perm_p_exact", "bh_q_mc"]].to_string())
    print("BH survivors MC:", int(rej_mc.sum()), " exact:", int(rej_ex.sum()),
          " min q mc/exact:", canon["bh_q_mc"].min(), canon["bh_q_exact"].min())
    print("joint null common:", joint_common)
    print("joint null indep :", joint_indep)
    print("per-market:", per_market_null)
    print("reverse:", reverse_summary)
    print(rev_flag[KEY_COLS + ["n_signal_weeks", "excess", "perm_p_reverse"]].to_string())
    print("pooled 72:", pooled_summ)
    print(pooled_flag.to_string())
    print("pooled reverse flagged:", len(pooled_rev_flag), "BH:", int(pooled_rev_bh.sum()))
    print(pooled_rev_flag[["lookback", "buy_th", "side", "horizon_w", "excess",
                           "n_mkts_positive_excess", "perm_p", "perm_p_reverse"]].to_string())


if __name__ == "__main__":
    main()

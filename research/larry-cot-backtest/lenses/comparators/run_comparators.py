#!/usr/bin/env python3
"""
run_comparators -- "comparators and incremental value" lens (SPEC.md v1).

Question: does the Larry Williams commercials index (idx26, 80/20) carry information about
forward returns beyond (a) a plain 200-day price trend and (b) speculator (managed-money)
positioning?

  (i)   Managed-money index mm_idx26, contrarian (long mm<=20 / short mm>=80) and
        trend-following (long mm>=80 / short mm<=20): canonical cell statistics (engine
        cell_stats / pooled_cell) for h = 4, 8, 13 weeks, every market + pooled, next to the
        commercials (Larry) cells. All three threshold pairs are computed; 80/20 is primary.
  (ii)  Pure price trend: long if entry close > 200-trading-day SMA, else short (h = 8 primary).
  (iii) Incremental value regressions of the forward 8-week return on standardised idx26,
        the trend dummy and standardised mm_idx26: crude alone (Newey-West HAC, lag 8) and a
        5-market panel with market fixed effects (Driscoll-Kraay lag 8 primary; clustered by
        report date secondary).
  (iv)  Agreement subsets for N=26 80/20: Larry setup and trend agree vs disagree, with a
        conditional permutation test (Larry setup shifted, trend state held fixed).
  Also: a descriptive table of every crude report since 2025-12-30 (the 2026 moves) with the
  position of each approach and its signed 8-week return.

Multiple testing: BH-FDR q-values are added for the per-market managed-money family in (i)
and for the per-market elementary agreement subsets in (iv).

Outputs (all in this script's directory):
  tables.json                      every table below, machine-readable
  i_mm_cells.csv                   (i) per-market + pooled cells
  ii_trend.csv                     (ii) trend cells
  iii_regressions.csv              (iii) coefficient table (long format)
  iv_agreement.csv                 (iv) agreement subsets

Run:  python3 -I run_comparators.py      (no cwd dependence; ~15 s)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))      # this lens' code dir
import comparators_lib as cl  # noqa: E402
lc = cl.lc
HERE = lc.DEFAULT_ROOT / "work" / "lens_comparators"          # outputs: ROOT/work/lens_comparators
HERE.mkdir(parents=True, exist_ok=True)

K, SEED = lc.K_PERM, lc.DEFAULT_SEED
MARKETS = list(lc.MARKETS)
H_MM = (4, 8, 13)
PRIMARY_H = 8

# index variants for (i): label -> (column, description)
VARIANTS = {
    "commercials_larry": ("idx26", "Larry: long comm idx>=buy, short comm idx<=sell"),
    "mm_contrarian": ("mmc_idx26", "long mm_idx26<=sell, short mm_idx26>=buy"),
    "mm_trendfollow": ("mm_idx26", "long mm_idx26>=buy, short mm_idx26<=sell"),
}


def _clean(o):
    """JSON-safe conversion (NaN -> None, numpy scalars -> Python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NaT:
        return None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def build():
    panel = lc.build_panel()
    prices = lc.load_prices()
    cl.add_comparator_columns(panel, prices)
    facts = cl.check_mm_masks(panel)
    # Sanity: engine cell_stats on idx26 reproduces the published canonical.csv cells.
    canon = pd.read_csv(cl.ENGINE_DIR / "canonical.csv")
    for _, row in canon[(canon.lookback == 26) & (canon.buy_th == 80)].iterrows():
        s = lc.cell_stats(panel[row.market], "idx26", 80, 20, row.side, f"r{row.horizon_w}")
        for c in ["n_signal_weeks", "n_episodes", "mean_ret", "excess", "perm_p"]:
            assert np.isclose(s[c], row[c], rtol=0, atol=1e-12), (row.market, c)
    facts.append("engine cell_stats(idx26, 80/20) reproduces canonical.csv for all 40 cells")
    # SMA200 is defined for every eligible week (prices start ~2000, COT 2006)
    for m in MARKETS:
        f = panel[m]
        e = lc.eligible_mask(f, "idx26", "r1")
        assert f.loc[e, "trend_up"].notna().all(), m
    facts.append("trend_up (SMA200) defined for every idx26-eligible week in every market")
    facts.append(_check_conditional_perm(panel))
    return panel, prices, facts


def _check_conditional_perm(panel, k=300):
    """Recompute the conditional permutation p of cl.subset_stats_markets for two subsets of
    crude and the pooled long_agree subset with explicit np.roll loops (same RNG draws)."""
    parts = cl.agreement_parts(panel, MARKETS, "idx26", 80, 20, "r8", "trend_up")
    for mk, comps in ((["crude"], [("long", 1.0)]), (["crude"], [("long", 0.0), ("short", 1.0)]),
                      (MARKETS, [("long", 1.0)])):
        sub = {m: parts[m] for m in mk}
        fast = cl.subset_stats_markets(sub, comps, k, SEED)
        rng = np.random.default_rng(SEED)
        offs = {m: rng.integers(lc.MIN_SHIFT, len(sub[m]["r"]) - lc.MIN_SHIFT, size=k,
                                endpoint=True) for m in mk}
        shifted = []
        for j in range(k):
            tot, cnt = 0.0, 0
            for m in mk:
                a = sub[m]
                lo, sh = np.roll(a["lo"], offs[m][j]), np.roll(a["sh"], offs[m][j])
                mask, sign = cl._subset_weeks(a, comps, lo, sh)
                tot += float((sign[mask] * a["r"][mask]).sum())
                cnt += int(mask.sum())
            if cnt:
                shifted.append(tot / cnt)
        shifted = np.array(shifted)
        p = (1 + int(np.sum(shifted >= fast["mean_ret"]))) / (1 + len(shifted))
        assert abs(p - fast["perm_p"]) < 1e-12, (mk, comps, p, fast["perm_p"])
    return "conditional permutation (agreement subsets) == explicit np.roll loop (k=300)"


# ----------------------------------------------------------------------------------------
# (i) managed money vs commercials
# ----------------------------------------------------------------------------------------
def part_i(panel):
    rows = []
    for var, (col, _) in VARIANTS.items():
        for bt, st in lc.THRESHOLDS:
            for side in lc.SIDES:
                for h in H_MM:
                    rc = f"r{h}"
                    for m in MARKETS:
                        s = lc.cell_stats(panel[m], col, bt, st, side, rc, k=K, seed=SEED,
                                          exact_p=True)
                        rows.append(dict(variant=var, market=m, buy_th=bt, sell_th=st,
                                         side=side, horizon_w=h, **s))
                    p = lc.pooled_cell(panel, col, bt, st, side, rc, MARKETS, k=K, seed=SEED)
                    rows.append(dict(variant=var, market="pooled", buy_th=bt, sell_th=st,
                                     side=side, horizon_w=h,
                                     n_signal_weeks=p["n_signal_weeks"],
                                     n_episodes=p["n_episodes"], mean_ret=p["mean_ret"],
                                     median_ret=p["median_ret"], hit_rate=p["hit_rate"],
                                     excess=p["excess_vs_own_baseline"],
                                     equal_weight_mkt_excess=p["equal_weight_mkt_excess"],
                                     n_mkts_positive_excess=p["n_mkts_positive_excess"],
                                     perm_p=p["perm_p"]))
    tab = pd.DataFrame(rows)
    # BH-FDR over the family of per-market managed-money cells at 80/20, h in (4, 8, 13)
    # (2 MM variants x 5 markets x 2 sides x 3 horizons = 60 tests), using the exact p.
    fam = ((tab.variant != "commercials_larry") & (tab.market != "pooled")
           & (tab.buy_th == 80))
    tab["bh_q_mm_family"] = np.nan
    tab.loc[fam, "bh_q_mm_family"] = multipletests(tab.loc[fam, "perm_p_exact"],
                                                   method="fdr_bh")[1]

    # Overlap between Larry setups and MM-contrarian setups, and index correlation (N=26).
    overlap = {}
    for m in MARKETS:
        f = panel[m]
        e = lc.eligible_mask(f, "idx26", "r8") & f["mm_idx26"].notna().to_numpy()
        out = dict(corr_idx26_mm_idx26=float(np.corrcoef(f.loc[e, "idx26"],
                                                         f.loc[e, "mm_idx26"])[0, 1]),
                   spearman_idx26_mm_idx26=float(f.loc[e, ["idx26", "mm_idx26"]]
                                                 .corr(method="spearman").iloc[0, 1]),
                   corr_comm_net_mm_net=float(np.corrcoef(f.loc[e, "comm_net"],
                                                          f.loc[e, "mm_net"])[0, 1]),
                   n_weeks=int(e.sum()))
        for side in lc.SIDES:
            a = lc.setup_mask(f["idx26"].to_numpy(), side, 80, 20) & e
            b = lc.setup_mask(f["mmc_idx26"].to_numpy(), side, 80, 20) & e
            out[f"{side}_larry_weeks"] = int(a.sum())
            out[f"{side}_mmcontra_weeks"] = int(b.sum())
            out[f"{side}_both_weeks"] = int((a & b).sum())
            out[f"{side}_share_of_larry_also_mmcontra"] = float((a & b).sum() / a.sum())
            out[f"{side}_jaccard"] = float((a & b).sum() / (a | b).sum())
            # weeks where Larry is in a setup and MM trend-following points the other way
            c = lc.setup_mask(f["mm_idx26"].to_numpy(), side, 80, 20) & e
            out[f"{side}_larry_and_mmtrend_same_dir_weeks"] = int((a & c).sum())
        overlap[m] = out
    return tab, overlap


# ----------------------------------------------------------------------------------------
# (ii) price trend
# ----------------------------------------------------------------------------------------
def part_ii(panel):
    rows, always = [], []
    for tcol, label in (("trend_idx", "entry_close_vs_sma200"),
                        ("trend_idx_lag", "prev_close_vs_sma200_lagged")):
        hs = H_MM if tcol == "trend_idx" else (PRIMARY_H,)
        for h in hs:
            rc = f"r{h}"
            for m in MARKETS:
                f = panel[m]
                for side in lc.SIDES:
                    s = lc.cell_stats(f, tcol, 50, 50, side, rc, k=K, seed=SEED, exact_p=True)
                    rows.append(dict(trend_def=label, market=m, side=side, horizon_w=h, **s))
                a = cl.trend_always_in(f, rc, tcol, K, SEED)
                always.append(dict(trend_def=label, market=m, horizon_w=h, **a))
            for side in lc.SIDES:
                p = lc.pooled_cell(panel, tcol, 50, 50, side, rc, MARKETS, k=K, seed=SEED)
                rows.append(dict(trend_def=label, market="pooled", side=side, horizon_w=h,
                                 n_signal_weeks=p["n_signal_weeks"],
                                 n_episodes=p["n_episodes"], mean_ret=p["mean_ret"],
                                 median_ret=p["median_ret"], hit_rate=p["hit_rate"],
                                 excess=p["excess_vs_own_baseline"],
                                 equal_weight_mkt_excess=p["equal_weight_mkt_excess"],
                                 n_mkts_positive_excess=p["n_mkts_positive_excess"],
                                 perm_p=p["perm_p"]))
            a = cl.trend_always_in_pooled(panel, MARKETS, rc, tcol, K, SEED)
            always.append(dict(trend_def=label, market="pooled", horizon_w=h, **a))
    return pd.DataFrame(rows), pd.DataFrame(always)


# ----------------------------------------------------------------------------------------
# (iii) incremental-value regressions
# ----------------------------------------------------------------------------------------
MODELS = {
    "comm_only": ["z_idx26"],
    "trend_only": ["trend"],
    "mm_only": ["z_mm_idx26"],
    "trend_mm": ["trend", "z_mm_idx26"],
    "full": ["z_idx26", "trend", "z_mm_idx26"],
    "full_discrete": ["s_comm", "trend", "s_mm_trendfollow"],
}


def _fit(df, xcols, ycol, fe, cov, maxlags=8):
    X = df[xcols].astype(float)
    if fe:
        d = pd.get_dummies(df["market"], prefix="fe", drop_first=True, dtype=float)
        X = pd.concat([X, d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[ycol].to_numpy(dtype=float)
    mod = sm.OLS(y, X.to_numpy())
    names = list(X.columns)
    if cov == "hac":
        res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov == "dk":
        res = mod.fit(cov_type="hac-groupsum",
                      cov_kwds={"time": df["time_code"].to_numpy(), "maxlags": maxlags})
    elif cov == "cluster_date":
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["time_code"].to_numpy()})
    else:
        raise ValueError(cov)
    t = cl.ols_table(res, names)
    t["coefs"] = {k: v for k, v in t["coefs"].items() if not k.startswith("fe_")}
    return t


def part_iii(panel):
    reg = cl.regression_frame(panel, MARKETS, "r8", "trend_up")
    out = {"sample": {}, "single_market": {}, "panel": {}}
    long_rows = []

    def add_rows(scope, model, spec, t):
        for v, c in t["coefs"].items():
            long_rows.append(dict(scope=scope, model=model, spec=spec, var=v, nobs=t["nobs"],
                                  r2=t["r2"], **c))

    # descriptive: correlations and VIFs of the predictors
    for m in MARKETS + ["pooled"]:
        g = reg if m == "pooled" else reg[reg.market == m]
        X = g[["z_idx26", "trend", "z_mm_idx26"]].to_numpy(dtype=float)
        out["sample"][m] = dict(
            nobs=int(len(g)), first=g.report_date.min(), last=g.report_date.max(),
            corr=g[["z_idx26", "trend", "z_mm_idx26", "y"]].corr().round(6).to_dict(),
            vif=cl.vif(X, ["z_idx26", "trend", "z_mm_idx26"]),
            share_trend_up=float(g["trend"].mean()))

    # single markets: Newey-West HAC lag 8 (lag 16 sensitivity for the full model)
    for m in MARKETS:
        g = reg[reg.market == m].sort_values("report_date")
        res_m = {}
        for name, xc in MODELS.items():
            t = _fit(g, xc, "y", fe=False, cov="hac", maxlags=8)
            res_m[name] = t
            add_rows(m, name, "NW_HAC_lag8", t)
        t16 = _fit(g, MODELS["full"], "y", fe=False, cov="hac", maxlags=16)
        res_m["full_NW_lag16"] = t16
        add_rows(m, "full", "NW_HAC_lag16", t16)
        res_m["delta_r2_full_minus_trend_mm"] = res_m["full"]["r2"] - res_m["trend_mm"]["r2"]
        out["single_market"][m] = res_m

    # pooled panel with market fixed effects
    reg = reg.sort_values(["market", "report_date"]).reset_index(drop=True)
    res_p = {}
    for name, xc in MODELS.items():
        t = _fit(reg, xc, "y", fe=True, cov="dk", maxlags=8)
        res_p[name] = t
        add_rows("pooled_FE", name, "DriscollKraay_lag8", t)
    for spec, kw in (("cluster_date", dict(cov="cluster_date")),
                     ("DriscollKraay_lag16", dict(cov="dk", maxlags=16))):
        t = _fit(reg, MODELS["full"], "y", fe=True, **kw)
        res_p[f"full_{spec}"] = t
        add_rows("pooled_FE", "full", spec, t)
    # returns scaled by each market's r8 sd (equalises volatility across markets)
    for name in ("comm_only", "full"):
        t = _fit(reg, MODELS[name], "y_volstd", fe=True, cov="dk", maxlags=8)
        res_p[f"{name}_volstd_y"] = t
        add_rows("pooled_FE_volstd_y", name, "DriscollKraay_lag8", t)
    res_p["delta_r2_full_minus_trend_mm"] = res_p["full"]["r2"] - res_p["trend_mm"]["r2"]
    out["panel"] = res_p
    return out, pd.DataFrame(long_rows)


# ----------------------------------------------------------------------------------------
# (iv) agreement subsets
# ----------------------------------------------------------------------------------------
SUBSETS = {
    "long_agree(trend_up)": [("long", 1.0)],
    "long_disagree(trend_down)": [("long", 0.0)],
    "short_agree(trend_down)": [("short", 0.0)],
    "short_disagree(trend_up)": [("short", 1.0)],
    "all_agree": [("long", 1.0), ("short", 0.0)],
    "all_disagree": [("long", 0.0), ("short", 1.0)],
    # all Larry setups, for reference (state-unconditional)
    "all_setups": [("long", 1.0), ("long", 0.0), ("short", 0.0), ("short", 1.0)],
}


def part_iv(panel):
    rows = []
    for h in H_MM:
        parts = cl.agreement_parts(panel, MARKETS, "idx26", 80, 20, f"r{h}", "trend_up")
        for name, comps in SUBSETS.items():
            for m in MARKETS:
                s = cl.subset_stats_markets({m: parts[m]}, comps, K, SEED)
                rows.append(dict(subset=name, market=m, horizon_w=h, **s))
            s = cl.subset_stats_markets(parts, comps, K, SEED)
            rows.append(dict(subset=name, market="pooled", horizon_w=h, **s))
    tab = pd.DataFrame(rows)
    # BH-FDR within each horizon over the 20 per-market elementary subsets
    # (long/short x agree/disagree x 5 markets).
    tab["bh_q_elementary_family"] = np.nan
    for h in H_MM:
        fam = ((tab.horizon_w == h) & (tab.market != "pooled")
               & tab.subset.str.startswith(("long_", "short_")))
        tab.loc[fam, "bh_q_elementary_family"] = multipletests(tab.loc[fam, "perm_p"],
                                                               method="fdr_bh")[1]
    return tab


# ----------------------------------------------------------------------------------------
# Latest crude readings (context for the 2026 case; descriptive only)
# ----------------------------------------------------------------------------------------
def latest_crude(panel, prices, start="2025-12-30"):
    """Every crude report from `start` on (covers the 2026 moves), with the three approaches'
    positions and, where the 8-week return exists, each approach's signed r8. Descriptive."""
    f = panel["crude"]
    f = f[f.report_date >= pd.Timestamp(start)]
    rows = []
    for _, r in f.iterrows():
        larry = 1 if r.idx26 >= 80 else -1 if r.idx26 <= 20 else 0
        mm_tf = 1 if r.mm_idx26 >= 80 else -1 if r.mm_idx26 <= 20 else 0
        trend = (np.nan if np.isnan(r.trend_up) else (1 if r.trend_up == 1.0 else -1))
        rows.append(dict(report_date=r.report_date, entry_date=r.entry_date,
                         shutdown=bool(r.shutdown), entry_price=r.entry_price,
                         sma200_at_entry=r.sma200_entry, idx26=r.idx26, idx52=r.idx52,
                         idx156=r.idx156, mm_idx26=r.mm_idx26,
                         larry_pos=larry, mm_trendfollow_pos=mm_tf, trend_pos=trend,
                         r4=r.r4, r8=r.r8,
                         larry_signed_r8=(larry * r.r8 if larry else np.nan),
                         mm_trendfollow_signed_r8=(mm_tf * r.r8 if mm_tf else np.nan),
                         trend_signed_r8=trend * r.r8))
    c = prices["crude"]
    sma = c.rolling(cl.SMA_N, min_periods=cl.SMA_N).mean()
    last = dict(date=c.index[-1], close=float(c.iloc[-1]), sma200=float(sma.iloc[-1]),
                above_sma200=bool(c.iloc[-1] > sma.iloc[-1]))
    return rows, last


def main():
    panel, prices, facts = build()
    print("\n".join(facts), flush=True)
    t_i, overlap = part_i(panel)
    print("(i) done", flush=True)
    t_ii, t_ii_always = part_ii(panel)
    print("(ii) done", flush=True)
    t_iii, t_iii_long = part_iii(panel)
    print("(iii) done", flush=True)
    t_iv = part_iv(panel)
    print("(iv) done", flush=True)
    latest, last_px = latest_crude(panel, prices)

    t_i.to_csv(HERE / "i_mm_cells.csv", index=False)
    pd.concat([t_ii.assign(kind="side_cell"), t_ii_always.assign(kind="always_in")],
              ignore_index=True).to_csv(HERE / "ii_trend.csv", index=False)
    t_iii_long.to_csv(HERE / "iii_regressions.csv", index=False)
    t_iv.to_csv(HERE / "iv_agreement.csv", index=False)

    tables = dict(
        meta=dict(seed=SEED, k_perm=K, markets=MARKETS, primary="N=26, 80/20, h=8",
                  trend="entry close > SMA200 of cleaned closes (window incl. entry day)",
                  self_checks=facts),
        i_mm_cells=t_i.to_dict(orient="records"),
        i_overlap_N26_80_20=overlap,
        ii_trend_side_cells=t_ii.to_dict(orient="records"),
        ii_trend_always_in=t_ii_always.to_dict(orient="records"),
        iii_regressions=t_iii,
        iv_agreement=t_iv.to_dict(orient="records"),
        latest_crude=dict(reports=latest, last_price=last_px),
    )
    with open(HERE / "tables.json", "w") as fh:
        json.dump(_clean(tables), fh, indent=1)
    print("wrote", HERE / "tables.json")


if __name__ == "__main__":
    main()

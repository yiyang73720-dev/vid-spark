#!/usr/bin/env python3
"""
run_rank -- steps 1, 2, 4 and 5 of the rank step of the Larry Williams fundamentals battery
(PREREG.md sections 6-8); step 3 (confluence) is run_confluence.py and is read from its output.

  1. One table of every tool's PRE-REGISTERED primary: excess h8, perm p, split halves,
     walk-forward, S1/S2 after costs vs buy & hold, the tradability check, and the adversarial
     verifiers' corrections (numbers read from each results.json, never typed in).
     The COT (v1) row is recomputed with the shared harness, which reproduces v1 bit-for-bit
     (the v1 numbers are cross-checked here).
  2. Family-wide multiple testing: every test row of the five SPEC2 results.json files plus the
     360-cell v1 COT grid; Benjamini-Hochberg at q = 0.10 on the +1-corrected exhaustive p where
     available (else the Monte-Carlo p); survivors per tool. Also h8-only, primaries-only,
     Benjamini-Yekutieli, and a second family that adds the confluence rows.
  4. Ranking by the pre-declared scorecard / tiers (PREREG 7).
  5. chart_data.json: per tool primary excess h8 with a 95% circular block-bootstrap interval
     (13-week blocks), OOS halves, S2 Sharpe vs buy & hold; weekly equity curves of the top tool's
     primary strategy vs buy & hold.

Outputs (ROOT/work2/rank/): results.json, primary_table.csv, family_tests_with_q.csv,
family_bh_summary.json, ranking.csv, chart_data.json, run_rank_log.txt.
Run: python3 -I run_rank.py   (after run_confluence.py)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import os                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rank_lib (battery/rank)
import hashlib                      # noqa: E402
import json                         # noqa: E402
import time                         # noqa: E402

import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

import rank_lib as rl               # noqa: E402

fh, se, lc = rl.fh, rl.se, rl.lc
HERE = rl.RANK                      # outputs: ROOT/work2/rank
HERE.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOG: list[str] = []
Q = 0.10                            # BH level (task / PREREG 6)


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    LOG.append(line)
    print(line, flush=True)


def pick(rows, **kw) -> dict:
    """The single row of a list-of-dicts / DataFrame matching all key=value filters."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if int(m.sum()) != 1:
        raise AssertionError(f"expected one row for {kw}, got {int(m.sum())}")
    return df[m].iloc[0].to_dict()


def fnum(x):
    """float or None (JSON-safe)."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def strat_cols(met: pd.DataFrame) -> dict:
    """S1/S2 long-only and long+short Sharpe, CAGR, max DD, trades, null p; buy & hold."""
    out = {}
    for st, key in (("S1_long_only", "S1_LO"), ("S1_long_short", "S1_LS"),
                    ("S2_long_only", "S2_LO"), ("S2_long_short", "S2_LS")):
        r = pick(met, strategy=st)
        out.update({f"{key}_sharpe": fnum(r["sharpe"]), f"{key}_cagr": fnum(r["ann_return"]),
                    f"{key}_max_dd": fnum(r["max_dd"]), f"{key}_trades": fnum(r.get("n_trades")),
                    f"{key}_null_p": fnum(r.get("null_p_sharpe"))})
    b = pick(met, strategy="buy_hold")
    out.update(bh_sharpe=fnum(b["sharpe"]), bh_cagr=fnum(b["ann_return"]),
               bh_max_dd=fnum(b["max_dd"]), strat_window=f"{b['start']}..{b['end']}")
    return out


def half_cols(h1: dict, h2: dict, p_key: str = "perm_p_exact") -> dict:
    p1 = h1.get(p_key) if h1.get(p_key) is not None and not pd.isna(h1.get(p_key)) else h1["perm_p"]
    p2 = h2.get(p_key) if h2.get(p_key) is not None and not pd.isna(h2.get(p_key)) else h2["perm_p"]
    return dict(split_date=h1.get("split_date"), h1_n=int(h1["n"]), h1_excess=fnum(h1["excess"]),
                h1_p=fnum(p1), h2_n=int(h2["n"]), h2_excess=fnum(h2["excess"]), h2_p=fnum(p2))


# =============================================================================================
# Step 1: the primary table
# =============================================================================================
def cot_row() -> tuple[dict, dict]:
    """v1 COT primary recomputed with the harness (crude N26 80/20 long h8 on crude_cl, timing
    cot), its halves, identity walk-forward, S1/S2 (null_k 2000) and the crude_roll check."""
    spec = rl.primary_cot()
    sf, ser, timing = spec["parts"][0]
    p = fh.evaluate(sf, ser, (8,), timing=timing, sides=("long",)).iloc[0].to_dict()
    hv = fh.split_half(sf, ser, timing=timing, horizon=8, sides=("long",))
    st = fh.simulate(sf, ser, timing=timing,
                     labels=dict(filter="cot", variant="primary", market="crude"))
    met = st["metrics"]
    # cross-check against the v1 strategy lens (metrics.csv)
    v1 = pd.read_csv(rl.ROOT / "work" / "lens_strategy" / "metrics.csv",
                     float_precision="round_trip")
    v1 = v1[v1.market == "crude"].drop_duplicates("strategy").set_index("strategy")
    for s_ in fh.STRATEGIES:
        a, b = float(pick(met, strategy=s_)["sharpe"]), float(v1.loc[s_, "sharpe"])
        if a != b:
            raise AssertionError(f"COT {s_} Sharpe {a} differs from v1 metrics.csv {b}")
    trad = fh.evaluate(sf, "crude_roll", (8,), timing=timing, sides=("long",)).iloc[0].to_dict()
    row = dict(tool="cot", rule="crude: Larry COT index of commercials (N=26) >= 80 -> LONG; "
                                "h=8w; v1 CL continuous (crude_cl), timing cot",
               market="crude", series=ser, timing=timing, side="long",
               **{k: p[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                    "perm_p", "perm_p_exact", "period")},
               **half_cols(pick(hv, half="H1"), pick(hv, half="H2")),
               wf_kind="identity (trailing 26-report index, no fitted parameter) = static rule",
               wf_n=int(p["n"]), wf_excess=fnum(p["excess"]), wf_p=fnum(p["perm_p_exact"]),
               **strat_cols(met), strat_series="crude_cl",
               tradable_desc="v1 primary signal re-evaluated on crude_roll (computed here)",
               tradable_n=int(trad["n"]), tradable_excess=fnum(trad["excess"]),
               tradable_p=fnum(trad["perm_p_exact"]),
               tool_verdict="no edge (v1)", verifier_verdict="no edge (v1 canonical table reproduced by two "
                                                             "independent implementations)",
               verifier_corrections="none to the headline")
    return row, dict(spec=spec, strategy=st)


def premium_row() -> dict:
    res = rl.load_results("premium")
    p = pick(res["tests"], is_primary=True)
    hv = pd.DataFrame(res["oos"]["split_half"])
    wf = pick(res["oos"]["walk_forward"]["rows"], side="long", horizon_w=8)
    met = pd.DataFrame(res["strategy"])
    met = met[(met["variant"] == "primary") & (met["series"] == "crude_roll")]
    sub = res["verification"]["independent_rederivation"]["H_subperiods_long_h8"]
    a, b = sub["1985_2009"], sub["2010-01-01_2024-02-02"]
    spot = pick(res["tests"], variant="primary", series="crude_spot", side="long", horizon_w=8)
    # The sub-period numbers below are the verifier's re-derivation on ITS OWN roll index
    # (print-calendar roll dates); on the harness crude_roll index they are 1985-2009
    # +0.0234 (exact p 0.033) and 2010-2024-02 -0.0154 (exact p 0.742) -- see
    # results.json verification.subperiods_long_h8 of tool_premium. (Label added by the final
    # audit; the values themselves were already correct for that index.)
    corr = (f"edge is 1985-2009 only (verifier's independent roll index): excess "
            f"{a['excess']:+.4f} (exact p {a['perm_p_exact']:.3f}, n {a['n']}); 2010-2024-02 "
            f"{b['excess']:+.4f} (p {b['perm_p_exact']:.3f}, {b['episodes']} episodes). Same "
            f"signal on EIA spot: {spot['excess']:+.4f} (p {spot['perm_p_exact']:.3f}) -> the edge "
            f"is roll yield (carry), not price.")
    return dict(tool="premium", rule="crude: contract 1 > contract 2 (backwardation) at the "
                                      "Friday close -> LONG; h=8w; crude_roll; timing price",
                market="crude", series=p["series"], timing=p["timing"], side=p["side"],
                **{k: p[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                     "perm_p", "perm_p_exact", "period")},
                **half_cols(pick(hv, half="H1", side="long", horizon_w=8),
                            pick(hv, half="H2", side="long", horizon_w=8)),
                wf_kind=res["oos"]["walk_forward"]["description"],
                wf_n=int(wf["n"]), wf_excess=fnum(wf["excess"]), wf_p=fnum(wf["perm_p_exact"]),
                **strat_cols(met), strat_series="crude_roll",
                tradable_desc="primary itself (crude_roll, roll-adjusted)",
                tradable_n=int(p["n"]), tradable_excess=fnum(p["excess"]),
                tradable_p=fnum(p["perm_p_exact"]),
                tool_verdict=res["verdict"],
                verifier_verdict=res["verification"].get("overall_verdict", "positive EV, fragile"),
                verifier_corrections=corr)


def seasonality_row() -> dict:
    res = rl.load_results("seasonality")
    t = pd.DataFrame(res["tests"])
    p = pick(t, is_primary=True)
    hv = res["oos"]["split_half_pooled"]
    met = pd.DataFrame(res["strategy"])
    met = met[met["market"] == "EW7_portfolio"]
    trad = pick(t, variant="tradable_primary_full", market="pooled", side="net", horizon_w=8)
    tp = pd.DataFrame(res["strategy_tradable_portfolio"])
    tp_s2 = pick(tp, strategy="S2_long_only")
    ver = res["verification"]
    wo_gas = pick(t, variant="posthoc_primary_without_gasoline", side="net", horizon_w=8)
    corr = (f"verifier practical verdict '{ver['verdict']['verifier_practical_verdict']}': "
            f"gasoline spot supplies {ver['robustness']['share_of_total_excess_from_gasoline_spot']:.1%}"
            f" of the pooled excess; without it {wo_gas['excess']:+.4f} (p {wo_gas['perm_p']:.3f}); "
            f"tradable pooled check {trad['excess']:+.5f} (p {trad['perm_p']:.3f}); tradable "
            f"portfolio S2 long-only Sharpe {tp_s2['sharpe']:.3f} vs buy & hold "
            f"{tp_s2['bh_sharpe']:.3f} (null p {tp_s2['null_p_sharpe']:.3f}).")
    return dict(tool="seasonality",
                rule="7 markets pooled (crude/HO/gasoline spot, gold, silver, soybean, sugar v1), "
                     "walk-forward week-of-year tendency 70/30, NET side, h=8w; timing price",
                market="pooled7", series=p["series"], timing=p["timing"], side=p["side"],
                **{k: p.get(k) for k in ("n", "episodes", "T", "mean", "baseline", "excess",
                                         "hit", "perm_p", "perm_p_exact", "period")},
                **half_cols(pick(hv, half="H1"), pick(hv, half="H2"), p_key="perm_p"),
                wf_kind="primary is walk-forward by construction (prior-year analogs only; "
                        "asserted, truncation test 168/168)",
                wf_n=int(p["n"]), wf_excess=fnum(p["excess"]), wf_p=fnum(p["perm_p"]),
                **strat_cols(met), strat_series="EW7 portfolio (spot energies + v1 series)",
                tradable_desc="pre-registered tradable check: estimated and evaluated on "
                              "roll-adjusted futures (crude_roll + pysystemtrade), pooled net h8",
                tradable_n=int(trad["n"]), tradable_excess=fnum(trad["excess"]),
                tradable_p=fnum(trad["perm_p"]),
                tool_verdict=res["primary"]["verdict"]["verdict"],
                verifier_verdict=ver["verdict"]["verifier_practical_verdict"],
                verifier_corrections=corr)


def valuation_row() -> dict:
    res = rl.load_results("valuation")
    p = pick(res["tests"], is_primary=True)
    hv = pd.DataFrame(res["oos"]["split_half"])
    wf = pick(res["oos"]["walk_forward"]["selection"], side="long", horizon_w=8)
    met = pd.DataFrame(res["strategy"])
    met = met[(met["variant"] == "primary") & (met["series"] == "crude_roll")]
    w20 = res["episode_sensitivity_audit"]["without_2020"]
    corr = (f"without the 2020 COVID episode: primary {w20['primary']['excess']:+.4f} "
            f"(p {w20['primary']['perm_p']:.3f}); H2 {w20['primary_H2']['excess']:+.4f} "
            f"(p {w20['primary_H2']['perm_p']:.3f}); selection walk-forward "
            f"{w20['walk_forward_selection']['excess']:+.4f} "
            f"(p {w20['walk_forward_selection']['perm_p']:.3f}).")
    return dict(tool="valuation",
                rule="crude (CL) vs London gold, WillVal EMA2-EMA22, 156-week range, VI <= 20 "
                     "-> LONG; h=8w; crude_roll; timing price",
                market="crude", series=p["series"], timing=p["timing"], side=p["side"],
                **{k: p[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                     "perm_p", "perm_p_exact", "period")},
                **half_cols(pick(hv, half="H1", side="long", horizon_w=8),
                            pick(hv, half="H2", side="long", horizon_w=8)),
                wf_kind="pre-registered yearly parameter-selection walk-forward (2008+)",
                wf_n=int(wf["n"]), wf_excess=fnum(wf["excess"]), wf_p=fnum(wf["perm_p_exact"]),
                **strat_cols(met), strat_series="crude_roll",
                tradable_desc="primary itself (crude_roll)", tradable_n=int(p["n"]),
                tradable_excess=fnum(p["excess"]), tradable_p=fnum(p["perm_p_exact"]),
                tool_verdict=res["verdict"], verifier_verdict="no edge",
                verifier_corrections=corr)


def open_interest_row() -> dict:
    res = rl.load_results("open_interest")
    p = pick(res["tests"], is_primary=True)
    hv = pd.DataFrame(res["oos"]["split_half"])
    wf = res["oos"]["walk_forward"]["WF1_long_h8"]
    met = pd.DataFrame(res["strategy"])
    met = met[(met["variant"] == "primary") & (met["series"] == "crude_roll")]
    av = res["adversarial_verification"]["result"]
    lo = av["leave_one_episode_out_excess"]
    corr = (f"single-episode dependence: without the {lo['min_episode']} episode the excess is "
            f"{lo['min']:+.4f}; one observation per episode {av['first_week_of_episode']['excess']:+.4f}"
            f" (n {av['first_week_of_episode']['n']}); S1 long-only is inside the random-timing null"
            f" (exhaustive null p {av['crude_S1_long_only']['null']['p_plus1']:.3f}).")
    return dict(tool="open_interest",
                rule="crude: 4-report F+O OI change in the bottom 10% of 156 reports while the "
                     "4-report |price change| <= its median -> LONG; h=8w; crude_roll; timing cot",
                market="crude", series=p["series"], timing=p["timing"], side=p["side"],
                **{k: p[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                     "perm_p", "perm_p_exact", "period")},
                **half_cols(pick(hv, half="H1", side="long", horizon_w=8),
                            pick(hv, half="H2", side="long", horizon_w=8)),
                wf_kind="pre-registered WF-1: yearly choice among 24 OI rules from realised past "
                        "outcomes (2010+)",
                wf_n=int(wf["n"]), wf_excess=fnum(wf["excess"]), wf_p=fnum(wf["perm_p_exact"]),
                **strat_cols(met), strat_series="crude_roll",
                tradable_desc="primary itself (crude_roll)", tradable_n=int(p["n"]),
                tradable_excess=fnum(p["excess"]), tradable_p=fnum(p["perm_p_exact"]),
                tool_verdict=res["verdict"], verifier_verdict="no edge",
                verifier_corrections=corr)


def intermarket_row() -> dict:
    res = rl.load_results("intermarket")
    t = pd.DataFrame(res["tests"])
    p = pick(t, is_primary=True)
    hv = pd.DataFrame(res["oos"]["split_half"])
    wf = pick(res["oos"]["walk_forward_selection"]["rows"], side="long", horizon_w=8)
    met = pd.DataFrame(res["strategy"])
    trad = pick(t, grid="H", series="pst_gold", variant="lead_usd_L8", side="long", horizon_w=8)
    corr = ("wording only (C4/C5/C7/C10); every primary, OOS and strategy number reproduced. "
            "The pooled 5-market secondary (+0.48%, p 0.072) is first-half and 2008-09 "
            "dependent.")
    return dict(tool="intermarket",
                rule="gold: broad dollar (DTWEXBGS, 7-day lag) fell over 8 weeks -> LONG gold; "
                     "h=8w; London gold; timing price",
                market="gold", series=p["series"], timing=p["timing"], side=p["side"],
                **{k: p[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                     "perm_p", "perm_p_exact", "period")},
                **half_cols(pick(hv, half="H1", side="long", horizon_w=8),
                            pick(hv, half="H2", side="long", horizon_w=8)),
                wf_kind="pre-registered yearly leader/lookback/sign selection walk-forward (2011+)",
                wf_n=int(wf["n"]), wf_excess=fnum(wf["excess"]), wf_p=fnum(wf["perm_p_exact"]),
                **strat_cols(met), strat_series="gold (London)",
                tradable_desc="primary signal on the pysystemtrade gold futures roll index "
                              "(pst_gold)", tradable_n=int(trad["n"]),
                tradable_excess=fnum(trad["excess"]), tradable_p=fnum(trad["perm_p_exact"]),
                tool_verdict=res["verdict"],
                verifier_verdict=res["adversarial_verification"]["verdict_check"],
                verifier_corrections=corr)


# =============================================================================================
# Step 2: family-wide multiple testing
# =============================================================================================
FAM_COLS = ["tool", "variant", "market", "series", "timing", "side", "horizon_w", "n",
            "episodes", "excess", "perm_p", "perm_p_exact", "period", "is_primary"]


def v1_cot_grid() -> pd.DataFrame:
    """The 360-cell v1 COT grid through the harness (exact +1 p), checked bit-for-bit
    against S/work/engine/canonical_exact_p.csv."""
    panel = rl.cot_panel()
    ser = {"crude": "crude_cl", "gold": "gold", "silver": "silver", "soybean": "soybean",
           "sugar": "sugar"}
    rows = []
    for m in lc.MARKETS:
        for n in lc.LOOKBACKS:
            for bt, st in lc.THRESHOLDS:
                sf = fh.cot_signal_frame(m, f"idx{n}", bt, st, panel=panel)
                r = fh.evaluate(sf, ser[m], lc.HORIZONS_W, timing="cot",
                                sides=("long", "short"))
                r["market"], r["lookback"], r["buy_th"], r["sell_th"] = m, n, bt, st
                rows.append(r)
    g = pd.concat(rows, ignore_index=True)
    ref = pd.read_csv(rl.ROOT / "work" / "engine" / "canonical_exact_p.csv",
                      float_precision="round_trip")
    j = ref.merge(g, on=["market", "lookback", "buy_th", "sell_th", "side", "horizon_w"],
                  suffixes=("_v1", ""))
    if len(j) != 360:
        raise AssertionError("v1 grid merge failed")
    for a, b in (("n_signal_weeks", "n"), ("excess_v1", "excess"), ("perm_p_v1", "perm_p"),
                 ("perm_p_exact_v1", "perm_p_exact_raw")):
        x, y = j[a].to_numpy(float), j[b].to_numpy(float)
        if not ((x == y) | (np.isnan(x) & np.isnan(y))).all():
            raise AssertionError(f"v1 grid column {a} not reproduced")
    g["tool"] = "cot"
    g["variant"] = ("N" + g["lookback"].astype(str) + "_" + g["buy_th"].astype(str) + "/"
                    + g["sell_th"].astype(str))
    g["is_primary"] = ((g.market == "crude") & (g.lookback == 26) & (g.buy_th == 80)
                       & (g.side == "long") & (g.horizon_w == 8))
    return g


def family_table(cot_grid: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for t in ("premium", "seasonality", "valuation", "open_interest", "intermarket"):
        d = pd.DataFrame(rl.load_results(t)["tests"])
        d["tool"] = t
        parts.append(d.reindex(columns=FAM_COLS))
    parts.append(cot_grid.reindex(columns=FAM_COLS))
    fam = pd.concat(parts, ignore_index=True)
    fam["is_primary"] = fam["is_primary"].fillna(False).astype(bool)
    ex = fam["perm_p_exact"].astype(float)
    fam["p_used"] = np.where(ex.notna(), ex, fam["perm_p"].astype(float))
    fam["p_source"] = np.where(ex.notna(), "exact+1", np.where(fam["perm_p"].notna(), "mc",
                                                               "none"))
    return fam


def bh_family(fam: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    """q-values over the rows with a p-value; survivors per tool at q <= Q."""
    f = fam[fam["p_used"].notna()].copy()
    f["q_bh_all"] = rl.bh_qvalues(f["p_used"].to_numpy())
    f["q_by_all"] = rl.by_qvalues(f["p_used"].to_numpy())
    f["q_bh_h8"] = np.nan
    h8 = (f["horizon_w"] == 8).to_numpy()
    f.loc[h8, "q_bh_h8"] = rl.bh_qvalues(f.loc[h8, "p_used"].to_numpy())
    f["q_bh_primaries"] = np.nan
    pm = f["is_primary"].to_numpy(bool)
    f.loc[pm, "q_bh_primaries"] = rl.bh_qvalues(f.loc[pm, "p_used"].to_numpy())
    # within-tool BH (each tool's own grid as its family) -- context for the family result
    f["q_bh_within_tool"] = np.nan
    for t, g in f.groupby("tool", sort=False):
        f.loc[g.index, "q_bh_within_tool"] = rl.bh_qvalues(g["p_used"].to_numpy())
    # resolution floor: permutation p-values cannot go below 1/(1+K) (MC) or 1/(1+#offsets)
    # (exhaustive). BH can only reject when some k has p_(k) <= k*Q/m, so with every p >= p_min
    # at least ceil(p_min*m/Q) rows must sit at (or below) that floor.
    m_rows = int(len(f))
    p_min = float(f["p_used"].min())
    floor = dict(m=m_rows, p_min=p_min, bh_threshold_k1=Q / m_rows,
                 min_rows_needed_at_p_min=int(np.ceil(p_min * m_rows / Q)),
                 rows_at_p_min=int((f["p_used"] <= p_min + 1e-15).sum()),
                 rows_p_le_0001=int((f["p_used"] <= 0.001).sum()),
                 rows_p_le_0005=int((f["p_used"] <= 0.005).sum()),
                 bh_threshold_at_k_rows_p_le_0005=Q * int((f["p_used"] <= 0.005).sum()) / m_rows)
    per_tool = {}
    for t, g in f.groupby("tool", sort=False):
        surv = g[g["q_bh_all"] <= Q].sort_values("p_used")
        per_tool[t] = dict(
            rows=int(len(g)), rows_p_le_005=int((g["p_used"] <= 0.05).sum()),
            expected_p_le_005_if_null=round(0.05 * len(g), 1),
            bh_all_survivors=int(len(surv)),
            bh_h8_survivors=int((g["q_bh_h8"] <= Q).sum()),
            by_all_survivors=int((g["q_by_all"] <= Q).sum()),
            within_tool_bh_survivors=int((g["q_bh_within_tool"] <= Q).sum()),
            within_tool_bh_survivors_h8=int(((g["q_bh_within_tool"] <= Q)
                                             & (g["horizon_w"] == 8)).sum()),
            min_p=fnum(g["p_used"].min()),
            min_q_bh_all=fnum(g["q_bh_all"].min()),
            primary_q_bh_all=fnum(g.loc[g["is_primary"], "q_bh_all"].min())
            if g["is_primary"].any() else None,
            primary_q_bh_primaries=fnum(g.loc[g["is_primary"], "q_bh_primaries"].min())
            if g["is_primary"].any() else None,
            survivors_top=surv.head(12)[["variant", "market", "series", "side", "horizon_w",
                                         "n", "episodes", "excess", "p_used", "p_source",
                                         "q_bh_all"]].to_dict("records"),
            within_tool_survivors_top=g[g["q_bh_within_tool"] <= Q].sort_values("p_used").head(
                12)[["variant", "market", "series", "side", "horizon_w", "n", "episodes",
                     "excess", "p_used", "p_source", "q_bh_within_tool"]].to_dict("records"),
            smallest_p_rows=g.sort_values("p_used").head(5)[
                ["variant", "market", "series", "side", "horizon_w", "n", "excess", "p_used",
                 "p_source", "q_bh_all"]].to_dict("records"))
    summ = dict(family=label, rows_total=int(len(fam)), rows_with_p=int(len(f)),
                rows_without_p_dropped=int(fam["p_used"].isna().sum()),
                p_source_counts=f["p_source"].value_counts().to_dict(),
                q_level=Q, bh_all_survivors=int((f["q_bh_all"] <= Q).sum()),
                bh_h8_rows=int(h8.sum()), bh_h8_survivors=int((f["q_bh_h8"] <= Q).sum()),
                by_all_survivors=int((f["q_by_all"] <= Q).sum()),
                primaries_only_bh_survivors=int((f.loc[pm, "q_bh_primaries"] <= Q).sum()),
                resolution_floor=floor,
                primaries=f.loc[pm, ["tool", "variant", "market", "series", "side", "excess",
                                     "p_used", "q_bh_all", "q_bh_primaries"]].to_dict("records"),
                per_tool=per_tool)
    return f, summ


# =============================================================================================
# Step 5 helper: bootstrap of each primary's excess
# =============================================================================================
def primary_bootstrap(tool: str, spec: dict, published_excess: float) -> dict:
    """95% circular block-bootstrap interval (13-week blocks, B 2000, seed 11) of the primary
    excess; the point estimate is recomputed from the eligible rows and must equal the
    published excess."""
    if spec["kind"] == "single":
        sf, ser, timing = spec["parts"][0]
        rows = rl.eligible_rows(sf, ser, timing, 8)
        b = rl.bootstrap_excess(rows["signal"], rows["r"], spec["side"])
    else:
        stk = []
        for (sf, ser, timing), name in zip(spec["parts"], spec["names"]):
            e = rl.eligible_rows(sf, ser, timing, 8)
            e["market"] = name
            stk.append(e)
        stk = pd.concat(stk, ignore_index=True)
        b = rl.bootstrap_excess(stk["signal"], stk["r"], spec["side"],
                                group=stk["market"].to_numpy(),
                                cal=stk["decision_date"].to_numpy())
    if abs(b["excess"] - published_excess) > 1e-12:
        raise AssertionError(f"{tool}: bootstrap point {b['excess']} != published "
                             f"{published_excess}")
    return b


# =============================================================================================
# Step 4: ranking
# =============================================================================================
def scorecard(row: dict, q_primary: float | None) -> dict:
    p_used = row["perm_p_exact"] if row.get("perm_p_exact") is not None and not pd.isna(
        row.get("perm_p_exact")) else row["perm_p"]
    R = dict(
        R1_excess_pos=bool(row["excess"] > 0),
        R2_p_le_005=bool(p_used <= 0.05),
        R3_family_bh_survives=bool(q_primary is not None and q_primary <= Q),
        R4_both_halves_pos=bool(row["h1_excess"] > 0 and row["h2_excess"] > 0),
        R5_walk_forward_pos=bool(row["wf_excess"] > 0),
        R6_s2_long_only_beats_bh_after_costs=bool(
            row["S2_LO_sharpe"] > row["bh_sharpe"] and row["S2_LO_null_p"] is not None
            and row["S2_LO_null_p"] <= 0.10),
        R7_tradable_excess_pos=bool(row["tradable_excess"] > 0))
    v = list(R.values())
    if all(v):
        tier, tname = 1, "robust positive EV after costs"
    elif (R["R1_excess_pos"] and R["R4_both_halves_pos"] and R["R5_walk_forward_pos"]
          and R["R7_tradable_excess_pos"] and p_used <= 0.20):
        tier, tname = 2, "positive EV, fragile (tradable series)"
    elif R["R1_excess_pos"] and R["R2_p_le_005"] and not R["R7_tradable_excess_pos"]:
        tier, tname = 3, "positive in-sample, not tradable"
    else:
        tier, tname = 4, "no edge"
    return dict(**R, n_pass=int(sum(v)), tier=tier, tier_name=tname, p_used=float(p_used))


# =============================================================================================
# Plain-Chinese one-sentence summaries (numbers inserted from the computed values; the
# qualitative words are guarded by assertions on those values)
# =============================================================================================
def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def chinese_summary(t: str, r: dict, extra: dict) -> str:
    if t == "cot":
        assert r["excess"] < 0 and r["S2_LO_null_p"] > 0.10
        return (f"COT商业持仓指数（原油N=26≥80做多，持有8周）的超额收益是{pct(r['excess'])}，"
                f"置换检验p={r['perm_p_exact']:.2f}，前后两半{pct(r['h1_excess'])}/"
                f"{pct(r['h2_excess'])}，扣成本后S2只做多夏普{r['S2_LO_sharpe']:.2f}对买入持有"
                f"{r['bh_sharpe']:.2f}且随机择时p={r['S2_LO_null_p']:.2f}——没有正期望值。")
    if t == "premium":
        assert r["excess"] > 0 and r["h2_excess"] < r["h1_excess"]
        assert extra["post2010"] < 0 and extra["spot"] < 0
        # Final-audit wording: "failed since 2010" is true for the pre-registered statistic
        # (weekly LONG excess), not for every form of the signal, so the measure is named, and
        # the pre-specified long+short strategy result (SPEC2 rule 6, secondary) is mentioned.
        assert r["S2_LS_sharpe"] > r["bh_sharpe"] and abs(extra["h2_net"]) < 0.005
        return (f"升贴水（原油近月高于次月就做多，持有8周）超额{pct(r['excess'])}但p="
                f"{r['perm_p_exact']:.3f}不显著，前半段{pct(r['h1_excess'])}、后半段只剩"
                f"{pct(r['h2_excess'])}（2010年后{pct(extra['post2010'])}），赚的是展期收益而不是"
                f"价格上涨（同一信号在现货上{pct(extra['spot'])}），扣成本后只做多夏普"
                f"{r['S2_LO_sharpe']:.2f}低于买入持有{r['bh_sharpe']:.2f}——六个工具里最不差的一个，"
                f"但很脆弱，按周平均的做多超额在2010年以来为负。次要结果：同一信号做多+做空"
                f"（贴水做空），扣成本后S2夏普{r['S2_LS_sharpe']:.2f}高于买入持有"
                f"{r['bh_sharpe']:.2f}（随机择时p={r['S2_LS_null_p']:.3f}），但2004年后按周的多空"
                f"超额只有{pct(extra['h2_net'])}，2010年后的收益主要来自几笔暴跌中的空单——"
                f"只能算线索，不是已证实的优势。")
    if t == "seasonality":
        assert r["excess"] > 0 and r["tradable_excess"] < 0
        return (f"季节性（7个市场合并，持有8周）超额{pct(r['excess'])}、p={r['perm_p']:.3f}，"
                f"前后两半都为正，但一半以上来自无法交易的纽约港汽油现货"
                f"（占{extra['gas_share']:.0%}），在可交易的期货展期指数上是"
                f"{pct(r['tradable_excess'])}（p={r['tradable_p']:.2f}）——统计上最强，实际上不能赚钱。")
    if t == "valuation":
        assert r["excess"] < 0 and r["h1_excess"] < 0 < r["h2_excess"]
        assert extra["no2020"] < 0 and extra["no2020_h2"] < 0
        return (f"估值指数（原油相对黄金被低估就做多，持有8周）超额{pct(r['excess'])}、p="
                f"{r['perm_p_exact']:.2f}，前半段{pct(r['h1_excess'])}，后半段{pct(r['h2_excess'])}"
                f"全靠2020年一次暴跌后的反弹（去掉2020年后半段为{pct(extra['no2020_h2'])}、整体为"
                f"{pct(extra['no2020'])}），扣成本后S2只做多夏普{r['S2_LO_sharpe']:.2f}——没有优势。")
    if t == "open_interest":
        assert r["h1_excess"] < 0 < r["h2_excess"] and extra["loo_min"] < 0
        return (f"持仓量规则（原油横盘时持仓量大降就做多，持有8周）超额{pct(r['excess'])}、p="
                f"{r['perm_p_exact']:.2f}不显著，前后两半符号相反（{pct(r['h1_excess'])}/"
                f"{pct(r['h2_excess'])}），去掉2021年一次行情就变成{pct(extra['loo_min'])}——"
                f"没有可检测的优势。")
    if t == "intermarket":
        assert r["excess"] < 0 and r["h2_excess"] < 0 and r["S2_LO_sharpe"] < r["bh_sharpe"]
        return (f"跨市场（美元8周下跌就做多黄金，持有8周）超额{pct(r['excess'])}、p="
                f"{r['perm_p_exact']:.2f}，后半段{pct(r['h2_excess'])}，扣成本后S2只做多夏普"
                f"{r['S2_LO_sharpe']:.2f}低于直接持有黄金{r['bh_sharpe']:.2f}——没有优势。")
    if t == "confluence":
        assert extra["verdict"] == "confluence does not beat the tools alone"
        return (f"多工具共振（原油上至少2个工具同时看多且多于看空，持有8周）超额"
                f"{pct(extra['excess'])}、p={extra['p']:.2f}，没有显著胜过任何一个单独工具，"
                f"5个市场合并也只有{pct(extra['pooled_excess'])}（p={extra['pooled_p']:.2f}）——"
                f"把条件叠加起来并没有带来优势。")
    raise KeyError(t)


# =============================================================================================
# main
# =============================================================================================
def main() -> None:
    prereg_sha = hashlib.sha256((rl.RANK_CODE / "PREREG.md").read_bytes()).hexdigest()
    if prereg_sha != (rl.RANK_CODE / "PREREG.sha256").read_text().split()[0]:
        raise SystemExit("PREREG.md changed after it was hashed")
    conf = json.load(open(HERE / "confluence_results.json", encoding="utf-8"))
    if conf["prereg_sha256"] != prereg_sha:
        raise SystemExit("confluence_results.json was produced under another PREREG")

    # ---------------- Step 1 ----------------
    cot, _ = cot_row()
    log("COT row recomputed; S1/S2 Sharpe equal v1 metrics.csv")
    rows = [cot, premium_row(), seasonality_row(), valuation_row(), open_interest_row(),
            intermarket_row()]
    table = pd.DataFrame(rows)
    log("primary table assembled")

    # ---------------- Step 2 ----------------
    cot_grid = v1_cot_grid()
    log("v1 COT grid (360 cells) reproduced bit-for-bit through the harness")
    fam = family_table(cot_grid)
    fam_q, fam_summ = bh_family(fam, "five SPEC2 results.json tests + v1 COT grid")
    ctests = pd.read_csv(HERE / "confluence_tests.csv", low_memory=False)
    ctests["tool"] = "confluence"
    ctests["is_primary"] = ctests["is_primary"].fillna(False).astype(bool)
    fam2 = pd.concat([fam, ctests.reindex(columns=FAM_COLS).assign(
        p_used=lambda d: np.where(d["perm_p_exact"].notna(), d["perm_p_exact"], d["perm_p"]),
        p_source=lambda d: np.where(d["perm_p_exact"].notna(), "exact+1", "mc"))],
        ignore_index=True)
    fam2_q, fam2_summ = bh_family(fam2, "main family + confluence rows")
    fam_q.to_csv(HERE / "family_tests_with_q.csv", index=False)
    (HERE / "family_bh_summary.json").write_text(json.dumps(fh.to_jsonable(
        dict(main=fam_summ, with_confluence=fam2_summ)), indent=1))
    log(f"family BH: {fam_summ['rows_with_p']} rows with p; survivors q<=0.10: "
        f"{fam_summ['bh_all_survivors']} (h8 family {fam_summ['bh_h8_survivors']}, BY "
        f"{fam_summ['by_all_survivors']})")
    for t, v in fam_summ["per_tool"].items():
        log(f"   {t:14s} rows {v['rows']:5d}  p<=.05 {v['rows_p_le_005']:4d} (exp "
            f"{v['expected_p_le_005_if_null']})  BH {v['bh_all_survivors']:4d}  BH-h8 "
            f"{v['bh_h8_survivors']:4d}  BY {v['by_all_survivors']:4d}  within-tool BH "
            f"{v['within_tool_bh_survivors']:4d} (h8 {v['within_tool_bh_survivors_h8']})  "
            f"min p {v['min_p']:.5f}  primary q {v['primary_q_bh_all']:.3f}")
    log(f"primaries-only BH (6 rows): survivors {fam_summ['primaries_only_bh_survivors']}; "
        + "; ".join(f"{r['tool']} p={r['p_used']:.4f} q={r['q_bh_primaries']:.4f}"
                    for r in fam_summ["primaries"]))
    log(f"resolution floor: {fam_summ['resolution_floor']}")

    # ---------------- Step 5 (bootstrap intervals) ----------------
    specs = {t: rl.PRIMARY_BUILDERS[t]() for t in rl.TOOLS}
    boots = {t: primary_bootstrap(t, specs[t], float(table.loc[table.tool == t, "excess"].iloc[0]))
             for t in rl.TOOLS}
    for t, b in boots.items():
        log(f"bootstrap {t}: excess {b['excess']:+.5f} 95% [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}]")
    table["boot_ci_lo"] = [boots[t]["ci_lo"] for t in table.tool]
    table["boot_ci_hi"] = [boots[t]["ci_hi"] for t in table.tool]

    # ---------------- Step 4 ----------------
    qprim = {t: fam_summ["per_tool"][t]["primary_q_bh_all"] for t in rl.TOOLS}
    sc = pd.DataFrame([dict(tool=r["tool"], **scorecard(r, qprim[r["tool"]]))
                       for r in table.to_dict("records")])
    sc["boot_ci_lo"] = [boots[t]["ci_lo"] for t in sc.tool]
    sc = sc.sort_values(["tier", "n_pass", "boot_ci_lo"], ascending=[True, False, False],
                        kind="mergesort").reset_index(drop=True)
    sc["rank"] = np.arange(1, len(sc) + 1)
    sc.to_csv(HERE / "ranking.csv", index=False)
    table = table.merge(sc[["tool", "rank", "tier", "tier_name", "n_pass"]], on="tool")
    table = table.sort_values("rank").reset_index(drop=True)
    table.to_csv(HERE / "primary_table.csv", index=False)
    best = sc.iloc[0]["tool"]
    any_tier1 = bool((sc["tier"] == 1).any())
    log("ranking: " + " > ".join(f"{r.tool}(T{r.tier},{r.n_pass})" for r in sc.itertuples()))

    # ---------------- Step 5: equity of the best tool's primary strategy ----------------
    spec = specs[best]
    if spec["kind"] == "single":
        sf, ser, timing = spec["parts"][0]
        st = fh.simulate(sf, ser, timing=timing, null_k=0)
        pub = table[table.tool == best].iloc[0]
        for s_, key in (("S1_long_only", "S1_LO"), ("S2_long_only", "S2_LO"),
                        ("S1_long_short", "S1_LS"), ("S2_long_short", "S2_LS")):
            a = float(pick(st["metrics"], strategy=s_)["sharpe"])
            if abs(a - pub[f"{key}_sharpe"]) > 1e-12:
                raise AssertionError(f"{best} {s_}: rebuilt Sharpe {a} != published")
        eq = fh.weekly_equity(st["equity"])
        eq_series = ser
    else:
        port = se.portfolio_simulation(spec["parts"], spec["names"], null_k=0)
        eq = fh.weekly_equity(port["equity"])
        eq_series = "EW7 portfolio"
    log(f"equity curves of {best} rebuilt ({len(eq['dates'])} weeks)")

    # ---------------- Chinese summaries ----------------
    prem_res = rl.load_results("premium")
    sub = prem_res["verification"]["independent_rederivation"]["H_subperiods_long_h8"]
    spot = pick(prem_res["tests"], variant="primary", series="crude_spot", side="long",
                horizon_w=8)
    seas_res = rl.load_results("seasonality")
    val_res = rl.load_results("valuation")
    oi_res = rl.load_results("open_interest")
    extras = dict(
        cot={}, intermarket={},
        premium=dict(post2010=sub["2010-01-01_2024-02-02"]["excess"], spot=spot["excess"],
                     h2_net=pick(prem_res["oos"]["split_half"], half="H2", side="net",
                                 horizon_w=8)["excess"]),
        seasonality=dict(gas_share=seas_res["verification"]["robustness"][
            "share_of_total_excess_from_gasoline_spot"]),
        valuation=dict(
            no2020=val_res["episode_sensitivity_audit"]["without_2020"]["primary"]["excess"],
            no2020_h2=val_res["episode_sensitivity_audit"]["without_2020"]["primary_H2"][
                "excess"]),
        open_interest=dict(loo_min=oi_res["adversarial_verification"]["result"][
            "leave_one_episode_out_excess"]["min"]))
    tr = {r["tool"]: r for r in table.to_dict("records")}
    zh = {t: chinese_summary(t, tr[t], extras[t]) for t in rl.TOOLS}
    pooled_cell = pick(conf["pooled5"]["cell"], side="long", horizon_w=8)
    zh["confluence"] = chinese_summary("confluence", {}, dict(
        verdict=conf["verdict"], excess=conf["primary"]["excess"],
        p=conf["primary"]["perm_p_exact"], pooled_excess=pooled_cell["excess"],
        pooled_p=pooled_cell["perm_p"]))

    # ---------------- chart_data.json ----------------
    label = {"cot": "COT commercials (v1)", "premium": "Premium / backwardation",
             "seasonality": "Seasonality (7-mkt pooled)", "valuation": "Valuation (WillVal)",
             "open_interest": "Open interest", "intermarket": "Intermarket (dollar -> gold)"}
    tools_chart = []
    for r in table.to_dict("records"):
        t = r["tool"]
        tools_chart.append(dict(
            tool=t, label=label[t], rank=int(r["rank"]), tier=int(r["tier"]),
            tier_name=r["tier_name"], rule=r["rule"], series=r["series"], side=r["side"],
            period=r["period"], n=int(r["n"]), episodes=int(r["episodes"]),
            excess_h8=r["excess"], ci95=[boots[t]["ci_lo"], boots[t]["ci_hi"]],
            perm_p=r["perm_p_exact"] if r["perm_p_exact"] is not None and not pd.isna(
                r["perm_p_exact"]) else r["perm_p"],
            oos_halves=dict(split_date=r["split_date"],
                            H1=dict(excess=r["h1_excess"], p=r["h1_p"], n=r["h1_n"]),
                            H2=dict(excess=r["h2_excess"], p=r["h2_p"], n=r["h2_n"])),
            walk_forward=dict(kind=r["wf_kind"], excess=r["wf_excess"], p=r["wf_p"]),
            tradable_check=dict(desc=r["tradable_desc"], excess=r["tradable_excess"],
                                p=r["tradable_p"]),
            s2=dict(long_only_sharpe=r["S2_LO_sharpe"], long_only_null_p=r["S2_LO_null_p"],
                    long_short_sharpe=r["S2_LS_sharpe"], long_short_null_p=r["S2_LS_null_p"],
                    bh_sharpe=r["bh_sharpe"], strategy_series=r["strat_series"],
                    window=r["strat_window"]),
            s1=dict(long_only_sharpe=r["S1_LO_sharpe"], long_short_sharpe=r["S1_LS_sharpe"]),
            summary_zh=zh[t]))
    cprim = conf["primary"]
    cmet = pd.DataFrame(conf["strategy"])
    chalf = pd.DataFrame(conf["split_half"])
    conf_chart = dict(
        tool="confluence", label="Confluence (>=2 tools agree, crude)",
        rule=conf["primary_rule"], series="crude_roll", side="long", period=cprim["period"],
        n=cprim["n"], episodes=cprim["episodes"], excess_h8=cprim["excess"],
        ci95=[conf["primary_bootstrap_excess_ci"]["ci_lo"],
              conf["primary_bootstrap_excess_ci"]["ci_hi"]],
        perm_p=cprim["perm_p_exact"],
        oos_halves=dict(split_date=pick(chalf, half="H1", side="long")["split_date"],
                        **{h: dict(excess=pick(chalf, half=h, side="long")["excess"],
                                   p=pick(chalf, half=h, side="long")["perm_p_exact"],
                                   n=pick(chalf, half=h, side="long")["n"]) for h in ("H1", "H2")}),
        s2=dict(long_only_sharpe=pick(cmet, strategy="S2_long_only")["sharpe"],
                long_only_null_p=pick(cmet, strategy="S2_long_only")["null_p_sharpe"],
                long_short_sharpe=pick(cmet, strategy="S2_long_short")["sharpe"],
                long_short_null_p=pick(cmet, strategy="S2_long_short")["null_p_sharpe"],
                bh_sharpe=pick(cmet, strategy="buy_hold")["sharpe"]),
        verdict=conf["verdict"], summary_zh=zh["confluence"])
    chart = dict(
        meta=dict(description="Larry Williams fundamentals battery: pre-registered primary of "
                              "each tool. excess_h8 = mean 8-week signed return minus the "
                              "signed baseline (decimal). ci95 = circular moving-block "
                              "bootstrap, 13-week blocks, B=2000, seed 11. perm_p = exhaustive "
                              "+1-corrected circular-shift p (Monte-Carlo for pooled cells).",
                  ranking_rule="PREREG.md section 7", prereg_sha256=prereg_sha,
                  units="decimal returns (0.01 = 1%)"),
        tools=tools_chart, confluence=conf_chart,
        equity_best=dict(tool=best, series=eq_series,
                         note="weekly equity (last trading day of each week), 1 = start; "
                              "S1/S2 after 0.05% per side; buy_hold without cost",
                         dates=eq["dates"], series_values=eq["series"]),
        equity_confluence=dict(series="crude_roll", dates=conf["charts"]["equity_primary"]["dates"],
                               series_values=conf["charts"]["equity_primary"]["series"]))
    (HERE / "chart_data.json").write_text(json.dumps(fh.to_jsonable(chart), indent=1))
    log("wrote chart_data.json")

    # ---------------- caveats (numbers inserted from the computed values) ----------------
    fl = fam_summ["resolution_floor"]
    act = conf["activity_counts_common_weeks"]
    cvar = pd.DataFrame(json.load(open(HERE / "confluence_results.json"))["variants"])
    unan = pick(cvar, variant="k2_unanimous", side="long", horizon_w=8)
    cs2 = pick(cmet, strategy="S2_long_short")
    pr = tr["premium"]
    caveats = [
        f"Family BH: {fl['m']} rows with a p-value; permutation p-values cannot go below "
        f"{fl['p_min']:.5f} here, so BH at q=0.10 needs at least {fl['min_rows_needed_at_p_min']} "
        f"rows at that floor (only {fl['rows_at_p_min']} are; {fl['rows_p_le_0005']} rows have "
        f"p<=0.005). Zero survivors therefore also reflects p resolution and heavy row "
        f"duplication (long/short/net complements, nested grids). The 6-row primaries-only family "
        f"is the informative one: only the seasonality primary survives (q "
        f"{[r['q_bh_primaries'] for r in fam_summ['primaries'] if r['tool'] == 'seasonality'][0]:.3f}),"
        f" and it is not tradable.",
        "The ranking rule (PREREG section 7) was written after the tools' published results were "
        "known; it encodes the task's criteria (out-of-sample, after costs, tradable) and was not "
        "tuned, but it is not blind.",
        f"Premium passes R4 only because H2 excess is {pr['h2_excess']:+.5f} (> 0 by a hair, p "
        f"{pr['h2_p']:.2f}); the verifier's 2010-2024 sub-period is "
        f"{extras['premium']['post2010']:+.4f}, and on EIA spot the same signal is "
        f"{extras['premium']['spot']:+.4f}: the positive EV is roll yield earned before 2010.",
        f"Confluence components premium and dollar are always-on regime signals (long/short/none "
        f"on the crude common weeks: premium {act['premium']}, dollar {act['dollar']}), so the >=2 "
        f"rule is dominated by them. Common window {conf['common_weeks']['first']}.."
        f"{conf['common_weeks']['last']} (OI percentiles need a year of history; crude_roll ends "
        f"2024-04-05).",
        f"Not promoted (secondary, hypotheses only): confluence S2 long+short Sharpe "
        f"{cs2['sharpe']:.3f} vs buy & hold {pick(cmet, strategy='buy_hold')['sharpe']:.3f} (null p "
        f"{cs2['null_p_sharpe']:.3f}), and the unanimous-k2 variant long h8 excess "
        f"{unan['excess']:+.4f} (exact p {unan['perm_p_exact']:.3f}, {int(unan['n'])} weeks, "
        f"{int(unan['episodes'])} episodes).",
        "Bootstrap intervals treat 13-week blocks as exchangeable; for few-episode primaries "
        "(valuation 14 episodes, open interest 27) they are very wide.",
        "Series caveats carried over from the tools: seasonality's primary uses spot energy and "
        "unadjusted v1 soybean/sugar series; COT's v1 primary uses unadjusted CL continuous "
        "(crude_cl); intermarket's tradable check uses the pysystemtrade gold index whose "
        "post-2013 prints are not always settlements; crude_roll, the EIA curve and the "
        "pysystemtrade curves end in 2024-03/04, so premium and confluence cannot be checked on "
        "2024-04..2026.",
        "Larry's own wording could not be read from his books for any tool (sandbox DNS); every "
        "definition rests on search-engine extracts and third-party re-implementations."]

    # ---------------- final-audit additions (ROOT/work2/audit_final) ----------------
    # Added by the final adversarial audit. Every number is read from the tools' results or
    # from the audit scripts' JSON outputs (audit_premium_independent.py,
    # audit_premium_strategy_trades.py, audit_family_bh.py, audit_confluence.py).
    wf_par = prem_res["oos"]["walk_forward"]["params_by_year"]
    n_long_kept = sum(bool(p_["trade_long"]) for p_ in wf_par)
    sh_prem = pd.DataFrame(prem_res["oos"]["split_half"])
    net_h2 = pick(sh_prem, half="H2", side="net", horizon_w=8)
    caveats.append(
        f"Premium R5: the pre-registered learned-side walk-forward kept the long side in "
        f"{n_long_kept} of {len(wf_par)} yearly refits from 1995, so its result "
        f"({pr['wf_excess']:+.4f}, p {pr['wf_p']:.3f}) is "
        f"{'exactly the static rule on 1995+ weeks' if n_long_kept == len(wf_par) else 'close to the static rule'}"
        f", not an independent out-of-sample confirmation.")
    caveats.append(
        f"Premium long+short (pre-specified by SPEC2 rule 6 but secondary; never the headline): "
        f"S1 Sharpe {pr['S1_LS_sharpe']:.3f} (null p {pr['S1_LS_null_p']:.4f}) and S2 "
        f"{pr['S2_LS_sharpe']:.3f} (null p {pr['S2_LS_null_p']:.4f}) vs buy & hold "
        f"{pr['bh_sharpe']:.3f}. This is the strongest after-cost, tradable result of the battery, "
        f"but the weekly long+short (net) excess in H2 is {net_h2['excess']:+.5f} (exact p "
        f"{net_h2['perm_p_exact']:.2f}): after 2004 the strategy's gain comes from its "
        f"episode-start entries (since 2010 mostly three crash shorts, see the final-audit "
        f"caveat), not from the weekly signal. Costs are charged only at entry and exit, not on "
        f"the ~2 contract rolls inside each 40-day hold.")
    audit_dir = rl.WORK2 / "audit_final"
    final_audit = None
    try:
        ai = json.load(open(audit_dir / "audit_premium_independent.json", encoding="utf-8"))
        at = json.load(open(audit_dir / "audit_premium_strategy_trades.json", encoding="utf-8"))
        ab = json.load(open(audit_dir / "audit_family_bh.json", encoding="utf-8"))
        ac = json.load(open(audit_dir / "audit_confluence.json", encoding="utf-8"))
        ip, ih2 = ai["primary_long_h8"], ai["H2_long"]
        sfull, s10 = ai["strategy_full"], at["from2010"]
        top3 = [s10[k]["top3_share_of_log_growth"] for k in ("S1_long_short", "S2_long_short")]
        exc = at["full"]
        strat_surv = ab["strategy_family_posthoc"]["survivors_six"]
        cal = ai["calendar_vs_harness"]
        sc_h = [d["score_harness"][0] for d in cal["detail"]]
        sc_a = [d["score_audit"][0] for d in cal["detail"]]
        n_fav = sum(h > a for h, a in zip(sc_h, sc_a))
        naive = ai["naive_data_driven_calendar_diagnostic"]
        ver_ind = prem_res["verification"]["independent_rederivation"]
        caveats.append(
            f"Final audit, independent rebuild from the raw EIA files (own code, own NYMEX-rule roll "
            f"calendar; it differs from the harness in {cal['months_differing']} months by one day, "
            f"and in {n_fav} of them the curve-shape evidence favours the harness day): premium "
            f"primary n {ip['n']}, {ip['episodes']} episodes, excess {ip['excess']:+.5f}, exact p "
            f"{ip['perm_p_exact']:.4f} (published {pr['excess']:+.5f}, {pr['perm_p_exact']:.4f}); "
            f"H2 {ih2['excess']:+.5f} (p {ih2['perm_p_exact']:.2f}); S2 long-only Sharpe "
            f"{sfull['S2_long_only']['sharpe']:.3f} vs buy & hold {sfull['buy_hold']['sharpe']:.3f}. "
            f"Roll dating matters: a deliberately naive data-driven roll calendar that mis-dates "
            f"{naive['months_differing_from_rule']} months moves the excess to "
            f"{naive['primary_long_h8']['excess']:+.5f} (p "
            f"{naive['primary_long_h8']['perm_p_exact']:.3f}); on the correctly dated calendars "
            f"(harness {pr['perm_p_exact']:.4f}, verifier "
            f"{ver_ind['A_primary_my_index']['perm_p_exact']:.4f}, audit "
            f"{ip['perm_p_exact']:.4f}) p stays above 0.05.")
        if min(ip["perm_p_exact"], pr["perm_p_exact"],
               ver_ind["A_primary_my_index"]["perm_p_exact"]) <= 0.05:
            raise AssertionError("premium p on a correct calendar fell to <= 0.05; revise caveat")
        caveats.append(
            f"Final audit, POST-HOC (not pre-registered): the premium long+short strategies keep "
            f"their edge over the full window when every trade overlapping the 2008, 2014-15 and "
            f"2020 crashes is removed from the real and the null paths (S1 Sharpe "
            f"{exc['S1_long_short']['ex_crash_trades']['sharpe']:.3f}, null p "
            f"{exc['S1_long_short']['ex_crash_trades']['null_p']:.4f}; S2 "
            f"{exc['S2_long_short']['ex_crash_trades']['sharpe']:.3f}, p "
            f"{exc['S2_long_short']['ex_crash_trades']['null_p']:.4f}), and they survive BH "
            f"(q=0.10) across the 24 primary-strategy null tests of the six tools "
            f"({', '.join(f'{s['tool']} {s['strategy']} q {s['q_bh']:.3f}' for s in strat_surv)}). "
            f"But since 2010, {min(top3):.0%}-{max(top3):.0%} of their log growth comes from three "
            f"short trades in crude crashes (2014-11, 2018-10, 2020-01).")
        final_audit = dict(
            scripts=[str(rl.BATTERY / "audit_final" / f) for f in (
                "audit_premium_independent.py", "audit_premium_strategy_trades.py",
                "audit_family_bh.py", "audit_confluence.py")],
            premium_independent=dict(
                primary=ip, H1=ai["H1_long"], H2=ih2, pre2010=ai["pre2010_long"],
                from2010=ai["from2010_long"], spot=ai["spot_long"],
                walk_forward=ai["walk_forward_long"], strategy=sfull,
                calendar_months_differing_from_harness=ai["calendar_vs_harness"][
                    "months_differing"]),
            premium_long_short_posthoc=dict(
                full_ex_crash={k: exc[k]["ex_crash_trades"] for k in
                               ("S1_long_short", "S2_long_short")},
                from2010={k: {kk: s10[k][kk] for kk in ("sharpe", "null_p", "n_trades",
                                                         "top3_share_of_log_growth")}
                          for k in ("S1_long_short", "S2_long_short")},
                strategy_family_bh_survivors=strat_surv),
            family_bh=dict(rows=ab["bh"]["m"], survivors_all=ab["bh"]["survivors_all"],
                           survivors_by=ab["bh"]["survivors_by"],
                           survivors_h8=ab["bh"]["survivors_h8"],
                           p_identical_to_published=ab["p_identical_to_published_sorted"],
                           v1_plus1_consistent=ab["v1_plus1_check"]["p_used_gt_raw"]),
            confluence=dict(timing=ac["timing"], raw_mismatches=ac[
                "raw_recomputation_mismatches"], truncation_tests=ac["truncation_tests"],
                primary_reproduced=ac["primary_recomputed"] == ac["primary_published"]))
        log("final-audit outputs found and folded into caveats / results.json")
    except FileNotFoundError:
        log("final-audit outputs not found; caveats without the audit lines")

    # ---------------- results.json ----------------
    out = dict(
        step="rank", prereg_sha256=prereg_sha,
        primary_table=table, ranking=sc, best_tool=best, any_robust_positive_ev=any_tier1,
        family_testing=dict(main=fam_summ, with_confluence=fam2_summ),
        confluence=dict(verdict=conf["verdict"], primary=conf["primary"],
                        bootstrap=conf["primary_bootstrap_excess_ci"],
                        H_a=conf["H_a_beats_random_timing"], H_b=conf["H_b_beats_each_component"],
                        split_half=conf["split_half"], strategy=conf["strategy"],
                        pooled5_long_h8=pooled_cell,
                        pooled5_H_b=conf["pooled5"]["H_b_long_h8"]),
        bootstrap_primary=boots, summaries_zh=zh, caveats=caveats, final_audit=final_audit,
        run_log=LOG, run_seconds=round(time.time() - T0, 1))
    (HERE / "results.json").write_text(json.dumps(fh.to_jsonable(out), indent=1))
    (HERE / "run_rank_log.txt").write_text("\n".join(LOG) + "\n")
    log("wrote results.json, primary_table.csv, ranking.csv, family_tests_with_q.csv")


if __name__ == "__main__":
    main()

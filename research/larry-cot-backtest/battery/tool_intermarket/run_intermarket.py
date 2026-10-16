#!/usr/bin/env python3
"""
run_intermarket -- the pre-registered Larry Williams intermarket test (PREREG.md in this
directory; its sha256 is stored in results.json). Run with:

    python3 -I run_intermarket.py

What it does (section numbers refer to PREREG.md):
  3  PRIMARY: gold, long when the broad dollar's 8-week change (read with a 7-day publication
     lag) is negative, h = 8 weeks, traded on the v1 London gold series; plus the pooled
     5-market secondary headline.
  5  Variant grid A-H (leader-change grid, Larry's literal gold -> bonds rule, Willspread, vote
     composite, extended pysystemtrade history, own-trend control, pooled cells, other gold series).
  6  Out-of-sample: split-half and the pre-registered walk-forward selection rule.
  7  S1/S2 strategy simulation with costs and a random-timing null.
  8  Benjamini-Hochberg q-values (information only).
Outputs (this directory): results.json (SPEC2 rule-8 contract + charts), tests.csv,
trades_primary.csv, run_log.txt. Every number is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parent         # this tool's code and PREREG.md
sys.path.insert(0, str(CODE))
import intermarket_lib as il  # noqa: E402

fh = il.fh
HERE = fh.WORK2 / "tool_intermarket"           # outputs: ROOT/work2/tool_intermarket
HERE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------
# Settings (all pre-registered)
# ---------------------------------------------------------------------------------------------
FILTER = "intermarket"
TRADED = {"crude": "crude_roll", "gold": "gold", "silver": "silver", "soybean": "soybean",
          "sugar": "sugar"}
MARKETS = list(TRADED)
LOOKBACKS = (4, 8, 13)
H = (1, 4, 8, 13)
SIDES = ("long", "short", "net")
PRIMARY = dict(market="gold", series="gold", leader="usd", lookback_w=8, side="long",
               horizon_w=8, variant="lead_usd_L8")
WF_FIRST_TEST = "2011-01-01"
WF_MIN_LONG_WEEKS = 26
TIMING = "price"

T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    LOG.append(line)
    print(line, flush=True)


TESTS: list[pd.DataFrame] = []      # every evaluate() block
FRAMES: dict[tuple, tuple] = {}     # (grid, variant, market, series) -> (signal frame, series)
SIGS: dict[tuple, np.ndarray] = {}  # (variant, market) -> signal on that market's Friday grid


def run_eval(grid: str, variant: str, market: str, series, fridays, sig, *, horizons=H,
             sides=SIDES, **extra) -> pd.DataFrame:
    """Build a signal frame and evaluate it with the shared harness; label and keep the rows."""
    s, sname = fh._resolve(series, None)
    sf = fh.make_signal_frame(fridays, sig)
    rows = fh.evaluate(sf, s, horizons, timing=TIMING, sides=sides, series_name=sname,
                       labels=dict(filter=FILTER, variant=variant, market=market,
                                   is_primary=False, grid=grid, **extra))
    TESTS.append(rows)
    FRAMES[(grid, variant, market, sname)] = (sf, s)
    return rows


def main() -> None:
    prereg = CODE / "PREREG.md"
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    log(f"PREREG.md sha256 {prereg_sha}")

    series = {m: fh.load_series(TRADED[m]) for m in MARKETS}
    fridays = {m: il.fridays_for(series[m]) for m in MARKETS}

    # =========================================================================================
    # A. Leader-change grid (the primary is the cell gold / usd / L8 / long / h8)
    # =========================================================================================
    for leader in ("usd", "bond", "gold"):
        for L in LOOKBACKS:
            v = f"lead_{leader}_L{L}"
            for m in MARKETS:
                if leader == "gold" and m == "gold":
                    continue                       # gold is not its own leader
                sig = il.leader_signal(leader, fridays[m], L)
                SIGS[(v, m)] = sig
                run_eval("A", v, m, series[m], fridays[m], sig, leader=leader, lookback_w=L,
                         sign=il.LEADERS[leader][2])
            # context: crude on the unadjusted CL continuous (2000..2026)
            cl = fh.load_series("crude_cl")
            fcl = il.fridays_for(cl)
            run_eval("A_ctx", v, "crude", cl, fcl, il.leader_signal(leader, fcl, L),
                     leader=leader, lookback_w=L, sign=il.LEADERS[leader][2])
    log("grid A done")

    primary_sf, _ = FRAMES[("A", PRIMARY["variant"], "gold", "gold")]

    # =========================================================================================
    # B. Larry's literal gold -> bonds rule (gold below its close 20 trading days ago = long bonds)
    # =========================================================================================
    pst_gold, pst_gold_info = il.pst_index("gold")
    gold_inputs = {"gold_v1": series["gold"], "gold_pst": pst_gold}
    oos_extra = {}
    for gname, gin in gold_inputs.items():
        for tname in ("ust10_price", "ust10_tr"):
            ts = fh.load_series(tname)
            fr = il.fridays_for(ts, start=gin.index[0], end=gin.index[-1])
            sig = il.larry_gold_bonds_signal(gin, fr, 20)
            run_eval("B", f"larry_gold20_to_bonds[{gname}]", "ust10", ts, fr, sig,
                     leader="gold", lookback_w=4, sign=-1)
        sf, s = FRAMES[("B", f"larry_gold20_to_bonds[{gname}]", "ust10", "ust10_price")]
        oos_extra[f"split_half_larry_gold20_to_bonds[{gname}]"] = fh.split_half(
            sf, s, timing=TIMING, horizon=8, sides=("long", "short", "net"))
    log("grid B done")

    # =========================================================================================
    # C. Willspread: EMA5 - EMA20 of 100 * market / secondary (daily), long when > 0
    # =========================================================================================
    secondaries = {"gold": (series["gold"], 0), "bond": (fh.load_series("ust10_price"), 0),
                   "usd": (fh.load_series("usd_broad"), il.USD_LAG_DAYS)}
    for m in MARKETS:
        for sec, (xs, lag) in secondaries.items():
            if sec == "gold" and m == "gold":
                continue
            sig = il.willspread_signal(series[m], xs, fridays[m], lag_days=lag, fast=5, slow=20)
            run_eval("C", f"willspread_{sec}_5_20", m, series[m], fridays[m], sig,
                     leader=sec, sign=+1)
            sig2 = il.willspread_signal(series[m], xs, fridays[m], lag_days=lag, fast=7, slow=11)
            run_eval("C", f"willspread_{sec}_7_11", m, series[m], fridays[m], sig2,
                     horizons=(8,), leader=sec, sign=+1)
    log("grid C done")

    # =========================================================================================
    # D. Vote composite at L = 8
    # =========================================================================================
    for m in MARKETS:
        if m == "gold":
            comp = [SIGS[("lead_usd_L8", m)], SIGS[("lead_bond_L8", m)]]
        else:
            comp = [SIGS[("lead_usd_L8", m)], SIGS[("lead_bond_L8", m)], SIGS[("lead_gold_L8", m)]]
        sig = il.vote_signal(comp, 2)
        SIGS[("vote_L8", m)] = sig
        run_eval("D", "vote_L8", m, series[m], fridays[m], sig, lookback_w=8)
    log("grid D done")

    # =========================================================================================
    # E. Extended history on pysystemtrade roll-adjusted indexes (bond and gold leaders)
    # =========================================================================================
    pst_info = {"gold": pst_gold_info}
    pst = {"gold": pst_gold}
    for m in MARKETS:
        if m != "gold":
            pst[m], pst_info[m] = il.pst_index(m)
    for m in MARKETS:
        fr = il.fridays_for(pst[m])
        for L in LOOKBACKS:
            run_eval("E", f"ext_lead_bond_L{L}", m, pst[m], fr,
                     il.leader_signal("bond", fr, L), leader="bond", lookback_w=L, sign=-1)
            if m != "gold":
                sig = il.sign_signal(il.leader_change(pst["gold"], fr, L, 0), +1)
                run_eval("E", f"ext_lead_gold_L{L}", m, pst[m], fr, sig, leader="gold",
                         lookback_w=L, sign=+1)
    sf, s = FRAMES[("E", "ext_lead_bond_L8", "gold", "pst_gold")]
    oos_extra["split_half_ext_lead_bond_L8[gold on pst_gold]"] = fh.split_half(
        sf, s, timing=TIMING, horizon=8, sides=("long", "short", "net"))
    log("grid E done")

    # =========================================================================================
    # F. Own-trend control and descriptive (no forward return) context
    # =========================================================================================
    for m in MARKETS:
        sig = il.own_trend_signal(series[m], fridays[m], 8)
        SIGS[("own_L8", m)] = sig
        run_eval("F", "own_trend_L8", m, series[m], fridays[m], sig, lookback_w=8, sign=+1)
    p_sig, own = SIGS[("lead_usd_L8", "gold")], SIGS[("own_L8", "gold")]
    both = ~np.isnan(p_sig) & ~np.isnan(own) & (p_sig != 0) & (own != 0)
    usd = fh.load_series("usd_broad")
    usd_same = il.leader_change(usd, fridays["gold"], 8, 0)          # same window, NO lag
    usd_lag = il.leader_change(usd, fridays["gold"], 8, il.USD_LAG_DAYS)
    gold_chg = il.leader_change(series["gold"], fridays["gold"], 8, 0)

    def _corr(a, b, step=1):
        ok = np.flatnonzero(~np.isnan(a) & ~np.isnan(b))[::step]
        return dict(corr=float(np.corrcoef(a[ok], b[ok])[0, 1]), n=int(len(ok)))
    context = dict(
        primary_signal_agrees_with_gold_own_8w_trend=dict(
            share=float((p_sig[both] == own[both]).mean()), n_weeks=int(both.sum())),
        corr_usd_chg8_same_window_vs_gold_chg8=dict(
            all_weeks=_corr(usd_same, gold_chg), every_8th_week=_corr(usd_same, gold_chg, 8)),
        corr_usd_chg8_lag7_primary_input_vs_gold_trailing_chg8=dict(
            all_weeks=_corr(usd_lag, gold_chg), every_8th_week=_corr(usd_lag, gold_chg, 8)),
        note="Descriptive only (no forward returns): how strongly the dollar and gold move "
             "together over the same 8 weeks, and how often the primary signal just repeats "
             "gold's own trailing 8-week trend.")
    log("grid F done")

    # =========================================================================================
    # G. Pooled cells
    # =========================================================================================
    pooled = []
    for v in [f"lead_{ld}_L{L}" for ld in ("usd", "bond", "gold") for L in LOOKBACKS] + ["vote_L8"]:
        grid = "D" if v == "vote_L8" else "A"
        parts = [(FRAMES[(grid, v, m, TRADED[m])][0], series[m], TIMING)
                 for m in MARKETS if (grid, v, m, TRADED[m]) in FRAMES]
        pr = fh.evaluate_pooled(parts, H, sides=("long", "short"),
                                labels=dict(filter=FILTER, variant=v, market="pooled",
                                            is_primary=False, grid="G",
                                            n_markets=len(parts)))
        pooled.append(pr)
    pooled_df = pd.concat(pooled, ignore_index=True)
    TESTS.append(pooled_df)
    secondary = pooled_df[(pooled_df["variant"] == "lead_usd_L8") & (pooled_df["side"] == "long")
                          & (pooled_df["horizon_w"] == 8)].iloc[0].to_dict()
    log("grid G done")

    # =========================================================================================
    # H. Primary rule on the pysystemtrade gold futures roll index (2006..2024)
    # =========================================================================================
    fr = il.fridays_for(pst["gold"])
    run_eval("H", "lead_usd_L8", "gold", pst["gold"], fr, il.leader_signal("usd", fr, 8),
             leader="usd", lookback_w=8, sign=-1)
    log("grid H done")

    # =========================================================================================
    # Assemble the tests table, mark the primary, BH q-values (info only)
    # =========================================================================================
    tests = pd.concat(TESTS, ignore_index=True)
    is_p = ((tests["grid"] == "A") & (tests["variant"] == PRIMARY["variant"])
            & (tests["market"] == "gold") & (tests["series"] == "gold")
            & (tests["side"] == "long") & (tests["horizon_w"] == 8))
    assert int(is_p.sum()) == 1, "primary row must be unique"
    tests["is_primary"] = is_p
    h8 = (tests["horizon_w"] == 8) & (tests["market"] != "pooled")
    tests["bh_q_h8"] = np.nan
    tests.loc[h8, "bh_q_h8"] = il.bh_qvalues(tests.loc[h8, "perm_p"].to_numpy())
    prim = tests[is_p].iloc[0].to_dict()
    log(f"tests: {len(tests)} rows; primary excess {prim['excess']:.6f} perm_p {prim['perm_p']}")

    # =========================================================================================
    # 6. Out-of-sample
    # =========================================================================================
    split = fh.split_half(primary_sf, "gold", timing=TIMING, horizon=8,
                          sides=("long", "short", "net"))
    # Identity walk-forward (no estimated parameter: same signal as the static rule)
    gold_fr = fridays["gold"]
    base_frame = pd.DataFrame({"decision_date": gold_fr, "sig": SIGS[("lead_usd_L8", "gold")]})
    wf_id_sf = fh.walk_forward(base_frame, fit=lambda hist, cutoff: None,
                               apply=lambda p, blk: blk["sig"].to_numpy(),
                               first_test=WF_FIRST_TEST)
    wf_id = fh.evaluate(wf_id_sf, "gold", H, timing=TIMING, sides=("long",),
                        labels=dict(filter=FILTER, variant="wf_identity_lead_usd_L8",
                                    market="gold", is_primary=False))

    # Selection walk-forward: leader {usd, bond} x L {4, 8, 13} x sign {prereg, reversed}
    ft = fh.forward_table(gold_fr, "gold", TIMING, (8,))
    wf_frame = pd.DataFrame({"decision_date": gold_fr, "r8": ft["r8"].to_numpy(),
                             "exit_date8": ft["exit_date8"].to_numpy()})
    cfgs = []
    for ld in ("usd", "bond"):
        for L in LOOKBACKS:
            for flip, tag in ((1, "prereg"), (-1, "reversed")):
                name = f"{ld}_L{L}_{tag}"
                wf_frame[name] = il.leader_signal(ld, gold_fr, L, sign=il.LEADERS[ld][2] * flip)
                cfgs.append(name)
    choices = []

    def wf_fit(hist: pd.DataFrame, cutoff: pd.Timestamp):
        """Best long-side h=8 excess among the configs, using only outcomes realised before
        the cutoff; None (no trade) if nothing qualifies or the best excess is <= 0."""
        done = hist[(hist["exit_date8"] < cutoff) & hist["r8"].notna()]
        best, best_ex, table = None, -np.inf, {}
        for c in cfgs:
            el = done[done[c].notna()]
            lg = el[el[c] == 1]
            if len(lg) < WF_MIN_LONG_WEEKS:
                table[c] = None
                continue
            ex = float(lg["r8"].mean() - el["r8"].mean())
            table[c] = ex
            if ex > best_ex:
                best, best_ex = c, ex
        chosen = best if (best is not None and best_ex > 0) else None
        choices.append(dict(cutoff=cutoff.date().isoformat(), chosen=chosen,
                            best_in_sample_excess=None if best is None else best_ex,
                            in_sample_excess=table))
        return chosen

    def wf_apply(params, blk: pd.DataFrame):
        if params is None:
            return np.zeros(len(blk))                      # no trade, week still valid
        x = blk[params].to_numpy(dtype=float)
        return np.where(np.isnan(x), np.nan, np.where(x == 1, 1.0, 0.0))

    wf_sel_sf = fh.walk_forward(wf_frame, wf_fit, wf_apply, first_test=WF_FIRST_TEST)
    wf_sel = fh.evaluate(wf_sel_sf, "gold", H, timing=TIMING, sides=("long",),
                         labels=dict(filter=FILTER, variant="wf_selection", market="gold",
                                     is_primary=False))
    # Static primary over the same 2011+ weeks
    same = primary_sf.copy()
    m = same["decision_date"] >= pd.Timestamp(WF_FIRST_TEST)
    same["valid"] = same["valid"] & m
    same["signal"] = np.where(same["valid"], same["signal"], 0).astype(np.int8)
    static_2011 = fh.evaluate(same, "gold", H, timing=TIMING, sides=("long",),
                              labels=dict(filter=FILTER, variant="static_primary_2011plus",
                                          market="gold", is_primary=False))
    log("OOS done")

    # =========================================================================================
    # 7. Strategy
    # =========================================================================================
    st = fh.simulate(primary_sf, "gold", timing=TIMING,
                     labels=dict(filter=FILTER, variant=PRIMARY["variant"], market="gold"))
    log("primary strategy done")
    ctx_metrics = []
    for m in ("crude", "silver", "soybean", "sugar"):
        sf, s = FRAMES[("A", "lead_usd_L8", m, TRADED[m])]
        r = fh.simulate(sf, s, timing=TIMING, keep_equity=False,
                        labels=dict(filter=FILTER, variant="lead_usd_L8", market=m))
        ctx_metrics.append(r["metrics"])
    sf, s = FRAMES[("B", "larry_gold20_to_bonds[gold_v1]", "ust10", "ust10_price")]
    r = fh.simulate(sf, s, timing=TIMING, keep_equity=False,
                    labels=dict(filter=FILTER, variant="larry_gold20_to_bonds[gold_v1]",
                                market="ust10"))
    ctx_metrics.append(r["metrics"])
    ctx_metrics = pd.concat(ctx_metrics, ignore_index=True)
    log("context strategies done")

    # =========================================================================================
    # Verdict (pre-declared criteria, PREREG section 4)
    # =========================================================================================
    sp = split[(split["side"] == "long") & (split["horizon_w"] == 8)].set_index("half")
    wf8 = wf_sel[wf_sel["horizon_w"] == 8].iloc[0]
    met = st["metrics"].set_index("strategy")
    s1lo = met.loc["S1_long_only"]
    checks = dict(
        primary_excess_gt0=bool(prim["excess"] > 0),
        primary_perm_p_le_005=bool(prim["perm_p"] <= 0.05),
        primary_perm_p_le_020=bool(prim["perm_p"] <= 0.20),
        split_H1_excess_gt0=bool(sp.loc["H1", "excess"] > 0),
        split_H2_excess_gt0=bool(sp.loc["H2", "excess"] > 0),
        walk_forward_selection_excess_gt0=bool(wf8["excess"] > 0),
        S1_long_only_sharpe_gt0=bool(s1lo["sharpe"] > 0),
        S1_long_only_null_p_le_010=bool(s1lo["null_p_sharpe"] <= 0.10),
    )
    robust = all(checks[k] for k in ("primary_excess_gt0", "primary_perm_p_le_005",
                                     "split_H1_excess_gt0", "split_H2_excess_gt0",
                                     "walk_forward_selection_excess_gt0",
                                     "S1_long_only_sharpe_gt0", "S1_long_only_null_p_le_010"))
    if robust:
        verdict = "positive EV, robust"
    elif checks["primary_excess_gt0"] and checks["primary_perm_p_le_020"]:
        verdict = "positive EV, fragile"
    elif prim["excess"] < 0 and prim["perm_p"] >= 0.95:
        verdict = "negative (anti-signal)"
    else:
        verdict = "no edge"
    log(f"verdict: {verdict}  {checks}")

    # =========================================================================================
    # Write outputs
    # =========================================================================================
    primary_out = dict(
        rule="gold LONG when the broad dollar (DTWEXBGS, read as of Friday-7d) fell over 8 weeks; "
             "h=8w; entry first gold close after the Friday; traded on v1 London gold",
        **{k: prim[k] for k in ("filter", "variant", "market", "series", "timing", "side",
                                "horizon_w", "n", "episodes", "mean", "median", "baseline",
                                "excess", "hit", "perm_p", "perm_p_exact", "perm_p_exact_raw",
                                "T", "period", "is_primary")},
        split_half_long_h8={h: dict(excess=float(sp.loc[h, "excess"]),
                                    perm_p=float(sp.loc[h, "perm_p"]), n=int(sp.loc[h, "n"]),
                                    period=sp.loc[h, "period"]) for h in ("H1", "H2")},
        walk_forward_selection_long_h8=dict(excess=float(wf8["excess"]),
                                            perm_p=float(wf8["perm_p"]), n=int(wf8["n"]),
                                            period=wf8["period"]),
        S1_long_only=dict(sharpe=float(s1lo["sharpe"]), ann_return=float(s1lo["ann_return"]),
                          max_dd=float(s1lo["max_dd"]), n_trades=int(s1lo["n_trades"]),
                          win_rate=float(s1lo["win_rate"]),
                          null_p_sharpe=float(s1lo["null_p_sharpe"]),
                          bh_sharpe=float(s1lo["bh_sharpe"]),
                          bh_ann_return=float(s1lo["bh_ann_return"])),
    )
    oos = dict(
        split_half=split,
        walk_forward_identity=dict(
            note="The primary has no estimated parameter, so its walk-forward signal equals the "
                 "static rule; evaluated from 2011-01-01.", rows=wf_id),
        walk_forward_selection=dict(
            note="Each 1 Jan from 2011 pick, among leader {usd, bond} x L {4,8,13} x sign "
                 "{prereg, reversed}, the config with the best long-side h=8 excess on gold using "
                 "only weeks whose 8-week exit is before the cutoff (>= 26 long weeks); no trade "
                 "if the best excess <= 0.",
            rows=wf_sel, static_primary_same_window=static_2011, yearly_choices=choices),
        **oos_extra,
    )
    caveats = [
        "No verbatim Larry Williams rule for using the dollar, bonds or gold to time crude, "
        "silver, soybeans or sugar could be found; all sources are search-engine extracts "
        "(the pages themselves are unreachable from this sandbox). The primary is the battery's "
        "suggested dollar-change rule applied to gold, the commodity Larry ties to the dollar.",
        "Larry's only explicit intermarket rule with a parameter (gold below its close 20 days "
        "ago = buy bonds) comes from a third-party reconstruction of his GSV bond system, and it "
        "trades bonds, not a commodity. It is tested in grid B on a 10y price proxy (6% coupon "
        "bond at DGS10), not on actual T-bond futures.",
        "The broad dollar (DTWEXBGS) is a weekly-published Fed index, read with a 7-day lag; "
        "Larry would watch the ICE dollar index (DX) futures in real time. Same-day dollar data "
        "would shorten the lag by a week; it is not tested because the archive has no DX series.",
        "Gold is traded on the London fix series (no futures carry). The pst_gold rows (grid H, "
        "E) use a pysystemtrade futures roll index, whose post-2013 prints are sometimes hourly "
        "stamps rather than settlements.",
        "Willspread's sign convention and parameters come from third-party code (5/20 EMAs; "
        "MetaStock 7/11); the opposite reading equals the short side of each row.",
        "Overlapping 8-week returns: only the circular-shift permutation p-values are evidence; "
        "BH q-values over all non-pooled h=8 rows are information only, and no grid cell is "
        "promoted to the result.",
        "crude_roll ends 2024-04-05, so crude rows end there; crude_cl context rows run to 2026.",
        "A leader-change signal is almost never exactly 0, so its long and short setups are "
        "complements: the shifted long mean and the shifted short mean move one-for-one, and the "
        "long, short and net rows of one frame share the same permutation p. They are one test, "
        "not three.",
        "PREREG.md's header says 09:10Z; the file was hashed at 2026-10-16T09:09:46Z, before the "
        "first backtest run (run_log.txt).",
    ]
    # Computed grid summary (distinct long-side h=8 frames, non-pooled), information only
    lh8 = tests[(tests["horizon_w"] == 8) & (tests["side"] == "long")
                & (tests["market"] != "pooled")]
    grid_summary = dict(
        long_h8_frames=int(len(lh8)),
        long_h8_frames_perm_p_le_005=int((lh8["perm_p"] <= 0.05).sum()),
        long_h8_frames_excess_gt0=int((lh8["excess"] > 0).sum()),
        h8_rows_in_bh=int(h8.sum()),
        min_bh_q_h8=float(tests.loc[h8, "bh_q_h8"].min()),
        h8_rows_bh_q_le_010=int((tests.loc[h8, "bh_q_h8"] <= 0.10).sum()),
    )
    log(f"grid summary: {grid_summary}")
    larry_def = (
        "Larry Williams uses related markets as leading indicators: he looks at the short-term "
        "trend of the market that 'influences' the one he trades (bonds influence stocks; gold "
        "influences bonds, inversely; the dollar is tied to gold; inflation follows gold). His "
        "Willspread (Long-Term Secrets to Short-Term Trading) measures the flow of price between "
        "the traded market and its influencing market as EMA(5)-EMA(20) of 100*primary/secondary; "
        "his bond system (third-party reconstruction) buys bonds only when gold is below its close "
        "20 days earlier. Tested here as: follow the leader's L-week change (dollar and bond "
        "inverse, gold direct), with the primary = gold long when the broad dollar's 8-week change "
        "is negative.")
    sources = [
        "Stocks & Commodities interview with Larry Williams, July 1997, "
        "traders.com/Documentation/FEEDbk_docs/1997/07/0797Williams.html (search extract)",
        "Larry Williams, Long-Term Secrets to Short-Term Trading (Wiley 1999; 2nd ed. 2011), "
        "Willspread chapter, quoted in usethinkscript.com thread 6049 (p.170, 2nd ed.; extract)",
        "Willspread formula: MQL5 code base 8528; TradingView script Rak6df4K; "
        "traderyam.blogspot.com/2016/03; MetaStock forum m164619 (7/11 EMAs) (extracts)",
        "GSV T-bond system reconstruction with gold 20-day filter, TradingView chart idea 0uyx2uHR "
        "(extract)",
        "Larry Williams Investors Daily Edge interview, ireallytrade.com/investorsdailyedge.html "
        "(extract)",
        "StockCharts ChartWatchers 2023-03, 'Trading Legend Larry Williams' 2023 Market Forecast "
        "and Trading Setups' (extract)",
        "Mail & Guardian 2006 interview, mg.co.za/?p=319359 (extract)",
        "emini-watch.com, 'Larry Williams' Trading Strategy: What I Saw Him Do Live' (extract)",
        "Data: FRED DTWEXBGS and DGS10, LBMA gold/silver, CBOT/ICE continuous (v1), EIA RCLC1/2 "
        "(crude_roll), pysystemtrade multiple_prices_csv @bbe29e19 (pst indexes)",
    ]
    charts = dict(
        equity_primary=fh.weekly_equity(st["equity"]),
        primary_signal=dict(dates=[d.date().isoformat() for d in gold_fr],
                            usd_chg8_lag7=[None if np.isnan(x) else round(float(x), 6)
                                           for x in usd_lag],
                            signal=[None if np.isnan(x) else int(x)
                                    for x in SIGS[("lead_usd_L8", "gold")]]),
        h8_long_excess_by_cell=fh.to_jsonable(
            tests[(tests["horizon_w"] == 8) & (tests["side"] == "long")
                  & (tests["market"] != "pooled")][
                ["grid", "variant", "market", "series", "n", "excess", "perm_p", "bh_q_h8"]]),
    )
    fh.write_results(
        HERE / "results.json", filter=FILTER, larry_definition=larry_def, sources=sources,
        primary=primary_out, tests=tests, strategy=st["metrics"], oos=oos, caveats=caveats,
        verdict=verdict, verdict_checks=checks, prereg_sha256=prereg_sha,
        secondary_headline_pooled_usd_L8_long_h8=secondary, strategy_context=ctx_metrics,
        context=context, grid_summary=grid_summary, pst_index_info=pst_info, charts=charts,
        run_info=dict(script="run_intermarket.py", lib="intermarket_lib.py",
                      harness="battery/harness/fund_harness.py", k_perm=fh.K_PERM, seed=fh.SEED,
                      timing=TIMING, generated_utc=pd.Timestamp.now(tz="UTC").isoformat()))
    tests.to_csv(HERE / "tests.csv", index=False)
    st["trades"].to_csv(HERE / "trades_primary.csv", index=False)
    log("wrote results.json, tests.csv, trades_primary.csv")
    (HERE / "run_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

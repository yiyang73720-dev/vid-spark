#!/usr/bin/env python3
"""
run_confluence -- step 3 of the rank step: does requiring >= 2 of Larry's tools to point the
same way beat each tool alone?  Implements PREREG.md sections 1-5 exactly.

PRIMARY: crude, components {cot, premium, seasonal, valuation, oi, dollar} on the COT report
grid (timing cot), long if L >= 2 and L > S (short mirror-image), LONG side, h = 8 weeks, traded
on crude_roll. Every single-tool comparison uses the same eligible weeks (same baseline, same T).

Outputs (this directory):
  confluence_results.json   primary, tests H-a / H-b, verdict, halves, strategy, secondary
  confluence_tests.csv      every confluence / single-component test row (harness columns)
  confluence_weekly_crude.csv  the crude common-grid table (components, confluence, r8)

Run: python3 -I run_confluence.py
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

fh = rl.fh
HERE = rl.RANK                      # outputs: ROOT/work2/rank
HERE.mkdir(parents=True, exist_ok=True)
H = (1, 4, 8, 13)
SIDES = ("long", "short", "net")
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    LOG.append(line)
    print(line, flush=True)


def frame(df: pd.DataFrame, sig: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    """Signal frame on the report grid: undefined outside `valid`, shutdown weeks excluded."""
    return fh.make_signal_frame(df["report_date"], np.where(valid, sig, np.nan),
                                excluded=df["shutdown"].to_numpy(bool))


def ev(sf, series, *, variant, market, horizons=H, sides=SIDES, comps="", rule="") -> pd.DataFrame:
    """fh.evaluate with confluence labels (timing cot)."""
    return fh.evaluate(sf, series, horizons, timing="cot", sides=sides,
                       labels=dict(filter="confluence", variant=variant, market=market,
                                   is_primary=False, components=comps, rule=rule))


def main() -> None:
    prereg_sha = hashlib.sha256((rl.RANK_CODE / "PREREG.md").read_bytes()).hexdigest()
    stored = (rl.RANK_CODE / "PREREG.sha256").read_text().split()[0]
    if prereg_sha != stored:
        raise SystemExit("PREREG.md changed after it was hashed")
    log(f"PREREG.md sha256 {prereg_sha} (matches PREREG.sha256)")

    # ------------------------------------------------------------------------------------
    # Components on each market's COT report grid
    # ------------------------------------------------------------------------------------
    comp = {m: rl.report_grid_components(m) for m in rl.MARKETS5}
    comps_of = {m: [c for c in rl.COMPONENTS if c in comp[m].columns] for m in rl.MARKETS5}
    valid_of = {m: comp[m][comps_of[m]].notna().all(axis=1).to_numpy() for m in rl.MARKETS5}
    for m in rl.MARKETS5:
        d = comp[m]["report_date"][valid_of[m]]
        log(f"{m}: components {comps_of[m]}; all defined on {int(valid_of[m].sum())} reports "
            f"({d.min().date()}..{d.max().date()})")
    tests: list[pd.DataFrame] = []

    # ------------------------------------------------------------------------------------
    # PRIMARY (crude) and the single components on the same weeks
    # ------------------------------------------------------------------------------------
    cdf, comps, valid = comp["crude"], comps_of["crude"], valid_of["crude"]
    conf = rl.confluence_signal(cdf, comps, 2, "majority")
    sf_conf = frame(cdf, conf, valid)
    prim_rows = ev(sf_conf, "crude_roll", variant="primary_k2_majority", market="crude",
                   comps="+".join(comps), rule="L>=2 & L>S")
    prim_rows["is_primary"] = (prim_rows["side"] == "long") & (prim_rows["horizon_w"] == 8)
    tests.append(prim_rows)
    primary = prim_rows[prim_rows["is_primary"]].iloc[0].to_dict()
    log("PRIMARY confluence crude long h8: " + ", ".join(
        f"{k}={primary[k]}" for k in ("n", "episodes", "mean", "baseline", "excess", "hit",
                                       "perm_p", "perm_p_exact", "T", "period")))

    single_rows = []
    sf_comp = {}
    for c in comps:
        sf_comp[c] = frame(cdf, cdf[c].to_numpy(float), valid)
        rows = ev(sf_comp[c], "crude_roll", variant=f"single_{c}", market="crude", comps=c,
                  rule="component alone, common weeks")
        single_rows.append(rows)
    single_rows = pd.concat(single_rows, ignore_index=True)
    tests.append(single_rows)

    # Common eligible weeks at h=8: the same rows for the confluence and every component
    el = rl.eligible_rows(sf_conf, "crude_roll", "cot", 8)
    wk = pd.DataFrame({"report_date": el["decision_date"], "confluence": el["signal"],
                       "r8": el["r"]})
    for c in comps:
        e2 = rl.eligible_rows(sf_comp[c], "crude_roll", "cot", 8)
        if not np.array_equal(e2["decision_date"].to_numpy(), el["decision_date"].to_numpy()):
            raise AssertionError(f"eligible weeks of {c} differ from the confluence's")
        wk[c] = e2["signal"].to_numpy()
    wk.to_csv(HERE / "confluence_weekly_crude.csv", index=False)

    # H-a and H-b
    r8 = wk["r8"].to_numpy()
    cs = wk["confluence"].to_numpy()
    h_b = {}
    for c in comps:
        xs = wk[c].to_numpy()
        n_long = int((xs == 1).sum())
        d = rl.bootstrap_diff(cs, xs, r8, "long")
        xrow = single_rows[(single_rows["variant"] == f"single_{c}")
                           & (single_rows["side"] == "long")
                           & (single_rows["horizon_w"] == 8)].iloc[0]
        h_b[c] = dict(n_long_weeks=n_long, eligible_for_test=n_long >= 10,
                      excess_component=float(xrow["excess"]),
                      perm_p_exact_component=float(xrow["perm_p_exact"]),
                      excess_confluence=float(primary["excess"]), **d,
                      passes=bool(n_long >= 10 and d["diff"] > 0 and d["p_boot"] <= 0.05))
        log(f"H-b vs {c}: n_long={n_long} D={d['diff']:.6f} p_boot={d['p_boot']:.4f} "
            f"CI=[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
    h_a = bool(primary["excess"] > 0 and primary["perm_p_exact"] <= 0.05)
    tested = [c for c in comps if h_b[c]["eligible_for_test"]]
    hb_all = all(h_b[c]["passes"] for c in tested)
    comp_exc = [h_b[c]["excess_component"] for c in tested]
    if h_a and hb_all:
        verdict = "confluence beats each tool alone"
    elif primary["excess"] > 0 and all(primary["excess"] > x for x in comp_exc):
        verdict = "confluence better than every single tool but not significantly"
    else:
        verdict = "confluence does not beat the tools alone"
    log(f"VERDICT: {verdict} (H-a {h_a}; H-b all {hb_all})")

    boot_primary = rl.bootstrap_excess(cs, r8, "long")

    # Split halves, strategy
    halves = fh.split_half(sf_conf, "crude_roll", timing="cot", horizon=8,
                           sides=("long", "short", "net"))
    log("split halves: " + "; ".join(
        f"{r.half} {r.side} n={r.n} excess={r.excess:.6f} p={r.perm_p_exact:.4f}"
        for r in halves.itertuples()))
    st = fh.simulate(sf_conf, "crude_roll", timing="cot",
                     labels=dict(filter="confluence", variant="primary_k2_majority",
                                 market="crude"))
    met = st["metrics"]
    log("strategy: " + "; ".join(f"{r.strategy} sharpe={r.sharpe:.3f}" for r in met.itertuples()))
    st["trades"].to_csv(HERE / "confluence_trades_primary.csv", index=False)

    # ------------------------------------------------------------------------------------
    # Secondary 2: rule variants on crude (same eligible weeks)
    # ------------------------------------------------------------------------------------
    variants = {
        "k3_majority": (comps, 3, "majority", None, "L>=3 & L>S"),
        "k2_unanimous": (comps, 2, "unanimous", None, "L>=2 & S==0"),
        "event_only_k2": (["cot", "seasonal", "valuation", "oi"], 2, "majority", None,
                          "event tools only, L>=2 & L>S"),
        "cot_anchored": (comps, 2, "majority", "cot",
                         "cot==+1 & others L_o>=1 & L_o>S_o"),
    }
    var_sf = {}
    for v, (cc, need, rule, anchor, desc) in variants.items():
        s = rl.confluence_signal(cdf, cc, need, rule, anchor)
        var_sf[v] = frame(cdf, s, valid)
        tests.append(ev(var_sf[v], "crude_roll", variant=v, market="crude", comps="+".join(cc),
                        rule=desc))
    log("variants done")

    # Secondary 4: primary on other crude series
    for ser in ("crude_roll_lead5", "crude_cl"):
        tests.append(ev(sf_conf, ser, variant="primary_k2_majority", market="crude",
                        comps="+".join(comps), rule="L>=2 & L>S"))
    log("other crude series done")

    # ------------------------------------------------------------------------------------
    # Secondary 3: pooled 5 markets
    # ------------------------------------------------------------------------------------
    parts, stack = [], []
    per_market = []
    for m in rl.MARKETS5:
        df, cc, vv = comp[m], comps_of[m], valid_of[m]
        s = rl.confluence_signal(df, cc, 2, "majority")
        sf = frame(df, s, vv)
        parts.append((sf, rl.TRADED5[m], "cot"))
        per_market.append(ev(sf, rl.TRADED5[m], variant="primary_rule_market", market=m,
                             horizons=(8,), comps="+".join(cc), rule="L>=2 & L>S"))
        e = rl.eligible_rows(sf, rl.TRADED5[m], "cot", 8)
        t = pd.DataFrame({"market": m, "report_date": e["decision_date"],
                          "week": rl.friday_of_report_week(e["decision_date"]),
                          "confluence": e["signal"], "r8": e["r"]})
        dd = df.set_index("report_date").loc[pd.DatetimeIndex(e["decision_date"])]
        for c in rl.COMPONENTS:
            # a component that does not exist for m (valuation for gold) never sets up there
            t[c] = dd[c].to_numpy(float) if c in cc else 0.0
        stack.append(t)
    per_market = pd.concat(per_market, ignore_index=True)
    tests.append(per_market)
    pooled = fh.evaluate_pooled(parts, H, sides=("long", "short"),
                                labels=dict(filter="confluence", variant="pooled5_k2_majority",
                                            market="pooled5", is_primary=False))
    tests.append(pooled)
    stack = pd.concat(stack, ignore_index=True)
    stack.to_csv(HERE / "confluence_weekly_pooled5.csv", index=False)
    grp, cal = stack["market"].to_numpy(), stack["week"].to_numpy()
    pooled_conf_long = pooled[(pooled["side"] == "long") & (pooled["horizon_w"] == 8)].iloc[0]
    pooled_hb = {}
    for c in rl.COMPONENTS:
        xs = stack[c].to_numpy(float)
        d = rl.bootstrap_diff(stack["confluence"].to_numpy(float), xs, stack["r8"].to_numpy(),
                              "long", group=grp, cal=cal)
        pooled_hb[c] = dict(n_long_weeks=int((xs == 1).sum()),
                            excess_component_vs_own_baseline=rl.excess_from_rows(
                                xs, stack["r8"].to_numpy(), "long", grp), **d)
        log(f"pooled H-b vs {c}: D={d['diff']:.6f} p_boot={d['p_boot']:.4f}")
    pooled_boot = rl.bootstrap_excess(stack["confluence"].to_numpy(float), stack["r8"].to_numpy(),
                                      "long", group=grp, cal=cal)
    # single components pooled (fh.evaluate_pooled, MC p), same eligible weeks per market
    pooled_single = []
    for c in rl.COMPONENTS:
        pp = []
        for m in rl.MARKETS5:
            if c not in comps_of[m]:
                continue
            pp.append((frame(comp[m], comp[m][c].to_numpy(float), valid_of[m]),
                       rl.TRADED5[m], "cot"))
        r = fh.evaluate_pooled(pp, (8,), sides=("long", "short"),
                               labels=dict(filter="confluence", variant=f"pooled_single_{c}",
                                           market=f"pooled{len(pp)}", is_primary=False))
        pooled_single.append(r)
    pooled_single = pd.concat(pooled_single, ignore_index=True)
    tests.append(pooled_single)
    log("pooled 5 markets done")

    # ------------------------------------------------------------------------------------
    # Secondary 5: pairwise agreement of the components on the crude common weeks
    # ------------------------------------------------------------------------------------
    agree = []
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            x, y = wk[a].to_numpy(), wk[b].to_numpy()
            both = (x != 0) & (y != 0)
            agree.append(dict(a=a, b=b, n_both_active=int(both.sum()),
                              share_same_sign_when_both_active=float((x[both] == y[both]).mean())
                              if both.any() else None,
                              corr=float(np.corrcoef(x, y)[0, 1])
                              if x.std() > 0 and y.std() > 0 else None))
    activity = {c: dict(long=int((wk[c] == 1).sum()), short=int((wk[c] == -1).sum()),
                        none=int((wk[c] == 0).sum())) for c in comps + ["confluence"]}

    # ------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------
    tdf = pd.concat(tests, ignore_index=True)
    tdf.to_csv(HERE / "confluence_tests.csv", index=False)
    out = dict(
        prereg_sha256=prereg_sha,
        primary_rule="crude; components cot, premium, seasonal, valuation, oi, dollar on the "
                     "COT report grid (timing cot); long if L>=2 and L>S, short if S>=2 and S>L;"
                     " LONG side, h=8 weeks, traded on crude_roll",
        common_weeks=dict(n_eligible_h8=int(len(wk)),
                          first=str(wk["report_date"].min().date()),
                          last=str(wk["report_date"].max().date())),
        primary=primary,
        primary_bootstrap_excess_ci=boot_primary,
        H_a_beats_random_timing=dict(passes=h_a, excess=primary["excess"],
                                     perm_p_exact=primary["perm_p_exact"],
                                     perm_p_mc=primary["perm_p"]),
        H_b_beats_each_component=h_b,
        verdict=verdict,
        single_components_common_weeks=single_rows,
        split_half=halves,
        strategy=met,
        walk_forward="identity: no fitted parameter (PREREG 4); every component uses only "
                     "information available at its decision time",
        activity_counts_common_weeks=activity,
        pairwise_agreement=agree,
        variants=tdf[tdf["variant"].isin(list(variants))],
        other_series=tdf[(tdf["variant"] == "primary_k2_majority")
                         & tdf["series"].isin(["crude_roll_lead5", "crude_cl"])],
        pooled5=dict(cell=pooled, per_market=per_market,
                     bootstrap_excess_long_h8=pooled_boot, H_b_long_h8=pooled_hb,
                     single_components=pooled_single,
                     note="pooled excess = vs each market's own baseline; perm_p is the "
                          "Monte-Carlo pooled shift (no exhaustive version)"),
        charts=dict(equity_primary=fh.weekly_equity(st["equity"])),
        run_log=LOG, run_seconds=round(time.time() - T0, 1))
    (HERE / "confluence_results.json").write_text(json.dumps(fh.to_jsonable(out), indent=1))
    log("wrote confluence_results.json, confluence_tests.csv")


if __name__ == "__main__":
    main()

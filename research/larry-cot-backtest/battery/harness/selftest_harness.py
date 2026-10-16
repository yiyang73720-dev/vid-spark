#!/usr/bin/env python3
"""
selftest_harness -- checks that fund_harness reproduces the v1 COT study exactly and that its
new pieces (roll-adjusted crude index, timing rules, OOS helpers) behave as documented.

Usage:  python3 -I selftest_harness.py [--fast]
        --fast skips the 2000-draw strategy null reproduction (the slowest check).
Writes ROOT/work2/harness/selftest_output.json and exits non-zero on any failed check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import numpy as np        # noqa: E402
import pandas as pd       # noqa: E402
import fund_harness as fh  # noqa: E402

lc = fh.lc
V1 = fh.WORK_V1
RESULTS: dict = {}
FAILED: list[str] = []


def check(name: str, ok: bool, detail=None) -> None:
    RESULTS.setdefault("checks", []).append(dict(check=name, ok=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"))
    if not ok:
        FAILED.append(name)


# ----------------------------------------------------------------------------------------
# A. v1 canonical table (360 cells) and the headline
# ----------------------------------------------------------------------------------------
def test_v1_canonical(panel) -> None:
    ref = pd.read_csv(V1 / "engine" / "canonical_exact_p.csv", float_precision="round_trip")
    rows = []
    for m in lc.MARKETS:
        for n in lc.LOOKBACKS:
            for bt, st in lc.THRESHOLDS:
                sf = fh.cot_signal_frame(m, f"idx{n}", bt, st, panel=panel)
                r = fh.evaluate(sf, panel_prices[m], lc.HORIZONS_W, timing="cot",
                                sides=("long", "short"), series_name=m)
                r["market"], r["lookback"], r["buy_th"], r["sell_th"] = m, n, bt, st
                rows.append(r)
    mine = pd.concat(rows, ignore_index=True)
    key = ["market", "lookback", "buy_th", "sell_th", "side", "horizon_w"]
    j = ref.merge(mine, on=key, how="outer", indicator=True)
    check("v1 canonical: same 360 cells", (j["_merge"] == "both").all() and len(j) == 360,
          int(len(j)))
    pairs = [("n_signal_weeks", "n"), ("n_episodes", "episodes"), ("mean_ret", "mean"),
             ("median_ret", "median"), ("hit_rate", "hit"), ("baseline_mean", "baseline"),
             ("excess", "excess_y"), ("perm_p", "perm_p_y"), ("perm_p_exact", "perm_p_exact_raw")]
    j = j.rename(columns={"excess_x": "excess_ref", "perm_p_x": "perm_p_ref",
                          "perm_p_exact_x": "perm_p_exact_ref"})
    bad = {}
    for a, b in pairs:
        a2 = {"excess": "excess_ref", "perm_p": "perm_p_ref",
              "perm_p_exact": "perm_p_exact_ref"}.get(a, a)
        x, y = j[a2].to_numpy(float), j[b].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        bad[a] = int((~same).sum())
    check("v1 canonical: every column bit-identical in all 360 cells",
          all(v == 0 for v in bad.values()), bad)
    h = mine[(mine.market == "crude") & (mine.lookback == 26) & (mine.buy_th == 80)
             & (mine.side == "long") & (mine.horizon_w == 8)].iloc[0]
    head = {c: h[c] for c in ("n", "episodes", "mean", "median", "baseline", "excess", "hit",
                              "perm_p", "perm_p_exact", "perm_p_exact_raw", "T", "period")}
    RESULTS["v1_headline_reproduced"] = head
    check("headline crude N26 80/20 long h8 = 273 weeks / 51 episodes",
          head["n"] == 273 and head["episodes"] == 51, (head["n"], head["episodes"]))
    check("headline mean/baseline/excess exact",
          head["mean"] == -0.013815102282036938 and head["baseline"] == 0.018282105406205888
          and head["excess"] == -0.032097207688242826,
          (head["mean"], head["baseline"], head["excess"]))
    check("headline perm_p (seed 11) and exhaustive p equal v1",
          head["perm_p"] == 0.9625187406296851
          and head["perm_p_exact_raw"] == 0.9596602972399151,
          (head["perm_p"], head["perm_p_exact_raw"], head["perm_p_exact"]))


# ----------------------------------------------------------------------------------------
# B. v1 pooled headline
# ----------------------------------------------------------------------------------------
def test_v1_pooled(panel) -> None:
    ref = pd.read_csv(V1 / "engine" / "pooled.csv", float_precision="round_trip")
    ref = ref[(ref.lookback == 26) & (ref.buy_th == 80) & (ref.side == "long")
              & (ref.horizon_w == 8)].iloc[0]
    parts = [(fh.cot_signal_frame(m, panel=panel), panel_prices[m], "cot") for m in lc.MARKETS]
    p = fh.evaluate_pooled(parts, (8,), sides=("long",)).iloc[0]
    ok = (p["n"] == ref["n_signal_weeks"] and p["episodes"] == ref["n_episodes"]
          and p["mean"] == ref["mean_ret"] and p["excess"] == ref["excess_vs_own_baseline"]
          and p["perm_p"] == ref["perm_p"])
    check("v1 pooled N26 80/20 long h8 reproduced exactly", ok,
          dict(n=int(p["n"]), mean=p["mean"], excess=p["excess"], perm_p=p["perm_p"]))


# ----------------------------------------------------------------------------------------
# C. v1 strategy lens (crude S1/S2) incl. the random-timing null
# ----------------------------------------------------------------------------------------
def test_v1_strategy(panel, null_k: int) -> None:
    ref = pd.read_csv(V1 / "lens_strategy" / "metrics.csv", float_precision="round_trip")
    ref = ref[ref.market == "crude"].drop_duplicates("strategy").set_index("strategy")
    sf = fh.cot_signal_frame("crude", panel=panel)
    t0 = time.time()
    res = fh.simulate(sf, "crude_cl", timing="cot", null_k=null_k)
    met = res["metrics"].set_index("strategy")
    cols = ["n_trades", "win_rate", "avg_trade", "ann_return", "ann_vol", "sharpe", "max_dd",
            "time_in_mkt"] + (["null_p_sharpe", "null_sharpe_median"] if null_k else [])
    diffs = {}
    for st in fh.STRATEGIES:
        for c in cols:
            if float(met.loc[st, c]) != float(ref.loc[st, c]):
                diffs[f"{st}.{c}"] = (float(met.loc[st, c]), float(ref.loc[st, c]))
    for c in ("ann_return", "sharpe", "max_dd"):
        if float(met.loc["buy_hold", c]) != float(ref.loc["buy_hold", c]):
            diffs[f"buy_hold.{c}"] = (float(met.loc["buy_hold", c]), float(ref.loc["buy_hold", c]))
    check(f"v1 strategy lens crude S1/S2 metrics identical (null_k={null_k})", not diffs,
          diffs or f"{time.time() - t0:.1f}s")
    RESULTS["v1_strategy_reproduced"] = fh.to_jsonable(
        res["metrics"][["strategy", "n_trades", "win_rate", "ann_return", "sharpe", "max_dd"]
                       + (["null_p_sharpe"] if null_k else [])])


# ----------------------------------------------------------------------------------------
# D. Roll-adjusted crude index
# ----------------------------------------------------------------------------------------
def test_roll_index() -> None:
    ri = fh.build_crude_roll_index(0)
    v = fh.validate_roll_index(ri)
    RESULTS["roll_index_info"] = {k: x for k, x in ri.info.items()
                                  if k != "eia_gap_days_treated_as_business"}
    RESULTS["roll_index_validation"] = v
    d = ri.detail
    # Identity: on a non-roll day after holding contract 1, the index return is CL1's return.
    prev_held = d["held"].shift(1)
    m = (~d["roll"]) & (prev_held == 1)
    m.iloc[0] = False
    check("roll index: non-roll days holding contract 1 earn exactly the CL1 return",
          bool((d.loc[m, "ret"] == d.loc[m, "naive_ret"]).all()), int(m.sum()))
    check("roll index: one identity change per LTD inside the data",
          v["roll_day_adjusted"]["n"] == ri.info["n_roll_days"]
          and ri.info["n_roll_days"] == int(((ri.ltd >= d.index[0])
                                             & (ri.ltd < d.index[-1])).sum()),
          (ri.info["n_roll_days"], int(len(ri.ltd))))
    check("roll days are no longer jumps: corr(naive - adjusted, spread) > 0.99",
          v["corr_naive_minus_adjusted_vs_spread"] > 0.99,
          round(v["corr_naive_minus_adjusted_vs_spread"], 4))
    check("roll days are no longer jumps: |corr(adjusted, spread)| < 0.1",
          abs(v["corr_adjusted_vs_spread"]) < 0.1, round(v["corr_adjusted_vs_spread"], 4))
    ratio = v["roll_day_adjusted"]["std"] / v["other_days"]["std"]
    check("roll-day adjusted return std within 15% of other days' std", abs(ratio - 1) < 0.15,
          dict(adjusted_roll_std=round(v["roll_day_adjusted"]["std"], 5),
               naive_roll_std=round(v["roll_day_naive"]["std"], 5),
               other_std=round(v["other_days"]["std"], 5)))
    check("calendar roll days confirmed by back-contract data score (share > 0 well above "
          "non-roll base rate)",
          v["share_score_positive_on_roll_days"] > v["share_score_positive_on_other_days"] + 0.3,
          (round(v["share_score_positive_on_roll_days"], 4),
           round(v["share_score_positive_on_other_days"], 4)))
    check("roll index strictly positive", v["min_level"] > 0, round(v["min_level"], 4))
    ri5 = fh.build_crude_roll_index(5)
    a = ri.level.pct_change().dropna()
    b = ri5.level.reindex(ri.level.index).pct_change().dropna()
    RESULTS["roll_lead5_vs_0"] = dict(
        total_return_lead0=float(ri.level.iloc[-1] / 100 - 1),
        total_return_lead5=float(ri5.level.iloc[-1] / 100 - 1),
        weekly_corr=float(ri.level.resample("W-FRI").last().pct_change()
                          .corr(ri5.level.resample("W-FRI").last().pct_change())))
    check("roll_lead=5 variant builds and differs only around rolls",
          float((a - b.reindex(a.index)).abs().gt(1e-12).mean()) < 0.35,
          RESULTS["roll_lead5_vs_0"])


# ----------------------------------------------------------------------------------------
# E. Timing rules
# ----------------------------------------------------------------------------------------
def test_timing() -> None:
    px = fh.load_series("crude_roll")
    fri = fh.weekly_decision_dates("1990-01-05", "2024-03-29")
    ep = fh.entry_positions(fri, px.index, "price")
    ok = ep >= 0
    ed = px.index.values[ep[ok]]
    check("price timing: entry strictly after the decision Friday, within 7 days",
          bool(np.all(ed > fri.values[ok])
               and np.all((ed - fri.values[ok]) <= np.timedelta64(8, "D"))),
          dict(weeks=int(len(fri)), no_entry=int((~ok).sum()),
               weekday_counts=pd.Series(pd.DatetimeIndex(ed).day_name()).value_counts()
               .to_dict()))
    ep2 = fh.entry_positions(fri, px.index, "eia")
    ok2 = ep2 >= 0
    ed2 = px.index.values[ep2[ok2]]
    check("eia timing: entry on/after the Thursday after the week-ending Friday",
          bool(np.all(ed2 >= fri.values[ok2] + np.timedelta64(6, "D"))))
    f = lc.build_panel(markets=["crude"])["crude"]
    ep3 = fh.entry_positions(f["report_date"], panel_prices["crude"].index, "cot")
    check("cot timing identical to the v1 engine", bool(np.all(ep3 == f["entry_pos"].to_numpy())))
    early = fh.entry_positions(pd.DatetimeIndex(["1970-01-02"]), px.index, "price")
    check("no entry when the series has not started (lag guard)", int(early[0]) == -1)


# ----------------------------------------------------------------------------------------
# F. Permutation direction, net side, split-half, walk-forward, expanding stats
# ----------------------------------------------------------------------------------------
def test_mechanics() -> None:
    px = fh.load_series("crude_roll")
    fri = fh.weekly_decision_dates("1990-01-05", "2023-12-29")
    ft = fh.forward_table(fri, px, "price", (8,))
    oracle = np.where(ft["r8"].isna(), np.nan, np.sign(ft["r8"]).fillna(0))
    sf = fh.make_signal_frame(fri, oracle)
    r = fh.evaluate(sf, px, (8,), timing="price", sides=("long", "short", "net"), k=500)
    pmin = 1 / 501
    check("oracle signal: perm_p at its minimum for long, short and net",
          bool((r["perm_p"] == pmin).all()), r[["side", "perm_p", "perm_p_exact"]]
          .to_dict("records"))
    anti = fh.make_signal_frame(fri, -oracle)
    ra = fh.evaluate(anti, px, (8,), timing="price", sides=("long",), k=500)
    check("anti-oracle signal: perm_p ~ 1", float(ra["perm_p"].iloc[0]) > 0.99,
          float(ra["perm_p"].iloc[0]))

    rng = np.random.default_rng(3)
    lo_only = fh.make_signal_frame(fri, (rng.random(len(fri)) < 0.3).astype(float))
    a = fh.evaluate(lo_only, px, (8,), timing="price", sides=("long", "net"), k=200)
    same = all(a.iloc[0][c] == a.iloc[1][c] for c in ("n", "episodes", "mean", "excess", "hit",
                                                      "perm_p_exact"))
    check("net side equals long side for a long-only signal", same)

    sh = fh.split_half(lo_only, px, timing="price", horizon=8, sides=("long",), k=200)
    full = a.iloc[0]
    check("split-half: H1 + H2 signal weeks = full sample",
          int(sh["n"].sum()) == int(full["n"]), (sh[["half", "n", "period"]]
                                                 .to_dict("records"), int(full["n"])))

    # Walk-forward: fit() must only ever see history strictly before its cutoff, and an
    # outcome-based fit restricted to realised returns must never use an exit after it.
    frame = ft.copy()
    seen = []

    def fit(hist, cutoff):
        seen.append(bool((hist["decision_date"] < cutoff).all()))
        realised = hist[hist["exit_date8"] < cutoff]
        seen.append(bool((realised["exit_date8"] < cutoff).all()))
        return float(realised["r8"].mean()) if len(realised) else np.nan

    def apply(mu, test):
        return np.full(len(test), np.nan if np.isnan(mu) else (1.0 if mu > 0 else -1.0))
    wf = fh.walk_forward(frame, fit, apply, first_test="1995-01-01")
    check("walk-forward: fit sees only the past; output valid from first_test",
          all(seen) and bool((wf.loc[wf.valid, "decision_date"] >= "1995-01-01").all()),
          dict(blocks=len(seen) // 2, valid_weeks=int(wf["valid"].sum())))

    x = rng.standard_normal(400)
    z1, p1 = fh.expanding_zscore(x, 52), fh.expanding_percentile(x, 52)
    x2 = x.copy()
    x2[300:] += 100.0
    z2, p2 = fh.expanding_zscore(x2, 52), fh.expanding_percentile(x2, 52)
    check("expanding z-score / percentile use no future data",
          bool(np.allclose(z1[:300], z2[:300], equal_nan=True)
               and np.allclose(p1[:300], p2[:300], equal_nan=True)))


# ----------------------------------------------------------------------------------------
# G. Other series
# ----------------------------------------------------------------------------------------
def test_series() -> None:
    check("10y price proxy: 6% coupon bond at 6% yield = par",
          abs(float(fh.par_bond_price(0.06)) - 100.0) < 1e-9)
    check("10y price proxy falls when yields rise",
          float(fh.par_bond_price(0.07)) < 100 < float(fh.par_bond_price(0.05)))
    desc = fh.describe_series()
    RESULTS["series"] = fh.to_jsonable(desc)
    check("every registered series loads, positive, sorted, no duplicate dates",
          all(len(fh.load_series(n)) > 1000 for n in fh.SERIES), desc[["series", "first", "last",
                                                                       "n"]].to_dict("records"))
    ho_dropped = fh.drop_stale_holiday_prints(fh.load_ext_raw("HO_SPOT"))[1]
    RESULTS["stale_holiday_prints_dropped"] = {
        sid: [d.date().isoformat() for d in fh.drop_stale_holiday_prints(fh.load_ext_raw(sid))[1]]
        for sid in ("HO_SPOT", "GAS_SPOT", "RBRTE", "DTWEXBGS", "HO_F1", "RBOB_F1")}
    check("stale holiday prints are reported", isinstance(ho_dropped, pd.DatetimeIndex))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the 2000-draw strategy null")
    args = ap.parse_args(argv)
    t0 = time.time()
    global panel_prices
    panel = lc.build_panel()
    panel_prices = lc.load_prices()
    test_v1_canonical(panel)
    test_v1_pooled(panel)
    test_v1_strategy(panel, 0 if args.fast else fh.K_PERM)
    test_roll_index()
    test_timing()
    test_mechanics()
    test_series()
    RESULTS["runtime_s"] = round(time.time() - t0, 1)
    RESULTS["failed"] = FAILED
    out = fh.WORK2 / "harness"                 # outputs: ROOT/work2/harness
    out.mkdir(parents=True, exist_ok=True)
    (out / "selftest_output.json").write_text(json.dumps(fh.to_jsonable(RESULTS), indent=1))
    print(f"{len(RESULTS['checks']) - len(FAILED)}/{len(RESULTS['checks'])} checks passed "
          f"in {RESULTS['runtime_s']}s -> {out / 'selftest_output.json'}")
    return 1 if FAILED else 0


panel_prices: dict = {}

if __name__ == "__main__":
    sys.exit(main())

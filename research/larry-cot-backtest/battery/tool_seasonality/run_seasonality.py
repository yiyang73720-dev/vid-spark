#!/usr/bin/env python3
"""
run_seasonality -- driver of the Larry Williams SEASONALITY test (SPEC2 battery).

Implements exactly what PREREG.md fixes (written before any of this ran):
  1. look-ahead checks (walk-forward re-implementation, truncation test, leak canary);
  2. the PRIMARY rule: seasonal walk-forward, 7 markets pooled, NET side, h = 8 weeks,
     th = 70/30, L = 15 (crude/HO/gasoline) or 10 (gold/silver/soybean/sugar), week bucket,
     estimation series = evaluation series (crude on EIA spot);
  3. out-of-sample: split halves (per market + pooled, common split date), per-year breakdown;
  4. strategy: S1/S2 per market (fh.simulate) + equal-weight 7-market portfolio with its null;
  5. tradability confirmation on roll-adjusted futures indices (L = 15);
  6. crude on CL continuous / roll index, extended-history variant, and the full variant grid;
  7. results.json (SPEC2 rule 8) + tests.csv + chart-ready JSON.

Run:  python3 -I run_seasonality.py [--fast]     (--fast: K = 200 for a quick dry run)
All numbers in results.json are computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent         # this tool's code and PREREG files
sys.path.insert(0, str(CODE))
import seas_data as sd  # noqa: E402
import seasonal as se   # noqa: E402

fh = se.fh
HERE = fh.WORK2 / "tool_seasonality"           # outputs: ROOT/work2/tool_seasonality
HERE.mkdir(parents=True, exist_ok=True)
FAST = "--fast" in sys.argv
K = 200 if FAST else fh.K_PERM            # permutation draws (SPEC2: 2000)
NULL_K = 200 if FAST else fh.K_PERM       # strategy null draws
OUT = HERE / ("results_fast.json" if FAST else "results.json")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------
# Markets (PREREG section 3)
# ----------------------------------------------------------------------------------------
MARKETS = [  # key, estimation = evaluation series of the primary, primary L, tradable key
    dict(key="crude", series="crude_spot", L=15),
    dict(key="heatoil", series="ho_spot", L=15),
    dict(key="gasoline", series="gas_spot", L=15),
    dict(key="gold", series="gold", L=10),
    dict(key="silver", series="silver", L=10),
    dict(key="soybean", series="soybean", L=10),
    dict(key="sugar", series="sugar", L=10),
]
KEYS = [m["key"] for m in MARKETS]
PRIMARY = dict(th=70, h=8, bucket="week")
HORIZONS = (1, 4, 8, 13)
GRID_TH = (60, 65, 70, 75)
GRID_L = (10, 15, 20)
GRID_H = (4, 8, 13)
GRID_BUCKET = ("week", "month")
FRI = fh.weekly_decision_dates("1986-01-03", "2026-10-16")
TRADABLE_COMMON = ("2001-01-01", "2024-03-28")


def series_obj(name: str) -> pd.Series:
    """A traded series by name: harness series, or a roll-adjusted futures index 'psys:<key>'."""
    if name.startswith("psys:"):
        return sd.tradable_series(name.split(":", 1)[1]).rename(name)
    return fh.load_series(name)


def sframe(sig) -> pd.DataFrame:
    return fh.make_signal_frame(FRI, sig)


def restrict(sf: pd.DataFrame, start, end) -> pd.DataFrame:
    """Signal frame with weeks outside [start, end] made invalid (sub-period evaluation)."""
    q = sf.copy()
    d = pd.DatetimeIndex(q["decision_date"])
    m = np.asarray((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))
    q["valid"] = q["valid"].to_numpy(bool) & m
    q["signal"] = np.where(q["valid"], q["signal"], 0).astype(np.int8)
    return q


def evaluate(sf, ser_name, horizons, labels, sides=("long", "short", "net")):
    rows = fh.evaluate(sf, series_obj(ser_name), horizons, timing="price", sides=sides,
                       series_name=ser_name, labels=labels, k=K)
    return rows


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ========================================================================================
# 0. Estimation tables
# ========================================================================================
log("building estimation tables")
EST_H = (4, 8, 13)
TAB = {m["key"]: se.est_table(series_obj(m["series"]), m["series"], EST_H) for m in MARKETS}
TRAD_NAME = {k: ("crude_roll" if k == "crude" else f"psys:{k}") for k in KEYS}
TAB_TRAD = {k: se.est_table(series_obj(TRAD_NAME[k]), TRAD_NAME[k], EST_H) for k in KEYS}
TAB_LEAD5 = se.est_table(series_obj("crude_roll_lead5"), "crude_roll_lead5", EST_H)
EXT_KEYS = ("gold", "silver", "soybean", "sugar")
TAB_EXT = {k: se.splice_tables(TAB[k], TAB_TRAD[k], f"{k}+psys:{k}") for k in EXT_KEYS}

# ========================================================================================
# 1. Look-ahead checks (PREREG 2(a), 2(b)) + leak canary
# ========================================================================================
log("look-ahead checks")
checks = dict(walk_forward_reimplementation=[], truncation=[], leak_canary=None)


def cuts_for(ser: pd.Series, L: int) -> pd.DatetimeIndex:
    """12 cut dates spread from the first possible signal (series start + L years) to the end."""
    return se.nearest_friday(pd.date_range(ser.index[0] + pd.DateOffset(years=L),
                                           ser.index[-1], periods=12))
for m in MARKETS:
    for tabset, label, L in ((TAB, m["series"], m["L"]), (TAB_TRAD, TRAD_NAME[m["key"]], 15)):
        tab = tabset[m["key"]]
        vec = se.seasonal_signal(FRI, tab, PRIMARY["h"], L, PRIMARY["th"], "week")
        wf = se.signals_walk_forward(FRI, tab, PRIMARY["h"], L, PRIMARY["th"])
        wfv = np.where(wf["valid"], wf["signal"].astype(float), np.nan)
        same = np.array_equal(np.nan_to_num(vec, nan=9.0), np.nan_to_num(wfv, nan=9.0))
        checks["walk_forward_reimplementation"].append(
            dict(market=m["key"], est_series=label, L=L, identical=bool(same),
                 n_defined=int(np.isfinite(vec).sum())))
        if not same:
            raise AssertionError(f"walk-forward re-implementation differs: {label}")
    ser = series_obj(m["series"])
    for bucket in ("week", "month"):
        tt = se.truncation_test(FRI, ser, m["series"], PRIMARY["h"], m["L"], PRIMARY["th"],
                                bucket, cuts_for(ser, m["L"]))
        for t in tt:
            t.update(market=m["key"], series=m["series"], bucket=bucket)
        checks["truncation"].extend(tt)
        if not all(t["identical"] for t in tt):
            raise AssertionError(f"truncation test failed: {m['series']} {bucket}")
try:   # the assertion must fire if an analog were taken from the decision year itself
    se._assert_past(np.array([[np.datetime64("2020-03-06")]], dtype="datetime64[ns]"),
                    pd.DatetimeIndex(["2020-01-03"]), "canary")
    checks["leak_canary"] = "FAILED: no assertion"
    raise AssertionError("leak canary did not fire")
except AssertionError as e:
    if "canary did not fire" in str(e):
        raise
    checks["leak_canary"] = f"assertion fired as expected: {e}"

# ========================================================================================
# 2. PRIMARY
# ========================================================================================
log("primary")
SIG, SF = {}, {}
for m in MARKETS:
    SIG[m["key"]] = se.seasonal_signal(FRI, TAB[m["key"]], PRIMARY["h"], m["L"], PRIMARY["th"])
    SF[m["key"]] = sframe(SIG[m["key"]])

base_lab = dict(filter="seasonality", th=PRIMARY["th"], h_est=PRIMARY["h"],
                bucket=PRIMARY["bucket"], set="primary")
tests = []
primary_rows = []
for m in MARKETS:
    lab = dict(base_lab, variant="primary", market=m["key"], est_series=m["series"], L=m["L"],
               is_primary=False)
    rows = evaluate(SF[m["key"]], m["series"], HORIZONS, lab)
    for side in ("long", "short", "net"):
        oc = se.offcal_p(SF[m["key"]], series_obj(m["series"]), "price", 8, side)
        msk = (rows["side"] == side) & (rows["horizon_w"] == 8)
        for kk, v in oc.items():
            rows.loc[msk, kk] = v
    primary_rows.append(rows)
primary_rows = pd.concat(primary_rows, ignore_index=True)

PARTS = [(SF[m["key"]], series_obj(m["series"]), "price") for m in MARKETS]
pooled_rows = []
for h in HORIZONS:
    r = se.pooled_net(PARTS, h, k=K, labels=dict(base_lab, variant="primary", L="15/10",
                                                  est_series="own", is_primary=(h == 8)))
    pooled_rows.append(r)
pl = fh.evaluate_pooled(PARTS, (8,), sides=("long", "short"), k=K,
                        labels=dict(base_lab, filter="seasonality", variant="primary",
                                    L="15/10", est_series="own", is_primary=False))
pooled_df = pd.concat([pd.DataFrame(pooled_rows), pl], ignore_index=True)
PRIMARY_ROW = pooled_df[(pooled_df["side"] == "net") & (pooled_df["horizon_w"] == 8)].iloc[0]

# Holm across the 7 per-market primary NET h=8 cells
pm_net = primary_rows[(primary_rows["side"] == "net") & (primary_rows["horizon_w"] == 8)]
holm_tab = pd.DataFrame(dict(market=pm_net["market"].to_numpy(),
                             excess=pm_net["excess"].to_numpy(),
                             perm_p=pm_net["perm_p"].to_numpy(),
                             perm_p_holm=holm(pm_net["perm_p"].to_numpy()),
                             perm_p_offcal=pm_net["perm_p_offcal"].to_numpy()))
tests += [primary_rows, pooled_df]

# ========================================================================================
# 3. Out-of-sample
# ========================================================================================
log("out-of-sample")
split_rows = []
for m in MARKETS:
    sp = fh.split_half(SF[m["key"]], series_obj(m["series"]), timing="price", horizon=8,
                       sides=("long", "short", "net"), k=K,
                       labels=dict(base_lab, variant="primary_split", market=m["key"],
                                   est_series=m["series"], L=m["L"], is_primary=False),
                       series_name=m["series"])
    split_rows.append(sp)
split_rows = pd.concat(split_rows, ignore_index=True)
pooled_split = pd.DataFrame(se.pooled_split_half(PARTS, 8, k=K))
by_year = se.pooled_by_year(PARTS, 8)

# ========================================================================================
# 4. Strategy (rule 6)
# ========================================================================================
log("strategy per market")
strat_rows, equity_charts, trades_all = [], {}, []
for m in MARKETS:
    st = fh.simulate(SF[m["key"]], series_obj(m["series"]), timing="price",
                     series_name=m["series"], null_k=NULL_K,
                     labels=dict(filter="seasonality", variant="primary", market=m["key"]))
    strat_rows.append(st["metrics"])
    equity_charts[m["key"]] = fh.weekly_equity(st["equity"][["S1_long_short", "buy_hold"]])
    tr = st["trades"]
    tr.insert(0, "market", m["key"])
    trades_all.append(tr)
log("strategy portfolio (primary)")
port = se.portfolio_simulation(PARTS, KEYS, null_k=NULL_K)
strat_df = pd.concat(strat_rows + [port["metrics"]], ignore_index=True)

# ========================================================================================
# 5. Tradability confirmation (roll-adjusted futures, L = 15)
# ========================================================================================
log("tradability confirmation")
SF_TR = {}
trad_rows = []
for k in KEYS:
    sig = se.seasonal_signal(FRI, TAB_TRAD[k], 8, 15, PRIMARY["th"])
    SF_TR[k] = sframe(sig)
    lab = dict(base_lab, set="tradable", variant="tradable_primary", market=k,
               est_series=TRAD_NAME[k], L=15, is_primary=False)
    rows = evaluate(SF_TR[k], TRAD_NAME[k], HORIZONS, lab)
    for side in ("long", "short", "net"):
        oc = se.offcal_p(SF_TR[k], series_obj(TRAD_NAME[k]), "price", 8, side)
        msk = (rows["side"] == side) & (rows["horizon_w"] == 8)
        for kk, v in oc.items():
            rows.loc[msk, kk] = v
    trad_rows.append(rows)
trad_rows = pd.concat(trad_rows, ignore_index=True)
PARTS_TR = [(SF_TR[k], series_obj(TRAD_NAME[k]), "price") for k in KEYS]
trad_pooled = [se.pooled_net(PARTS_TR, 8, k=K, labels=dict(base_lab, set="tradable",
                                                            variant="tradable_primary_full",
                                                            L=15, est_series="roll-adjusted",
                                                            is_primary=False))]
PARTS_TR_C = [(restrict(sf, *TRADABLE_COMMON), ser, t) for sf, ser, t in PARTS_TR]
trad_pooled.append(se.pooled_net(PARTS_TR_C, 8, k=K, labels=dict(
    base_lab, set="tradable", variant="tradable_primary_2001_2024", L=15,
    est_series="roll-adjusted", is_primary=False)))
trad_pooled_df = pd.DataFrame(trad_pooled)
trad_split = pd.DataFrame(se.pooled_split_half(PARTS_TR, 8, k=K))
log("strategy portfolio (tradable)")
port_tr = se.portfolio_simulation(PARTS_TR, KEYS, null_k=NULL_K)
port_tr["metrics"]["market"] = "EW7_portfolio_tradable"
tests += [trad_rows, trad_pooled_df]

# The primary signal (estimated on spot / v1) evaluated on the tradable series over the
# common window: does a spot-learned seasonal pay on futures? (diagnostic)
cross_rows = []
for m in MARKETS:
    lab = dict(base_lab, set="cross", variant="primary_signal_on_tradable", market=m["key"],
               est_series=m["series"], L=m["L"], is_primary=False)
    cross_rows.append(evaluate(SF[m["key"]], TRAD_NAME[m["key"]], (8,), lab))
cross_rows = pd.concat(cross_rows, ignore_index=True)
cross_pooled = pd.DataFrame([se.pooled_net(
    [(SF[m["key"]], series_obj(TRAD_NAME[m["key"]]), "price") for m in MARKETS], 8, k=K,
    labels=dict(base_lab, set="cross", variant="primary_signal_on_tradable", L="15/10",
                est_series="own", is_primary=False))])
tests += [cross_rows, cross_pooled]

# ========================================================================================
# 6. Crude extras and extended-history variant
# ========================================================================================
log("crude extras and extended history")
extra_rows = []
for ser in ("crude_cl", "crude_roll"):
    extra_rows.append(evaluate(SF["crude"], ser, HORIZONS, dict(
        base_lab, set="crude_eval", variant=f"primary_signal_on_{ser}", market="crude",
        est_series="crude_spot", L=15, is_primary=False)))
sig_l5 = se.seasonal_signal(FRI, TAB_LEAD5, 8, 15, PRIMARY["th"])
extra_rows.append(evaluate(sframe(sig_l5), "crude_roll_lead5", HORIZONS, dict(
    base_lab, set="tradable", variant="tradable_primary_lead5", market="crude",
    est_series="crude_roll_lead5", L=15, is_primary=False)))
SF_EXT = dict(SF)
for k in EXT_KEYS:
    sig = se.seasonal_signal(FRI, TAB_EXT[k], 8, 15, PRIMARY["th"])
    SF_EXT[k] = sframe(sig)
    ser = next(m["series"] for m in MARKETS if m["key"] == k)
    extra_rows.append(evaluate(SF_EXT[k], ser, HORIZONS, dict(
        base_lab, set="extended", variant="extended_L15", market=k,
        est_series=f"{ser}+psys:{k}", L=15, is_primary=False)))
extra_rows = pd.concat(extra_rows, ignore_index=True)
ext_pooled = pd.DataFrame([se.pooled_net(
    [(SF_EXT[m["key"]], series_obj(m["series"]), "price") for m in MARKETS], 8, k=K,
    labels=dict(base_lab, set="extended", variant="primary_with_extended_L15", L=15,
                est_series="own (+psys before 2000 for metals/ags)", is_primary=False))])
tests += [extra_rows, ext_pooled]

# POST-HOC (NOT pre-registered, added after the dry run showed one market dominating):
# leave-one-market-out pooled NET h=8 of the primary. Labelled set="posthoc".
log("post-hoc leave-one-out")
loo = []
for i, m in enumerate(MARKETS):
    loo.append(se.pooled_net([p for j, p in enumerate(PARTS) if j != i], 8, k=K, labels=dict(
        base_lab, set="posthoc", variant=f"posthoc_primary_without_{m['key']}", L="15/10",
        est_series="own", is_primary=False)))
loo_df = pd.DataFrame(loo)
tests += [loo_df]

# ========================================================================================
# 7. Variant grid (robustness only)
# ========================================================================================
log("variant grid")
GRID_SETS = {   # set name -> list of (market key, estimation table dict, evaluation series)
    "primary": [(m["key"], TAB, m["series"]) for m in MARKETS],
    "crude_on_cl": [("crude", TAB, "crude_cl")],
    "crude_on_roll": [("crude", TAB, "crude_roll")],
    "tradable": [(k, TAB_TRAD, TRAD_NAME[k]) for k in KEYS],
}
grid_rows, grid_pooled = [], []
stats_cache = {}
for set_name, members in GRID_SETS.items():
    for bucket in GRID_BUCKET:
        for h in GRID_H:
            Ls = list(GRID_L) + (["mixed"] if set_name == "primary" else [])
            for L in Ls:
                for th in GRID_TH:
                    parts = []
                    for key, tabs, ev in members:
                        LL = (next(m["L"] for m in MARKETS if m["key"] == key)
                              if L == "mixed" else L)
                        ck = (tabs[key].name, bucket, h, LL)
                        if ck not in stats_cache:
                            stats_cache[ck] = (se.week_stats(FRI, tabs[key], h, LL)
                                               if bucket == "week"
                                               else se.month_stats(FRI, tabs[key], h, LL))
                        sf = sframe(se.signal_from_stats(stats_cache[ck], th))
                        parts.append((sf, series_obj(ev), "price"))
                        if L == "mixed":
                            continue      # per-market rows of mixed L = the fixed-L rows
                        variant = f"{bucket}_th{th}_L{L}_h{h}"
                        grid_rows.append(evaluate(sf, ev, (h,), dict(
                            filter="seasonality", variant=variant, market=key, set=set_name,
                            est_series=tabs[key].name, th=th, L=LL, h_est=h, bucket=bucket,
                            is_primary=False)))
                    if set_name in ("primary", "tradable"):
                        grid_pooled.append(se.pooled_net(parts, h, k=K, labels=dict(
                            filter="seasonality", variant=f"{bucket}_th{th}_L{L}_h{h}",
                            set=set_name, th=th, L=str(L), h_est=h, bucket=bucket,
                            est_series="own" if set_name == "primary" else "roll-adjusted",
                            is_primary=False)))
grid_rows = pd.concat(grid_rows, ignore_index=True)
grid_pooled = pd.DataFrame(grid_pooled)
tests += [grid_rows, grid_pooled]


def grid_summary(df: pd.DataFrame) -> dict:
    d = df[df["n"] > 0]
    return dict(cells=int(len(d)), share_excess_positive=float((d["excess"] > 0).mean()),
                share_perm_p_le_0_05=float((d["perm_p"] <= 0.05).mean()),
                share_perm_p_le_0_10=float((d["perm_p"] <= 0.10).mean()),
                median_excess=float(d["excess"].median()),
                min_excess=float(d["excess"].min()), max_excess=float(d["excess"].max()))


grid_sum = {}
for set_name in ("primary", "tradable"):
    gp = grid_pooled[grid_pooled["set"] == set_name]
    grid_sum[f"pooled_net_{set_name}"] = grid_summary(gp)
    for bucket in GRID_BUCKET:
        grid_sum[f"pooled_net_{set_name}_{bucket}"] = grid_summary(gp[gp["bucket"] == bucket])
for set_name in GRID_SETS:
    g = grid_rows[(grid_rows["set"] == set_name) & (grid_rows["side"] == "net")]
    grid_sum[f"per_market_net_{set_name}"] = grid_summary(g)

# one-at-a-time variants around the primary (pooled net, primary series)
ota = grid_pooled[(grid_pooled["set"] == "primary")].copy()


def pick(bucket, th, L, h):
    r = ota[(ota["bucket"] == bucket) & (ota["th"] == th) & (ota["L"] == str(L))
            & (ota["h_est"] == h)]
    return r.iloc[0][["variant", "n", "episodes", "mean", "excess", "hit", "perm_p"]].to_dict()


one_at_a_time = dict(
    primary_mixed=pick("week", 70, "mixed", 8),
    thresholds=[pick("week", th, "mixed", 8) for th in GRID_TH],
    lookbacks=[pick("week", 70, L, 8) for L in GRID_L],
    horizons=[pick("week", 70, "mixed", h) for h in GRID_H],
    month_of_year=pick("month", 70, "mixed", 8),
)

# ========================================================================================
# 8. Charts
# ========================================================================================
log("charts")


def heat(set_name, bucket, h):
    g = grid_pooled[(grid_pooled["set"] == set_name) & (grid_pooled["bucket"] == bucket)
                    & (grid_pooled["h_est"] == h)]
    Ls = [str(x) for x in GRID_L] + (["mixed"] if set_name == "primary" else [])
    return dict(rows_L=Ls, cols_th=list(GRID_TH),
                excess=[[float(g[(g["L"] == L) & (g["th"] == th)]["excess"].iloc[0])
                         for th in GRID_TH] for L in Ls],
                perm_p=[[float(g[(g["L"] == L) & (g["th"] == th)]["perm_p"].iloc[0])
                         for th in GRID_TH] for L in Ls])


latest_profile = {}
for m in MARKETS:   # the walk-forward estimate the rule used over the last 52 decision weeks
    st = se.week_stats(FRI, TAB[m["key"]], 8, m["L"])
    sel = np.flatnonzero(st["complete"])[-52:]
    latest_profile[m["key"]] = dict(
        decision_date=[FRI[i].date().isoformat() for i in sel],
        hit=[float(st["npos"][i] / m["L"]) for i in sel],
        mean=[float(st["mean"][i]) for i in sel],
        signal=[float(SIG[m["key"]][i]) for i in sel])

charts = dict(
    equity_portfolio_primary=fh.weekly_equity(port["equity"]),
    equity_portfolio_tradable=fh.weekly_equity(port_tr["equity"]),
    equity_by_market_S1_long_short=equity_charts,
    pooled_net_by_year=by_year.to_dict("list"),
    grid_heatmap_pooled_net_h8=dict(primary_week=heat("primary", "week", 8),
                                    primary_month=heat("primary", "month", 8),
                                    tradable_week=heat("tradable", "week", 8),
                                    tradable_month=heat("tradable", "month", 8)),
    latest_seasonal_estimates=latest_profile,
)

# ========================================================================================
# 9. Write results
# ========================================================================================
log("writing results")
tests_df = pd.concat(tests, ignore_index=True)
front = [c for c in fh.TEST_COLUMNS if c in tests_df.columns]
tests_df = tests_df[front + [c for c in tests_df.columns if c not in front]]
tests_df.to_csv(HERE / ("tests_fast.csv" if FAST else "tests.csv"), index=False)
trades_df = pd.concat(trades_all, ignore_index=True)
trades_df.to_csv(HERE / ("trades_fast.csv" if FAST else "trades.csv"), index=False)

p_ = PRIMARY_ROW.to_dict()
s1 = port["metrics"].set_index("strategy")
tp = trad_pooled_df.iloc[0]
c1 = bool(p_["excess"] > 0 and p_["perm_p"] <= 0.05)
c2 = bool((pooled_split["excess"] > 0).all())
c3 = bool(tp["excess"] > 0 and tp["perm_p"] <= 0.10)
c4 = bool(s1.loc["S1_long_short", "sharpe"] > 0 and s1.loc["S1_long_short", "null_p_sharpe"] <= 0.10)
if c1 and c2 and c3 and c4:
    verdict = "positive EV, robust"
elif p_["excess"] > 0 and p_["perm_p"] <= 0.10:
    verdict = "positive EV, fragile"
elif p_["excess"] < 0 and p_["perm_p_lower"] <= 0.05:
    verdict = "negative (anti-signal)"
else:
    verdict = "no edge"
verdict_detail = dict(verdict=verdict, c1_primary_excess_pos_p_le_005=c1,
                      c2_both_halves_excess_pos=c2, c3_tradable_excess_pos_p_le_010=c3,
                      c4_S1_LS_portfolio_sharpe_pos_null_p_le_010=c4)

larry_definition = (
    "Seasonality (Larry Williams; first book 'Sure Thing Commodity Trading: How Seasonal Factors "
    "Influence Commodity Prices'): the average pattern a market traces over the calendar year, "
    "built from many prior years; his Williams True Seasonal uses only data prior to the year "
    "analysed and he recommends multi-year lookbacks. He uses seasonals as timing CONFIRMATION "
    "(with COT etc.), not as a stand-alone trigger. Tested here as the task's mechanical "
    "walk-forward rule: for each decision Friday, the 8-week forward returns from the same "
    "calendar date in each of the prior L years (L = 15, or 10 where the series is too short); "
    "long if >= 70% of those years were up and their mean > 0, short if <= 30% and mean < 0.")
sources = [
    "Larry Williams bio / first seasonal book: https://futures.ireallytrade.com/aboutlarry/ ; "
    "https://williamspercentr.com/about-larry",
    "MoneyShow interview transcripts (seasonal = averaged annual pattern; congruency; "
    "confirmation not a stand-alone trigger; COT at seasonal lows; patterns change): "
    "https://moneyshow.com/articles/videotranstr-28669 ; "
    "https://moneyshow.com/articles/videotranstr-27772",
    "Larry Williams on the Seasonality of the Futures Markets (NinjaTrader webinar summary): "
    "https://ninjatrader.com/futures/blogs/larry-williams-on-the-seasonality-of-the-futures-markets/",
    "Williams True Seasonal (uses only data prior to the analysed year; default 1-year lookback, "
    "Williams recommends more): https://help.stockcharts.com/charts-and-tools/stockchartsacp-plug-ins/"
    "larry-williams-stock-trading-starter-pack-plug-in ; https://articles.stockcharts.com/article/"
    "articles-chartwatchers-2023-09-optimizing-your-stock-selectio-527/",
    "Industry convention (not Larry): MRCI 15 years, >= 80% reliability: https://mrci.com/web/"
    "help-pages/frequently-asked-questions/70-general/2204-from-our-editor-trade-selection-process.html",
    "Note: the pages above were seen only through WebSearch result summaries (WebFetch is "
    "DNS-blocked in this session); the exact construction in Larry's books was not verified.",
    "Data: EIA spot series (crude_spot, ho_spot, gas_spot) and v1 series via fund_harness; "
    "crude_roll from EIA RCLC1/RCLC2; roll-adjusted futures indices from pysystemtrade "
    f"multiple_prices_csv @ {sd.PSYS_COMMIT} (seas_data.py).",
]
caveats = [
    "Larry says he does not trade seasonals alone; this tests the tool in isolation, as the "
    "battery requires.",
    "Spot energy series (ho_spot, gas_spot, crude_spot) are not directly tradable; spot "
    "seasonality is largely priced into the futures curve, and the NY Harbor gasoline spot "
    "assessment changes seasonal grade. The tradability confirmation re-runs the rule on "
    "roll-adjusted futures (estimation = evaluation).",
    "v1 soybean (103.ZS00Y), sugar (SB=F) and crude CL (102.CL00Y) are unadjusted continuous "
    "futures; their roll jumps recur at nearly fixed calendar dates and can be learned as fake "
    "seasonality. The roll-adjusted indices avoid that.",
    "Roll-adjusted pysystemtrade indices end 2024-03-28; soybean holds the November contract "
    "only (deferred, rolls once a year); prices after the hourly era starts (2008-2013) are not "
    "always settlements; silver has 12 roll days with no usable quote (return set to 0).",
    "Gold/silver/soybean/sugar use L = 10 in the primary (their v1 series start in 2000), so "
    "their signals start about 2010.",
    "Shifts of about 52 weeks keep a seasonal signal aligned with the calendar, so the standard "
    "circular-shift p can be conservative; perm_p_offcal excludes those offsets (diagnostic, "
    "pre-registered). It is not uniformly smaller (gasoline: larger), and it also credits ANY "
    "calendar-locked pattern -- roll-day jumps of unadjusted futures, spot grade switches -- so a "
    "small perm_p_offcal is not evidence of a tradable seasonal.",
    "Overlapping 8-week returns: n signal weeks overstates the number of independent bets; "
    "use 'episodes' and the permutation p, not naive t-stats.",
    "Grid cells are highly correlated and were not used to choose the primary. Thresholds are "
    "integer counts of up-years: with L = 10, th = 65 and th = 70 both need 7 of 10 years, so "
    "those cells coincide.",
    "The leave-one-market-out table is post-hoc (added after a K=200 dry run) and is labelled so.",
] + sd.PSYS_CAVEATS

primary_doc = dict(
    rule=("Seasonal walk-forward, 7 markets pooled, NET side, h=8 weeks, th=70/30, L=15 (crude "
          "spot, HO spot, gasoline spot) / 10 (gold, silver, soybean, sugar), week-of-year "
          "(calendar-anchored) bucket; estimation series = evaluation series."),
    **{kk: p_[kk] for kk in ("market", "series", "side", "horizon_w", "n", "n_long", "n_short",
                             "episodes", "mean", "median", "excess", "hit", "perm_p",
                             "perm_p_lower", "T", "period")},
    per_market_net_h8=pm_net[["market", "series", "n", "episodes", "mean", "baseline", "excess",
                              "hit", "perm_p", "perm_p_exact", "perm_p_offcal"]],
    holm_per_market=holm_tab,
    split_half_pooled=pooled_split,
    strategy_portfolio=port["metrics"],
    tradability_confirmation=trad_pooled_df,
    verdict=verdict_detail,
)

oos = dict(
    note=("The primary is itself walk-forward: every seasonal estimate uses only prior years "
          "and outcomes that ended before the decision date (asserted)."),
    split_half_pooled=pooled_split, split_half_per_market=split_rows,
    tradable_split_half_pooled=trad_split, pooled_net_by_year=by_year,
    lookahead_checks=checks,
)
fh.write_results(
    OUT, filter="seasonality", larry_definition=larry_definition, sources=sources,
    primary=primary_doc, tests=tests_df, strategy=strat_df, oos=oos, caveats=caveats,
    strategy_tradable_portfolio=port_tr["metrics"], grid_summary=grid_sum,
    post_hoc=dict(note="NOT pre-registered; added after the dry run (K=200) showed the pooled "
                       "result concentrated in one market.", leave_one_market_out=loo_df),
    one_at_a_time_pooled_net=one_at_a_time,
    roll_index_validation={k: sd.validate_psys_index(k) for k in KEYS if k != "crude"},
    series_used={m["key"]: dict(primary_estimation=m["series"], primary_evaluation=m["series"],
                                primary_L=m["L"], tradable=TRAD_NAME[m["key"]]) for m in MARKETS},
    run=dict(K=K, null_k=NULL_K, seed=fh.SEED, fast=FAST, decision_grid=[
        FRI[0].date().isoformat(), FRI[-1].date().isoformat()],
        prereg_sha256=(CODE / "PREREG.sha256.txt").read_text().split()[1]),
    charts=charts,
)
log(f"wrote {OUT}")
print(json.dumps(fh.to_jsonable(dict(primary={kk: p_[kk] for kk in (
    "n", "episodes", "mean", "excess", "hit", "perm_p", "perm_p_lower", "period")},
    verdict=verdict_detail)), indent=1))

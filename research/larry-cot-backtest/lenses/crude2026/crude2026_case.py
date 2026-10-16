#!/usr/bin/env python3
"""
crude2026_case.py -- crude oil 2026 case study (lens of the Larry Williams COT backtest).

Question checked: did the latest crude COT data, read Larry Williams' way (commercials'
COT index, N=26 weeks, buy >= 80 / sell <= 20, SPEC.md v1), give a buy signal ahead of the
autumn-2026 rally, and what does history say about such signals in crude?

Every index, entry date and forward return comes from the reconciled engine
(work/engine/larry_cot.py); nothing in the spec is re-implemented here. This file only adds
case-study bookkeeping: tables, the hypothetical report, the rally measurements and the
base-rate / precursor counts.

Sections (match the lens brief) and outputs (all in work/lens_crude2026/):
  (a) timeline 2025-01-01 -> latest      timeline.json, timeline_weekly.csv, timeline_daily.csv
  (b) every 80/20 (N=26) setup 2025-26   setups_2025_2026.csv, setup_episodes_2025_2026.csv
  (c) the July-2026 buy setup            results.json["c_july_setup"]
  (d) the latest report (2026-09-29)     results.json["d_latest"]
  (e) HYPOTHETICAL 2026-10-06 report     results.json["e_hypothetical_1006"]  (not data)
  (f) the rally, measured                rally.csv, results.json["f_rally"]
  (g) base rates and precursor test      base_rates_updown.csv, base_rates_precursor.csv,
                                         rally_clusters.csv, results.json["g_base_rates"]

Run:  python3 -I crude2026_case.py          (no arguments, no network, ~10 s)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[2])   # ROOT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))           # engine code
import larry_cot as lc  # noqa: E402  (engine API, see work/engine/larry_cot.py)

OUT = S / "work" / "lens_crude2026"

# ---------------------------------------------------------------------------------------
# Case-study constants (primary Larry setting from the spec: N=26, 80/20)
# ---------------------------------------------------------------------------------------
N = 26
BUY, SELL = 80, 20
TL_START = pd.Timestamp("2025-01-01")          # timeline / setup-table start
SUSPICIOUS_SPOT = pd.Timestamp("2026-09-25")   # EIA spot 85.23 vs CL 92.41 that day
JULY_REPORT = pd.Timestamp("2026-07-14")       # idx26 reached 100 on this report
LATEST_REPORT = pd.Timestamp("2026-09-29")     # last report in cot_weekly.csv
HYPO_REPORT = pd.Timestamp("2026-10-06")       # NOT in our data
# Unverified web snippets, legacy FUTURES-ONLY commercial net (a different basis from ours):
LEGACY_FO_0929, LEGACY_FO_1006 = -142618, -126458
HYPO_DELTA = LEGACY_FO_1006 - LEGACY_FO_0929   # = +16,160, applied to OUR comm_net as a what-if
RALLY_TH = 0.15                                # "big rally" = 8-week forward return > +15 %
PRIOR_REPORTS = 8                              # "setup in the 8 weeks before the start"


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------
def d2s(x):
    """Timestamp -> 'YYYY-MM-DD'; NaT/None -> None."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def jsonable(o):
    """Recursively turn numpy / pandas scalars into JSON-safe values (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return d2s(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if o is pd.NaT:
        return None
    return o


def nominal_release(report_dates) -> pd.DatetimeIndex:
    """Nominal COT publication day = Friday of the report week (15:30 ET). Holiday weeks can
    slip to the following Monday and shutdown-window reports were published late (catch-up);
    the exact catch-up dates are not in our data."""
    d = pd.DatetimeIndex(pd.to_datetime(report_dates))
    return d - pd.to_timedelta(d.weekday, unit="D") + pd.Timedelta(days=4)


def exit_point(prices: pd.Series, entry_pos: int, n_days: int):
    """(exit_date, exit_close) n_days trading rows after entry_pos, or (None, nan)."""
    if entry_pos < 0 or entry_pos + n_days >= len(prices):
        return None, float("nan")
    return prices.index[entry_pos + n_days], float(prices.iloc[entry_pos + n_days])


def row_at(frame: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    r = frame[frame["report_date"] == report_date]
    if len(r) != 1:
        raise KeyError(f"report {report_date.date()} not found exactly once")
    return r.iloc[0]


def required_comm_net(others: np.ndarray, th: float) -> dict:
    """Smallest integer comm_net v that makes the Larry index >= th when v is the newest
    report and `others` are the other N-1 reports of its window. Inside [lo, hi] the index
    is 100*(v-lo)/(hi-lo), so v* = lo + th/100*(hi-lo). The answer is verified with the
    engine's lc.larry_index (v* reaches th, v*-1 does not)."""
    others = np.asarray(others, dtype=np.int64)
    lo, hi = int(others.min()), int(others.max())
    v_real = lo + th / 100.0 * (hi - lo)
    v_int = int(math.ceil(v_real))
    chk_hit = lc.larry_index(np.append(others, v_int), len(others) + 1)[-1]
    chk_miss = lc.larry_index(np.append(others, v_int - 1), len(others) + 1)[-1]
    assert chk_hit >= th > chk_miss, (th, chk_hit, chk_miss)
    return dict(window_min_others=lo, window_max_others=hi, threshold=th,
                comm_net_needed_real=v_real, comm_net_needed_int=v_int,
                idx_at_needed=float(chk_hit), idx_at_needed_minus_1=float(chk_miss))


def path_after(prices: pd.Series, entry_date, entry_price: float) -> dict:
    """Max / min close strictly after entry, and the latest close, with returns vs entry."""
    after = prices[prices.index > entry_date]
    out = dict(series_last_date=d2s(prices.index[-1]))
    if len(after) == 0:
        return out
    out.update(max_close=float(after.max()), max_date=d2s(after.idxmax()),
               max_ret=float(after.max() / entry_price - 1),
               min_close=float(after.min()), min_date=d2s(after.idxmin()),
               min_ret=float(after.min() / entry_price - 1),
               latest_close=float(after.iloc[-1]), latest_date=d2s(after.index[-1]),
               latest_ret=float(after.iloc[-1] / entry_price - 1))
    return out


# ---------------------------------------------------------------------------------------
# Load data and build the crude frames with the engine
# ---------------------------------------------------------------------------------------
def load_all():
    cl = lc.load_prices(markets=["crude"])["crude"]          # CL continuous, cleaned
    spot = lc.load_wti_spot()                                 # EIA spot, as-is (keeps 85.23)
    f_cl = lc.build_panel(markets=["crude"])["crude"]
    f_sp = lc.build_panel(markets=["crude"], price_overrides={"crude": spot})["crude"]
    # Sanity: the engine's crude frame reproduces the published crude_timeseries.csv.
    ts = pd.read_csv(S / "work" / "engine" / "crude_timeseries.csv",
                     parse_dates=["report_date"], float_precision="round_trip")
    assert len(ts) == len(f_cl)
    assert (ts["comm_net"].to_numpy() == f_cl["comm_net"].to_numpy()).all()
    for c in ["idx26", "idx52", "idx156", "mm_idx26"]:
        assert np.allclose(ts[c].to_numpy(), f_cl[c].to_numpy(), equal_nan=True, atol=1e-9)
    return cl, spot, f_cl, f_sp


# ---------------------------------------------------------------------------------------
# (a) timeline
# ---------------------------------------------------------------------------------------
def build_timeline(cl, spot, f_cl, f_sp):
    lo_m, sh_m = lc.signal_masks(f_cl["idx26"].to_numpy(), BUY, SELL)
    w = pd.DataFrame({
        "report_date": f_cl["report_date"],
        "release_nominal": nominal_release(f_cl["report_date"]),
        "shutdown_delayed": f_cl["shutdown"].astype(bool),
        "comm_net": f_cl["comm_net"], "mm_net": f_cl["mm_net"], "oi": f_cl["oi"],
        "idx26": f_cl["idx26"], "idx52": f_cl["idx52"], "idx156": f_cl["idx156"],
        "mm_idx26": f_cl["mm_idx26"],
        "long_setup_80": lo_m, "short_setup_20": sh_m,
        "entry_date_cl": f_cl["entry_date"], "entry_price_cl": f_cl["entry_price"],
        "r4_cl": f_cl["r4"], "r8_cl": f_cl["r8"], "r13_cl": f_cl["r13"],
        "entry_date_spot": f_sp["entry_date"], "entry_price_spot": f_sp["entry_price"],
        "r4_spot": f_sp["r4"], "r8_spot": f_sp["r8"], "r13_spot": f_sp["r13"],
    })
    w = w[w["report_date"] >= TL_START].reset_index(drop=True)

    daily = pd.concat([cl.rename("cl_close"), spot.rename("spot_close")], axis=1, sort=True)
    daily = daily[daily.index >= TL_START]
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["spot_minus_cl"] = daily["spot_close"] - daily["cl_close"]
    daily["spot_flag"] = np.where(daily["date"] == SUSPICIOUS_SPOT,
                                  "suspicious print (spot 85.23 vs CL 92.41)", "")
    # A spot print identical to the previous spot print (possible holiday fill).
    sp_prev = spot.shift(1).reindex(daily["date"]).to_numpy()
    same = (daily["spot_close"].to_numpy() == sp_prev) & daily["spot_close"].notna().to_numpy()
    daily.loc[same & (daily["date"] >= pd.Timestamp("2026-10-01")), "spot_flag"] = \
        "identical to previous spot print"
    return w, daily


# ---------------------------------------------------------------------------------------
# (b) every 80/20 setup week in 2025-2026, and the episodes they belong to
# ---------------------------------------------------------------------------------------
def build_setups(cl, spot, f_cl, f_sp):
    idx = f_cl["idx26"].to_numpy()
    lo_m, sh_m = lc.signal_masks(idx, BUY, SELL)
    ep = {"long": lc.episode_ids(lo_m), "short": lc.episode_ids(sh_m)}
    rel = nominal_release(f_cl["report_date"])
    rows = []
    for i in np.flatnonzero((lo_m | sh_m) & (f_cl["report_date"] >= TL_START).to_numpy()):
        side = "long" if lo_m[i] else "short"
        sgn = 1.0 if side == "long" else -1.0
        r = dict(side=side, episode=f"{side[0].upper()}{ep[side][i]}",
                 report_date=f_cl["report_date"].iloc[i], release_nominal=rel[i],
                 shutdown_delayed=bool(f_cl["shutdown"].iloc[i]),
                 tradable_per_spec=not bool(f_cl["shutdown"].iloc[i]),
                 idx26=float(idx[i]), comm_net=int(f_cl["comm_net"].iloc[i]),
                 entry_date_cl=f_cl["entry_date"].iloc[i],
                 entry_price_cl=float(f_cl["entry_price"].iloc[i]))
        for h in (4, 8, 13):
            ed, _ = exit_point(cl, int(f_cl["entry_pos"].iloc[i]), 5 * h)
            r[f"exit_date_r{h}_cl"] = ed
            r[f"r{h}_cl"] = float(f_cl[f"r{h}"].iloc[i])
            r[f"signed_r{h}_cl"] = sgn * r[f"r{h}_cl"]
        r["entry_date_spot"] = f_sp["entry_date"].iloc[i]
        r["entry_price_spot"] = float(f_sp["entry_price"].iloc[i])
        touches = r["entry_date_spot"] == SUSPICIOUS_SPOT
        for h in (4, 8, 13):
            ed, _ = exit_point(spot, int(f_sp["entry_pos"].iloc[i]), 5 * h)
            r[f"exit_date_r{h}_spot"] = ed
            r[f"r{h}_spot"] = float(f_sp[f"r{h}"].iloc[i])
            touches = touches or (ed == SUSPICIOUS_SPOT)
        r["spot_uses_suspicious_0925"] = bool(touches)
        rows.append(r)
    setups = pd.DataFrame(rows)

    # Episodes (maximal runs on the full report sequence, as in the engine's "full" mode).
    erows = []
    for (side, epi), g in setups.groupby(["side", "episode"], sort=False):
        k = int(epi[1:])
        full = f_cl[ep[side] == k]                       # whole run, may start before 2025
        trad = g[g["tradable_per_spec"]]
        first_t = trad.iloc[0] if len(trad) else None
        e = dict(side=side, episode=epi,
                 first_report=full["report_date"].iloc[0], last_report=full["report_date"].iloc[-1],
                 n_weeks=len(full), n_weeks_in_2025_2026=len(g), n_tradable_weeks=len(trad),
                 idx26_min=float(full["idx26"].min()), idx26_max=float(full["idx26"].max()),
                 first_tradable_report=None if first_t is None else first_t["report_date"])
        for key in ["entry_date_cl", "entry_price_cl", "signed_r4_cl", "signed_r8_cl",
                    "signed_r13_cl"]:
            e[f"first_tradable_{key}"] = None if first_t is None else first_t[key]
        e["mean_signed_r8_cl_tradable_weeks"] = (float(trad["signed_r8_cl"].mean())
                                                 if trad["signed_r8_cl"].notna().any() else np.nan)
        e["n_tradable_weeks_with_r8"] = int(trad["signed_r8_cl"].notna().sum())
        erows.append(e)
    episodes = pd.DataFrame(erows).sort_values("first_report").reset_index(drop=True)

    # Aggregate of tradable 2025-26 setup weeks (CL; spot excludes weeks touching 09-25).
    agg = {}
    for side in ("long", "short"):
        g = setups[(setups["side"] == side) & setups["tradable_per_spec"]]
        for h in (4, 8, 13):
            v = g[f"signed_r{h}_cl"].dropna()
            agg[f"{side}_h{h}_cl"] = dict(n=int(len(v)),
                                          mean=float(v.mean()) if len(v) else None,
                                          hit=float((v > 0).mean()) if len(v) else None)
    return setups, episodes, agg


# ---------------------------------------------------------------------------------------
# (c) the July-2026 buy setup
# ---------------------------------------------------------------------------------------
def july_setup(cl, spot, f_cl, f_sp, setups):
    out = {}
    for name, px, fr in (("cl", cl, f_cl), ("spot", spot, f_sp)):
        row = row_at(fr, JULY_REPORT)
        pos = int(row["entry_pos"])
        d = dict(report_date=d2s(JULY_REPORT),
                 release_nominal=d2s(nominal_release([JULY_REPORT])[0]),
                 idx26=float(row["idx26"]), idx52=float(row["idx52"]),
                 idx156=float(row["idx156"]), mm_idx26=float(row["mm_idx26"]),
                 comm_net=int(row["comm_net"]),
                 entry_date=d2s(row["entry_date"]), entry_price=float(row["entry_price"]))
        for h in (1, 4, 8, 13):
            ed, ep_ = exit_point(px, pos, 5 * h)
            d[f"r{h}"] = float(row[f"r{h}"])
            d[f"exit_date_r{h}"] = d2s(ed)
            d[f"exit_price_r{h}"] = ep_
        july = px[(px.index >= "2026-07-01") & (px.index <= "2026-07-31")]
        d["july_low_close"], d["july_low_date"] = float(july.min()), d2s(july.idxmin())
        d["entry_vs_july_low"] = float(row["entry_price"] / july.min() - 1)
        rel_day = px[px.index <= nominal_release([JULY_REPORT])[0]]
        d["close_on_release_day"] = float(rel_day.iloc[-1])
        d["close_on_release_day_date"] = d2s(rel_day.index[-1])
        d["path_after_entry"] = path_after(px, row["entry_date"], float(row["entry_price"]))
        out[name] = d
    ep_label = setups.loc[setups["report_date"] == JULY_REPORT, "episode"].iloc[0]
    epw = setups[setups["episode"] == ep_label]
    out["episode"] = dict(label=ep_label,
                          reports=[d2s(x) for x in epw["report_date"]],
                          idx26=[float(x) for x in epw["idx26"]],
                          entry_dates_cl=[d2s(x) for x in epw["entry_date_cl"]],
                          entry_prices_cl=[float(x) for x in epw["entry_price_cl"]],
                          signed_r8_cl=[float(x) for x in epw["signed_r8_cl"]])
    # Near misses in 2026: 75 <= idx26 < 80 (would be a long setup under the 75/25 pair).
    nm = f_cl[(f_cl["report_date"] >= "2026-01-01") & (f_cl["idx26"] >= 75) & (f_cl["idx26"] < 80)]
    out["near_miss_75_80_in_2026"] = [
        dict(report_date=d2s(r.report_date), idx26=float(r.idx26),
             entry_date_cl=d2s(r.entry_date), entry_price_cl=float(r.entry_price),
             r4_cl=float(r.r4), r8_cl=float(r.r8), r13_cl=float(r.r13))
        for r in nm.itertuples()]
    # What the COT index read at the July low (last reports published before 2026-07-06).
    pre = f_cl[(f_cl["report_date"] >= "2026-06-09") & (f_cl["report_date"] <= "2026-07-07")]
    out["readings_around_july_low"] = [
        dict(report_date=d2s(r.report_date), release_nominal=d2s(nominal_release([r.report_date])[0]),
             idx26=float(r.idx26), mm_idx26=float(r.mm_idx26), comm_net=int(r.comm_net))
        for r in pre.itertuples()]
    return out


# ---------------------------------------------------------------------------------------
# (d) the latest report and (e) the hypothetical 2026-10-06 report
# ---------------------------------------------------------------------------------------
def latest_and_hypo(cl, spot, f_cl, f_sp):
    cn = f_cl["comm_net"].to_numpy(dtype=np.int64)
    i_last = int(np.flatnonzero(f_cl["report_date"].to_numpy() == LATEST_REPORT.to_datetime64())[0])
    assert i_last == len(f_cl) - 1, "2026-09-29 should be the last report"
    row = f_cl.iloc[i_last]
    rd = f_cl["report_date"]

    sig = {}
    for n in lc.LOOKBACKS:
        for b, s in lc.THRESHOLDS:
            v = float(row[f"idx{n}"])
            sig[f"idx{n}_{b}_{s}"] = dict(value=v, long=v >= b, short=v <= s)
    win = slice(i_last - N + 1, i_last + 1)
    win_vals, win_dates = cn[win], rd.iloc[win]
    d = dict(report_date=d2s(LATEST_REPORT),
             release_nominal=d2s(nominal_release([LATEST_REPORT])[0]),
             entry_target=d2s(lc.entry_target([LATEST_REPORT])[0]),
             comm_net=int(row["comm_net"]), mm_net=int(row["mm_net"]), oi=int(row["oi"]),
             idx26=float(row["idx26"]), idx52=float(row["idx52"]), idx156=float(row["idx156"]),
             mm_idx26=float(row["mm_idx26"]),
             signal_checks=sig,
             any_long_signal=any(x["long"] for x in sig.values()),
             any_short_signal=any(x["short"] for x in sig.values()),
             cl_entry_available=bool(row["entry_pos"] >= 0),
             cl_last_date=d2s(cl.index[-1]),
             window26_first=d2s(win_dates.iloc[0]), window26_last=d2s(win_dates.iloc[-1]),
             window26_min=int(win_vals.min()), window26_min_date=d2s(win_dates.iloc[int(np.argmin(win_vals))]),
             window26_max=int(win_vals.max()), window26_max_date=d2s(win_dates.iloc[int(np.argmax(win_vals))]))
    srow = row_at(f_sp, LATEST_REPORT)
    d["spot_entry_date"] = d2s(srow["entry_date"])
    d["spot_entry_price"] = float(srow["entry_price"])
    d["spot_path_after_entry"] = path_after(spot, srow["entry_date"], float(srow["entry_price"]))
    prev = f_cl.iloc[i_last - 1]
    d["previous_report"] = dict(report_date=d2s(prev["report_date"]), idx26=float(prev["idx26"]),
                                mm_idx26=float(prev["mm_idx26"]), comm_net=int(prev["comm_net"]))

    # Required comm_net for idx26 = 75 / 80:
    #  (i)  re-reading 2026-09-29 itself (same 26-week window, other 25 reports fixed)
    #  (ii) the next report, 2026-10-06 (window rolls forward one week)
    req = {}
    others_same = cn[i_last - N + 1:i_last]               # 25 reports before 09-29
    others_next = cn[i_last - N + 2:i_last + 1]           # 25 reports ending 09-29
    for label, others in (("same_window_0929", others_same), ("next_report_1006", others_next)):
        for th in (75, 80):
            q = required_comm_net(others, th)
            q["change_vs_0929"] = q["comm_net_needed_int"] - int(row["comm_net"])
            q["change_vs_0929_real"] = q["comm_net_needed_real"] - int(row["comm_net"])
            req[f"{label}_th{th}"] = q
    d["required"] = req
    # How often has comm_net risen by at least that much in one week (all crude history)?
    dif = np.diff(cn)
    d["weekly_change_history"] = dict(
        n_weekly_changes=int(len(dif)),
        share_ge_needed_75=float(np.mean(dif >= req["next_report_1006_th75"]["change_vs_0929"])),
        share_ge_needed_80=float(np.mean(dif >= req["next_report_1006_th80"]["change_vs_0929"])),
        share_ge_hypo_delta=float(np.mean(dif >= HYPO_DELTA)),
        median_abs_change=float(np.median(np.abs(dif))))

    # (e) HYPOTHETICAL: our comm_net + 16,160 on 2026-10-06 (window rolls forward one week).
    v_h = int(row["comm_net"]) + HYPO_DELTA
    ext = np.append(cn, v_h)
    e = dict(LABEL="HYPOTHETICAL - NOT DATA. 2026-10-06 is not in cot_weekly.csv; the +16,160 is "
                   "the change in LEGACY FUTURES-ONLY commercial net from unverified web snippets "
                   "(-142,618 -> -126,458), applied to our DISAGGREGATED F+O comm_net.",
             legacy_fo_0929=LEGACY_FO_0929, legacy_fo_1006=LEGACY_FO_1006, delta=HYPO_DELTA,
             our_comm_net_0929=int(row["comm_net"]), hypo_comm_net_1006=v_h,
             basis_gap_0929=int(row["comm_net"]) - LEGACY_FO_0929)
    for n in lc.LOOKBACKS:
        e[f"hypo_idx{n}"] = float(lc.larry_index(ext, n)[-1])
    e["hypo_long_80"] = e["hypo_idx26"] >= 80
    e["hypo_long_75"] = e["hypo_idx26"] >= 75
    e["hypo_window26_min"] = int(ext[-N:].min())
    e["hypo_window26_max"] = int(ext[-N:].max())
    e["hypo_entry_target"] = d2s(lc.entry_target([HYPO_REPORT])[0])
    e["hypo_release_nominal"] = d2s(nominal_release([HYPO_REPORT])[0])
    # Conditional on the hypothetical 10-06 value: what would 2026-10-13 need?
    for th in (75, 80):
        q = required_comm_net(ext[-(N - 1):], th)
        q["change_vs_hypo_1006"] = q["comm_net_needed_int"] - v_h
        e[f"next_report_1013_th{th}_conditional"] = q
    # Spot prices around the hypothetical report (CL ends 2026-10-05).
    for lab, dt in (("release_day_1009", "2026-10-09"), ("entry_1013", "2026-10-13"),
                    ("latest", None)):
        s = spot if dt is None else spot[spot.index <= dt]
        e[f"spot_{lab}"] = dict(date=d2s(s.index[-1]), close=float(s.iloc[-1]))
    e["spot_entry_1013_to_latest"] = e["spot_latest"]["close"] / e["spot_entry_1013"]["close"] - 1
    return d, e


# ---------------------------------------------------------------------------------------
# (f) the rally, defined and measured in both price series
# ---------------------------------------------------------------------------------------
def rally_measures(px: pd.Series, label: str) -> dict:
    """Points (all closes):
      july_low       = lowest close in calendar July 2026
      peak           = highest close after the July low
      late_sep_low   = lowest close after the peak, up to 2026-09-30
      latest         = last close in the series
      at_2026_10_05  = close on 2026-10-05 (last CL date, common end for comparison)
      oct_low        = lowest close from 2026-10-01 to the latest close"""
    july = px[(px.index >= "2026-07-01") & (px.index <= "2026-07-31")]
    jl_d, jl = july.idxmin(), float(july.min())
    after = px[px.index >= jl_d]
    pk_d, pk = after.idxmax(), float(after.max())
    late = px[(px.index > pk_d) & (px.index <= "2026-09-30")]
    ls_d, ls = late.idxmin(), float(late.min())
    lt_d, lt = px.index[-1], float(px.iloc[-1])
    c_d = px[px.index <= "2026-10-05"].index[-1]
    c = float(px.loc[c_d])
    octo = px[px.index >= "2026-10-01"]
    oc_d, oc = octo.idxmin(), float(octo.min())
    return dict(series=label,
                july_low_date=d2s(jl_d), july_low=jl, peak_date=d2s(pk_d), peak=pk,
                late_sep_low_date=d2s(ls_d), late_sep_low=ls, latest_date=d2s(lt_d), latest=lt,
                close_2026_10_05_date=d2s(c_d), close_2026_10_05=c,
                july_low_to_peak=pk / jl - 1, july_low_to_latest=lt / jl - 1,
                peak_to_late_sep_low=ls / pk - 1, late_sep_low_to_latest=lt / ls - 1,
                late_sep_low_to_2026_10_05=c / ls - 1, latest_vs_peak=lt / pk - 1,
                oct_low_date=d2s(oc_d), oct_low=oc, oct_low_to_latest=lt / oc - 1)


def build_rally(cl, spot):
    spot_x = spot.drop(SUSPICIOUS_SPOT)            # spot without the suspicious print
    rows = [rally_measures(cl, "CL continuous (to 2026-10-05)"),
            rally_measures(spot, "EIA spot as-is (incl. 2026-09-25 = 85.23)"),
            rally_measures(spot_x, "EIA spot excl. 2026-09-25")]
    rally = pd.DataFrame(rows)
    # Spot-minus-CL gap before / after the late-September break.
    both = pd.concat([cl.rename("cl"), spot.rename("sp")], axis=1, sort=True).dropna()
    gap = both["sp"] - both["cl"]
    g = dict(mean_gap_2026_09_01_to_09_24=float(gap["2026-09-01":"2026-09-24"].mean()),
             gap_2026_09_25=float(gap.loc["2026-09-25"]),
             mean_gap_2026_09_28_to_10_05=float(gap["2026-09-28":"2026-10-05"].mean()),
             mean_gap_2025=float(gap["2025-01-01":"2025-12-31"].mean()),
             max_abs_gap_2025=float(gap["2025-01-01":"2025-12-31"].abs().max()),
             cl_0923_to_0928=float(cl.loc["2026-09-28"] / cl.loc["2026-09-23"] - 1),
             spot_0923_to_0928=float(spot.loc["2026-09-28"] / spot.loc["2026-09-23"] - 1),
             spot_0928_to_latest=float(spot.iloc[-1] / spot.loc["2026-09-28"] - 1))
    # [verifier addition] 2025 is not the only comparison base: 2026 itself already had a
    # sustained multi-dollar spot premium over CL (2026-04-23..06-22). Gap regime changes in
    # 2026 start 1-3 trading days after CL contract expiries (CME rule: 3 business days before
    # the 25th of the prior month), which points to a roll / term-structure mismatch between
    # the two series rather than a spot data error (the single 09-25 = 85.23 print excepted).
    g26 = gap["2026-01-01":"2026-09-24"]
    g.update(mean_gap_2026_04_23_to_06_22=float(gap["2026-04-23":"2026-06-22"].mean()),
             n_days_gap_gt3_2026_before_0925=int((g26 > 3).sum()),
             max_abs_gap_2026_before_0925=float(g26.abs().max()),
             max_abs_gap_2026_before_0925_date=d2s(g26.abs().idxmax()),
             max_abs_gap_2010_2025=float(gap["2010-01-01":"2025-12-31"].abs().max()),
             max_abs_gap_2010_2025_date=d2s(gap["2010-01-01":"2025-12-31"].abs().idxmax()))
    cl_expiries_2026 = ["2026-03-20", "2026-04-21", "2026-06-22", "2026-09-22"]
    g["gap_around_cl_expiries_2026"] = {
        e: {d2s(k): float(v) for k, v in
            gap[pd.Timestamp(e) - pd.Timedelta(days=2):pd.Timestamp(e) + pd.Timedelta(days=6)].items()}
        for e in cl_expiries_2026}
    # Price on the publication days of the reports around the move (closes at/before day).
    pub = {}
    for lab, dt in (("rel_0929_report_fri_1002", "2026-10-02"),
                    ("rel_1006_report_fri_1009", "2026-10-09")):
        for nm, px in (("cl", cl), ("spot", spot)):
            s = px[px.index <= dt]
            pub[f"{lab}_{nm}"] = dict(date=d2s(s.index[-1]), close=float(s.iloc[-1]))
    spot_low = rows[2]["late_sep_low"]   # spot low excluding the suspicious print
    pub["spot_late_sep_low_excl_to_1009"] = pub["rel_1006_report_fri_1009_spot"]["close"] / spot_low - 1
    pub["spot_1009_to_latest"] = float(spot.iloc[-1]) / pub["rel_1006_report_fri_1009_spot"]["close"] - 1
    pub["spot_1002_to_latest"] = float(spot.iloc[-1]) / pub["rel_0929_report_fri_1002_spot"]["close"] - 1
    return rally, dict(points=rows, spot_minus_cl_gap=g, publication_days=pub)


# ---------------------------------------------------------------------------------------
# (g) base rates: idx26 >= 80 vs unconditional, and the rally-precursor test
# ---------------------------------------------------------------------------------------
def updown_rates(frame: pd.DataFrame, label: str, include_shutdown=False) -> list[dict]:
    rows = []
    for h in (8, 13):
        a = lc.cell_arrays(frame, "idx26", BUY, SELL, "long", f"r{h}", include_shutdown)
        sig, r = a["sig"], a["r"]
        up = (r > 0).astype(float)
        st = lc.cell_stats(frame, "idx26", BUY, SELL, "long", f"r{h}",
                           include_shutdown=include_shutdown, exact_p=True)
        rows.append(dict(prices=label, include_shutdown=include_shutdown, horizon_w=h,
                         T_eligible_weeks=int(len(sig)), n_setup_weeks=int(sig.sum()),
                         n_episodes=st["n_episodes"],
                         p_up_given_setup=float(up[sig].mean()),
                         p_up_unconditional=float(up.mean()),
                         p_up_given_no_setup=float(up[~sig].mean()),
                         mean_r_given_setup=float(r[sig].mean()),
                         mean_r_unconditional=float(r.mean()),
                         median_r_given_setup=float(np.median(r[sig])),
                         median_r_unconditional=float(np.median(r)),
                         perm_p_hit_rate=lc.perm_pvalue(sig, up),
                         perm_p_hit_rate_exact=lc.perm_pvalue_exact(sig, up),
                         perm_p_mean_ret=st["perm_p"], perm_p_mean_ret_exact=st["perm_p_exact"]))
    return rows


def updown_subperiods(frame: pd.DataFrame, label: str) -> list[dict]:
    """[verifier addition] Same up/down base rates (h=8) split into 2006-2016 and 2017-2026,
    to show where the full-sample shortfall comes from. Permutation p (exact, circular shift
    within the sub-period) via the engine's lc.perm_pvalue_exact."""
    rows = []
    a = lc.cell_arrays(frame, "idx26", BUY, SELL, "long", "r8", False)
    dates = frame["report_date"].to_numpy()[a["elig"]]
    for lo, hi in (("2006-01-01", "2016-12-31"), ("2017-01-01", "2026-12-31")):
        m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
        sig, r = a["sig"][m], a["r"][m]
        up = (r > 0).astype(float)
        rows.append(dict(prices=label, period=f"{lo[:4]}-{hi[:4]}", horizon_w=8,
                         T_eligible_weeks=int(m.sum()), n_setup_weeks=int(sig.sum()),
                         p_up_given_setup=float(up[sig].mean()), p_up_unconditional=float(up.mean()),
                         mean_r_given_setup=float(r[sig].mean()), mean_r_unconditional=float(r.mean()),
                         excess=float(r[sig].mean() - r.mean()),
                         perm_p_hit_rate_exact=lc.perm_pvalue_exact(sig, up),
                         perm_p_mean_ret_exact=lc.perm_pvalue_exact(sig, r)))
    return rows


def precursor(frame: pd.DataFrame, label: str, include_shutdown=False,
              strict_flag=False, rally_th=RALLY_TH, want_clusters=False):
    """8-week windows start at each eligible report's entry date; 'rally' = r8 > rally_th.
    flag = a long setup (idx26 >= 80) on any of the last 8 reports (i-7..i), all of which are
    published before the window starts. strict_flag drops shutdown-delayed reports from the
    flag (they were not published in real time)."""
    a = lc.cell_arrays(frame, "idx26", BUY, SELL, "long", "r8", include_shutdown)
    setup_full, elig = a["setup"].copy(), a["elig"]
    if strict_flag:
        setup_full &= ~frame["shutdown"].to_numpy(dtype=bool)
    flag_full = (pd.Series(setup_full.astype(int)).rolling(PRIOR_REPORTS, min_periods=1)
                 .max().to_numpy() > 0)
    r8 = frame["r8"].to_numpy(dtype=float)
    rally_full = elig & (np.nan_to_num(r8, nan=-np.inf) > rally_th)
    flag, rally, now = flag_full[elig], rally_full[elig], a["sig"]
    cl_ids = lc.episode_ids(rally_full)            # clusters of consecutive rally weeks
    starts = [int(np.flatnonzero(cl_ids == k)[0]) for k in range(1, cl_ids.max() + 1)]
    res = dict(prices=label, include_shutdown=include_shutdown, strict_flag=strict_flag,
               rally_threshold=rally_th, T_eligible_weeks=int(len(flag)),
               n_rally_weeks=int(rally.sum()), n_rally_clusters=len(starts),
               n_rally_weeks_flagged=int((flag & rally).sum()), n_weeks_flagged=int(flag.sum()),
               n_weeks_setup_now=int(now.sum()),
               n_rally_clusters_flagged_at_start=int(sum(flag_full[i] for i in starts)),
               share_rally_weeks_flagged=float(flag[rally].mean()),
               share_all_weeks_flagged=float(flag.mean()),
               share_nonrally_weeks_flagged=float(flag[~rally].mean()),
               share_all_weeks_setup_now=float(now.mean()),
               share_rally_weeks_setup_now=float(now[rally].mean()),
               share_rally_clusters_flagged_at_start=float(np.mean([flag_full[i] for i in starts])),
               p_rally=float(rally.mean()), p_rally_given_flag=float(rally[flag].mean()),
               p_rally_given_no_flag=float(rally[~flag].mean()),
               perm_p_rally_given_flag=lc.perm_pvalue(flag, rally.astype(float)),
               perm_p_rally_given_flag_exact=lc.perm_pvalue_exact(flag, rally.astype(float)))
    res["lift_p_rally_given_flag"] = res["p_rally_given_flag"] / res["p_rally"]
    if not want_clusters:
        return res, None
    rd = frame["report_date"]
    idx = frame["idx26"].to_numpy()
    crow = []
    for k, i in enumerate(starts, 1):
        members = np.flatnonzero(cl_ids == k)
        lo_i = max(0, i - PRIOR_REPORTS + 1)
        crow.append(dict(cluster=k, first_report=d2s(rd.iloc[i]),
                         first_entry_date=d2s(frame["entry_date"].iloc[i]),
                         first_entry_price=float(frame["entry_price"].iloc[i]),
                         last_report=d2s(rd.iloc[members[-1]]), n_weeks=len(members),
                         max_r8=float(np.nanmax(r8[members])),
                         r8_first_week=float(r8[i]),
                         flag_setup_in_prior8_at_start=bool(flag_full[i]),
                         idx26_at_start=float(idx[i]),
                         max_idx26_prior8=float(np.nanmax(idx[lo_i:i + 1])),
                         any_week_of_cluster_flagged=bool(flag_full[members].any())))
    return res, pd.DataFrame(crow)


def build_base_rates(f_cl, f_sp):
    up = (updown_rates(f_cl, "CL") + updown_rates(f_sp, "EIA spot")
          + updown_rates(f_cl, "CL", include_shutdown=True))
    pre_rows, clusters = [], None
    main, clusters = precursor(f_cl, "CL", want_clusters=True)
    pre_rows.append(main)
    for kw in (dict(strict_flag=True), dict(include_shutdown=True)):
        pre_rows.append(precursor(f_cl, "CL", **kw)[0])
    pre_rows.append(precursor(f_sp, "EIA spot")[0])
    for th in (0.10, 0.20):
        pre_rows.append(precursor(f_cl, "CL", rally_th=th)[0])
    sub = updown_subperiods(f_cl, "CL") + updown_subperiods(f_sp, "EIA spot")
    return pd.DataFrame(up), pd.DataFrame(pre_rows), clusters, pd.DataFrame(sub)


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cl, spot, f_cl, f_sp = load_all()

    # (a)
    weekly, daily = build_timeline(cl, spot, f_cl, f_sp)
    weekly.to_csv(OUT / "timeline_weekly.csv", index=False, date_format="%Y-%m-%d")
    daily.to_csv(OUT / "timeline_daily.csv", index=False, date_format="%Y-%m-%d")

    # (b)
    setups, episodes, setup_agg = build_setups(cl, spot, f_cl, f_sp)
    setups.to_csv(OUT / "setups_2025_2026.csv", index=False, date_format="%Y-%m-%d")
    episodes.to_csv(OUT / "setup_episodes_2025_2026.csv", index=False, date_format="%Y-%m-%d")

    # (c), (d), (e), (f), (g)
    july = july_setup(cl, spot, f_cl, f_sp, setups)
    latest, hypo = latest_and_hypo(cl, spot, f_cl, f_sp)
    rally_tbl, rally_info = build_rally(cl, spot)
    rally_tbl.to_csv(OUT / "rally.csv", index=False)
    br_up, br_pre, clusters, br_sub = build_base_rates(f_cl, f_sp)
    br_up.to_csv(OUT / "base_rates_updown.csv", index=False)
    br_sub.to_csv(OUT / "base_rates_updown_subperiods.csv", index=False)
    br_pre.to_csv(OUT / "base_rates_precursor.csv", index=False)
    clusters.to_csv(OUT / "rally_clusters.csv", index=False)

    results = dict(
        meta=dict(engine="work/engine/larry_cot.py", lookback=N, buy_th=BUY, sell_th=SELL,
                  cot_basis="CFTC disaggregated, futures+options combined; comm_net = "
                            "(prod/merc + swap) long - short",
                  prices_primary="NYMEX CL continuous front month (unadjusted), ends 2026-10-05",
                  prices_alt="EIA Cushing WTI spot, ends 2026-10-14 (2026-09-25 = 85.23 suspicious)",
                  perm=dict(k=lc.K_PERM, seed=lc.DEFAULT_SEED, min_shift=lc.MIN_SHIFT)),
        b_setups_2025_2026=dict(n_setup_weeks=int(len(setups)),
                                n_long_weeks=int((setups["side"] == "long").sum()),
                                n_short_weeks=int((setups["side"] == "short").sum()),
                                n_tradable_long_weeks=int(((setups["side"] == "long")
                                                           & setups["tradable_per_spec"]).sum()),
                                n_tradable_short_weeks=int(((setups["side"] == "short")
                                                            & setups["tradable_per_spec"]).sum()),
                                episodes=episodes.to_dict(orient="records"),
                                tradable_aggregate_cl=setup_agg),
        c_july_setup=july, d_latest=latest, e_hypothetical_1006=hypo, f_rally=rally_info,
        g_base_rates=dict(updown=br_up.to_dict(orient="records"),
                          updown_subperiods=br_sub.to_dict(orient="records"),
                          precursor=br_pre.to_dict(orient="records"),
                          clusters_since_2025=clusters[clusters["first_report"] >= "2025-01-01"]
                          .to_dict(orient="records")))
    (OUT / "results.json").write_text(json.dumps(jsonable(results), indent=1))

    # timeline.json: chart-ready weekly COT + daily prices + annotated events.
    rp = {r["series"]: r for r in rally_info["points"]}
    clp = rp["CL continuous (to 2026-10-05)"]
    spx = rp["EIA spot excl. 2026-09-25"]
    events = [
        dict(date=clp["july_low_date"], kind="price", label=f"CL July low {clp['july_low']:.2f}"),
        dict(date=d2s(JULY_REPORT), kind="cot", label="Report: idx26 = 100 (long setup)"),
        dict(date=july["cl"]["release_nominal"], kind="cot", label="That report published (Fri)"),
        dict(date=july["cl"]["entry_date"], kind="entry",
             label=f"Spec entry, CL close {july['cl']['entry_price']:.2f}"),
        dict(date=clp["peak_date"], kind="price", label=f"CL peak {clp['peak']:.2f}"),
        dict(date=clp["late_sep_low_date"], kind="price",
             label=f"CL late-Sep low {clp['late_sep_low']:.2f}"),
        dict(date=spx["late_sep_low_date"], kind="price",
             label=f"Spot late-Sep low excl. 09-25: {spx['late_sep_low']:.2f}"),
        dict(date=d2s(SUSPICIOUS_SPOT), kind="data_issue", label="Suspicious spot print 85.23"),
        dict(date=d2s(LATEST_REPORT), kind="cot",
             label=f"Latest report in data: idx26 = {latest['idx26']:.2f} (no signal)"),
        dict(date=latest["release_nominal"], kind="cot", label="Latest report published (Fri)"),
        dict(date=d2s(HYPO_REPORT), kind="hypothetical",
             label=f"HYPOTHETICAL (not data): idx26 = {hypo['hypo_idx26']:.2f} if comm_net +16,160"),
    ]
    wk = weekly.copy()
    for c in ["report_date", "release_nominal", "entry_date_cl", "entry_date_spot"]:
        wk[c] = wk[c].map(d2s)
    dl = daily.copy()
    dl["date"] = dl["date"].map(d2s)
    timeline = dict(
        meta=dict(title="Crude oil: Larry COT index (commercials) vs price, 2025-01-01 -> latest",
                  weekly_note="One row per COT report (as-of Tuesday). idx = 100*(comm_net-min)/(max-min) "
                              "over the last N reports incl. the current one. Long setup idx26>=80, short <=20. "
                              "shutdown_delayed rows were published late (not tradable per spec). "
                              "release_nominal = Friday of the report week.",
                  daily_note="cl_close = NYMEX CL continuous (ends 2026-10-05); spot_close = EIA Cushing spot "
                             "(ends 2026-10-14). 2026-09-25 spot print is suspicious.",
                  thresholds=dict(buy=BUY, sell=SELL),
                  hypothetical_note=hypo["LABEL"]),
        weekly=wk.to_dict(orient="records"), daily=dl.to_dict(orient="records"),
        events=events,
        hypothetical=dict(report_date=d2s(HYPO_REPORT), comm_net=hypo["hypo_comm_net_1006"],
                          idx26=hypo["hypo_idx26"], idx52=hypo["hypo_idx52"],
                          idx156=hypo["hypo_idx156"], label=hypo["LABEL"]))
    (OUT / "timeline.json").write_text(json.dumps(jsonable(timeline)))

    # Console summary
    print(f"(d) latest {latest['report_date']}: idx26={latest['idx26']:.4f} idx52={latest['idx52']:.4f} "
          f"idx156={latest['idx156']:.4f} mm_idx26={latest['mm_idx26']:.4f} "
          f"any long={latest['any_long_signal']} any short={latest['any_short_signal']}")
    for k, q in latest["required"].items():
        print(f"    {k}: need comm_net >= {q['comm_net_needed_int']} (change {q['change_vs_0929']:+d})")
    print(f"(e) HYPOTHETICAL 10-06: comm_net={hypo['hypo_comm_net_1006']} idx26={hypo['hypo_idx26']:.4f}")
    print(f"(c) July setup CL: entry {july['cl']['entry_date']} @ {july['cl']['entry_price']}, "
          f"r4={july['cl']['r4']:.4f} r8={july['cl']['r8']:.4f} r13={july['cl']['r13']}")
    print(rally_tbl.T.to_string())
    print(br_up.T.to_string())
    print(br_pre.T.to_string())
    print(f"wrote outputs to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

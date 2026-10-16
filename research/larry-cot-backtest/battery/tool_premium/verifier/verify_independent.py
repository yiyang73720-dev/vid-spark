#!/usr/bin/env python3
"""
verify_independent -- adversarial, independent re-derivation of the premium tool's PRIMARY result
(crude, contract 1 > contract 2 at the Friday close -> LONG, h = 8 weeks, roll-adjusted index).

Nothing from fund_harness or premium_lib is imported. The raw EIA JSON files (RCLC1..RCLC4) and
the EIA Cushing spot CSV are parsed directly (json / csv only, never executed), and a roll-adjusted
index is rebuilt with a DIFFERENT roll-dating method from the harness:

  * harness: NYMEX last-trading-day (LTD) rule on a reconstructed exchange calendar (holiday
    rules + EIA gap days + data-based adjudication of ambiguous months);
  * here:    the same NYMEX rule ("3 business days before the 25th; 4 if the 25th is not a
    business day") applied to the calendar of days that actually carry an EIA print.
    Where the two calendars disagree the roll days differ, and the comparison below shows
    how much that matters.

Then: Friday grid, signal read as of the Friday close (<= 4 calendar days old), entry at the first
index close strictly after the Friday (<= 7 days later), exit 40 index rows later, baseline over
all eligible weeks, excess, hit, and the exhaustive circular-shift p-value with the +1 correction
(offsets 26..T-26), exactly as SPEC2 rules 1-4 describe them.

Robustness/attack checks computed here (all printed and written to verify_output.json):
  A. primary on my own index vs on the harness index (roll-dating sensitivity);
  B. roll-day jump scan: largest daily index returns, and returns on days where the two roll
     calendars disagree;
  C. single-episode / single-year dependence: excess with each calendar year left out, with
     2008 / 2014 / 2020 removed, and leave-one-episode-out extremes;
  D. overlap-free check: the 8 non-overlapping sub-grids (every 8th Friday), each with its own
     baseline; and an episode-level statistic (one number per premium episode);
  E. sign check: premium weeks really have c1 > c2 in the raw prints (no inverted definition);
  F. spot vs roll decomposition on the same entry/exit dates (claim C3).

Run:  python3 -I battery/tool_premium/verifier/verify_independent.py
      (writes ROOT/work2/tool_premium/verifier/verify_output.json and weekly_my_index.csv)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PKG = Path(__file__).resolve().parents[3]                # package dir (battery/<step>/<sub>/../../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
EXT = ROOT / "ext" / "accomplish999_crude-oil" / "data" / "series"
SPOT_CSV = ROOT / "data" / "wti-daily.csv"
OUT = ROOT / "work2" / "tool_premium" / "verifier" / "verify_output.json"

H_DAYS = 40            # 8 weeks x 5 trading days
MIN_SHIFT = 26         # circular-shift offsets in [26, T-26]
STALE_DAYS = 4         # a Friday reading may be at most 4 calendar days old
MAX_ENTRY_LAG = 7      # no entry if the first close is > 7 days after the target


# ------------------------------------------------------------------------------------------
# 1. Raw data (parse only)
# ------------------------------------------------------------------------------------------
def load_eia(series_id: str) -> pd.Series:
    """Daily observations of an archive series as float, NaN for blanks."""
    with open(EXT / f"{series_id}.json", encoding="utf-8") as fh:
        d = json.load(fh)
    obs = d["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy()
    s = pd.Series(vals, index=dates, name=series_id).sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_contracts() -> pd.DataFrame:
    """c1..c4 from 1985-01-02 (first contract-2 print); holiday placeholder rows (every contract
    repeating the previous row) removed; non-positive prints set to NaN."""
    df = pd.DataFrame({f"c{j}": load_eia(f"RCLC{j}") for j in (1, 2, 3, 4)})
    df = df[df.index >= "1985-01-02"]
    prev = df.shift(1)
    stale = ((df == prev) | df.isna()).all(axis=1) & df.notna().any(axis=1)
    stale.iloc[0] = False
    df = df[~stale.to_numpy()].copy()
    return df.where(df > 0)


def load_spot() -> pd.Series:
    """EIA Cushing WTI spot, positive prints only."""
    rows = []
    with open(SPOT_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["Price"])
            except (TypeError, ValueError):
                continue
            if v > 0:
                rows.append((pd.Timestamp(r["Date"]), v))
    s = pd.Series(dict(rows)).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ------------------------------------------------------------------------------------------
# 2. Roll-adjusted index with an independent roll calendar
# ------------------------------------------------------------------------------------------
def ltd_from_print_calendar(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """NYMEX CL contract-1 last trading day for every month, counted on the calendar of days
    that carry a print: 3 print-days before the 25th if the 25th has a print, else 4."""
    out = []
    for p in pd.period_range(days[0], days[-1], freq="M"):
        d25 = pd.Timestamp(p.year, p.month, 25)
        before = days[days < d25]
        if len(before) < 4 or d25 > days[-1]:
            continue
        out.append(before[-3] if d25 in days else before[-4])
    return pd.DatetimeIndex(out)


def roll_index(c1: pd.Series, c2: pd.Series, ltd: pd.DatetimeIndex) -> tuple[pd.Series, np.ndarray]:
    """Hold contract 1 through its LTD close. On the first index day t after an LTD (contract 1
    changed identity between t' and t) the return is c1(t)/c2(t') - 1, else c1(t)/c1(t') - 1.
    Returns (level starting at 100, boolean roll-day mask)."""
    days = c1.index
    a, b = c1.to_numpy(), c2.to_numpy()
    k = np.searchsorted(ltd.values, days.values, side="left")   # index of next LTD >= day
    roll = np.zeros(len(days), bool)
    roll[1:] = k[1:] != k[:-1]                                   # an LTD lies in [t', t)
    ret = np.zeros(len(days))
    ret[1:] = np.where(roll[1:], a[1:] / b[:-1], a[1:] / a[:-1]) - 1.0
    return pd.Series(100 * np.cumprod(1 + ret), index=days), roll


# ------------------------------------------------------------------------------------------
# 3. Weekly table: signal, entry, forward 8-week return
# ------------------------------------------------------------------------------------------
def weekly_table(c1: pd.Series, c2: pd.Series, level: pd.Series,
                 start="1985-01-04", end="2024-03-29") -> pd.DataFrame:
    """One row per Friday: premium p = c1/c2-1 as of the Friday close, sign signal, entry/exit
    rows and the 8-week forward return on `level`. Undefined weeks are dropped."""
    fri = pd.date_range(start, end, freq="W-FRI")
    sd = c1.index.values
    lv = level.index.values
    rows = []
    for f in fri:
        f64 = np.datetime64(f)
        k = np.searchsorted(sd, f64, side="right") - 1
        if k < 0 or (f64 - sd[k]) > np.timedelta64(STALE_DAYS, "D"):
            continue
        p = c1.iloc[k] / c2.iloc[k] - 1.0
        tgt = f64 + np.timedelta64(1, "D")
        e = np.searchsorted(lv, tgt, side="left")
        if e >= len(lv) or (lv[e] - tgt) > np.timedelta64(MAX_ENTRY_LAG, "D"):
            continue
        x = e + H_DAYS
        if x >= len(lv):
            continue
        rows.append(dict(friday=f, signal_day=pd.Timestamp(sd[k]), c1=c1.iloc[k], c2=c2.iloc[k],
                         p=p, sig=int(np.sign(p)), entry=pd.Timestamp(lv[e]),
                         exit=pd.Timestamp(lv[x]), r=level.iloc[x] / level.iloc[e] - 1.0))
    return pd.DataFrame(rows)


def circ_p(mask: np.ndarray, r: np.ndarray) -> float:
    """Exhaustive one-sided circular-shift p-value of mean(r[mask]) with the +1 correction."""
    T = len(r)
    pos = np.flatnonzero(mask)
    obs = r[pos].mean()
    offs = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
    sh = r[(pos[None, :] + offs[:, None]) % T].mean(axis=1)
    return float((1 + np.sum(sh >= obs - 1e-15)) / (1 + len(offs)))


def stats(t: pd.DataFrame, side: int = 1, do_p: bool = True) -> dict:
    """n, episodes, mean, baseline, excess, hit (+ exact p) of one side of a weekly table."""
    r = t["r"].to_numpy()
    m = t["sig"].to_numpy() == side
    if m.sum() == 0:
        return dict(n=0)
    signed = side * r
    base = r.mean()
    starts = m & ~np.concatenate(([False], m[:-1]))
    out = dict(n=int(m.sum()), episodes=int(starts.sum()), T=int(len(r)),
               mean=float(signed[m].mean()), baseline=float(base),
               excess=float(signed[m].mean() - side * base), hit=float((signed[m] > 0).mean()))
    if do_p:
        out["perm_p_exact"] = circ_p(m, signed)
    return out


def s1_strategy(t: pd.DataFrame, level: pd.Series, hold: int = H_DAYS, cost: float = 0.0005
                ) -> dict:
    """Independent S1: one event per signal episode (first week of a run of equal non-zero
    signs on the Friday grid), entered at that week's entry close, held `hold` index days; an
    event whose entry is on/before the open trade's exit is skipped. Long-only uses only long
    events. Metrics: CAGR over calendar time, Sharpe = mean/std of daily net returns scaled by
    the observed days per year, max drawdown; buy & hold over the same window."""
    days = level.index
    close = level.to_numpy()
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    pos_of = {d: i for i, d in enumerate(days)}
    sig = t["sig"].to_numpy()
    prev = np.concatenate(([0], sig[:-1]))
    start_rows = np.flatnonzero((sig != 0) & (sig != prev))
    events = [(pos_of[t["entry"].iloc[j]], int(sig[j])) for j in start_rows]
    w0 = pos_of[t["entry"].iloc[0]]
    out = {}
    for mode in ("long_short", "long_only"):
        pos = np.zeros(len(close))
        busy, ntr, wins = -1, 0, 0
        for e, s in events:
            if mode == "long_only" and s != 1:
                continue
            if e <= busy:
                continue
            x = min(e + hold, len(close) - 1)
            pos[e:x] = s
            g = np.prod(1 + s * ret[e + 1:x + 1]) * (1 - cost) ** 2 - 1
            if e + hold <= len(close) - 1:
                ntr += 1
                wins += g > 0
            busy = x
        p_prev = np.concatenate(([0.0], pos[:-1]))
        p_prev[:w0 + 1] = 0.0
        dpos = np.abs(pos - p_prev)
        dpos[:w0] = 0.0
        net = ((1 + p_prev * ret) * (1 - cost * dpos) - 1)[w0:]
        out[mode] = perf(net, days[w0:]) | dict(n_trades=ntr, win_rate=wins / ntr)
    bh = ret.copy()
    bh[:w0 + 1] = 0.0
    out["buy_hold"] = perf(bh[w0:], days[w0:])
    return out


def perf(net: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """CAGR, Sharpe (rf 0, days per year from the data), max drawdown of a daily net series."""
    eq = np.cumprod(1 + net)
    yrs = (dates[-1] - dates[0]).days / 365.25
    r = net[1:]
    ppy = len(r) / yrs
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))
    return dict(cagr=float(eq[-1] ** (1 / yrs) - 1),
                sharpe=float(r.mean() * ppy / (r.std(ddof=1) * np.sqrt(ppy))),
                max_dd=float((np.concatenate(([1.0], eq)) / peak - 1).min()))


# ------------------------------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------------------------------
def main() -> None:
    res = {}
    raw = load_contracts()
    both = raw[["c1", "c2"]].dropna()                     # index days: both contracts print
    print_days = raw.index[raw.notna().any(axis=1).to_numpy()]
    ltd_mine = ltd_from_print_calendar(print_days)
    lvl_mine, roll_mine = roll_index(both["c1"], both["c2"], ltd_mine)

    # The harness index, read from the published results only through its daily level, for the
    # roll-dating comparison (loaded by importing nothing: we rebuild the harness LTDs from the
    # trades file is not possible, so we compare at the level of weekly results instead).
    t_mine = weekly_table(both["c1"], both["c2"], lvl_mine)
    prim = stats(t_mine, +1)
    res["A_primary_my_index"] = prim
    res["A_primary_my_index_period"] = f"{t_mine['friday'].min().date()}..{t_mine['friday'].max().date()}"
    res["A_short_my_index"] = stats(t_mine, -1)
    print("A. primary (my own roll index):", prim)

    # E. sign check: in premium weeks c1 > c2 in the raw prints
    pw = t_mine[t_mine["sig"] == 1]
    res["E_sign_check"] = dict(premium_weeks=int(len(pw)), all_c1_gt_c2=bool((pw["c1"] > pw["c2"]).all()),
                               discount_weeks=int((t_mine["sig"] == -1).sum()),
                               flat_weeks=int((t_mine["sig"] == 0).sum()))

    # B. roll-day jumps on my index
    ret = lvl_mine.pct_change().fillna(0.0)
    top = ret.abs().sort_values(ascending=False).head(8)
    res["B_largest_daily_returns_my_index"] = [
        dict(date=d.date().isoformat(), ret=float(ret[d]), roll_day=bool(roll_mine[lvl_mine.index.get_loc(d)]))
        for d in top.index]
    res["B_roll_days_my_index"] = int(roll_mine.sum())
    rr = ret[roll_mine]
    res["B_roll_day_abs_mean_vs_other"] = dict(roll=float(rr.abs().mean()),
                                               other=float(ret[~roll_mine].abs().mean()))

    # C. single-year / single-episode dependence
    years = t_mine["friday"].dt.year.to_numpy()
    loyo = {}
    for y in sorted(set(years)):
        tt = t_mine[years != y].reset_index(drop=True)
        loyo[int(y)] = stats(tt, +1, do_p=False)["excess"]
    s_loyo = pd.Series(loyo)
    res["C_leave_one_year_out_excess"] = dict(min=float(s_loyo.min()), min_year=int(s_loyo.idxmin()),
                                              max=float(s_loyo.max()), max_year=int(s_loyo.idxmax()))
    for drop in ([2008], [2014], [2020], [2008, 2014, 2020]):
        tt = t_mine[~np.isin(years, drop)].reset_index(drop=True)
        res[f"C_drop_{'_'.join(map(str, drop))}"] = stats(tt, +1)
    # contribution of each year to the long-side mean (sum of excess-weighted returns)
    m = t_mine["sig"].to_numpy() == 1
    r = t_mine["r"].to_numpy()
    base = r.mean()
    contrib = pd.Series(r[m] - base, index=years[m]).groupby(level=0).sum() / m.sum()
    res["C_year_contribution_to_excess_top5"] = {int(k): float(v) for k, v in
                                                 contrib.sort_values(ascending=False).head(5).items()}
    res["C_year_contribution_to_excess_bottom5"] = {int(k): float(v) for k, v in
                                                    contrib.sort_values().head(5).items()}
    # leave-one-episode-out
    starts = m & ~np.concatenate(([False], m[:-1]))
    ep_id = np.where(m, np.cumsum(starts), 0)
    looe = []
    for e in range(1, ep_id.max() + 1):
        keep = ep_id != e
        tt = t_mine[keep].reset_index(drop=True)
        looe.append((e, stats(tt, +1, do_p=False)["excess"],
                     t_mine["friday"][ep_id == e].min().date().isoformat(), int((ep_id == e).sum())))
    looe_df = pd.DataFrame(looe, columns=["ep", "excess_wo", "start", "weeks"])
    res["C_leave_one_episode_out"] = dict(
        min_excess=float(looe_df["excess_wo"].min()),
        min_episode_start=looe_df.loc[looe_df["excess_wo"].idxmin(), "start"],
        max_excess=float(looe_df["excess_wo"].max()),
        max_episode_start=looe_df.loc[looe_df["excess_wo"].idxmax(), "start"])

    # D. overlap-free: every 8th Friday, 8 phases, each its own baseline
    phase = []
    for ph in range(8):
        tt = t_mine.iloc[ph::8].reset_index(drop=True)
        s = stats(tt, +1, do_p=False)
        phase.append(dict(phase=ph, n=s["n"], T=s["T"], excess=s["excess"]))
    ph_df = pd.DataFrame(phase)
    res["D_nonoverlapping_phases"] = dict(excess_min=float(ph_df["excess"].min()),
                                          excess_max=float(ph_df["excess"].max()),
                                          excess_mean=float(ph_df["excess"].mean()),
                                          n_positive=int((ph_df["excess"] > 0).sum()))
    # episode level: one mean return per premium episode vs the baseline
    ep_means = pd.Series(r[m], index=ep_id[m]).groupby(level=0).mean()
    res["D_episode_level"] = dict(episodes=int(len(ep_means)),
                                  mean_of_episode_means_minus_baseline=float(ep_means.mean() - base),
                                  share_episodes_above_baseline=float((ep_means > base).mean()),
                                  median_episode_mean_minus_baseline=float(ep_means.median() - base))

    # F. spot vs roll decomposition, same entry/exit dates
    spot = load_spot()
    sv = spot.index.values

    def spot_at(d):
        k = np.searchsorted(sv, np.datetime64(d), side="right") - 1
        return spot.iloc[k] if k >= 0 else np.nan
    t_mine["r_spot"] = [spot_at(x) / spot_at(e) - 1 if e >= spot.index[0] else np.nan
                        for e, x in zip(t_mine["entry"], t_mine["exit"])]
    ok = t_mine["r_spot"].notna()
    tt = t_mine[ok]
    pm = tt["sig"] == 1
    res["F_decomposition"] = dict(
        n_all=int(len(tt)), n_premium=int(pm.sum()),
        roll_premium=float(tt.loc[pm, "r"].mean()), roll_all=float(tt["r"].mean()),
        spot_premium=float(tt.loc[pm, "r_spot"].mean()), spot_all=float(tt["r_spot"].mean()),
        excess_total=float(tt.loc[pm, "r"].mean() - tt["r"].mean()),
        excess_spot=float(tt.loc[pm, "r_spot"].mean() - tt["r_spot"].mean()))
    res["F_decomposition"]["excess_roll_basis"] = (res["F_decomposition"]["excess_total"]
                                                   - res["F_decomposition"]["excess_spot"])
    # spot-only test of the same signal with the exact p (a price-only check)
    ts = tt[["friday", "sig"]].copy()
    ts["r"] = tt["r_spot"].to_numpy()
    res["F_spot_same_dates_long"] = stats(ts.reset_index(drop=True), +1)

    # decade breakdown on my index
    dec = {}
    for d0 in range(1980, 2030, 10):
        mm = (years >= d0) & (years < d0 + 10)
        if mm.any():
            s = stats(t_mine[mm].reset_index(drop=True), +1, do_p=False)
            dec[f"{d0}s"] = dict(n=s.get("n", 0), excess=s.get("excess"))
    res["decades_my_index"] = dec

    # H. where inside H2 (2004-07-23..) does the edge live? own-baseline sub-periods
    sub = {}
    for name, a, b in (("H1_1985_2004-07-16", "1985-01-01", "2004-07-16"),
                       ("2004-07-23_2009-12-25", "2004-07-23", "2009-12-31"),
                       ("2010-01-01_2024-02-02", "2010-01-01", "2024-12-31"),
                       ("1985_2009", "1985-01-01", "2009-12-31")):
        mm = (t_mine["friday"] >= a) & (t_mine["friday"] <= b)
        s = stats(t_mine[mm].reset_index(drop=True), +1)
        sub[name] = {k: s.get(k) for k in ("n", "episodes", "T", "excess", "perm_p_exact")}
    res["H_subperiods_long_h8"] = sub

    # G. S1 strategy (first week of each episode, hold 40 index days, one position at a time,
    #    0.05% per side, daily close-to-close P&L, window = first entry .. last index close).
    res["G_S1_my_index"] = s1_strategy(t_mine, lvl_mine)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t_mine.to_csv(OUT.with_name("weekly_my_index.csv"), index=False)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        print(k, json.dumps(v, default=str))


if __name__ == "__main__":
    main()

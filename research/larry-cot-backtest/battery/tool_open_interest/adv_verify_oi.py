#!/usr/bin/env python3
"""
adv_verify_oi -- ADVERSARIAL, independent re-derivation of the open-interest PRIMARY result.

It does NOT import fund_harness, larry_cot, strat_lib or oi_lib. Everything is rebuilt from the
raw files with pandas/numpy only:

  * F+O total OI:   S/data/cot_weekly.csv  (market == "crude", column "oi")
  * crude_roll:     my own roll-adjusted front-month index from the EIA RCLC1/RCLC2 prints
                    (S/ext/accomplish999_crude-oil/data/series/RCLC{1,2,3,4}.json, json.load only).
                    Contract-1 last trading days (LTD) come from the NYMEX rule applied to the
                    data's own trading calendar (days with a non-stale print); hold contract 1
                    through its LTD, and on the next day use c1(t)/c2(t-1) - 1.
  * timing (v1 COT rule): entry = first index day on/after the Tuesday of the week after the
    report date (and at most 7 days late); exit = 40 index days later (h = 8 weeks).

Then it re-computes the PRIMARY cell (pre-registered rule oi4w_p10_range, LONG, h=8): n, episodes,
mean, baseline, excess, hit, exhaustive circular-shift p with and without the +1 correction, and
a battery of adversarial diagnostics (per-episode table, non-overlapping first-week-of-episode
version, leave-one-year-out, 2008/2014/2020 dependence, split halves, the 2020-04-20 negative
print, literal 30% coverage facts, the post-hoc gold cell).

Run:  python3 -I adv_verify_oi.py      (writes ROOT/work2/tool_open_interest/adv_verify_oi.json)
      python3 -I adv_verify_oi.py --harness-ltd
            Same code, but the roll dates are the harness's contract-1 last trading days (the
            only harness import). Use it for an exact comparison with results.json. My own NYMEX
            rule on the EIA print calendar misdates 23 of 470 months, most before 2007. It
            treats days EIA did not print as holidays, but NYMEX traded on some of them, e.g.
            the Fridays after Thanksgiving 2016-11-25 and 2017-11-24. That sets the Jan-2017
            and Jan-2018 expiries one business day early: 2016-11-18 instead of 2016-11-21,
            and 2017-11-17 instead of 2017-11-20. Writes adv_verify_oi_harnessltd.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PKG = Path(__file__).resolve().parents[2]                # package dir
S = Path(os.environ.get("LARRY_COT_ROOT") or PKG)        # ROOT: data/ ext/ work2/
OUT = S / "work2" / "tool_open_interest" / "adv_verify_oi.json"
SERIES_DIR = S / "ext" / "accomplish999_crude-oil" / "data" / "series"

SHUTDOWNS = [("2013-10-01", "2013-10-31"), ("2018-12-18", "2019-02-26"),
             ("2025-09-30", "2026-01-20")]           # v1 SPEC: excluded report windows
MIN_SHIFT = 26                                       # circular-shift offsets in [26, T-26]
H_DAYS = 40                                          # h = 8 weeks = 40 trading days
MAX_ENTRY_LAG = 7                                    # entry at most 7 days after the target


# ------------------------------------------------------------------------------------------
# 1. Raw data
# ------------------------------------------------------------------------------------------
def load_eia(series_id: str) -> pd.Series:
    """EIA daily series from the third-party JSON (parsed with json.load only)."""
    doc = json.load(open(SERIES_DIR / f"{series_id}.json", encoding="utf-8"))
    ob = pd.DataFrame(doc["observations"])
    s = pd.Series(pd.to_numeric(ob["value"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(ob["date"]), name=series_id)
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_cot(market: str) -> pd.DataFrame:
    cot = pd.read_csv(S / "data" / "cot_weekly.csv", dtype={"code": str})
    g = cot[cot["market"] == market].copy()
    g["report_date"] = pd.to_datetime(g["report_date"])
    g = g.sort_values("report_date").reset_index(drop=True)
    assert not g["report_date"].duplicated().any(), "duplicate report dates"
    g["oi"] = g["oi"].astype(float)
    sd = np.zeros(len(g), bool)
    for a, b in SHUTDOWNS:
        sd |= (g["report_date"] >= a).to_numpy() & (g["report_date"] <= b).to_numpy()
    g["shutdown"] = sd
    return g[["report_date", "oi", "shutdown"]]


# ------------------------------------------------------------------------------------------
# 2. My own roll-adjusted crude index (roll_lead = 0)
# ------------------------------------------------------------------------------------------
def my_ltds(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """NYMEX CL rule on the given business calendar: 3 business days before the 25th of the
    month; if the 25th is not a business day, 3 business days before the last business day
    preceding the 25th."""
    out = []
    for p in pd.period_range(cal[0] + pd.Timedelta(days=40), cal[-1], freq="M"):
        d25 = pd.Timestamp(p.year, p.month, 25)
        if d25 > cal[-1]:
            continue
        before = cal[cal < d25]
        out.append(before[-3] if d25 in cal else before[-4])
    return pd.DatetimeIndex(out)


def build_roll_index(ltd: pd.DatetimeIndex | None = None) -> tuple:
    """Index level on days where c1 > 0 and c2 > 0 (and the row is not a stale repeat of the
    previous row). Between consecutive index days t' < t: if an LTD L with t' <= L < t was
    crossed, the held contract became contract 2 at t', so ret = c1(t)/c2(t') - 1; otherwise
    ret = c1(t)/c1(t') - 1."""
    df = pd.DataFrame({f"c{j}": load_eia(f"RCLC{j}") for j in (1, 2, 3, 4)})
    df = df[df.index >= "1985-01-02"]
    prev = df.shift(1)
    stale = (((df == prev) | df.isna()).all(axis=1) & df.notna().any(axis=1)).to_numpy().copy()
    stale[0] = False
    cal = df.index[~stale & df["c1"].notna().to_numpy()]          # trading calendar from data
    ltd_used = my_ltds(cal) if ltd is None else ltd
    ok = (~stale) & (df["c1"] > 0).to_numpy() & (df["c2"] > 0).to_numpy()
    v = df.loc[ok, ["c1", "c2"]]
    c1, c2 = v["c1"].to_numpy(), v["c2"].to_numpy()
    days = v.index
    k = np.searchsorted(ltd_used.values, days.values, side="left")   # #LTDs strictly before
    crossed = np.zeros(len(days), bool)
    crossed[1:] = (k[1:] - k[:-1]) >= 1
    assert ((k[1:] - k[:-1]) <= 1).all()
    ret = np.zeros(len(days))
    ret[1:] = np.where(crossed[1:], c1[1:] / c2[:-1], c1[1:] / c1[:-1]) - 1.0
    lvl = pd.Series(100 * np.cumprod(1 + ret), index=days, name="my_crude_roll")
    info = dict(n_days=int(len(days)), first=days[0].date().isoformat(),
                last=days[-1].date().isoformat(), n_roll_days=int(crossed.sum()),
                stale_rows=[d.date().isoformat() for d in df.index[stale]],
                dropped_nonpositive_c1=[d.date().isoformat()
                                        for d in df.index[(df["c1"] <= 0).to_numpy()]],
                max_abs_daily_ret=float(np.abs(ret).max()),
                max_abs_daily_ret_date=days[int(np.abs(ret).argmax())].date().isoformat())
    return lvl, info, ltd_used, cal


# ------------------------------------------------------------------------------------------
# 3. Features of the pre-registered primary rule
# ------------------------------------------------------------------------------------------
def asof(series: pd.Series, dates: pd.Series, max_stale: int = 4) -> np.ndarray:
    """Last close on or before each date, NaN if > max_stale calendar days old."""
    idx = series.index.values
    d = pd.DatetimeIndex(dates).values
    pos = np.searchsorted(idx, d, side="right") - 1
    out = np.full(len(d), np.nan)
    for i, p in enumerate(pos):
        if p >= 0 and (d[i] - idx[p]) / np.timedelta64(1, "D") <= max_stale:
            out[i] = series.iloc[p]
    return out


def change(x: np.ndarray, dates: pd.Series, w: int, max_days: int) -> np.ndarray:
    """x_t / x_{t-w} - 1 over report rows, NaN if report t-w is > max_days before t."""
    d = pd.DatetimeIndex(dates)
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        if (d[t] - d[t - w]).days <= max_days:
            out[t] = x[t] / x[t - w] - 1.0
    return out


def pct_window(x: np.ndarray, window: int = 156, min_n: int = 52) -> np.ndarray:
    """100 * share of the defined values among the last `window` rows (incl. t) that are
    <= x_t; NaN if x_t is NaN or fewer than min_n defined values. Uses data <= t only."""
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        if np.isnan(x[t]):
            continue
        past = x[max(0, t - window + 1):t + 1]
        past = past[~np.isnan(past)]
        if len(past) >= min_n:
            out[t] = 100.0 * np.mean(past <= x[t])
    return out


def primary_signal(rep: pd.DataFrame, price: pd.Series, w: int = 4, p: float = 10) -> pd.DataFrame:
    f = rep.copy()
    f["px"] = asof(price, f["report_date"])
    f["dOI"] = change(f["oi"].to_numpy(), f["report_date"], w, 7 * w + 7)
    f["dP"] = change(f["px"].to_numpy(), f["report_date"], w, 7 * w + 7)
    f["pOI"] = pct_window(f["dOI"].to_numpy())
    f["pP"] = pct_window(np.abs(f["dP"].to_numpy()))
    valid = f["pOI"].notna() & f["pP"].notna()
    rng = f["pP"] <= 50
    sig = np.where(valid & rng & (f["pOI"] <= p), 1, np.where(valid & rng & (f["pOI"] >= 100 - p), -1, 0))
    f["valid"] = valid.to_numpy()
    f["signal"] = sig
    return f


# ------------------------------------------------------------------------------------------
# 4. Timing, forward returns, cell statistics
# ------------------------------------------------------------------------------------------
def forward(rep_dates: pd.Series, lvl: pd.Series, h_days: int = H_DAYS) -> pd.DataFrame:
    d = pd.DatetimeIndex(rep_dates)
    target = d - pd.to_timedelta(d.weekday, unit="D") + pd.Timedelta(days=8)   # next Tuesday
    idx = lvl.index.values
    pos = np.searchsorted(idx, target.values, side="left")
    ent = np.full(len(d), -1)
    for i, p in enumerate(pos):
        if p < len(idx) and (idx[p] - target.values[i]) / np.timedelta64(1, "D") <= MAX_ENTRY_LAG:
            ent[i] = p
    r = np.full(len(d), np.nan)
    ex_date = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    en_date = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    L = lvl.to_numpy()
    for i, p in enumerate(ent):
        if p >= 0:
            en_date[i] = idx[p]
            if p + h_days < len(L):
                r[i] = L[p + h_days] / L[p] - 1.0
                ex_date[i] = idx[p + h_days]
    return pd.DataFrame({"entry_date": en_date, "exit_date": ex_date, "r": r})


def cell(f: pd.DataFrame, r: np.ndarray, side: int = 1, extra_mask=None) -> dict:
    """Primary statistics on eligible weeks (valid, not excluded, forward return defined)."""
    valid = f["valid"].to_numpy(bool)
    if extra_mask is not None:
        valid = valid & extra_mask
    elig = valid & ~f["shutdown"].to_numpy(bool) & ~np.isnan(r)
    setup = valid & (f["signal"].to_numpy() == side)
    re = r[elig]
    T = int(elig.sum())
    base = float(re.mean())
    sig = setup[elig]
    signed = side * re
    n = int(sig.sum())
    mean = float(signed[sig].mean())
    # episodes: runs of the setup on the full report sequence with >= 1 counted week
    run_id = np.cumsum(setup & ~np.concatenate(([False], setup[:-1])))
    run_id = np.where(setup, run_id, 0)
    episodes = len(np.unique(run_id[elig & setup]))
    # exhaustive circular shift; the offset set [26, T-26] is symmetric so direction is moot
    offs = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
    pos_ = np.flatnonzero(sig)
    shifted = signed[(pos_[None, :] + offs[:, None]) % T].mean(axis=1)
    cnt = int(np.sum(shifted >= mean))
    cnt_tol = int(np.sum(shifted >= mean - 1e-12))
    d = f["report_date"].to_numpy()[elig]
    return dict(n=n, episodes=int(episodes), T=T, mean=mean, baseline=base,
                excess=mean - side * base, hit=float((signed[sig] > 0).mean()),
                median=float(np.median(signed[sig])),
                perm_p_exact=(1 + cnt) / (1 + len(offs)), perm_p_exact_raw=cnt / len(offs),
                perm_p_exact_tol=(1 + cnt_tol) / (1 + len(offs)),
                period=f"{pd.Timestamp(d.min()).date()}..{pd.Timestamp(d.max()).date()}")


def entry_rows(rep_dates: pd.Series, lvl: pd.Series) -> np.ndarray:
    """Row of the entry close (v1 COT timing), -1 if none within MAX_ENTRY_LAG days."""
    d = pd.DatetimeIndex(rep_dates)
    target = d - pd.to_timedelta(d.weekday, unit="D") + pd.Timedelta(days=8)
    idx = lvl.index.values
    pos = np.searchsorted(idx, target.values, side="left")
    ok = pos < len(idx)
    lag = np.full(len(d), np.inf)
    lag[ok] = (idx[pos[ok]] - target.values[ok]) / np.timedelta64(1, "D")
    return np.where(ok & (lag <= MAX_ENTRY_LAG), pos, -1)


def s1_long_only(f: pd.DataFrame, lvl: pd.Series, sig: np.ndarray | None = None,
                 hold: int = 40, cost: float = 0.0005) -> dict:
    """S1 long-only: one event per long episode (its first tradable week: not excluded, entry
    inside the data); enter at that entry close, hold `hold` days, ignore events whose entry is
    on/before the current exit; daily net = (1 + pos[t-1] r[t]) (1 - cost |dpos|) - 1.
    Window = first valid week's entry close .. last close. Sharpe/CAGR as in SPEC2 rule 6."""
    sig = f["signal"].to_numpy() if sig is None else sig
    valid = f["valid"].to_numpy(bool)
    ep = entry_rows(f["report_date"], lvl)
    tradable = (ep >= 0) & ~f["shutdown"].to_numpy(bool)
    setup = valid & (sig == 1)
    starts = []
    i = 0
    while i < len(setup):
        if setup[i]:
            j = i
            while j + 1 < len(setup) and setup[j + 1]:
                j += 1
            members = [k for k in range(i, j + 1) if tradable[k]]
            if members:
                starts.append(int(ep[members[0]]))
            i = j + 1
        else:
            i += 1
    close = lvl.to_numpy()
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1
    first_valid = np.flatnonzero(valid & (ep >= 0))[0]
    start = int(ep[first_valid])
    pos = np.zeros(n)
    busy, trades = -1, []
    for e in sorted(starts):
        if e <= busy:
            continue
        x = min(e + hold, n - 1)
        pos[e:x] = 1.0                       # held at the close of e .. x-1, flat from x
        trades.append((e, x, x == e + hold))
        busy = x
    p_prev = np.concatenate(([0.0], pos[:-1]))
    p_prev[:start + 1] = 0.0
    dpos = np.abs(pos - p_prev)
    dpos[:start] = 0.0
    net = ((1 + p_prev * ret) * (1 - cost * dpos) - 1)[start:]
    days = lvl.index[start:]
    years = (days[-1] - days[0]).days / 365.25
    eq = np.cumprod(1 + net)
    r = net[1:]
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(len(r) / years))
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))
    dd = float((np.concatenate(([1.0], eq)) / peak - 1).min())
    tr_net = [(1 - cost) * np.prod(1 + ret[e + 1:x + 1]) * (1 - cost) - 1
              for e, x, closed in trades if closed]
    bh = close[start:] / close[start]
    bh_r = bh[1:] / bh[:-1] - 1
    bpk = np.maximum.accumulate(bh)
    return dict(start=str(days[0].date()), end=str(days[-1].date()), n_trades=len(tr_net),
                win_rate=float(np.mean(np.array(tr_net) > 0)) if tr_net else float("nan"),
                cagr=float(eq[-1] ** (1 / years) - 1), sharpe=sharpe, max_dd=dd,
                bh_cagr=float(bh[-1] ** (1 / years) - 1),
                bh_sharpe=float(np.mean(bh_r) / np.std(bh_r, ddof=1) * np.sqrt(len(bh_r) / years)),
                bh_max_dd=float((bh / bpk - 1).min()), _years=years)


def s1_null_exhaustive(f: pd.DataFrame, lvl: pd.Series, real_sharpe: float) -> dict:
    """Random-timing null for S1 long-only over EVERY circular shift k in [26, T-26] of the
    signal over the valid weeks (T = number of valid weeks); p with the +1 correction."""
    valid = np.flatnonzero(f["valid"].to_numpy(bool))
    T = len(valid)
    base = f["signal"].to_numpy().astype(float)
    sh = []
    for k in range(MIN_SHIFT, T - MIN_SHIFT + 1):
        s = base.copy()
        s[valid] = np.roll(base[valid], k)
        sh.append(s1_long_only(f, lvl, sig=s)["sharpe"])
    sh = np.array(sh)
    return dict(n_offsets=int(len(sh)), median=float(np.median(sh)),
                p_plus1=float((1 + np.sum(sh >= real_sharpe)) / (1 + len(sh))))


def episode_table(f: pd.DataFrame, r: np.ndarray, side: int = 1) -> tuple[pd.DataFrame, float]:
    """One row per setup episode with >= 1 eligible week: first report, weeks, first-week and
    mean forward return, and the episode's summed excess over the all-weeks baseline."""
    elig = f["valid"].to_numpy(bool) & ~f["shutdown"].to_numpy(bool) & ~np.isnan(r)
    setup = f["valid"].to_numpy(bool) & (f["signal"].to_numpy() == side)
    run_id = np.where(setup, np.cumsum(setup & ~np.concatenate(([False], setup[:-1]))), 0)
    base = float(r[elig].mean())
    rows = []
    for e in np.unique(run_id[setup & elig]):
        idx = np.flatnonzero((run_id == e) & elig)
        rows.append(dict(first_report=f["report_date"].iloc[idx[0]].date().isoformat(),
                         year=int(f["report_date"].iloc[idx[0]].year), weeks=int(len(idx)),
                         r_first=float(side * r[idx[0]]), r_mean=float(side * r[idx].mean()),
                         sum_excess=float((side * r[idx] - side * base).sum())))
    return pd.DataFrame(rows), base


# ------------------------------------------------------------------------------------------
def main() -> None:
    res: dict = {}
    rep = load_cot("crude")
    out_path = OUT
    if "--harness-ltd" in sys.argv:
        sys.path.insert(0, str(PKG / "battery" / "harness"))
        import fund_harness as fh  # noqa: E402  (ONLY for the contract-1 last trading days)
        lvl, info, ltd_mine, cal = build_roll_index(fh.build_crude_roll_index(0).ltd)
        res["ltd_source"] = "harness crude_last_trade_dates (roll_lead=0)"
        out_path = OUT.with_name("adv_verify_oi_harnessltd.json")
    else:
        lvl, info, ltd_mine, cal = build_roll_index()
        res["ltd_source"] = "own NYMEX rule on the EIA print calendar"
    res["my_roll_index"] = info
    f = primary_signal(rep, lvl)
    fw = forward(f["report_date"], lvl)
    r = fw["r"].to_numpy()
    prim = cell(f, r, +1)
    res["primary_long_h8"] = prim
    res["primary_short_h8"] = cell(f, r, -1)
    print("PRIMARY (independent):", prim)

    # ---- per-episode table (long) ----
    elig = f["valid"].to_numpy() & ~f["shutdown"].to_numpy() & ~np.isnan(r)
    setup = f["valid"].to_numpy() & (f["signal"].to_numpy() == 1)
    run_id = np.where(setup, np.cumsum(setup & ~np.concatenate(([False], setup[:-1]))), 0)
    base = prim["baseline"]
    eps = []
    for e in np.unique(run_id[setup & elig]):
        m = (run_id == e) & elig
        rows = np.flatnonzero(m)
        eps.append(dict(first_report=f["report_date"].iloc[rows[0]].date().isoformat(),
                        weeks=int(len(rows)),
                        entry=pd.Timestamp(fw["entry_date"].iloc[rows[0]]).date().isoformat(),
                        exit_first=pd.Timestamp(fw["exit_date"].iloc[rows[0]]).date().isoformat(),
                        r8_first=float(r[rows[0]]), r8_mean=float(r[rows].mean()),
                        sum_excess=float((r[rows] - base).sum())))
    ep = pd.DataFrame(eps)
    res["episodes_long"] = ep.to_dict("records")
    # non-overlapping version: one observation per episode (its first counted week)
    res["first_week_of_episode"] = dict(n=int(len(ep)), mean=float(ep["r8_first"].mean()),
                                        excess=float(ep["r8_first"].mean() - base),
                                        hit=float((ep["r8_first"] > 0).mean()))
    # concentration: share of the total excess contributed by the largest episode
    tot = float(ep["sum_excess"].sum())
    res["excess_concentration"] = dict(
        total_sum_excess=tot, n_weeks=prim["n"],
        top_episode=ep.loc[ep["sum_excess"].idxmax()].to_dict(),
        bottom_episode=ep.loc[ep["sum_excess"].idxmin()].to_dict())
    # leave-one-EPISODE-out: excess without each episode's weeks (baseline unchanged)
    n_w = prim["n"]
    loeo = {row["first_report"]: float((tot - row["sum_excess"]) / (n_w - row["weeks"]))
            for _, row in ep.iterrows()}
    res["leave_one_episode_out_excess"] = dict(
        min=min(loeo.values()), min_episode=min(loeo, key=loeo.get),
        max=max(loeo.values()), max_episode=max(loeo, key=loeo.get),
        n_negative=int(sum(v < 0 for v in loeo.values())), n_episodes=len(loeo))
    # leave-one-year-out on the signal weeks (baseline kept on all eligible weeks)
    years = f["report_date"].dt.year.to_numpy()
    loyo = {}
    for y in sorted(set(years[setup & elig])):
        keep = setup & elig & (years != y)
        loyo[int(y)] = dict(n=int(keep.sum()), excess=float(r[keep].mean() - base))
    res["leave_one_year_out_excess"] = loyo
    # drop the named crisis windows entirely (both signal and baseline)
    for tag, a, b in (("ex_2008", "2008-01-01", "2009-06-30"), ("ex_2014_16", "2014-06-01", "2016-03-31"),
                      ("ex_2020", "2020-01-01", "2020-12-31")):
        dd = f["report_date"]
        mask = ~((dd >= a) & (dd <= b)).to_numpy()
        c = cell(f, r, +1, extra_mask=mask)
        res[f"primary_long_h8_{tag}"] = {k: c[k] for k in ("n", "episodes", "T", "excess", "perm_p_exact")}

    # ---- split halves (midpoint of the eligible decision dates) ----
    d_el = f["report_date"][elig]
    mid = d_el.iloc[0] + (d_el.iloc[-1] - d_el.iloc[0]) / 2
    halves = {}
    for nm, m in (("H1", (f["report_date"] <= mid).to_numpy()), ("H2", (f["report_date"] > mid).to_numpy())):
        c = cell(f, r, +1, extra_mask=m)
        halves[nm] = {k: c[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess",
                                        "perm_p_exact", "period")}
    res["split_half_long_h8"] = dict(split=str(mid.date()), **halves)

    # ---- the 2020-04-20 negative print: which long weeks hold across it? ----
    a20 = np.datetime64("2020-04-20")
    across = [(f["report_date"].iloc[i].date().isoformat(), float(r[i]))
              for i in np.flatnonzero(setup & elig)
              if fw["entry_date"].iloc[i] < a20 <= fw["exit_date"].iloc[i]]
    res["long_weeks_holding_across_2020_04_20"] = across
    allw = [i for i in np.flatnonzero(elig)
            if fw["entry_date"].iloc[i] < a20 <= fw["exit_date"].iloc[i]]
    res["eligible_weeks_holding_across_2020_04_20"] = len(allw)

    # ---- literal 30% coverage facts (claim C7 / caveat 2) ----
    oi = f["oi"]
    for n_ in (26, 52):
        mx = oi.rolling(n_, min_periods=n_).max()
        ratio = (oi / mx)[mx.notna()]
        res[f"crude_oi_le_70pct_of_{n_}w_high_share"] = float((ratio <= 0.70).mean())
        res[f"crude_oi_min_ratio_to_{n_}w_high"] = float(ratio.min())
    g = load_cot("gold")["oi"]
    mx = g.rolling(52, min_periods=52).max()
    res["gold_oi_le_70pct_of_52w_high_share"] = float(((g / mx)[mx.notna()] <= 0.70).mean())

    # ---- post-hoc gold cell (claim C11), on the v1 gold series ----
    px = pd.read_csv(S / "data" / "prices_daily.csv")
    gs = px[px["market"] == "gold"].copy()
    gs["date"] = pd.to_datetime(gs["date"])
    gs = gs[gs["close"] > 0].set_index("date")["close"].sort_index()
    gs = gs[~gs.index.duplicated(keep="last")]
    grep = load_cot("gold")
    gf = primary_signal(grep, gs)
    gr = forward(gf["report_date"], gs)["r"].to_numpy()
    res["gold_primary_rule_long_h8"] = cell(gf, gr, +1)
    print("GOLD (independent):", res["gold_primary_rule_long_h8"])
    gep, gbase = episode_table(gf, gr, +1)
    res["gold_episodes_long"] = gep.to_dict("records")
    res["gold_excess_by_year_of_episode_start"] = (
        gep.groupby("year")["sum_excess"].sum().round(6).to_dict())
    g_el = gf["valid"].to_numpy(bool) & ~gf["shutdown"].to_numpy(bool) & ~np.isnan(gr)
    g_set = gf["valid"].to_numpy(bool) & (gf["signal"].to_numpy() == 1) & g_el
    gy = gf["report_date"].dt.year.to_numpy()
    res["gold_leave_one_year_out_excess"] = {
        int(y): float(gr[g_set & (gy != y)].mean() - gbase) for y in sorted(set(gy[g_set]))}
    res["gold_first_week_of_episode"] = dict(n=int(len(gep)),
                                             excess=float(gep["r_first"].mean() - gbase),
                                             hit=float((gep["r_first"] > 0).mean()))
    # gold without its last 3 years (2024-2026 bull market) and without 2008-2009
    for tag, a, b in (("ex_2024_26", "2024-01-01", "2026-12-31"), ("ex_2008_09", "2008-01-01", "2009-12-31")):
        m = ~((gf["report_date"] >= a) & (gf["report_date"] <= b)).to_numpy()
        c = cell(gf, gr, +1, extra_mask=m)
        res[f"gold_primary_rule_long_h8_{tag}"] = {k: c[k] for k in ("n", "episodes", "T", "excess", "perm_p_exact")}

    # ---- S1 long-only strategy on my index, with an exhaustive random-timing null ----
    s1 = s1_long_only(f, lvl)
    s1["null"] = s1_null_exhaustive(f, lvl, s1["sharpe"])
    res["crude_S1_long_only"] = s1
    print("crude S1 long-only (independent):", s1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=str))
    print("episodes:\n", ep.to_string())
    for k in ("first_week_of_episode", "split_half_long_h8", "leave_one_year_out_excess",
              "leave_one_episode_out_excess", "excess_concentration",
              "primary_long_h8_ex_2008", "primary_long_h8_ex_2014_16", "primary_long_h8_ex_2020",
              "long_weeks_holding_across_2020_04_20", "eligible_weeks_holding_across_2020_04_20",
              "my_roll_index", "gold_excess_by_year_of_episode_start",
              "gold_leave_one_year_out_excess", "gold_first_week_of_episode",
              "gold_primary_rule_long_h8_ex_2024_26", "gold_primary_rule_long_h8_ex_2008_09",
              "crude_oi_le_70pct_of_26w_high_share", "crude_oi_min_ratio_to_26w_high",
              "crude_oi_le_70pct_of_52w_high_share", "gold_oi_le_70pct_of_52w_high_share"):
        print(k, res[k])


if __name__ == "__main__":
    main()

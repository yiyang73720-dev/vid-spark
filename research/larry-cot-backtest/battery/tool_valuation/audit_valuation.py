#!/usr/bin/env python3
"""
audit_valuation -- adversarial, independent re-derivation of the valuation tool's PRIMARY cell
(crude vs gold, EMA2-EMA22, N=156, LONG when VI <= 20, h=8 weeks, traded on a roll-adjusted
WTI front-month index), plus the episode / roll / overlap checks of the audit.

Independence: nothing here imports fund_harness, larry_cot or valuation_lib.
  * prices are parsed directly from S/data/prices_daily.csv, S/data/wti-daily.csv and the EIA
    RCLC1/RCLC2 JSON files (json.load only; third-party files are never executed);
  * the roll-adjusted crude index is REBUILT here from RCLC1/RCLC2 with the NYMEX CL expiry
    rule applied to the data's own trading calendar (the harness uses a reconciled calendar
    plus a back-contract score; agreement between the two is reported);
  * VI, entries, exits, the baseline and the exhaustive circular-shift p are re-implemented.

Outputs: ROOT/work2/tool_valuation/audit_output.json (every number printed below).

Run:  python3 -I audit_valuation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/tool_valuation/../..)
S = Path(os.environ.get("LARRY_COT_ROOT") or PKG)        # ROOT: data/ ext/ work2/
HERE = S / "work2" / "tool_valuation"
EXT = S / "ext" / "accomplish999_crude-oil" / "data" / "series"

H_DAYS = 40                 # h = 8 weeks = 40 trading rows after entry
MIN_SHIFT = 26              # circular-shift offsets in [26, T-26]
STALE_DAYS = 4              # an as-of close older than 4 calendar days -> undefined week
MAX_ENTRY_LAG = 7           # no entry if the first close after Friday is > 7 days later
OUT: dict = {}


def say(key: str, val) -> None:
    """Print and store one audit number."""
    OUT[key] = val
    print(f"{key}: {val}")


# =============================================================================================
# 1. Raw inputs (own parsers)
# =============================================================================================
def v1_close(market: str) -> pd.Series:
    """Daily close of one market from prices_daily.csv, closes <= 0 dropped (v1 data rule)."""
    px = pd.read_csv(S / "data" / "prices_daily.csv", float_precision="round_trip")
    g = px[px["market"] == market]
    s = pd.Series(g["close"].to_numpy(float), index=pd.to_datetime(g["date"])).sort_index()
    return s[s > 0]


def wti_spot() -> pd.Series:
    w = pd.read_csv(S / "data" / "wti-daily.csv", float_precision="round_trip")
    s = pd.Series(pd.to_numeric(w["Price"], errors="coerce").to_numpy(float),
                  index=pd.to_datetime(w["Date"])).sort_index().dropna()
    return s[s > 0]


def eia_json(series_id: str) -> pd.Series:
    """EIA series from the third-party archive (json.load only)."""
    with open(EXT / f"{series_id}.json", encoding="utf-8") as f:
        obs = json.load(f)["observations"]
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy(float)
    return pd.Series(vals, index=pd.to_datetime([o["date"] for o in obs])).sort_index().dropna()


# =============================================================================================
# 2. Own roll-adjusted WTI index (hold contract 1 through its last trading day)
# =============================================================================================
def own_roll_index() -> tuple[pd.Series, pd.DatetimeIndex]:
    """Index days = days where both contract 1 and 2 print > 0 and the row is not an exact
    repeat of the previous row (holiday placeholder). Last trading day (LTD) of contract 1 in
    month m = 3 trading days before the 25th if the 25th is a trading day, else 3 trading days
    before the last trading day preceding the 25th; the trading calendar is simply the set of
    index days. Return on the first index day after an LTD = CL1(t)/CL2(t-1) - 1 (the old
    contract 2 is the new contract 1); otherwise CL1(t)/CL1(t-1) - 1."""
    c = pd.DataFrame({"c1": eia_json("RCLC1"), "c2": eia_json("RCLC2")}).dropna()
    c = c[(c["c1"] > 0) & (c["c2"] > 0)]
    rep = (c == c.shift(1)).all(axis=1)
    c = c[~rep.to_numpy()]
    c = c[c.index >= "1985-01-02"]
    days = c.index
    ltds = []
    for p in pd.period_range(days[0], days[-1], freq="M"):
        d25 = pd.Timestamp(p.year, p.month, 25)
        before = days[days < d25]
        if d25 > days[-1] or len(before) < 5:
            continue
        ltds.append(before[-3] if d25 in days else before[-4])
    ltd = pd.DatetimeIndex(ltds)
    c1, c2 = c["c1"].to_numpy(), c["c2"].to_numpy()
    # an expiry lies between consecutive index days t-1 and t when t-1 <= LTD < t
    k_prev = np.searchsorted(ltd.values, days.values[:-1], side="left")
    k_now = np.searchsorted(ltd.values, days.values[1:], side="left")
    roll = (k_now - k_prev) == 1
    ret = np.where(roll, c1[1:] / c2[:-1] - 1.0, c1[1:] / c1[:-1] - 1.0)
    level = pd.Series(100.0 * np.concatenate(([1.0], np.cumprod(1.0 + ret))), index=days)
    return level, ltd


# =============================================================================================
# 3. Weekly valuation index
# =============================================================================================
FRI = pd.date_range("1985-01-04", "2026-10-02", freq="W-FRI")


def weekly_asof(s: pd.Series) -> np.ndarray:
    """Last close on or before each Friday; NaN if none or older than STALE_DAYS."""
    pos = np.searchsorted(s.index.values, FRI.values, side="right") - 1
    out = np.full(len(FRI), np.nan)
    ok = pos >= 0
    age = (FRI.values[ok] - s.index.values[pos[ok]]) / np.timedelta64(1, "D")
    vals = s.to_numpy()[pos[ok]]
    out[np.flatnonzero(ok)[age <= STALE_DAYS]] = vals[age <= STALE_DAYS]
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a, y = 2.0 / (n + 1.0), np.empty(len(x))
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def vi_index(m: np.ndarray, c: np.ndarray, s: int | None = 2, l: int = 22,
             N: int = 156) -> np.ndarray:
    """VI on the Friday grid: undefined weeks are skipped by the EMAs and the window.
    s=None is the 'none' detrending variant (min/max range of the raw ratio)."""
    R = 100.0 * m / c
    ok = np.isfinite(R)
    r = R[ok]
    if s is None:
        pal = r.copy()
    else:
        pal = ema(r, s) - ema(r, l)
        pal[: l - 1] = np.nan
    vi = np.full(len(r), np.nan)
    for i in range(len(r)):
        w = pal[max(0, i - N + 1): i + 1]
        if i - N + 1 < 0 or np.isnan(w).any():
            continue
        lo, hi = w.min(), w.max()
        if hi > lo:
            vi[i] = 100.0 * (pal[i] - lo) / (hi - lo)
    out = np.full(len(m), np.nan)
    out[ok] = vi
    return out


# =============================================================================================
# 4. Forward returns, cell statistics and the exhaustive circular-shift p
# =============================================================================================
def fwd(px: pd.Series, h_days: int = H_DAYS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entry = first close strictly after the Friday (within MAX_ENTRY_LAG days); exit =
    h_days rows later. Returns (r, entry_row, exit_row) aligned on FRI."""
    idx = px.index.values
    tgt = (FRI + pd.Timedelta(days=1)).values
    pos = np.searchsorted(idx, tgt, side="left")
    r = np.full(len(FRI), np.nan)
    ent = np.full(len(FRI), -1)
    for i, p in enumerate(pos):
        if p >= len(idx) or (idx[p] - tgt[i]) / np.timedelta64(1, "D") > MAX_ENTRY_LAG:
            continue
        ent[i] = p
        if p + h_days < len(idx):
            r[i] = px.iloc[p + h_days] / px.iloc[p] - 1.0
    return r, ent, np.where((ent >= 0) & (ent + h_days < len(idx)), ent + h_days, -1)


def cell(vi: np.ndarray, r: np.ndarray, mask_extra: np.ndarray | None = None,
         side: int = 1, th: tuple = (20.0, 80.0)) -> dict:
    """Long (side=+1, VI<=buy) or short (side=-1, VI>=sell) cell on eligible weeks."""
    elig = np.isfinite(vi) & np.isfinite(r)
    if mask_extra is not None:
        elig &= mask_extra
    setup = (vi <= th[0]) if side == 1 else (vi >= th[1])
    sig = setup[elig]
    signed = side * r[elig]
    T, n = int(elig.sum()), int(sig.sum())
    if n == 0:
        return dict(n=0, T=T)
    mean, base = float(signed[sig].mean()), float(r[elig].mean())
    pos = np.flatnonzero(sig)
    offs = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
    shifted = signed[(pos[None, :] + offs[:, None]) % T].mean(axis=1)
    cnt = int((shifted >= mean).sum())
    rng = np.random.default_rng(11)
    mc = rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=2000, endpoint=True)
    mc_cnt = int((signed[(pos[None, :] + mc[:, None]) % T].mean(axis=1) >= mean).sum())
    d = FRI[elig]
    return dict(n=n, T=T, mean=mean, baseline=base, excess=mean - side * base,
                hit=float((signed[sig] > 0).mean()), median=float(np.median(signed[sig])),
                perm_p_exact=(1 + cnt) / (1 + len(offs)), perm_p_mc=(1 + mc_cnt) / 2001,
                period=f"{d[0].date()}..{d[-1].date()}")


def episodes(setup: np.ndarray, vi_defined: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive setup weeks on the Friday grid (an undefined week breaks a
    run). Returns (first_index, last_index) pairs."""
    runs, start = [], None
    for i in range(len(setup)):
        on = bool(setup[i]) and bool(vi_defined[i])
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(setup) - 1))
    return runs


# =============================================================================================
# 5. Run
# =============================================================================================
def main() -> None:
    crude_cl, gold = v1_close("crude"), v1_close("gold")
    roll, ltd = own_roll_index()
    spot = wti_spot()
    say("own_roll_index", f"{roll.index[0].date()}..{roll.index[-1].date()} "
                          f"({len(roll)} days, {len(ltd)} LTDs)")

    # (the own roll index is compared with the harness crude_roll at the end of main(); the
    #  harness is loaded there only as a data source)
    vi = vi_index(weekly_asof(crude_cl), weekly_asof(gold))
    okv = np.isfinite(vi)
    say("vi_defined", f"{FRI[okv][0].date()}..{FRI[okv][-1].date()} ({int(okv.sum())} weeks)")

    r_roll, ent, ext = fwd(roll)
    prim = cell(vi, r_roll)
    say("PRIMARY_own_roll_index", prim)

    # --- episode table of the long setups (own index) ---------------------------------------
    long_setup = okv & (vi <= 20.0)
    eps = episodes(long_setup, okv)
    r_spot, _, _ = fwd(spot)
    r_cl, _, _ = fwd(crude_cl)
    elig = okv & np.isfinite(r_roll)
    base = float(r_roll[elig].mean())
    rows = []
    for a, b in eps:
        w = np.arange(a, b + 1)
        w_el = w[np.isfinite(r_roll[w])]
        rows.append(dict(first=str(FRI[a].date()), last=str(FRI[b].date()), weeks=int(b - a + 1),
                         counted=int(len(w_el)),
                         mean_r8_roll=float(np.mean(r_roll[w_el])) if len(w_el) else None,
                         sum_excess_roll=float(np.sum(r_roll[w_el] - base)) if len(w_el) else 0.0,
                         mean_r8_spot=float(np.nanmean(r_spot[w])) if np.isfinite(r_spot[w]).any() else None,
                         mean_r8_cl=float(np.nanmean(r_cl[w])) if np.isfinite(r_cl[w]).any() else None))
    say("episodes_long_ownroll", rows)
    # leave-one-episode-out excess (drop that episode's weeks from the signal and the baseline)
    loo = []
    for k, (a, b) in enumerate(eps):
        keep = np.ones(len(FRI), bool)
        keep[a:b + 1] = False
        c = cell(vi, r_roll, keep)
        loo.append(dict(dropped=f"{FRI[a].date()}..{FRI[b].date()}", n=c["n"],
                        excess=c.get("excess"), perm_p_exact=c.get("perm_p_exact")))
    say("leave_one_episode_out_ownroll", loo)

    # --- split half (midpoint date of eligible weeks) -------------------------------------
    d_el = FRI[elig]
    mid = d_el[0] + (d_el[-1] - d_el[0]) / 2
    h1 = cell(vi, r_roll, np.asarray(FRI <= mid))
    h2 = cell(vi, r_roll, np.asarray(FRI > mid))
    say("split_date_ownroll", str(mid.date()))
    say("H1_ownroll", h1)
    say("H2_ownroll", h2)
    # H2 without the 2020 episode (weeks 2020-01..2020-12 removed)
    no2020 = ~np.asarray((FRI >= "2020-01-01") & (FRI <= "2020-12-31"))
    say("H2_without_2020_ownroll", cell(vi, r_roll, np.asarray(FRI > mid) & no2020))
    say("FULL_without_2020_ownroll", cell(vi, r_roll, no2020))
    no0809 = ~np.asarray((FRI >= "2008-07-01") & (FRI <= "2009-06-30"))
    say("FULL_without_2008H2_2009H1_ownroll", cell(vi, r_roll, no0809))

    # --- overlap check: one observation per episode (its first counted week) -----------------
    first_wk = np.zeros(len(FRI), bool)
    for a, b in eps:
        w = [i for i in range(a, b + 1) if np.isfinite(r_roll[i])]
        if w:
            first_wk[w[0]] = True
    fr = r_roll[first_wk]
    say("first_week_of_episode_only_ownroll", dict(n=int(first_wk.sum()), mean=float(fr.mean()),
                                           baseline=base, excess=float(fr.mean() - base),
                                           hit=float((fr > 0).mean())))

    # --- roll-day and 2020-04 expiry checks on the own index ----------------------------------
    rr = roll.pct_change()
    big = rr.abs().sort_values(ascending=False).head(8)
    say("largest_daily_moves_own_roll", {str(d.date()): round(float(rr[d]), 4) for d in big.index})
    in_2020 = [str(FRI[i].date()) for i in np.flatnonzero(long_setup & np.isfinite(r_roll))
               if roll.index[ent[i]] <= pd.Timestamp("2020-04-21") <= roll.index[ext[i]]]
    say("long_weeks_whose_8w_window_contains_2020-04-21", in_2020)

    # --- the same primary on other crude series (same VI) -----------------------------------
    say("PRIMARY_on_crude_spot", cell(vi, r_spot))
    say("PRIMARY_on_crude_cl", cell(vi, r_cl))

    # --- compare with the harness crude_roll and its _lead5 variant (DATA only) --------------
    sys.path.insert(0, str(PKG / "battery" / "harness"))
    import fund_harness as fh  # noqa: E402  (used only to load the two series for comparison)
    hr = fh.load_series("crude_roll")
    j = roll.index.intersection(hr.index)
    d1, d2 = roll.reindex(j).pct_change(), hr.reindex(j).pct_change()
    diff = (d1 - d2).abs()
    say("own_vs_harness_roll", dict(common_days=int(len(j)), own_days=int(len(roll)),
                                    harness_days=int(len(hr)),
                                    days_return_diff_gt_1e9=int((diff > 1e-9).sum()),
                                    max_abs_return_diff=float(diff.max()),
                                    final_level_ratio=float((roll.reindex(j).iloc[-1] / roll.reindex(j).iloc[0])
                                                            / (hr.reindex(j).iloc[-1] / hr.reindex(j).iloc[0]))))
    r_h, _, _ = fwd(hr)
    say("PRIMARY_on_harness_crude_roll", cell(vi, r_h))
    r_l5, _, _ = fwd(fh.load_series("crude_roll_lead5"))
    say("PRIMARY_on_crude_roll_lead5", cell(vi, r_l5))
    say("H2_on_crude_roll_lead5", cell(vi, r_l5, np.asarray(FRI > mid)))
    say("H1_on_crude_roll_lead5", cell(vi, r_l5, np.asarray(FRI <= mid)))
    # episodes on lead5
    for row, (a, b) in zip(rows, eps):
        w = [i for i in range(a, b + 1) if np.isfinite(r_l5[i])]
        row["mean_r8_lead5"] = float(np.mean(r_l5[w])) if w else None
    say("episodes_long_with_lead5", rows)

    extended_checks(vi, hr, spot, r_h, eps)
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "audit_output.json").write_text(json.dumps(OUT, indent=1, default=str))


# =============================================================================================
# 7. Extended checks: secondary claims, single-episode dependence, walk-forward, pooling
# =============================================================================================
def ust10_price(y_pct: pd.Series) -> pd.Series:
    """Price of a 10-year, 6% semi-annual coupon bond at yield y (percent)."""
    y = y_pct.to_numpy(float) / 200.0
    disc = (1.0 + y) ** -20
    return pd.Series(3.0 * (1.0 - disc) / y + 100.0 * disc, index=y_pct.index)


def weekly_asof_lag(s: pd.Series, lag_days: int) -> np.ndarray:
    """As weekly_asof, but read as of Friday - lag_days (inputs published with a delay)."""
    d = FRI - pd.Timedelta(days=lag_days)
    pos = np.searchsorted(s.index.values, d.values, side="right") - 1
    out = np.full(len(FRI), np.nan)
    for i, p in enumerate(pos):
        if p >= 0 and (d.values[i] - s.index.values[p]) / np.timedelta64(1, "D") <= STALE_DAYS:
            out[i] = s.iloc[p]
    return out


def extended_checks(vi_prim: np.ndarray, hr: pd.Series, spot: pd.Series, r_h: np.ndarray,
                    eps: list) -> None:
    no2020 = ~np.asarray((FRI >= "2020-01-01") & (FRI <= "2020-12-31"))
    crude_w, gold_w = weekly_asof(v1_close("crude")), weekly_asof(v1_close("gold"))
    usd_w = weekly_asof_lag(eia_json("DTWEXBGS")[lambda s: s > 0], 7)
    dgs10 = eia_json("DGS10")
    bond_w = weekly_asof(ust10_price(dgs10[dgs10 > 0]))

    # --- claim 7/8: the crude 'nominal hits' and whether they survive dropping 2020 -------
    vi_usd = vi_index(crude_w, usd_w)
    vi_bond = vi_index(crude_w, bond_w)
    say("crude_vs_usd_long_h8", cell(vi_usd, r_h))
    say("crude_vs_usd_long_h8_without_2020", cell(vi_usd, r_h, no2020))
    # 2-of-3 vote: long when at least two of the three VIs <= 20 (defined only if all are)
    V = np.vstack([vi_prim, vi_bond, vi_usd])
    alldef = np.isfinite(V).all(axis=0)
    vote = np.where(alldef, np.where((V <= 20).sum(axis=0) >= 2, 0.0, 50.0), np.nan)
    say("crude_vote2of3_long_h8", cell(vote, r_h))
    say("crude_vote2of3_long_h8_without_2020", cell(vote, r_h, no2020))
    # cross-up variant: long in the week VI goes from <= 20 to > 20
    prev = np.concatenate(([np.nan], vi_prim[:-1]))
    cross = np.where(np.isfinite(vi_prim) & np.isfinite(prev),
                     np.where((prev <= 20) & (vi_prim > 20), 0.0, 50.0), np.nan)
    say("cross_up_long_h8", cell(cross, r_h))
    say("cross_up_long_h8_without_2020", cell(cross, r_h, no2020))
    say("cross_up_weeks", [str(FRI[i].date()) for i in np.flatnonzero(cross == 0.0)])

    # --- claim 9: post-hoc spot/roll decomposition, with and without 2020 -----------------
    idx = hr.index
    tgt = (FRI + pd.Timedelta(days=1)).values
    pos = np.searchsorted(idx.values, tgt, side="left")
    sp_idx = spot.index.values

    def spot_asof(d):
        p = np.searchsorted(sp_idx, np.datetime64(d), side="right") - 1
        if p < 0 or (np.datetime64(d) - sp_idx[p]) / np.timedelta64(1, "D") > STALE_DAYS:
            return np.nan
        return spot.iloc[p]
    spot_r = np.full(len(FRI), np.nan)
    for i in np.flatnonzero(np.isfinite(r_h) & np.isfinite(vi_prim)):
        spot_r[i] = spot_asof(idx[pos[i] + H_DAYS]) / spot_asof(idx[pos[i]]) - 1.0
    ok = np.isfinite(r_h) & np.isfinite(vi_prim) & np.isfinite(spot_r)
    dec = {}
    for name, msk in (("all_eligible", ok), ("long_setup", ok & (vi_prim <= 20)),
                      ("long_setup_without_2020", ok & (vi_prim <= 20) & no2020),
                      ("all_eligible_without_2020", ok & no2020),
                      ("long_setup_2020_only", ok & (vi_prim <= 20) & ~no2020)):
        dec[name] = dict(n=int(msk.sum()), roll=float(r_h[msk].mean()),
                         spot=float(spot_r[msk].mean()),
                         roll_minus_spot=float((r_h[msk] - spot_r[msk]).mean()))
    say("decomposition_h8", dec)
    april = spot[(spot.index >= "2020-04-15") & (spot.index <= "2020-04-30")]
    say("eia_spot_2020-04-15..30", {str(d.date()): float(v) for d, v in april.items()})

    # --- claim 3: extended history, crude vs bond halves (not reported by the tool) --------
    rclc1 = eia_json("RCLC1")
    rclc1 = rclc1[rclc1 > 0]
    cd = pd.read_csv(S / "work2" / "scout" / "curve_daily.csv", usecols=["market", "date", "price"],
                     parse_dates=["date"])
    gps = cd[(cd["market"] == "gold") & (cd["price"] > 0)].set_index("date")["price"].sort_index()
    vi_eg = vi_index(weekly_asof(rclc1), weekly_asof(gps))
    vi_eb = vi_index(weekly_asof(rclc1), bond_w)
    say("extended_vs_gold_long_h8", cell(vi_eg, r_h))
    say("extended_vs_bond_long_h8", cell(vi_eb, r_h))
    for nm, v in (("gold", vi_eg), ("bond", vi_eb)):
        el = np.isfinite(v) & np.isfinite(r_h)
        d_el = FRI[el]
        mid = d_el[0] + (d_el[-1] - d_el[0]) / 2
        say(f"extended_vs_{nm}_split", dict(split=str(mid.date()),
                                            H1=cell(v, r_h, np.asarray(FRI <= mid)),
                                            H2=cell(v, r_h, np.asarray(FRI > mid))))
    say("extended_vs_gold_long_h8_without_2020", cell(vi_eg, r_h, no2020))

    # --- claim 10: the pre-registered parameter-selection walk-forward, rebuilt ------------
    # 60 crude-vs-gold configurations; each 1 Jan from 2008 pick the one with the highest
    # long h=8 excess over weeks whose exit date is before the cutoff (>= 26 long weeks).
    exit_date = np.array([idx[p + H_DAYS] if (p + H_DAYS) < len(idx) else np.datetime64("NaT", "ns")
                          for p in pos], dtype="datetime64[ns]")
    cfgs = {}
    for ek, (s_, l_) in (("ema2_22", (2, 22)), ("ema5_15", (5, 15)), ("none", (None, 0))):
        for N in (52, 65, 104, 156, 260):
            v = vi_index(crude_w, gold_w, s_, l_, N)
            for (b, s2) in ((20, 80), (25, 75), (15, 85), (15, 75)):
                cfgs[f"{ek}|N{N}|{b}/{s2}"] = (v, b)
    wf_sig = np.full(len(FRI), np.nan)
    chosen_log = []
    cutoffs = pd.date_range("2008-01-01", "2026-01-01", freq="YS")
    for k, cut in enumerate(cutoffs):
        nxt = cutoffs[k + 1] if k + 1 < len(cutoffs) else pd.Timestamp("2027-01-01")
        past = np.asarray(FRI < cut) & np.isfinite(r_h) & (exit_date < np.datetime64(cut))
        best, best_ex = None, 0.0
        for name, (v, b) in cfgs.items():
            el = past & np.isfinite(v)
            lw = el & (v <= b)
            if lw.sum() < 26:
                continue
            ex = r_h[lw].mean() - r_h[el].mean()
            if ex > best_ex:
                best, best_ex = name, ex
        chosen_log.append(f"{cut.year}:{best}")
        blk = np.asarray((FRI >= cut) & (FRI < nxt))
        if best is None:
            wf_sig[blk] = np.where(np.isfinite(vi_prim[blk]), 50.0, np.nan)
        else:
            v, b = cfgs[best]
            wf_sig[blk] = np.where(np.isfinite(v[blk]), np.where(v[blk] <= b, 0.0, 50.0), np.nan)
    say("wf_selection_choices", chosen_log)
    say("wf_selection_long_h8", cell(wf_sig, r_h))
    say("wf_selection_long_h8_without_2020", cell(wf_sig, r_h, no2020))
    in08 = np.asarray(FRI >= "2008-01-01")
    say("static_primary_2008plus", cell(np.where(in08, vi_prim, np.nan), r_h))
    say("static_primary_2008plus_without_2020", cell(np.where(in08, vi_prim, np.nan), r_h, no2020))

    # --- claim 5: pooled 4 markets vs gold, long h8 (own pooling, harness RNG order) -------
    parts = [("crude", vi_prim, r_h)]
    for mk in ("silver", "soybean", "sugar"):
        px = v1_close(mk)
        parts.append((mk, vi_index(weekly_asof(px), gold_w), fwd(px)[0]))
    rng = np.random.default_rng(11)
    sums, allr, allx = np.zeros(2000), [], []
    for mk, v, r in parts:
        el = np.isfinite(v) & np.isfinite(r)
        sig = (v <= 20)[el]
        re = r[el]
        T = len(re)
        offs = rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=2000, endpoint=True)
        p_ = np.flatnonzero(sig)
        sums += re[(p_[None, :] + offs[:, None]) % T].sum(axis=1)
        allr.append(re[sig])
        allx.append(re[sig] - re.mean())
    sr = np.concatenate(allr)
    say("pooled4_vs_gold_long_h8", dict(n=int(len(sr)), mean=float(sr.mean()),
                                        excess=float(np.concatenate(allx).mean()),
                                        hit=float((sr > 0).mean()),
                                        perm_p=(1 + int((sums / len(sr) >= sr.mean()).sum())) / 2001))

    # --- claim 8: pooled vs usd and 2-of-3 vote, long h8 / h13, with and without 2020 -------
    def pooled(parts_, keep=None):
        xs, rs = [], []
        for v, r in parts_:
            el = np.isfinite(v) & np.isfinite(r) & (np.ones(len(v), bool) if keep is None else keep)
            sig = el & (v <= 20)
            rs.append(r[sig])
            xs.append(r[sig] - r[el].mean())
        return dict(n=int(sum(len(a) for a in rs)), excess=float(np.concatenate(xs).mean()))
    mk_px = {mk: v1_close(mk) for mk in ("silver", "soybean", "sugar")}
    for hw in (8, 13):
        r_c = fwd(hr, 5 * hw)[0]
        pu, pv = [(vi_usd, r_c)], [(vote, r_c)]
        for mk, px in mk_px.items():
            w = weekly_asof(px)
            r_m = fwd(px, 5 * hw)[0]
            vu = vi_index(w, usd_w)
            vg = vi_index(w, gold_w)
            vb = vi_index(w, bond_w)
            Vm = np.vstack([vg, vb, vu])
            vm = np.where(np.isfinite(Vm).all(axis=0),
                          np.where((Vm <= 20).sum(axis=0) >= 2, 0.0, 50.0), np.nan)
            pu.append((vu, r_m))
            pv.append((vm, r_m))
        say(f"pooled4_vs_usd_long_h{hw}", pooled(pu))
        say(f"pooled4_vs_usd_long_h{hw}_without_2020", pooled(pu, no2020))
        say(f"pooled4_vote2of3_long_h{hw}", pooled(pv))
        say(f"pooled4_vote2of3_long_h{hw}_without_2020", pooled(pv, no2020))

    # --- claim 9: primary traded on spot / unadjusted CL without 2020 ------------------------
    say("PRIMARY_on_crude_spot_without_2020", cell(vi_prim, fwd(spot)[0], no2020))
    say("PRIMARY_on_crude_cl_without_2020", cell(vi_prim, fwd(v1_close("crude"))[0], no2020))

    # --- claim 4: S1 long-only trades rebuilt (first week of each long episode, 40-day hold,
    #     one position at a time, 0.05% per side) on the harness crude_roll -----------------
    trades, busy_until = [], -1
    for a, b in eps:
        i = a
        p = pos[i]
        if p >= len(idx) or p <= busy_until:      # no entry, or a trade is still open
            continue
        x = min(p + H_DAYS, len(idx) - 1)       # exit after 40 rows (or at the series end)
        net = (hr.iloc[x] / hr.iloc[p]) * (1 - 0.0005) * (1 - 0.0005) - 1.0
        trades.append(dict(signal=str(FRI[i].date()), entry=str(idx[p].date()),
                           exit=str(idx[x].date()), net=round(float(net), 6)))
        busy_until = x
    say("S1_long_only_trades_rebuilt", trades)
    eq = float(np.prod([1 + t["net"] for t in trades]))
    yrs = (pd.Timestamp("2024-04-05") - pd.Timestamp("2004-01-20")).days / 365.25
    say("S1_long_only_rebuilt_summary", dict(n=len(trades),
                                             win_rate=float(np.mean([t["net"] > 0 for t in trades])),
                                             total=eq - 1, cagr=eq ** (1 / yrs) - 1))

    # --- claim 11: latest readings at the last Friday -------------------------------------
    last = {}
    for mk in ("crude", "silver", "soybean", "sugar", "gold"):
        w = weekly_asof(v1_close(mk))
        for cn, cw in (("gold", gold_w), ("bond", bond_w), ("usd", usd_w)):
            if mk == "gold" and cn == "gold":
                continue
            v = vi_index(w, cw)
            j = np.flatnonzero(np.isfinite(v))[-1]
            last[f"{mk}_vs_{cn}"] = (str(FRI[j].date()), round(float(v[j]), 1))
    say("latest_readings", last)


if __name__ == "__main__":
    main()

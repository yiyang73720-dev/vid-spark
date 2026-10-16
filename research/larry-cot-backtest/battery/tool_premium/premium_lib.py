#!/usr/bin/env python3
"""
premium_lib -- data and signal builders for the Larry Williams premium/discount test
(SPEC2 battery, tool_premium). Evaluation itself is done by the shared harness
(battery/harness/fund_harness.py); this module only turns raw inputs into weekly signals and
builds the extra traded series the harness does not have (pysystemtrade roll-adjusted indexes).

Larry's tool (see PREREG.md): nearby contract ABOVE the deferred contract = "premium" market
(inverted / backwardated) = bullish; nearby BELOW deferred = "discount" / carrying-charge market
= bearish. The premium is measured as p = nearby / deferred - 1 (> 0 means premium).

No-look-ahead conventions used throughout:
  * A weekly value is read "as of" the Friday close: the last daily observation dated on or before
    the Friday, and only if it is at most STALE_DAYS calendar days old (else undefined / NaN).
  * Rolling / expanding statistics (52-week range index, z-scores, 5-year inventory norms) use only
    observations up to and including the decision week.
Run with `python3 -I`; third-party files are parsed (pandas.read_csv / json), never executed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
HARNESS = PKG / "battery" / "harness"                    # fund_harness.py
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))
import fund_harness as fh  # noqa: E402  (sets sys.dont_write_bytecode)

CURVE_CSV = ROOT / "work2" / "scout" / "curve_daily.csv"   # scout's tidy pysystemtrade curve
STALE_DAYS = 4                 # a Friday reading may come from Mon..Fri of the same week
PST_MARKETS = ("crude", "gold", "silver", "soybean", "sugar", "corn", "wheat", "copper",
               "natgas", "heatoil")
ROLL_LOOKBACK_ROWS = 10        # how far back an unbridged pysystemtrade roll may be moved


# =============================================================================================
# 1. Weekly "as of Friday" sampling
# =============================================================================================
def weekly_asof(daily: pd.Series, fridays: pd.DatetimeIndex, max_age_days: int = STALE_DAYS
                ) -> np.ndarray:
    """Value of `daily` at the last observation dated on or before each Friday; NaN when there is
    none or when it is more than `max_age_days` calendar days old (data gap / series not started).
    Only data available at the Friday close are used."""
    s = daily.dropna().sort_index()
    idx = s.index.values
    f = pd.DatetimeIndex(fridays).values
    pos = np.searchsorted(idx, f, side="right") - 1
    out = np.full(len(f), np.nan)
    ok = pos >= 0
    age = np.full(len(f), np.inf)
    age[ok] = (f[ok] - idx[pos[ok]]) / np.timedelta64(1, "D")
    ok &= age <= max_age_days
    out[ok] = s.to_numpy(dtype=float)[pos[ok]]
    return out


# =============================================================================================
# 2. Crude curve from the EIA NYMEX contract prints (RCLC1..RCLC4)
# =============================================================================================
def crude_premium_daily(near: int = 1, far: int = 2) -> pd.Series:
    """Daily premium p = c_near / c_far - 1 from the EIA contract prints, on days where both
    contracts print a positive value and the row is not a stale holiday placeholder (the same day
    filter the harness uses for the crude_roll index)."""
    raw = fh.load_cl_contracts()
    stale = fh._stale_rows(raw)            # holiday rows that repeat the previous prints
    a, b = raw[f"c{near}"], raw[f"c{far}"]
    ok = (~stale) & (a > 0).to_numpy() & (b > 0).to_numpy()
    p = (a[ok] / b[ok] - 1.0).rename(f"p_c{near}c{far}")
    p.index.name = "date"
    return p


# =============================================================================================
# 3. Signal transforms of a weekly premium series (all +1 / 0 / -1, NaN = undefined)
# =============================================================================================
def sig_sign(p: np.ndarray) -> np.ndarray:
    """Larry's literal rule: premium (> 0) long, discount (< 0) short, equal -> 0."""
    p = np.asarray(p, float)
    return np.where(np.isnan(p), np.nan, np.sign(p))


def sig_threshold(p: np.ndarray, x: float) -> np.ndarray:
    """Spread scaled by price with a dead band: long if p >= +x, short if p <= -x, else 0."""
    p = np.asarray(p, float)
    s = np.where(p >= x, 1.0, np.where(p <= -x, -1.0, 0.0))
    return np.where(np.isnan(p), np.nan, s)


def range_index(p: np.ndarray, n: int = 52) -> np.ndarray:
    """Larry-style 0-100 index of the premium over the last n weekly values INCLUDING the current
    one: 100 * (p - min_n) / (max_n - min_n). NaN until n defined values are in the window, if any
    value in the window is undefined, or if max == min."""
    s = pd.Series(np.asarray(p, float))
    lo = s.rolling(n, min_periods=n).min()
    hi = s.rolling(n, min_periods=n).max()
    rng = (hi - lo).where(hi > lo)
    return (100.0 * (s - lo) / rng).to_numpy()


def sig_range(p: np.ndarray, n: int = 52, buy: float = 80.0, sell: float = 20.0) -> np.ndarray:
    """Long when the range index >= buy, short when <= sell (the COT-index convention)."""
    ix = range_index(p, n)
    s = np.where(ix >= buy, 1.0, np.where(ix <= sell, -1.0, 0.0))
    return np.where(np.isnan(ix), np.nan, s)


def sig_steepen(p: np.ndarray, lag: int = 4) -> np.ndarray:
    """Spread change: long if the premium rose over the last `lag` weeks, short if it fell."""
    s = pd.Series(np.asarray(p, float))
    d = (s - s.shift(lag)).to_numpy()
    return np.where(np.isnan(d), np.nan, np.sign(d))


def sig_cross(p: np.ndarray, window: int = 4) -> np.ndarray:
    """Fresh cross (Larry: the cross from a discount to a premium market is bullish):
    long if p(t) > 0 and p was <= 0 in any of the previous `window` weeks; short if p(t) < 0 and
    p was >= 0 in any of the previous `window` weeks; else 0. Undefined previous weeks never count
    as a cross."""
    s = pd.Series(np.asarray(p, float))
    prev = pd.concat([s.shift(k) for k in range(1, window + 1)], axis=1)
    was_le0 = (prev <= 0).any(axis=1).to_numpy()
    was_ge0 = (prev >= 0).any(axis=1).to_numpy()
    v = s.to_numpy()
    out = np.where((v > 0) & was_le0, 1.0, np.where((v < 0) & was_ge0, -1.0, 0.0))
    return np.where(np.isnan(v), np.nan, out)


def sig_zexp(p: np.ndarray, min_periods: int = 104, z: float = 1.0) -> np.ndarray:
    """Premium z-scored against its EXPANDING mean/std (past + current weekly values only,
    fh.expanding_zscore): long if z >= +1, short if z <= -1."""
    zz = fh.expanding_zscore(np.asarray(p, float), min_periods=min_periods)
    s = np.where(zz >= z, 1.0, np.where(zz <= -z, -1.0, 0.0))
    return np.where(np.isnan(zz), np.nan, s)


# =============================================================================================
# 4. pysystemtrade curves (other markets) and roll-adjusted held-contract indexes
# =============================================================================================
_CURVE_CACHE: dict[str, pd.DataFrame] = {}


def pst_curve(market: str) -> pd.DataFrame:
    """The scout's tidy daily curve for one market (one row per weekday; last row of each date;
    rows whose two legs were the same contract already blanked by the scout)."""
    if not _CURVE_CACHE:
        c = pd.read_csv(CURVE_CSV, parse_dates=["date"])
        for m, g in c.groupby("market"):
            _CURVE_CACHE[str(m)] = g.sort_values("date").reset_index(drop=True)
    return _CURVE_CACHE[market].copy()


def pst_premium_daily(market: str) -> pd.Series:
    """Daily premium near / far - 1 on days where both legs are positive."""
    c = pst_curve(market)
    ok = (c["near"] > 0) & (c["far"] > 0)
    p = pd.Series((c.loc[ok, "near"] / c.loc[ok, "far"] - 1.0).to_numpy(),
                  index=pd.DatetimeIndex(c.loc[ok, "date"], name="date"), name=f"p_{market}")
    return p


def pst_roll_index(market: str) -> tuple[pd.Series, dict]:
    """Roll-adjusted (multiplicative) excess-return index of the contract pysystemtrade HOLDS
    (the PRICE column), 100 on the first day.

    Index days: rows with a positive PRICE. Same contract on consecutive index days:
    r = P(t)/P(t-1) - 1. Roll (PRICE contract changes X -> Y between t-1 and t): r = P_Y(t) /
    P_Y(t-1) - 1 with P_Y(t-1) read from the FORWARD or CARRY column of row t-1 when that column
    holds contract Y. Otherwise, if the OLD contract X is still priced on row t (CARRY/FORWARD
    column), the switch is made at the close of t instead: r = P_X(t)/P_X(t-1) - 1. Otherwise
    (unbridged) the switch is moved back to the latest earlier
    index day j (at most ROLL_LOOKBACK_ROWS rows back) where Y has a price, and the return over
    (t-1, t] becomes [P_Y(t)/P_Y(j)] / [P_X(t-1)/P_X(j)] - 1, i.e. X's already-booked return
    from j to t-1 is replaced by Y's (an exact switch at the close of j). If no such j exists, the
    roll-day return is set to 0 and counted as unresolved.
    Returns (level, info)."""
    c = pst_curve(market)
    c = c[c["price"].notna() & (c["price"] > 0)].reset_index(drop=True)
    pc = c["price_contract"].to_numpy()
    px = c["price"].to_numpy(dtype=float)
    fc, fp = c["forward_contract"].to_numpy(), c["forward"].to_numpy(dtype=float)
    cc, cp = c["carry_contract"].to_numpy(), c["carry"].to_numpy(dtype=float)

    def price_of(contract, j):
        """Price of `contract` on row j from the FORWARD/CARRY/PRICE columns (NaN if absent)."""
        if pc[j] == contract:
            return px[j]
        if fc[j] == contract and np.isfinite(fp[j]) and fp[j] > 0:
            return fp[j]
        if cc[j] == contract and np.isfinite(cp[j]) and cp[j] > 0:
            return cp[j]
        return np.nan

    n = len(c)
    ret = np.zeros(n)
    counts = dict(rolls=0, bridged_t_minus_1=0, bridged_old_contract_at_t=0,
                  bridged_lookback=0, unresolved=0)
    moved = []
    for t in range(1, n):
        if pc[t] == pc[t - 1]:
            ret[t] = px[t] / px[t - 1] - 1.0
            continue
        counts["rolls"] += 1
        y_prev = price_of(pc[t], t - 1)
        if np.isfinite(y_prev):
            ret[t] = px[t] / y_prev - 1.0
            counts["bridged_t_minus_1"] += 1
            continue
        x_now = price_of(pc[t - 1], t)      # old contract still priced at t: switch at t's close
        if np.isfinite(x_now):
            ret[t] = x_now / px[t - 1] - 1.0
            counts["bridged_old_contract_at_t"] += 1
            continue
        done = False
        for j in range(t - 2, max(-1, t - 2 - ROLL_LOOKBACK_ROWS), -1):
            if pc[j] != pc[t - 1]:          # never reach back past the previous roll
                break
            y_j = price_of(pc[t], j)
            if np.isfinite(y_j):
                ret[t] = (px[t] / y_j) / (px[t - 1] / px[j]) - 1.0
                counts["bridged_lookback"] += 1
                moved.append(dict(date=c.loc[t, "date"].date().isoformat(),
                                  switch_moved_to=c.loc[j, "date"].date().isoformat()))
                done = True
                break
        if not done:
            ret[t] = 0.0
            counts["unresolved"] += 1
            moved.append(dict(date=c.loc[t, "date"].date().isoformat(), switch_moved_to=None))
    level = pd.Series(100.0 * np.cumprod(1.0 + ret),
                      index=pd.DatetimeIndex(c["date"], name="date"), name=f"pst_{market}")
    absr = pd.Series(np.abs(ret), index=level.index)
    top = absr.sort_values(ascending=False).head(3)
    held_months = (fh_months(pc) - fh_months_dates(c["date"]))
    info = dict(market=market, first=level.index[0].date().isoformat(),
                last=level.index[-1].date().isoformat(), n_days=int(n), **counts,
                rolls_moved_or_unresolved=moved,
                largest_abs_daily_returns=[dict(date=d.date().isoformat(), ret=float(ret[level.index.get_loc(d)]))
                                           for d in top.index],
                held_contract_months_to_delivery_median=float(np.median(held_months)),
                total_return=float(level.iloc[-1] / level.iloc[0] - 1.0),
                min_level=float(level.min()))
    if not (level > 0).all():
        raise AssertionError(f"{market}: roll index went non-positive")
    return level, info


def fh_months(ids: np.ndarray) -> np.ndarray:
    """Contract id YYYYMMDD -> absolute month number (year*12 + month-1)."""
    ids = np.asarray(ids, dtype=np.int64)
    return (ids // 10000) * 12 + ((ids // 100) % 100 - 1)


def fh_months_dates(dates: pd.Series) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.asarray(d.year * 12 + d.month - 1, dtype=np.int64)


# =============================================================================================
# 5. Non-Larry comparator: EIA commercial crude stocks vs their 5-year seasonal norm
# =============================================================================================
def inventory_vs_norm(series_id: str = "WCESTUS1", years: int = 5, tol_days: int = 3
                      ) -> pd.DataFrame:
    """Weekly stocks (week-ending Fridays) and dev = stocks / norm - 1, where norm is the mean of
    the stocks on the week-ending dates nearest to (date - k years), k = 1..years, each within
    +-tol_days. All `years` matches are required; only past weeks are used."""
    w = fh.load_ext_raw(series_id)
    idx = w.index.values
    vals = w.to_numpy(dtype=float)
    past = np.full((len(w), years), np.nan)
    for k in range(1, years + 1):
        tgt = (w.index - pd.DateOffset(years=k)).values
        pos = np.searchsorted(idx, tgt)
        best = np.full(len(w), -1)
        bestd = np.full(len(w), np.inf)
        for cand in (pos - 1, pos):
            okc = (cand >= 0) & (cand < len(idx))
            d = np.full(len(w), np.inf)
            d[okc] = np.abs((idx[cand[okc]] - tgt[okc]) / np.timedelta64(1, "D"))
            better = d < bestd
            best = np.where(better, cand, best)
            bestd = np.where(better, d, bestd)
        ok = (best >= 0) & (bestd <= tol_days)
        past[ok, k - 1] = vals[best[ok]]
    norm = np.where(np.isnan(past).any(axis=1), np.nan, past.mean(axis=1))
    return pd.DataFrame({"stocks": vals, "norm": norm, "dev": vals / norm - 1.0},
                        index=pd.DatetimeIndex(w.index, name="week_ending"))


# =============================================================================================
# 6. Small statistics helpers
# =============================================================================================
def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p-values stay NaN)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q

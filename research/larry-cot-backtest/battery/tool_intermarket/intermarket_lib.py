#!/usr/bin/env python3
"""
intermarket_lib -- signal builders for the Larry Williams intermarket test (SPEC2 battery,
tool_intermarket). The evaluation itself (forward returns, baseline, excess, permutation tests,
split-half, walk-forward, S1/S2 strategy) is done by the shared harness
`battery/harness/fund_harness.py`. This module only turns raw daily inputs into weekly +1/0/-1
signals on the Friday decision grid.

Larry's idea (see PREREG.md): a market is led by the short-term TREND of the market that
"influences" it. Gold influences bonds, inversely. Bonds influence stocks. The dollar is tied to
gold. Inflation follows gold. This module builds:

  * leader-change signals: the sign of a leading market's L-week change (dollar and bond inverse,
    gold direct);
  * Larry's literal gold -> bonds filter: gold below its close 20 trading days earlier = long bonds;
  * Willspread: EMA(fast) - EMA(slow) of 100 * market / secondary on common daily dates;
  * a 2-of-3 / 2-of-2 vote composite and an own-trend control.

No-look-ahead conventions used throughout:
  * A weekly value is read "as of" a date: the last daily observation dated on or before it, and
    only if it is at most STALE_DAYS calendar days old (else NaN, i.e. the week is undefined).
  * Inputs published with a delay are read as of (Friday - lag): the broad dollar DTWEXBGS comes
    out on Mondays for the previous week, so lag = 7 days.
  * EMAs are recursive (pandas ewm, adjust=False), so a value uses only past and current data.
Run with `python3 -I`; third-party files are parsed by the harness / premium_lib (read_csv / json),
never executed.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True      # never write .pyc files next to shared code

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
HARNESS = PKG / "battery" / "harness"
PREMIUM = PKG / "battery" / "tool_premium"     # reused (read-only): pysystemtrade roll indexes
for p in (HARNESS, PREMIUM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import fund_harness as fh  # noqa: E402
import premium_lib as pl   # noqa: E402

STALE_DAYS = 4             # a Friday reading may come from Mon..Fri of the same week
USD_LAG_DAYS = 7           # DTWEXBGS (H.10) is released Mondays for the previous week

# Leader name -> (series loader name in the harness, publication lag in days, pre-registered sign).
# sign -1 = inverse (leader down -> long the market), +1 = direct.
LEADERS = {
    "usd": ("usd_broad", USD_LAG_DAYS, -1),
    "bond": ("ust10_price", 0, -1),
    "gold": ("gold", 0, +1),
}


# =============================================================================================
# As-of readings
# =============================================================================================
def asof_stale(series: pd.Series, dates, lag_days: int = 0,
               max_age_days: int = STALE_DAYS) -> np.ndarray:
    """Value of `series` at the last observation dated on or before (date - lag_days), NaN when
    there is none or when that observation is more than `max_age_days` older than (date - lag)."""
    s = series.dropna().sort_index()
    d = pd.DatetimeIndex(pd.to_datetime(dates)) - pd.Timedelta(days=lag_days)
    idx = s.index.values
    pos = np.searchsorted(idx, d.values, side="right") - 1
    out = np.full(len(d), np.nan)
    ok = pos >= 0
    age = np.full(len(d), np.inf)
    age[ok] = (d.values[ok] - idx[pos[ok]]) / np.timedelta64(1, "D")
    ok &= age <= max_age_days
    out[ok] = s.to_numpy(dtype=float)[pos[ok]]
    return out


def leader_change(series: pd.Series, fridays, weeks: int, lag_days: int = 0) -> np.ndarray:
    """chg_L(t) = x(t - lag) / x(t - lag - 7L days) - 1, both read with the staleness rule."""
    a = asof_stale(series, fridays, lag_days)
    b = asof_stale(series, fridays, lag_days + 7 * int(weeks))
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b - 1.0


def sign_signal(x: np.ndarray, sign: int) -> np.ndarray:
    """+1 / -1 / 0 = sign * sign(x); NaN where x is NaN (undefined week)."""
    x = np.asarray(x, dtype=float)
    out = sign * np.sign(x)
    out[np.isnan(x)] = np.nan
    return out + 0.0          # turn -0.0 into 0.0


def leader_signal(leader: str, fridays, weeks: int, *, series: pd.Series | None = None,
                  sign: int | None = None, lag_days: int | None = None) -> np.ndarray:
    """Signal for 'follow the leader's L-week trend' with the pre-registered sign (LEADERS),
    unless `sign` / `lag_days` / `series` are given explicitly."""
    name, lag0, sign0 = LEADERS[leader]
    s = fh.load_series(name) if series is None else series
    chg = leader_change(s, fridays, weeks, lag0 if lag_days is None else lag_days)
    return sign_signal(chg, sign0 if sign is None else sign)


# =============================================================================================
# Larry's literal gold -> bonds filter (GSV reconstruction, PREREG source 3)
# =============================================================================================
def gold_vs_n_days_ago(gold: pd.Series, fridays, n: int = 20) -> np.ndarray:
    """On gold's own daily calendar: compare the last close on or before the Friday (staleness
    rule) with the close n rows earlier. Returns the change g(pos)/g(pos-n) - 1 (NaN if
    undefined)."""
    g = gold.dropna().sort_index()
    d = pd.DatetimeIndex(pd.to_datetime(fridays))
    idx = g.index.values
    pos = np.searchsorted(idx, d.values, side="right") - 1
    v = g.to_numpy(dtype=float)
    out = np.full(len(d), np.nan)
    ok = pos >= n
    age = np.full(len(d), np.inf)
    age[pos >= 0] = (d.values[pos >= 0] - idx[pos[pos >= 0]]) / np.timedelta64(1, "D")
    ok &= age <= STALE_DAYS
    out[ok] = v[pos[ok]] / v[pos[ok] - n] - 1.0
    return out


def larry_gold_bonds_signal(gold: pd.Series, fridays, n: int = 20) -> np.ndarray:
    """Long bonds (+1) when gold is BELOW its close n gold trading days earlier, short (-1) when
    above, 0 when equal; NaN when undefined (inverse: rising gold is bad for bonds)."""
    return sign_signal(gold_vs_n_days_ago(gold, fridays, n), -1)


# =============================================================================================
# Willspread (PREREG source 2)
# =============================================================================================
def willspread_daily(market: pd.Series, secondary: pd.Series, fast: int = 5,
                     slow: int = 20) -> pd.Series:
    """WS = EMA_fast(S) - EMA_slow(S), S = 100 * market / secondary on their COMMON daily dates.
    EMA alpha = 2/(n+1), recursive, seeded with the first S; WS is NaN before the slow-th common
    day."""
    j = pd.concat([market.rename("m"), secondary.rename("x")], axis=1, join="inner").dropna()
    j = j[(j["m"] > 0) & (j["x"] > 0)]
    s = 100.0 * j["m"] / j["x"]
    ws = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    ws.iloc[: slow - 1] = np.nan
    return ws.rename("willspread")


def willspread_signal(market: pd.Series, secondary: pd.Series, fridays, *, lag_days: int = 0,
                      fast: int = 5, slow: int = 20) -> np.ndarray:
    """Long the market when WS > 0 (market gaining on its influencing market), short when < 0.
    Read as of (Friday - lag_days) with the staleness rule (lag 7 when the secondary is the
    dollar, so both legs are public)."""
    ws = willspread_daily(market, secondary, fast, slow)
    return sign_signal(asof_stale(ws, fridays, lag_days), +1)


# =============================================================================================
# Composites and controls
# =============================================================================================
def vote_signal(signals: list[np.ndarray], need: int) -> np.ndarray:
    """Long when at least `need` component signals are +1, short when at least `need` are -1,
    else 0. Valid (non-NaN) only when every component is defined. For 3 components need=2 (2 of
    3); for 2 components need=2 (both agree)."""
    a = np.vstack([np.asarray(s, dtype=float) for s in signals])
    defined = ~np.isnan(a).any(axis=0)
    up = (a == 1).sum(axis=0)
    dn = (a == -1).sum(axis=0)
    out = np.where(up >= need, 1.0, np.where(dn >= need, -1.0, 0.0))
    out[~defined] = np.nan
    return out


def own_trend_signal(series: pd.Series, fridays, weeks: int = 8) -> np.ndarray:
    """Control: long when the market's OWN L-week change at the Friday close is > 0."""
    return sign_signal(leader_change(series, fridays, weeks, 0), +1)


# =============================================================================================
# Frames, series and multiple testing
# =============================================================================================
def fridays_for(series: pd.Series, start=None, end=None) -> pd.DatetimeIndex:
    """Friday grid clipped to the traded series' own date range (and to [start, end])."""
    a = series.index[0] if start is None else max(series.index[0], pd.Timestamp(start))
    b = series.index[-1] if end is None else min(series.index[-1], pd.Timestamp(end))
    return fh.weekly_decision_dates(a, b)


def pst_index(market: str) -> tuple[pd.Series, dict]:
    """pysystemtrade roll-adjusted excess-return index of the held contract (tool_premium's
    builder, reused read-only). Named pst_<market>."""
    lvl, info = pl.pst_roll_index(market)
    lvl = lvl[lvl > 0].sort_index()
    lvl = lvl[~lvl.index.duplicated(keep="last")]
    return lvl.rename(f"pst_{market}"), info


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p-values stay NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok], kind="mergesort")]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q

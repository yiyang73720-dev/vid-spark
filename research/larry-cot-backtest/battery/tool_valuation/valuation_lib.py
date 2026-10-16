#!/usr/bin/env python3
"""
valuation_lib -- Larry Williams' Valuation Index (WillVal) on a weekly Friday grid.

Pre-registered definition (PREREG.md, section 2):

    R_t   = 100 * M_t / C_t                     market price relative to a comparator
    Pal_t = EMA_s(R)_t - EMA_l(R)_t             short minus long EMA (s=2, l=22 primary)
    VI_t  = 100 * (Pal_t - min_N) / (max_N - min_N)
                                                min/max over the last N defined Pal values,
                                                including the current one (N=156 primary)
    long  setup (undervalued) when VI <= buy_th  (20)
    short setup (overvalued)  when VI >= sell_th (80)

Everything here is causal: a value at Friday t uses only prices dated on or before t (or
before t - lag for inputs published with a delay), EMAs are recursive and the min/max window
ends at t. Nothing reads a forward return.

Run with `python3 -I`; the third-party files (S/ext*, the scout's curve file) are parsed with
pandas/json only, never executed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
sys.path.insert(0, str(PKG / "battery" / "harness"))
import fund_harness as fh  # noqa: E402  shared battery harness

# Pre-registered constants ------------------------------------------------------------------
MAX_AGE_DAYS = 4            # an as-of close older than this (calendar days) makes the week undefined
USD_LAG_DAYS = 7            # DTWEXBGS is published on Mondays for the week to the previous Friday
PRIMARY = dict(ema=(2, 22), N=156, buy_th=20.0, sell_th=80.0)

# Grid B (PREREG section 5)
LOOKBACKS = (52, 65, 104, 156, 260)
EMAS = {"ema2_22": (2, 22), "ema5_15": (5, 15), "none": None}
THRESHOLDS = ((20.0, 80.0), (25.0, 75.0), (15.0, 85.0), (15.0, 75.0))

GRID_START, GRID_END = "1985-01-04", "2026-10-02"     # all Fridays with any possible data


# ===========================================================================================
# Reading inputs as of each Friday
# ===========================================================================================
def asof_fresh(series: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 0,
               max_age_days: int = MAX_AGE_DAYS) -> np.ndarray:
    """Value of `series` at its last observation on or before (date - lag_days); NaN when there
    is none or when that observation is more than `max_age_days` calendar days older than
    (date - lag_days). Same lookup as fh.asof_values plus a staleness limit."""
    s = series.dropna().sort_index()
    d = pd.DatetimeIndex(dates) - pd.Timedelta(days=lag_days)
    pos = np.searchsorted(s.index.values, d.values, side="right") - 1
    out = np.full(len(d), np.nan)
    ok = pos >= 0
    age = np.full(len(d), np.inf)
    age[ok] = (d.values[ok] - s.index.values[pos[ok]]) / np.timedelta64(1, "D")
    ok &= age <= max_age_days
    out[ok] = s.to_numpy(dtype=float)[pos[ok]]
    return out


# ===========================================================================================
# The index
# ===========================================================================================
def ema_recursive(x: np.ndarray, n: int) -> np.ndarray:
    """EMA with alpha = 2/(n+1), y_0 = x_0, y_t = alpha*x_t + (1-alpha)*y_{t-1} (no NaN in x)."""
    a = 2.0 / (n + 1.0)
    y = np.empty(len(x))
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def valuation_index(m: np.ndarray, c: np.ndarray, ema: tuple[int, int] | None = (2, 22),
                    N: int = 156) -> dict:
    """Valuation index on aligned weekly arrays (NaN = undefined week).

    Weeks where either input is undefined are DROPPED before smoothing (the EMAs and the
    min/max window run over consecutive defined weeks) and come back as NaN.
    Returns dict(R, pal, vi) as full-length arrays."""
    m = np.asarray(m, float)
    c = np.asarray(c, float)
    R = 100.0 * m / c
    ok = np.isfinite(R) & (c > 0) & (m > 0)
    r = R[ok]
    if ema is None:
        pal_d = r.copy()                                   # variant "none": range of the ratio
    else:
        s, l = ema
        pal_d = ema_recursive(r, s) - ema_recursive(r, l)
        pal_d[: l - 1] = np.nan                            # defined from the l-th observation on
    pal_s = pd.Series(pal_d)
    lo = pal_s.rolling(N, min_periods=N).min().to_numpy()
    hi = pal_s.rolling(N, min_periods=N).max().to_numpy()
    rng = hi - lo
    vi_d = np.where(rng > 0, 100.0 * (pal_d - lo) / np.where(rng > 0, rng, 1.0), np.nan)
    vi_d[~np.isfinite(pal_d)] = np.nan
    out = {k: np.full(len(m), np.nan) for k in ("R", "pal", "vi")}
    out["R"][ok] = r
    out["pal"][ok] = pal_d
    out["vi"][ok] = vi_d
    return out


def setups(vi: np.ndarray, buy_th: float, sell_th: float) -> np.ndarray:
    """+1 when VI <= buy_th (undervalued), -1 when VI >= sell_th (overvalued), 0 otherwise,
    NaN where VI is undefined."""
    vi = np.asarray(vi, float)
    sig = np.where(vi <= buy_th, 1.0, np.where(vi >= sell_th, -1.0, 0.0))
    return np.where(np.isfinite(vi), sig, np.nan)


def cross_setups(vi: np.ndarray, buy_th: float, sell_th: float) -> np.ndarray:
    """Grid D: +1 in the week VI crosses from <= buy_th to > buy_th ("crosses back above the
    lower line"), -1 in the week it crosses from >= sell_th to < sell_th; 0 otherwise; NaN
    where VI or its previous defined value is undefined."""
    vi = np.asarray(vi, float)
    prev = np.concatenate(([np.nan], vi[:-1]))
    up = (prev <= buy_th) & (vi > buy_th)
    dn = (prev >= sell_th) & (vi < sell_th)
    sig = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    return np.where(np.isfinite(vi) & np.isfinite(prev), sig, np.nan)


def composite_vote(vis: list[np.ndarray], buy_th: float, sell_th: float, k: int) -> np.ndarray:
    """At least k of the indices undervalued -> +1; at least k overvalued -> -1; else 0.
    Defined only when every index is defined. (With 3 indices and k=2, or 2 and k=2, a week
    cannot be both.)"""
    V = np.vstack([np.asarray(v, float) for v in vis])
    defined = np.isfinite(V).all(axis=0)
    nlo = (V <= buy_th).sum(axis=0)
    nhi = (V >= sell_th).sum(axis=0)
    if ((nlo >= k) & (nhi >= k)).any():
        raise AssertionError("a week cannot be both a long and a short composite setup")
    sig = np.where(nlo >= k, 1.0, np.where(nhi >= k, -1.0, 0.0))
    return np.where(defined, sig, np.nan)


def composite_avg(vis: list[np.ndarray]) -> np.ndarray:
    """Mean of the indices; NaN unless all are defined."""
    V = np.vstack([np.asarray(v, float) for v in vis])
    return np.where(np.isfinite(V).all(axis=0), V.mean(axis=0), np.nan)


# ===========================================================================================
# Inputs
# ===========================================================================================
def friday_grid() -> pd.DatetimeIndex:
    return fh.weekly_decision_dates(GRID_START, GRID_END)


def market_inputs(fri: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Valuation inputs of the traded markets as of each Friday (no lag: market closes).
    crude = crude_cl (NYMEX CL front month, what Larry charts); the others = their v1 series."""
    out = {"crude": asof_fresh(fh.load_series("crude_cl"), fri)}
    for mk in ("gold", "silver", "soybean", "sugar"):
        out[mk] = asof_fresh(fh.load_series(mk), fri)
    return out


def comparator_inputs(fri: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Comparators as of each Friday: gold (v1 London spot), the 10y Treasury price proxy
    (DGS10 is a market yield, no lag) and the broad dollar (published Mondays: read as of
    Friday - 7 days)."""
    return {"gold": asof_fresh(fh.load_series("gold"), fri),
            "bond": asof_fresh(fh.load_series("ust10_price"), fri),
            "usd": asof_fresh(fh.load_series("usd_broad"), fri, lag_days=USD_LAG_DAYS)}


def load_psys_gold() -> pd.Series:
    """Grid E: pysystemtrade GOLD held-contract (PRICE) daily price from the scout's tidy
    curve file (1975..2024-03-28). Unadjusted: small steps at contract rolls."""
    df = pd.read_csv(ROOT / "work2" / "scout" / "curve_daily.csv",
                     usecols=["market", "date", "price"], parse_dates=["date"])
    g = df[df["market"] == "gold"].set_index("date")["price"].astype(float)
    g = g[g > 0].sort_index()
    if g.index.duplicated().any():
        raise ValueError("duplicate gold dates in curve_daily.csv")
    return g.rename("gold_psys")


def load_rclc1() -> pd.Series:
    """Grid E: EIA NYMEX WTI contract-1 settlements (RCLC1), positive, stale holiday rows
    removed (1983..2024-04-05)."""
    s = fh.load_ext_raw("RCLC1")
    s = s[s > 0]
    s, _ = fh.drop_stale_holiday_prints(s)
    return s.rename("rclc1")

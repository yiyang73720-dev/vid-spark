#!/usr/bin/env python3
"""
larry_cot -- reconciled engine for the Larry Williams COT-index backtest (SPEC.md v1).

This module reconciles two independent implementations of the spec
(crosscheck/implA/backtest_canonical.py and crosscheck/implB/backtest_b.py). They agree on every
deterministic quantity to within 5e-16. Two things differed:
  * the permutation RNG seed (A=11, B=22): Monte-Carlo noise only. This engine defaults to
    seed 11, so its perm_p equals A's; with --seed 22 it equals B's.
  * the CSV float parsing: pandas' default C parser rounds about 8% of the price strings
    1 ulp away from the correctly rounded double, which is what A did. This engine parses
    with float_precision="round_trip", the same as B's Python float(). Its deterministic
    columns therefore match B exactly and A to within 5e-16.

Importable API (downstream "lens" agents)
-----------------------------------------
    import sys; sys.path.insert(0, "<package>/engine"); import larry_cot as lc

    panel = lc.build_panel()                       # dict market -> per-report DataFrame
    f = panel["crude"]                             # columns: report_date, target_date,
                                                   #   entry_pos, entry_date, entry_price,
                                                   #   comm_net, mm_net, oi, shutdown,
                                                   #   idx26/idx52/idx156, mm_idx26, r1/r4/r8/r13
    st = lc.cell_stats(f, "idx26", 80, 20, "long", "r8")      # one canonical cell (dict)
    canon = lc.canonical_table(panel)                         # the 360-row table
    pooled = lc.pooled_cell(panel, "idx26", 80, 20, "long", "r8")

    # Robustness hooks:
    panel_spot = lc.build_panel(price_overrides={"crude": lc.load_wti_spot()})
    lc.cell_stats(f, "idx26", 80, 20, "long", "r8", include_shutdown=True)
    lc.cell_stats(f, "mm_idx26", 80, 20, "short", "r8")       # speculator index as comparator
    lc.add_forward_returns(f, prices_series, horizons_days=[3, 10])   # adds rd3, rd10

CLI
---
    python3 -I larry_cot.py [--root ROOT] [--out DIR] [--seed 11] [--k 2000] [--no-extras]

ROOT (data in ROOT/data, outputs in ROOT/work/engine) defaults to $LARRY_COT_ROOT, or to the
package directory (the parent of engine/) when that variable is unset.

The CLI writes canonical.csv and crude_timeseries.csv, both in the spec's exact format. Unless
--no-extras is given it also writes pooled.csv (the 72 pooled cells across the 5 markets)
and canonical_exact_p.csv (the canonical keys plus the exhaustive permutation p over every
allowed offset, which has no Monte-Carlo noise). It then runs its self-checks.

Every spec ambiguity resolved here is marked "SPEC-CHOICE".
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Constants (verbatim from SPEC.md)
# ----------------------------------------------------------------------------------------
# Root holding data/, ext/ and work/: $LARRY_COT_ROOT, else the package dir (parent of engine/).
DEFAULT_ROOT = Path(os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[1])

MARKETS = ("crude", "gold", "silver", "soybean", "sugar")      # us10y/us30y ignored
LOOKBACKS = (26, 52, 156)
THRESHOLDS = ((80, 20), (75, 25), (90, 10))                     # (buy_th, sell_th), spec order
SIDES = ("long", "short")
HORIZONS_W = (1, 4, 8, 13)
DAYS_PER_WEEK = 5                                               # exit = 5*h trading days
K_PERM = 2000
MIN_SHIFT = 26                                                  # offsets uniform in [26, T-26]
DEFAULT_SEED = 11                                               # = implementation A

# Shutdown publication delays: report_date inside these windows (inclusive) is not tradable.
SHUTDOWN_WINDOWS = (
    ("2013-10-01", "2013-10-31"),
    ("2018-12-18", "2019-02-26"),
    ("2025-09-30", "2026-01-20"),
)

CANONICAL_COLUMNS = ["market", "lookback", "buy_th", "sell_th", "side", "horizon_w",
                     "n_signal_weeks", "n_episodes", "mean_ret", "median_ret", "hit_rate",
                     "baseline_mean", "excess", "perm_p"]
TIMESERIES_COLUMNS = ["report_date", "entry_date", "comm_net", "mm_net", "oi",
                      "idx26", "idx52", "idx156", "mm_idx26", "entry_price"]


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def load_cot(path: str | Path | None = None, markets: Iterable[str] = MARKETS) -> pd.DataFrame:
    """Load the weekly COT file and return one row per (market, report).

    Columns: market, report_date (Timestamp), oi, comm_net, mm_net (int64), sorted by
    market then report_date.
      comm_net = (prod_merc_long + swap_long) - (prod_merc_short + swap_short)  # Larry's commercials
      mm_net   = m_money_long - m_money_short                                    # managed money
    """
    path = Path(path) if path is not None else DEFAULT_ROOT / "data" / "cot_weekly.csv"
    cot = pd.read_csv(path, dtype={"code": str}, float_precision="round_trip")
    cot = cot[cot["market"].isin(list(markets))].copy()
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    pos_cols = ["oi", "prod_merc_long", "prod_merc_short", "swap_long", "swap_short",
                "m_money_long", "m_money_short"]
    for c in pos_cols:
        # Positions are integer contract counts that some files store as "42587.0".
        v = cot[c].to_numpy(dtype=float)
        if np.isnan(v).any():
            raise ValueError(f"missing COT values in column {c}")
        if np.any(np.abs(v - np.round(v)) > 1e-9):
            raise ValueError(f"non-integer COT values in column {c}")
        cot[c] = np.round(v).astype("int64")
    cot["comm_net"] = (cot["prod_merc_long"] + cot["swap_long"]) - (
        cot["prod_merc_short"] + cot["swap_short"])
    cot["mm_net"] = cot["m_money_long"] - cot["m_money_short"]
    cot = cot.sort_values(["market", "report_date"]).reset_index(drop=True)
    if cot.duplicated(["market", "report_date"]).any():
        raise ValueError("duplicate (market, report_date) rows in COT file")
    return cot[["market", "report_date", "oi", "comm_net", "mm_net"]]


def _clean_close(s: pd.Series) -> pd.Series:
    """Drop missing and non-positive closes (spec: drop any close <= 0 before computing).
    The surviving rows ARE the market's trading-day calendar."""
    s = s[s.notna() & (s > 0)].sort_index()
    if s.index.duplicated().any():
        raise ValueError("duplicate price dates")
    return s.astype(float)


def load_prices(path: str | Path | None = None,
                markets: Iterable[str] = MARKETS) -> dict[str, pd.Series]:
    """Return {market: cleaned close Series indexed by trading date (DatetimeIndex)}.
    crude = NYMEX CL continuous front month; the -37.63 print on 2020-04-20 is removed,
    so 2020-04-20 is NOT a trading day for crude."""
    path = Path(path) if path is not None else DEFAULT_ROOT / "data" / "prices_daily.csv"
    # round_trip = correctly rounded decimal->double (pandas' default parser is not).
    px = pd.read_csv(path, float_precision="round_trip")
    out = {}
    for m in markets:
        g = px[px["market"] == m]
        s = pd.Series(g["close"].to_numpy(dtype=float),
                      index=pd.DatetimeIndex(pd.to_datetime(g["date"]), name="date"),
                      name=m)
        out[m] = _clean_close(s)
    return out


def load_wti_spot(path: str | Path | None = None) -> pd.Series:
    """EIA Cushing WTI spot (Date, Price), cleaned the same way. This is the alternative crude
    series for robustness checks only, not used in the canonical run. It is left as-is apart
    from the cleaning, so the suspicious 2026-09-25 = 85.23 print is still in it."""
    path = Path(path) if path is not None else DEFAULT_ROOT / "data" / "wti-daily.csv"
    w = pd.read_csv(path, float_precision="round_trip")
    s = pd.Series(pd.to_numeric(w["Price"], errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(pd.to_datetime(w["Date"]), name="date"),
                  name="crude_wti_spot")
    return _clean_close(s)


# ----------------------------------------------------------------------------------------
# Larry COT index
# ----------------------------------------------------------------------------------------
def larry_index(values, n: int) -> np.ndarray:
    """Larry Williams index over the last n reports, *including* the current one:
        idx = 100 * (v - min_n) / (max_n - min_n)
    NaN for the first n-1 reports and NaN when max_n == min_n. The window counts reports
    (rows), not calendar weeks."""
    s = pd.Series(np.asarray(values, dtype=float))
    lo = s.rolling(n, min_periods=n).min()
    hi = s.rolling(n, min_periods=n).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng.where(rng != 0)).to_numpy()


def exact_threshold_mask(values, n: int, th: float, ge: bool) -> np.ndarray:
    """Exact-integer version of idx >= th (ge=True) or idx <= th (ge=False) for integer
    inputs: 100*(v-min) >= th*(max-min). Used only as a self-check that float rounding at
    the threshold never changes a signal."""
    v = np.asarray(values, dtype=np.int64)
    s = pd.Series(v)
    lo = s.rolling(n, min_periods=n).min().to_numpy()
    hi = s.rolling(n, min_periods=n).max().to_numpy()
    ok = ~np.isnan(lo) & (hi != lo)
    out = np.zeros(len(v), dtype=bool)
    num = 100 * (v[ok] - lo[ok].astype(np.int64))
    den = hi[ok].astype(np.int64) - lo[ok].astype(np.int64)
    out[ok] = (num >= th * den) if ge else (num <= th * den)
    return out


# ----------------------------------------------------------------------------------------
# Timing: shutdown flag, entry date, forward returns
# ----------------------------------------------------------------------------------------
def shutdown_mask(report_dates, windows=SHUTDOWN_WINDOWS) -> np.ndarray:
    """True where report_date falls in a shutdown window (endpoints inclusive)."""
    d = pd.DatetimeIndex(pd.to_datetime(report_dates))
    out = np.zeros(len(d), dtype=bool)
    for a, b in windows:
        out |= np.asarray((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b)))
    return out


def entry_target(report_dates) -> pd.DatetimeIndex:
    """Tuesday of the week after report_date: report_date - weekday + 1 + 7 days (Mon=0).
    Tuesday report -> +7, Monday -> +8, Wednesday -> +6 days."""
    d = pd.DatetimeIndex(pd.to_datetime(report_dates))
    return d - pd.to_timedelta(d.weekday, unit="D") + pd.Timedelta(days=8)


def entry_positions(report_dates, price_index: pd.DatetimeIndex) -> np.ndarray:
    """Row position in the cleaned price series of the first trading day on or after the
    target Tuesday (holidays roll forward). -1 when the target is past the end of the data."""
    tgt = entry_target(report_dates)
    pos = np.searchsorted(price_index.values, tgt.values, side="left")
    return np.where(pos < len(price_index), pos, -1).astype(np.int64)


def forward_returns(prices: pd.Series, entry_pos: np.ndarray, n_days: int) -> np.ndarray:
    """r = close[entry_pos + n_days] / close[entry_pos] - 1 on the cleaned series. NaN when
    there is no entry or not enough data afterwards (the observation is dropped)."""
    c = prices.to_numpy(dtype=float)
    entry_pos = np.asarray(entry_pos, dtype=np.int64)
    exit_pos = entry_pos + int(n_days)
    ok = (entry_pos >= 0) & (exit_pos < len(c))
    r = np.full(len(entry_pos), np.nan)
    r[ok] = c[exit_pos[ok]] / c[entry_pos[ok]] - 1.0
    return r


def add_forward_returns(frame: pd.DataFrame, prices: pd.Series,
                        horizons_w: Iterable[int] = (),
                        horizons_days: Iterable[int] = ()) -> pd.DataFrame:
    """Add forward-return columns in place: r{h} for h weeks (5*h trading days) and rd{d}
    for d trading days. `frame` must have been built against the same `prices` series,
    because entry_pos indexes into it."""
    for h in horizons_w:
        frame[f"r{h}"] = forward_returns(prices, frame["entry_pos"].to_numpy(),
                                         DAYS_PER_WEEK * h)
    for dd in horizons_days:
        frame[f"rd{dd}"] = forward_returns(prices, frame["entry_pos"].to_numpy(), dd)
    return frame


def build_market_frame(cot_m: pd.DataFrame, prices_m: pd.Series,
                       lookbacks: Iterable[int] = LOOKBACKS,
                       mm_lookbacks: Iterable[int] = (26,),
                       horizons_w: Iterable[int] = HORIZONS_W,
                       horizons_days: Iterable[int] = (),
                       windows=SHUTDOWN_WINDOWS) -> pd.DataFrame:
    """Return one row per report for one market, with indices, entry and forward returns.

    The rolling min/max uses every report, shutdown weeks included (spec: the data exists in
    hindsight). Shutdown weeks are only flagged here; cell_stats excludes them."""
    f = cot_m.sort_values("report_date").reset_index(drop=True).copy()
    for n in lookbacks:
        f[f"idx{n}"] = larry_index(f["comm_net"].to_numpy(), n)
    for n in mm_lookbacks:
        f[f"mm_idx{n}"] = larry_index(f["mm_net"].to_numpy(), n)
    f["shutdown"] = shutdown_mask(f["report_date"], windows)
    f["target_date"] = entry_target(f["report_date"])
    pos = entry_positions(f["report_date"], prices_m.index)
    f["entry_pos"] = pos
    has = pos >= 0
    f["entry_date"] = pd.NaT
    f.loc[has, "entry_date"] = prices_m.index.values[pos[has]]
    f["entry_date"] = pd.to_datetime(f["entry_date"])
    f["entry_price"] = np.where(has, prices_m.to_numpy()[np.where(has, pos, 0)], np.nan)
    add_forward_returns(f, prices_m, horizons_w, horizons_days)
    return f


def build_panel(root: str | Path | None = None, markets: Iterable[str] = MARKETS,
                price_overrides: Mapping[str, pd.Series] | None = None,
                **frame_kwargs) -> dict[str, pd.DataFrame]:
    """Load the data and build {market: per-report frame} for every market.
    price_overrides maps market -> an already-loaded close Series, which replaces that market's
    prices_daily.csv series (e.g. {"crude": load_wti_spot()}). Overrides are cleaned again.
    Extra keyword args go to build_market_frame (lookbacks, horizons_w, horizons_days, ...)."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    markets = list(markets)
    cot = load_cot(root / "data" / "cot_weekly.csv", markets)
    prices = load_prices(root / "data" / "prices_daily.csv", markets)
    for m, s in (price_overrides or {}).items():
        prices[m] = _clean_close(s)
    return {m: build_market_frame(cot[cot["market"] == m], prices[m], **frame_kwargs)
            for m in markets}


# ----------------------------------------------------------------------------------------
# Signals, episodes, eligibility
# ----------------------------------------------------------------------------------------
def signal_masks(idx, buy_th: float, sell_th: float) -> tuple[np.ndarray, np.ndarray]:
    """(long_setup, short_setup) over the full report sequence: long when idx >= buy_th,
    short when idx <= sell_th. NaN idx means no setup."""
    idx = np.asarray(idx, dtype=float)
    ok = ~np.isnan(idx)
    return ok & (idx >= buy_th), ok & (idx <= sell_th)


def setup_mask(idx, side: str, buy_th: float, sell_th: float) -> np.ndarray:
    lo, sh = signal_masks(idx, buy_th, sell_th)
    if side not in SIDES:
        raise ValueError(side)
    return lo if side == "long" else sh


def episode_ids(active) -> np.ndarray:
    """Label maximal runs of consecutive True values 1, 2, ...; 0 where inactive."""
    a = np.asarray(active, dtype=bool)
    start = a & ~np.concatenate(([False], a[:-1]))
    return np.where(a, np.cumsum(start), 0)


def eligible_mask(frame: pd.DataFrame, idx_col: str, ret_col: str,
                  include_shutdown: bool = False) -> np.ndarray:
    """Eligible weeks: idx defined, outside the shutdown windows (unless include_shutdown),
    and a valid forward return in ret_col."""
    e = frame[idx_col].notna().to_numpy() & frame[ret_col].notna().to_numpy()
    if not include_shutdown:
        e &= ~frame["shutdown"].to_numpy(dtype=bool)
    return e


def count_episodes(setup: np.ndarray, elig: np.ndarray, mode: str = "full") -> int:
    """Number of episodes that contain at least one counted signal week (setup & eligible).

    SPEC-CHOICE (both A and B): mode="full" builds the runs on the market's full report
    sequence. Shutdown weeks and weeks without a forward return do NOT break a run; NaN idx
    does, because there is no setup then. Robustness alternatives:
      "strict"   - any non-eligible week breaks a run (runs of setup & eligible),
      "eligible" - runs over the compressed sequence of eligible weeks only.
    """
    setup = np.asarray(setup, dtype=bool)
    elig = np.asarray(elig, dtype=bool)
    counted = setup & elig
    if mode == "full":
        ids = episode_ids(setup)[counted]
    elif mode == "strict":
        ids = episode_ids(counted)[counted]
    elif mode == "eligible":
        s_e = setup[elig]
        ids = episode_ids(s_e)[s_e]
    else:
        raise ValueError(mode)
    return int(len(np.unique(ids)))


# ----------------------------------------------------------------------------------------
# Permutation test (circular shift over the T eligible weeks)
# ----------------------------------------------------------------------------------------
def _shifted_means(sig: np.ndarray, signed: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean signed return over np.roll(sig, k) for each offset k. The signal at eligible
    position i moves to (i + k) mod T."""
    T = len(sig)
    pos = np.flatnonzero(sig)
    return signed[(pos[None, :] + np.asarray(offsets)[:, None]) % T].mean(axis=1)


def perm_offsets(T: int, k: int = K_PERM, seed: int = DEFAULT_SEED,
                 min_shift: int = MIN_SHIFT) -> np.ndarray:
    """SPEC-CHOICE (A and B): a fresh np.random.default_rng(seed) per cell, so results do
    not depend on loop order. Offsets are integers uniform on the CLOSED interval
    [min_shift, T-min_shift]."""
    rng = np.random.default_rng(seed)
    return rng.integers(min_shift, T - min_shift, size=k, endpoint=True)


def perm_pvalue(sig, signed, observed: float | None = None, k: int = K_PERM,
                seed: int = DEFAULT_SEED, min_shift: int = MIN_SHIFT) -> float:
    """One-sided Monte-Carlo p: (1 + #{shifted mean >= observed}) / (1 + k).
    sig and signed are aligned on the T eligible weeks in chronological order."""
    sig = np.asarray(sig, dtype=bool)
    signed = np.asarray(signed, dtype=float)
    if not sig.any():
        return float("nan")
    if observed is None:
        observed = float(signed[sig].mean())
    sm = _shifted_means(sig, signed, perm_offsets(len(sig), k, seed, min_shift))
    return (1 + int(np.sum(sm >= observed))) / (1 + k)


def perm_pvalue_exact(sig, signed, observed: float | None = None,
                      min_shift: int = MIN_SHIFT) -> float:
    """Exhaustive version without Monte-Carlo noise: the share of ALL allowed offsets
    k in [min_shift, T-min_shift] whose shifted mean is >= observed. This is the limit of
    perm_pvalue as k grows. A diagnostic only, not a spec column."""
    sig = np.asarray(sig, dtype=bool)
    signed = np.asarray(signed, dtype=float)
    if not sig.any():
        return float("nan")
    if observed is None:
        observed = float(signed[sig].mean())
    offs = np.arange(min_shift, len(sig) - min_shift + 1)
    return float(np.mean(_shifted_means(sig, signed, offs) >= observed))


# ----------------------------------------------------------------------------------------
# One cell, the canonical table, pooled cells
# ----------------------------------------------------------------------------------------
def cell_arrays(frame: pd.DataFrame, idx_col: str, buy_th: float, sell_th: float,
                side: str, ret_col: str, include_shutdown: bool = False) -> dict:
    """Return the aligned arrays of one cell: the eligible mask (full sequence), the setup
    (full sequence), and on the T eligible weeks the signal, raw r and signed r."""
    sgn = 1.0 if side == "long" else -1.0
    setup = setup_mask(frame[idx_col].to_numpy(), side, buy_th, sell_th)
    elig = eligible_mask(frame, idx_col, ret_col, include_shutdown)
    r_e = frame[ret_col].to_numpy(dtype=float)[elig]
    return dict(sgn=sgn, setup=setup, elig=elig, sig=setup[elig], r=r_e, signed=sgn * r_e)


def cell_stats(frame: pd.DataFrame, idx_col: str, buy_th: float, sell_th: float,
               side: str, ret_col: str, *, include_shutdown: bool = False,
               k: int = K_PERM, seed: int = DEFAULT_SEED, min_shift: int = MIN_SHIFT,
               episode_mode: str = "full", exact_p: bool = False,
               do_perm: bool = True) -> dict:
    """Statistics of one (market, index, threshold pair, side, horizon) cell.

    mean_ret / median_ret / hit_rate use SIGNED returns (+r long, -r short); a hit is a
    signed return strictly > 0. baseline_mean is the UNSIGNED mean raw r over all eligible
    weeks, so it is the same for long and short. excess = mean_ret - sgn * baseline_mean.
    """
    a = cell_arrays(frame, idx_col, buy_th, sell_th, side, ret_col, include_shutdown)
    sig, signed, sgn = a["sig"], a["signed"], a["sgn"]
    baseline = float(a["r"].mean()) if len(a["r"]) else float("nan")
    sr = signed[sig]
    n = int(sig.sum())
    out = dict(n_signal_weeks=n,
               n_episodes=count_episodes(a["setup"], a["elig"], episode_mode),
               mean_ret=np.nan, median_ret=np.nan, hit_rate=np.nan,
               baseline_mean=baseline, excess=np.nan, perm_p=np.nan, T=int(len(sig)))
    if n > 0:
        mean_ret = float(sr.mean())
        out.update(mean_ret=mean_ret, median_ret=float(np.median(sr)),
                   hit_rate=float((sr > 0).mean()), excess=mean_ret - sgn * baseline)
        if do_perm:
            out["perm_p"] = perm_pvalue(sig, signed, mean_ret, k, seed, min_shift)
        if exact_p:
            out["perm_p_exact"] = perm_pvalue_exact(sig, signed, mean_ret, min_shift)
    elif exact_p:
        out["perm_p_exact"] = np.nan
    return out


def canonical_table(panel: Mapping[str, pd.DataFrame], markets: Iterable[str] = MARKETS,
                    lookbacks: Iterable[int] = LOOKBACKS, thresholds=THRESHOLDS,
                    horizons_w: Iterable[int] = HORIZONS_W, *, k: int = K_PERM,
                    seed: int = DEFAULT_SEED, exact_p: bool = False,
                    include_shutdown: bool = False, episode_mode: str = "full",
                    idx_prefix: str = "idx") -> pd.DataFrame:
    """The spec's canonical table in spec row order: market, then lookback, then threshold
    pair, then side, then horizon. With exact_p=True an extra perm_p_exact column is added.
    idx_prefix="mm_idx" runs the same grid on the managed-money index, if those columns were
    built (mm_lookbacks)."""
    rows = []
    for m in markets:
        f = panel[m]
        for n in lookbacks:
            for bt, st in thresholds:
                for sd in SIDES:
                    for h in horizons_w:
                        s = cell_stats(f, f"{idx_prefix}{n}", bt, st, sd, f"r{h}",
                                       include_shutdown=include_shutdown, k=k, seed=seed,
                                       episode_mode=episode_mode, exact_p=exact_p)
                        s.pop("T")
                        rows.append(dict(market=m, lookback=n, buy_th=bt, sell_th=st,
                                         side=sd, horizon_w=h, **s))
    cols = CANONICAL_COLUMNS + (["perm_p_exact"] if exact_p else [])
    return pd.DataFrame(rows)[cols]


def pooled_cell(panel: Mapping[str, pd.DataFrame], idx_col: str, buy_th: float,
                sell_th: float, side: str, ret_col: str, markets: Iterable[str] = MARKETS,
                *, k: int = K_PERM, seed: int = DEFAULT_SEED, min_shift: int = MIN_SHIFT,
                include_shutdown: bool = False, episode_mode: str = "full") -> dict:
    """The same cell pooled across markets (not part of the canonical file).

    The pooled mean, median and hit rate are over all signal weeks of all markets, each
    market-week weighted equally. excess_vs_own_baseline is the mean of (signed r minus the
    signed baseline of that week's own market). n_episodes is the sum over markets.
    Permutation: one default_rng(seed) per pooled cell. Each market gets its own independent
    offset vector, drawn in `markets` order, and the statistic is the pooled mean signed
    return. A and B used exactly this scheme."""
    markets = list(markets)
    parts, epis = [], 0
    for m in markets:
        a = cell_arrays(panel[m], idx_col, buy_th, sell_th, side, ret_col, include_shutdown)
        a["baseline"] = float(a["r"].mean())
        parts.append(a)
        epis += count_episodes(a["setup"], a["elig"], episode_mode)
    sgn = 1.0 if side == "long" else -1.0
    all_sr = np.concatenate([p["signed"][p["sig"]] for p in parts])
    all_ex = np.concatenate([p["signed"][p["sig"]] - sgn * p["baseline"] for p in parts])
    mkt_ex = {m: float(p["signed"][p["sig"]].mean() - sgn * p["baseline"])
              for m, p in zip(markets, parts) if p["sig"].any()}
    n_tot = int(len(all_sr))
    out = dict(n_signal_weeks=n_tot, n_episodes=int(epis))
    if n_tot == 0:
        return {**out, "mean_ret": np.nan, "median_ret": np.nan, "hit_rate": np.nan,
                "excess_vs_own_baseline": np.nan, "equal_weight_mkt_excess": np.nan,
                "n_mkts_positive_excess": 0, "perm_p": np.nan}
    obs = float(all_sr.mean())
    rng = np.random.default_rng(seed)
    sums = np.zeros(k)
    for p in parts:
        T = len(p["sig"])
        offs = rng.integers(min_shift, T - min_shift, size=k, endpoint=True)
        pos = np.flatnonzero(p["sig"])
        if len(pos):
            sums += p["signed"][(pos[None, :] + offs[:, None]) % T].sum(axis=1)
    pval = (1 + int(np.sum(sums / n_tot >= obs))) / (1 + k)
    return {**out, "mean_ret": obs, "median_ret": float(np.median(all_sr)),
            "hit_rate": float((all_sr > 0).mean()),
            "excess_vs_own_baseline": float(all_ex.mean()),
            "equal_weight_mkt_excess": float(np.mean(list(mkt_ex.values()))),
            "n_mkts_positive_excess": int(sum(v > 0 for v in mkt_ex.values())),
            "perm_p": pval}


def pooled_table(panel: Mapping[str, pd.DataFrame], lookbacks=LOOKBACKS,
                 thresholds=THRESHOLDS, horizons_w=HORIZONS_W, *, k: int = K_PERM,
                 seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Every pooled cell (lookback x threshold x side x horizon = 72 rows)."""
    rows = []
    for n in lookbacks:
        for bt, st in thresholds:
            for sd in SIDES:
                for h in horizons_w:
                    rows.append(dict(lookback=n, buy_th=bt, sell_th=st, side=sd, horizon_w=h,
                                     **pooled_cell(panel, f"idx{n}", bt, st, sd, f"r{h}",
                                                   k=k, seed=seed)))
    return pd.DataFrame(rows)


def crude_timeseries(panel_or_frame) -> pd.DataFrame:
    """The spec's crude_timeseries.csv content: one row per crude report, shutdown weeks
    included. Dates are YYYY-MM-DD. entry_date and entry_price are empty when the target is
    past the end of the prices (only the final report, 2026-09-29)."""
    f = panel_or_frame["crude"] if isinstance(panel_or_frame, Mapping) else panel_or_frame
    f = f.sort_values("report_date")
    return pd.DataFrame({
        "report_date": f["report_date"].dt.strftime("%Y-%m-%d"),
        "entry_date": f["entry_date"].dt.strftime("%Y-%m-%d"),
        "comm_net": f["comm_net"].astype("int64"),
        "mm_net": f["mm_net"].astype("int64"),
        "oi": f["oi"].astype("int64"),
        "idx26": f["idx26"], "idx52": f["idx52"], "idx156": f["idx156"],
        "mm_idx26": f["mm_idx26"],
        "entry_price": f["entry_price"],
    })[TIMESERIES_COLUMNS].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------------------------
def self_checks(panel: Mapping[str, pd.DataFrame], canon: pd.DataFrame) -> list[str]:
    """Assert the invariants of the spec and return a list of human-readable facts."""
    facts = []
    assert len(canon) == 360, len(canon)
    assert not canon[CANONICAL_COLUMNS].isna().any().any(), "NaN in canonical table"
    nb = canon.groupby(["market", "lookback", "horizon_w"])["baseline_mean"].nunique()
    assert (nb == 1).all(), "baseline must not depend on side/threshold"
    for m, f in panel.items():
        for c in [c for c in f.columns if c.startswith(("idx", "mm_idx"))]:
            v = f[c].dropna()
            assert ((v >= 0) & (v <= 100)).all(), (m, c)
        has = f["entry_pos"].to_numpy() >= 0
        ed, rd = f.loc[has, "entry_date"], f.loc[has, "report_date"]
        assert (ed > rd + pd.Timedelta(days=5)).all(), m          # never before publication
        assert (ed >= f.loc[has, "target_date"]).all(), m
        gap = (ed - rd).dt.days
        facts.append(f"{m}: entry gap days min/max {gap.min()}/{gap.max()}, "
                     f"no entry for {int((~has).sum())} report(s), "
                     f"shutdown reports {int(f['shutdown'].sum())}")
        # Float threshold comparison == exact-integer comparison (no rounding flips).
        for n in LOOKBACKS:
            for bt, st in THRESHOLDS:
                lo, sh = signal_masks(f[f"idx{n}"].to_numpy(), bt, st)
                assert (lo == exact_threshold_mask(f["comm_net"].to_numpy(), n, bt, True)).all()
                assert (sh == exact_threshold_mask(f["comm_net"].to_numpy(), n, st, False)).all()
    facts.append("float vs exact-integer threshold comparison: identical for all cells")
    return facts


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="root dir ROOT (default: $LARRY_COT_ROOT or package dir)")
    ap.add_argument("--out", default=None, help="output dir (default: ROOT/work/engine)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--k", type=int, default=K_PERM)
    ap.add_argument("--no-extras", action="store_true",
                    help="write only canonical.csv and crude_timeseries.csv")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "work" / "engine"
    out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(root)
    canon = canonical_table(panel, k=args.k, seed=args.seed, exact_p=not args.no_extras)
    canon[CANONICAL_COLUMNS].to_csv(out / "canonical.csv", index=False)
    crude_timeseries(panel).to_csv(out / "crude_timeseries.csv", index=False)
    if not args.no_extras:
        canon.to_csv(out / "canonical_exact_p.csv", index=False)
        pooled_table(panel, k=args.k, seed=args.seed).to_csv(out / "pooled.csv", index=False)

    facts = self_checks(panel, canon)
    head = canon[(canon.market == "crude") & (canon.lookback == 26) & (canon.buy_th == 80)
                 & (canon.side == "long") & (canon.horizon_w == 8)].iloc[0]
    print("\n".join(facts))
    print(f"wrote {out}/canonical.csv ({len(canon)} rows), crude_timeseries.csv")
    print("headline crude N=26 80/20 long h=8:",
          {c: head[c] for c in CANONICAL_COLUMNS[6:] + (["perm_p_exact"]
                                                       if "perm_p_exact" in canon else [])})
    return 0


if __name__ == "__main__":
    sys.exit(main())

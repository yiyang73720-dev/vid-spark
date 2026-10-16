#!/usr/bin/env python3
"""
seas_data -- data module of the Larry Williams SEASONALITY test (fundamentals battery, SPEC2).

What it adds on top of the shared harness (fund_harness):

1. Roll-adjusted ("tradable") futures excess-return indices for heating oil, gasoline, gold,
   silver, soybean and sugar, built from the pysystemtrade multiple-prices files
   (S/ext2/robcarver17_pysystemtrade/data/futures/multiple_prices_csv/, pinned commit
   bbe29e19dab73271570c94b4ee7e2f182d4b9ef1). Third-party files are PARSED ONLY (pandas.read_csv);
   nothing from S/ext2 is executed. Run every script that imports this module with `python3 -I`.

   Why these are needed: a seasonal estimator learns whatever repeats at the same calendar date
   every year. Unadjusted continuous futures series (v1 soybean "ZS00Y", v1 sugar "SB=F", v1 crude
   "CL00Y") jump on roll days, and roll days sit at (almost) the same calendar dates every year,
   so the size of the old/new-contract spread becomes a fake "seasonal" pattern -- learned from
   the series and then "confirmed" on the same series. Spot energy series have a similar issue
   (NY Harbor gasoline spot switches RVP grade with the season; spot seasonality is priced into
   the futures curve and cannot be captured by a futures position). A contract-consistent
   (roll-adjusted) futures index is free of both and is a return a futures trader can earn --
   but for the contract it holds, which is NOT always the front month (see PSYS_CAVEATS).

   Construction (same rule as fund_harness.build_crude_roll_index, generalised): one row per
   weekday date = the LAST row of that date with a PRICE print (in the hourly era from 2008-2013
   on, that is the 23:00 "daily close" row, which has all three legs). The index holds the
   pysystemtrade PRICE contract. Between two consecutive rows t' < t:
       same PRICE contract:     r = PRICE(t) / PRICE(t') - 1
       PRICE contract changed:  r = PRICE(t) / P_new(t') - 1,
   where P_new(t') is the price at t' of the NEW contract, read from the CARRY or FORWARD leg of
   row t' (its contract id must match). If the new contract has no print at t' (`roll_bridged`):
   (a) if the OLD contract still has a print at t, the index stays in it for that day and
   switches at t's close; (b) else the switch is made at the latest earlier row (<= 10 rows back)
   where both contracts have prints; (c) else the day's return is set to 0 (`n_roll_failed`).
   No return is ever taken across two different contracts.
   No collateral/financing return is added (an EXCESS-return index, like crude_roll).

   Caveats (reported in results.json): pysystemtrade holds a deferred November soybean contract
   (rolls once a year) rather than the front month; HEATOIL, GASOILINE and SUGAR11 roll 60 days
   before expiry (csvconfig/rollconfig.csv RollOffsetDays = -60), so those indices hold a
   deferred contract too; prices after the hourly era starts are not always exchange
   settlements and are partly one trading day out of step with EIA settlements; all files end
   2024-03-28. [Added by the adversarial verification, see verify/.]
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
PSYS_DIR = ROOT / "ext2" / "robcarver17_pysystemtrade" / "data" / "futures" / "multiple_prices_csv"
PSYS_COMMIT = "bbe29e19dab73271570c94b4ee7e2f182d4b9ef1"

# market key -> pysystemtrade instrument file (GASOILINE is pysystemtrade's name for NYMEX
# gasoline: unleaded HU before the 2006 switch, RBOB after it).
PSYS_FILES = {
    "crude": "CRUDE_W",       # cross-check only; crude uses the EIA-based fh "crude_roll"
    "heatoil": "HEATOIL",
    "gasoline": "GASOILINE",
    "gold": "GOLD",
    "silver": "SILVER",
    "soybean": "SOYBEAN",
    "sugar": "SUGAR11",
}


# Caveats on these indices, reported in results.json (added by the adversarial verification;
# the numbers behind them are in verify/verify_roll_proxy.json).
PSYS_CAVEATS = [
    "The pysystemtrade-based roll-adjusted indices hold pysystemtrade's PRICE contract. Per "
    "csvconfig/rollconfig.csv that contract is rolled 60 days before expiry for HEATOIL, "
    "GASOILINE and SUGAR11 (RollOffsetDays = -60), so these indices hold a DEFERRED contract, "
    "not the front month; SOYBEAN holds November only. A deferred contract prices most of the "
    "season in advance, so the tradability check is a test of deferred-contract seasonality. "
    "With settlement-based front-month proxies (EIA HO_F1 with month-start roll days zeroed; v1 "
    "SB with Mar/May/Jul/Oct roll days zeroed) heating oil alone shows a positive, not "
    "multiplicity-robust result, while the pooled tradability cell stays null "
    "(results.json: verification.tradability_front_month_proxies).",
    "After about 2008 the pysystemtrade daily rows are partly one trading day out of step with "
    "EIA settlements (their daily returns correlate with EIA contract 1 of the PREVIOUS day in "
    "2013-2017). This adds noise to 8-week returns; it creates no look-ahead because the "
    "seasonal signal never uses recent prices.",
]


@functools.lru_cache(maxsize=None)
def psys_daily(instrument: str) -> pd.DataFrame:
    """One row per weekday date: the last row of that date that has a PRICE print.
    Columns: price, price_contract, carry, carry_contract, forward, forward_contract."""
    df = pd.read_csv(PSYS_DIR / f"{instrument}.csv")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["date"] = df["DATETIME"].dt.normalize()
    df = df[(df["date"].dt.weekday < 5) & (df["PRICE"] > 0)]
    df = df.sort_values("DATETIME", kind="mergesort").groupby("date").tail(1)
    out = pd.DataFrame({
        "price": df["PRICE"].to_numpy(float),
        "price_contract": df["PRICE_CONTRACT"].to_numpy("int64"),
        "carry": df["CARRY"].to_numpy(float),
        "carry_contract": df["CARRY_CONTRACT"].to_numpy("int64"),
        "forward": df["FORWARD"].to_numpy(float),
        "forward_contract": df["FORWARD_CONTRACT"].to_numpy("int64"),
    }, index=pd.DatetimeIndex(df["date"], name="date"))
    if out.index.duplicated().any():
        raise AssertionError(f"{instrument}: duplicate dates after reduction")
    return out


def _price_of(row: pd.Series, contract: int) -> float:
    """Price of `contract` in a daily row (from whichever leg holds it), NaN if absent."""
    for leg in ("price", "carry", "forward"):
        if int(row[f"{leg}_contract"]) == contract and row[leg] > 0:
            return float(row[leg])
    return float("nan")


@functools.lru_cache(maxsize=None)
def psys_roll_index(market: str) -> dict:
    """Contract-consistent excess-return index of one market (see module docstring).
    Returns dict(level=Series (100 at start), detail=DataFrame, info=dict)."""
    d = psys_daily(PSYS_FILES[market])
    pc = d["price_contract"].to_numpy()
    p = d["price"].to_numpy()
    n = len(d)
    ret = np.zeros(n)
    roll = np.zeros(n, bool)
    bridged = np.zeros(n, bool)
    failed = 0
    for i in range(1, n):
        if pc[i] == pc[i - 1]:
            ret[i] = p[i] / p[i - 1] - 1.0
            continue
        roll[i] = True
        new = int(pc[i])
        pn = _price_of(d.iloc[i - 1], new)
        if np.isfinite(pn):
            ret[i] = p[i] / pn - 1.0
            continue
        old = int(pc[i - 1])
        # New contract not quoted at t'. (a) If the OLD contract is still quoted at t, stay in it
        # for this day and switch at t's close (the next return is then same-contract).
        po = _price_of(d.iloc[i], old)
        if np.isfinite(po):
            ret[i] = po / p[i - 1] - 1.0
            bridged[i] = True
            continue
        # (b) Otherwise switch at the latest earlier row j where both the old (held) and the new
        # contract have prints: undo the old contract's return from j to i-1 (already in the
        # index) and apply the new contract's return from j to i.
        for j in range(i - 2, max(i - 12, 0), -1):
            pn_j, po_j = _price_of(d.iloc[j], new), _price_of(d.iloc[j], old)
            if np.isfinite(pn_j) and np.isfinite(po_j):
                ret[i] = (p[i] / pn_j) / (p[i - 1] / po_j) - 1.0
                bridged[i] = True
                break
        else:
            failed += 1          # no common quote within 10 rows: flat return on that day
    level = pd.Series(100.0 * np.cumprod(1.0 + ret), index=d.index, name=f"{market}_psys_roll")
    naive = np.zeros(n)
    naive[1:] = p[1:] / p[:-1] - 1.0
    detail = pd.DataFrame(dict(price=p, price_contract=pc, ret=ret, naive_ret=naive, roll=roll,
                               roll_bridged=bridged), index=d.index)
    months = sorted({int(c) // 100 % 100 for c in pc})
    info = dict(market=market, instrument=PSYS_FILES[market], first=d.index[0].date().isoformat(),
                last=d.index[-1].date().isoformat(), n_days=int(n), n_rolls=int(roll.sum()),
                n_roll_bridged=int(bridged.sum()), n_roll_failed=int(failed),
                held_contract_months=months)
    if not (level > 0).all():
        raise AssertionError(f"{market}: index went non-positive")
    return dict(level=level, detail=detail, info=info)


def validate_psys_index(market: str) -> dict:
    """Evidence that the roll adjustment removes the roll jumps: on roll days the unadjusted
    return contains the spread (large), the adjusted one looks like any other day."""
    ix = psys_roll_index(market)
    det = ix["detail"].iloc[1:]
    rd, od = det[det["roll"]], det[~det["roll"]]

    def _st(x):
        x = np.abs(np.asarray(x, float))
        return dict(n=int(len(x)), mean_abs=float(x.mean()) if len(x) else None,
                    p95_abs=float(np.quantile(x, 0.95)) if len(x) else None,
                    max_abs=float(x.max()) if len(x) else None)
    return dict(**ix["info"], roll_day_adjusted=_st(rd["ret"]), roll_day_unadjusted=_st(rd["naive_ret"]),
                other_days=_st(od["ret"]),
                total_return_index=float(ix["level"].iloc[-1] / ix["level"].iloc[0] - 1.0),
                total_return_unadjusted_price=float(det["price"].iloc[-1] / ix["detail"]["price"].iloc[0] - 1.0))


def tradable_series(market: str) -> pd.Series:
    """The roll-adjusted futures series used for the tradability check, by market key."""
    if market == "crude":
        import fund_harness as fh            # EIA RCLC1/RCLC2 based, SPEC2 rule 2a
        return fh.load_series("crude_roll")
    return psys_roll_index(market)["level"]

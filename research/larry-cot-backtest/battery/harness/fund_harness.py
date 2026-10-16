#!/usr/bin/env python3
"""
fund_harness -- shared, look-ahead-free evaluation harness for the Larry Williams
"fundamentals battery" (SPEC2.md, rules 1-6), for ARBITRARY weekly signals.

Every filter agent of the battery imports this module, builds a weekly *signal frame*
(one row per decision week, +1 = long setup, -1 = short setup, 0 = none) and hands it to the
same evaluation code, so all tools are measured with one ruler:

  * loaders for every traded series of SPEC2 rule 2 (incl. a roll-adjusted crude front-month
    excess-return index built here from the EIA NYMEX contract-1/contract-2 prints),
  * entry-date rules for price-based, COT-based and EIA-inventory-based signals (rule 1),
  * forward returns, baseline, excess, hit rate, episodes and the circular-shift permutation
    test, Monte-Carlo and exhaustive (rules 3-4),
  * split-half and walk-forward helpers (rule 5),
  * the S1/S2 trading simulation with costs, buy & hold and a random-timing null (rule 6).

The COT machinery of the v1 study is REUSED, not re-implemented: the reconciled engine
`engine/larry_cot.py` (permutation test, episodes, COT timing, v1 price loaders) and the
strategy library `lenses/strategy/strat_lib.py` (S1/S2 trade logic, P&L, metrics). Feeding
the v1 COT idx26 >= 80 long signal for crude through `evaluate` reproduces the v1 headline
bit-for-bit (see selftest_harness.py).

Quick start
-----------
    import sys; sys.path.insert(0, "<package>/battery/harness"); import fund_harness as fh

    px = fh.load_series("crude_roll")                 # any name in fh.SERIES
    fri = fh.weekly_decision_dates("1990-01-05", "2024-03-29")
    sig = ...                                         # your rule: +1 / 0 / -1 per Friday (NaN = undefined)
    sf = fh.make_signal_frame(fri, sig)               # valid = sig is not NaN
    rows = fh.evaluate(sf, "crude_roll", timing="price", horizons=(1, 4, 8, 13),
                       labels=dict(filter="curve", variant="primary", market="crude",
                                   is_primary=True))
    halves = fh.split_half(sf, "crude_roll", timing="price", horizon=8, sides=("long",))
    strat = fh.simulate(sf, "crude_roll", timing="price")        # S1/S2, LS + LO, null p
    print(strat["metrics"])

Conventions (all choices are marked "HARNESS-CHOICE")
------------------------------------------------------
* A *series* is a pandas Series of positive closes on a DatetimeIndex; its rows ARE the
  trading-day calendar of that series (forward returns count rows, as in v1).
* timing="price": the signal is computed from the close of the decision date (normally a
  Friday); entry = close of the first trading day STRICTLY AFTER the decision date.
  timing="cot":   decision_date = COT report_date; entry = first trading day on/after the
  Tuesday of the following week (v1 rule, lc.entry_target).
  timing="eia":   decision_date = EIA week-ending Friday; data are released the following
  Wednesday 10:30 ET (Thursday in holiday weeks); entry = first trading day on/after the
  Thursday = week_ending + 6 days.
  HARNESS-CHOICE: if the first available close is more than MAX_ENTRY_LAG_DAYS calendar days
  after the target (series not yet started / long data gap) the week has no entry.
* Forward return for h weeks = close[entry + 5h] / close[entry] - 1 on the traded series.
* A week is ELIGIBLE (counts for the baseline and for T) when its signal statistic is defined
  (valid), it is not excluded (e.g. COT shutdown weeks) and it has a forward return.
* Episodes are maximal runs of the same setup over the FULL weekly sequence (v1 "full" mode):
  excluded weeks and weeks without a forward return do not break a run, undefined weeks do.
"""
from __future__ import annotations

import bisect
import functools
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True     # never write .pyc files next to the code

# ----------------------------------------------------------------------------------------
# Paths and reused v1 code
# ----------------------------------------------------------------------------------------
# Code: <package>/engine and <package>/lenses/strategy (v1), <package>/battery (this battery).
# Data and outputs: ROOT = $LARRY_COT_ROOT, else the package dir. ROOT/data and ROOT/ext are
# the v1 inputs, ROOT/ext2 the battery inputs, ROOT/work the v1 outputs, ROOT/work2 ours.
PKG = Path(__file__).resolve().parents[2]                # package dir (battery/harness/../..)
BATTERY = PKG / "battery"                                # battery code
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
WORK_V1 = ROOT / "work"                                  # v1 outputs (engine/, lens_strategy/)
WORK2 = ROOT / "work2"                                   # battery outputs, one dir per step
EXT_SERIES_DIR = ROOT / "ext" / "accomplish999_crude-oil" / "data" / "series"

for _p in (PKG / "engine", PKG / "lenses" / "strategy"):     # v1 code (larry_cot, strat_lib)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import larry_cot as lc   # noqa: E402  reconciled v1 COT engine (permutation, episodes, timing)
import strat_lib as sl   # noqa: E402  v1 strategy primitives (S1/S2, P&L, metrics)

# ----------------------------------------------------------------------------------------
# Constants (SPEC2 rules 3, 4, 6 -- identical to v1)
# ----------------------------------------------------------------------------------------
HORIZONS_W = lc.HORIZONS_W            # (1, 4, 8, 13) weeks
DAYS_PER_WEEK = lc.DAYS_PER_WEEK      # exit = 5*h trading days after entry
K_PERM = lc.K_PERM                    # 2000 Monte-Carlo circular shifts
MIN_SHIFT = lc.MIN_SHIFT              # offsets uniform on [26, T-26]
SEED = lc.DEFAULT_SEED                # 11, as in v1
COST = sl.COST                        # 0.05% per side per unit of position change
HOLD = sl.HOLD                        # 40 trading days
TRIG_WIN = sl.TRIG_WIN                # S2: wait up to 20 trading days
SMA_N = sl.SMA_N                      # S2: 20-day SMA of closes
PRIMARY_H = 8                         # SPEC2 rule 7: primary horizon = 8 weeks
MAX_ENTRY_LAG_DAYS = 7                # HARNESS-CHOICE, see module docstring
TIMINGS = ("price", "cot", "eia")
SIDES = ("long", "short", "net")      # "net" = long and short setups pooled, signed by setup

TEST_COLUMNS = ["filter", "variant", "market", "series", "timing", "side", "horizon_w", "n",
                "episodes", "mean", "median", "baseline", "excess", "hit", "perm_p",
                "perm_p_exact", "perm_p_exact_raw", "T", "period", "is_primary"]


# ========================================================================================
# 1. Raw third-party series (parse only)
# ========================================================================================
def load_ext_raw(series_id: str) -> pd.Series:
    """Parse one series of the third-party accomplish999 archive (JSON, values are decimal
    strings). Only json.load is used; nothing from S/ext is executed. Returns a float Series on
    a sorted DatetimeIndex named 'date', missing values dropped. float(str) is correctly
    rounded, i.e. the same as pandas' float_precision="round_trip" used by the v1 engine."""
    path = EXT_SERIES_DIR / f"{series_id}.json"
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    obs = doc["observations"]

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    s = pd.Series([_num(o["value"]) for o in obs],
                  index=pd.DatetimeIndex(pd.to_datetime([o["date"] for o in obs]), name="date"),
                  name=series_id).sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"{series_id}: duplicate dates")
    return s.dropna()


def ext_catalog() -> pd.DataFrame:
    """The archive's own catalog (ids, names, units, coverage) for reference."""
    return pd.read_csv(ROOT / "ext" / "accomplish999_crude-oil" / "public" / "downloads"
                       / "catalog.csv")


# ========================================================================================
# 2. US exchange holidays (used for the NYMEX business-day calendar and stale-print cleanup)
# ========================================================================================
# Unscheduled full-day closures of the New York futures/stock exchanges in the data period.
SPECIAL_CLOSURES = pd.DatetimeIndex(pd.to_datetime([
    "1985-09-27",                                   # Hurricane Gloria
    "1994-04-27",                                   # President Nixon's funeral
    "1996-01-08",                                   # blizzard (NYMEX closed)
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",   # September 11
    "2004-06-11",                                   # President Reagan's funeral
    "2007-01-02",                                   # President Ford's funeral
    "2012-10-29", "2012-10-30",                     # Hurricane Sandy
    "2018-12-05",                                   # President G.H.W. Bush's funeral
]))


def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous Gregorian / Meeus algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: pd.Timestamp, saturday_to_friday: bool = True) -> pd.Timestamp | None:
    if d.weekday() == 5:
        return d - pd.Timedelta(days=1) if saturday_to_friday else None
    if d.weekday() == 6:
        return d + pd.Timedelta(days=1)
    return d


def us_exchange_holidays(first_year: int, last_year: int) -> pd.DatetimeIndex:
    """Rule-based full-day holidays of the New York exchanges (NYSE/NYMEX schedule):
    New Year (no Friday make-up when Jan 1 is a Saturday), MLK day (from 1998), Presidents day,
    Good Friday, Memorial day, Juneteenth (from 2022), Independence day, Labor day,
    Thanksgiving, Christmas -- plus SPECIAL_CLOSURES. Weekend holidays move to the observed
    weekday. This is a reference list only: where a data source has a genuine (non-stale) print
    on one of these days, the data win (e.g. NYMEX traded on MLK day before 1998)."""
    out = set()
    for y in range(first_year, last_year + 1):
        days = [_observed(pd.Timestamp(y, 1, 1), saturday_to_friday=False),
                _nth_weekday(y, 2, 0, 3), _easter(y) - pd.Timedelta(days=2),
                _last_weekday(y, 5, 0), _observed(pd.Timestamp(y, 7, 4)),
                _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4),
                _observed(pd.Timestamp(y, 12, 25))]
        if y >= 1998:
            days.append(_nth_weekday(y, 1, 0, 3))
        if y >= 2022:
            days.append(_observed(pd.Timestamp(y, 6, 19)))
        out.update(d for d in days if d is not None)
    out.update(d for d in SPECIAL_CLOSURES if first_year <= d.year <= last_year)
    return pd.DatetimeIndex(sorted(out), name="date")


def drop_stale_holiday_prints(s: pd.Series) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Remove placeholder prints: a value on an exchange holiday that merely repeats the
    previous print (some EIA files carry the last value forward over holidays). Entering at such
    a 'close' would really be entering at the previous day's price. Returns (clean, dropped)."""
    hol = us_exchange_holidays(s.index[0].year, s.index[-1].year)
    stale = s.index.isin(hol) & (s.to_numpy() == s.shift(1).to_numpy())
    return s[~stale], s.index[stale]


# ========================================================================================
# 3. Crude: roll-adjusted front-month excess-return index (SPEC2 rule 2a)
# ========================================================================================
@dataclass
class RollIndex:
    """Result of build_crude_roll_index.

    level  -- index level, 100 on the first day (pd.Series, the traded series)
    ret    -- daily index returns (first day 0)
    ltd    -- contract-1 last trading days used (DatetimeIndex)
    detail -- one row per index day: c1, c2 (EIA prints), held (contract held INTO the next
              day, set at this close: 1 or 2), roll (contract 1 changed identity since the
              previous index day), ret, naive_ret (= c1(t)/c1(t-1) - 1, the unadjusted
              continuous return), spread_prev (= c2(t-1)/c1(t-1) - 1), roll_gap (identity
              change crossed while the switch day itself was missing from the data)
    info   -- calendar facts: stale prints removed, EIA gap days, adjudicated months
    """
    level: pd.Series
    ret: pd.Series
    ltd: pd.DatetimeIndex
    detail: pd.DataFrame
    info: dict = field(default_factory=dict)


def load_cl_contracts() -> pd.DataFrame:
    """EIA NYMEX WTI contracts 1-4 (RCLC1..RCLC4), one column each (c1..c4), from 1985-01-02
    (first print of contract 2) to 2024-04-05. Raw prints; NaN where a contract has no print."""
    df = pd.DataFrame({f"c{j}": load_ext_raw(f"RCLC{j}") for j in (1, 2, 3, 4)})
    df.index.name = "date"
    return df[df.index >= "1985-01-02"]


def _stale_rows(df: pd.DataFrame) -> np.ndarray:
    """Rows where every available contract repeats the previous row exactly: holiday
    placeholders (e.g. 2018-11-22 Thanksgiving, 2018-12-25 Christmas, 1996-01-08 blizzard)."""
    prev = df.shift(1)
    same = ((df == prev) | df.isna()).all(axis=1) & df.notna().any(axis=1)
    same.iloc[0] = False
    return same.to_numpy()


def nymex_calendar(contracts: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict]:
    """Business-day calendar for the CL expiry rule, reconciled from the data.

    data days  = dates with at least one non-stale contract print;
    gap days   = weekdays without a print that are NOT exchange holidays (EIA publication gaps,
                 e.g. Columbus/Veterans day 2015-2019, some Fridays after Thanksgiving);
    business   = data days + gap days, extended 70 calendar days past the last print with
                 rule-based weekdays (needed for the expiry date of the last contract 1).
    Returns (business_days, data_days, facts)."""
    stale = _stale_rows(contracts)
    data_days = contracts.index[~stale & contracts.notna().any(axis=1).to_numpy()]
    hol = us_exchange_holidays(data_days[0].year, data_days[-1].year + 1)
    weekdays = pd.bdate_range(data_days[0], data_days[-1])
    gaps = weekdays.difference(data_days).difference(hol)
    future = pd.bdate_range(data_days[-1] + pd.Timedelta(days=1),
                            data_days[-1] + pd.Timedelta(days=70)).difference(hol)
    business = data_days.union(gaps).union(future)
    facts = dict(stale_rows_removed=[d.date().isoformat() for d in contracts.index[stale]],
                 eia_gap_days_treated_as_business=[d.date().isoformat() for d in gaps],
                 holidays_with_real_prints=[d.date().isoformat()
                                            for d in data_days.intersection(hol)])
    return business, data_days, facts


def _ltd_rule(year: int, month: int, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """NYMEX CL rule: trading in contract 1 ends 3 business days before the 25th calendar day
    of the month (the month before delivery); if the 25th is not a business day, 3 business
    days before the last business day preceding the 25th (= 4 business days before the 25th)."""
    d25 = pd.Timestamp(year, month, 25)
    before = cal[cal < d25]
    return before[-3] if d25 in cal else before[-4]


def back_contract_roll_score(contracts: pd.DataFrame) -> pd.Series:
    """Data evidence that contract identities changed between two consecutive complete rows
    (t', t), using only contracts 2-4 (never the expiring contract 1, whose last-day prints are
    often dislocated):
        no_roll_err = |dc2 - dc3|                         (same contracts on both days)
        roll_err    = |(c2(t) - c3(t')) - (c3(t) - c4(t'))| (old c3 -> new c2, old c4 -> new c3)
        score       = no_roll_err - roll_err  (> 0 favours 'identity change at t')
    The score is weak when the curve is close to linear, and that is exactly when mis-dating a
    roll by a day costs the index almost nothing (the error is the curve's curvature)."""
    v = contracts[["c2", "c3", "c4"]].dropna()
    c2, c3, c4 = (v[c].to_numpy() for c in ("c2", "c3", "c4"))
    no_roll = np.abs((c2[1:] - c2[:-1]) - (c3[1:] - c3[:-1]))
    roll = np.abs((c2[1:] - c3[:-1]) - (c3[1:] - c4[:-1]))
    return pd.Series(no_roll - roll, index=v.index[1:], name="roll_score")


def crude_last_trade_dates(contracts: pd.DataFrame | None = None) -> tuple[pd.DatetimeIndex, dict]:
    """Contract-1 last trading day of every month covered by the data.

    The rule is applied on the reconciled business calendar. In the few months where an EIA
    gap day sits inside the counting window (the Friday after Thanksgiving before 2007 and in
    2016/2017, a missing Christmas Eve), the rule gives two candidates depending on whether that
    day was a NYMEX business day; HARNESS-CHOICE: the candidate whose first roll day has the
    larger back_contract_roll_score is used, and the decision is recorded in facts."""
    contracts = load_cl_contracts() if contracts is None else contracts
    business, data_days, facts = nymex_calendar(contracts)
    score = back_contract_roll_score(contracts[~_stale_rows(contracts)])
    s_idx = score.index
    out, adjudicated = [], []
    months = pd.period_range(business[0], business[-1], freq="M")
    for p in months:
        d25 = pd.Timestamp(p.year, p.month, 25)
        if d25 <= business[0] + pd.Timedelta(days=7) or d25 > business[-1]:
            continue
        la = _ltd_rule(p.year, p.month, business)        # EIA gap days are business days
        if d25 > data_days[-1]:                          # beyond the data: rule calendar only
            out.append(la)
            continue
        lb = _ltd_rule(p.year, p.month, data_days)       # EIA gap days are holidays
        if la == lb:
            out.append(la)
            continue
        ra = s_idx[np.searchsorted(s_idx.values, la.to_datetime64(), side="right")]
        rb = s_idx[np.searchsorted(s_idx.values, lb.to_datetime64(), side="right")]
        sa, sb = float(score[ra]), float(score[rb])
        # Both candidates may fall on the same scoring day (missing back-contract prints):
        # then the data carry no evidence and the rule with the gap day as a business day wins.
        evidence = ra != rb
        pick = lb if (evidence and sb > sa) else la
        out.append(pick)
        adjudicated.append(dict(month=str(p), ltd_if_gap_is_business=la.date().isoformat(),
                                ltd_if_gap_is_holiday=lb.date().isoformat(),
                                score_a=round(sa, 4), score_b=round(sb, 4),
                                data_evidence=bool(evidence), chosen=pick.date().isoformat()))
    facts["ambiguous_months_adjudicated"] = adjudicated
    return pd.DatetimeIndex(out, name="ltd"), facts


@functools.lru_cache(maxsize=None)
def build_crude_roll_index(roll_lead: int = 0) -> RollIndex:
    """Roll-adjusted front-month WTI excess-return index from the EIA contract-1/2 prints.

    Position: contract 1, switched into contract 2 at the close `roll_lead` NYMEX business days
    before contract 1's last trading day (LTD). roll_lead=0 is SPEC2 rule 2 literally: hold
    contract 1 through its LTD close; on the next day (contract 1 has changed identity) the
    return is CL1(t)/CL2(t-1) - 1. roll_lead=k>0: from the close of LTD-k the position is the
    next contract, whose returns are CL2(t)/CL2(t-1)-1 until the LTD and CL1(t)/CL2(t-1)-1 on the
    day after it. No financing / collateral return is added (an EXCESS-return index).

    Index days are the dates where both contract 1 and contract 2 have a positive, non-stale
    print. Missing days are bridged: the return from the previous index day to the next one is
    computed on the contract actually held. HARNESS-CHOICE: if an identity change is crossed
    while the scheduled switch day itself is missing (roll_gap), the return is still
    CL1(t)/CL2(t') - 1 (switch at the last available close before expiry). The single
    non-positive contract-1 print (2020-04-20, -37.63, the May-2020 contract one day before
    its expiry) is dropped like every close <= 0 (SPEC2 data rule), so with roll_lead=0 the
    index jumps from 2020-04-17 to 2020-04-21 on the May contract; with roll_lead >= 2 it holds
    the June contract over those days instead.

    Roll timing is not innocuous: the last days before expiry carry their own (sometimes
    dislocated) returns, so roll_lead=0 and roll_lead=5 differ by a few percent a year over
    1985-2024 (selftest_output.json, "roll_lead5_vs_0"). Carry/premium filters should report
    both "crude_roll" and "crude_roll_lead5".
    """
    raw = load_cl_contracts()
    ltd, facts = crude_last_trade_dates(raw)
    business, _, _ = nymex_calendar(raw)
    stale = _stale_rows(raw)
    ok = (~stale) & (raw["c1"] > 0).to_numpy() & (raw["c2"] > 0).to_numpy()
    v = raw.loc[ok, ["c1", "c2"]]
    days = v.index
    c1, c2 = v["c1"].to_numpy(), v["c2"].to_numpy()

    # Business-day distance from each index day to the next LTD on or after it.
    bpos = business.get_indexer(days)
    if (bpos < 0).any():
        raise AssertionError("index days must be business days")
    nxt = np.searchsorted(ltd.values, days.values, side="left")      # next LTD >= day
    if (nxt >= len(ltd)).any():                                       # after the last LTD
        tail = nxt >= len(ltd)
        raise AssertionError(f"{tail.sum()} index days after the last LTD")
    dist = business.get_indexer(ltd[nxt]) - bpos
    held = np.where(dist <= roll_lead, 2, 1)                          # contract held at close

    # LTDs crossed between consecutive index days: t' <= LTD < t.
    n_cross = np.zeros(len(days), dtype=int)
    k_prev = np.searchsorted(ltd.values, days.values[:-1], side="left")
    k_now = np.searchsorted(ltd.values, days.values[1:], side="left")
    n_cross[1:] = k_now - k_prev
    if (n_cross > 1).any():
        raise AssertionError("two contract expiries between consecutive index days")

    ret = np.zeros(len(days))
    roll_gap = np.zeros(len(days), dtype=bool)
    h_prev = held[:-1]
    roll = n_cross[1:] == 1
    same1 = ~roll & (h_prev == 1)
    same2 = ~roll & (h_prev == 2)
    ret[1:][same1] = c1[1:][same1] / c1[:-1][same1] - 1.0
    ret[1:][same2] = c2[1:][same2] / c2[:-1][same2] - 1.0
    ret[1:][roll] = c1[1:][roll] / c2[:-1][roll] - 1.0
    roll_gap[1:] = roll & (h_prev == 1)

    level = pd.Series(100.0 * np.cumprod(1.0 + ret), index=days,
                      name=f"crude_roll{'' if roll_lead == 0 else f'_lead{roll_lead}'}")
    naive = np.zeros(len(days))
    naive[1:] = c1[1:] / c1[:-1] - 1.0
    spread_prev = np.full(len(days), np.nan)
    spread_prev[1:] = c2[:-1] / c1[:-1] - 1.0
    detail = pd.DataFrame(dict(c1=c1, c2=c2, held=held, roll=n_cross == 1, ret=ret,
                               naive_ret=naive, spread_prev=spread_prev, roll_gap=roll_gap),
                          index=days)
    facts.update(roll_lead=int(roll_lead), first_day=days[0].date().isoformat(),
                 last_day=days[-1].date().isoformat(), n_days=int(len(days)),
                 n_ltd=int(len(ltd)), n_roll_days=int(detail["roll"].sum()),
                 n_roll_gap=int(roll_gap.sum()),
                 dropped_nonpositive_c1=[d.date().isoformat()
                                         for d in raw.index[(raw["c1"] <= 0).to_numpy()]])
    if not (level > 0).all():
        raise AssertionError("roll index went non-positive")
    return RollIndex(level=level, ret=pd.Series(ret, index=days), ltd=ltd, detail=detail,
                     info=facts)


def validate_roll_index(ri: RollIndex) -> dict:
    """Evidence that the roll adjustment works (numbers are computed, nothing is assumed):

    * on roll days the UNADJUSTED continuous return (c1(t)/c1(t-1)-1) contains the calendar
      spread: corr(naive - adjusted, spread_prev) should be ~1, while the adjusted return should
      be unrelated to the spread (corr ~0) -- i.e. roll days are no longer jumps;
    * the size of adjusted roll-day returns should look like any other day's;
    * the data-based back_contract_roll_score should favour 'identity change' far more often on
      the calendar's roll days than on other days."""
    d = ri.detail.iloc[1:]
    rd, nd = d[d["roll"]], d[~d["roll"]]

    def _stats(x):
        x = np.asarray(x, float)
        return dict(n=int(len(x)), mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                    mean_abs=float(np.mean(np.abs(x))),
                    p99_abs=float(np.quantile(np.abs(x), 0.99)), max_abs=float(np.max(np.abs(x))))
    jump = rd["naive_ret"] - rd["ret"]
    raw = load_cl_contracts()
    score = back_contract_roll_score(raw[~_stale_rows(raw)])
    roll_days = d.index[d["roll"]]
    on = score.reindex(roll_days).dropna()
    off = score.drop(roll_days, errors="ignore")
    big = rd.loc[rd["ret"].abs().sort_values(ascending=False).index[:5],
                 ["c1", "c2", "ret", "naive_ret", "spread_prev"]]
    return dict(
        roll_day_adjusted=_stats(rd["ret"]), roll_day_naive=_stats(rd["naive_ret"]),
        other_days=_stats(nd["ret"]),
        corr_naive_minus_adjusted_vs_spread=float(np.corrcoef(jump, rd["spread_prev"])[0, 1]),
        corr_adjusted_vs_spread=float(np.corrcoef(rd["ret"], rd["spread_prev"])[0, 1]),
        corr_naive_vs_spread=float(np.corrcoef(rd["naive_ret"], rd["spread_prev"])[0, 1]),
        mean_abs_spread_on_roll_days=float(rd["spread_prev"].abs().mean()),
        share_score_positive_on_roll_days=float((on > 0).mean()),
        share_score_positive_on_other_days=float((off > 0).mean()),
        largest_abs_adjusted_roll_day_returns=[
            dict(date=i.date().isoformat(), **{k: float(v) for k, v in r.items()})
            for i, r in big.iterrows()],
        n_roll_gap=int(d["roll_gap"].sum()),
        total_return_index=float(ri.level.iloc[-1] / ri.level.iloc[0] - 1.0),
        total_return_c1_price=float(d["c1"].iloc[-1] / ri.detail["c1"].iloc[0] - 1.0),
        min_level=float(ri.level.min()),
    )


# ========================================================================================
# 4. Other traded series
# ========================================================================================
def par_bond_price(y, coupon: float = 0.06, maturity: float = 10.0, freq: int = 2):
    """Price per 100 face of a bond paying `coupon` (annual rate, `freq` times a year) with
    `maturity` years left, at bond-equivalent yield y (decimal):
        P = 100 * [ (c/f) * (1 - (1+y/f)^-n) / (y/f) + (1+y/f)^-n ],  n = maturity*f.
    Fractional n is allowed (standard approximation, used by the total-return variant)."""
    y = np.asarray(y, dtype=float)
    n = maturity * freq
    i = y / freq
    disc = (1.0 + i) ** (-n)
    small = np.abs(i) < 1e-12
    annuity = np.where(small, n, (1.0 - disc) / np.where(small, 1.0, i))
    return 100.0 * (coupon / freq * annuity + disc)


def load_ust10(kind: str = "price", coupon: float = 0.06) -> pd.Series:
    """10-year Treasury proxy from FRED DGS10 (constant-maturity yield, percent).

    kind="price" (default; SPEC2's 'PRICE proxy'): the price of a hypothetical bond with exactly
      10 years to maturity and a FIXED 6% semi-annual coupon (the notional coupon of the CBOT
      10-year T-note future) at each day's DGS10 yield. Only yield changes move it: no coupon
      income, no roll-down (like a futures price, which also excludes carry).
    kind="tr": constant-maturity par-bond TOTAL return: each day buy a 10-year par bond
      (coupon = yesterday's yield), revalue it at today's yield with 10 - dt years left and add
      the accrued coupon y(t-1)*dt (dt in years of 365.25 days); level = 100 * cumprod.
    DGS10 is an H.15 market yield; using a Friday's value at that Friday's close is fine for a
    signal (the yield is a market price observable at the close)."""
    y = load_ext_raw("DGS10") / 100.0
    y = y[y > 0]
    if kind == "price":
        return pd.Series(par_bond_price(y.to_numpy(), coupon), index=y.index, name="ust10_price")
    if kind == "tr":
        yv = y.to_numpy()
        dt = np.diff(y.index.values).astype("timedelta64[D]").astype(float) / 365.25
        p_new = par_bond_price(yv[1:], coupon=yv[:-1], maturity=10.0 - dt)
        r = np.concatenate(([0.0], p_new / 100.0 - 1.0 + yv[:-1] * dt))
        return pd.Series(100.0 * np.cumprod(1.0 + r), index=y.index, name="ust10_tr")
    raise ValueError(kind)


def _ext_price(series_id: str, name: str) -> pd.Series:
    """An EIA/FRED price series from the archive: positive values, stale holiday prints removed."""
    s = load_ext_raw(series_id)
    s = s[s > 0]
    s, _ = drop_stale_holiday_prints(s)
    return s.rename(name)


@functools.lru_cache(maxsize=None)
def _v1_prices() -> dict:
    return lc.load_prices()


# name -> (loader, description). Every loader returns a cleaned Series of positive closes.
SERIES: dict[str, tuple[Callable[[], pd.Series], str]] = {
    "crude_roll": (lambda: build_crude_roll_index(0).level,
                   "WTI front-month roll-adjusted EXCESS-return index from EIA RCLC1/RCLC2 "
                   "(hold contract 1 through its last trading day; SPEC2 rule 2a), 1985-01-02.."
                   "2024-04-05. Includes roll yield: the series for carry/premium tests."),
    "crude_roll_lead5": (lambda: build_crude_roll_index(5).level,
                         "Same index but switching to contract 2 five business days before "
                         "expiry (robustness: avoids expiry-week dislocations)."),
    "crude_cl": (lambda: _v1_prices()["crude"].rename("crude_cl"),
                 "NYMEX CL continuous front month, unadjusted (v1 series, 2000..2026-10)."),
    "crude_spot": (lambda: lc.load_wti_spot().rename("crude_spot"),
                   "EIA Cushing WTI spot (v1 series, 1986..2026-10)."),
    "gold": (lambda: _v1_prices()["gold"], "London spot gold (v1 series)."),
    "silver": (lambda: _v1_prices()["silver"], "London spot silver (v1 series)."),
    "soybean": (lambda: _v1_prices()["soybean"], "CBOT ZS continuous (v1 series)."),
    "sugar": (lambda: _v1_prices()["sugar"], "ICE SB continuous (v1 series)."),
    "ust10_price": (lambda: load_ust10("price"),
                    "10y Treasury PRICE proxy: 10-year 6%-coupon bond priced at DGS10 "
                    "(1962..2026-10)."),
    "ust10_tr": (lambda: load_ust10("tr"),
                 "10y constant-maturity par-bond TOTAL-return proxy from DGS10 (robustness)."),
    "usd_broad": (lambda: _ext_price("DTWEXBGS", "usd_broad"),
                  "Fed nominal broad dollar index DTWEXBGS (2006..2026-10). Published weekly "
                  "(H.10, Mondays): as a SIGNAL input use last week's values."),
    "ho_spot": (lambda: _ext_price("HO_SPOT", "ho_spot"),
                "NY Harbor No.2 heating oil spot, $/gal (EIA, 1986-06..2026-09)."),
    "gas_spot": (lambda: _ext_price("GAS_SPOT", "gas_spot"),
                 "NY Harbor conventional gasoline spot, $/gal (EIA, 1986-06..2026-09)."),
    "brent_spot": (lambda: _ext_price("RBRTE", "brent_spot"),
                   "Europe Brent spot (EIA, 1987-05..2026-09)."),
    "ho_f1": (lambda: _ext_price("HO_F1", "ho_f1"),
              "Heating-oil futures contract 1, UNADJUSTED (roll jumps!), 1980..2024-04."),
    "rbob_f1": (lambda: _ext_price("RBOB_F1", "rbob_f1"),
                "RBOB gasoline futures contract 1, UNADJUSTED (roll jumps!), 2005-10..2024-04."),
}


@functools.lru_cache(maxsize=None)
def _load_series_cached(name: str) -> pd.Series:
    if name not in SERIES:
        raise KeyError(f"unknown series {name!r}; known: {sorted(SERIES)}")
    s = SERIES[name][0]()
    s = s[s.notna() & (s > 0)].sort_index().astype(float)
    if s.index.duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    s.index.name = "date"
    return s


def load_series(name: str) -> pd.Series:
    """A traded series by name (see SERIES). Returns a copy, safe to modify."""
    return _load_series_cached(name).copy()


def describe_series(names: Iterable[str] | None = None) -> pd.DataFrame:
    """Coverage of the registered series."""
    rows = []
    for n in (names or SERIES):
        s = _load_series_cached(n)
        rows.append(dict(series=n, first=s.index[0].date().isoformat(),
                         last=s.index[-1].date().isoformat(), n=int(len(s)),
                         description=SERIES[n][1]))
    return pd.DataFrame(rows)


def _resolve(series: str | pd.Series, series_name: str | None) -> tuple[pd.Series, str]:
    if isinstance(series, str):
        return _load_series_cached(series), series_name or series
    s = series[series.notna() & (series > 0)].sort_index().astype(float)
    return s, series_name or str(series.name)


# ========================================================================================
# 5. Decision dates and entry timing (SPEC2 rule 1)
# ========================================================================================
def weekly_decision_dates(start, end) -> pd.DatetimeIndex:
    """All Fridays in [start, end] (the weekly decision grid of price/macro signals)."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-FRI", name="decision_date")


def asof_values(series: pd.Series, dates, lag_days: int = 0) -> np.ndarray:
    """Value of `series` at the last observation dated on or before (date - lag_days), NaN if
    none. Use it to read a signal input as of each decision Friday (lag_days > 0 for inputs that
    are published with a delay)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)) - pd.Timedelta(days=lag_days)
    pos = np.searchsorted(series.index.values, d.values, side="right") - 1
    out = np.full(len(d), np.nan)
    ok = pos >= 0
    out[ok] = series.to_numpy(dtype=float)[pos[ok]]
    return out


def entry_target(decision_dates, timing: str) -> pd.DatetimeIndex:
    """Earliest calendar date whose close may be used as the entry (see module docstring)."""
    d = pd.DatetimeIndex(pd.to_datetime(decision_dates))
    if timing == "price":
        return d + pd.Timedelta(days=1)           # strictly after the decision close
    if timing == "cot":
        return lc.entry_target(d)                 # v1: Tuesday of the following week
    if timing == "eia":
        return d + pd.Timedelta(days=6)           # week-ending Friday -> Thursday after release
    raise ValueError(f"timing must be one of {TIMINGS}, got {timing!r}")


def entry_positions(decision_dates, price_index: pd.DatetimeIndex, timing: str,
                    max_lag_days: int = MAX_ENTRY_LAG_DAYS) -> np.ndarray:
    """Row of the entry close in the traded series (first trading day on/after the target);
    -1 when there is none (past the end, or more than max_lag_days after the target)."""
    tgt = entry_target(decision_dates, timing)
    pos = np.searchsorted(price_index.values, tgt.values, side="left")
    ok = pos < len(price_index)
    lag = np.full(len(pos), np.inf)
    lag[ok] = (price_index.values[pos[ok]] - tgt.values[ok]) / np.timedelta64(1, "D")
    return np.where(ok & (lag <= max_lag_days), pos, -1).astype(np.int64)


def forward_returns(series: pd.Series, entry_pos: np.ndarray, h_weeks: int) -> np.ndarray:
    """close[entry + 5h] / close[entry] - 1 (NaN without entry or future data) -- v1 code."""
    return lc.forward_returns(series, entry_pos, DAYS_PER_WEEK * int(h_weeks))


def forward_table(decision_dates, series: str | pd.Series, timing: str,
                  horizons: Iterable[int] = HORIZONS_W,
                  max_lag_days: int = MAX_ENTRY_LAG_DAYS) -> pd.DataFrame:
    """Per decision date: entry date/price and, per horizon, the exit date and forward return.
    Walk-forward fits that learn from outcomes must only use rows with exit_date{h} < cutoff."""
    s, _ = _resolve(series, None)
    d = pd.DatetimeIndex(pd.to_datetime(decision_dates))
    ep = entry_positions(d, s.index, timing, max_lag_days)
    has = ep >= 0
    out = pd.DataFrame({"decision_date": d})
    out["entry_date"] = pd.NaT
    out.loc[has, "entry_date"] = s.index.values[ep[has]]
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["entry_price"] = np.where(has, s.to_numpy()[np.where(has, ep, 0)], np.nan)
    for h in horizons:
        xp = ep + DAYS_PER_WEEK * int(h)
        okx = has & (xp < len(s))
        ex = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        ex[okx] = s.index.values[xp[okx]]
        out[f"exit_date{h}"] = ex
        out[f"r{h}"] = forward_returns(s, ep, h)
    return out


# ========================================================================================
# 6. Signal frames
# ========================================================================================
def make_signal_frame(decision_dates, signal=None, *, long=None, short=None, valid=None,
                      excluded=None) -> pd.DataFrame:
    """Canonical weekly signal frame: decision_date, signal (int8: +1 long setup, -1 short
    setup, 0 none), valid (signal statistic defined this week), excluded (not tradable in real
    time, e.g. COT shutdown weeks).

    Give either `signal` (+1/0/-1; NaN means undefined -> valid False) or boolean `long`
    and/or `short` masks (then valid defaults to True). Rows are sorted by decision_date."""
    d = pd.DatetimeIndex(pd.to_datetime(decision_dates))
    n = len(d)
    if signal is not None:
        sv = np.asarray(signal, dtype=float)
        v_auto = ~np.isnan(sv)
        sv = np.where(v_auto, sv, 0.0)
        if not np.isin(sv, (-1.0, 0.0, 1.0)).all():
            raise ValueError("signal must be in {-1, 0, +1} (or NaN)")
    else:
        lo = np.zeros(n, bool) if long is None else np.asarray(long, bool)
        sh = np.zeros(n, bool) if short is None else np.asarray(short, bool)
        if (lo & sh).any():
            raise ValueError("a week cannot be both a long and a short setup")
        sv = lo.astype(float) - sh.astype(float)
        v_auto = np.ones(n, bool)
    v = v_auto if valid is None else (np.asarray(valid, bool) & v_auto)
    x = np.zeros(n, bool) if excluded is None else np.asarray(excluded, bool)
    sf = pd.DataFrame(dict(decision_date=d, signal=np.where(v, sv, 0).astype(np.int8),
                           valid=v, excluded=x))
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    if sf["decision_date"].duplicated().any():
        raise ValueError("duplicate decision dates")
    return sf


def cot_signal_frame(market: str = "crude", idx_col: str = "idx26", buy_th: float = 80,
                     sell_th: float = 20, panel: Mapping[str, pd.DataFrame] | None = None
                     ) -> pd.DataFrame:
    """The v1 COT setups of one market as a signal frame (timing="cot"): decision_date =
    report_date, valid = index defined, excluded = report in a shutdown window."""
    f = (panel or lc.build_panel(markets=[market]))[market]
    lo, sh = lc.signal_masks(f[idx_col].to_numpy(), buy_th, sell_th)
    return make_signal_frame(f["report_date"], long=lo, short=sh,
                             valid=f[idx_col].notna().to_numpy(),
                             excluded=f["shutdown"].to_numpy(dtype=bool))


def cot_shutdown_mask(report_dates) -> np.ndarray:
    """True for COT report dates inside the v1 shutdown windows (not tradable in real time)."""
    return lc.shutdown_mask(report_dates)


# ========================================================================================
# 7. Evaluation (SPEC2 rules 3-4)
# ========================================================================================
def _episode_ids_values(vals: np.ndarray) -> np.ndarray:
    """Episodes of a {-1,0,1} sequence: a new episode starts when a non-zero value starts or
    changes sign; 0 where inactive."""
    v = np.asarray(vals)
    prev = np.concatenate(([0], v[:-1]))
    start = (v != 0) & (v != prev)
    return np.where(v != 0, np.cumsum(start), 0)


def _perm_offsets_all(T: int, min_shift: int) -> np.ndarray:
    return np.arange(min_shift, T - min_shift + 1)


def _cell(sig_full: np.ndarray, valid: np.ndarray, excluded: np.ndarray, r: np.ndarray,
          side: str, *, k: int, seed: int, min_shift: int, exact: bool, do_perm: bool) -> dict:
    """Statistics of one (side, horizon) cell on aligned full-sequence arrays."""
    elig = valid & ~excluded & ~np.isnan(r)
    r_e = r[elig]
    T = int(elig.sum())
    baseline = float(r_e.mean()) if T else float("nan")
    nan = float("nan")
    out = dict(n=0, episodes=0, mean=nan, median=nan, baseline=baseline, excess=nan, hit=nan,
               perm_p=nan, perm_p_exact=nan, perm_p_exact_raw=nan, T=T)
    if side in ("long", "short"):
        sgn = 1 if side == "long" else -1
        setup = valid & (sig_full == sgn)
        sig = setup[elig]
        signed = sgn * r_e
        n = int(sig.sum())
        out["episodes"] = lc.count_episodes(setup, elig, "full")
        signed_base = sgn * baseline
    elif side == "net":
        s_full = np.where(valid, sig_full, 0)
        s_e = s_full[elig].astype(float)
        sig = s_e != 0
        signed = s_e * r_e
        n = int(sig.sum())
        ids = _episode_ids_values(s_full)[elig]
        out["episodes"] = int(len(np.unique(ids[sig])))
        signed_base = float(s_e[sig].mean() * baseline) if n else nan
    else:
        raise ValueError(f"side must be one of {SIDES}")
    out["n"] = n
    if n == 0:
        return out
    sr = signed[sig]
    mean = float(sr.mean())
    out.update(mean=mean, median=float(np.median(sr)), hit=float((sr > 0).mean()),
               excess=mean - signed_base)
    if not do_perm or T < 2 * min_shift + 1:
        return out
    if side in ("long", "short"):
        # v1 engine code: shifted mean of the signed return over the shifted signal weeks.
        out["perm_p"] = lc.perm_pvalue(sig, signed, mean, k, seed, min_shift)
        if exact:
            offs = _perm_offsets_all(T, min_shift)
            cnt = int(np.sum(lc._shifted_means(sig, signed, offs) >= mean))
            out["perm_p_exact_raw"] = cnt / len(offs)
            out["perm_p_exact"] = (1 + cnt) / (1 + len(offs))
    else:
        # Net: shift the +1/-1 setup vector; statistic = mean of shifted setup * r over the
        # shifted setup weeks (number of long and short weeks is preserved by the shift).
        pos = np.flatnonzero(sig)
        vals = s_e[pos]

        def _stats(offs):
            return (vals[None, :] * r_e[(pos[None, :] + np.asarray(offs)[:, None]) % T]).mean(1)
        rng_offs = lc.perm_offsets(T, k, seed, min_shift)
        out["perm_p"] = (1 + int(np.sum(_stats(rng_offs) >= mean))) / (1 + k)
        if exact:
            offs = _perm_offsets_all(T, min_shift)
            cnt = int(np.sum(_stats(offs) >= mean))
            out["perm_p_exact_raw"] = cnt / len(offs)
            out["perm_p_exact"] = (1 + cnt) / (1 + len(offs))
    return out


def _period(dates: pd.Series | pd.DatetimeIndex) -> str:
    d = pd.DatetimeIndex(dates)
    return f"{d.min().date().isoformat()}..{d.max().date().isoformat()}" if len(d) else ""


def evaluate(sf: pd.DataFrame, series: str | pd.Series, horizons: Iterable[int] = HORIZONS_W,
             *, timing: str, sides: Sequence[str] = ("long", "short"),
             series_name: str | None = None, labels: Mapping | None = None,
             k: int = K_PERM, seed: int = SEED, min_shift: int = MIN_SHIFT, exact: bool = True,
             do_perm: bool = True, max_lag_days: int = MAX_ENTRY_LAG_DAYS) -> pd.DataFrame:
    """Canonical test rows (SPEC2 rule 8 'tests') for one signal frame on one traded series.

    For every side in `sides` and horizon h (weeks):
      n         signal weeks with a forward return (and not excluded)
      episodes  setup episodes containing at least one counted week (v1 "full" mode)
      mean      mean SIGNED forward return over the signal weeks (+r long, -r short)
      median    median signed return
      baseline  mean UNSIGNED forward return over all eligible weeks of this series/horizon
      excess    mean - signed baseline (signed baseline = +baseline long, -baseline short,
                mean setup sign * baseline for "net")
      hit       share of signed returns > 0
      perm_p    Monte-Carlo circular-shift p (K shifts, offsets uniform on [26, T-26],
                default_rng(seed) per cell, (1 + #{shifted >= observed}) / (1 + K)) -- v1 test
      perm_p_exact      exhaustive over every allowed offset with the +1 correction:
                        (1 + #{shifted >= observed}) / (1 + #offsets)        (SPEC2 rule 4)
      perm_p_exact_raw  exhaustive share without the correction (= v1's perm_p_exact column)
      T         eligible weeks; period = first..last eligible decision date
    Overlapping forward returns make naive t-stats invalid; none are reported.
    `labels` (filter, variant, market, is_primary, ...) are copied into every row."""
    s, sname = _resolve(series, series_name)
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    ep = entry_positions(sf["decision_date"], s.index, timing, max_lag_days)
    sig_full = sf["signal"].to_numpy(dtype=np.int64)
    valid = sf["valid"].to_numpy(dtype=bool)
    excl = sf["excluded"].to_numpy(dtype=bool)
    rows = []
    base = dict(filter=None, variant=None, market=None, is_primary=False)
    base.update(labels or {})
    for side in sides:
        for h in horizons:
            r = forward_returns(s, ep, h)
            c = _cell(sig_full, valid, excl, r, side, k=k, seed=seed, min_shift=min_shift,
                      exact=exact, do_perm=do_perm)
            elig = valid & ~excl & ~np.isnan(r)
            rows.append({**base, "series": sname, "timing": timing, "side": side,
                         "horizon_w": int(h), **c,
                         "period": _period(sf["decision_date"][elig])})
    out = pd.DataFrame(rows)
    extra = [c for c in out.columns if c not in TEST_COLUMNS]
    return out[[c for c in TEST_COLUMNS if c in out.columns] + extra]


def evaluate_pooled(parts: Sequence[tuple[pd.DataFrame, str | pd.Series, str]],
                    horizons: Iterable[int] = (PRIMARY_H,), *, sides=("long", "short"),
                    labels: Mapping | None = None, k: int = K_PERM, seed: int = SEED,
                    min_shift: int = MIN_SHIFT,
                    max_lag_days: int = MAX_ENTRY_LAG_DAYS) -> pd.DataFrame:
    """One cell pooled across markets, exactly like lc.pooled_cell: parts = [(signal frame,
    series, timing), ...]. Pooled mean/median/hit over all signal weeks of all parts;
    excess_vs_own_baseline = mean of (signed r - signed baseline of its own part); one
    default_rng(seed) per pooled cell, each part drawing its own K offsets in `parts` order;
    statistic = pooled mean signed return. (No exhaustive version: the joint offset space is
    the product of the parts'.)"""
    rows = []
    base = dict(filter=None, variant=None, market="pooled", is_primary=False)
    base.update(labels or {})
    prepared = []
    for sf, ser, timing in parts:
        s, sname = _resolve(ser, None)
        sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
        prepared.append((sf, s, sname, timing,
                         entry_positions(sf["decision_date"], s.index, timing, max_lag_days)))
    for side in sides:
        if side not in ("long", "short"):
            raise ValueError("pooled cells support long/short")
        sgn = 1 if side == "long" else -1
        for h in horizons:
            all_sr, all_ex, epis, arrays = [], [], 0, []
            for sf, s, sname, timing, ep in prepared:
                r = forward_returns(s, ep, h)
                valid = sf["valid"].to_numpy(bool)
                elig = valid & ~sf["excluded"].to_numpy(bool) & ~np.isnan(r)
                setup = valid & (sf["signal"].to_numpy(np.int64) == sgn)
                sig, signed = setup[elig], sgn * r[elig]
                b = float(r[elig].mean())
                all_sr.append(signed[sig])
                all_ex.append(signed[sig] - sgn * b)
                epis += lc.count_episodes(setup, elig, "full")
                arrays.append((sig, signed))
            sr = np.concatenate(all_sr)
            n = int(len(sr))
            row = {**base, "series": "+".join(p[2] for p in prepared), "timing":
                   "+".join(sorted({p[3] for p in prepared})), "side": side, "horizon_w": int(h),
                   "n": n, "episodes": int(epis)}
            if n == 0:
                rows.append(row)
                continue
            obs = float(sr.mean())
            rng = np.random.default_rng(seed)
            sums = np.zeros(k)
            for sig, signed in arrays:
                T = len(sig)
                offs = rng.integers(min_shift, T - min_shift, size=k, endpoint=True)
                pos = np.flatnonzero(sig)
                if len(pos):
                    sums += signed[(pos[None, :] + offs[:, None]) % T].sum(axis=1)
            row.update(mean=obs, median=float(np.median(sr)), hit=float((sr > 0).mean()),
                       excess=float(np.concatenate(all_ex).mean()),
                       perm_p=(1 + int(np.sum(sums / n >= obs))) / (1 + k),
                       perm_p_exact=float("nan"), perm_p_exact_raw=float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


# ========================================================================================
# 8. Out-of-sample helpers (SPEC2 rule 5)
# ========================================================================================
def eligible_dates(sf: pd.DataFrame, series: str | pd.Series, *, timing: str,
                   horizon: int = PRIMARY_H, max_lag_days: int = MAX_ENTRY_LAG_DAYS
                   ) -> pd.DatetimeIndex:
    """Decision dates that are eligible at `horizon` (valid, not excluded, forward return)."""
    s, _ = _resolve(series, None)
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    ep = entry_positions(sf["decision_date"], s.index, timing, max_lag_days)
    r = forward_returns(s, ep, horizon)
    m = sf["valid"].to_numpy(bool) & ~sf["excluded"].to_numpy(bool) & ~np.isnan(r)
    return pd.DatetimeIndex(sf["decision_date"][m])


def split_half(sf: pd.DataFrame, series: str | pd.Series, *, timing: str,
               horizon: int = PRIMARY_H, sides: Sequence[str] = ("long",),
               horizons: Iterable[int] | None = None, **eval_kw) -> pd.DataFrame:
    """The rule on the two halves of its OWN sample, split at the midpoint DATE of the eligible
    decision dates (at `horizon`): H1 = decision_date <= midpoint, H2 = after. Each half is a
    full evaluate() run (own baseline, own T, own permutation); weeks of the other half are
    marked invalid, so an episode crossing the midpoint is counted in both halves.
    Rows get half = "H1"/"H2" and split_date."""
    el = eligible_dates(sf, series, timing=timing, horizon=horizon)
    if len(el) == 0:
        raise ValueError("no eligible weeks")
    mid = el[0] + (el[-1] - el[0]) / 2
    out = []
    for half, m in (("H1", sf["decision_date"] <= mid), ("H2", sf["decision_date"] > mid)):
        part = sf.copy()
        part["valid"] = part["valid"].to_numpy(bool) & m.to_numpy()
        part["signal"] = np.where(part["valid"], part["signal"], 0).astype(np.int8)
        rows = evaluate(part, series, horizons or (horizon,), timing=timing, sides=sides,
                        **eval_kw)
        rows["half"] = half
        rows["split_date"] = mid.date().isoformat()
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def walk_forward(frame: pd.DataFrame, fit: Callable[[pd.DataFrame, pd.Timestamp], object],
                 apply: Callable[[object, pd.DataFrame], Sequence[float]], *, first_test,
                 refit: str = "YS", last=None) -> pd.DataFrame:
    """Generic walk-forward signal builder.

    frame: weekly DataFrame with a 'decision_date' column plus whatever columns `fit`/`apply`
           need (and optionally 'excluded').
    For each block [b0, b1) of the calendar grid pd.date_range(first_test, ..., freq=refit):
        params = fit(frame[decision_date < b0], b0)        # history strictly before the block
        signal = apply(params, frame[b0 <= decision_date < b1])   # +1/0/-1 or NaN
    fit() receives the cutoff b0 so that any OUTCOME it learns from can be restricted to
    returns already realised before b0 (use forward_table(...)[exit_date{h} < b0]).
    Returns a signal frame: valid only on weeks inside a tested block with a finite signal."""
    f = frame.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    d = pd.DatetimeIndex(f["decision_date"])
    last = pd.Timestamp(last) if last is not None else d[-1]
    starts = pd.date_range(pd.Timestamp(first_test), last, freq=refit)
    if len(starts) == 0 or starts[0] != pd.Timestamp(first_test):
        starts = pd.DatetimeIndex([pd.Timestamp(first_test)]).append(starts)
    sig = np.full(len(f), np.nan)
    ends = list(starts[1:]) + [last + pd.Timedelta(days=1)]
    for b0, b1 in zip(starts, ends):
        m = np.asarray((d >= b0) & (d < b1))
        if not m.any():
            continue
        params = fit(f.loc[np.asarray(d < b0)].copy(), b0)
        sig[m] = np.asarray(apply(params, f.loc[m].copy()), dtype=float)
    excl = f["excluded"].to_numpy(bool) if "excluded" in f else None
    return make_signal_frame(d, sig, excluded=excl)


def expanding_zscore(x, min_periods: int = 52, window: int | None = None, ddof: int = 1
                     ) -> np.ndarray:
    """z of each value against the mean/std of the values up to AND including it (all known at
    the decision time; no future data). window=None: expanding; else rolling over `window`
    observations. NaN before min_periods."""
    s = pd.Series(np.asarray(x, dtype=float))
    roll = (s.expanding(min_periods=min_periods) if window is None
            else s.rolling(window, min_periods=min_periods))
    mu, sd = roll.mean(), roll.std(ddof=ddof)
    return ((s - mu) / sd.where(sd > 0)).to_numpy()


def expanding_percentile(x, min_periods: int = 52, window: int | None = None) -> np.ndarray:
    """100 * share of the values up to AND including the current one that are <= it
    (expanding, or rolling over `window` observations). NaN inputs are skipped in the history
    and give NaN. NaN before min_periods valid observations."""
    v = np.asarray(x, dtype=float)
    out = np.full(len(v), np.nan)
    hist: list[float] = []          # sorted history for the expanding case
    for i, xi in enumerate(v):
        if np.isnan(xi):
            continue
        if window is not None:
            lo = max(0, i - window + 1)
            past = v[lo:i + 1]
            past = past[~np.isnan(past)]
            if len(past) >= min_periods:
                out[i] = 100.0 * np.mean(past <= xi)
            continue
        bisect.insort(hist, xi)
        if len(hist) >= min_periods:
            out[i] = 100.0 * bisect.bisect_right(hist, xi) / len(hist)
    return out


# ========================================================================================
# 9. Strategy simulation (SPEC2 rule 6), reusing strat_lib
# ========================================================================================
STRATEGIES = ("S1_long_short", "S1_long_only", "S2_long_short", "S2_long_only")


def make_ctx(sf: pd.DataFrame, series: pd.Series, timing: str, name: str = "",
             max_lag_days: int = MAX_ENTRY_LAG_DAYS) -> sl.MarketCtx:
    """strat_lib context for an arbitrary signal frame. ctx.idx holds the +1/0/-1 signal (NaN
    where invalid); the window starts at the entry close of the first valid week (v1: first
    week with a defined index), and ends at the last close of the series."""
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    close = series.to_numpy(dtype=float)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    sma = pd.Series(close).rolling(SMA_N, min_periods=SMA_N).mean().to_numpy()
    ep = entry_positions(sf["decision_date"], series.index, timing, max_lag_days)
    valid = sf["valid"].to_numpy(bool)
    idx = np.where(valid, sf["signal"].to_numpy(dtype=float), np.nan)
    ok = np.flatnonzero(valid & (ep >= 0))
    if len(ok) == 0:
        raise ValueError("no valid week with an entry inside the series")
    return sl.MarketCtx(market=name, dates=series.index, close=close, ret=ret, sma=sma,
                        start=int(ep[ok[0]]), report_date=sf["decision_date"].to_numpy(),
                        entry_pos=ep, shutdown=sf["excluded"].to_numpy(bool), idx=idx,
                        idx_defined=valid)


def signal_events(ctx: sl.MarketCtx, sig: np.ndarray, sides=("long", "short")) -> list[dict]:
    """Generic version of strat_lib.setup_events (first_of_episode=True): one event per setup
    episode (maximal run on the full weekly sequence), dated at the episode's FIRST TRADABLE
    week (not excluded, entry inside the data); an episode with no tradable week gives none.
    `sig` = +1/0/-1 per week with NaN where invalid."""
    trad = sl.tradable_mask(ctx)
    s = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0)
    ev = []
    for side in sides:
        setup = s == (1.0 if side == "long" else -1.0)
        ids = lc.episode_ids(setup)
        for e in np.unique(ids[ids > 0]):
            members = np.flatnonzero(ids == e)
            rows = members[trad[members]]
            if len(rows) == 0:
                continue
            j = int(rows[0])
            ev.append(dict(side=side, pos=int(ctx.entry_pos[j]), report=j,
                           episode_first_report=int(members[0]),
                           delayed_by_shutdown=bool(j != members[0])))
    ev.sort(key=lambda dd: (dd["pos"], dd["report"]))
    return ev


def _run(ctx: sl.MarketCtx, sig: np.ndarray, hold: int, win: int, s2_kwargs: dict) -> dict:
    """{strategy: (position path, trades)} for S1/S2 x long+short/long-only."""
    ev_ls = signal_events(ctx, sig, ("long", "short"))
    ev_l = [e for e in ev_ls if e["side"] == "long"]
    out = {}
    for mode, ev in (("long_short", ev_ls), ("long_only", ev_l)):
        t1 = sl.s1_trades(ctx, ev, hold)
        out[f"S1_{mode}"] = (sl.trades_to_pos(ctx, t1), t1)
        t2 = sl.s2_trades(ctx, ev, hold, win, **s2_kwargs)
        out[f"S2_{mode}"] = (sl.trades_to_pos(ctx, t2), t2)
    return out


def _shift_signal(ctx: sl.MarketCtx, k: int) -> np.ndarray:
    """Random-timing null: the signal circularly shifted by k over the VALID weeks (np.roll),
    exactly like strat_lib.shift_idx; timing, exclusions and prices stay real."""
    out = ctx.idx.copy()
    d = np.flatnonzero(ctx.idx_defined)
    out[d] = np.roll(ctx.idx[d], k)
    return out


def simulate(sf: pd.DataFrame, series: str | pd.Series, *, timing: str,
             series_name: str | None = None, labels: Mapping | None = None,
             hold: int = HOLD, win: int = TRIG_WIN, cost: float = COST,
             null_k: int = K_PERM, seed: int = SEED, min_shift: int = MIN_SHIFT,
             s2_kwargs: Mapping | None = None, max_lag_days: int = MAX_ENTRY_LAG_DAYS,
             keep_equity: bool = True) -> dict:
    """SPEC2 rule 6 for the primary rule.

    S1: enter at the entry close of the first tradable week of each setup episode, hold `hold`
        trading days, one position at a time (strat_lib.s1_trades).
    S2: same events, but wait up to `win` trading days (entry day included) for a close above
        (long) / below (short) its 20-day SMA (strat_lib.s2_trades).
    Each in long+short and long-only form; daily close-to-close P&L, cost per side per unit of
    position change, rf = 0; buy & hold of the same series over the same window (no cost).
    Random-timing null: the weekly signal circularly shifted over the valid weeks by null_k
    offsets uniform on [26, T-26] (one default_rng(seed)); p = (1 + #{null Sharpe >= real})
    / (1 + null_k). null_k=0 skips it.

    Returns dict(metrics=DataFrame, trades=DataFrame, equity=DataFrame (daily equity, 1 before
    the first close), null=dict(strategy -> array of null Sharpes), ctx=MarketCtx)."""
    s, sname = _resolve(series, series_name)
    ctx = make_ctx(sf, s, timing, sname, max_lag_days)
    s2_kwargs = dict(s2_kwargs or {})
    runs = _run(ctx, ctx.idx, hold, win, s2_kwargs)
    bh = np.zeros(ctx.n)
    bh[ctx.start:] = 1.0
    bh_net = sl.daily_net(ctx, bh, cost=0.0)
    bh_m = sl.perf_metrics(bh_net, sl.held_mask(ctx, bh))
    years = (ctx.dates[-1] - ctx.dates[ctx.start]).days / 365.25

    null = {}
    if null_k:
        T = int(ctx.idx_defined.sum())
        offs = np.random.default_rng(seed).integers(min_shift, T - min_shift, size=null_k,
                                                    endpoint=True)
        null = {st: np.empty(null_k) for st in STRATEGIES}
        for i, kk in enumerate(offs):
            res = _run(ctx, _shift_signal(ctx, int(kk)), hold, win, s2_kwargs)
            for st, (pos, _) in res.items():
                null[st][i] = sl.sharpe_only(sl.daily_net(ctx, pos, cost).to_numpy(), years)

    base = dict(filter=None, variant=None, market=None)
    base.update(labels or {})
    rows, trades, equity = [], [], {}
    for st in STRATEGIES:
        pos, tr = runs[st]
        net = sl.daily_net(ctx, pos, cost)
        pm = sl.perf_metrics(net, sl.held_mask(ctx, pos))
        tr = sl.trade_returns(ctx, tr, cost)
        row = {**base, "series": sname, "timing": timing, "strategy": st,
               **sl.trade_stats(tr), **pm,
               **{f"bh_{kk}": bh_m[kk] for kk in ("ann_return", "ann_vol", "sharpe", "max_dd",
                                                   "total_return")}}
        if null_k:
            v = null[st]
            row.update(null_sharpe_median=float(np.median(v)),
                       null_sharpe_p05=float(np.quantile(v, 0.05)),
                       null_sharpe_p95=float(np.quantile(v, 0.95)),
                       null_p_sharpe=(1 + int(np.sum(v >= pm["sharpe"]))) / (1 + null_k))
        rows.append(row)
        for t in tr:
            trades.append(dict(strategy=st, side=t["side"],
                               signal_decision_date=pd.Timestamp(
                                   ctx.report_date[t["signal_report"]]).date().isoformat(),
                               trigger_date=ctx.dates[t["trigger"]].date().isoformat(),
                               entry_date=t["entry_date"].date().isoformat(),
                               exit_date=t["exit_date"].date().isoformat(),
                               entry_price=t["entry_price"], exit_price=t["exit_price"],
                               days_held=t["days_held"], open_at_end=t["open"],
                               price_ret=t["price_ret"], net_ret=t["net_ret"]))
        if keep_equity:
            equity[st] = (1.0 + net).cumprod()
    rows.append({**base, "series": sname, "timing": timing, "strategy": "buy_hold", **bh_m})
    if keep_equity:
        equity["buy_hold"] = (1.0 + bh_net).cumprod()
    return dict(metrics=pd.DataFrame(rows), trades=pd.DataFrame(trades),
                equity=pd.DataFrame(equity) if keep_equity else None, null=null, ctx=ctx)


def weekly_equity(equity: pd.DataFrame) -> dict:
    """Chart-ready weekly sample (last trading day of each week) of an equity DataFrame."""
    wk = equity.groupby(equity.index.to_period("W-FRI")).tail(1)
    return {"dates": [d.date().isoformat() for d in wk.index],
            "series": {c: [round(float(v), 6) for v in wk[c]] for c in wk.columns}}


# ========================================================================================
# 10. Output helpers (SPEC2 rule 8)
# ========================================================================================
def to_jsonable(obj):
    """Recursively convert numpy/pandas objects to plain JSON types (NaN/inf -> None)."""
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).date().isoformat()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if obj is pd.NaT:
        return None
    return obj


def write_results(path: str | Path, *, filter: str, larry_definition: str,
                  sources: Sequence[str], primary: Mapping, tests: pd.DataFrame,
                  strategy, oos: Mapping, caveats: Sequence[str], **extra) -> Path:
    """Write results.json in the SPEC2 rule-8 contract."""
    missing = [c for c in ("filter", "variant", "market", "series", "side", "horizon_w", "n",
                           "episodes", "mean", "baseline", "excess", "hit", "perm_p",
                           "perm_p_exact", "period", "is_primary") if c not in tests.columns]
    if missing:
        raise ValueError(f"tests lacks columns {missing}")
    doc = {"filter": filter, "larry_definition": larry_definition, "sources": list(sources),
           "primary": primary, "tests": tests, "strategy": strategy, "oos": oos,
           "caveats": list(caveats), **extra}
    p = Path(path)
    p.write_text(json.dumps(to_jsonable(doc), indent=1))
    return p

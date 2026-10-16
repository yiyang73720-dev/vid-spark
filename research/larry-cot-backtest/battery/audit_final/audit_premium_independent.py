#!/usr/bin/env python3
"""
audit_premium_independent -- final auditor's independent rebuild of the top-ranked tool
(premium / backwardation: crude contract 1 > contract 2 -> LONG, h = 8 weeks, crude_roll).

Everything here is re-implemented from the RAW files, without fund_harness, premium_lib or the
v1 engine:

  * EIA NYMEX WTI contracts 1-4 (RCLC1..RCLC4 JSON, parsed with json only),
  * EIA Cushing WTI spot (S/data/wti-daily.csv).

The one deliberate methodological difference from the battery is the ROLL CALENDAR. The harness
dates contract-1 identity changes with the NYMEX expiry rule on a reconciled business calendar
(adjudicating ambiguous months with a back-contract score); the premium verifier used the same
rule on the EIA print calendar. This script instead DETECTS the identity change from the price
data alone: in every calendar month it picks the one day whose curve shape is best explained by
"every contract moved up one slot" (see detect_roll_days). If the published numbers are robust
they must survive a roll calendar built in a completely different way.

Conventions copied from SPEC2 (they are definitions, not code):
  decision = every Friday 1985-01-04..2024-03-29; signal read from the last index day on or
  before the Friday if it is at most 4 calendar days old; entry = first index day strictly after
  the Friday (at most 7 days later); exit = entry + 40 index days; baseline = mean over all
  eligible weeks; excess = mean signed return - signed baseline; episodes = runs of the setup on
  the full weekly sequence (undefined weeks break a run, weeks without a forward return do not);
  circular-shift p over offsets [26, T-26] with the +1 correction (exhaustive) and a Monte-Carlo
  version (K = 2000, default_rng(11)); S1/S2 as in SPEC2 rule 6 (0.05% per side).

Run:  python3 -I audit_premium_independent.py
      (writes ROOT/work2/audit_final/audit_premium_independent.json)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True    # never write .pyc files next to the published scripts

import json                       # noqa: E402
import os                         # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
SERIES_DIR = ROOT / "ext" / "accomplish999_crude-oil" / "data" / "series"
SPOT_CSV = ROOT / "data" / "wti-daily.csv"
PREMIUM_RESULTS = ROOT / "work2" / "tool_premium" / "results.json"
OUT = ROOT / "work2" / "audit_final" / "audit_premium_independent.json"

FIRST_FRIDAY, LAST_FRIDAY = "1985-01-04", "2024-03-29"   # the pre-registered crude grid
HOLD_DAYS = 40            # 8 weeks x 5 trading days
MAX_SIGNAL_AGE = 4        # calendar days between the last print and the Friday
MAX_ENTRY_LAG = 7         # calendar days between the Friday and the entry close
MIN_SHIFT, K_MC, SEED = 26, 2000, 11
COST = 0.0005             # 0.05% per side per unit of position change
SMA_N, TRIG_WIN = 20, 20  # S2: close above/below its 20-day SMA within 20 trading days


# =============================================================================================
# 1. Raw data
# =============================================================================================
def load_eia_json(series_id: str) -> pd.Series:
    """One EIA series from the third-party JSON dump (parsed with json only, never executed)."""
    with open(SERIES_DIR / f"{series_id}.json", encoding="utf-8") as fh_:
        obs = json.load(fh_)["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs
                   if o.get("value") not in (None, "", ".")}, name=series_id)
    return s.sort_index()


def load_contracts() -> pd.DataFrame:
    """c1..c4 on the union of their dates, from 1985-01-02 (first print of contract 2).
    Non-positive prints (only 2020-04-20 c1 = -37.63) become NaN, as SPEC2 drops closes <= 0."""
    df = pd.concat({f"c{j}": load_eia_json(f"RCLC{j}") for j in (1, 2, 3, 4)}, axis=1,
                   sort=True)
    df = df[df.index >= "1985-01-02"]
    return df.where(df > 0)


def drop_placeholder_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove rows in which every printed contract repeats the previous row exactly: EIA
    carries the last settlement forward over some exchange holidays, and such a row is not a
    tradable close. (Independent criterion: no holiday calendar is used.)"""
    prev = df.shift(1)
    same = ((df == prev) | df.isna()).all(axis=1) & df.notna().any(axis=1)
    same.iloc[0] = False
    return df[~same], [d.date().isoformat() for d in df.index[same]]


def load_spot() -> pd.Series:
    """EIA Cushing WTI spot, positive prints only."""
    s = pd.read_csv(SPOT_CSV, parse_dates=["Date"]).set_index("Date")["Price"].astype(float)
    return s[s > 0].sort_index()


# =============================================================================================
# 2. Data-driven roll calendar and the roll-adjusted front-month index
# =============================================================================================
def detect_roll_days(v: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """First day of each new contract 1, found from the curve shape alone.

    For consecutive index days (t', t) let a_k = log(c_k / c_(k+1)) (k = 1, 2, 3).
      no_roll_err = |a1(t) - a1(t')| + |a2(t) - a2(t')|   (same contracts on both days)
      roll_err    = |a1(t) - a2(t')| + |a2(t) - a3(t')|   (every contract moved up one slot)
      score       = no_roll_err - roll_err               (> 0 favours a roll at t)
    Terms whose inputs are missing are left out. CL has one expiry per month, around the 20th,
    so in every calendar month the day with the largest score among days 14..28 is taken as the
    roll day. The margin to the runner-up is recorded so weak months can be inspected."""
    a1 = np.log(v["c1"] / v["c2"]).to_numpy()
    a2 = np.log(v["c2"] / v["c3"]).to_numpy()
    a3 = np.log(v["c3"] / v["c4"]).to_numpy()
    no_roll = np.nansum(np.vstack([np.abs(a1[1:] - a1[:-1]), np.abs(a2[1:] - a2[:-1])]), axis=0)
    roll = np.nansum(np.vstack([np.abs(a1[1:] - a2[:-1]), np.abs(a2[1:] - a3[:-1])]), axis=0)
    score = pd.Series(np.concatenate(([np.nan], no_roll - roll)), index=v.index)
    window = score[(score.index.day >= 14) & (score.index.day <= 28)].dropna()
    picks, rows = [], []
    for period, grp in window.groupby(window.index.to_period("M")):
        srt = grp.sort_values(ascending=False)
        picks.append(srt.index[0])
        rows.append(dict(month=str(period), roll_day=srt.index[0].date().isoformat(),
                         score=float(srt.iloc[0]),
                         margin=float(srt.iloc[0] - srt.iloc[1]) if len(srt) > 1 else np.nan))
    return pd.DatetimeIndex(picks), pd.DataFrame(rows)


def _nth_weekday(y: int, m: int, wd: int, n: int) -> pd.Timestamp:
    """n-th (1-based) weekday `wd` (Mon=0) of month m; n = -1 for the last one."""
    days = pd.date_range(pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))
    hits = days[days.weekday == wd]
    return hits[n - 1] if n > 0 else hits[-1]


def _easter(y: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, month, day)


# One-off NYMEX closures (storms, national days of mourning, 9/11).
SPECIAL_CLOSED = ["1985-09-27", "1994-04-27", "2001-09-11", "2001-09-12", "2001-09-13",
                  "2001-09-14", "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30",
                  "2018-12-05"]


def rule_holidays(y0: int, y1: int) -> pd.DatetimeIndex:
    """Candidate NYMEX holidays by rule (written for this audit): New Year (Sunday -> Monday),
    MLK day, Presidents' day, Good Friday, Memorial day, July 4 and Christmas (Saturday ->
    Friday, Sunday -> Monday), Labor day, Thanksgiving, plus the one-off closures above.
    The data decide in the end: see rule_roll_days."""
    out = []
    for y in range(y0, y1 + 1):
        nyd = pd.Timestamp(y, 1, 1)
        out.append(nyd + pd.Timedelta(days=1) if nyd.weekday() == 6 else nyd)
        out += [_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3),
                _easter(y) - pd.Timedelta(days=2), _nth_weekday(y, 5, 0, -1),
                _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4)]
        for m, dd in ((7, 4), (12, 25)):
            x = pd.Timestamp(y, m, dd)
            out.append(x - pd.Timedelta(days=1) if x.weekday() == 5 else
                       x + pd.Timedelta(days=1) if x.weekday() == 6 else x)
    out += [pd.Timestamp(s) for s in SPECIAL_CLOSED]
    return pd.DatetimeIndex(sorted(set(out)))


def rule_roll_days(clean: pd.DataFrame, v: pd.DataFrame) -> tuple[pd.DatetimeIndex, list]:
    """First day of each new contract 1 from the NYMEX expiry rule: trading ends 3 business
    days before the 25th of the month before delivery; if the 25th is not a business day,
    3 business days before the last business day preceding the 25th.

    Business days = weekdays that are not rule holidays, EXCEPT that a rule holiday on which
    EIA has a genuine (non-placeholder) contract print counts as a business day (the data
    override the rule). Weekdays without any print that are not rule holidays (EIA publication
    gaps) stay business days. The roll day is the first index day after the last trading day.
    Returns (roll days, list of rule holidays overridden by real prints)."""
    printed = clean.index[clean.notna().any(axis=1).to_numpy()]
    wk = pd.bdate_range("1984-12-01", printed[-1] + pd.Timedelta(days=60))
    hol = rule_holidays(1984, printed[-1].year + 1)
    overridden = hol.intersection(printed)
    business = wk.difference(hol.difference(overridden))
    rolls = []
    for p in pd.period_range(v.index[0], v.index[-1], freq="M"):
        d25 = pd.Timestamp(p.year, p.month, 25)
        before = business[business < d25]
        ltd = before[-3] if d25 in business else before[-4]
        k = np.searchsorted(v.index.values, ltd.to_datetime64(), side="right")
        if 0 < k < len(v) and v.index[k] > ltd and v.index[k - 1] <= ltd:
            rolls.append(v.index[k])
    return pd.DatetimeIndex(sorted(set(rolls))), [d.date().isoformat() for d in overridden]


def curve_score(v: pd.DataFrame) -> pd.Series:
    """Per index day, evidence that contract identities changed since the previous index day
    (positive favours a roll); the same score as detect_roll_days, kept for adjudication."""
    a1 = np.log(v["c1"] / v["c2"]).to_numpy()
    a2 = np.log(v["c2"] / v["c3"]).to_numpy()
    a3 = np.log(v["c3"] / v["c4"]).to_numpy()
    no_roll = np.nansum(np.vstack([np.abs(a1[1:] - a1[:-1]), np.abs(a2[1:] - a2[:-1])]), axis=0)
    roll = np.nansum(np.vstack([np.abs(a1[1:] - a2[:-1]), np.abs(a2[1:] - a3[:-1])]), axis=0)
    return pd.Series(np.concatenate(([np.nan], no_roll - roll)), index=v.index)


def build_roll_index(v: pd.DataFrame, roll_days: pd.DatetimeIndex) -> pd.Series:
    """Hold contract 1 through its last print; on the first day of the new contract the return
    is c1(t) / c2(t') - 1 (the old contract 2 IS the new contract 1). Level 100 on day one."""
    c1, c2 = v["c1"].to_numpy(), v["c2"].to_numpy()
    is_roll = v.index.isin(roll_days)
    ret = np.zeros(len(v))
    ret[1:] = np.where(is_roll[1:], c1[1:] / c2[:-1], c1[1:] / c1[:-1]) - 1.0
    return pd.Series(100.0 * np.cumprod(1.0 + ret), index=v.index, name="my_crude_roll")


# =============================================================================================
# 3. Weekly signal, forward returns, statistics
# =============================================================================================
def friday_signal(v: pd.DataFrame, fridays: pd.DatetimeIndex) -> np.ndarray:
    """+1 if c1 > c2, -1 if c1 < c2, 0 if equal, NaN if no print within 4 days before Friday."""
    pos = np.searchsorted(v.index.values, fridays.values, side="right") - 1
    out = np.full(len(fridays), np.nan)
    ok = pos >= 0
    age = (fridays.values[ok] - v.index.values[pos[ok]]) / np.timedelta64(1, "D")
    good = np.flatnonzero(ok)[age <= MAX_SIGNAL_AGE]
    diff = v["c1"].to_numpy()[pos[good]] - v["c2"].to_numpy()[pos[good]]
    out[good] = np.sign(diff)
    return out


def forward_return(px: pd.Series, fridays: pd.DatetimeIndex, hold: int = HOLD_DAYS
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(entry row, return) of each Friday: entry = first close strictly after the Friday within
    7 days, exit = entry + hold rows. Row -1 / NaN when unavailable."""
    idx = px.index.values
    tgt = (fridays + pd.Timedelta(days=1)).values
    e = np.searchsorted(idx, tgt, side="left")
    ok = e < len(idx)
    lag = np.full(len(e), np.inf)
    lag[ok] = (idx[e[ok]] - tgt[ok]) / np.timedelta64(1, "D")
    e = np.where(ok & (lag <= MAX_ENTRY_LAG), e, -1)
    c = px.to_numpy()
    r = np.full(len(e), np.nan)
    good = (e >= 0) & (e + hold < len(c))
    r[good] = c[e[good] + hold] / c[e[good]] - 1.0
    return e, r


def count_episodes(setup: np.ndarray, elig: np.ndarray) -> int:
    """Runs of `setup` on the full weekly sequence that contain at least one eligible week."""
    start = setup & ~np.concatenate(([False], setup[:-1]))
    run_id = np.cumsum(start)
    return int(len(np.unique(run_id[setup & elig])))


def shifted_means(sig: np.ndarray, x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean of x over the signal positions circularly shifted by each offset."""
    pos = np.flatnonzero(sig)
    return x[(pos[None, :] + offsets[:, None]) % len(x)].mean(axis=1)


def cell(sig_w: np.ndarray, r: np.ndarray, dates: pd.DatetimeIndex, side: str = "long",
         mask: np.ndarray | None = None) -> dict:
    """Statistics of one side at h = 8 over the weeks in `mask` (default: all)."""
    valid = ~np.isnan(sig_w)
    if mask is not None:
        valid &= mask
    elig = valid & ~np.isnan(r)
    sgn = 1 if side == "long" else -1
    setup = valid & (np.nan_to_num(sig_w) == sgn)
    re = r[elig]
    sig = setup[elig]
    signed = sgn * re
    T, n = int(elig.sum()), int(sig.sum())
    base = float(re.mean())
    mean = float(signed[sig].mean())
    offs_all = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
    cnt = int(np.sum(shifted_means(sig, signed, offs_all) >= mean))
    offs_mc = np.random.default_rng(SEED).integers(MIN_SHIFT, T - MIN_SHIFT, size=K_MC,
                                                   endpoint=True)
    mc = (1 + int(np.sum(shifted_means(sig, signed, offs_mc) >= mean))) / (1 + K_MC)
    d = dates[elig]
    return dict(n=n, episodes=count_episodes(setup, elig), T=T, mean=mean, baseline=base,
                excess=mean - sgn * base, hit=float(np.mean(signed[sig] > 0)),
                perm_p_exact=(1 + cnt) / (1 + len(offs_all)), perm_p_mc=mc,
                period=f"{d.min().date()}..{d.max().date()}")


def block_bootstrap_ci(sig: np.ndarray, r: np.ndarray, block: int = 13, B: int = 2000,
                       seed: int = SEED) -> dict:
    """95% interval of the long excess by a circular moving-block bootstrap of the eligible
    weeks (13-week blocks), drawing block starts as rng.integers(0, T, size=(B, nblocks))."""
    T = len(r)
    nb = int(np.ceil(T / block))
    starts = np.random.default_rng(seed).integers(0, T, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(B, -1)[:, :T]
    stats = np.array([r[i][sig[i]].mean() - r[i].mean() if sig[i].any() else np.nan
                      for i in idx])
    stats = stats[~np.isnan(stats)]
    return dict(ci_lo=float(np.quantile(stats, 0.025)), ci_hi=float(np.quantile(stats, 0.975)))


# =============================================================================================
# 4. Walk-forward (learned side, annual refits from 1995, outcomes realised before the cutoff)
# =============================================================================================
def walk_forward_long(sig_w, r, entry_row, px, fridays) -> dict:
    """Each 1 January from 1995 the long side is kept for the coming year only if, among weeks
    whose 8-week exit is before that date, premium weeks beat the all-weeks baseline."""
    exit_date = np.full(len(fridays), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = ~np.isnan(r)
    exit_date[ok] = px.index.values[entry_row[ok] + HOLD_DAYS]
    wf = np.full(len(fridays), np.nan)
    years = range(1995, fridays[-1].year + 1)
    kept = {}
    for y in years:
        cut = np.datetime64(pd.Timestamp(y, 1, 1))
        hist = ok & ~np.isnan(sig_w) & (exit_date < cut)
        base = r[hist].mean()
        long_ex = r[hist & (sig_w == 1)].mean() - base
        kept[y] = bool(long_ex > 0)
        blk = (fridays.year == y) & ~np.isnan(sig_w)
        wf[blk] = np.where((sig_w[blk] == 1) & kept[y], 1.0, 0.0)
    res = cell(wf, r, fridays, "long")
    res["years_long_side_kept"] = int(sum(kept.values()))
    res["years_tested"] = len(kept)
    return res


# =============================================================================================
# 5. S1 / S2 strategies (SPEC2 rule 6), re-implemented
# =============================================================================================
def episode_events(sig_w: np.ndarray, entry_row: np.ndarray, sides: tuple) -> list[tuple]:
    """(entry row, side) at the first week with an entry of every setup episode."""
    ev = []
    s = np.nan_to_num(sig_w)
    for side in sides:
        setup = s == (1 if side == "long" else -1)
        start = setup & ~np.concatenate(([False], setup[:-1]))
        run = np.cumsum(start)
        for k in np.unique(run[setup]):
            rows = np.flatnonzero((run == k) & setup & (entry_row >= 0))
            if len(rows):
                ev.append((int(entry_row[rows[0]]), side))
    return sorted(ev)


def strategy_position(ev: list[tuple], close: np.ndarray, sma: np.ndarray, s2: bool
                      ) -> np.ndarray:
    """Daily position path. One trade at a time, held HOLD_DAYS rows; a signal arriving on or
    before the open trade's exit row is ignored. S2 waits up to 20 rows (entry row included,
    and only until the next event) for a close beyond the 20-day SMA and enters at that close."""
    n = len(close)
    pos = np.zeros(n)
    busy = -1
    for i, (p, side) in enumerate(ev):
        if p <= busy:
            continue
        entry = p
        if s2:
            hi = min(p + TRIG_WIN - 1, n - 1, (ev[i + 1][0] - 1) if i + 1 < len(ev) else n - 1)
            if hi < p:
                continue
            seg_c, seg_s = close[p:hi + 1], sma[p:hi + 1]
            cond = seg_c > seg_s if side == "long" else seg_c < seg_s
            hits = np.flatnonzero(cond)
            if not len(hits):
                continue
            entry = p + int(hits[0])
            if entry <= busy:
                continue
        exit_ = min(entry + HOLD_DAYS, n - 1)
        end = n if entry + HOLD_DAYS > n - 1 else exit_
        pos[entry:end] = 1.0 if side == "long" else -1.0
        busy = exit_
    return pos


def sharpe_of(pos: np.ndarray, ret: np.ndarray, start: int, dates: pd.DatetimeIndex,
              cost: float = COST) -> dict:
    """Daily net P&L from the start row: (1 + p(t-1) r(t)) (1 - cost |dp|) - 1; Sharpe = mean
    * obs-per-year / (sd * sqrt(obs-per-year)) over the days after the start day; CAGR over
    calendar time."""
    p_prev = np.concatenate(([0.0], pos[:-1]))
    p_prev[:start + 1] = 0.0
    dpos = np.abs(pos - p_prev)
    dpos[:start] = 0.0
    net = ((1.0 + p_prev * ret) * (1.0 - cost * dpos) - 1.0)[start:]
    years = (dates[-1] - dates[start]).days / 365.25
    x = net[1:]
    ppy = len(x) / years
    sd = np.std(x, ddof=1)
    eq = np.cumprod(1.0 + net)
    return dict(sharpe=float(np.mean(x) * ppy / (sd * np.sqrt(ppy))) if sd > 0 else np.nan,
                cagr=float(eq[-1] ** (1.0 / years) - 1.0))


def strategies(sig_w, entry_row, px, null_k: int = K_MC) -> dict:
    """S1/S2 x long-only/long+short with costs, buy & hold, random-timing null p (signal
    circularly shifted over the valid weeks, offsets from default_rng(11) on [26, T-26])."""
    close = px.to_numpy()
    ret = np.concatenate(([0.0], close[1:] / close[:-1] - 1.0))
    sma = pd.Series(close).rolling(SMA_N, min_periods=SMA_N).mean().to_numpy()
    valid = ~np.isnan(sig_w)
    start = int(entry_row[np.flatnonzero(valid & (entry_row >= 0))[0]])
    dates = px.index
    bh = np.zeros(len(close))
    bh[start:] = 1.0
    out = {"buy_hold": sharpe_of(bh, ret, start, dates, cost=0.0)}

    def run(sig):
        res = {}
        for mode, sides in (("long_only", ("long",)), ("long_short", ("long", "short"))):
            ev = episode_events(sig, entry_row, sides)
            for name, s2 in (("S1", False), ("S2", True)):
                res[f"{name}_{mode}"] = strategy_position(ev, close, sma, s2)
        return res

    real = run(sig_w)
    for k, pos in real.items():
        out[k] = sharpe_of(pos, ret, start, dates)
    if null_k:
        vidx = np.flatnonzero(valid)
        T = len(vidx)
        offs = np.random.default_rng(SEED).integers(MIN_SHIFT, T - MIN_SHIFT, size=null_k,
                                                    endpoint=True)
        null = {k: [] for k in real}
        for o in offs:
            sh = sig_w.copy()
            sh[vidx] = np.roll(sig_w[vidx], int(o))
            for k, pos in run(sh).items():
                null[k].append(sharpe_of(pos, ret, start, dates)["sharpe"])
        for k in real:
            v = np.asarray(null[k])
            out[k]["null_p"] = float((1 + np.sum(v >= out[k]["sharpe"])) / (1 + null_k))
    return out


# =============================================================================================
# 6. Main
# =============================================================================================
def main() -> None:
    raw = load_contracts()
    clean, placeholders = drop_placeholder_rows(raw)
    v = clean[clean["c1"].notna() & clean["c2"].notna()]          # index days
    roll_days, overridden = rule_roll_days(clean, v)
    px = build_roll_index(v, roll_days)
    spot = load_spot()

    fridays = pd.date_range(FIRST_FRIDAY, LAST_FRIDAY, freq="W-FRI")
    sig_w = friday_signal(v, fridays)
    entry_row, r8 = forward_return(px, fridays)
    _, r8_spot = forward_return(spot, fridays)

    # Diagnostic only: a purely data-driven roll calendar (argmax of the curve-shape score in
    # each month). It mis-dates months whose curve is close to linear (see the audit notes),
    # so its numbers measure how sensitive the result is to roll-dating errors.
    naive_days, naive_tab = detect_roll_days(v)
    _, r8_naive = forward_return(build_roll_index(v, naive_days), fridays)
    score = curve_score(v)

    res: dict = dict(
        roll_calendar=dict(method="NYMEX expiry rule on an audit-written holiday list, "
                                  "rule holidays with real EIA prints count as business days",
                           n_index_days=int(len(v)), n_roll_days=int(len(roll_days)),
                           placeholder_rows_removed=len(placeholders),
                           rule_holidays_with_real_prints=overridden),
        signal_counts=dict(valid=int((~np.isnan(sig_w)).sum()), long=int((sig_w == 1).sum()),
                           short=int((sig_w == -1).sum()), flat=int((sig_w == 0).sum())))
    res["naive_data_driven_calendar_diagnostic"] = dict(
        months_differing_from_rule=int(len(roll_days.difference(naive_days))),
        primary_long_h8=cell(sig_w, r8_naive, fridays, "long"))

    # Disagreements with the harness calendar, with the curve-shape evidence for each side.
    try:
        sys.path.insert(0, str(PKG / "battery" / "harness"))
        import fund_harness as fh                               # comparison only
        ri = fh.build_crude_roll_index(0)
        h_days = ri.detail.index[ri.detail["roll"].to_numpy()]
        rows = []
        for d in h_days.symmetric_difference(roll_days):
            per = d.to_period("M")
            mine = roll_days[roll_days.to_period("M") == per]
            theirs = h_days[h_days.to_period("M") == per]
            rows.append(dict(month=str(per),
                             audit=[x.date().isoformat() for x in mine],
                             harness=[x.date().isoformat() for x in theirs],
                             score_audit=[round(float(score[x]), 5) for x in mine],
                             score_harness=[round(float(score[x]), 5) for x in theirs]))
        dis = pd.DataFrame(rows).drop_duplicates("month") if rows else pd.DataFrame()
        res["calendar_vs_harness"] = dict(
            same_index_days=bool(ri.detail.index.equals(v.index)),
            months_differing=int(len(dis)), detail=dis.to_dict("records"),
            max_abs_level_ratio_diff=float(np.max(np.abs(
                px.to_numpy() / ri.level.reindex(px.index).to_numpy() - 1.0))))
    except Exception as exc:                                     # never block the audit
        res["calendar_vs_harness"] = dict(error=repr(exc))

    # Primary and its out-of-sample cuts
    res["primary_long_h8"] = cell(sig_w, r8, fridays, "long")
    elig = ~np.isnan(sig_w) & ~np.isnan(r8)
    res["primary_long_h8"].update(block_bootstrap_ci((sig_w[elig] == 1), r8[elig]))
    res["primary_net_components"] = dict(short=cell(sig_w, r8, fridays, "short"))
    ed = fridays[elig]
    mid = ed[0] + (ed[-1] - ed[0]) / 2
    res["split_date"] = mid.date().isoformat()
    res["H1_long"] = cell(sig_w, r8, fridays, "long", np.asarray(fridays <= mid))
    res["H2_long"] = cell(sig_w, r8, fridays, "long", np.asarray(fridays > mid))
    res["H1_short"] = cell(sig_w, r8, fridays, "short", np.asarray(fridays <= mid))
    res["H2_short"] = cell(sig_w, r8, fridays, "short", np.asarray(fridays > mid))
    res["pre2010_long"] = cell(sig_w, r8, fridays, "long", np.asarray(fridays < "2010-01-01"))
    res["from2010_long"] = cell(sig_w, r8, fridays, "long", np.asarray(fridays >= "2010-01-01"))
    res["spot_long"] = cell(sig_w, r8_spot, fridays, "long")
    res["walk_forward_long"] = walk_forward_long(sig_w, r8, entry_row, px, fridays)

    # Strategies: full window, and the signal restricted to 2010+ (post hoc)
    res["strategy_full"] = strategies(sig_w, entry_row, px)
    sig_2010 = np.where(np.asarray(fridays >= "2010-01-01"), sig_w, np.nan)
    res["strategy_from2010_posthoc"] = strategies(sig_2010, entry_row, px)

    # Comparison with the published numbers
    pub = json.load(open(PREMIUM_RESULTS, encoding="utf-8"))
    prim = pub["primary"]
    sh = pd.DataFrame(pub["oos"]["split_half"])
    sh8 = sh[(sh["horizon_w"] == 8) & (sh["side"] == "long")].set_index("half")
    st = pd.DataFrame(pub["strategy"])
    st = st[st["series"] == "crude_roll"].set_index("strategy")
    res["published"] = dict(
        primary={k: prim[k] for k in ("n", "episodes", "T", "mean", "baseline", "excess", "hit",
                                      "perm_p", "perm_p_exact", "period")},
        H1_excess=float(sh8.loc["H1", "excess"]), H2_excess=float(sh8.loc["H2", "excess"]),
        H2_p=float(sh8.loc["H2", "perm_p_exact"]),
        strategy_sharpe={k: float(st.loc[k, "sharpe"]) for k in st.index},
        strategy_null_p={k: (None if pd.isna(st.loc[k, "null_p_sharpe"])
                             else float(st.loc[k, "null_p_sharpe"])) for k in st.index})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("roll_calendar", "calendar_vs_harness",
                                          "naive_data_driven_calendar_diagnostic",
                                          "signal_counts", "primary_long_h8",
                                          "split_date", "H1_long", "H2_long", "pre2010_long",
                                          "from2010_long", "spot_long", "walk_forward_long",
                                          "strategy_full", "strategy_from2010_posthoc",
                                          "published")}, indent=1, default=str))


if __name__ == "__main__":
    main()

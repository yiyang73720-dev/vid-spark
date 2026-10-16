#!/usr/bin/env python3
"""
seasonal -- walk-forward seasonal-tendency signals (Larry Williams seasonality, SPEC2 battery).

Definitions are fixed in PREREG.md (section 2). In short, for a decision Friday d, lookback L
years and horizon h weeks:

  analog_k(d) = the Friday nearest (+-3 days, unique) to d's calendar date k years earlier
                (k = 1..L; Feb 29 -> Feb 28),
  R_k         = h-week forward return of analog_k on the ESTIMATION series (entry = first close
                strictly after the analog Friday, exit 5h trading days later: fh.forward_table),
  hit = #{R_k > 0} / L,  mean = mean_k R_k   (undefined unless all L analogs exist),
  signal = +1 if hit >= th and mean > 0;  -1 if hit <= 1-th and mean < 0;  0 otherwise.

Month-of-year version: bucket = calendar month of d; each prior year contributes the mean of R
over all Fridays of that month (NaN unless all exist); hit/mean across the L yearly values.

No-look-ahead is guaranteed by construction (only prior years) AND asserted: every analog's
exit date must be strictly before d. Two independent checks live here as well:
  * signals_walk_forward(): the same signal rebuilt through fh.walk_forward with WEEKLY refit,
    where fit() only sees forward-table rows whose exit date is before the cutoff;
  * truncation_test(): signals recomputed from estimation series truncated at a cut date X must
    be identical for all decision dates <= X.

Pooled evaluation (net side) and the off-calendar permutation p are implemented here with the
same conventions as fund_harness (see the function docstrings).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))   # battery/harness
import fund_harness as fh  # noqa: E402

WEEKS_PER_YEAR = 365.25 / 7.0      # 52.18: shift distance that keeps the calendar aligned
OFFCAL_TOL = 3                     # off-calendar null: drop offsets within +-3 weeks of k*52.18


# ========================================================================================
# 1. Estimation tables (forward returns of every Friday of the estimation series)
# ========================================================================================
@dataclass
class EstTable:
    """Forward returns of every Friday on an estimation series.
    r[h], exit[h]: numpy arrays aligned with `fridays` (NaN / NaT where not available)."""
    name: str
    fridays: pd.DatetimeIndex
    r: dict
    exit: dict

    def lookup(self, dates: pd.DatetimeIndex, h: int) -> tuple[np.ndarray, np.ndarray]:
        """(returns, exit dates as datetime64) of the given Fridays; NaN/NaT if not in table."""
        pos = self.fridays.get_indexer(dates)
        ok = pos >= 0
        r = np.full(len(dates), np.nan)
        e = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
        r[ok] = self.r[h][pos[ok]]
        e[ok] = self.exit[h][pos[ok]]
        return r, e


def est_table(series: pd.Series, name: str, horizons=(4, 8, 13)) -> EstTable:
    """Forward table (timing="price") of every Friday from one week before the series' first
    close to its last close."""
    s = series[series.notna() & (series > 0)].sort_index()
    fri = fh.weekly_decision_dates(s.index[0] - pd.Timedelta(days=7), s.index[-1])
    ft = fh.forward_table(fri, s, "price", horizons)
    r = {h: ft[f"r{h}"].to_numpy(float) for h in horizons}
    ex = {h: ft[f"exit_date{h}"].to_numpy("datetime64[ns]") for h in horizons}
    # a return without an exit date (or vice versa) would be a harness bug
    for h in horizons:
        if not np.array_equal(np.isnan(r[h]), np.isnat(ex[h])):
            raise AssertionError(f"{name}: r{h} / exit_date{h} mismatch")
    return EstTable(name=name, fridays=pd.DatetimeIndex(fri), r=r, exit=ex)


def splice_tables(primary: EstTable, fallback: EstTable, name: str) -> EstTable:
    """Per-Friday splice: use `primary`'s forward return where it exists, else `fallback`'s
    (extended-history variant: v1 series first, roll-adjusted futures index before 2000)."""
    fri = primary.fridays.union(fallback.fridays)
    r, ex = {}, {}
    for h in primary.r:
        rp, ep = primary.lookup(fri, h)
        rf, ef = fallback.lookup(fri, h)
        use_p = ~np.isnan(rp)
        r[h] = np.where(use_p, rp, rf)
        ex[h] = np.where(use_p, ep, ef)
    return EstTable(name=name, fridays=fri, r=r, exit=ex)


def truncate_table(t: EstTable, series: pd.Series, cut, horizons) -> EstTable:
    """Rebuild a table from the series truncated at `cut` (used by the truncation test)."""
    s = series[series.index <= pd.Timestamp(cut)]
    return est_table(s, f"{t.name}<= {pd.Timestamp(cut).date()}", horizons)


# ========================================================================================
# 2. Analogs and seasonal statistics
# ========================================================================================
def nearest_friday(dates) -> pd.DatetimeIndex:
    """The unique Friday within +-3 days of each date."""
    d = pd.DatetimeIndex(dates)
    delta = (4 - d.weekday.to_numpy()) % 7
    delta = np.where(delta > 3, delta - 7, delta)
    return d + pd.to_timedelta(delta, unit="D")


def years_back(dates, k: int) -> pd.DatetimeIndex:
    """Same month/day k years earlier (Feb 29 -> Feb 28 in non-leap years)."""
    d = pd.DatetimeIndex(dates)
    y = d.year.to_numpy() - k
    m = d.month.to_numpy()
    day = d.day.to_numpy()
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    day = np.where((m == 2) & (day == 29) & ~leap, 28, day)
    return pd.DatetimeIndex(pd.to_datetime(pd.DataFrame(dict(year=y, month=m, day=day))))


def analog_fridays(dec, L: int) -> list[pd.DatetimeIndex]:
    """[analog_k(d) for k = 1..L] for every decision date."""
    return [nearest_friday(years_back(dec, k)) for k in range(1, L + 1)]


def _assert_past(exits: np.ndarray, dec: pd.DatetimeIndex, what: str):
    """Every non-missing analog exit must be strictly before its decision date."""
    d = dec.to_numpy("datetime64[ns]")
    ok = ~np.isnat(exits)
    bad = ok & ~(exits < d[:, None] if exits.ndim == 2 else exits < d)
    if bad.any():
        raise AssertionError(f"LOOK-AHEAD in {what}: {int(bad.sum())} analog exits >= decision")


def week_stats(dec, tab: EstTable, h: int, L: int) -> dict:
    """Week-of-year (calendar-anchored) analog statistics. Returns dict(npos, mean, complete,
    R (n x L analog returns), A (analog Fridays, k = 1..L))."""
    dec = pd.DatetimeIndex(dec)
    A = analog_fridays(dec, L)
    R = np.full((len(dec), L), np.nan)
    E = np.full((len(dec), L), np.datetime64("NaT"), dtype="datetime64[ns]")
    for j, a in enumerate(A):
        R[:, j], E[:, j] = tab.lookup(a, h)
    _assert_past(E, dec, f"week_stats({tab.name}, h={h}, L={L})")
    complete = ~np.isnan(R).any(axis=1)
    npos = np.where(complete, (R > 0).sum(axis=1), -1)
    with np.errstate(invalid="ignore"):
        mean = np.where(complete, np.nanmean(np.where(complete[:, None], R, 0.0), axis=1), np.nan)
    return dict(npos=npos, mean=mean, complete=complete, R=R, A=A, L=L)


def month_stats(dec, tab: EstTable, h: int, L: int) -> dict:
    """Month-of-year analog statistics: per prior year, the mean h-week return of all Fridays
    in d's calendar month (NaN unless all of them have a return)."""
    dec = pd.DatetimeIndex(dec)
    f = pd.DataFrame(dict(year=tab.fridays.year, month=tab.fridays.month, r=tab.r[h],
                          ex=tab.exit[h]))
    g = f.groupby(["year", "month"])
    ym = pd.DataFrame(dict(mean=g["r"].mean(), n_nan=g["r"].apply(lambda x: int(x.isna().sum())),
                           last_exit=g["ex"].max()))
    ym.loc[ym["n_nan"] > 0, "mean"] = np.nan
    key = pd.MultiIndex.from_frame(ym.reset_index()[["year", "month"]])
    vals = ym["mean"].to_numpy(float)
    lex = ym["last_exit"].to_numpy("datetime64[ns]")
    R = np.full((len(dec), L), np.nan)
    E = np.full((len(dec), L), np.datetime64("NaT"), dtype="datetime64[ns]")
    for k in range(1, L + 1):
        q = pd.MultiIndex.from_arrays([dec.year.to_numpy() - k, dec.month.to_numpy()])
        pos = key.get_indexer(q)
        ok = pos >= 0
        R[ok, k - 1] = vals[pos[ok]]
        E[ok, k - 1] = np.where(np.isnan(vals[pos[ok]]), np.datetime64("NaT"), lex[pos[ok]])
    _assert_past(E, dec, f"month_stats({tab.name}, h={h}, L={L})")
    complete = ~np.isnan(R).any(axis=1)
    npos = np.where(complete, (R > 0).sum(axis=1), -1)
    mean = np.where(complete, np.where(complete[:, None], R, 0.0).mean(axis=1), np.nan)
    return dict(npos=npos, mean=mean, complete=complete, R=R, L=L)


def signal_from_stats(st: dict, th: int) -> np.ndarray:
    """+1 / 0 / -1 (NaN where the estimate is undefined). th in percent; integer comparisons:
    long  100*npos >= th*L and mean > 0;  short 100*npos <= (100-th)*L and mean < 0."""
    L = st["L"]
    npos = st["npos"]
    mean = st["mean"]
    lo = st["complete"] & (100 * npos >= th * L) & (mean > 0)
    sh = st["complete"] & (100 * npos <= (100 - th) * L) & (mean < 0)
    sig = np.where(lo, 1.0, np.where(sh, -1.0, 0.0))
    return np.where(st["complete"], sig, np.nan)


def seasonal_signal(dec, tab: EstTable, h: int, L: int, th: int, bucket: str = "week"
                    ) -> np.ndarray:
    st = week_stats(dec, tab, h, L) if bucket == "week" else month_stats(dec, tab, h, L)
    return signal_from_stats(st, th)


# ========================================================================================
# 3. Independent look-ahead checks
# ========================================================================================
def signals_walk_forward(dec, tab: EstTable, h: int, L: int, th: int) -> pd.DataFrame:
    """Week-bucket signal rebuilt through fh.walk_forward with WEEKLY refit. fit() receives the
    history strictly before the block and the cutoff; it keeps ONLY outcomes whose exit date is
    before the cutoff (returns known at decision time) and builds a {Friday: return} map.
    apply() looks up the L analogs of the block's decision date in that map. Shares no code
    with week_stats: the calendar arithmetic is redone with the standard-library datetime."""
    import calendar
    import datetime as dt

    dec = pd.DatetimeIndex(dec)
    frame = pd.DataFrame(dict(decision_date=dec))
    fr = pd.DataFrame(dict(friday=tab.fridays, r=tab.r[h], ex=tab.exit[h])).dropna()
    fr = fr.sort_values("ex", kind="mergesort")
    ex_sorted = fr["ex"].to_numpy("datetime64[ns]")
    fri_py = [x.date() for x in pd.DatetimeIndex(fr["friday"])]
    r_py = fr["r"].to_numpy(float)

    def analog(d: dt.date, k: int) -> dt.date:
        y = d.year - k
        day = 28 if (d.month == 2 and d.day == 29 and not calendar.isleap(y)) else d.day
        x = dt.date(y, d.month, day)
        delta = (4 - x.weekday()) % 7          # days forward to the next Friday (0..6)
        if delta > 3:
            delta -= 7                         # nearer Friday is behind
        return x + dt.timedelta(days=delta)

    def fit(hist: pd.DataFrame, cutoff: pd.Timestamp):
        n_known = int(np.searchsorted(ex_sorted, np.datetime64(cutoff, "ns"), side="left"))
        return dict(zip(fri_py[:n_known], r_py[:n_known]))     # outcomes with exit < cutoff

    def apply(params: dict, block: pd.DataFrame):
        out = []
        for d in pd.DatetimeIndex(block["decision_date"]):
            rs = []
            for k in range(1, L + 1):
                rs.append(params.get(analog(d.date(), k), np.nan))
            rs = np.array(rs, float)
            if np.isnan(rs).any():
                out.append(np.nan)
                continue
            npos, mean = int((rs > 0).sum()), float(rs.mean())
            if 100 * npos >= th * L and mean > 0:
                out.append(1.0)
            elif 100 * npos <= (100 - th) * L and mean < 0:
                out.append(-1.0)
            else:
                out.append(0.0)
        return out

    return fh.walk_forward(frame, fit, apply, first_test=dec[0], refit="W-FRI", last=dec[-1])


def truncation_test(dec, series: pd.Series, name: str, h: int, L: int, th: int, bucket: str,
                    cuts) -> list[dict]:
    """For each cut date X: signals from the estimation series truncated at X must equal the
    full-data signals on every decision date <= X (value and definedness)."""
    dec = pd.DatetimeIndex(dec)
    full = seasonal_signal(dec, est_table(series, name, (h,)), h, L, th, bucket)
    out = []
    for X in cuts:
        X = pd.Timestamp(X)
        m = dec <= X
        tab_x = est_table(series[series.index <= X], f"{name}<=X", (h,))
        part = seasonal_signal(dec[m], tab_x, h, L, th, bucket)
        same = np.array_equal(np.nan_to_num(part, nan=9.0), np.nan_to_num(full[m], nan=9.0))
        out.append(dict(cut=X.date().isoformat(), n_decisions=int(m.sum()),
                        n_defined=int(np.isfinite(part).sum()), identical=bool(same)))
    return out


# ========================================================================================
# 4. Pooled NET evaluation and off-calendar permutation p
# ========================================================================================
def _prep_part(sf: pd.DataFrame, series, timing: str, h: int):
    s, sname = fh._resolve(series, None)
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    ep = fh.entry_positions(sf["decision_date"], s.index, timing)
    r = fh.forward_returns(s, ep, h)
    valid = sf["valid"].to_numpy(bool)
    elig = valid & ~sf["excluded"].to_numpy(bool) & ~np.isnan(r)
    s_full = np.where(valid, sf["signal"].to_numpy(np.int64), 0)
    return dict(sname=sname, s_full=s_full, elig=elig, s_e=s_full[elig].astype(float),
                r_e=r[elig], dates=pd.DatetimeIndex(sf["decision_date"])[elig])


def pooled_net(parts, h: int, *, k: int = fh.K_PERM, seed: int = fh.SEED,
               min_shift: int = fh.MIN_SHIFT, labels=None) -> dict:
    """Pooled NET cell across markets (fh.evaluate_pooled conventions, extended to net):
    pooled mean of setup_sign * r over all setup weeks of all parts; excess = mean of
    (signed r - setup_sign * own-part baseline); hit = share of signed r > 0. Permutation: one
    default_rng(seed); each part (in order) draws k offsets on [min_shift, T_part - min_shift]
    and circularly shifts its +1/-1 setup vector over its eligible weeks; statistic = pooled
    mean. perm_p = (1 + #{>= obs})/(1 + k); perm_p_lower = (1 + #{<= obs})/(1 + k)."""
    P = [_prep_part(sf, ser, timing, h) for sf, ser, timing in parts]
    signed, exc, episodes, T = [], [], 0, 0
    for p in P:
        sig = p["s_e"] != 0
        b = float(p["r_e"].mean()) if len(p["r_e"]) else np.nan
        signed.append(p["s_e"][sig] * p["r_e"][sig])
        exc.append(p["s_e"][sig] * (p["r_e"][sig] - b))
        ids = fh._episode_ids_values(p["s_full"])[p["elig"]]
        episodes += int(len(np.unique(ids[sig])))
        T += len(p["r_e"])
    sr = np.concatenate(signed) if signed else np.array([])
    n = int(len(sr))
    row = dict(filter="seasonality", market="pooled", series="+".join(p["sname"] for p in P),
               timing="+".join(sorted({t for _, _, t in parts})), side="net", horizon_w=int(h),
               n=n, episodes=episodes, T=T, baseline=np.nan, median=np.nan, mean=np.nan,
               excess=np.nan, hit=np.nan, perm_p=np.nan, perm_p_lower=np.nan,
               perm_p_exact=np.nan, perm_p_exact_raw=np.nan,
               n_long=int(sum((p["s_e"] > 0).sum() for p in P)),
               n_short=int(sum((p["s_e"] < 0).sum() for p in P)),
               period=fh._period(pd.DatetimeIndex(np.concatenate(
                   [p["dates"].to_numpy() for p in P])) if P else []))
    row.update(labels or {})
    if n == 0:
        return row
    obs = float(sr.mean())
    rng = np.random.default_rng(seed)
    sums = np.zeros(k)
    for p in P:
        Tp = len(p["r_e"])
        offs = rng.integers(min_shift, Tp - min_shift, size=k, endpoint=True)
        pos = np.flatnonzero(p["s_e"] != 0)
        if len(pos):
            sums += (p["s_e"][pos][None, :] * p["r_e"][(pos[None, :] + offs[:, None]) % Tp]).sum(1)
    null = sums / n
    row.update(mean=obs, median=float(np.median(sr)), hit=float((sr > 0).mean()),
               excess=float(np.concatenate(exc).mean()),
               perm_p=(1 + int(np.sum(null >= obs))) / (1 + k),
               perm_p_lower=(1 + int(np.sum(null <= obs))) / (1 + k))
    return row


def offcal_offsets(T: int, min_shift: int = fh.MIN_SHIFT, tol: int = OFFCAL_TOL) -> np.ndarray:
    """All offsets in [min_shift, T - min_shift] that are NOT within tol weeks of a multiple of
    52.18 weeks, neither forward (o) nor through the wrap (T - o): such shifts would keep a
    seasonal signal aligned with the calendar."""
    o = np.arange(min_shift, T - min_shift + 1)

    def near(x):
        return np.abs(x - np.round(x / WEEKS_PER_YEAR) * WEEKS_PER_YEAR) <= tol
    return o[~(near(o) | near(T - o))]


def offcal_p(sf: pd.DataFrame, series, timing: str, h: int, side: str) -> dict:
    """Exhaustive circular-shift p over the off-calendar offsets only, with the +1 correction:
    (1 + #{shifted >= obs}) / (1 + #offsets). Also the lower tail."""
    p = _prep_part(sf, series, timing, h)
    s_e, r_e = p["s_e"], p["r_e"]
    if side == "long":
        vals_all = (s_e > 0).astype(float)
    elif side == "short":
        vals_all = -(s_e < 0).astype(float)
    else:
        vals_all = s_e
    pos = np.flatnonzero(vals_all != 0)
    T = len(r_e)
    if len(pos) == 0 or T < 2 * fh.MIN_SHIFT + 1:
        return dict(perm_p_offcal=np.nan, perm_p_offcal_lower=np.nan, n_offcal_offsets=0)
    vals = vals_all[pos]
    obs = float((vals * r_e[pos]).mean())
    offs = offcal_offsets(T)
    sh = (vals[None, :] * r_e[(pos[None, :] + offs[:, None]) % T]).mean(1)
    return dict(perm_p_offcal=(1 + int(np.sum(sh >= obs))) / (1 + len(offs)),
                perm_p_offcal_lower=(1 + int(np.sum(sh <= obs))) / (1 + len(offs)),
                n_offcal_offsets=int(len(offs)))


def pooled_by_year(parts, h: int) -> pd.DataFrame:
    """Calendar-year breakdown of the pooled NET cell (own full-sample baselines)."""
    rows = []
    for sf, ser, timing in parts:
        p = _prep_part(sf, ser, timing, h)
        b = float(p["r_e"].mean())
        sig = p["s_e"] != 0
        rows.append(pd.DataFrame(dict(year=p["dates"][sig].year, market=p["sname"],
                                      signed=p["s_e"][sig] * p["r_e"][sig],
                                      excess=p["s_e"][sig] * (p["r_e"][sig] - b))))
    df = pd.concat(rows, ignore_index=True)
    g = df.groupby("year")
    return pd.DataFrame(dict(n=g.size(), mean=g["signed"].mean(), excess=g["excess"].mean(),
                             hit=g["signed"].apply(lambda x: float((x > 0).mean())))).reset_index()


def pooled_split_half(parts, h: int, **kw) -> list[dict]:
    """Pooled net cell on the two halves split at ONE common date: the midpoint of the union of
    the parts' eligible decision dates. Each half re-evaluated (own baselines, own shifts)."""
    first, last = [], []
    for sf, ser, timing in parts:
        el = fh.eligible_dates(sf, ser, timing=timing, horizon=h)
        first.append(el[0])
        last.append(el[-1])
    mid = min(first) + (max(last) - min(first)) / 2
    out = []
    for half in ("H1", "H2"):
        hp = []
        for sf, ser, timing in parts:
            q = sf.copy()
            d = pd.DatetimeIndex(q["decision_date"])
            m = np.asarray(d <= mid) if half == "H1" else np.asarray(d > mid)
            q["valid"] = q["valid"].to_numpy(bool) & m
            q["signal"] = np.where(q["valid"], q["signal"], 0).astype(np.int8)
            if fh.eligible_dates(q, ser, timing=timing, horizon=h).size > 2 * fh.MIN_SHIFT:
                hp.append((q, ser, timing))
        row = pooled_net(hp, h, **kw)
        row.update(half=half, split_date=mid.date().isoformat(), n_markets=len(hp))
        out.append(row)
    return out


# ========================================================================================
# 5. Equal-weight portfolio of per-market strategies (rule 6, pooled primary)
# ========================================================================================
def _portfolio(nets: list[pd.Series]) -> pd.Series:
    """Daily equal-weight mean over the markets whose window is active on each date. A market
    without a close on a date (inside its window) contributes 0 that day."""
    idx = nets[0].index
    for n in nets[1:]:
        idx = idx.union(n.index)
    M = np.full((len(idx), len(nets)), np.nan)
    for j, n in enumerate(nets):
        a = n.reindex(idx)
        inside = (idx >= n.index[0]) & (idx <= n.index[-1])
        M[:, j] = np.where(inside, a.fillna(0.0).to_numpy(), np.nan)
    return pd.Series(np.nanmean(M, axis=1), index=idx)


def portfolio_simulation(parts, names, *, hold=fh.HOLD, win=fh.TRIG_WIN, cost=fh.COST,
                         null_k=fh.K_PERM, seed=fh.SEED, min_shift=fh.MIN_SHIFT) -> dict:
    """EW portfolio of S1/S2 (long+short, long-only) over the parts, vs EW buy & hold.
    Null: every market's signal circularly shifted over its valid weeks by the SAME offsets
    fh.simulate draws for that market (default_rng(seed) per market, offsets on
    [min_shift, T - min_shift]); null Sharpe of the portfolio per draw;
    p = (1 + #{null >= real}) / (1 + null_k)."""
    sl = fh.sl
    ctxs = []
    for (sf, ser, timing), nm in zip(parts, names):
        s, _ = fh._resolve(ser, None)
        ctxs.append(fh.make_ctx(sf, s, timing, nm))

    def nets_for(sigs):
        res = [fh._run(c, sg, hold, win, {}) for c, sg in zip(ctxs, sigs)]
        return {st: _portfolio([sl.daily_net(c, r[st][0], cost) for c, r in zip(ctxs, res)])
                for st in fh.STRATEGIES}

    real = nets_for([c.idx for c in ctxs])
    bh = _portfolio([sl.daily_net(c, np.where(np.arange(c.n) >= c.start, 1.0, 0.0), 0.0)
                     for c in ctxs])
    bh_m = sl.perf_metrics(bh)
    years = (bh.index[-1] - bh.index[0]).days / 365.25
    null = {st: np.empty(null_k) for st in fh.STRATEGIES}
    if null_k:
        offs = [np.random.default_rng(seed).integers(min_shift, int(c.idx_defined.sum())
                                                     - min_shift, size=null_k, endpoint=True)
                for c in ctxs]
        for i in range(null_k):
            nn = nets_for([fh._shift_signal(c, int(o[i])) for c, o in zip(ctxs, offs)])
            for st in fh.STRATEGIES:
                null[st][i] = sl.sharpe_only(nn[st].to_numpy(), years)
    rows = []
    for st in fh.STRATEGIES:
        m = sl.perf_metrics(real[st])
        row = dict(filter="seasonality", market="EW7_portfolio", strategy=st, **m,
                   **{f"bh_{kk}": bh_m[kk] for kk in ("ann_return", "ann_vol", "sharpe",
                                                       "max_dd", "total_return")})
        if null_k:
            v = null[st]
            row.update(null_sharpe_median=float(np.median(v)),
                       null_sharpe_p05=float(np.quantile(v, 0.05)),
                       null_sharpe_p95=float(np.quantile(v, 0.95)),
                       null_p_sharpe=(1 + int(np.sum(v >= m["sharpe"]))) / (1 + null_k))
        rows.append(row)
    rows.append(dict(filter="seasonality", market="EW7_portfolio", strategy="buy_hold", **bh_m))
    equity = pd.DataFrame({st: (1.0 + real[st]).cumprod() for st in fh.STRATEGIES})
    equity["buy_hold"] = (1.0 + bh).cumprod()
    return dict(metrics=pd.DataFrame(rows), equity=equity, null=null)

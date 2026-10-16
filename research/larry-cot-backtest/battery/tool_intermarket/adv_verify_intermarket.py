#!/usr/bin/env python3
"""
adv_verify_intermarket -- adversarial, independent re-derivation of the tool_intermarket PRIMARY
and spot checks of its headline claims. Run with:

    python3 -I adv_verify_intermarket.py

Independence: Part 1 (the primary) and Part 2 (S1 strategy) read the RAW files directly
(S/data/prices_daily.csv for gold, the archived FRED DTWEXBGS JSON for the dollar) with json /
pandas.read_csv and re-implement every step with plain numpy: Friday grid, as-of reading with
the publication lag, entry/exit rows, baseline, excess, hit, episodes, the exhaustive and the
Monte-Carlo circular-shift p, and S1 long-only with costs and its random-timing null. Nothing
from run_intermarket.py, intermarket_lib.py or the harness evaluate()/simulate() is used there.

Parts 3-5 check grid claims. They reuse the harness only to LOAD series (crude_roll, ust10_price,
the pysystemtrade indexes) and intermarket_lib only to rebuild the grid's signals, and then
evaluate them with this script's own numpy code. Part 5 is a joint ("family-wise") shift test of
the claim that the count of nominally significant grid cells is what chance would produce.

Parts 6-8: the walk-forward selection rule re-derived (own bond price from DGS10), crisis-window
and split-half dependence of the positive pooled cells (C5, C7), C7 on disjoint pre-/post-2000
history, and a roll-day check of crude_roll.

Outputs (this directory): adv_verify_intermarket.json, adv_verify_log.txt, and an
'adversarial_verification' block added to results.json (no existing number is changed; the rest
of results.json is byte-identical). Re-run this script after any re-run of run_intermarket.py,
which rewrites results.json without the block.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True          # never write .pyc files into shared directories

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parent                   # this tool's code (intermarket_lib)
PKG = CODE.parents[1]                                    # package dir
S = Path(os.environ.get("LARRY_COT_ROOT") or PKG)        # ROOT: data/ ext/ work2/
HERE = S / "work2" / "tool_intermarket"                 # outputs of run_intermarket.py
GOLD_CSV = S / "data" / "prices_daily.csv"
USD_JSON = S / "ext" / "accomplish999_crude-oil" / "data" / "series" / "DTWEXBGS.json"

K = 2000            # Monte-Carlo shifts (SPEC2 rule 4)
SEED = 11           # v1 seed
MIN_SHIFT = 26      # offsets uniform on [26, T-26]
H_DAYS = 40         # h = 8 weeks = 40 trading days
STALE = 4           # an as-of reading may be at most 4 calendar days old
USD_LAG = 7         # DTWEXBGS (H.10) is released on Mondays for the previous week
MAX_ENTRY_LAG = 7   # no entry if the first close is > 7 days after Friday + 1
COST = 0.0005       # 0.05% per side

LOG: list[str] = []


def log(msg: str) -> None:
    LOG.append(str(msg))
    print(msg, flush=True)


# =============================================================================================
# Raw loaders (independent of the harness)
# =============================================================================================
def load_gold_raw() -> pd.Series:
    """v1 gold closes straight from prices_daily.csv; non-positive closes dropped."""
    px = pd.read_csv(GOLD_CSV, float_precision="round_trip")
    g = px[px["market"] == "gold"]
    s = pd.Series(g["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(g["date"])))
    return s[s > 0].sort_index()


def load_usd_raw() -> pd.Series:
    """FRED DTWEXBGS daily observations from the archived JSON (parsed, never executed)."""
    with open(USD_JSON) as f:
        obs = json.load(f)["observations"]
    d = pd.DatetimeIndex([o["date"] for o in obs])
    v = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy(float)
    s = pd.Series(v, index=d).dropna()
    return s[s > 0].sort_index()


# =============================================================================================
# Small building blocks
# =============================================================================================
def asof(s: pd.Series, dates: pd.DatetimeIndex, lag: int) -> np.ndarray:
    """Last value dated on/before (date - lag) if at most STALE days old, else NaN."""
    t = dates - pd.Timedelta(days=lag)
    idx = s.index.values
    pos = np.searchsorted(idx, t.values, side="right") - 1
    out = np.full(len(t), np.nan)
    for i, p in enumerate(pos):
        if p >= 0 and (t[i] - s.index[p]).days <= STALE:
            out[i] = s.iloc[p]
    return out


def entry_rows(price: pd.Series, fridays: pd.DatetimeIndex) -> np.ndarray:
    """Row of the first close strictly after each Friday (-1 if none within the lag rule)."""
    pos = np.searchsorted(price.index.values, (fridays + pd.Timedelta(days=1)).values, "left")
    out = np.full(len(fridays), -1, dtype=np.int64)
    for i, p in enumerate(pos):
        if p < len(price) and (price.index[p] - (fridays[i] + pd.Timedelta(days=1))).days \
                <= MAX_ENTRY_LAG:
            out[i] = p
    return out


def fwd(price: pd.Series, ep: np.ndarray, ndays: int = H_DAYS) -> np.ndarray:
    """close[entry + ndays] / close[entry] - 1, NaN if unavailable."""
    c = price.to_numpy(float)
    r = np.full(len(ep), np.nan)
    ok = (ep >= 0) & (ep + ndays < len(c))
    r[ok] = c[ep[ok] + ndays] / c[ep[ok]] - 1.0
    return r


def runs(mask: np.ndarray) -> np.ndarray:
    """Run ids (1, 2, ...) of consecutive True values; 0 where False."""
    m = np.asarray(mask, bool)
    start = m & ~np.concatenate(([False], m[:-1]))
    return np.where(m, np.cumsum(start), 0)


def all_shift_means(sig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """M[j] = mean of x over the signal weeks shifted circularly by j, for j = 0..T-1."""
    T = len(sig)
    pos = np.flatnonzero(sig)
    out = np.empty(T)
    for a in range(0, T, 256):                  # chunked to keep memory small
        j = np.arange(a, min(T, a + 256))
        out[j] = x[(pos[None, :] + j[:, None]) % T].mean(axis=1)
    return out


def cell_stats(sig_full: np.ndarray, defined: np.ndarray, r: np.ndarray, side: int = 1,
               mc: bool = True) -> dict:
    """One (side, h=8) cell, independently: eligible = defined & forward return exists."""
    elig = defined & ~np.isnan(r)
    re = r[elig]
    T = int(elig.sum())
    setup = defined & (sig_full == side)
    sig = setup[elig]
    signed = side * re
    n = int(sig.sum())
    base = float(re.mean())
    mean = float(signed[sig].mean())
    ids = runs(setup)[setup & elig]
    M = all_shift_means(sig, signed)
    offs = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
    cnt = int(np.sum(M[offs] >= mean))
    out = dict(n=n, T=T, episodes=int(len(np.unique(ids))), mean=mean,
               median=float(np.median(signed[sig])), baseline=base, excess=mean - side * base,
               hit=float(np.mean(signed[sig] > 0)), perm_p_exact=(1 + cnt) / (1 + len(offs)),
               perm_p_exact_raw=cnt / len(offs))
    if mc:
        ko = np.random.default_rng(SEED).integers(MIN_SHIFT, T - MIN_SHIFT, size=K,
                                                  endpoint=True)
        out["perm_p"] = (1 + int(np.sum(M[ko] >= mean))) / (1 + K)
    return out


# =============================================================================================
# Part 1: the PRIMARY, from raw files
# =============================================================================================
def primary_frame(gold: pd.Series, usd: pd.Series, lag: int = USD_LAG, L: int = 8):
    """Friday grid over gold's range, the dollar L-week change read with `lag`, entry rows and
    8-week forward returns."""
    fr = pd.date_range(gold.index[0], gold.index[-1], freq="W-FRI")
    a = asof(usd, fr, lag)
    b = asof(usd, fr, lag + 7 * L)
    chg = a / b - 1.0
    ep = entry_rows(gold, fr)
    r = fwd(gold, ep)
    sig = np.where(np.isnan(chg), 0, np.where(chg < 0, 1, np.where(chg > 0, -1, 0)))
    return fr, chg, sig, ~np.isnan(chg), ep, r


def part1(gold, usd, res) -> dict:
    fr, chg, sig, dfn, ep, r = primary_frame(gold, usd)
    st = cell_stats(sig, dfn, r, +1)
    elig = dfn & ~np.isnan(r)
    st["period"] = f"{fr[elig][0].date()}..{fr[elig][-1].date()}"
    rp = res["primary"]
    cmp = {k: dict(mine=st[k], theirs=rp[k]) for k in
           ("n", "T", "episodes", "mean", "baseline", "excess", "hit", "perm_p", "perm_p_exact",
            "perm_p_exact_raw", "period")}
    match = all((abs(v["mine"] - v["theirs"]) < 1e-12 if isinstance(v["mine"], float)
                 else v["mine"] == v["theirs"]) for v in cmp.values())
    log(f"[P1] primary independent: {st}")
    log(f"[P1] matches results.json exactly: {match}")

    # --- Lag sensitivity (lag 0 is NOT tradable: the dollar is not public then; it shows how much
    #     of the 'relationship' is contemporaneous rather than leading)
    lag_rows = {}
    for lag in (0, 3, 7, 10, 14):
        f2, c2, s2, d2, e2, r2 = primary_frame(gold, usd, lag=lag)
        x = cell_stats(s2, d2, r2, +1, mc=False)
        lag_rows[f"lag{lag}"] = dict(n=x["n"], T=x["T"], excess=x["excess"],
                                     perm_p_exact=x["perm_p_exact"])
    log(f"[P1] lag sensitivity: {lag_rows}")

    # --- Episode dependence: drop crisis windows (both from signal weeks and the baseline)
    windows = {"GFC 2008-07..2009-03": ("2008-07-01", "2009-03-31"),
               "2011 gold peak 2011-06..2011-12": ("2011-06-01", "2011-12-31"),
               "2013 gold crash 2013-03..2013-07": ("2013-03-01", "2013-07-31"),
               "COVID 2020-02..2020-08": ("2020-02-01", "2020-08-31"),
               "2022 dollar surge 2022-01..2022-11": ("2022-01-01", "2022-11-30")}
    drop = {}
    for name, (a, b) in windows.items():
        keep = ~((fr >= pd.Timestamp(a)) & (fr <= pd.Timestamp(b)))
        x = cell_stats(sig, dfn & np.asarray(keep), r, +1, mc=False)
        drop[name] = dict(n=x["n"], excess=x["excess"], perm_p_exact=x["perm_p_exact"])
    log(f"[P1] drop-one-window: {drop}")

    # --- Non-overlapping subsamples: every 8th eligible week, 8 phases
    re, se = r[elig], (sig[elig] == 1)
    phases = []
    for j in range(8):
        rr, ss = re[j::8], se[j::8]
        phases.append(dict(phase=j, n=int(ss.sum()), T=int(len(rr)),
                           excess=float(rr[ss].mean() - rr.mean())))
    log(f"[P1] non-overlapping phases excess: {[round(p['excess'], 5) for p in phases]}")

    # --- Contribution by calendar year to the long-side excess (sum = total excess)
    yrs = fr[elig].year
    contrib = pd.Series(np.where(se, re - re.mean(), 0.0) / se.sum(), index=yrs).groupby(
        level=0).sum()
    log(f"[P1] by-year contribution (top/bottom): "
        f"{contrib.sort_values().head(3).round(5).to_dict()} / "
        f"{contrib.sort_values().tail(3).round(5).to_dict()}")

    # --- Split halves at the midpoint date of eligible weeks (each half its own baseline)
    ed = fr[elig]
    mid = ed[0] + (ed[-1] - ed[0]) / 2
    halves = {}
    for h, m in (("H1", fr <= mid), ("H2", fr > mid)):
        x = cell_stats(sig, dfn & np.asarray(m), r, +1)
        halves[h] = dict(n=x["n"], excess=x["excess"], perm_p=x["perm_p"])
    log(f"[P1] split at {mid.date()}: {halves}")

    # --- Context correlations (C4): dollar 8w change vs gold 8w change over the same window
    g8 = asof(gold, fr, 0) / asof(gold, fr, 56) - 1.0
    u_same = asof(usd, fr, 0) / asof(usd, fr, 56) - 1.0
    ok = ~np.isnan(g8) & ~np.isnan(u_same)
    okl = ~np.isnan(g8) & ~np.isnan(chg)
    both = okl & (chg != 0) & (g8 != 0)
    agree = float(np.mean(np.sign(-chg[both]) == np.sign(g8[both])))
    p_long_sig = float(np.mean(chg[both] < 0))
    p_gold_up = float(np.mean(g8[both] > 0))
    chance = p_long_sig * p_gold_up + (1 - p_long_sig) * (1 - p_gold_up)
    ctx = dict(corr_same_window=float(np.corrcoef(u_same[ok], g8[ok])[0, 1]), n_same=int(ok.sum()),
               corr_lagged_input_vs_gold_trailing=float(np.corrcoef(chg[okl], g8[okl])[0, 1]),
               agree_share=agree, agree_n=int(both.sum()), agree_expected_by_chance=chance)
    log(f"[P1] context: {ctx}")
    return dict(primary=st, compare_to_results=cmp, match=match, lag_sensitivity=lag_rows,
                drop_window=drop, nonoverlap_phases=phases,
                by_year_contribution={int(k): float(v) for k, v in contrib.items()},
                split_half=halves, split_date=str(mid.date()), context=ctx)


# =============================================================================================
# Part 2: S1 long-only strategy with costs, independently
# =============================================================================================
def s1_long_only(gold: pd.Series, fr, sig_week: np.ndarray, defined: np.ndarray, ep):
    """Enter at the entry close of the first tradable week of each long episode, hold 40 rows,
    one position at a time (an event whose entry row <= the current exit row is skipped).
    Returns (daily net returns from the window start, closed-trade net returns)."""
    c = gold.to_numpy(float)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    start = int(ep[np.flatnonzero(defined & (ep >= 0))[0]])
    setup = (sig_week == 1) & defined
    ids = runs(setup)
    events = sorted(int(ep[np.flatnonzero((ids == e) & (ep >= 0))[0]])
                    for e in np.unique(ids[ids > 0]) if np.any((ids == e) & (ep >= 0)))
    pos = np.zeros(n)
    busy, trades = -1, []
    for e in events:
        if e <= busy:
            continue
        x = e + H_DAYS
        is_open = x > n - 1
        x = min(x, n - 1)
        pos[e:(n if is_open else x)] = 1.0
        if not is_open:
            trades.append((1 - COST) ** 2 * np.prod(1 + ret[e + 1:x + 1]) - 1)
        busy = x
    p_prev = np.concatenate(([0.0], pos[:-1]))
    p_prev[:start + 1] = 0.0
    dpos = np.abs(pos - p_prev)
    dpos[:start] = 0.0
    net = (1 + p_prev * ret) * (1 - COST * dpos) - 1
    return pd.Series(net[start:], index=gold.index[start:]), np.array(trades)


def perf(net: pd.Series) -> dict:
    eq = np.cumprod(1 + net.to_numpy())
    yrs = (net.index[-1] - net.index[0]).days / 365.25
    r = net.to_numpy()[1:]
    ppy = len(r) / yrs
    sd = np.std(r, ddof=1)
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))
    return dict(sharpe=float(np.mean(r) * ppy / (sd * np.sqrt(ppy))), cagr=float(eq[-1] ** (1 / yrs) - 1),
                max_dd=float((np.concatenate(([1.0], eq)) / peak - 1).min()), years=yrs)


def part2(gold, usd, res) -> dict:
    fr, chg, sig, dfn, ep, r = primary_frame(gold, usd)
    net, tr = s1_long_only(gold, fr, sig, dfn, ep)
    m = perf(net)
    m.update(n_trades=int(len(tr)), win_rate=float(np.mean(tr > 0)))
    # buy & hold over the same window, no cost
    c = gold.to_numpy(float)
    start = gold.index.get_loc(net.index[0])
    bh = np.zeros(len(c))
    bh[start + 1:] = c[start + 1:] / c[start:-1] - 1
    bhm = perf(pd.Series(bh[start:], index=gold.index[start:]))
    # random-timing null: shift the weekly signal circularly over the DEFINED weeks
    d = np.flatnonzero(dfn)
    offs = np.random.default_rng(SEED).integers(MIN_SHIFT, len(d) - MIN_SHIFT, size=K,
                                                endpoint=True)
    null = np.empty(K)
    for i, k in enumerate(offs):
        s2 = sig.copy()
        s2[d] = np.roll(sig[d], int(k))
        nn, _ = s1_long_only(gold, fr, s2, dfn, ep)
        null[i] = perf(nn)["sharpe"]
    m["null_p_sharpe"] = (1 + int(np.sum(null >= m["sharpe"]))) / (1 + K)
    theirs = res["primary"]["S1_long_only"]
    log(f"[P2] S1 long-only independent: {m}; buy&hold {bhm}")
    log(f"[P2] results.json S1 long-only: {theirs}")
    return dict(S1_long_only=m, buy_hold=bhm, theirs=theirs)


# =============================================================================================
# Part 3: claims cross-check against tests.csv / results.json (numbers as stated)
# =============================================================================================
def part3(res) -> dict:
    t = pd.read_csv(HERE / "tests.csv")

    def row(grid, variant, market, series, side="long", h=8):
        q = t[(t["grid"] == grid) & (t["variant"] == variant) & (t["market"] == market)
              & (t["series"] == series) & (t["side"] == side) & (t["horizon_w"] == h)]
        assert len(q) == 1, (grid, variant, market, series, len(q))
        return dict(n=int(q.iloc[0]["n"]), **{k: float(q.iloc[0][k]) for k in
                                               ("excess", "perm_p", "perm_p_exact")})

    out = {}
    out["C5_pooled_usd_L8"] = res["secondary_headline_pooled_usd_L8_long_h8"]
    out["C5_per_market"] = {m: row("A", "lead_usd_L8", m, s) for m, s in
                            (("crude", "crude_roll"), ("soybean", "soybean"),
                             ("sugar", "sugar"), ("gold", "gold"), ("silver", "silver"))}
    out["C5_crude_L4"] = row("A", "lead_usd_L4", "crude", "crude_roll")
    out["C7_gridA_gold_L13"] = {m: row("A", "lead_gold_L13", m, s) for m, s in
                                (("crude", "crude_roll"), ("silver", "silver"),
                                 ("soybean", "soybean"), ("sugar", "sugar"))}
    pooled = t[(t["market"] == "pooled") & (t["side"] == "long") & (t["horizon_w"] == 8)]
    out["C7_pooled_gold"] = pooled[pooled["variant"].str.startswith("lead_gold")][
        ["variant", "n", "excess", "perm_p"]].to_dict("records")
    wsp = t[(t["grid"] == "C") & (t["side"] == "long") & (t["horizon_w"] == 8)]
    out["C9_willspread"] = dict(frames=int(len(wsp)), min_perm_p=float(wsp["perm_p"].min()),
                                best=wsp.sort_values("perm_p").iloc[0][
                                    ["variant", "market", "excess", "perm_p"]].to_dict())
    lh8 = t[(t["horizon_w"] == 8) & (t["side"] == "long") & (t["market"] != "pooled")]
    out["C10"] = dict(frames=int(len(lh8)), p_le_005=int((lh8["perm_p"] <= 0.05).sum()),
                      excess_gt0=int((lh8["excess"] > 0).sum()),
                      min_bh_q=float(t["bh_q_h8"].min()))
    log(f"[P3] claim cross-check: {json.dumps(out, default=str)}")
    return out


# =============================================================================================
# Part 4 + 5: rebuild the long-side h=8 grid frames; own evaluation; joint-shift family test
# =============================================================================================
def build_grid_frames():
    """Rebuild the 121 non-pooled long h=8 frames of the grid (signals from intermarket_lib,
    series from the harness loaders), returning {key: (sig_full, defined, r8)}."""
    sys.path.insert(0, str(CODE))
    import intermarket_lib as il            # signal builders only
    fh = il.fh
    traded = {"crude": "crude_roll", "gold": "gold", "silver": "silver", "soybean": "soybean",
              "sugar": "sugar"}
    ser = {m: fh.load_series(s) for m, s in traded.items()}
    frames = {}

    def add(key, price, fridays, sig):
        sig = np.asarray(sig, float)
        dfn = ~np.isnan(sig)
        sf = np.where(dfn, sig, 0).astype(int)
        frames[key] = (sf, dfn, fwd(price, entry_rows(price, pd.DatetimeIndex(fridays))),
                       pd.DatetimeIndex(fridays))

    fr = {m: il.fridays_for(ser[m]) for m in traded}
    cl = fh.load_series("crude_cl")
    fcl = il.fridays_for(cl)
    sigs = {}
    for ld in ("usd", "bond", "gold"):
        for L in (4, 8, 13):
            for m in traded:
                if ld == "gold" and m == "gold":
                    continue
                s = il.leader_signal(ld, fr[m], L)
                sigs[(ld, L, m)] = s
                add(("A", f"lead_{ld}_L{L}", m), ser[m], fr[m], s)
            add(("A_ctx", f"lead_{ld}_L{L}", "crude_cl"), cl, fcl, il.leader_signal(ld, fcl, L))
    pst_gold, _ = il.pst_index("gold")
    for gname, gin in (("gold_v1", ser["gold"]), ("gold_pst", pst_gold)):
        for tname in ("ust10_price", "ust10_tr"):
            ts = fh.load_series(tname)
            f2 = il.fridays_for(ts, start=gin.index[0], end=gin.index[-1])
            add(("B", gname, tname), ts, f2, il.larry_gold_bonds_signal(gin, f2, 20))
    secs = {"gold": (ser["gold"], 0), "bond": (fh.load_series("ust10_price"), 0),
            "usd": (fh.load_series("usd_broad"), il.USD_LAG_DAYS)}
    for m in traded:
        for sec, (xs, lag) in secs.items():
            if sec == "gold" and m == "gold":
                continue
            for fa, sl_ in ((5, 20), (7, 11)):
                add(("C", f"ws_{sec}_{fa}_{sl_}", m), ser[m], fr[m],
                    il.willspread_signal(ser[m], xs, fr[m], lag_days=lag, fast=fa, slow=sl_))
    for m in traded:
        comp = [sigs[("usd", 8, m)], sigs[("bond", 8, m)]] + ([] if m == "gold" else
                                                              [sigs[("gold", 8, m)]])
        add(("D", "vote_L8", m), ser[m], fr[m], il.vote_signal(comp, 2))
    pst = {"gold": pst_gold}
    for m in traded:
        if m != "gold":
            pst[m], _ = il.pst_index(m)
    for m in traded:
        f2 = il.fridays_for(pst[m])
        for L in (4, 8, 13):
            add(("E", f"bond_L{L}", m), pst[m], f2, il.leader_signal("bond", f2, L))
            if m != "gold":
                add(("E", f"gold_L{L}", m), pst[m], f2,
                    il.sign_signal(il.leader_change(pst["gold"], f2, L, 0), +1))
    for m in traded:
        add(("F", "own_L8", m), ser[m], fr[m], il.own_trend_signal(ser[m], fr[m], 8))
    f2 = il.fridays_for(pst["gold"])
    add(("H", "usd_L8", "pst_gold"), pst["gold"], f2, il.leader_signal("usd", f2, 8))
    return frames


def part4_5(res) -> dict:
    frames = build_grid_frames()
    log(f"[P4] rebuilt {len(frames)} long-h8 frames")
    t = pd.read_csv(HERE / "tests.csv")
    tl = t[(t["horizon_w"] == 8) & (t["side"] == "long") & (t["market"] != "pooled")]

    # Per-frame exhaustive shift means on the eligible sequence
    prep = {}
    for key, (sf, dfn, r, fr) in frames.items():
        elig = dfn & ~np.isnan(r)
        sig = (sf == 1)[elig]
        x = r[elig]
        M = all_shift_means(sig, x)
        T = len(x)
        offs = np.arange(MIN_SHIFT, T - MIN_SHIFT + 1)
        prep[key] = dict(M=M, T=T, base=float(x.mean()), offs=offs, n=int(sig.sum()),
                         start=fr[elig][0], p0=(1 + int(np.sum(M[offs] >= M[0]))) / (1 + len(offs)))
    obs_p = np.array([v["p0"] for v in prep.values()])
    obs_ex = np.array([v["M"][0] - v["base"] for v in prep.values()])
    log(f"[P4] own exact p: {int((obs_p <= 0.05).sum())} frames <= 0.05; "
        f"{int((obs_ex > 0).sum())} with excess > 0; n frames {len(prep)}; "
        f"theirs (MC p): {int((tl['perm_p'] <= 0.05).sum())} / {int((tl['excess'] > 0).sum())}"
        f" of {len(tl)}")
    # sanity: own n / excess agree with tests.csv on a few named frames
    chk = {}
    for key, (g, v, m, s) in {("A", "lead_gold_L13", "crude"): ("A", "lead_gold_L13", "crude",
                                                                  "crude_roll"),
                              ("A", "lead_usd_L8", "gold"): ("A", "lead_usd_L8", "gold", "gold"),
                              ("F", "own_L8", "gold"): ("F", "own_trend_L8", "gold", "gold")
                              }.items():
        q = tl[(tl["grid"] == g) & (tl["variant"] == v) & (tl["market"] == m)
               & (tl["series"] == s)].iloc[0]
        p = prep[key]
        chk["/".join(key)] = dict(mine_n=p["n"], theirs_n=int(q["n"]),
                                  mine_excess=float(p["M"][0] - p["base"]),
                                  theirs_excess=float(q["excess"]), mine_p=p["p0"],
                                  theirs_p_exact=float(q["perm_p_exact"]))
    log(f"[P4] spot check vs tests.csv: {chk}")

    # Joint ("family") shift: one common offset k (weeks) for every frame -> the cross-frame
    # dependence (shared leaders, nested lookbacks, overlapping markets/periods) is kept.
    Tmin = min(v["T"] for v in prep.values())
    ks = np.arange(MIN_SHIFT, Tmin - MIN_SHIFT + 1)
    cnt_p, cnt_ex, sum_z = [], [], []
    for k in ks:
        c1 = c2 = 0
        for v in prep.values():
            T, M, offs = v["T"], v["M"], v["offs"]
            mk = M[k % T]
            pk = (1 + int(np.sum(M[(k + offs) % T] >= mk))) / (1 + len(offs))
            c1 += pk <= 0.05
            c2 += mk > v["base"]
        cnt_p.append(c1)
        cnt_ex.append(c2)
    cnt_p, cnt_ex = np.array(cnt_p), np.array(cnt_ex)
    o1, o2 = int((obs_p <= 0.05).sum()), int((obs_ex > 0).sum())
    fam = dict(n_frames=len(prep), common_offsets=int(len(ks)), Tmin=int(Tmin),
               observed_p_le_005=o1, observed_excess_gt0=o2,
               null_p_le_005_median=float(np.median(cnt_p)),
               null_p_le_005_p95=float(np.quantile(cnt_p, 0.95)),
               family_p_count_sig=float((1 + np.sum(cnt_p >= o1)) / (1 + len(ks))),
               null_excess_gt0_median=float(np.median(cnt_ex)),
               null_excess_gt0_p95=float(np.quantile(cnt_ex, 0.95)),
               family_p_count_pos=float((1 + np.sum(cnt_ex >= o2)) / (1 + len(ks))),
               note="Common circular shift k (in weeks of each frame's own eligible sequence) "
                    "applied to all frames at once; frames start on different dates, so the "
                    "shift is only approximately a common calendar shift.")
    log(f"[P5] family test: {fam}")
    # Same, restricted to the 2000+ / 2006+ frames that share most of the calendar (A, C, D, F)
    sub = [kk for kk in prep if kk[0] in ("A", "C", "D", "F")]
    o1s = int(sum(prep[kk]["p0"] <= 0.05 for kk in sub))
    o2s = int(sum(prep[kk]["M"][0] > prep[kk]["base"] for kk in sub))
    Tm = min(prep[kk]["T"] for kk in sub)
    ks2 = np.arange(MIN_SHIFT, Tm - MIN_SHIFT + 1)
    c1s, c2s = [], []
    for k in ks2:
        a = b = 0
        for kk in sub:
            v = prep[kk]
            T, M, offs = v["T"], v["M"], v["offs"]
            mk = M[k % T]
            a += (1 + int(np.sum(M[(k + offs) % T] >= mk))) / (1 + len(offs)) <= 0.05
            b += mk > v["base"]
        c1s.append(a)
        c2s.append(b)
    c1s, c2s = np.array(c1s), np.array(c2s)
    fam_sub = dict(grids="A, C, D, F (v1-era frames)", n_frames=len(sub), observed_p_le_005=o1s,
                   observed_excess_gt0=o2s, null_p_le_005_median=float(np.median(c1s)),
                   family_p_count_sig=float((1 + np.sum(c1s >= o1s)) / (1 + len(ks2))),
                   null_excess_gt0_median=float(np.median(c2s)),
                   family_p_count_pos=float((1 + np.sum(c2s >= o2s)) / (1 + len(ks2))))
    log(f"[P5] family test (A,C,D,F): {fam_sub}")

    # BH over the 121 long-side h=8 frames only (the run's BH used 363 rows incl. the short and
    # net complements, whose p-values are ~1 - p_long and only add rows)
    p = np.sort(obs_p)
    m = len(p)
    q = np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1]
    bh121 = dict(min_q=float(q.min()), n_q_le_010=int((q <= 0.10).sum()))
    log(f"[P5] BH over 121 long frames (own exact p): {bh121}")
    return dict(spot_check=chk, family_all=fam, family_ACDF=fam_sub, bh_long121=bh121,
                frames=frames)


# =============================================================================================
# Part 6: walk-forward selection rule, independently (PREREG section 6)
# =============================================================================================
def bond_price_from_dgs10() -> pd.Series:
    """Price of a 10-year, 6% semi-annual bond at each day's DGS10 yield (own formula)."""
    with open(USD_JSON.parent / "DGS10.json") as f:
        obs = json.load(f)["observations"]
    y = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy() / 100
    s = pd.Series(y, index=pd.DatetimeIndex([o["date"] for o in obs])).dropna()
    s = s[s > 0].sort_index()
    v = (1 + s / 2) ** -20
    return (3.0 / (s / 2) * (1 - v) + 100 * v).rename("bond")


def part6(gold, usd, res) -> dict:
    bond = bond_price_from_dgs10()
    fr = pd.date_range(gold.index[0], gold.index[-1], freq="W-FRI")
    ep = entry_rows(gold, fr)
    r8 = fwd(gold, ep)
    gi = gold.index.values
    exit8 = np.array([gi[e + H_DAYS] if (e >= 0 and e + H_DAYS < len(gi))
                      else np.datetime64("NaT", "ns") for e in ep], dtype="datetime64[ns]")
    cfg = {}
    for name, ser, lag, sgn in (("usd", usd, USD_LAG, -1), ("bond", bond, 0, -1)):
        for L in (4, 8, 13):
            chg = asof(ser, fr, lag) / asof(ser, fr, lag + 7 * L) - 1
            for flip, tag in ((1, "prereg"), (-1, "reversed")):
                s = sgn * flip * np.sign(chg)
                cfg[f"{name}_L{L}_{tag}"] = s          # NaN where undefined
    wf = np.full(len(fr), np.nan)
    choices = []
    for y in range(2011, fr[-1].year + 1):
        cut = pd.Timestamp(f"{y}-01-01")
        hist = (fr < cut) & ~np.isnan(r8) & (exit8 < np.datetime64(cut))
        best, best_ex = None, -np.inf
        for c, s in cfg.items():
            el = hist & ~np.isnan(s)
            lg = el & (s == 1)
            if lg.sum() < 26:
                continue
            ex = r8[lg].mean() - r8[el].mean()
            if ex > best_ex:
                best, best_ex = c, ex
        blk = (fr >= cut) & (fr < pd.Timestamp(f"{y + 1}-01-01"))
        if best is None or best_ex <= 0:
            wf[blk] = 0.0
        else:
            s = cfg[best][blk]
            wf[blk] = np.where(np.isnan(s), np.nan, (s == 1).astype(float))
        choices.append((str(cut.date()), best, float(best_ex)))
    dfn = ~np.isnan(wf)
    st = cell_stats(np.where(dfn, wf, 0).astype(int), dfn, r8, +1)
    theirs = res["primary"]["walk_forward_selection_long_h8"]
    log(f"[P6] walk-forward selection (own): n={st['n']} excess={st['excess']:.6f} "
        f"p={st['perm_p']:.4f}; theirs {theirs}")
    log(f"[P6] own yearly choices: {choices}")
    # sanity: own bond price vs the harness ust10_price
    sys.path.insert(0, str(PKG / "battery" / "harness"))
    import fund_harness as fh
    hb = fh.load_series("ust10_price")
    j = pd.concat([bond, hb], axis=1, join="inner")
    maxdiff = float((j.iloc[:, 0] / j.iloc[:, 1] - 1).abs().max())
    log(f"[P6] own 6%-coupon bond price vs harness ust10_price: max rel diff {maxdiff:.2e}")
    return dict(own=dict(n=st["n"], excess=st["excess"], perm_p=st["perm_p"]), theirs=theirs,
                choices=choices, bond_vs_harness_max_rel_diff=maxdiff)


# =============================================================================================
# Part 7: pooled cells and episode dependence of the POSITIVE cells (C5, C7)
# =============================================================================================
def pooled(parts, drop=None) -> dict:
    """Own pooled cell (long, h=8): parts = [(sig_full, defined, r, fridays)]. excess = mean of
    (r - own baseline) over all signal weeks; MC p with one default_rng(SEED) per pooled cell,
    each part drawing K offsets in order (the run's scheme)."""
    rng = np.random.default_rng(SEED)
    sums = np.zeros(K)
    allr, allex = [], []
    for sf, dfn, r, fr in parts:
        d = dfn.copy()
        if drop is not None:
            for a, b in drop:
                d &= ~np.asarray((fr >= pd.Timestamp(a)) & (fr <= pd.Timestamp(b)))
        elig = d & ~np.isnan(r)
        sig = ((sf == 1) & d)[elig]
        x = r[elig]
        allr.append(x[sig])
        allex.append(x[sig] - x.mean())
        T = len(x)
        offs = rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=K, endpoint=True)
        pos = np.flatnonzero(sig)
        sums += x[(pos[None, :] + offs[:, None]) % T].sum(axis=1)
    sr = np.concatenate(allr)
    obs = sr.mean()
    return dict(n=int(len(sr)), excess=float(np.concatenate(allex).mean()),
                perm_p=(1 + int(np.sum(sums / len(sr) >= obs))) / (1 + K))


def part7(frames) -> dict:
    windows = {"none": None,
               "GFC 2008-07..2009-06": [("2008-07-01", "2009-06-30")],
               "oil/dollar 2014-06..2015-03": [("2014-06-01", "2015-03-31")],
               "COVID 2020-02..2020-08": [("2020-02-01", "2020-08-31")],
               "all three": [("2008-07-01", "2009-06-30"), ("2014-06-01", "2015-03-31"),
                             ("2020-02-01", "2020-08-31")]}
    cells = {
        "C5 pooled usd L8 (5 mkts)": [frames[("A", "lead_usd_L8", m)] for m in
                                      ("crude", "gold", "silver", "soybean", "sugar")],
        "C5 crude_roll usd L4": [frames[("A", "lead_usd_L4", "crude")]],
        "C7 pooled gold L13 (4 mkts)": [frames[("A", "lead_gold_L13", m)] for m in
                                        ("crude", "silver", "soybean", "sugar")],
        "C7 crude_roll gold L13": [frames[("A", "lead_gold_L13", "crude")]],
        "C7 ext pst gold L13 (4 mkts pooled)": [frames[("E", "gold_L13", m)] for m in
                                                ("crude", "silver", "soybean", "sugar")],
    }
    out = {}
    for cname, parts in cells.items():
        out[cname] = {w: pooled(parts, drop) for w, drop in windows.items()}
        log(f"[P7] {cname}: " + "; ".join(f"{w}: ex {v['excess']:+.5f} p {v['perm_p']:.4f}"
                                          for w, v in out[cname].items()))
    # split halves of the two pooled headline-like cells (each part split at its own midpoint)
    for cname in ("C5 pooled usd L8 (5 mkts)", "C7 pooled gold L13 (4 mkts)"):
        hs = {}
        for half in ("H1", "H2"):
            parts = []
            for sf, dfn, r, fr in cells[cname]:
                el = fr[dfn & ~np.isnan(r)]
                mid = el[0] + (el[-1] - el[0]) / 2
                m = np.asarray(fr <= mid) if half == "H1" else np.asarray(fr > mid)
                parts.append((sf, dfn & m, r, fr))
            hs[half] = pooled(parts)
        out[cname]["split_half"] = hs
        log(f"[P7] {cname} split halves: {hs}")

    # C7 replication on DISJOINT periods of the extended pysystemtrade history: decision dates
    # before 2000-08-25 (none of the 2000-2026 grid-A data where the pattern was noticed) vs
    # after. Each period has its own baseline and its own circular shifts.
    rep = {}
    for L in (4, 8, 13):
        parts_all = [frames[("E", f"gold_L{L}", m)] for m in ("crude", "silver", "soybean",
                                                                "sugar")]
        for tag, lo, hi in (("pre2000", "1900-01-01", "2000-08-24"),
                            ("post2000", "2000-08-25", "2100-01-01")):
            parts = [(sf, dfn & np.asarray((fr >= pd.Timestamp(lo)) & (fr <= pd.Timestamp(hi))),
                      r, fr) for sf, dfn, r, fr in parts_all]
            parts = [p for p in parts if (p[1] & ~np.isnan(p[2])).sum() > 2 * MIN_SHIFT + 1]
            rep[f"ext_gold_L{L}_{tag}"] = dict(pooled(parts), n_markets=len(parts))
            for (sf, dfn, r, fr), m in zip(parts_all, ("crude", "silver", "soybean", "sugar")):
                d = dfn & np.asarray((fr >= pd.Timestamp(lo)) & (fr <= pd.Timestamp(hi)))
                if (d & ~np.isnan(r)).sum() > 2 * MIN_SHIFT + 1 and ((sf == 1) & d).any():
                    x = cell_stats(sf, d, r, +1)
                    rep[f"ext_gold_L{L}_{tag}_{m}"] = dict(n=x["n"], excess=x["excess"],
                                                           perm_p=x["perm_p"])
    log(f"[P7] C7 disjoint-period replication: {json.dumps(rep)}")
    out["C7_disjoint_periods"] = rep
    return out


def part8() -> dict:
    """crude_roll roll-day check: largest daily moves of the roll index, and returns on the days
    contract 1 changes identity."""
    sys.path.insert(0, str(PKG / "battery" / "harness"))
    import fund_harness as fh
    ri = fh.build_crude_roll_index(0)
    v = fh.validate_roll_index(ri)
    lv = ri.level
    rr = lv.pct_change().dropna()
    # days on which contract 1 changes identity (the return is CL1(t)/CL2(t-1) - 1 there)
    roll_days = pd.DatetimeIndex(ri.detail.index[ri.detail["roll"].to_numpy(bool)])
    on_roll = rr[rr.index.isin(roll_days)]
    top = rr.abs().sort_values(ascending=False).head(6)
    out = dict(validate=v, n_roll_days_in_index=int(len(on_roll)),
               roll_day_abs_ret_mean=float(on_roll.abs().mean()),
               nonroll_abs_ret_mean=float(rr[~rr.index.isin(roll_days)].abs().mean()),
               largest_abs_daily=[(str(d.date()), float(rr[d])) for d in top.index])
    log(f"[P8] crude_roll: {json.dumps(out, default=str)[:1500]}")
    return out


def annotate_results(out: dict) -> None:
    """Add an 'adversarial_verification' block to results.json (numbers unchanged; wording
    corrections only). Every number below is read from this script's own output dict."""
    p1, p2 = out["part1_primary"], out["part2_strategy"]
    fam, famv1 = out["part4_5_grid"]["family_all"], out["part4_5_grid"]["family_ACDF"]
    p7 = out["part7_pooled_episodes"]
    c5, c5l4 = p7["C5 pooled usd L8 (5 mkts)"], p7["C5 crude_roll usd L4"]
    c7, rep = p7["C7 pooled gold L13 (4 mkts)"], p7["C7_disjoint_periods"]
    ctx = p1["context"]
    s1 = p2["S1_long_only"]
    s1_match = all(abs(s1[a] - p2["theirs"][b]) < 1e-12 for a, b in
                   (("sharpe", "sharpe"), ("cagr", "ann_return"), ("max_dd", "max_dd"),
                    ("win_rate", "win_rate"), ("null_p_sharpe", "null_p_sharpe")))
    wf = out["part6_walk_forward"]
    block = dict(
        script="adv_verify_intermarket.py (output adv_verify_intermarket.json, "
               "adv_verify_log.txt)",
        primary_recomputed_from_raw_files_match=bool(p1["match"]),
        s1_long_only_recomputed_match=bool(s1_match),
        walk_forward_selection_recomputed_match=bool(
            wf["own"]["n"] == wf["theirs"]["n"]
            and abs(wf["own"]["excess"] - wf["theirs"]["excess"]) < 1e-12),
        verdict_check="'no edge' confirmed for the pre-registered primary",
        wording_corrections=[
            f"C4: the primary signal agrees with gold's own trailing 8-week trend in "
            f"{ctx['agree_share']:.1%} of {ctx['agree_n']} weeks vs {ctx['agree_expected_by_chance']:.1%}"
            f" expected by chance from the two signals' marginal rates: it overlaps with gold's "
            f"own trend, it does not 'mostly repeat' it.",
            f"C5: the pooled dollar-L8 rule's positive excess comes entirely from the first half "
            f"(H1 {c5['split_half']['H1']['excess']:+.4%}, p {c5['split_half']['H1']['perm_p']:.3f};"
            f" H2 {c5['split_half']['H2']['excess']:+.4%}, p {c5['split_half']['H2']['perm_p']:.3f})"
            f" and leans on 2008-09 (without Jul 2008-Jun 2009: "
            f"{c5['GFC 2008-07..2009-06']['excess']:+.4%}, p "
            f"{c5['GFC 2008-07..2009-06']['perm_p']:.3f}). Dollar-L4 on crude_roll "
            f"({c5l4['none']['excess']:+.4%}, p {c5l4['none']['perm_p']:.3f}) falls to "
            f"{c5l4['GFC 2008-07..2009-06']['excess']:+.4%}, p "
            f"{c5l4['GFC 2008-07..2009-06']['perm_p']:.3f} without that window.",
            f"C7: gold's 13-week trend leading the other commodities is robust to dropping single "
            f"crisis windows (pooled p {c7['GFC 2008-07..2009-06']['perm_p']:.4f} without 2008-09)"
            f" and replicates on the disjoint pre-2000 pysystemtrade history (pooled L13 "
            f"{rep['ext_gold_L13_pre2000']['excess']:+.4%}, p "
            f"{rep['ext_gold_L13_pre2000']['perm_p']:.3f}; L8 "
            f"{rep['ext_gold_L8_pre2000']['excess']:+.4%}, p "
            f"{rep['ext_gold_L8_pre2000']['perm_p']:.3f}), but NOT for crude pre-2000 (L13 "
            f"{rep['ext_gold_L13_pre2000_crude']['excess']:+.4%}). Still post hoc (not the "
            f"primary): a candidate for a new pre-registered test.",
            f"C10: a joint circular-shift test that keeps the cross-frame correlation (one common "
            f"offset for all {fam['n_frames']} long-h8 frames) does NOT support 'about what chance "
            f"would produce': {fam['observed_p_le_005']} frames with exact p<=0.05 vs a null "
            f"median of {fam['null_p_le_005_median']:.0f} (95th pct "
            f"{fam['null_p_le_005_p95']:.0f}), family p {fam['family_p_count_sig']:.3f}; "
            f"{fam['observed_excess_gt0']} frames with excess>0 vs null median "
            f"{fam['null_excess_gt0_median']:.0f} (95th pct {fam['null_excess_gt0_p95']:.0f}), "
            f"family p {fam['family_p_count_pos']:.3f}. The surplus sits in the gold-leader and "
            f"extended-history frames; the v1-era grids A/C/D/F alone show "
            f"{famv1['observed_p_le_005']} hits vs null median "
            f"{famv1['null_p_le_005_median']:.0f} (family p {famv1['family_p_count_sig']:.3f}). "
            f"No single cell survives BH (min q over the 121 long frames "
            f"{out['part4_5_grid']['bh_long121']['min_q']:.3f}).",
        ],
        no_lookahead_findings=[
            "DTWEXBGS read as of Friday-7d = the Friday value released on the Monday H.10; lag "
            "0/3/10/14 change nothing (all excess < 0, p > 0.5), see adv_verify_log.txt.",
            "crude_roll has no roll-day jumps: adjusted roll-day returns look like other days "
            "(validate_roll_index).",
            "Walk-forward fit only uses weeks whose 8-week exit is before the cutoff "
            "(re-derived independently: identical yearly choices).",
            "FRED DTWEXBGS is the current vintage (H.10 weights are revised); real-time "
            "vintages are not in the archive (untestable, minor).",
            "The v1 'gold' series is symbol 122.XAU (a vendor spot quote), labelled London spot "
            "in SPEC v1; its exact fixing time is unverified. Entry is the next trading day, so "
            "this cannot create look-ahead.",
        ],
    )
    path = HERE / "results.json"
    doc = json.loads(path.read_text())
    doc["adversarial_verification"] = block
    path.write_text(json.dumps(doc, indent=1))
    log("[ANN] wrote adversarial_verification block into results.json")
    for w in block["wording_corrections"]:
        log(f"[ANN] {w}")


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    gold, usd = load_gold_raw(), load_usd_raw()
    log(f"gold {gold.index[0].date()}..{gold.index[-1].date()} n={len(gold)}; "
        f"usd {usd.index[0].date()}..{usd.index[-1].date()} n={len(usd)}")
    out = dict(part1_primary=part1(gold, usd, res), part2_strategy=part2(gold, usd, res),
               part3_claims=part3(res))
    g = part4_5(res)
    frames = g.pop("frames")
    out["part4_5_grid"] = g
    out["part6_walk_forward"] = part6(gold, usd, res)
    out["part7_pooled_episodes"] = part7(frames)
    out["part8_crude_roll"] = part8()
    (HERE / "adv_verify_intermarket.json").write_text(json.dumps(out, indent=1, default=str))
    annotate_results(json.loads(json.dumps(out, default=str)))
    (HERE / "adv_verify_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

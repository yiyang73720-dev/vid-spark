#!/usr/bin/env python3
"""
rank_lib -- shared helpers for the cross-tool RANKING step of the Larry Williams fundamentals
battery (SPEC2). Nothing here invents a signal: every weekly signal is rebuilt with the tool's
OWN library (tool_*/..._lib.py, seasonal.py) or with the v1 COT engine, exactly as that tool's
runner built its pre-registered primary, and `verify_primaries()` checks the rebuilt primaries
against the published primary rows before anything else uses them.

Contents
  1. imports of the shared harness and of every tool library (read-only reuse)
  2. rebuilt PRIMARY signal frames of the six tools (COT v1 + the five SPEC2 tools)
  3. Friday-grid component signals per market and their mapping onto the COT report grid
     (inputs of the confluence test, see PREREG.md)
  4. statistics helpers: Benjamini-Hochberg, circular moving-block bootstrap of the excess
     return and of excess differences

Run every script that imports this module with `python3 -I` (third-party files are parsed
with pandas/json only, never executed). sys.dont_write_bytecode is set before any import so
no .pyc is written into the tool directories.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True     # never write __pycache__ into work2/ or work/

import json                       # noqa: E402
import os                         # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

PKG = Path(__file__).resolve().parents[2]          # package dir (battery/rank/../..)
BATTERY = PKG / "battery"                          # battery code (tool libraries, PREREG.md)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)   # data/ ext/ ext2/ work/ work2/
WORK2 = ROOT / "work2"                             # battery outputs
RANK = WORK2 / "rank"                              # outputs of this step
RANK_CODE = BATTERY / "rank"                       # PREREG.md, PREREG.sha256
TOOL_DIRS = {
    "premium": WORK2 / "tool_premium",
    "seasonality": WORK2 / "tool_seasonality",
    "valuation": WORK2 / "tool_valuation",
    "open_interest": WORK2 / "tool_open_interest",
    "intermarket": WORK2 / "tool_intermarket",
}
for _p in [BATTERY / "harness", *(BATTERY / _d.name for _d in TOOL_DIRS.values())]:   # code
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fund_harness as fh          # noqa: E402  shared evaluation harness
import premium_lib as pl           # noqa: E402  tool_premium
import seasonal as se              # noqa: E402  tool_seasonality
import valuation_lib as vl         # noqa: E402  tool_valuation
import oi_lib as ol                # noqa: E402  tool_open_interest
import intermarket_lib as il       # noqa: E402  tool_intermarket

lc = fh.lc                          # v1 COT engine (larry_cot.py)
MARKETS5 = ("crude", "gold", "silver", "soybean", "sugar")
TRADED5 = {"crude": "crude_roll", "gold": "gold", "silver": "silver", "soybean": "soybean",
           "sugar": "sugar"}
TOOLS = ("cot", "premium", "seasonality", "valuation", "open_interest", "intermarket")
COMPONENTS = ("cot", "premium", "seasonal", "valuation", "oi", "dollar")
BLOCK_WEEKS = 13          # bootstrap block length (weeks), fixed by the task
BOOT_B = 2000             # bootstrap resamples
BOOT_SEED = 11            # same seed convention as the battery


def load_results(tool: str) -> dict:
    """results.json of a SPEC2 tool (json.load only)."""
    return json.load(open(TOOL_DIRS[tool] / "results.json", encoding="utf-8"))


# =============================================================================================
# 2. Rebuilt PRIMARY signal frames
# =============================================================================================
_PANEL = None


def cot_panel() -> dict:
    """v1 per-report COT frames (idx26/52/156, shutdown flags) for the five markets."""
    global _PANEL
    if _PANEL is None:
        _PANEL = lc.build_panel(markets=list(MARKETS5))
    return _PANEL


def primary_cot() -> dict:
    """v1 COT primary: crude, Larry COT index N=26 >= 80 long (<= 20 short), timing cot,
    traded on the v1 CL continuous series (crude_cl), LONG h=8."""
    sf = fh.cot_signal_frame("crude", "idx26", 80, 20, panel=cot_panel())
    return dict(kind="single", parts=[(sf, "crude_cl", "cot")], side="long")


def premium_crude_weekly(fri: pd.DatetimeIndex) -> np.ndarray:
    """tool_premium primary signal: sign(RCLC1/RCLC2 - 1) as of each Friday close."""
    return pl.sig_sign(pl.weekly_asof(pl.crude_premium_daily(1, 2), fri))


def primary_premium() -> dict:
    """tool_premium primary: crude c1 > c2 at the Friday close, LONG h=8 on crude_roll."""
    fri = fh.weekly_decision_dates("1985-01-04", "2024-03-29")      # the tool's crude grid
    sf = fh.make_signal_frame(fri, premium_crude_weekly(fri))
    return dict(kind="single", parts=[(sf, "crude_roll", "price")], side="long")


SEAS_MARKETS = [  # tool_seasonality MARKETS: key, estimation = evaluation series, primary L
    ("crude", "crude_spot", 15), ("heatoil", "ho_spot", 15), ("gasoline", "gas_spot", 15),
    ("gold", "gold", 10), ("silver", "silver", 10), ("soybean", "soybean", 10),
    ("sugar", "sugar", 10)]
SEAS_FRI = fh.weekly_decision_dates("1986-01-03", "2026-10-16")    # the tool's Friday grid
_SEAS_TAB: dict = {}


def seasonal_weekly(series_name: str, L: int, fri: pd.DatetimeIndex | None = None) -> np.ndarray:
    """tool_seasonality primary signal of one market: walk-forward week-of-year analogs over
    the prior L years, h=8, threshold 70/30, estimated on `series_name` (signal per Friday
    depends only on that Friday's analogs, so any Friday grid gives the same values)."""
    if series_name not in _SEAS_TAB:
        _SEAS_TAB[series_name] = se.est_table(fh.load_series(series_name), series_name,
                                              (4, 8, 13))
    return se.seasonal_signal(SEAS_FRI if fri is None else fri, _SEAS_TAB[series_name], 8, L,
                              70, "week")


def primary_seasonality() -> dict:
    """tool_seasonality primary: 7 markets pooled, NET side, h=8 (spot / v1 series)."""
    parts = [(fh.make_signal_frame(SEAS_FRI, seasonal_weekly(ser, L)), ser, "price")
             for _, ser, L in SEAS_MARKETS]
    return dict(kind="pooled_net", parts=parts, side="net",
                names=[k for k, _, _ in SEAS_MARKETS])


_VAL = None


def valuation_vis() -> dict:
    """Valuation indices EMA2-EMA22, N=156, on tool_valuation's Friday grid, for every
    (market, comparator) pair the tool used: {(market, comparator): vi array} + grid."""
    global _VAL
    if _VAL is None:
        fri = vl.friday_grid()
        mi, ci = vl.market_inputs(fri), vl.comparator_inputs(fri)
        vis = {}
        for m in MARKETS5:
            for c in ("gold", "bond", "usd"):
                if m == "gold" and c == "gold":
                    continue
                vis[(m, c)] = vl.valuation_index(mi[m], ci[c], (2, 22), 156)["vi"]
        _VAL = dict(fri=fri, vis=vis)
    return _VAL


def primary_valuation() -> dict:
    """tool_valuation primary: crude (CL input) vs London gold, VI <= 20 LONG, h=8, crude_roll."""
    v = valuation_vis()
    sig = vl.setups(v["vis"][("crude", "gold")], 20.0, 80.0)
    sf = fh.make_signal_frame(v["fri"], sig)
    return dict(kind="single", parts=[(sf, "crude_roll", "price")], side="long")


_OI: dict = {}


def oi_signal(market: str) -> tuple[pd.DataFrame, np.ndarray]:
    """tool_open_interest primary rule (4-report F+O OI change in the bottom/top 10% of the
    last 156 reports while |4-report price change| <= its 156-report median) for one market,
    price input = that market's traded series. Returns (report frame, signal)."""
    if market not in _OI:
        fo = ol.load_fo_cot()
        f = ol.build_features(fo[market], fh.load_series(ol.TRADED[market]))
        _OI[market] = (fo[market], ol.rule_primary(f))
    return _OI[market]


def primary_open_interest() -> dict:
    """tool_open_interest primary: crude, LONG on a large 4-report OI decline in a range,
    h=8, crude_roll, timing cot."""
    rep, sig = oi_signal("crude")
    return dict(kind="single", parts=[(ol.signal_frame(rep, sig), "crude_roll", "cot")],
                side="long")


def dollar_weekly(fri: pd.DatetimeIndex) -> np.ndarray:
    """tool_intermarket leader rule: +1 when the broad dollar (DTWEXBGS, read as of Friday-7d)
    fell over 8 weeks, -1 when it rose (inverse sign)."""
    return il.leader_signal("usd", fri, 8)


def primary_intermarket() -> dict:
    """tool_intermarket primary: gold LONG when the dollar fell over 8 weeks, h=8, London gold."""
    gold = fh.load_series("gold")
    fri = il.fridays_for(gold)
    sf = fh.make_signal_frame(fri, dollar_weekly(fri))
    return dict(kind="single", parts=[(sf, "gold", "price")], side="long")


PRIMARY_BUILDERS = {"cot": primary_cot, "premium": primary_premium,
                    "seasonality": primary_seasonality, "valuation": primary_valuation,
                    "open_interest": primary_open_interest, "intermarket": primary_intermarket}


def evaluate_primary(spec: dict, k: int = fh.K_PERM) -> dict:
    """h=8 primary row of a rebuilt spec (single market: fh.evaluate on the pre-registered
    side; pooled net: tool_seasonality's pooled_net)."""
    if spec["kind"] == "single":
        sf, ser, timing = spec["parts"][0]
        return fh.evaluate(sf, ser, (8,), timing=timing, sides=(spec["side"],), k=k
                           ).iloc[0].to_dict()
    return se.pooled_net(spec["parts"], 8, k=k)


def verify_primaries(published: dict[str, dict], k: int = fh.K_PERM) -> dict:
    """Rebuild every primary and compare n, mean, excess, hit, perm_p (and T where present)
    with the published primary row. Returns {tool: dict(match, rebuilt, published, diffs)}."""
    out = {}
    for tool, build in PRIMARY_BUILDERS.items():
        spec = build()
        row = evaluate_primary(spec, k)
        pub = published[tool]
        diffs = {}
        for c in ("n", "mean", "excess", "hit", "perm_p", "T"):
            if c in pub and c in row and pub[c] is not None:
                a, b = float(row[c]), float(pub[c])
                if not (a == b or abs(a - b) <= 1e-12 * max(1.0, abs(b))):
                    diffs[c] = (a, b)
        out[tool] = dict(match=not diffs, diffs=diffs,
                         rebuilt={c: row.get(c) for c in ("n", "episodes", "mean", "excess",
                                                          "hit", "perm_p", "perm_p_exact",
                                                          "T", "period")})
    return out


# =============================================================================================
# 3. Confluence components on the COT report grid
# =============================================================================================
FRI_ALL = fh.weekly_decision_dates("1985-01-04", "2026-10-16")    # every Friday of the study


def friday_of_report_week(report_dates) -> pd.DatetimeIndex:
    """Friday of the calendar week of each COT report date (Mon=0 .. Fri=4): the Friday on
    whose 15:30 ET publication (Monday in holiday weeks) the report becomes public. Its CLOSE is
    known before the COT entry (first close on/after the following Tuesday)."""
    d = pd.DatetimeIndex(pd.to_datetime(report_dates))
    return d + pd.to_timedelta(4 - d.weekday.to_numpy(), unit="D")


def _on_report_grid(friday_values: np.ndarray, friday_grid: pd.DatetimeIndex,
                    report_dates) -> np.ndarray:
    """Value of a Friday-indexed signal at the Friday of each report week (NaN if absent)."""
    s = pd.Series(np.asarray(friday_values, float), index=pd.DatetimeIndex(friday_grid))
    return s.reindex(friday_of_report_week(report_dates)).to_numpy(dtype=float)


def friday_components(market: str) -> dict[str, tuple[np.ndarray, pd.DatetimeIndex]]:
    """Friday-grid component signals of one market (+1 / 0 / -1, NaN = undefined), each built
    exactly like the corresponding tool's pre-registered rule:
      premium   crude: sign(RCLC1/RCLC2 - 1); others: sign(near/far - 1) from the pysystemtrade
                curve (tool_premium 'pst_sign')
      seasonal  tool_seasonality primary of that market (crude estimated on crude_spot, L=15;
                gold/silver/soybean/sugar on their v1 series, L=10), h=8, 70/30
      valuation tool_valuation VI vs London gold (EMA2-EMA22, N=156, <=20 long, >=80 short);
                not defined for gold itself (gold is the comparator) -> component absent
      dollar    tool_intermarket lead_usd_L8 (inverse sign, 7-day publication lag)"""
    out = {}
    if market == "crude":
        out["premium"] = (premium_crude_weekly(FRI_ALL), FRI_ALL)
    else:
        out["premium"] = (pl.sig_sign(pl.weekly_asof(pl.pst_premium_daily(market), FRI_ALL)),
                          FRI_ALL)
    ser, L = {"crude": ("crude_spot", 15)}.get(market, (market, 10))
    out["seasonal"] = (seasonal_weekly(ser, L, FRI_ALL), FRI_ALL)
    if market != "gold":
        v = valuation_vis()
        out["valuation"] = (vl.setups(v["vis"][(market, "gold")], 20.0, 80.0), v["fri"])
    out["dollar"] = (dollar_weekly(FRI_ALL), FRI_ALL)
    return out


def report_grid_components(market: str) -> pd.DataFrame:
    """One row per COT report of `market`: report_date, friday (of the report week), shutdown,
    and the component signals cot, premium, seasonal, valuation, oi, dollar (+1/0/-1, NaN =
    undefined; the column is absent when the component does not exist for the market).
    COT and OI are read at the report date; the Friday components at the report week's Friday."""
    p = cot_panel()[market]
    rep = p["report_date"].reset_index(drop=True)
    df = pd.DataFrame({"report_date": rep, "friday": friday_of_report_week(rep),
                       "shutdown": p["shutdown"].to_numpy(bool)})
    lo, sh = lc.signal_masks(p["idx26"].to_numpy(), 80, 20)
    df["cot"] = np.where(p["idx26"].notna().to_numpy(), lo.astype(float) - sh.astype(float),
                         np.nan)
    oi_rep, oi_sig = oi_signal(market)
    if not np.array_equal(pd.DatetimeIndex(oi_rep["report_date"]).values,
                          pd.DatetimeIndex(rep).values):
        raise AssertionError(f"{market}: OI and COT report dates differ")
    df["oi"] = np.asarray(oi_sig, float)
    for name, (vals, grid) in friday_components(market).items():
        df[name] = _on_report_grid(vals, grid, rep)
    return df


def confluence_signal(df: pd.DataFrame, comps: list[str], need: int = 2,
                      rule: str = "majority", anchor: str | None = None) -> np.ndarray:
    """Confluence of the component columns `comps` (PREREG.md section 3).
      L = #components at +1, S = #components at -1 (undefined weeks are handled by the caller
      through `valid`).
      rule="majority":  long if L >= need and L > S;  short if S >= need and S > L.
      rule="unanimous": long if L >= need and S == 0; short if S >= need and L == 0.
      anchor="cot": COT-anchored (Larry's practice): long if cot == +1 and, among the OTHER
      components, L_o >= 1 and L_o > S_o; short mirror-image.
    Returns +1/0/-1 (float)."""
    A = df[comps].to_numpy(dtype=float)
    if anchor is not None:
        others = [c for c in comps if c != anchor]
        B = df[others].to_numpy(dtype=float)
        lo_o, sh_o = (B == 1).sum(1), (B == -1).sum(1)
        a = df[anchor].to_numpy(dtype=float)
        lo = (a == 1) & (lo_o >= need - 1) & (lo_o > sh_o)
        sh = (a == -1) & (sh_o >= need - 1) & (sh_o > lo_o)
        return lo.astype(float) - sh.astype(float)
    L, S = (A == 1).sum(1), (A == -1).sum(1)
    if rule == "majority":
        lo, sh = (L >= need) & (L > S), (S >= need) & (S > L)
    elif rule == "unanimous":
        lo, sh = (L >= need) & (S == 0), (S >= need) & (L == 0)
    else:
        raise ValueError(rule)
    return lo.astype(float) - sh.astype(float)


# =============================================================================================
# 4. Statistics helpers
# =============================================================================================
def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up, monotone); NaN p-values stay NaN and are not
    counted in m."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def by_qvalues(p) -> np.ndarray:
    """Benjamini-Yekutieli q-values (valid under arbitrary dependence): BH x sum_{i<=m} 1/i."""
    p = np.asarray(p, dtype=float)
    m = int((~np.isnan(p)).sum())
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if m else 1.0
    q = bh_qvalues(p) * c
    return np.where(np.isnan(q), np.nan, np.minimum(q, 1.0))


def block_starts(T: int, B: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap: (B, ceil(T/block)) random block start positions."""
    nb = int(np.ceil(T / block))
    return rng.integers(0, T, size=(B, nb))


def block_indices(starts: np.ndarray, T: int, block: int) -> np.ndarray:
    """Row indices (B, T) of the resampled series built from blocks starting at `starts`
    (each block = `block` consecutive rows, wrapping around), truncated to T rows."""
    off = np.arange(block)
    idx = (starts[:, :, None] + off[None, None, :]) % T
    return idx.reshape(starts.shape[0], -1)[:, :T]


def excess_from_rows(sig: np.ndarray, r: np.ndarray, side: str,
                     group: np.ndarray | None = None) -> float:
    """Excess of the signal rows (SPEC2 rule 3) on a set of eligible rows: mean signed return
    of the setup rows minus the signed baseline (mean r over all rows). With `group` (market
    ids for pooled cells) every row's baseline is its own market's mean r over the rows of that
    market (excess vs own baseline, fh.evaluate_pooled convention). side = long/short/net."""
    sig = np.asarray(sig, float)
    r = np.asarray(r, float)
    if group is None:
        base = np.full(len(r), r.mean())
    else:
        base = pd.Series(r).groupby(np.asarray(group)).transform("mean").to_numpy()
    if side == "long":
        m = sig == 1
        vals = r[m] - base[m]
    elif side == "short":
        m = sig == -1
        vals = -(r[m] - base[m])
    else:
        m = sig != 0
        vals = sig[m] * (r[m] - base[m])
    return float(vals.mean()) if m.any() else np.nan


def bootstrap_excess(sig, r, side: str, *, block: int = BLOCK_WEEKS, B: int = BOOT_B,
                     seed: int = BOOT_SEED, group=None, cal=None) -> dict:
    """95% interval of the excess by a circular moving-block bootstrap over WEEKS.
    Single market: blocks of `block` consecutive eligible weeks. Pooled (`group` + `cal` given):
    blocks of `block` consecutive CALENDAR weeks of the union calendar `cal` (one week = all
    markets' rows dated that week, so cross-market correlation is kept); baselines are
    recomputed per market inside every resample. Resamples without a setup row are dropped
    (reported as n_undefined)."""
    sig, r = np.asarray(sig, float), np.asarray(r, float)
    rng = np.random.default_rng(seed)
    obs = excess_from_rows(sig, r, side, group)
    stats = np.empty(B)
    if cal is None:
        T = len(r)
        idx = block_indices(block_starts(T, B, block, rng), T, block)
        for b in range(B):
            stats[b] = excess_from_rows(sig[idx[b]], r[idx[b]], side)
    else:
        cal = np.asarray(cal)
        weeks, inv = np.unique(cal, return_inverse=True)
        rows_of = [np.flatnonzero(inv == w) for w in range(len(weeks))]
        T = len(weeks)
        idx = block_indices(block_starts(T, B, block, rng), T, block)
        g = np.asarray(group)
        for b in range(B):
            rows = np.concatenate([rows_of[w] for w in idx[b]])
            # repeated weeks: tag market ids so baselines stay per market (not per copy)
            stats[b] = excess_from_rows(sig[rows], r[rows], side, g[rows])
    ok = ~np.isnan(stats)
    v = stats[ok]
    return dict(excess=obs, ci_lo=float(np.quantile(v, 0.025)), ci_hi=float(np.quantile(v, 0.975)),
                boot_sd=float(v.std(ddof=1)), share_le_0=float(np.mean(v <= 0)),
                B=int(ok.sum()), n_undefined=int((~ok).sum()), block_weeks=block, seed=seed)


def bootstrap_diff(sig_a, sig_b, r, side: str, *, block: int = BLOCK_WEEKS, B: int = BOOT_B,
                   seed: int = BOOT_SEED, group=None, cal=None) -> dict:
    """D = excess(sig_a) - excess(sig_b) on the SAME eligible rows, with a circular block
    bootstrap (weeks resampled jointly for both signals). One-sided p for 'a beats b':
    p = (1 + #{D* <= 0}) / (1 + B_valid). Also the 95% interval of D*."""
    a, b_, r = np.asarray(sig_a, float), np.asarray(sig_b, float), np.asarray(r, float)
    rng = np.random.default_rng(seed)
    g = None if group is None else np.asarray(group)
    obs = excess_from_rows(a, r, side, g) - excess_from_rows(b_, r, side, g)
    stats = np.empty(B)
    if cal is None:
        T = len(r)
        idx = block_indices(block_starts(T, B, block, rng), T, block)
        for k in range(B):
            i = idx[k]
            stats[k] = excess_from_rows(a[i], r[i], side) - excess_from_rows(b_[i], r[i], side)
    else:
        weeks, inv = np.unique(np.asarray(cal), return_inverse=True)
        rows_of = [np.flatnonzero(inv == w) for w in range(len(weeks))]
        T = len(weeks)
        idx = block_indices(block_starts(T, B, block, rng), T, block)
        for k in range(B):
            i = np.concatenate([rows_of[w] for w in idx[k]])
            stats[k] = (excess_from_rows(a[i], r[i], side, g[i])
                        - excess_from_rows(b_[i], r[i], side, g[i]))
    ok = ~np.isnan(stats)
    v = stats[ok]
    return dict(diff=float(obs), ci_lo=float(np.quantile(v, 0.025)),
                ci_hi=float(np.quantile(v, 0.975)),
                p_boot=(1 + int(np.sum(v <= 0))) / (1 + len(v)), B=int(ok.sum()),
                n_undefined=int((~ok).sum()))


def eligible_rows(sf: pd.DataFrame, series, timing: str, h: int = 8) -> pd.DataFrame:
    """Eligible weeks of a signal frame at horizon h (valid, not excluded, forward return on
    the traded series): decision_date, signal, r (unsigned forward return), in time order."""
    s, _ = fh._resolve(series, None)
    sf = sf.sort_values("decision_date", kind="mergesort").reset_index(drop=True)
    ep = fh.entry_positions(sf["decision_date"], s.index, timing)
    r = fh.forward_returns(s, ep, h)
    m = sf["valid"].to_numpy(bool) & ~sf["excluded"].to_numpy(bool) & ~np.isnan(r)
    return pd.DataFrame({"decision_date": sf["decision_date"][m].to_numpy(),
                         "signal": sf["signal"].to_numpy(float)[m], "r": r[m]})

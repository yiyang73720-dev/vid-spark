#!/usr/bin/env python3
"""
oi_lib -- data loading and signal construction for the Larry Williams OPEN INTEREST tool
(SPEC2 battery, see PREREG.md in this directory).

Everything here is look-ahead free at the COT decision time (report_date, a Tuesday as-of
date that is published the following Friday): every statistic of report t uses only reports
<= t and prices on or before report_date_t. The evaluation itself (entry timing, forward
returns, permutation tests, strategy) is done by the shared harness `fund_harness`.

Data sources (all parsed, never executed; third-party JSON/CSV read with json/pandas only):
  * F+O total OI and commercials:  S/data/cot_weekly.csv  via the v1 engine (lc.load_cot)
  * crude futures-only OI 2010+:   S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json
                                   (its last row, 2026-09-29, is mixed-basis and is dropped)
  * legacy futures-only OI 2000-2020 (scout, deduplicated): S/work2/scout/cot_smalltrader_weekly.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HARNESS = Path(__file__).resolve().parents[1] / "harness"     # battery/harness
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))
import fund_harness as fh  # noqa: E402

ROOT = fh.ROOT
CFTC_CL_JSON = ROOT / "ext" / "accomplish999_crude-oil" / "data" / "series" / "CFTC_CL.json"
LEGACY_CSV = ROOT / "work2" / "scout" / "cot_smalltrader_weekly.csv"
CFTC_CL_MIXED_ROW = pd.Timestamp("2026-09-29")   # mixed-basis row (equals the F+O total)
SPLICE_DATE = pd.Timestamp("2010-01-05")         # legacy before, CFTC_CL from this date on

MARKETS = ("crude", "gold", "silver", "soybean", "sugar")
# Traded series per market (SPEC2 rule 2): crude on the roll-adjusted index, others on v1.
TRADED = {"crude": "crude_roll", "gold": "gold", "silver": "silver", "soybean": "soybean",
          "sugar": "sugar"}

PCT_WINDOW = 156     # 3 years of weekly reports (Larry's long COT window)
PCT_MIN = 52         # at least one year of defined values before a percentile is used
PRICE_STALE_DAYS = 4  # price input must be at most 4 calendar days older than the report


# ========================================================================================
# 1. OI data
# ========================================================================================
def load_fo_cot() -> dict[str, pd.DataFrame]:
    """F+O disaggregated COT per market: report_date, oi, comm_net (v1 definitions), shutdown."""
    cot = fh.lc.load_cot()
    out = {}
    for m in MARKETS:
        g = cot[cot["market"] == m].sort_values("report_date").reset_index(drop=True)
        g = g[["report_date", "oi", "comm_net"]].copy()
        g["oi"] = g["oi"].astype(float)
        g["comm_net"] = g["comm_net"].astype(float)
        g["shutdown"] = fh.cot_shutdown_mask(g["report_date"])
        out[m] = g
    return out


def load_cftc_cl_futonly(fo_crude: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Crude futures-only OI from CFTC_CL.json (json.load only). Drops the 2026-09-29 row,
    which is mixed-basis: its OI equals the F+O total of that date (checked here)."""
    doc = json.load(open(CFTC_CL_JSON, encoding="utf-8"))
    ob = pd.DataFrame(doc["observations"])
    df = pd.DataFrame({"report_date": pd.to_datetime(ob["date"]),
                       "oi": pd.to_numeric(ob["open_interest"]).astype(float)})
    df = df.sort_values("report_date").reset_index(drop=True)
    info = {"rows_in_file": int(len(df)), "first": df["report_date"].iloc[0].date().isoformat(),
            "last_in_file": df["report_date"].iloc[-1].date().isoformat()}
    mixed = df["report_date"] == CFTC_CL_MIXED_ROW
    if fo_crude is not None and mixed.any():
        fo_val = fo_crude.loc[fo_crude["report_date"] == CFTC_CL_MIXED_ROW, "oi"]
        info["dropped_row_oi"] = float(df.loc[mixed, "oi"].iloc[0])
        info["dropped_row_equals_FO_oi"] = bool(len(fo_val) and
                                                float(fo_val.iloc[0]) == info["dropped_row_oi"])
    df = df[~mixed].reset_index(drop=True)
    info["rows_used"] = int(len(df))
    info["last_used"] = df["report_date"].iloc[-1].date().isoformat()
    df["shutdown"] = fh.cot_shutdown_mask(df["report_date"])
    return df, info


def load_legacy_futonly(market: str) -> pd.DataFrame:
    """Legacy futures-only OI 2000-2020 (scout file, already deduplicated; no 2007)."""
    lg = pd.read_csv(LEGACY_CSV, dtype={"code": str})
    g = lg[lg["market"] == market].copy()
    g["report_date"] = pd.to_datetime(g["report_date"])
    g = g.sort_values("report_date").reset_index(drop=True)
    if g["report_date"].duplicated().any():
        raise ValueError(f"duplicate legacy report dates for {market}")
    out = pd.DataFrame({"report_date": g["report_date"], "oi": g["oi"].astype(float)})
    out["shutdown"] = fh.cot_shutdown_mask(out["report_date"])
    return out


def splice_crude_futonly(cftc_cl: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Futures-only crude OI 2000-2026: legacy before SPLICE_DATE, CFTC_CL from it on.
    Returns the frame and the agreement check on the overlapping dates."""
    leg = load_legacy_futonly("crude")
    both = leg.merge(cftc_cl[["report_date", "oi"]], on="report_date", suffixes=("_leg", "_cl"))
    chk = {"overlap_dates": int(len(both)),
           "equal_oi": int((both["oi_leg"] == both["oi_cl"]).sum())}
    sp = pd.concat([leg[leg["report_date"] < SPLICE_DATE],
                    cftc_cl[cftc_cl["report_date"] >= SPLICE_DATE]], ignore_index=True)
    sp = sp[["report_date", "oi"]].sort_values("report_date").reset_index(drop=True)
    sp["shutdown"] = fh.cot_shutdown_mask(sp["report_date"])
    gaps = sp["report_date"].diff().dt.days
    chk["gaps_gt_8d"] = [f"{sp['report_date'][i-1].date()}->{sp['report_date'][i].date()}"
                         for i in np.flatnonzero(gaps.to_numpy() > 8)]
    return sp, chk


# ========================================================================================
# 2. Look-ahead-free building blocks
# ========================================================================================
def asof_price(series: pd.Series, dates, max_stale_days: int = PRICE_STALE_DAYS) -> np.ndarray:
    """Close of `series` on or before each date; NaN if that close is more than
    max_stale_days calendar days old (series not started / ended / data gap)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    pos = np.searchsorted(series.index.values, d.values, side="right") - 1
    out = np.full(len(d), np.nan)
    ok = pos >= 0
    age = np.full(len(d), np.inf)
    age[ok] = (d.values[ok] - series.index.values[pos[ok]]) / np.timedelta64(1, "D")
    ok &= age <= max_stale_days
    out[ok] = series.to_numpy(dtype=float)[pos[ok]]
    return out


def lag_change(x: np.ndarray, dates, w: int, max_days: int | None = None) -> np.ndarray:
    """x_t / x_{t-w} - 1 over REPORT rows; NaN if report t-w is more than max_days
    (default 7w+7) calendar days before report t (gap guard) or an input is NaN."""
    x = np.asarray(x, dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    out = np.full(len(x), np.nan)
    if len(x) <= w:
        return out
    lim = 7 * w + 7 if max_days is None else max_days
    span = (d[w:] - d[:-w]).days.to_numpy()
    ch = x[w:] / x[:-w] - 1.0
    out[w:] = np.where(span <= lim, ch, np.nan)
    return out


def pct156(x: np.ndarray) -> np.ndarray:
    """100 x share of the last <=156 report values (incl. the current one) that are <= x_t;
    NaN until 52 defined values exist. Uses only data known at t."""
    return fh.expanding_percentile(x, min_periods=PCT_MIN, window=PCT_WINDOW)


def range_index(x: np.ndarray, n: int) -> np.ndarray:
    """Larry-style 0-100 index of x within its min/max over the last n reports incl. t
    (NaN for the first n-1 reports or when max == min) -- the v1 COT-index formula."""
    return fh.lc.larry_index(np.asarray(x, dtype=float), n)


def rolling_extreme(x: np.ndarray, n: int, how: str) -> np.ndarray:
    """Rolling max/min of x over the last n reports incl. t (NaN before n reports)."""
    s = pd.Series(np.asarray(x, dtype=float))
    r = s.rolling(n, min_periods=n)
    return (r.max() if how == "max" else r.min()).to_numpy()


# ========================================================================================
# 3. Features and rules
# ========================================================================================
WINDOWS = (1, 2, 4, 13)


def build_features(rep: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Per report: OI and price changes over W reports, their 3-year percentiles, OI range
    indices, 12-month OI growth (Hong & Yogo) and WILLCO when comm_net is present.
    `price` = the traded series; P_t = its close as of report_date (<= 4 days stale)."""
    f = rep[["report_date"]].copy()
    d = f["report_date"]
    oi = rep["oi"].to_numpy(dtype=float)
    px = asof_price(price, d)
    f["oi"] = oi
    f["px"] = px
    for w in WINDOWS:
        f[f"dOI{w}"] = lag_change(oi, d, w)
        f[f"dP{w}"] = lag_change(px, d, w)
        f[f"pOI{w}"] = pct156(f[f"dOI{w}"].to_numpy())
        f[f"pP{w}"] = pct156(np.abs(f[f"dP{w}"].to_numpy()))
    for n in (26, 52):
        f[f"oi_idx{n}"] = range_index(oi, n)
        f[f"oi_max{n}"] = rolling_extreme(oi, n, "max")
        f[f"oi_min{n}"] = rolling_extreme(oi, n, "min")
    # Hong & Yogo (2012): 12-month log growth of OI in contracts (t-52 within 372 days).
    g = lag_change(oi, d, 52, max_days=372)
    f["hy_g"] = np.log1p(g)
    f["hy_pct"] = pct156(f["hy_g"].to_numpy())
    if "comm_net" in rep:
        w_ratio = rep["comm_net"].to_numpy(dtype=float) / oi
        f["willco26"] = range_index(w_ratio, 26)
    return f


def _sig(long_m, short_m, valid) -> np.ndarray:
    """+1/-1/0 with NaN where the rule's statistics are undefined."""
    s = np.where(long_m, 1.0, np.where(short_m, -1.0, 0.0))
    return np.where(valid, s, np.nan)


def rule_oi_pct(f: pd.DataFrame, w: int, p: float, range_filter: bool) -> np.ndarray:
    """Grid C (and the PRIMARY = w=4, p=10, range_filter=True): long when the w-report OI change
    is in the bottom p% of the last 3 years, short when in the top p%; with range_filter the
    price must also be range-bound (|dP_w| at or below its 3-year median)."""
    po = f[f"pOI{w}"].to_numpy()
    valid = ~np.isnan(po)
    rng = np.ones(len(f), bool)
    if range_filter:
        pp = f[f"pP{w}"].to_numpy()
        valid &= ~np.isnan(pp)
        rng = np.nan_to_num(pp, nan=101.0) <= 50.0
    po0 = np.nan_to_num(po, nan=50.0)
    return _sig((po0 <= p) & rng, (po0 >= 100.0 - p) & rng, valid)


def rule_primary(f: pd.DataFrame) -> np.ndarray:
    """PRIMARY: pOI4 <= 10 and pP4 <= 50 -> long; pOI4 >= 90 and pP4 <= 50 -> short."""
    return rule_oi_pct(f, 4, 10, True)


def rule_larry30(f: pd.DataFrame, n: int) -> np.ndarray:
    """Variant D: long if OI <= 0.70 x max(OI, last n), short if OI >= 1.30 x min(OI, last n),
    each with the 4-week range filter."""
    oi = f["oi"].to_numpy()
    mx, mn = f[f"oi_max{n}"].to_numpy(), f[f"oi_min{n}"].to_numpy()
    pp = f["pP4"].to_numpy()
    valid = ~np.isnan(mx) & ~np.isnan(pp)
    rng = np.nan_to_num(pp, nan=101.0) <= 50.0
    lo = (oi <= 0.70 * np.nan_to_num(mx, nan=np.inf)) & rng
    sh = (oi >= 1.30 * np.nan_to_num(mn, nan=np.inf)) & rng
    return _sig(lo, sh, valid)


def rule_liq_bottom(f: pd.DataFrame, w: int) -> np.ndarray:
    """Variant E: long if pOI_w <= 10 and dP_w < 0 (liquidation into a decline);
    short if pOI_w >= 90 and dP_w > 0 (blow-off build-up into a rally)."""
    po, dp = f[f"pOI{w}"].to_numpy(), f[f"dP{w}"].to_numpy()
    valid = ~np.isnan(po) & ~np.isnan(dp)
    po0, dp0 = np.nan_to_num(po, nan=50.0), np.nan_to_num(dp, nan=0.0)
    return _sig((po0 <= 10) & (dp0 < 0), (po0 >= 90) & (dp0 > 0), valid)


def quadrant(f: pd.DataFrame, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrant id 1..4 (0 when a change is exactly 0) and validity over w reports.
    Q1 price up/OI up, Q2 price up/OI down, Q3 price down/OI up, Q4 price down/OI down."""
    dp, do = f[f"dP{w}"].to_numpy(), f[f"dOI{w}"].to_numpy()
    valid = ~np.isnan(dp) & ~np.isnan(do)
    dp0, do0 = np.nan_to_num(dp), np.nan_to_num(do)
    q = np.select([(dp0 > 0) & (do0 > 0), (dp0 > 0) & (do0 < 0),
                   (dp0 < 0) & (do0 > 0), (dp0 < 0) & (do0 < 0)], [1, 2, 3, 4], 0)
    return q, valid


def rule_quad_classic(f: pd.DataFrame, w: int) -> np.ndarray:
    """Textbook reading: +1 in Q1 and Q4, -1 in Q2 and Q3 (= sign(dP) x sign(dOI))."""
    q, valid = quadrant(f, w)
    return _sig(np.isin(q, (1, 4)), np.isin(q, (2, 3)), valid)


def rule_quad_trend(f: pd.DataFrame, w: int) -> np.ndarray:
    """Trend-confirmation reading: +1 in Q1 (price up, OI up), -1 in Q3 (price down, OI up)."""
    q, valid = quadrant(f, w)
    return _sig(q == 1, q == 3, valid)


def rule_oi_range(f: pd.DataFrame, n: int) -> np.ndarray:
    """Variant G (Larry's contrarian reading): long if OI index_n <= 20, short if >= 80."""
    idx = f[f"oi_idx{n}"].to_numpy()
    valid = ~np.isnan(idx)
    i0 = np.nan_to_num(idx, nan=50.0)
    return _sig(i0 <= 20, i0 >= 80, valid)


def rule_hy(f: pd.DataFrame, use_sign: bool) -> np.ndarray:
    """Variant H (Hong & Yogo 2012, pro-cyclical): long when 12-month OI growth is above its
    trailing 3-year median (or > 0 for use_sign), short otherwise."""
    if use_sign:
        g = f["hy_g"].to_numpy()
        valid = ~np.isnan(g)
        g0 = np.nan_to_num(g)
        return _sig(g0 > 0, g0 < 0, valid)
    p = f["hy_pct"].to_numpy()
    valid = ~np.isnan(p)
    p0 = np.nan_to_num(p, nan=50.0)
    return _sig(p0 > 50, p0 <= 50, valid)


def rule_willco(f: pd.DataFrame) -> np.ndarray:
    """Variant I (Larry's WILLCO): 26-week index of comm_net/OI; long >= 80, short <= 20."""
    idx = f["willco26"].to_numpy()
    valid = ~np.isnan(idx)
    i0 = np.nan_to_num(idx, nan=50.0)
    return _sig(i0 >= 80, i0 <= 20, valid)


def signal_frame(rep: pd.DataFrame, sig: np.ndarray) -> pd.DataFrame:
    """Harness signal frame with COT timing semantics (decision_date = report_date, shutdown
    reports excluded)."""
    return fh.make_signal_frame(rep["report_date"], sig,
                                excluded=rep["shutdown"].to_numpy(dtype=bool))

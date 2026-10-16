"""Data-scout sanity checks for the Larry Williams fundamentals battery (spec v2).

What this script does
---------------------
1. Futures curve (two contract months per market) from the public pysystemtrade repo
   (robcarver17/pysystemtrade, GPL-3, `data/futures/multiple_prices_csv/<INSTR>.csv`).
   Each row holds three contracts: PRICE (the contract the system holds), CARRY (an adjacent
   contract used for carry) and FORWARD (next contract to roll into), each with its YYYYMMDD id.
   We collapse the file to one row per calendar date (last timestamp of the date; recent years
   are hourly), define NEAR / FAR as the nearer / further of CARRY and PRICE by contract id, and
   check that NEAR and FAR are genuinely distinct contracts (ids differ, prices not duplicated,
   imperfect return correlation, no one-leg staleness) and how close NEAR is to the front month.
   Crude and heating oil are cross-checked against EIA NYMEX contract-1..4 settlements that are
   already on disk (`S/ext/accomplish999_crude-oil`).
2. Small-trader (non-reportable) COT positions from bgithub1/cme_open_interest
   (`cot_history.csv` = CFTC legacy report, `cot_new_history.csv` = disaggregated report).
   We check coverage, the accounting identity OI = reportable + non-reportable, and which report
   variant (futures-only or futures+options combined) the files are by matching open interest
   against files whose variant is known.

All third-party files are UNTRUSTED: they are only parsed as data (run this with `python3 -I`).
Outputs (all under S/work2/scout/):
    curve_daily.csv           tidy daily NEAR/FAR curve per market
    cot_smalltrader_weekly.csv tidy weekly legacy COT incl. non-reportable positions
    scout_summary.json        every number reported in INVENTORY.md
    inventory_tables.md       markdown tables generated from scout_summary.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/scout/../..)
S = Path(os.environ.get("LARRY_COT_ROOT") or PKG)        # ROOT: data/ ext/ ext2/ work2/
PST = S / "ext2/robcarver17_pysystemtrade/data/futures/multiple_prices_csv"
BG = S / "ext2/bgithub1_cme_open_interest"
EIA = S / "ext/accomplish999_crude-oil/data/series"
OUT = S / "work2/scout"

# pysystemtrade instrument code -> market name used in this project
CURVE_MARKETS = {
    "CRUDE_W": "crude", "GOLD": "gold", "SILVER": "silver", "SOYBEAN": "soybean",
    "SUGAR11": "sugar", "CORN": "corn", "WHEAT": "wheat", "COPPER": "copper",
    "GAS_US": "natgas", "HEATOIL": "heatoil",
}

# CFTC contract market codes -> market name
COT_CODES = {
    "067651": "crude", "088691": "gold", "084691": "silver", "005602": "soybean",
    "080732": "sugar", "002602": "corn", "001602": "wheat", "085692": "copper",
    "023651": "natgas", "022651": "heatoil",
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def contract_month_index(ids: pd.Series) -> pd.Series:
    """Contract id YYYYMMDD (DD usually 00) -> absolute month number year*12 + (month-1)."""
    ids = ids.astype("Int64")
    year = ids // 10000
    month = (ids // 100) % 100
    return year * 12 + (month - 1)


def date_month_index(dates: pd.Series) -> pd.Series:
    """Calendar date -> absolute month number year*12 + (month-1)."""
    return dates.dt.year * 12 + (dates.dt.month - 1)


def r(x, nd=6):
    """Round floats for JSON output, pass through None/NaN as None."""
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def load_eia(series_id: str) -> pd.Series:
    """Load one EIA series (json written by a third-party repo) as a float Series by date."""
    with open(EIA / f"{series_id}.json") as fh:
        obs = json.load(fh)["observations"]
    s = pd.Series({o["date"]: o["value"] for o in obs})
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()


# ----------------------------------------------------------------------------------------------
# 1. futures curve from pysystemtrade multiple prices
# ----------------------------------------------------------------------------------------------
def load_multiple_prices(instr: str) -> pd.DataFrame:
    """Read one multiple-prices file and collapse it to one row per calendar date.

    Recent years are stored hourly; the daily close is stamped 23:00. We keep the LAST row of
    each calendar date as a whole row (23:00 when present; never mixing columns from different
    timestamps) and record whether that row was the 23:00 stamp. Saturday/Sunday dates (Sunday
    evening session bars in the hourly era) are dropped because they are not daily closes.
    """
    raw = pd.read_csv(PST / f"{instr}.csv")
    raw["ts"] = pd.to_datetime(raw["DATETIME"])
    raw = raw.sort_values("ts", kind="stable")
    raw["date"] = raw["ts"].dt.normalize()
    rows_per_date = raw.groupby("date").size()
    multi = rows_per_date[rows_per_date > 1]
    daily = raw.drop_duplicates("date", keep="last").reset_index(drop=True)
    weekend = daily["date"].dt.weekday >= 5
    daily.attrs["n_raw_rows"] = len(raw)
    daily.attrs["n_weekend_dates_dropped"] = int(weekend.sum())
    daily.attrs["first_intraday_date"] = str(multi.index.min().date()) if len(multi) else None
    daily.attrs["n_intraday_dates"] = int(len(multi))
    daily = daily[~weekend].reset_index(drop=True)
    daily["is_2300"] = daily["ts"].dt.hour.eq(23)
    return daily


def build_curve(instr: str, market: str) -> tuple[pd.DataFrame, dict]:
    """Return tidy NEAR/FAR daily curve for one market plus a dict of sanity statistics."""
    d = load_multiple_prices(instr)
    stats: dict = {"instrument": instr, "market": market, "raw_rows": d.attrs["n_raw_rows"],
                   "n_weekend_dates_dropped": d.attrs["n_weekend_dates_dropped"],
                   "first_date_with_intraday_rows": d.attrs["first_intraday_date"],
                   "n_dates_with_intraday_rows": d.attrs["n_intraday_dates"]}

    # NEAR / FAR = nearer / further of CARRY and PRICE by contract id
    carry_m = contract_month_index(d["CARRY_CONTRACT"])
    price_m = contract_month_index(d["PRICE_CONTRACT"])
    carry_is_near = carry_m < price_m
    near_c = np.where(carry_is_near, d["CARRY_CONTRACT"], d["PRICE_CONTRACT"])
    far_c = np.where(carry_is_near, d["PRICE_CONTRACT"], d["CARRY_CONTRACT"])
    near_p = np.where(carry_is_near, d["CARRY"], d["PRICE"])
    far_p = np.where(carry_is_near, d["PRICE"], d["CARRY"])

    cur = pd.DataFrame({
        "market": market, "date": d["date"],
        "near_contract": near_c.astype("int64"), "near": near_p.astype(float),
        "far_contract": far_c.astype("int64"), "far": far_p.astype(float),
        "price_contract": d["PRICE_CONTRACT"].astype("int64"), "price": d["PRICE"].astype(float),
        "carry_contract": d["CARRY_CONTRACT"].astype("int64"), "carry": d["CARRY"].astype(float),
        "forward_contract": d["FORWARD_CONTRACT"].astype("int64"), "forward": d["FORWARD"].astype(float),
        "close_stamp_2300": d["is_2300"],
    })

    # a row whose CARRY and PRICE are the SAME contract is a duplicated leg, not a spread:
    # blank both legs so it can never enter a curve signal
    same_id = cur["near_contract"] == cur["far_contract"]
    stats["n_same_contract_id_blanked"] = int(same_id.sum())
    cur.loc[same_id, ["near", "far"]] = np.nan
    both = cur["near"].notna() & cur["far"].notna()
    stats.update({
        "first_date": str(cur["date"].min().date()),
        "last_date": str(cur["date"].max().date()),
        "n_days": int(len(cur)),
        "n_days_not_2300_stamp": int((~cur["close_stamp_2300"]).sum()),
        "share_carry_is_near": r(float(carry_is_near.mean())),
        "n_carry_nan": int(d["CARRY"].isna().sum()),
        "n_price_nan": int(d["PRICE"].isna().sum()),
        "n_days_both_legs": int(both.sum()),
        "first_date_both_legs": str(cur.loc[both, "date"].min().date()),
        "n_nonpositive_price": int(((cur["near"] <= 0) | (cur["far"] <= 0)).sum()),
    })

    c = cur[both].copy()
    gap = contract_month_index(c["far_contract"]) - contract_month_index(c["near_contract"])
    stats["month_gap_far_minus_near"] = {str(k): int(v) for k, v in gap.value_counts().sort_index().items()}
    # months from the observation date to the NEAR / FAR delivery month (0 = delivery this month)
    m_near = contract_month_index(c["near_contract"]) - date_month_index(c["date"])
    m_far = contract_month_index(c["far_contract"]) - date_month_index(c["date"])
    stats["months_to_near_delivery"] = {"min": int(m_near.min()), "median": r(float(m_near.median()), 1),
                                        "max": int(m_near.max())}
    stats["months_to_far_delivery"] = {"min": int(m_far.min()), "median": r(float(m_far.median()), 1),
                                       "max": int(m_far.max())}

    # duplication checks: identical prices on the same day
    stats["share_days_near_equals_far"] = r(float((c["near"] == c["far"]).mean()))
    spread = c["near"] / c["far"] - 1.0  # > 0 = nearby premium (inverted / backwardated)
    stats["spread_near_over_far_minus_1"] = {
        "median": r(float(spread.median())), "p05": r(float(spread.quantile(0.05))),
        "p95": r(float(spread.quantile(0.95))),
    }
    stats["share_days_near_premium"] = r(float((spread > 0).mean()))
    dec = (c["date"].dt.year // 10) * 10
    stats["share_days_near_premium_by_decade"] = {
        str(k): r(float(v)) for k, v in (spread > 0).groupby(dec).mean().items()
    }

    # return-based distinctness: consecutive days with unchanged contract ids on both legs
    prev = c.shift(1)
    same = (c["near_contract"] == prev["near_contract"]) & (c["far_contract"] == prev["far_contract"])
    rn = np.log(c["near"] / prev["near"])[same]
    rf = np.log(c["far"] / prev["far"])[same]
    stats["n_same_contract_day_pairs"] = int(same.sum())
    stats["corr_daily_logret_near_far"] = r(float(np.corrcoef(rn, rf)[0, 1]))
    stats["std_daily_logret_near"] = r(float(rn.std()))
    stats["std_daily_logret_far"] = r(float(rf.std()))
    stats["share_near_unchanged_far_moved"] = r(float(((rn == 0) & (rf != 0)).mean()))
    stats["share_far_unchanged_near_moved"] = r(float(((rf == 0) & (rn != 0)).mean()))
    # same staleness metrics on the most recent 5 years (hourly-era data)
    recent = c.loc[same.index[same], "date"] >= (c["date"].max() - pd.DateOffset(years=5))
    stats["share_near_unchanged_far_moved_last5y"] = r(float(((rn == 0) & (rf != 0))[recent].mean()))
    stats["share_far_unchanged_near_moved_last5y"] = r(float(((rf == 0) & (rn != 0))[recent].mean()))

    # weekly decision coverage: Fridays (last available day of each W-FRI week) with both legs
    wk = c.set_index("date")[["near", "far"]].resample("W-FRI").last().dropna()
    stats["n_weeks_both_legs"] = int(len(wk))
    return cur, stats


def nymex_cl_last_trade(month_idx: int) -> pd.Timestamp:
    """Approximate CL last trading day for delivery month `month_idx` (absolute month number):
    3 business days before the 25th of the prior month (25th rolled back if not a weekday).
    Exchange holidays are ignored, so callers exclude dates within 2 business days of it."""
    y, m = divmod(month_idx - 1, 12)
    d25 = pd.Timestamp(year=y, month=m + 1, day=25)
    if d25.weekday() >= 5:
        d25 = d25 - pd.offsets.BDay(1)
    return d25 - pd.offsets.BDay(3)


def nymex_ho_last_trade(month_idx: int) -> pd.Timestamp:
    """Approximate HO last trading day: last business day of the month before delivery."""
    y, m = divmod(month_idx - 1, 12)
    return pd.Timestamp(year=y, month=m + 1, day=1) + pd.offsets.BMonthEnd(0)


def contract_number_on(date: pd.Timestamp, month_idx: int, ltd_fn) -> int | None:
    """Return k such that contract `month_idx` is NYMEX 'contract k' (1 = front) on `date`,
    or None when `date` is within 2 business days of a relevant expiry (ambiguous)."""
    # front = earliest delivery month whose last trading day is on/after date
    front = date.year * 12 + date.month - 1  # start from the current calendar month
    while ltd_fn(front) < date:
        front += 1
    for mm in (front - 1, front):  # ambiguity window around the nearest expiries
        if abs(len(pd.bdate_range(min(date, ltd_fn(mm)), max(date, ltd_fn(mm)))) - 1) <= 2:
            return None
    # monthly listing for CL and HO, so contract number = month distance + 1
    return int(month_idx - front + 1)


def eia_crosscheck(cur: pd.DataFrame, ltd_fn, eia_ids: dict[int, str]) -> dict:
    """Compare pysystemtrade contract prices with EIA NYMEX 'contract k' settlements."""
    eia = {k: load_eia(v) for k, v in eia_ids.items()}
    rows = []
    # weekly sample (Wednesdays) keeps this fast while covering the whole history
    sample = cur[cur["date"].dt.weekday == 2]
    for _, row in sample.iterrows():
        for leg in ("carry", "price", "forward"):
            px = row[leg]
            if not np.isfinite(px):
                continue
            mi = (row[f"{leg}_contract"] // 10000) * 12 + (row[f"{leg}_contract"] // 100) % 100 - 1
            k = contract_number_on(row["date"], int(mi), ltd_fn)
            if k is None or k not in eia:
                continue
            ref = eia[k].get(row["date"])
            if ref is None or not np.isfinite(ref) or ref <= 0:
                continue
            rows.append({"date": row["date"], "k": k, "rel_diff": px / ref - 1.0,
                         "stamp_2300": bool(row["close_stamp_2300"])})
    if not rows:
        return {"n_compared": 0}
    df = pd.DataFrame(rows)
    a = df["rel_diff"].abs()
    out = {
        "n_compared": int(len(df)),
        "n_by_contract_number": {str(k): int(v) for k, v in df["k"].value_counts().sort_index().items()},
        "median_abs_rel_diff": r(float(a.median())),
        "share_within_0p25pct": r(float((a <= 0.0025).mean())),
        "share_within_1pct": r(float((a <= 0.01).mean())),
        "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
    }
    # agreement by 5-year period: the hourly-era data (pysystemtrade's own IB feed) may not be
    # exchange settlements, so agreement is expected to drop in recent years
    period = (df["date"].dt.year // 5) * 5
    within = a <= 0.01
    out["share_within_1pct_by_5y"] = {f"{int(p)}-{int(p) + 4}": r(float(v)) for p, v in within.groupby(period).mean().items()}
    out["n_by_5y"] = {f"{int(p)}-{int(p) + 4}": int(v) for p, v in within.groupby(period).size().items()}
    return out


# ----------------------------------------------------------------------------------------------
# 2. small-trader COT from the legacy and disaggregated history files
# ----------------------------------------------------------------------------------------------
def load_cot_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two bgithub1 COT files restricted to our 10 markets, all columns as text."""
    leg = pd.read_csv(BG / "cot_history.csv", dtype=str)
    leg.columns = [c.strip().lstrip("_") for c in leg.columns]
    leg["code"] = leg["CFTC_Contract_Market_Code"].str.strip()
    leg = leg[leg["code"].isin(COT_CODES)].copy()
    leg["date"] = pd.to_datetime(leg["As_of_Date_in_Form_YYYY_MM_DD"].str.strip())

    dis = pd.read_csv(BG / "cot_new_history.csv", dtype=str)
    dis.columns = [c.strip() for c in dis.columns]
    dis["code"] = dis["CFTC_Contract_Market_Code"].str.strip()
    dis = dis[dis["code"].isin(COT_CODES)].copy()
    # despite its name this column holds ISO dates (YYYY-MM-DD) in this file
    dis["date"] = pd.to_datetime(dis["As_of_Date_In_Form_YYMMDD"].str.strip(), format="%Y-%m-%d")
    return leg, dis


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col].str.strip(), errors="coerce")


def cot_checks() -> tuple[pd.DataFrame, dict]:
    leg, dis = load_cot_files()
    out_rows, stats = [], {}

    # tidy legacy table
    t = pd.DataFrame({
        "market": leg["code"].map(COT_CODES), "code": leg["code"], "report_date": leg["date"],
        "oi": num(leg, "Open_Interest_All"),
        "noncomm_long": num(leg, "Noncommercial_Positions_Long_All"),
        "noncomm_short": num(leg, "Noncommercial_Positions_Short_All"),
        "noncomm_spread": num(leg, "Noncommercial_Positions_Spreading_All"),
        "comm_long": num(leg, "Commercial_Positions_Long_All"),
        "comm_short": num(leg, "Commercial_Positions_Short_All"),
        "rept_long": num(leg, "Total_Reportable_Positions_Long_All"),
        "rept_short": num(leg, "Total_Reportable_Positions_Short_All"),
        "nonrept_long": num(leg, "Nonreportable_Positions_Long_All"),
        "nonrept_short": num(leg, "Nonreportable_Positions_Short_All"),
    }).sort_values(["market", "report_date"], kind="stable")
    # the 2006 block of the legacy file is present twice; drop exact duplicate rows and keep
    # count of what was removed (any remaining duplicate date would have conflicting values)
    n_before = t.groupby("market").size()
    t = t.drop_duplicates()
    n_exact_dups = (n_before - t.groupby("market").size()).to_dict()

    # disaggregated OI / non-reportables, for report-variant matching
    dd = pd.DataFrame({
        "market": dis["code"].map(COT_CODES), "report_date": dis["date"],
        "oi_dis": num(dis, "Open_Interest_All"),
        "nonrept_long_dis": num(dis, "NonRept_Positions_Long_All"),
        "nonrept_short_dis": num(dis, "NonRept_Positions_Short_All"),
    }).drop_duplicates()

    # known futures+options combined disaggregated OI (project file) for 5 markets
    fo = pd.read_csv(S / "data/cot_weekly.csv", usecols=["market", "report_date", "oi"])
    fo["report_date"] = pd.to_datetime(fo["report_date"])
    fo = fo.rename(columns={"oi": "oi_fo"})
    # known futures-only disaggregated crude OI (accomplish999 CFTC_CL, method says futures-only)
    with open(EIA / "CFTC_CL.json") as fh:
        cl = json.load(fh)["observations"]
    fut_cl = pd.DataFrame({"report_date": pd.to_datetime([o["date"] for o in cl]),
                           "oi_fut": pd.to_numeric([o["open_interest"] for o in cl], errors="coerce"),
                           "nonrept_long_fut": pd.to_numeric([o.get("nonrept_long") for o in cl], errors="coerce")})
    fut_cl["market"] = "crude"

    for mkt, g in t.groupby("market"):
        dup = int(g["report_date"].duplicated().sum())  # conflicting duplicates after dedupe
        dates = g["report_date"].drop_duplicates().reset_index(drop=True)
        gaps = dates.diff().dt.days
        # each gap listed as (last report before, first report after, days); 2007 is absent
        # from this file (the 2006 block appears twice instead)
        gap_list = [f"{dates[i - 1].date()}->{dates[i].date()} ({int(gaps[i])}d)" for i in gaps[gaps > 8].index]
        id1 = (g["oi"] == g["rept_long"] + g["nonrept_long"])
        id2 = (g["oi"] == g["rept_short"] + g["nonrept_short"])
        share = g["nonrept_long"] / g["oi"]
        m = g.merge(dd[dd["market"] == mkt], on=["market", "report_date"], how="inner")
        s = {
            "first_date": str(g["report_date"].min().date()), "last_date": str(g["report_date"].max().date()),
            "n_reports": int(len(g)), "n_exact_duplicate_rows_dropped": int(n_exact_dups[mkt]),
            "n_conflicting_duplicate_dates": dup,
            "n_gaps_over_8_days": int((gaps > 8).sum()),
            "gaps_over_8_days": gap_list,
            "share_identity_long_oi_eq_rept_plus_nonrept": r(float(id1.mean())),
            "share_identity_short_oi_eq_rept_plus_nonrept": r(float(id2.mean())),
            "nonrept_long_share_of_oi_median": r(float(share.median())),
            "n_overlap_with_disagg_file": int(len(m)),
            "share_oi_equal_legacy_vs_disagg_file": r(float((m["oi"] == m["oi_dis"]).mean())) if len(m) else None,
            "share_nonrept_long_equal_legacy_vs_disagg_file":
                r(float((m["nonrept_long"] == m["nonrept_long_dis"]).mean())) if len(m) else None,
        }
        # coverage of the disaggregated file itself (also carries NonRept columns)
        dm = dd[dd["market"] == mkt]["report_date"].drop_duplicates().sort_values().reset_index(drop=True)
        dg = dm.diff().dt.days
        s["disagg_file"] = {
            "first_date": str(dm.min().date()), "last_date": str(dm.max().date()), "n_reports": int(len(dm)),
            "gaps_over_8_days": [f"{dm[i - 1].date()}->{dm[i].date()} ({int(dg[i])}d)" for i in dg[dg > 8].index],
        }
        mf = m.merge(fo[fo["market"] == mkt], on=["market", "report_date"], how="inner")
        s["n_overlap_with_project_FO_file"] = int(len(mf))
        s["share_disagg_file_oi_equal_project_FO_oi"] = r(float((mf["oi_dis"] == mf["oi_fo"]).mean())) if len(mf) else None
        lf = g.merge(fo[fo["market"] == mkt], on=["market", "report_date"], how="inner")
        s["share_legacy_oi_equal_project_FO_oi"] = r(float((lf["oi"] == lf["oi_fo"]).mean())) if len(lf) else None
        if mkt == "crude":
            mc = g.merge(fut_cl, on=["market", "report_date"], how="inner")
            s["n_overlap_with_known_futures_only_crude"] = int(len(mc))
            s["share_legacy_oi_equal_futures_only_crude_oi"] = r(float((mc["oi"] == mc["oi_fut"]).mean()))
            s["share_legacy_nonrept_long_equal_futures_only_crude"] = r(float((mc["nonrept_long"] == mc["nonrept_long_fut"]).mean()))
        stats[mkt] = s
        out_rows.append(g)

    tidy = pd.concat(out_rows)
    return tidy, stats


def crude_futures_only_nonrept() -> dict:
    """Coverage of the crude futures-only non-reportable series already on disk
    (S/ext/accomplish999_crude-oil CFTC_CL.json, disaggregated futures-only)."""
    with open(EIA / "CFTC_CL.json") as fh:
        obs = json.load(fh)["observations"]
    df = pd.DataFrame({"date": pd.to_datetime([o["date"] for o in obs]),
                       "nl": pd.to_numeric([o.get("nonrept_long") for o in obs], errors="coerce"),
                       "ns": pd.to_numeric([o.get("nonrept_short") for o in obs], errors="coerce")})
    ok = df.dropna()
    gaps = ok["date"].sort_values().reset_index(drop=True)
    g = gaps.diff().dt.days
    return {"n_obs_with_nonrept": int(len(ok)), "first_date": str(ok["date"].min().date()),
            "last_date": str(ok["date"].max().date()),
            "gaps_over_8_days": [f"{gaps[i - 1].date()}->{gaps[i].date()} ({int(g[i])}d)" for i in g[g > 8].index]}


# ----------------------------------------------------------------------------------------------
# markdown tables
# ----------------------------------------------------------------------------------------------
def curve_table(curve_stats: dict, xchk: dict) -> str:
    """Two markdown tables (coverage, distinctness) plus the EIA cross-check table."""
    txt = ("Coverage (one row per weekday; NEAR/FAR both present):\n\n"
           "| market | file | first date (both legs) | last date | days both legs | weeks (W-FRI) | "
           "hourly rows from | weekend dates dropped | same-contract rows blanked | CARRY NaN days |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    for instr, s in curve_stats.items():
        txt += (f"| {s['market']} | {instr}.csv | {s['first_date_both_legs']} | {s['last_date']} | "
                f"{s['n_days_both_legs']} | {s['n_weeks_both_legs']} | {s['first_date_with_intraday_rows']} | "
                f"{s['n_weekend_dates_dropped']} | {s['n_same_contract_id_blanked']} | {s['n_carry_nan']} |\n")

    txt += ("\nDistinctness (NEAR = nearer of CARRY/PRICE by contract id; stale = one leg unchanged while "
            "the other moved, same-contract consecutive days):\n\n"
            "| market | near leg | far-near gap in months (days) | months to near delivery min/med/max | "
            "near==far share | corr daily dlog | near-premium share | stale near / far (all) | "
            "stale near / far (last 5y) |\n|---|---|---|---|---|---|---|---|---|\n")
    for instr, s in curve_stats.items():
        gap = ", ".join(f"{k}m: {v}" for k, v in s["month_gap_far_minus_near"].items())
        ci = s["share_carry_is_near"]
        near_leg = "CARRY" if ci == 1 else ("PRICE" if ci == 0 else f"CARRY ({ci} of rows)")
        mn = s["months_to_near_delivery"]
        txt += (f"| {s['market']} | {near_leg} | {gap} | {mn['min']}/{mn['median']}/{mn['max']} | "
                f"{s['share_days_near_equals_far']} | {s['corr_daily_logret_near_far']} | "
                f"{s['share_days_near_premium']} | "
                f"{s['share_near_unchanged_far_moved']} / {s['share_far_unchanged_near_moved']} | "
                f"{s['share_near_unchanged_far_moved_last5y']} / {s['share_far_unchanged_near_moved_last5y']} |\n")

    txt += ("\nEIA cross-check (weekly Wednesday sample; pysystemtrade leg price vs EIA NYMEX contract-k "
            "settlement on the same date; dates within 2 business days of an expiry excluded):\n\n"
            "| market | EIA series | n compared | by contract k | median abs rel diff | within 0.25% | "
            "within 1% | within 1% by 5y period (n) |\n|---|---|---|---|---|---|---|---|\n")
    for mkt, x in xchk.items():
        if x.get("n_compared", 0) == 0:
            txt += f"| {mkt} | {x.get('eia')} | 0 | | | | | |\n"
            continue
        byk = ", ".join(f"k{k}: {v}" for k, v in x["n_by_contract_number"].items())
        by5 = ", ".join(f"{p}: {v} ({x['n_by_5y'][p]})" for p, v in x["share_within_1pct_by_5y"].items())
        txt += (f"| {mkt} | {x['eia']} | {x['n_compared']} | {byk} | {x['median_abs_rel_diff']} | "
                f"{x['share_within_0p25pct']} | {x['share_within_1pct']} | {by5} |\n")
    return txt


def cot_table(cs: dict) -> str:
    hdr = ("| market | first | last | reports (deduped) | exact dup rows dropped | conflicting dup dates | gaps>8d | OI = rept+nonrept (long) | "
           "nonrept long / OI (median) | legacy OI = disagg-file OI | disagg-file OI = project F+O OI | "
           "legacy OI = project F+O OI |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for mkt, s in sorted(cs.items()):
        lines.append(
            f"| {mkt} | {s['first_date']} | {s['last_date']} | {s['n_reports']} | "
            f"{s['n_exact_duplicate_rows_dropped']} | {s['n_conflicting_duplicate_dates']} | "
            f"{s['n_gaps_over_8_days']}: {'; '.join(s['gaps_over_8_days'])} | {s['share_identity_long_oi_eq_rept_plus_nonrept']} | "
            f"{s['nonrept_long_share_of_oi_median']} | {s['share_oi_equal_legacy_vs_disagg_file']} | "
            f"{s['share_disagg_file_oi_equal_project_FO_oi']} | {s['share_legacy_oi_equal_project_FO_oi']} |")
    txt = hdr + "\n".join(lines) + "\n"
    d0 = cs[sorted(cs)[0]]["disagg_file"]
    txt += ("\nDisaggregated file `cot_new_history.csv` (has NonRept columns too), same for all 10 markets "
            f"(first market shown): {d0['first_date']} to {d0['last_date']}, {d0['n_reports']} reports, "
            f"gaps>8d: {'; '.join(d0['gaps_over_8_days']) or 'none'}. Per-market values in scout_summary.json.\n")
    c = cs.get("crude", {})
    if "n_overlap_with_known_futures_only_crude" in c:
        txt += (f"\nCrude check against a known FUTURES-ONLY disaggregated file (S/ext/accomplish999 CFTC_CL, "
                f"{c['n_overlap_with_known_futures_only_crude']} common dates): legacy OI equal in "
                f"{c['share_legacy_oi_equal_futures_only_crude_oi']} of dates, legacy non-reportable long equal in "
                f"{c['share_legacy_nonrept_long_equal_futures_only_crude']} of dates.\n")
    return txt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, curve_stats = [], {}
    for instr, mkt in CURVE_MARKETS.items():
        cur, st = build_curve(instr, mkt)
        curves.append(cur)
        curve_stats[instr] = st
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(OUT / "curve_daily.csv", index=False, date_format="%Y-%m-%d")

    xchk = {}
    crude = curve[curve["market"] == "crude"]
    xchk["crude"] = eia_crosscheck(crude, nymex_cl_last_trade, {1: "RCLC1", 2: "RCLC2", 3: "RCLC3", 4: "RCLC4"})
    xchk["crude"]["eia"] = "RCLC1-4"
    ho = curve[curve["market"] == "heatoil"]
    xchk["heatoil"] = eia_crosscheck(ho, nymex_ho_last_trade, {1: "HO_F1"})
    xchk["heatoil"]["eia"] = "HO_F1"

    cot_tidy, cot_stats = cot_checks()
    cot_tidy.to_csv(OUT / "cot_smalltrader_weekly.csv", index=False, date_format="%Y-%m-%d")

    summary = {"curve": curve_stats, "eia_crosscheck": xchk, "cot_smalltrader": cot_stats,
               "crude_futures_only_nonrept_on_disk": crude_futures_only_nonrept()}
    with open(OUT / "scout_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=r)
    with open(OUT / "inventory_tables.md", "w") as fh:
        fh.write("### Futures curve: NEAR vs FAR contract (pysystemtrade multiple prices)\n\n")
        fh.write(curve_table(curve_stats, xchk))
        fh.write("\n### Small traders: CFTC legacy COT history (bgithub1/cme_open_interest)\n\n")
        fh.write(cot_table(cot_stats))
        cf = summary["crude_futures_only_nonrept_on_disk"]
        fh.write(f"\nAlready on disk (S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json, crude only, "
                 f"disaggregated futures-only): nonrept_long/short on {cf['n_obs_with_nonrept']} weekly reports, "
                 f"{cf['first_date']} to {cf['last_date']}; gaps>8d: "
                 f"{'; '.join(cf['gaps_over_8_days']) or 'none'}.\n")
    print(json.dumps(summary, indent=1, default=r))


if __name__ == "__main__":
    main()

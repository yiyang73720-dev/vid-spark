#!/usr/bin/env python3
"""
Larry Williams COT-index backtest -- canonical implementation A (spec v1).

Vectorized pandas approach: entry dates are resolved with `merge_asof(direction="forward")`
against each market's cleaned trading-day calendar, and exits are looked up by integer
trading-day position (entry position + 5*h) on that same calendar.

Usage:
    python3 -I backtest_canonical.py [ROOT]     (default: $LARRY_COT_ROOT or package dir)

Inputs  (read only):  <S>/data/cot_weekly.csv, <S>/data/prices_daily.csv
Outputs (written):    <S>/work/implA/canonical.csv          (360 rows)
                      <S>/work/implA/crude_timeseries.csv   (one row per crude report)
                      <S>/work/implA/pooled_headline.csv    (N=26, 80/20 pooled across 5 markets)

Every resolved ambiguity of the spec is marked with "SPEC-CHOICE" in the comments below.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Parameters (all straight from the spec)
# ----------------------------------------------------------------------------------------
MARKETS = ["crude", "gold", "silver", "soybean", "sugar"]   # us10y / us30y ignored
LOOKBACKS = [26, 52, 156]
THRESHOLDS = [(80, 20), (75, 25), (90, 10)]                   # (buy_th, sell_th); spec order
SIDES = ["long", "short"]
HORIZONS_W = [1, 4, 8, 13]
TRADING_DAYS_PER_WEEK = 5
K_PERM = 2000
SEED = 11
MIN_SHIFT = 26                                                # offsets uniform in [26, T-26]

# Report dates inside these windows are not tradable in real time (publication delayed).
SHUTDOWN_WINDOWS = [
    ("2013-10-01", "2013-10-31"),
    ("2018-12-18", "2019-02-26"),
    ("2025-09-30", "2026-01-20"),
]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def larry_index(s: pd.Series, n: int) -> pd.Series:
    """Larry Williams COT index over the last n reports *including* the current one.

    idx = 100 * (x - min_n) / (max_n - min_n); NaN for the first n-1 reports
    (min_periods=n) and NaN when max_n == min_n.
    """
    lo = s.rolling(n, min_periods=n).min()
    hi = s.rolling(n, min_periods=n).max()
    rng = hi - lo
    return 100.0 * (s - lo) / rng.where(rng != 0)


def load_cot(path: Path) -> pd.DataFrame:
    cot = pd.read_csv(path, dtype={"code": str})
    cot = cot[cot["market"].isin(MARKETS)].copy()
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    cot = cot.sort_values(["market", "report_date"]).reset_index(drop=True)
    # Commercials = Producer/Merchant + Swap Dealers (Larry's "insiders").
    cot["comm_net"] = (cot["prod_merc_long"] + cot["swap_long"]) - (
        cot["prod_merc_short"] + cot["swap_short"])
    # Managed money net (speculator comparator).
    cot["mm_net"] = cot["m_money_long"] - cot["m_money_short"]
    g_comm = cot.groupby("market", sort=False)["comm_net"]
    for n in LOOKBACKS:
        cot[f"idx{n}"] = g_comm.transform(lambda s, n=n: larry_index(s, n))
    cot["mm_idx26"] = cot.groupby("market", sort=False)["mm_net"].transform(
        lambda s: larry_index(s, 26))
    # Shutdown flag (inclusive windows).
    shut = np.zeros(len(cot), dtype=bool)
    for a, b in SHUTDOWN_WINDOWS:
        shut |= cot["report_date"].between(pd.Timestamp(a), pd.Timestamp(b)).to_numpy()
    cot["shutdown"] = shut
    # No-look-ahead entry target: Tuesday of the week following report_date.
    # report_date - weekday(report_date) + 1 + 7 days   (weekday: Mon=0)
    wd = cot["report_date"].dt.weekday
    cot["target_date"] = cot["report_date"] - pd.to_timedelta(wd, unit="D") + pd.Timedelta(days=8)
    return cot


def load_prices(path: Path) -> pd.DataFrame:
    px = pd.read_csv(path)
    px = px[px["market"].isin(MARKETS)].copy()
    px["date"] = pd.to_datetime(px["date"])
    # Spec: drop any close <= 0 (crude 2020-04-20 = -37.63) before computing anything.
    px = px[px["close"].notna() & (px["close"] > 0)]
    px = px.sort_values(["market", "date"]).reset_index(drop=True)
    # Trading-day position within each market's cleaned series (0, 1, 2, ...).
    px["tpos"] = px.groupby("market", sort=False).cumcount()
    return px[["market", "date", "close", "tpos"]]


def attach_entries_and_returns(cot: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Resolve entry date/price (first trading day on/after target) and h-week forward returns."""
    left = cot.sort_values("target_date")
    right = px.rename(columns={"date": "entry_date", "close": "entry_price",
                               "tpos": "entry_tpos"}).sort_values("entry_date")
    # merge_asof forward: first price row with entry_date >= target_date, within the same market.
    m = pd.merge_asof(left, right, left_on="target_date", right_on="entry_date",
                      by="market", direction="forward", allow_exact_matches=True)
    m = m.sort_values(["market", "report_date"]).reset_index(drop=True)

    # Exit lookup by (market, trading-day position) on the cleaned calendar.
    close_by_pos = px.set_index(["market", "tpos"])["close"]
    for h in HORIZONS_W:
        exit_pos = m["entry_tpos"] + TRADING_DAYS_PER_WEEK * h        # NaN if no entry
        key = pd.MultiIndex.from_arrays([m["market"], exit_pos.fillna(-1).astype("int64")])
        exit_px = close_by_pos.reindex(key).to_numpy()                  # NaN if beyond data
        m[f"r{h}"] = exit_px / m["entry_price"].to_numpy() - 1.0
    return m


def run_ids(active: np.ndarray) -> np.ndarray:
    """Label maximal runs of consecutive True values (1, 2, ...); 0 where inactive."""
    start = active & ~np.concatenate(([False], active[:-1]))
    ids = np.cumsum(start)
    return np.where(active, ids, 0)


def perm_pvalue(sig: np.ndarray, signed_r: np.ndarray, observed: float) -> float:
    """One-sided circular-shift permutation p-value.

    sig, signed_r are aligned on the T eligible weeks (chronological order).
    SPEC-CHOICE: a fresh np.random.default_rng(11) per cell (order-independent); offsets
    k ~ integers in [26, T-26] inclusive; shift convention = np.roll(sig, k), i.e. the
    signal that was at eligible week i is moved to eligible week (i + k) mod T.
    """
    T = len(sig)
    pos = np.flatnonzero(sig)
    if len(pos) == 0:
        return np.nan
    rng = np.random.default_rng(SEED)
    offs = rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=K_PERM, endpoint=True)
    shifted_pos = (pos[None, :] + offs[:, None]) % T            # K x n positions
    shifted_means = signed_r[shifted_pos].mean(axis=1)
    return (1 + int(np.sum(shifted_means >= observed))) / (1 + K_PERM)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main(S: str) -> None:
    S = Path(S)
    out_dir = S / "work" / "implA"
    out_dir.mkdir(parents=True, exist_ok=True)

    cot = load_cot(S / "data" / "cot_weekly.csv")
    px = load_prices(S / "data" / "prices_daily.csv")
    m = attach_entries_and_returns(cot, px)

    rows = []
    pooled_parts = []                                           # for the pooled headline note
    for mkt in MARKETS:
        g = m[m["market"] == mkt].sort_values("report_date").reset_index(drop=True)
        for n in LOOKBACKS:
            idx = g[f"idx{n}"].to_numpy()
            idx_ok = ~np.isnan(idx)
            for buy_th, sell_th in THRESHOLDS:
                # Setups on the FULL report sequence (NaN idx => no setup).
                setups = {"long": idx_ok & (idx >= buy_th),
                          "short": idx_ok & (idx <= sell_th)}
                # SPEC-CHOICE: episodes = maximal runs of consecutive report weeks with the
                # setup active, built on the full report sequence (shutdown weeks and weeks
                # without a forward return do NOT break a run).
                episodes = {sd: run_ids(setups[sd]) for sd in SIDES}
                for sd in SIDES:
                    sgn = 1.0 if sd == "long" else -1.0
                    for h in HORIZONS_W:
                        r = g[f"r{h}"].to_numpy()
                        # Eligible weeks: idx defined, outside shutdown windows, valid fwd return.
                        elig = idx_ok & ~g["shutdown"].to_numpy() & ~np.isnan(r)
                        r_e = r[elig]
                        baseline = r_e.mean()
                        sig_e = setups[sd][elig]                  # signal on eligible weeks
                        signed_e = sgn * r_e
                        sr = signed_e[sig_e]
                        n_sig = int(sig_e.sum())
                        # SPEC-CHOICE: n_episodes = distinct episodes with >=1 counted week.
                        n_epi = int(len(np.unique(episodes[sd][elig][sig_e])))
                        if n_sig > 0:
                            mean_ret = float(sr.mean())
                            median_ret = float(np.median(sr))
                            hit = float((sr > 0).mean())
                            excess = mean_ret - sgn * baseline
                            p = perm_pvalue(sig_e, signed_e, mean_ret)
                        else:
                            mean_ret = median_ret = hit = excess = p = np.nan
                        rows.append(dict(market=mkt, lookback=n, buy_th=buy_th, sell_th=sell_th,
                                         side=sd, horizon_w=h, n_signal_weeks=n_sig,
                                         n_episodes=n_epi, mean_ret=mean_ret,
                                         median_ret=median_ret, hit_rate=hit,
                                         baseline_mean=float(baseline), excess=excess,
                                         perm_p=p))
                        if n == 26 and (buy_th, sell_th) == (80, 20):
                            pooled_parts.append(dict(market=mkt, side=sd, h=h, sig=sig_e,
                                                     signed=signed_e, baseline=baseline,
                                                     n_epi=n_epi))

    canon = pd.DataFrame(rows)
    assert len(canon) == 360, len(canon)
    canon.to_csv(out_dir / "canonical.csv", index=False)

    # ---- crude time series --------------------------------------------------------------
    c = m[m["market"] == "crude"].sort_values("report_date")
    ts = pd.DataFrame({
        "report_date": c["report_date"].dt.strftime("%Y-%m-%d"),
        "entry_date": c["entry_date"].dt.strftime("%Y-%m-%d"),   # NaT -> empty
        "comm_net": c["comm_net"].round().astype("int64"),
        "mm_net": c["mm_net"].round().astype("int64"),
        "oi": c["oi"].round().astype("int64"),
        "idx26": c["idx26"], "idx52": c["idx52"], "idx156": c["idx156"],
        "mm_idx26": c["mm_idx26"],
        "entry_price": c["entry_price"],
    })
    ts.to_csv(out_dir / "crude_timeseries.csv", index=False)

    # ---- pooled headline (N=26, 80/20) across 5 markets ---------------------------------
    # Not part of the canonical file; reported as a note.
    # Pooled mean = mean over all signal-week signed returns (each market-week weighted equally);
    # pooled excess = mean of (signed r - signed own-market baseline) over pooled signal weeks;
    # pooled perm p: each permutation draws one independent offset per market (rng seed 11,
    # markets in fixed order) and recomputes the pooled mean signed return.
    prow = []
    for sd in SIDES:
        for h in HORIZONS_W:
            parts = [p for p in pooled_parts if p["side"] == sd and p["h"] == h]
            sgn = 1.0 if sd == "long" else -1.0
            all_sr = np.concatenate([p["signed"][p["sig"]] for p in parts])
            all_ex = np.concatenate([p["signed"][p["sig"]] - sgn * p["baseline"] for p in parts])
            n_tot = len(all_sr)
            obs = all_sr.mean()
            rng = np.random.default_rng(SEED)
            sums = np.zeros(K_PERM)
            for p in parts:
                T = len(p["sig"])
                pos = np.flatnonzero(p["sig"])
                offs = rng.integers(MIN_SHIFT, T - MIN_SHIFT, size=K_PERM, endpoint=True)
                sums += p["signed"][(pos[None, :] + offs[:, None]) % T].sum(axis=1)
            pooled_p = (1 + int(np.sum(sums / n_tot >= obs))) / (1 + K_PERM)
            mkt_ex = [float(p["signed"][p["sig"]].mean() - sgn * p["baseline"])
                      for p in parts if p["sig"].any()]
            prow.append(dict(side=sd, horizon_w=h, n_signal_weeks=n_tot,
                             n_episodes=int(sum(p["n_epi"] for p in parts)),
                             mean_ret=float(obs), median_ret=float(np.median(all_sr)),
                             hit_rate=float((all_sr > 0).mean()),
                             excess_vs_own_baseline=float(all_ex.mean()),
                             equal_weight_mkt_excess=float(np.mean(mkt_ex)),
                             n_mkts_positive_excess=int(sum(e > 0 for e in mkt_ex)),
                             perm_p=pooled_p))
    pd.DataFrame(prow).to_csv(out_dir / "pooled_headline.csv", index=False)

    # ---- self sanity checks ---------------------------------------------------------------
    for col in ["idx26", "idx52", "idx156", "mm_idx26"]:
        v = m[col].dropna()
        assert (v >= 0).all() and (v <= 100).all(), col
    has_entry = m["entry_date"].notna()
    assert (m.loc[has_entry, "entry_date"] > m.loc[has_entry, "report_date"]
            + pd.Timedelta(days=5)).all()
    assert (m.loc[has_entry, "entry_date"] >= m.loc[has_entry, "target_date"]).all()
    nb = canon.groupby(["market", "lookback", "horizon_w"])["baseline_mean"].nunique()
    assert (nb == 1).all(), "baseline must not depend on side/threshold"
    print("canonical rows:", len(canon), "| crude ts rows:", len(ts))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[2])

#!/usr/bin/env python3
"""
verify_primary -- independent re-derivation of the SEASONALITY primary rule (adversarial check).

This file shares NO signal / statistics code with seasonal.py or run_seasonality.py. It only uses
fund_harness.load_series() to read the cleaned price series (data loading), and re-implements:

  * forward returns: entry = first close strictly after the decision Friday (at most 7 calendar
    days after Saturday), exit = the close 40 rows (8 weeks x 5 trading days) after entry;
  * analogs: for decision Friday d and k = 1..L, the Friday nearest to (d - k years), using
    pandas DateOffset (Feb 29 -> Feb 28); its 8-week return on the same series;
  * signal: long if #up >= 0.7 L and mean > 0; short if #up <= 0.3 L and mean < 0
    (undefined unless all L analog returns exist);
  * pooled NET statistics (mean, excess over each market's own baseline, hit, episodes);
  * a Monte-Carlo circular-shift p with its OWN random generator (seed 2024), plus a variant
    where every market is shifted by the SAME calendar offset (keeps cross-market correlation);
  * sub-sample diagnostics (drop one market, drop one year, drop the largest episodes).

Run:  python3 -I verify_primary.py      (writes ROOT/work2/tool_seasonality/verify/verify_primary.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))   # battery/harness
import fund_harness as fh  # noqa: E402  (used ONLY for load_series)

HERE = fh.WORK2 / "tool_seasonality" / "verify"     # outputs: ROOT/work2/tool_seasonality/verify
H_W = 8                      # horizon in weeks
N_ROWS = 5 * H_W             # 40 trading rows
TH = 0.70
MARKETS = [("crude", "crude_spot", 15), ("heatoil", "ho_spot", 15), ("gasoline", "gas_spot", 15),
           ("gold", "gold", 10), ("silver", "silver", 10), ("soybean", "soybean", 10),
           ("sugar", "sugar", 10)]
GRID = pd.date_range("1986-01-03", "2026-10-16", freq="W-FRI")          # decision Fridays
GRID_EXT = pd.date_range("1965-01-01", "2026-10-16", freq="W-FRI")      # analog Fridays (estimation)


def forward(series: pd.Series, fridays: pd.DatetimeIndex) -> pd.DataFrame:
    """Entry/exit dates and 8-week return for each Friday (merge_asof, forward direction)."""
    px = pd.DataFrame({"date": series.index, "close": series.to_numpy(), "row": np.arange(len(series))})
    q = pd.DataFrame({"friday": fridays, "target": fridays + pd.Timedelta(days=1)})
    m = pd.merge_asof(q, px, left_on="target", right_on="date", direction="forward",
                      tolerance=pd.Timedelta(days=7))
    out = pd.DataFrame(index=fridays)
    row = m["row"].to_numpy()
    ok = ~np.isnan(row)
    row_i = np.where(ok, row, 0).astype(int)
    xr = row_i + N_ROWS
    ok2 = ok & (xr < len(series))
    c = series.to_numpy()
    r = np.full(len(fridays), np.nan)
    r[ok2] = c[xr[ok2]] / c[row_i[ok2]] - 1.0
    ex = np.full(len(fridays), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ex[ok2] = series.index.values[xr[ok2]]
    out["r"] = r
    out["exit"] = ex
    return out


def nearest_fri(ts: pd.Timestamp) -> pd.Timestamp:
    """Nearest Friday (within +-3 days) via ((4 - weekday + 3) mod 7) - 3."""
    return ts + pd.Timedelta(days=((4 - ts.weekday() + 3) % 7) - 3)


def seasonal_signal(fw: pd.DataFrame, L: int) -> pd.DataFrame:
    """Per decision Friday: #up, mean, latest analog exit and the +1/0/-1 signal (NaN if undefined)."""
    rmap = fw["r"].to_dict()
    emap = fw["exit"].to_dict()
    rows = []
    for d in GRID:
        rs, ex = [], []
        for k in range(1, L + 1):
            a = nearest_fri(d - pd.DateOffset(years=k))
            rs.append(rmap.get(a, np.nan))
            ex.append(emap.get(a, pd.NaT))
        rs = np.array(rs, float)
        if np.isnan(rs).any():
            rows.append((d, np.nan, np.nan, pd.NaT, np.nan))
            continue
        nup, mu = int((rs > 0).sum()), float(rs.mean())
        sig = 1.0 if (nup >= TH * L and mu > 0) else (-1.0 if (nup <= (1 - TH) * L + 1e-9 and mu < 0) else 0.0)
        rows.append((d, nup, mu, max(ex), sig))
    return pd.DataFrame(rows, columns=["d", "nup", "mean", "last_exit", "sig"]).set_index("d")


def market_frame(name: str, L: int) -> pd.DataFrame:
    s = fh.load_series(name)
    sg = seasonal_signal(forward(s, GRID_EXT), L)     # analogs may predate the decision grid
    df = sg.join(forward(s, GRID))
    # leak check: the latest analog exit must be strictly before the decision date
    lk = df.dropna(subset=["sig"])
    gap_days = (lk.index - pd.DatetimeIndex(lk["last_exit"])).days
    df.attrs["min_gap_days"] = int(gap_days.min())
    elig = df["sig"].notna() & df["r"].notna()
    return df[elig].copy()


def episodes(sig: np.ndarray) -> int:
    prev = np.concatenate(([0.0], sig[:-1]))
    return int(((sig != 0) & (sig != prev)).sum())


def pooled(frames: dict, keys=None, drop_years=()) -> dict:
    keys = keys or list(frames)
    sr, ex, n_ep = [], [], 0
    for k in keys:
        f = frames[k]
        f = f[~f.index.year.isin(drop_years)]
        b = f["r"].mean()
        s = f["sig"].to_numpy()
        r = f["r"].to_numpy()
        m = s != 0
        sr.append(s[m] * r[m])
        ex.append(s[m] * (r[m] - b))
        n_ep += episodes(s)
    sr, ex = np.concatenate(sr), np.concatenate(ex)
    return dict(n=int(len(sr)), episodes=n_ep, mean=float(sr.mean()), median=float(np.median(sr)),
                excess=float(ex.mean()), hit=float((sr > 0).mean()))


def perm_independent(frames: dict, K=2000, seed=2024, keys=None) -> float:
    """Each market shifted by its own random offset in [26, T-26] (the pre-registered scheme)."""
    keys = keys or list(frames)
    rng = np.random.default_rng(seed)
    obs_parts, sums, n = [], np.zeros(K), 0
    for k in keys:
        f = frames[k]
        s, r = f["sig"].to_numpy(), f["r"].to_numpy()
        T = len(s)
        pos = np.flatnonzero(s != 0)
        n += len(pos)
        obs_parts.append((s[pos] * r[pos]).sum())
        offs = rng.integers(26, T - 26 + 1, size=K)
        sums += (s[pos][None, :] * r[(pos[None, :] + offs[:, None]) % T]).sum(1)
    obs = sum(obs_parts) / n
    return (1 + int((sums / n >= obs).sum())) / (1 + K)


def perm_common(frames: dict, K=2000, seed=2024) -> float:
    """Every market's signal shifted by the SAME number of weeks on the common Friday grid, each
    circularly over its own eligible window: keeps the cross-market alignment of signals."""
    rng = np.random.default_rng(seed)
    Tmin = min(len(f) for f in frames.values())
    offs = rng.integers(26, Tmin - 26 + 1, size=K)
    sums, n, obs = np.zeros(K), 0, 0.0
    for f in frames.values():
        s, r = f["sig"].to_numpy(), f["r"].to_numpy()
        T = len(s)
        pos = np.flatnonzero(s != 0)
        n += len(pos)
        obs += (s[pos] * r[pos]).sum()
        sums += (s[pos][None, :] * r[(pos[None, :] + offs[:, None]) % T]).sum(1)
    return (1 + int((sums / n >= obs / n).sum())) / (1 + K)


def main():
    frames, gaps = {}, {}
    for key, name, L in MARKETS:
        f = market_frame(name, L)
        gaps[key] = f.attrs.get("min_gap_days")
        frames[key] = f
    out = dict(min_days_decision_minus_last_analog_exit=gaps)
    P = pooled(frames)
    P["n_long"] = int(sum((f["sig"] > 0).sum() for f in frames.values()))
    P["n_short"] = int(sum((f["sig"] < 0).sum() for f in frames.values()))
    P["T"] = int(sum(len(f) for f in frames.values()))
    P["period"] = f"{min(f.index.min() for f in frames.values()).date()}..{max(f.index.max() for f in frames.values()).date()}"
    P["perm_p_independent_seed2024"] = perm_independent(frames)
    P["perm_p_common_offset_seed2024"] = perm_common(frames)
    out["primary_pooled_net_h8"] = P
    # per market
    pm = {}
    for k, f in frames.items():
        q = pooled({k: f})
        q["perm_p_seed2024"] = perm_independent({k: f})
        q["baseline"] = float(f["r"].mean())
        pm[k] = q
    out["per_market"] = pm
    # leave-one-market-out, and the pool without the two artifact-prone v1/spot markets + gasoline
    out["leave_one_out"] = {k: dict(**pooled(frames, [x for x in frames if x != k]),
                                    perm_p=perm_independent(frames, keys=[x for x in frames if x != k]))
                            for k in frames}
    keep = ["crude", "heatoil", "gold", "silver"]
    out["without_gasoline_sugar_soybean"] = dict(**pooled(frames, keep),
                                                 perm_p=perm_independent(frames, keys=keep))
    # drop single crisis years
    out["drop_years"] = {str(y): pooled(frames, drop_years=(y,)) for y in (2008, 2014, 2015, 2020, 2022)}
    out["drop_2008_2014_2020"] = pooled(frames, drop_years=(2008, 2014, 2020))
    # by year, with each market's baseline taken WITHIN that year (results.json's by-year table
    # uses full-sample baselines instead, so the two differ)
    yr = {}
    for y in range(2000, 2027):
        try:
            yr[y] = round(pooled(frames, drop_years=tuple(x for x in range(1980, 2030) if x != y))["excess"], 5)
        except ValueError:
            pass
    out["excess_by_year_within_year_baseline"] = yr
    # largest episodes: drop the top-N episodes by summed excess contribution
    rows = []
    for k, f in frames.items():
        s = f["sig"].to_numpy()
        prev = np.concatenate(([0.0], s[:-1]))
        eid = np.cumsum((s != 0) & (s != prev))
        b = f["r"].mean()
        m = s != 0
        for e in np.unique(eid[m]):
            sel = m & (eid == e)
            rows.append(dict(market=k, start=str(f.index[sel][0].date()), weeks=int(sel.sum()),
                             side=int(s[sel][0]), excess_sum=float((s[sel] * (f["r"].to_numpy()[sel] - b)).sum())))
    ep = pd.DataFrame(rows).sort_values("excess_sum", ascending=False)
    tot_n = P["n"]
    tot = ep["excess_sum"].sum()
    out["top_episodes"] = ep.head(10).to_dict("records")
    out["excess_without_top_episodes"] = {str(N): float((tot - ep["excess_sum"].iloc[:N].sum()) /
                                                        (tot_n - ep["weeks"].iloc[:N].sum()))
                                          for N in (1, 5, 10, 20)}
    # symmetric trim: drop the N best AND the N worst episodes (dropping only winners is biased)
    sym = {}
    for N in (5, 10, 20):
        keep = ep.iloc[N:len(ep) - N]
        sym[str(N)] = float(keep["excess_sum"].sum() / keep["weeks"].sum())
    out["excess_symmetric_trim_episodes"] = sym
    out["n_episodes_eligible_only"] = int(len(ep))
    # pooled halves at the author's common split date, with and without gasoline spot
    split = pd.Timestamp("2013-10-21")
    halves = {}
    for lab, keys in (("all", list(frames)), ("without_gasoline", [k for k in frames if k != "gasoline"])):
        for h, early in (("H1", True), ("H2", False)):
            fr = {k: frames[k][(frames[k].index <= split) == early] for k in keys}
            fr = {k: f for k, f in fr.items() if len(f) > 52}
            halves[f"{lab}_{h}"] = dict(**pooled(fr), perm_p=perm_independent(fr))
    out["split_halves_2013_10_21"] = halves
    out["share_of_total_excess_from_gasoline"] = float(ep.loc[ep.market == "gasoline", "excess_sum"].sum() / tot)
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "verify_primary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

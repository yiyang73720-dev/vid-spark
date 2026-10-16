#!/usr/bin/env python3
"""
Larry Williams COT-index backtest -- independent implementation B of SPEC.md (v1).

Style: explicit loops over report weeks, plain Python lists/dicts for all date
lookups (bisect on a sorted list of trading dates); no pandas merge_asof.

Usage:
    python3 -I backtest_b.py [ROOT]     (default: $LARRY_COT_ROOT or package dir)

Inputs (read-only):
    S/data/cot_weekly.csv     CFTC disaggregated futures+options COT, weekly
    S/data/prices_daily.csv   daily closes per market

Outputs (written to ROOT/work/implB/):
    canonical.csv             360 rows: market x lookback x threshold x side x horizon
    crude_timeseries.csv      one row per crude report
    pooled.csv                (extra) same cells pooled across the 5 markets
    diagnostics.txt           (extra) sanity checks and tie/boundary counts
"""

import bisect
import csv
import datetime as dt
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Configuration (all taken verbatim from SPEC.md)
# ----------------------------------------------------------------------------
S = sys.argv[1] if len(sys.argv) > 1 else \
    (os.environ.get("LARRY_COT_ROOT")
     or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = os.path.join(S, "work", "implB")
os.makedirs(OUT_DIR, exist_ok=True)

MARKETS = ["crude", "gold", "silver", "soybean", "sugar"]     # us10y/us30y ignored
LOOKBACKS = [26, 52, 156]
THRESHOLDS = [(80, 20), (75, 25), (90, 10)]                    # (buy_th, sell_th)
SIDES = ["long", "short"]
HORIZONS = [1, 4, 8, 13]                                       # weeks; exit = 5*h trading days
K_PERM = 2000
SEED = 22
MIN_SHIFT = 26                                                 # offsets uniform in [26, T-26]

# Government-shutdown publication delays (inclusive windows): signals excluded.
SHUTDOWNS = [
    (dt.date(2013, 10, 1), dt.date(2013, 10, 31)),
    (dt.date(2018, 12, 18), dt.date(2019, 2, 26)),
    (dt.date(2025, 9, 30), dt.date(2026, 1, 20)),
]


def parse_date(s):
    return dt.date.fromisoformat(s.strip()[:10])


def in_shutdown(d):
    for lo, hi in SHUTDOWNS:
        if lo <= d <= hi:
            return True
    return False


def as_int(x):
    """COT position fields are stored as e.g. '42587.0'; they are integer counts."""
    f = float(x)
    i = int(round(f))
    assert abs(f - i) < 1e-9, f"non-integer position value {x!r}"
    return i


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_cot(path):
    """Return {market: [row dicts sorted by report_date]} for the 5 markets."""
    by_mkt = {m: [] for m in MARKETS}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = row["market"]
            if m not in by_mkt:
                continue
            rd = parse_date(row["report_date"])
            pm_l, pm_s = as_int(row["prod_merc_long"]), as_int(row["prod_merc_short"])
            sw_l, sw_s = as_int(row["swap_long"]), as_int(row["swap_short"])
            mm_l, mm_s = as_int(row["m_money_long"]), as_int(row["m_money_short"])
            by_mkt[m].append({
                "report_date": rd,
                "oi": as_int(row["oi"]),
                # Commercials = Producer/Merchant + Swap Dealers
                "comm_net": (pm_l + sw_l) - (pm_s + sw_s),
                # Managed money (speculator comparator)
                "mm_net": mm_l - mm_s,
            })
    for m in MARKETS:
        by_mkt[m].sort(key=lambda r: r["report_date"])
        dates = [r["report_date"] for r in by_mkt[m]]
        assert len(dates) == len(set(dates)), f"duplicate report dates in {m}"
    return by_mkt


def load_prices(path):
    """Return {market: (dates_list, closes_list)}; closes <= 0 or missing are dropped.
    The remaining rows define that market's trading-day calendar."""
    raw = {m: {} for m in MARKETS}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = row["market"]
            if m not in raw:
                continue
            c = row["close"].strip()
            if c == "":
                continue
            c = float(c)
            if not math.isfinite(c) or c <= 0:
                continue                      # e.g. crude 2020-04-20 = -37.63
            d = parse_date(row["date"])
            assert d not in raw[m], f"duplicate price date {m} {d}"
            raw[m][d] = c
    out = {}
    for m in MARKETS:
        ds = sorted(raw[m])
        out[m] = (ds, [raw[m][d] for d in ds])
    return out


# ----------------------------------------------------------------------------
# Larry COT index (explicit loop, window = last N reports incl. current)
# ----------------------------------------------------------------------------
def rolling_index(values, n):
    """For each position i return (idx_float, num, den) where
    idx = 100*(v - min_N)/(max_N - min_N), num = v - min_N, den = max_N - min_N.
    Returns (None, None, None) for the first n-1 reports or when den == 0."""
    res = []
    for i in range(len(values)):
        if i < n - 1:
            res.append((None, None, None))
            continue
        window = values[i - n + 1: i + 1]
        lo, hi = min(window), max(window)
        den = hi - lo
        if den == 0:
            res.append((None, None, None))
            continue
        num = values[i] - lo
        res.append((100.0 * num / den, num, den))
    return res


def setup_flags(idx_triplets, buy_th, sell_th):
    """Exact integer comparison of idx >= buy_th / idx <= sell_th, i.e.
    100*num >= buy_th*den (den > 0); avoids float rounding at the boundary."""
    longs, shorts = [], []
    for idx, num, den in idx_triplets:
        if idx is None:
            longs.append(False)
            shorts.append(False)
        else:
            longs.append(100 * num >= buy_th * den)
            shorts.append(100 * num <= sell_th * den)
    return longs, shorts


def episode_ids(flags):
    """Label maximal runs of consecutive True report weeks (over the full report
    sequence). Returns list with an int id for True weeks, None otherwise."""
    ids, cur, prev = [], 0, False
    for f in flags:
        if f and not prev:
            cur += 1
        ids.append(cur if f else None)
        prev = f
    return ids


# ----------------------------------------------------------------------------
# Timing: entry / exit
# ----------------------------------------------------------------------------
def entry_target(report_date):
    """Tuesday of the week following report_date:
    report_date - weekday(report_date) + 1 + 7 days (Mon=0)."""
    return report_date - dt.timedelta(days=report_date.weekday()) + dt.timedelta(days=8)


def entry_position(price_dates, report_date):
    """Index into the cleaned price series of the first trading day on/after the
    target Tuesday, or None if beyond the end of the data."""
    pos = bisect.bisect_left(price_dates, entry_target(report_date))
    return pos if pos < len(price_dates) else None


# ----------------------------------------------------------------------------
# Permutation test (circular shift of the weekly signal over eligible weeks)
# ----------------------------------------------------------------------------
def perm_pvalue(sig, sret, observed, offsets):
    """sig: bool array over the T eligible weeks; sret: signed returns (same length).
    Shifted signal = np.roll(sig, k); stat = mean of sret over shifted-signal weeks.
    p = (1 + #{stat_k >= observed}) / (1 + K)."""
    ge = 0
    for k in offsets:
        sh = np.roll(sig, int(k))
        if sret[sh].mean() >= observed:
            ge += 1
    return (1 + ge) / (1 + len(offsets))


def draw_offsets(T):
    """Fresh default_rng(SEED) per cell (order-independent, reproducible).
    Offsets uniform on the integers [26, T-26] inclusive."""
    rng = np.random.default_rng(SEED)
    return rng.integers(MIN_SHIFT, T - MIN_SHIFT, endpoint=True, size=K_PERM)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def fmt(x):
    """CSV float formatting: shortest round-trip repr; NaN/None -> empty."""
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    x = float(x)
    if not math.isfinite(x):
        return ""
    return repr(x)


def main():
    cot = load_cot(f"{S}/data/cot_weekly.csv")
    prices = load_prices(f"{S}/data/prices_daily.csv")
    diag = []

    # Per-market, per-report precomputation ------------------------------------
    # mk[m] holds lists aligned with the report sequence.
    mk = {}
    for m in MARKETS:
        rows = cot[m]
        pdates, pclose = prices[m]
        comm = [r["comm_net"] for r in rows]
        idx = {n: rolling_index(comm, n) for n in LOOKBACKS}
        entry_pos, entry_date, entry_px, shut = [], [], [], []
        fwd = {h: [] for h in HORIZONS}            # raw forward return r or None
        for r in rows:
            rd = r["report_date"]
            p = entry_position(pdates, rd)
            entry_pos.append(p)
            entry_date.append(pdates[p] if p is not None else None)
            entry_px.append(pclose[p] if p is not None else None)
            shut.append(in_shutdown(rd))
            for h in HORIZONS:
                if p is None or p + 5 * h >= len(pdates):
                    fwd[h].append(None)
                else:
                    fwd[h].append(pclose[p + 5 * h] / pclose[p] - 1.0)
        mk[m] = dict(rows=rows, comm=comm, idx=idx, entry_date=entry_date,
                     entry_px=entry_px, shut=shut, fwd=fwd)

        # sanity: entry strictly later than report_date + 5 days
        for r, ed in zip(rows, entry_date):
            if ed is not None:
                assert ed > r["report_date"] + dt.timedelta(days=5), (m, r["report_date"], ed)
        # boundary diagnostics: weeks where exact vs float comparison would differ
        for n in LOOKBACKS:
            for bt, st in THRESHOLDS:
                for idxf, num, den in idx[n]:
                    if idxf is None:
                        continue
                    if (idxf >= bt) != (100 * num >= bt * den) or \
                       (idxf <= st) != (100 * num <= st * den):
                        diag.append(f"float/exact boundary mismatch {m} N={n} {bt}/{st} idx={idxf!r}")
                    if 100 * num == bt * den or 100 * num == st * den:
                        diag.append(f"exact tie at threshold {m} N={n} {bt}/{st} idx={idxf!r}")

    # Canonical table -----------------------------------------------------------
    header = ["market", "lookback", "buy_th", "sell_th", "side", "horizon_w",
              "n_signal_weeks", "n_episodes", "mean_ret", "median_ret", "hit_rate",
              "baseline_mean", "excess", "perm_p"]
    out_rows = []
    # cache of per-cell arrays so pooled analysis reuses them
    cell_arrays = {}
    for m in MARKETS:
        d = mk[m]
        nrep = len(d["rows"])
        for n in LOOKBACKS:
            idxl = d["idx"][n]
            for bt, st in THRESHOLDS:
                longs, shorts = setup_flags(idxl, bt, st)
                for side in SIDES:
                    flags = longs if side == "long" else shorts
                    sgn = 1.0 if side == "long" else -1.0
                    epi = episode_ids(flags)
                    for h in HORIZONS:
                        # eligible weeks: idx defined, not in a shutdown window,
                        # valid forward return at horizon h
                        elig_ret, elig_sig, elig_epi = [], [], []
                        for i in range(nrep):
                            if idxl[i][0] is None or d["shut"][i] or d["fwd"][h][i] is None:
                                continue
                            elig_ret.append(d["fwd"][h][i])
                            elig_sig.append(flags[i])
                            elig_epi.append(epi[i])
                        T = len(elig_ret)
                        r_arr = np.array(elig_ret, dtype=float)
                        s_arr = np.array(elig_sig, dtype=bool)
                        baseline = float(r_arr.mean())
                        sret = sgn * r_arr                     # signed returns
                        sig_ret = sret[s_arr]
                        nsig = int(s_arr.sum())
                        neps = len({e for e, s in zip(elig_epi, elig_sig) if s})
                        if nsig > 0:
                            mean_r = float(sig_ret.mean())
                            med_r = float(np.median(sig_ret))
                            hit = float((sig_ret > 0).mean())
                            excess = mean_r - sgn * baseline
                            pval = perm_pvalue(s_arr, sret, mean_r, draw_offsets(T))
                        else:
                            mean_r = med_r = hit = excess = pval = float("nan")
                        cell_arrays[(m, n, bt, st, side, h)] = (s_arr, sret, baseline)
                        out_rows.append([m, n, bt, st, side, h, nsig, neps, mean_r, med_r,
                                         hit, baseline, excess, pval])

    with open(f"{OUT_DIR}/canonical.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in out_rows:
            w.writerow([fmt(v) if not isinstance(v, str) else v for v in r])

    # Crude time series ---------------------------------------------------------
    d = mk["crude"]
    mm = [r["mm_net"] for r in d["rows"]]
    mm_idx26 = rolling_index(mm, 26)
    with open(f"{OUT_DIR}/crude_timeseries.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["report_date", "entry_date", "comm_net", "mm_net", "oi",
                    "idx26", "idx52", "idx156", "mm_idx26", "entry_price"])
        for i, r in enumerate(d["rows"]):
            ed = d["entry_date"][i]
            w.writerow([r["report_date"].isoformat(), ed.isoformat() if ed else "",
                        r["comm_net"], r["mm_net"], r["oi"],
                        fmt(d["idx"][26][i][0]), fmt(d["idx"][52][i][0]),
                        fmt(d["idx"][156][i][0]), fmt(mm_idx26[i][0]),
                        fmt(d["entry_px"][i])])

    # Pooled across the 5 markets (extra, for the headline note) ---------------
    # Pooled stat = mean signed return over all signal weeks of all markets.
    # Permutation: each market's signal is circularly shifted by its own
    # independent offset each iteration (one default_rng(SEED) per pooled cell,
    # offsets drawn market by market in MARKETS order).
    pooled_rows = []
    for n in LOOKBACKS:
        for bt, st in THRESHOLDS:
            for side in SIDES:
                for h in HORIZONS:
                    parts = [cell_arrays[(m, n, bt, st, side, h)] for m in MARKETS]
                    sgn = 1.0 if side == "long" else -1.0
                    sig_all = np.concatenate([p[1][p[0]] for p in parts])
                    exc_all = np.concatenate([p[1][p[0]] - sgn * p[2] for p in parts])
                    nsig = len(sig_all)
                    neps = sum(int(c[7]) for c in out_rows
                               if c[1] == n and c[2] == bt and c[4] == side and c[5] == h)
                    if nsig == 0:
                        pooled_rows.append([n, bt, st, side, h, 0, neps] + [float("nan")] * 5)
                        continue
                    obs = float(sig_all.mean())
                    rng = np.random.default_rng(SEED)
                    offs = [rng.integers(MIN_SHIFT, len(p[0]) - MIN_SHIFT, endpoint=True,
                                         size=K_PERM) for p in parts]
                    ge = 0
                    for k in range(K_PERM):
                        tot, cnt = 0.0, 0
                        for p, o in zip(parts, offs):
                            sh = np.roll(p[0], int(o[k]))
                            tot += float(p[1][sh].sum())
                            cnt += int(sh.sum())
                        if tot / cnt >= obs:
                            ge += 1
                    pooled_rows.append([n, bt, st, side, h, nsig, neps, obs,
                                        float(np.median(sig_all)), float((sig_all > 0).mean()),
                                        float(exc_all.mean()), (1 + ge) / (1 + K_PERM)])
    with open(f"{OUT_DIR}/pooled.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lookback", "buy_th", "sell_th", "side", "horizon_w", "n_signal_weeks",
                    "n_episodes", "mean_ret", "median_ret", "hit_rate", "excess_vs_own_baseline",
                    "perm_p"])
        for r in pooled_rows:
            w.writerow([fmt(v) if not isinstance(v, str) else v for v in r])

    # Self sanity checks ----------------------------------------------------------
    assert len(out_rows) == 360
    for m in MARKETS:
        for n in LOOKBACKS:
            for i, (v, _, _) in enumerate(mk[m]["idx"][n]):
                if v is not None:
                    assert -1e-9 <= v <= 100 + 1e-9, (m, n, i, v)
    base = {}
    for r in out_rows:
        key = (r[0], r[1], r[5])
        base.setdefault(key, set()).add(r[11])
    assert all(len(v) == 1 for v in base.values()), "baseline differs across sides/thresholds"
    diag.append(f"rows={len(out_rows)}; baselines consistent over {len(base)} (market,N,h) keys")
    with open(f"{OUT_DIR}/diagnostics.txt", "w") as fh:
        fh.write("\n".join(diag) + "\n")
    print("\n".join(diag[-20:]))


if __name__ == "__main__":
    main()

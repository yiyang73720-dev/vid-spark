#!/usr/bin/env python3
"""
verify_primary -- independent re-computation of the PRIMARY cell of the open-interest tool
(crude, F+O total OI, crude_roll, long, h=8), written WITHOUT the harness's evaluate /
forward-return / permutation / percentile code. It only takes the crude_roll price levels from
the harness (data, not logic), then rebuilds the signal, entry dates, forward returns, baseline,
excess, hit, episodes and the exhaustive +1-corrected circular-shift p from scratch and compares
them with results.json.

Run:  python3 -I verify_primary.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
PKG = Path(__file__).resolve().parents[2]                # package dir
S = Path(os.environ.get("LARRY_COT_ROOT") or PKG)        # ROOT: data/ ext/ work2/
WORK = S / "work2" / "tool_open_interest"               # outputs of run_open_interest.py
sys.path.insert(0, str(PKG / "battery" / "harness"))
import fund_harness as fh  # noqa: E402  (used ONLY to load the crude_roll price levels)

SHUTDOWNS = [("2013-10-01", "2013-10-31"), ("2018-12-18", "2019-02-26"),
             ("2025-09-30", "2026-01-20")]                     # v1 SPEC windows


def main() -> int:
    # 1. COT: crude F+O total OI, one row per report
    cot = pd.read_csv(S / "data" / "cot_weekly.csv", dtype={"code": str},
                      float_precision="round_trip")
    c = cot[cot["market"] == "crude"].copy()
    c["report_date"] = pd.to_datetime(c["report_date"])
    c = c.sort_values("report_date").reset_index(drop=True)
    oi = c["oi"].astype(float)
    dates = c["report_date"]

    # 2. Price as of each report date (<= 4 days stale) from crude_roll
    px = fh.load_series("crude_roll")
    pdf = pd.DataFrame({"pdate": px.index, "close": px.to_numpy()})
    m = pd.merge_asof(pd.DataFrame({"report_date": dates}), pdf, left_on="report_date",
                      right_on="pdate", direction="backward")
    stale = (m["report_date"] - m["pdate"]).dt.days > 4
    P = m["close"].where(~stale)

    # 3. 4-report changes with the 35-day gap guard
    span = (dates - dates.shift(4)).dt.days
    dOI = (oi / oi.shift(4) - 1).where(span <= 35)
    dP = (P / P.shift(4) - 1).where(span <= 35)

    # 4. Rolling 156-report percentile incl. current, >= 52 defined values (plain loop)
    def pct(x: pd.Series) -> np.ndarray:
        v = x.to_numpy()
        out = np.full(len(v), np.nan)
        for i in range(len(v)):
            if np.isnan(v[i]):
                continue
            w = v[max(0, i - 155): i + 1]
            w = w[~np.isnan(w)]
            if len(w) >= 52:
                out[i] = 100.0 * np.mean(w <= v[i])
        return out
    pOI, pP = pct(dOI), pct(dP.abs())
    valid = ~np.isnan(pOI) & ~np.isnan(pP)
    long_ = valid & (np.nan_to_num(pOI, nan=50) <= 10) & (np.nan_to_num(pP, nan=101) <= 50)

    # 5. Shutdown exclusion and COT entry: first close on/after the next week's Tuesday
    shut = np.zeros(len(c), bool)
    for a, b in SHUTDOWNS:
        shut |= ((dates >= a) & (dates <= b)).to_numpy()
    target = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D") + pd.Timedelta(days=8)
    idx = px.index
    pos = np.searchsorted(idx.values, target.values, side="left")
    ok = pos < len(idx)
    lag = np.full(len(pos), np.inf)
    lag[ok] = (idx.values[pos[ok]] - target.values[ok]) / np.timedelta64(1, "D")
    pos = np.where(ok & (lag <= 7), pos, -1)
    close = px.to_numpy()
    r8 = np.full(len(pos), np.nan)
    has = (pos >= 0) & (pos + 40 < len(close))
    r8[has] = close[pos[has] + 40] / close[pos[has]] - 1

    # 6. Cell statistics
    elig = valid & ~shut & ~np.isnan(r8)
    r_e = r8[elig]
    sig = long_[elig]
    T = int(elig.sum())
    base = float(r_e.mean())
    mean = float(r_e[sig].mean())
    n = int(sig.sum())
    hit = float((r_e[sig] > 0).mean())
    # episodes: maximal runs of long setups over the full report sequence that contain a
    # counted (eligible) week
    run_id = np.cumsum(long_ & ~np.r_[False, long_[:-1]]) * long_
    episodes = len(set(run_id[long_ & elig]))
    # exhaustive circular-shift p, offsets 26..T-26, +1 corrected
    p_idx = np.flatnonzero(sig)
    offs = np.arange(26, T - 26 + 1)
    shifted = np.array([r_e[(p_idx + k) % T].mean() for k in offs])
    p_exact = (1 + int(np.sum(shifted >= mean))) / (1 + len(offs))

    mine = dict(n=n, episodes=episodes, mean=mean, baseline=base, excess=mean - base, hit=hit,
                perm_p_exact=p_exact, T=T)
    ref = json.load(open(WORK / "results.json"))["primary"]
    ok_all = True
    for k, v in mine.items():
        same = abs(float(ref[k]) - float(v)) <= 1e-12
        ok_all &= same
        print(f"{k:13s} independent={v!r:24} results.json={ref[k]!r:24} {'OK' if same else 'DIFF'}")
    print("ALL MATCH" if ok_all else "MISMATCH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

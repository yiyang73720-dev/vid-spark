#!/usr/bin/env python3
"""
verify_s1 -- independent minimal S1 simulation (enter at the first week of each signal episode,
hold 40 trading days, one position at a time, 0.05% per side) for single markets, to sanity-check
the per-market S1 long+short numbers in results.json. Uses verify_primary's signal code.

Run:  python3 -I verify_s1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_primary as vp  # noqa: E402

fh = vp.fh
HOLD, COST = 40, 0.0005


def s1(name: str, L: int) -> dict:
    s = fh.load_series(name)
    sg = vp.seasonal_signal(vp.forward(s, vp.GRID_EXT), L)["sig"]
    fw = vp.forward(s, vp.GRID)
    # episodes on the full Friday grid (undefined weeks break an episode)
    v = sg.fillna(0.0).to_numpy()
    prev = np.r_[0.0, v[:-1]]
    starts = np.flatnonzero((v != 0) & (v != prev))
    pos = np.zeros(len(s))
    idx = s.index
    busy_until = -1
    n_tr = 0
    for i in starts:
        d = vp.GRID[i]
        e = idx.searchsorted(d + pd.Timedelta(days=1))
        if e >= len(idx) or (idx[e] - d).days > 8 or e <= busy_until:
            continue
        x = min(e + HOLD, len(idx) - 1)
        pos[e + 1:x + 1] = v[i]          # earns close-to-close returns from entry to exit
        busy_until = x
        n_tr += 1
    r = s.pct_change().fillna(0.0).to_numpy()
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    # cost charged when the position changes: entry at e (pos changes on e+1 in this array)
    net = pos * r - COST * turn
    first = vp.GRID[np.flatnonzero(sg.notna().to_numpy())[0]]
    m = idx > first
    netm = pd.Series(net[m], index=idx[m])
    yrs = (netm.index[-1] - netm.index[0]).days / 365.25
    cagr = float((1 + netm).prod() ** (1 / yrs) - 1)
    sh = float(netm.mean() / netm.std() * np.sqrt(len(netm) / yrs))
    bh = pd.Series(r[m], index=idx[m])
    return dict(series=name, n_trades=n_tr, cagr=cagr, sharpe=sh,
                bh_cagr=float((1 + bh).prod() ** (1 / yrs) - 1),
                start=str(netm.index[0].date()), end=str(netm.index[-1].date()))


if __name__ == "__main__":
    out = [s1(n, L) for _, n, L in vp.MARKETS]
    print(json.dumps(out, indent=1))

#!/usr/bin/env python3
"""
verify_primary -- independent re-computation of the PRIMARY row (crude, c1 > c2 premium, long,
h = 8 weeks, crude_roll, price timing) WITHOUT fund_harness.evaluate / premium_lib, to check that
results.json was produced from the intended rule. Re-implements: Friday grid, as-of reading of
the EIA contract prints, entry at the first crude_roll close strictly after the Friday (<= 7
days), exit 40 rows later, baseline over all eligible weeks, excess, hit, and an exhaustive
circular-shift p-value with the +1 correction. Prints a comparison with results.json.

    python3 -I battery/tool_premium/verify_primary.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True   # never write .pyc files next to the published scripts

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
sys.path.insert(0, str(PKG / "battery" / "harness"))
import fund_harness as fh  # noqa: E402  (only for the data loaders)

WORK = ROOT / "work2" / "tool_premium"


def main() -> None:
    ri = fh.build_crude_roll_index(0)
    d = ri.detail                                   # index days with c1, c2 prints
    level = ri.level
    fridays = pd.date_range("1985-01-04", "2024-03-29", freq="W-FRI")
    days = d.index.values
    lv_days = level.index.values
    rows = []
    for f in fridays:
        k = np.searchsorted(days, np.datetime64(f), side="right") - 1
        if k < 0 or (np.datetime64(f) - days[k]) > np.timedelta64(4, "D"):
            continue
        c1, c2 = d["c1"].iloc[k], d["c2"].iloc[k]
        sig = 1 if c1 > c2 else (-1 if c1 < c2 else 0)
        e = np.searchsorted(lv_days, np.datetime64(f + pd.Timedelta(days=1)), side="left")
        if e >= len(lv_days) or (lv_days[e] - np.datetime64(f + pd.Timedelta(days=1))) > \
                np.timedelta64(7, "D"):
            continue
        x = e + 40
        if x >= len(lv_days):
            continue
        rows.append((f, sig, level.iloc[x] / level.iloc[e] - 1.0))
    t = pd.DataFrame(rows, columns=["friday", "sig", "r8"])
    long_ = t["sig"].to_numpy() == 1
    r = t["r8"].to_numpy()
    mean = r[long_].mean()
    base = r.mean()
    T = len(r)
    offs = np.arange(26, T - 26 + 1)
    pos = np.flatnonzero(long_)
    shifted = np.array([r[(pos + o) % T].mean() for o in offs])
    p_exact = (1 + np.sum(shifted >= mean)) / (1 + len(offs))
    mine = dict(n=int(long_.sum()), T=int(T), mean=float(mean), baseline=float(base),
                excess=float(mean - base), hit=float((r[long_] > 0).mean()),
                perm_p_exact=float(p_exact))
    res = json.loads((WORK / "results.json").read_text())
    prim = res["primary"]
    print("independent:", mine)
    print("results.json:", {k: prim[k] for k in mine})
    ok = all(abs(float(prim[k]) - float(v)) < 1e-12 for k, v in mine.items())
    print("MATCH" if ok else "MISMATCH")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

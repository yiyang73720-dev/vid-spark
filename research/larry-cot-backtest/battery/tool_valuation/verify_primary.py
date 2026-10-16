#!/usr/bin/env python3
"""
verify_primary -- independent re-computation of the PRIMARY cell (crude vs gold, EMA2-EMA22,
N=156, long VI<=20, h=8, crude_roll), written without valuation_lib and without
fh.evaluate / fh.entry_positions / fh.forward_returns:

  * weekly inputs via pandas merge_asof (with the same 4-day staleness limit),
  * EMAs via pandas ewm(adjust=False), Pal defined from the 22nd ratio observation,
  * min/max via rolling(156),
  * entry = first crude_roll close strictly after the Friday (within 7 days),
    exit = 40 rows later,
  * n, mean, baseline, excess, hit compared with results.json.
Only the raw price loaders of the harness are used (fh.load_series).

Run:  python3 -I verify_primary.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
HERE = ROOT / "work2" / "tool_valuation"       # outputs of run_valuation.py
sys.path.insert(0, str(PKG / "battery" / "harness"))
sys.dont_write_bytecode = True
import fund_harness as fh  # noqa: E402  (loaders only)

fri = pd.DataFrame({"fri": pd.date_range("1985-01-04", "2026-10-02", freq="W-FRI")})


def weekly(series: pd.Series, name: str) -> pd.Series:
    s = series.rename(name).reset_index().rename(columns={"date": "d"})
    s["obs"] = s["d"]
    m = pd.merge_asof(fri, s, left_on="fri", right_on="d", direction="backward")
    stale = (m["fri"] - m["obs"]).dt.days > 4
    return m[name].where(~stale)


crude = weekly(fh.load_series("crude_cl"), "crude")
gold = weekly(fh.load_series("gold"), "gold")
R = (100 * crude / gold)
ok = R.notna()
r = R[ok].reset_index(drop=True)
pal = r.ewm(span=2, adjust=False).mean() - r.ewm(span=22, adjust=False).mean()
pal.iloc[:21] = np.nan
lo, hi = pal.rolling(156, min_periods=156).min(), pal.rolling(156, min_periods=156).max()
vi_d = 100 * (pal - lo) / (hi - lo)
vi = pd.Series(np.nan, index=fri.index)
vi[ok[ok].index] = vi_d.to_numpy()

px = fh.load_series("crude_roll")
idx = px.index.values
pos = np.searchsorted(idx, (fri["fri"] + pd.Timedelta(days=1)).values, side="left")
has = pos < len(idx)
lag = np.full(len(pos), np.inf)
lag[has] = (idx[pos[has]] - (fri["fri"] + pd.Timedelta(days=1)).values[has]) / np.timedelta64(1, "D")
has &= lag <= 7
xp = pos + 40
has_x = has & (xp < len(idx))
v = px.to_numpy()
r8 = np.full(len(pos), np.nan)
r8[has_x] = v[xp[has_x]] / v[pos[has_x]] - 1

elig = vi.notna().to_numpy() & np.isfinite(r8)
sig = elig & (vi.to_numpy() <= 20)
mean = r8[sig].mean()
base = r8[elig].mean()
mine = dict(n=int(sig.sum()), mean=float(mean), baseline=float(base), excess=float(mean - base),
            hit=float((r8[sig] > 0).mean()), T=int(elig.sum()))
res = json.loads((HERE / "results.json").read_text())["primary"]
print("independent:", mine)
print("results.json:", {k: res[k] for k in mine})
for k in mine:
    assert abs(mine[k] - res[k]) < 1e-12, (k, mine[k], res[k])
print("PRIMARY CELL VERIFIED (independent implementation matches to 1e-12)")

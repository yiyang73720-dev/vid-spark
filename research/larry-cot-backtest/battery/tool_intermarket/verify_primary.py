#!/usr/bin/env python3
"""
verify_primary -- independent re-computation of the PRIMARY row of results.json with plain
pandas/numpy loops (no harness evaluate(), no intermarket_lib signal code), as a cross-check.

Primary (PREREG.md section 3): gold LONG when the broad dollar's 8-week change, read as of
Friday - 7 days (DTWEXBGS is published Mondays for the previous week), is negative; entry at the
first gold close strictly after the Friday (no entry if that close is > 7 days after Friday+1);
exit 40 gold trading days later; baseline = mean over all weeks with a defined dollar change and
a forward return. Also recomputes the exhaustive circular-shift p (+1 correction).
Run: python3 -I verify_primary.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
HERE = ROOT / "work2" / "tool_intermarket"     # outputs of run_intermarket.py
sys.path.insert(0, str(PKG / "battery" / "harness"))
import fund_harness as fh  # noqa: E402  (used ONLY to load the two raw series)

gold = fh.load_series("gold")
usd = fh.load_series("usd_broad")


def last_on_or_before(s: pd.Series, d: pd.Timestamp):
    """Last value dated <= d if at most 4 days old, else None."""
    sub = s.loc[:d]
    if sub.empty or (d - sub.index[-1]).days > 4:
        return None
    return float(sub.iloc[-1])


rows = []
for fri in pd.date_range(gold.index[0], gold.index[-1], freq="W-FRI"):
    a = last_on_or_before(usd, fri - pd.Timedelta(days=7))
    b = last_on_or_before(usd, fri - pd.Timedelta(days=63))
    if a is None or b is None:
        continue
    chg = a / b - 1.0
    after = gold.index[gold.index > fri]
    if len(after) == 0 or (after[0] - (fri + pd.Timedelta(days=1))).days > 7:
        continue
    i = gold.index.get_loc(after[0])
    if i + 40 >= len(gold):
        continue
    rows.append(dict(fri=fri, chg=chg, r=gold.iloc[i + 40] / gold.iloc[i] - 1.0))
df = pd.DataFrame(rows)
long = df["chg"] < 0
r = df["r"].to_numpy()
n = int(long.sum())
mean = float(r[long].mean())
base = float(r.mean())
T = len(df)
sig = long.to_numpy()
pos = np.flatnonzero(sig)
offs = np.arange(26, T - 26 + 1)
shifted = np.array([r[(pos + k) % T].mean() for k in offs])
p_exact = (1 + int(np.sum(shifted >= mean))) / (1 + len(offs))
mine = dict(n=n, T=T, mean=mean, baseline=base, excess=mean - base,
            hit=float((r[long] > 0).mean()), perm_p_exact=p_exact,
            period=f"{df['fri'].min().date()}..{df['fri'].max().date()}")
res = json.loads((HERE / "results.json").read_text())["primary"]
print("independent:", mine)
print("results.json:", {k: res[k] for k in mine})
ok = (mine["n"] == res["n"] and mine["T"] == res["T"]
      and all(abs(mine[k] - res[k]) < 1e-12 for k in ("mean", "baseline", "excess", "hit",
                                                      "perm_p_exact")))
print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

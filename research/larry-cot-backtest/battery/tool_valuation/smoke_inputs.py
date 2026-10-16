#!/usr/bin/env python3
"""Coverage-only smoke checks of the valuation inputs (no forward returns are read)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))   # valuation_lib
import numpy as np
import pandas as pd
import valuation_lib as vl

fri = vl.friday_grid()
M, C = vl.market_inputs(fri), vl.comparator_inputs(fri)
for k, v in {**{f"M:{a}": b for a, b in M.items()}, **{f"C:{a}": b for a, b in C.items()}}.items():
    ok = np.isfinite(v)
    print(f"{k:10s} defined weeks={ok.sum():5d} first={fri[ok][0].date()} last={fri[ok][-1].date()}")
# primary index coverage
out = vl.valuation_index(M["crude"], C["gold"], (2, 22), 156)
vi = out["vi"]; ok = np.isfinite(vi)
print("primary VI defined weeks", ok.sum(), fri[ok][0].date(), fri[ok][-1].date())
# sign flip symmetry check: Pal' = -Pal => VI' = 100 - VI
r = out["R"]; okr = np.isfinite(r)
from valuation_lib import ema_recursive
pal = out["pal"]
neg = -pal[okr]
s = pd.Series(neg); lo = s.rolling(156, min_periods=156).min(); hi = s.rolling(156, min_periods=156).max()
vi_neg = (100 * (s - lo) / (hi - lo)).to_numpy()
d = vi[okr] + vi_neg
print("max |VI + VI_flipped - 100| =", np.nanmax(np.abs(d - 100)))
# psys gold / rclc1 coverage
g = vl.load_psys_gold(); c1 = vl.load_rclc1()
print("psys gold", g.index[0].date(), g.index[-1].date(), len(g), "| rclc1", c1.index[0].date(), c1.index[-1].date(), len(c1))

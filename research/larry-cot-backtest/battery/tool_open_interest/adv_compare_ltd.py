#!/usr/bin/env python3
"""adv_compare_ltd -- compare the adversarial verifier's own crude roll index (adv_verify_oi.py)
with the harness's crude_roll: last-trade dates, daily levels and the primary-cell forward returns.
Run: python3 -I adv_compare_ltd.py"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
W = Path(__file__).resolve().parents[1]        # battery code dir (tool_open_interest/, harness/)
sys.path.insert(0, str(W / "harness")); sys.path.insert(0, str(W / "tool_open_interest"))
import fund_harness as fh
import adv_verify_oi as av

lvl, info, ltd_mine, cal = av.build_roll_index()
ri = fh.build_crude_roll_index(0)
h_ltd = ri.ltd[(ri.ltd >= ltd_mine[0]) & (ri.ltd <= ltd_mine[-1])]
only_mine = ltd_mine.difference(h_ltd); only_h = h_ltd.difference(ltd_mine)
print("LTDs mine", len(ltd_mine), "harness(in range)", len(h_ltd), "differ:", len(only_mine), len(only_h))
print(" only mine:", [d.date().isoformat() for d in only_mine])
print(" only harness:", [d.date().isoformat() for d in only_h])
# rebuild my index using the harness LTDs -> should match harness level exactly
lvl_h, *_ = av.build_roll_index(ri.ltd)
hl = fh.load_series("crude_roll")
common = hl.index.intersection(lvl_h.index)
print("days mine", len(lvl_h), "harness", len(hl), "common", len(common))
print("max rel diff (my code + harness LTDs vs harness level):",
      float(np.max(np.abs(lvl_h[common].to_numpy() / hl[common].to_numpy() - 1))))
# primary with my code but harness LTDs
rep = av.load_cot("crude")
f = av.primary_signal(rep, lvl_h)
fw = av.forward(f["report_date"], lvl_h)
c = av.cell(f, fw["r"].to_numpy(), +1)
print("PRIMARY with harness LTDs (my code):", c)

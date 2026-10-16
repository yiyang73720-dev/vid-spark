#!/usr/bin/env python3
"""
Re-run the whole robustness & data-snooping lens, in order, each script in its own isolated
interpreter (python3 -I). Every number in the lens report comes from these outputs.

    python3 -I run_all.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code dir; outputs go to ROOT/work/lens_robustness
# Children inherit -B (no .pyc) from this interpreter; cwd is a neutral temp dir (the scripts
# use absolute paths only, and -I keeps cwd off sys.path).
CHILD_FLAGS = ["-I"] + (["-B"] if sys.flags.dont_write_bytecode else [])
SCRIPTS = ["a_multiple_testing.py",      # (a) counts vs 5%, BH, joint shift null, reverse tail
           "b_heatmaps.py",              # (b) heatmaps.json
           "c_subperiods.py",            # (c) 2006-2015 vs 2016-2026 (+ crude by year)
           "defg_variants.py",           # (d) EIA spot, (e) shutdown in, (f) comm/oi, (g) Monday
           "silver_short_followup.py",   # follow-up on the only recurring pocket
           "h_rank_ic.py"]               # (h) Spearman IC with block bootstrap

for s in SCRIPTS:
    print(f"=== {s}", flush=True)
    r = subprocess.run([sys.executable, *CHILD_FLAGS, str(HERE / s)], cwd=tempfile.gettempdir())
    if r.returncode:
        sys.exit(f"{s} failed with exit code {r.returncode}")
print("all lens scripts finished")

#!/usr/bin/env python3
"""
Reconciliation and verification for the Larry COT engine.

  1. Diff implementation A against implementation B: canonical.csv cell by cell, joined on
     the key columns, plus crude_timeseries.csv.
  2. Check that the engine (default seed 11) matches A's perm_p exactly and A's deterministic
     columns to <1e-15, and that the engine with --seed 22 reproduces B's canonical.csv
     exactly (every column, bit for bit after round-trip parsing).
  2b. Show the root cause of the ulp-level A-vs-B float differences: pandas' default CSV float
     parser against correctly rounded parsing of prices_daily.csv.
  3. Check that the A-vs-B perm_p differences are pure Monte-Carlo noise: compare both with
     the exhaustive (noise-free) p and with the binomial standard error sqrt(p(1-p)/K).
  4. Check the pooled cells against A's pooled_headline.csv and B's pooled.csv.

Usage:  python3 -I verify_engine.py [ROOT]     (default: $LARRY_COT_ROOT or package dir)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(sys.argv[1] if len(sys.argv) > 1 else
         os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[1])
ENG = S / "work" / "engine"                                   # engine outputs
sys.path.insert(0, str(Path(__file__).resolve().parent))     # engine code (larry_cot.py)
import larry_cot as lc  # noqa: E402

KEYS = ["market", "lookback", "buy_th", "sell_th", "side", "horizon_w"]
INTS = ["n_signal_weeks", "n_episodes"]
FLOATS = ["mean_ret", "median_ret", "hit_rate", "baseline_mean", "excess"]
TOL = 1e-9


def rd(path) -> pd.DataFrame:
    """Read a CSV with correctly rounded float parsing, so equality checks are exact."""
    return pd.read_csv(path, float_precision="round_trip")


def diff_canonical(a: pd.DataFrame, b: pd.DataFrame, la: str, lb: str) -> pd.DataFrame:
    """Join on keys and report mismatches; return the merged frame."""
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "key sets differ"
    print(f"--- {la} vs {lb}: {len(m)} joined rows, same row order: "
          f"{bool((a[KEYS].values == b[KEYS].values).all())}")
    for c in INTS:
        print(f"  {c:15s} mismatches: {int((m[c + '_a'] != m[c + '_b']).sum())}")
    for c in FLOATS:
        d = (m[c + "_a"] - m[c + "_b"]).abs()
        print(f"  {c:15s} max|diff| {d.max():.3e}  cells >{TOL:g}: {int((d > TOL).sum())}  "
              f"bit-identical: {int((d == 0).sum())}/{len(m)}")
    d = m["perm_p_a"] - m["perm_p_b"]
    print(f"  perm_p          max|diff| {d.abs().max():.4f}  mean {d.mean():+.5f}  "
          f"sd {d.std():.5f}  cells >0.03: {int((d.abs() > 0.03).sum())}  "
          f"identical: {int((d == 0).sum())}")
    return m


def diff_timeseries(a: pd.DataFrame, b: pd.DataFrame, la: str, lb: str) -> None:
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    bad = []
    for c in a.columns:
        if c.endswith("date"):
            n = int((a[c].fillna("") != b[c].fillna("")).sum())
        else:
            nan_mis = int((a[c].isna() != b[c].isna()).sum())
            n = nan_mis + int(((a[c] - b[c]).abs() > TOL).sum())
        if n:
            bad.append((c, n))
    print(f"--- {la} vs {lb} crude_timeseries: {len(a)} rows, mismatching columns: {bad or 'none'}")


def main() -> None:
    A = rd(S / "work/implA/canonical.csv")
    B = rd(S / "work/implB/canonical.csv")
    E = rd(ENG / "canonical.csv")
    Ex = rd(ENG / "canonical_exact_p.csv")
    tsA = rd(S / "work/implA/crude_timeseries.csv")
    tsB = rd(S / "work/implB/crude_timeseries.csv")
    tsE = rd(ENG / "crude_timeseries.csv")

    # 1. A vs B
    mAB = diff_canonical(A, B, "A(seed11)", "B(seed22)")
    diff_timeseries(tsA, tsB, "A", "B")
    sig = mAB[(mAB.perm_p_a < 0.05) != (mAB.perm_p_b < 0.05)]
    print(f"  cells where p<0.05 verdict differs A vs B: {len(sig)}")
    if len(sig):
        print(sig[KEYS + ["perm_p_a", "perm_p_b"]].to_string(index=False))

    # 2. Engine vs A (same seed): expect identical outputs, perm_p included.
    mEA = diff_canonical(E, A, "engine(seed11)", "A(seed11)")
    assert (mEA.perm_p_a == mEA.perm_p_b).all(), "engine perm_p != A perm_p with same seed"
    diff_timeseries(tsE, tsA, "engine", "A")

    # 2b. Root cause of the ulp-level A/B float differences: A used pandas' default parser.
    import csv
    with open(S / "data/prices_daily.csv", newline="") as fh:
        exact = np.array([float(r["close"]) for r in csv.DictReader(fh)])
    dflt = pd.read_csv(S / "data/prices_daily.csv")["close"].to_numpy()
    rt = rd(S / "data/prices_daily.csv")["close"].to_numpy()
    print(f"  prices_daily closes: pandas default parser != correctly rounded in "
          f"{int((dflt != exact).sum())}/{len(exact)} rows; round_trip parser: "
          f"{int((rt != exact).sum())} rows")

    # Engine with B's seed: expect B's perm_p exactly (same offsets, same shift convention).
    with tempfile.TemporaryDirectory() as td:
        lc.main(["--root", str(S), "--out", td, "--seed", "22", "--no-extras"])
        E22 = rd(Path(td) / "canonical.csv")
    mEB = diff_canonical(E22, B, "engine(seed22)", "B(seed22)")
    n_same = int((mEB.perm_p_a == mEB.perm_p_b).sum())
    all_cols = [c for c in lc.CANONICAL_COLUMNS if c not in KEYS]
    n_bit = int(np.all([(mEB[c + "_a"] == mEB[c + "_b"]).to_numpy() for c in all_cols],
                       axis=0).sum())
    print(f"  engine(seed22) perm_p identical to B in {n_same}/360 cells; rows identical "
          f"in every column: {n_bit}/360")
    assert n_bit == 360, "engine(seed22) must reproduce B exactly"
    diff_timeseries(tsE, tsB, "engine", "B")

    # 3. Monte-Carlo noise check against the exhaustive p.
    x = Ex.merge(B[KEYS + ["perm_p"]].rename(columns={"perm_p": "perm_p_B"}), on=KEYS)
    k = lc.K_PERM
    pe = x["perm_p_exact"].clip(1e-4, 1 - 1e-4)
    se = np.sqrt(pe * (1 - pe) / k)
    for lab, col in [("A/engine seed11", "perm_p"), ("B seed22", "perm_p_B")]:
        z = (x[col] - x["perm_p_exact"]) / se
        print(f"  {lab}: |p - p_exact| max {np.abs(x[col] - x['perm_p_exact']).max():.4f}, "
              f"z sd {z.std():.3f} (expect ~1), |z|>3: {int((z.abs() > 3).sum())}")
    print(f"  p_exact < 0.05 cells: {int((x.perm_p_exact < 0.05).sum())}; seed11: "
          f"{int((x.perm_p < 0.05).sum())}; seed22: {int((x.perm_p_B < 0.05).sum())}")
    print(x[(x.perm_p_exact < 0.06)][KEYS + ["n_signal_weeks", "excess", "perm_p", "perm_p_B",
                                             "perm_p_exact"]].to_string(index=False))

    # 4. Pooled cells.
    P = rd(ENG / "pooled.csv")
    PA = rd(S / "work/implA/pooled_headline.csv")
    pe26 = P[(P.lookback == 26) & (P.buy_th == 80)].reset_index(drop=True)
    cols = ["n_signal_weeks", "n_episodes", "mean_ret", "median_ret", "hit_rate",
            "excess_vs_own_baseline", "equal_weight_mkt_excess", "perm_p"]
    dA = max(float(np.max(np.abs(pe26[c].to_numpy(float) - PA[c].to_numpy(float)))) for c in cols)
    print(f"--- pooled N=26 80/20 (8 rows) engine(seed11) vs A: max|diff| {dA:.2e}")
    with tempfile.TemporaryDirectory():
        panel = lc.build_panel(S)
        P22 = lc.pooled_table(panel, seed=22)
    PB = rd(S / "work/implB/pooled.csv")
    cB = ["n_signal_weeks", "n_episodes", "mean_ret", "median_ret", "hit_rate",
          "excess_vs_own_baseline", "perm_p"]
    dB = max(float(np.max(np.abs(P22[c].to_numpy(float) - PB[c].to_numpy(float)))) for c in cB)
    print(f"--- pooled all 72 cells engine(seed22) vs B: max|diff| {dB:.2e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
adv_verify_wf -- ADVERSARIAL re-derivation of the walk-forward result WF-1 (PREREG section 5)
and of "primary restricted to 2010+", with the verifier's own code (adv_verify_oi.py).

WF-1: each calendar year from 2010, choose among the 24 grid-C rules (W in 1,2,4,13; p in 5,10,20;
range filter on/off) the one with the highest PAST long excess at h=8. The excess is measured
only on weeks whose 8-week exit date is before 1 January of the test year (not excluded, rule
defined, forward return defined), against those same weeks' mean; a rule needs >= 15 past long
weeks. The chosen rule's signal is used for that year. Evaluated long, h=8.

It runs twice: on the verifier's own roll index (own last-trade dates) and, for an exact
comparison, on the same code with the harness's last-trade dates (the only harness import).

Run:  python3 -I adv_verify_wf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parents[1]        # battery code dir (tool_open_interest/, harness/)
sys.path.insert(0, str(W / "tool_open_interest"))
import adv_verify_oi as av  # noqa: E402

GRID = [(w, p, rf) for w in (1, 2, 4, 13) for p in (5, 10, 20) for rf in (True, False)]


def rule_signal(rep: pd.DataFrame, lvl: pd.Series, w: int, p: float, rf: bool) -> np.ndarray:
    """+1/-1/0, NaN where undefined (same construction as the primary, any W, p, range on/off)."""
    f = rep.copy()
    f["px"] = av.asof(lvl, f["report_date"])
    dOI = av.change(f["oi"].to_numpy(), f["report_date"], w, 7 * w + 7)
    dP = av.change(f["px"].to_numpy(), f["report_date"], w, 7 * w + 7)
    pOI = av.pct_window(dOI)
    pP = av.pct_window(np.abs(dP))
    valid = ~np.isnan(pOI) & (~np.isnan(pP) if rf else True)
    rng = (np.nan_to_num(pP, nan=101) <= 50) if rf else np.ones(len(f), bool)
    s = np.where(rng & (pOI <= p), 1.0, np.where(rng & (pOI >= 100 - p), -1.0, 0.0))
    return np.where(valid, s, np.nan)


def run(lvl: pd.Series, tag: str) -> None:
    rep = av.load_cot("crude")
    fw = av.forward(rep["report_date"], lvl)
    r8, ex8 = fw["r"].to_numpy(), fw["exit_date"].to_numpy()
    excl = rep["shutdown"].to_numpy(bool)
    dates = rep["report_date"].to_numpy()
    sigs = {f"oi{w}w_p{p}_{'range' if rf else 'any'}": rule_signal(rep, lvl, w, p, rf)
            for (w, p, rf) in GRID}
    wf = np.full(len(rep), np.nan)
    choices = []
    for y in range(2010, pd.Timestamp(dates[-1]).year + 1):
        cut = np.datetime64(f"{y}-01-01")
        hist = dates < cut
        done = hist & ~np.isnat(ex8) & (ex8 < cut) & ~excl & ~np.isnan(r8)
        best, best_ex = None, -np.inf
        for name, s in sigs.items():
            el = done & ~np.isnan(s)
            lo = el & (s == 1)
            if lo.sum() < 15:
                continue
            ex = r8[lo].mean() - r8[el].mean()
            if ex > best_ex:
                best, best_ex = name, ex
        blk = (dates >= cut) & (dates < np.datetime64(f"{y + 1}-01-01"))
        wf[blk] = sigs[best][blk] if best else 0.0
        choices.append(f"{y}:{best}")
    f = pd.DataFrame({"report_date": rep["report_date"], "shutdown": excl,
                      "valid": ~np.isnan(wf), "signal": np.nan_to_num(wf).astype(int)})
    c = av.cell(f, r8, +1)
    print(f"[{tag}] WF-1 long h8:", {k: c[k] for k in ("n", "episodes", "T", "mean", "baseline",
                                                         "excess", "hit", "perm_p_exact", "period")})
    print(f"[{tag}] choices:", "; ".join(choices))
    ep, base = av.episode_table(f, r8, +1)
    print(f"[{tag}] WF-1 episodes:\n", ep.to_string())
    top = ep.loc[ep["sum_excess"].idxmax()]
    rest = (ep["sum_excess"].sum() - top["sum_excess"]) / (c["n"] - top["weeks"])
    print(f"[{tag}] WF-1 excess without its largest episode ({top['first_report']}, "
          f"{top['weeks']} wk, {top['sum_excess'] / ep['sum_excess'].sum():.3f} of the summed "
          f"excess): {rest}")
    # the primary restricted to decision dates >= 2010-01-01 (own baseline and T)
    fp = av.primary_signal(rep, lvl)
    m = (rep["report_date"] >= "2010-01-01").to_numpy()
    c2 = av.cell(fp, r8, +1, extra_mask=m)
    print(f"[{tag}] primary 2010+:", {k: c2[k] for k in ("n", "episodes", "T", "excess", "perm_p_exact")})


if __name__ == "__main__":
    lvl_own, *_ = av.build_roll_index()
    run(lvl_own, "own LTDs")
    sys.path.insert(0, str(W / "harness"))
    import fund_harness as fh  # noqa: E402  (only for the harness's last-trade dates)
    lvl_h, *_ = av.build_roll_index(fh.build_crude_roll_index(0).ltd)
    run(lvl_h, "harness LTDs")

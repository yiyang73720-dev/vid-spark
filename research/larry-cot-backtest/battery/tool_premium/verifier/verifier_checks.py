#!/usr/bin/env python3
"""
verifier_checks -- adversarial checks of tool_premium that need the SHARED harness series
(crude_roll as published), so their numbers are directly comparable with results.json.
All of these are POST-HOC verifier diagnostics: they never change the pre-registered primary
row or the pre-declared verdict; they test whether the claims' wording is supported.

  1. Sub-periods of the primary rule (long, h=8, crude_roll), each with its own baseline and
     exhaustive circular-shift p: where inside the split-half H2 (2004-07-23..2024) the result
     comes from (2004-07..2009 vs 2010..2024).
  2. Single-year dependence: the primary with 2008, 2014, 2020 (and all three) removed.
  3. Baseline hit rate (share of ALL eligible weeks with a positive 8-week return), to put the
     62.9% hit rate in context.
  4. Strategy claim check: which strategy rows beat random timing (null p <= 0.05) on which
     series, and whether long-only is above or below the random-timing median.
  5. Writes everything (plus verify_output.json from verify_independent.py, if present) into
     results.json under the key "verification" (overwriting only that key).

Run after verify_independent.py:
    python3 -I battery/tool_premium/verifier/verifier_checks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent         # code: verify_independent.py
sys.path.insert(0, str(HERE.parent))           # code: premium_lib.py
sys.path.insert(0, str(HERE))      # verify_independent (python -I drops the script dir)
import premium_lib as pl  # noqa: E402  (signal builders, as used by run_premium.py)
fh = pl.fh
WORK = fh.WORK2 / "tool_premium"               # outputs: ROOT/work2/tool_premium

FRI = fh.weekly_decision_dates("1985-01-04", "2024-03-29")


def primary_frame() -> pd.DataFrame:
    """The pre-registered primary signal frame, rebuilt exactly as run_premium.py does."""
    p12 = pl.weekly_asof(pl.crude_premium_daily(1, 2), FRI)
    return fh.make_signal_frame(FRI, pl.sig_sign(p12))


def restrict(sf: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Signal frame with weeks outside `mask` made invalid (own baseline / own T)."""
    part = sf.copy()
    part["valid"] = part["valid"].to_numpy(bool) & np.asarray(mask, bool)
    part["signal"] = np.where(part["valid"], part["signal"], 0).astype(np.int8)
    return part


def long8(sf: pd.DataFrame) -> dict:
    """Primary-cell statistics (long, h=8, crude_roll, price timing) of a signal frame."""
    r = fh.evaluate(sf, "crude_roll", (8,), timing="price", sides=("long",)).iloc[0]
    return {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
            ("n", "episodes", "T", "mean", "baseline", "excess", "hit", "perm_p",
             "perm_p_exact", "period")}


def roll_dating_check() -> dict:
    """Compare fh.build_crude_roll_index(0) with verify_independent's print-calendar index.
    For every day whose daily return differs, test the two readings against the BACK contracts
    (never the expiring contract 1): if contract identities changed between t' and t, then
    new c2/c3 continue old c3/c4; otherwise c2/c3 continue themselves. The reading whose two
    back-contract returns agree more closely with each other is the better-supported one
    (|r_a - r_b| is reported for both readings; smaller = more consistent)."""
    import verify_independent as vi            # same directory; parse-only loaders
    raw = vi.load_contracts()
    both = raw[["c1", "c2"]].dropna()
    days = raw.index[raw.notna().any(axis=1).to_numpy()]
    mine, _ = vi.roll_index(both["c1"], both["c2"], vi.ltd_from_print_calendar(days))
    hz = fh.build_crude_roll_index(0)
    a = mine.pct_change()
    b = hz.level.pct_change().reindex(a.index)
    diff = (a - b).abs()
    rows = []
    for t in diff.index[(diff > 1e-9).to_numpy()]:
        k = raw.index.get_loc(t)
        cur, prv = raw.iloc[k], raw.iloc[k - 1]
        no_roll = [cur["c2"] / prv["c2"] - 1, cur["c3"] / prv["c3"] - 1]
        roll = [cur["c2"] / prv["c3"] - 1, cur["c3"] / prv["c4"] - 1]
        rows.append(dict(date=t.date().isoformat(), ret_independent=float(a[t]),
                         ret_harness=float(b[t]), harness_rolls_here=bool(hz.detail.loc[t, "roll"]),
                         back_c2c3_no_roll=no_roll, back_c2c3_roll=roll,
                         spread_no_roll=float(abs(no_roll[0] - no_roll[1])),
                         spread_roll=float(abs(roll[0] - roll[1]))))
    return dict(n_days_differing=len(rows), days=rows,
                note="NaN spreads = a back contract has no print that day (undecidable).")


def main() -> None:
    sf = primary_frame()
    d = pd.DatetimeIndex(sf["decision_date"])
    out: dict = dict(note="POST-HOC verifier diagnostics; they do not change the pre-registered "
                          "primary row or the pre-declared verdict.")

    # 1. sub-periods
    periods = {"1985-01-04..2004-07-16 (split H1)": ("1985-01-01", "2004-07-19"),
               "2004-07-23..2009-12-25": ("2004-07-20", "2009-12-31"),
               "2010-01-01..2024-02-02": ("2010-01-01", "2024-12-31"),
               "1985-01-04..2009-12-25": ("1985-01-01", "2009-12-31")}
    out["subperiods_long_h8"] = {k: long8(restrict(sf, (d >= a) & (d <= b)))
                                 for k, (a, b) in periods.items()}

    # 2. single-year dependence
    yrs = d.year.to_numpy()
    out["drop_years_long_h8"] = {"_".join(map(str, ys)): long8(restrict(sf, ~np.isin(yrs, ys)))
                                 for ys in ([2008], [2014], [2020], [2008, 2014, 2020])}

    # 3. baseline hit rate on the same eligible weeks
    ft = fh.forward_table(FRI, "crude_roll", "price", (8,))
    elig = sf["valid"].to_numpy(bool) & ft["r8"].notna().to_numpy()
    r8 = ft["r8"].to_numpy()[elig]
    out["baseline_hit_all_eligible_weeks"] = dict(T=int(elig.sum()), hit=float((r8 > 0).mean()))

    # 4. strategy rows vs random timing
    res = json.loads((WORK / "results.json").read_text())
    st = pd.DataFrame(res["strategy"])
    st = st[st["strategy"] != "buy_hold"]
    out["strategy_vs_random_timing"] = [
        dict(series=r.series, strategy=r.strategy, sharpe=r.sharpe,
             null_median=r.null_sharpe_median, null_p=r.null_p_sharpe,
             beats_random_p05=bool(r.null_p_sharpe <= 0.05),
             above_null_median=bool(r.sharpe > r.null_sharpe_median))
        for r in st.itertuples()]

    # 5. roll-dating comparison with the independent index, and the raw-print evidence
    out["roll_dating"] = roll_dating_check()

    # 6. merge the independent re-derivation and write
    ind = WORK / "verifier" / "verify_output.json"
    if ind.exists():
        out["independent_rederivation"] = json.loads(ind.read_text())
        a = out["independent_rederivation"]["A_primary_my_index"]
        out["roll_dating"]["primary_effect"] = dict(
            harness=dict(excess=res["primary"]["excess"], perm_p_exact=res["primary"]["perm_p_exact"]),
            independent=dict(excess=a["excess"], perm_p_exact=a["perm_p_exact"]))
    res["verification"] = fh.to_jsonable(out)
    (WORK / "results.json").write_text(json.dumps(res, indent=1, allow_nan=False))
    print(json.dumps(fh.to_jsonable({k: v for k, v in out.items()
                                     if k != "independent_rederivation"}), indent=1))


if __name__ == "__main__":
    main()

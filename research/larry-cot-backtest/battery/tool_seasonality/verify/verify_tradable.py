#!/usr/bin/env python3
"""
verify_tradable -- independent re-derivation of the TRADABILITY confirmation (claims C4, C5, C6)
and data checks on the roll-adjusted futures indices.

Signal / statistics code comes from verify_primary.py (independent of seasonal.py). The
roll-adjusted indices themselves are the author's construction (seas_data.tradable_series); this
script cross-checks them against the EIA contract-1 series (ho_f1, rbob_f1) on non-roll days.

Run:  python3 -I verify_tradable.py      (writes ROOT/work2/tool_seasonality/verify/verify_tradable.json)
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
sys.path.insert(0, str(HERE.parent))
import verify_primary as vp  # noqa: E402
import seas_data as sd       # noqa: E402  (author's roll-adjusted indices: data only)

fh = vp.fh
KEYS = ["crude", "heatoil", "gasoline", "gold", "silver", "soybean", "sugar"]
PRIMARY_SER = dict(zip(KEYS, ["crude_spot", "ho_spot", "gas_spot", "gold", "silver", "soybean", "sugar"]))
PRIMARY_L = dict(zip(KEYS, [15, 15, 15, 10, 10, 10, 10]))


def tradable(k: str) -> pd.Series:
    return sd.tradable_series(k)


def frame_from(est: pd.Series, ev: pd.Series, L: int) -> pd.DataFrame:
    """Signal estimated on `est`, returns measured on `ev` (both via verify_primary code)."""
    fw_est = vp.forward(est, vp.GRID_EXT)
    sg = vp.seasonal_signal(fw_est, L)
    fw_ev = vp.forward(ev, vp.GRID)
    df = sg.join(fw_ev, rsuffix="_ev")
    return df[df["sig"].notna() & df["r"].notna()].copy()


def restrict(frames: dict, start: str, end: str) -> dict:
    return {k: f[(f.index >= start) & (f.index <= end)] for k, f in frames.items()}


def main():
    out = {}
    # C4: estimation = evaluation = roll-adjusted futures, L = 15
    tr = {k: frame_from(tradable(k), tradable(k), 15) for k in KEYS}
    P = vp.pooled(tr)
    P["perm_p_seed2024"] = vp.perm_independent(tr)
    P["period"] = f"{min(f.index.min() for f in tr.values()).date()}..{max(f.index.max() for f in tr.values()).date()}"
    out["C4_tradable_full"] = P
    trc = restrict(tr, "2001-01-01", "2024-03-28")
    out["C4_tradable_2001_2024"] = dict(**vp.pooled(trc), perm_p_seed2024=vp.perm_independent(trc))
    out["C4_per_market"] = {k: dict(**vp.pooled({k: f}), perm_p_seed2024=vp.perm_independent({k: f}))
                            for k, f in tr.items()}
    # C5: primary (spot / v1-learned) signal evaluated on the tradable series
    cr = {k: frame_from(fh.load_series(PRIMARY_SER[k]), tradable(k), PRIMARY_L[k]) for k in KEYS}
    out["C5_primary_signal_on_tradable"] = dict(**vp.pooled(cr), perm_p_seed2024=vp.perm_independent(cr))
    out["C5_per_market"] = {k: dict(**vp.pooled({k: f}), perm_p_seed2024=vp.perm_independent({k: f}))
                            for k, f in cr.items()}
    # Gasoline: spot vs roll-adjusted futures on the gasoline-spot setup weeks
    gs = vp.market_frame("gas_spot", 15)
    fut = vp.forward(tradable("gasoline"), vp.GRID)
    j = gs[gs["sig"] != 0].join(fut, rsuffix="_fut").dropna(subset=["r_fut"])
    fut_base = float(fut.loc[gs.index.intersection(fut.dropna().index), "r"].mean())
    out["C6_gasoline_setup_weeks"] = dict(
        n=int(len(j)), signed_spot=float((j["sig"] * j["r"]).mean()),
        signed_fut=float((j["sig"] * j["r_fut"]).mean()), fut_baseline_on_spot_eligible_weeks=fut_base,
        months_long=j[j["sig"] > 0].index.month.value_counts().sort_index().to_dict(),
        months_short=j[j["sig"] < 0].index.month.value_counts().sort_index().to_dict())
    # Spot-minus-futures log drift by calendar quarter (seasonal basis that futures cannot earn)
    g = fh.load_series("gas_spot")
    f = tradable("gasoline")
    d = pd.concat([np.log(g).diff().rename("spot"), np.log(f).diff().rename("fut")], axis=1, sort=True).dropna()
    d = d[d.index >= "1990-01-01"]
    nyrs = d.index.year.nunique()
    out["gas_spot_minus_fut_log_drift_per_year_by_quarter"] = {
        f"Q{q}": float((d["spot"] - d["fut"])[d.index.quarter == q].sum() / nyrs) for q in (1, 2, 3, 4)}
    # Gasoline spot vs RBOB contract-1 log basis by ISO week (2006-06 on): the futures switch
    # grade (April contract = summer grade, October contract = winter grade) on fixed dates
    rb = fh.load_series("rbob_f1")
    jb = pd.concat([g.rename("spot"), rb.rename("rb")], axis=1, join="inner", sort=True)
    basis = np.log(jb["spot"] / jb["rb"])
    basis = basis[basis.index >= "2006-06-01"]
    wk = basis.groupby(basis.index.isocalendar().week.astype(int)).mean()
    out["gas_spot_vs_rbob_f1_log_basis_by_iso_week"] = {int(k): round(float(v), 4) for k, v in wk.items()}
    out["gas_basis_step_week8_to_week10"] = float(wk.loc[10] - wk.loc[8])
    out["gas_basis_step_week35_to_week36"] = float(wk.loc[36] - wk.loc[35])
    # Index sanity: correlation of daily returns with EIA contract-1 series on non-roll days
    chk = {}
    for k, raw in (("heatoil", "ho_f1"), ("gasoline", "rbob_f1")):
        idx = sd.psys_roll_index(k)["detail"]
        c1 = fh.load_series(raw)
        a = pd.concat([idx["ret"].rename("idx"), c1.pct_change().rename("c1"),
                       idx["roll"].rename("roll")], axis=1, sort=True).dropna()
        a = a[a.index >= "2006-01-01"]
        nr = a[~a["roll"].astype(bool)]
        chk[k] = dict(n=int(len(nr)), corr_nonroll=float(nr["idx"].corr(nr["c1"])),
                      share_abs_diff_lt_0_5pct=float(((nr["idx"] - nr["c1"]).abs() < 0.005).mean()))
    out["index_vs_eia_contract1_from_2006"] = chk
    vp.HERE.mkdir(parents=True, exist_ok=True)       # outputs: ROOT/work2/tool_seasonality/verify
    (vp.HERE / "verify_tradable.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

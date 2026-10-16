#!/usr/bin/env python3
"""
adv_verify_robust -- ADVERSARIAL re-derivation of the claim-C5 robustness rows (the primary
rule, long h=8) with the verifier's own code (adv_verify_oi.py), from raw files:

  * crude_cl:   S/data/prices_daily.csv, market == "crude", closes > 0
  * crude_spot: S/data/wti-daily.csv, Price > 0
  * futures-only OI: S/ext/.../CFTC_CL.json (json.load only), last row 2026-09-29 dropped,
    traded on the verifier's own crude roll index (own last-trade dates)

Run:  python3 -I adv_verify_robust.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parents[1]        # battery code dir (tool_open_interest/, harness/)
sys.path.insert(0, str(W / "tool_open_interest"))
import adv_verify_oi as av  # noqa: E402

KEYS = ("n", "episodes", "T", "excess", "perm_p_exact", "period")


def series_from_csv(path: Path, date_col: str, val_col: str, market: str | None = None) -> pd.Series:
    df = pd.read_csv(path)
    if market is not None:
        df = df[df["market"] == market]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col]))
    s = s[s > 0].sort_index()
    return s[~s.index.duplicated(keep="last")]


def run(rep: pd.DataFrame, lvl: pd.Series, tag: str) -> None:
    f = av.primary_signal(rep, lvl)
    r = av.forward(f["report_date"], lvl)["r"].to_numpy()
    c = av.cell(f, r, +1)
    print(tag, {k: c[k] for k in KEYS})


if __name__ == "__main__":
    rep = av.load_cot("crude")
    cl = series_from_csv(av.S / "data" / "prices_daily.csv", "date", "close", "crude")
    spot = series_from_csv(av.S / "data" / "wti-daily.csv", "Date", "Price")
    run(rep, cl, "F+O on crude_cl  :")
    run(rep, spot, "F+O on crude_spot:")
    doc = json.load(open(av.SERIES_DIR / "CFTC_CL.json", encoding="utf-8"))
    ob = pd.DataFrame(doc["observations"])
    fut = pd.DataFrame({"report_date": pd.to_datetime(ob["date"]),
                        "oi": pd.to_numeric(ob["open_interest"]).astype(float)})
    fut = fut[fut["report_date"] != "2026-09-29"].sort_values("report_date").reset_index(drop=True)
    sd = pd.Series(False, index=fut.index)
    for a, b in av.SHUTDOWNS:
        sd |= (fut["report_date"] >= a) & (fut["report_date"] <= b)
    fut["shutdown"] = sd.to_numpy()
    lvl, *_ = av.build_roll_index()
    run(fut, lvl, "fut-only CFTC_CL on own crude_roll:")

#!/usr/bin/env python3
"""
basis_check_ext.py -- does the COT basis (futures-only vs futures+options) change the Larry
reading for crude in 2025-2026?

Our canonical data (S/data/cot_weekly.csv) is CFTC DISAGGREGATED FUTURES+OPTIONS. Web snippets
the user may be reading quote LEGACY FUTURES-ONLY numbers. We have no legacy file, but the
third-party archive S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json says it holds the
DISAGGREGATED FUTURES-ONLY crude series (2010-2026). This script:
  * parses that JSON as plain data (never runs anything from S/ext/; run with python3 -I),
  * builds comm_net_fo = (prod + swap) long - short and its 26-week Larry index with the
    engine's lc.larry_index,
  * compares it with our F+O index on the same report dates.
Its last row (2026-09-29) has exactly our F+O open interest and positions, i.e. that file
mixes bases in its final row, so the row is dropped. Treat everything here as an unverified
third-party cross-check, not as data for the main results.

Outputs: basis_check_ext.csv, basis_check_ext.json (work/lens_crude2026/).
Run:  python3 -I basis_check_ext.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[2])   # ROOT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))           # engine code
import larry_cot as lc  # noqa: E402

OUT = S / "work" / "lens_crude2026"
OUT.mkdir(parents=True, exist_ok=True)
EXT = S / "ext" / "accomplish999_crude-oil" / "data" / "series" / "CFTC_CL.json"


def main() -> int:
    raw = json.loads(EXT.read_text())
    x = pd.DataFrame(raw["observations"])
    x["report_date"] = pd.to_datetime(x["date"])
    for c in ["open_interest", "prod_long", "prod_short", "swap_long", "swap_short", "mm_net"]:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("int64")
    x = x.sort_values("report_date").reset_index(drop=True)

    ours = lc.build_panel(markets=["crude"])["crude"][["report_date", "oi", "comm_net", "mm_net",
                                                         "idx26"]]
    m = x.merge(ours, on="report_date", how="left", suffixes=("_ext", ""))
    # Rows whose open interest equals our F+O open interest are not futures-only: drop them.
    mixed = m["open_interest"] == m["oi"]
    mixed_dates = m.loc[mixed, "report_date"].dt.strftime("%Y-%m-%d").tolist()
    fo = m[~mixed].reset_index(drop=True)
    fo["comm_net_fo"] = (fo["prod_long"] + fo["swap_long"]) - (fo["prod_short"] + fo["swap_short"])
    fo["idx26_fo"] = lc.larry_index(fo["comm_net_fo"].to_numpy(), 26)

    sel = fo[fo["report_date"] >= "2025-01-01"].dropna(subset=["idx26_fo", "idx26"])
    lo_fo, sh_fo = lc.signal_masks(sel["idx26_fo"].to_numpy(), 80, 20)
    lo_us, sh_us = lc.signal_masks(sel["idx26"].to_numpy(), 80, 20)
    full = fo.dropna(subset=["idx26_fo", "idx26"])
    lo_fo_all, _ = lc.signal_masks(full["idx26_fo"].to_numpy(), 80, 20)
    lo_us_all, _ = lc.signal_masks(full["idx26"].to_numpy(), 80, 20)

    def at(d):
        r = fo[fo["report_date"] == d].iloc[0]
        return dict(report_date=d, comm_net_fo=int(r["comm_net_fo"]), idx26_fo=float(r["idx26_fo"]),
                    comm_net_fplus_o=int(r["comm_net"]), idx26_fplus_o=float(r["idx26"]))

    summary = dict(
        source=str(EXT.relative_to(S)), label="UNVERIFIED THIRD-PARTY CROSS-CHECK",
        n_rows_file=int(len(x)), file_first=x["report_date"].min().strftime("%Y-%m-%d"),
        file_last=x["report_date"].max().strftime("%Y-%m-%d"),
        rows_dropped_identical_to_our_fplus_o=mixed_dates,
        since_2025=dict(n_reports=int(len(sel)),
                        corr_idx26=float(np.corrcoef(sel["idx26_fo"], sel["idx26"])[0, 1]),
                        mean_abs_diff_idx26=float((sel["idx26_fo"] - sel["idx26"]).abs().mean()),
                        max_abs_diff_idx26=float((sel["idx26_fo"] - sel["idx26"]).abs().max()),
                        long_setup_weeks_fo=int(lo_fo.sum()), long_setup_weeks_fplus_o=int(lo_us.sum()),
                        long_setup_agree_share=float((lo_fo == lo_us).mean()),
                        short_setup_weeks_fo=int(sh_fo.sum()), short_setup_weeks_fplus_o=int(sh_us.sum())),
        full_overlap=dict(n_reports=int(len(full)),
                          corr_idx26=float(np.corrcoef(full["idx26_fo"], full["idx26"])[0, 1]),
                          long_setup_agree_share=float((lo_fo_all == lo_us_all).mean())),
        key_dates=[at(pd.Timestamp(d)) for d in ("2026-07-07", "2026-07-14", "2026-07-21",
                                                  "2026-09-15", "2026-09-22")],
    )
    for k in summary["key_dates"]:
        k["report_date"] = k["report_date"].strftime("%Y-%m-%d")
    out = fo.loc[fo["report_date"] >= "2025-01-01",
                 ["report_date", "open_interest", "oi", "comm_net_fo", "comm_net", "idx26_fo", "idx26"]]
    out = out.rename(columns={"open_interest": "oi_fo", "oi": "oi_fplus_o",
                              "comm_net": "comm_net_fplus_o", "idx26": "idx26_fplus_o"})
    out.to_csv(OUT / "basis_check_ext.csv", index=False, date_format="%Y-%m-%d")
    (OUT / "basis_check_ext.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

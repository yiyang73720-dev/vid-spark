#!/usr/bin/env python3
"""
audit_family_bh -- final auditor's independent check of the family-wide multiple-testing step
(rank/PREREG.md section 6) and of the R2/R3 inputs of the ranking.

1. Rebuilds the family from the source files (the five tool results.json `tests` + the v1 COT
   grid as published by the rank step), compares it row-for-row with
   rank/family_tests_with_q.csv (row count, p_used, p source).
2. Checks the v1 COT rows' +1-corrected p against the v1 file canonical_exact_p.csv: from
   p_used = (1 + c) / (1 + N) and raw = c / N the implied number of offsets N must be a
   positive integer and identical for the long and short rows of the same cell.
3. Re-implements Benjamini-Hochberg (as "largest k with p_(k) <= k q / m", no shared code) and
   Benjamini-Yekutieli, and recounts survivors: all rows, h=8 rows, the 6 primaries, within
   each tool; plus the resolution floor (rows needed at the smallest attainable p).
4. POST-HOC (not pre-registered): a strategy-level family -- the random-timing null p of the
   S1/S2 x long-only/long+short strategies of every tool's primary (rank/primary_table.csv),
   with and without the confluence primary's four strategies.

Run:  python3 -I audit_family_bh.py   (writes ROOT/work2/audit_final/audit_family_bh.json)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json                       # noqa: E402
import os                         # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
W2 = ROOT / "work2"
RANK = W2 / "rank"
TOOLS = {"premium": "tool_premium", "seasonality": "tool_seasonality",
         "valuation": "tool_valuation", "open_interest": "tool_open_interest",
         "intermarket": "tool_intermarket"}
Q = 0.10
OUT = W2 / "audit_final" / "audit_family_bh.json"


def bh_reject(p: np.ndarray, q: float = Q) -> np.ndarray:
    """Benjamini-Hochberg step-up: reject the k smallest p where k is the largest index with
    p_(k) <= k q / m."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ok = p[order] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if ok.any():
        k = int(np.max(np.flatnonzero(ok))) + 1
        rej[order[:k]] = True
    return rej


def bh_q(p: np.ndarray) -> np.ndarray:
    """BH q-values: q_(i) = min_{j >= i} p_(j) m / j, capped at 1."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum(1.0, np.minimum.accumulate(adj[::-1])[::-1])
    out = np.empty(m)
    out[order] = adj
    return out


def main() -> None:
    res: dict = {}
    # ---------------------------------------------------------------- 1. rebuild the family
    parts = []
    for tool, d in TOOLS.items():
        t = pd.DataFrame(json.load(open(W2 / d / "results.json", encoding="utf-8"))["tests"])
        t["tool"] = tool
        parts.append(t)
    pub = pd.read_csv(RANK / "family_tests_with_q.csv", low_memory=False,
                      float_precision="round_trip")
    cot = pub[pub["tool"] == "cot"].copy()          # v1 grid as recomputed by the rank step
    fam = pd.concat(parts + [cot], ignore_index=True)
    ex = pd.to_numeric(fam["perm_p_exact"], errors="coerce")
    mc = pd.to_numeric(fam["perm_p"], errors="coerce")
    fam["p_mine"] = np.where(ex.notna(), ex, mc)
    fam = fam[fam["p_mine"].notna()].reset_index(drop=True)
    res["rebuilt_rows_with_p"] = int(len(fam))
    res["published_rows_with_p"] = int(len(pub))
    res["rows_per_tool"] = fam["tool"].value_counts().to_dict()
    res["p_identical_to_published_sorted"] = bool(np.array_equal(
        np.sort(fam["p_mine"].to_numpy()), np.sort(pub["p_used"].to_numpy())))
    res["mc_p_rows"] = int((ex.isna() & mc.notna()).sum())

    # ---------------------------------------------------------------- 2. v1 COT +1 correction
    ref = pd.read_csv(ROOT / "work" / "engine" / "canonical_exact_p.csv",
                      float_precision="round_trip")
    v = cot.assign(lookback=cot["variant"].str.extract(r"N(\d+)_")[0].astype(int),
                   buy_th=cot["variant"].str.extract(r"_(\d+)/")[0].astype(float))
    j = v.merge(ref, on=["market", "lookback", "buy_th", "side", "horizon_w"],
                suffixes=("", "_v1"))
    N = (1 - j["p_used"]) / (j["p_used"] - j["perm_p_exact_v1"])
    j["N_implied"] = N
    cons = j.groupby(["market", "lookback", "buy_th", "horizon_w"])["N_implied"].agg(
        lambda s: float(s.max() - s.min()))
    res["v1_plus1_check"] = dict(
        rows_merged=int(len(j)),
        p_used_gt_raw=bool((j["p_used"] > j["perm_p_exact_v1"]).all()),
        N_integer_max_dev=float(np.nanmax(np.abs(N - np.round(N)))),
        N_same_for_long_and_short_max_range=float(cons.max()),
        N_range=[float(N.min()), float(N.max())])

    # ---------------------------------------------------------------- 3. BH / BY recount
    p = fam["p_mine"].to_numpy()
    m = len(p)
    rej_all = bh_reject(p)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    rej_by = bh_reject(p, Q / c_m)
    h8 = (fam["horizon_w"] == 8).to_numpy()
    rej_h8 = bh_reject(p[h8])
    prim = fam["is_primary"].fillna(False).astype(bool).to_numpy()
    prim_tab = fam.loc[prim, ["tool", "variant", "market", "side", "excess", "p_mine"]].copy()
    prim_tab["q_bh_6"] = bh_q(prim_tab["p_mine"].to_numpy())
    within = {}
    for tool, g in fam.groupby("tool"):
        within[tool] = dict(rows=int(len(g)), p_le_005=int((g["p_mine"] <= 0.05).sum()),
                            expected_005=round(0.05 * len(g), 1),
                            within_tool_bh_survivors=int(bh_reject(g["p_mine"].to_numpy()).sum()),
                            min_p=float(g["p_mine"].min()))
    p_min = float(p.min())
    q_mine = bh_q(p)
    pub_sorted = pub.sort_values(["tool", "variant", "market", "series", "side", "horizon_w"])
    res["bh"] = dict(
        m=m, survivors_all=int(rej_all.sum()), survivors_by=int(rej_by.sum()),
        h8_rows=int(h8.sum()), survivors_h8=int(rej_h8.sum()),
        min_q_all=float(q_mine.min()),
        published_min_q_all=float(pub["q_bh_all"].min()),
        p_min=p_min, rows_needed_at_p_min=int(np.ceil(p_min * m / Q)),
        rows_at_p_min=int(np.sum(p <= p_min + 1e-15)),
        primaries=prim_tab.to_dict("records"), per_tool=within)

    # ---------------------------------------------------------------- 4. strategy family
    pt = pd.read_csv(RANK / "primary_table.csv", float_precision="round_trip")
    rows = []
    for _, r in pt.iterrows():
        for s in ("S1_LO", "S1_LS", "S2_LO", "S2_LS"):
            rows.append(dict(tool=r["tool"], strategy=s, sharpe=float(r[f"{s}_sharpe"]),
                             bh_sharpe=float(r["bh_sharpe"]), null_p=float(r[f"{s}_null_p"]),
                             series=r["strat_series"]))
    st = pd.DataFrame(rows)
    st["q_bh"] = bh_q(st["null_p"].to_numpy())
    st["bh_survivor_q10"] = bh_reject(st["null_p"].to_numpy())
    conf = json.load(open(RANK / "confluence_results.json", encoding="utf-8"))["strategy"]
    conf = pd.DataFrame(conf)
    conf = conf[conf["strategy"] != "buy_hold"]
    st2 = pd.concat([st[["tool", "strategy", "sharpe", "bh_sharpe", "null_p", "series"]],
                     pd.DataFrame(dict(tool="confluence", strategy=conf["strategy"].to_numpy(),
                                       sharpe=conf["sharpe"].to_numpy(float),
                                       bh_sharpe=conf["bh_sharpe"].to_numpy(float),
                                       null_p=conf["null_p_sharpe"].to_numpy(float),
                                       series="crude_roll"))], ignore_index=True)
    st2["q_bh"] = bh_q(st2["null_p"].to_numpy())
    st2["bh_survivor_q10"] = bh_reject(st2["null_p"].to_numpy())
    res["strategy_family_posthoc"] = dict(
        note="POST-HOC auditor family, not pre-registered: random-timing null p of the four "
             "S1/S2 strategies of each primary (BH q=0.10).",
        six_primaries=st.to_dict("records"),
        survivors_six=st.loc[st["bh_survivor_q10"], ["tool", "strategy", "null_p", "q_bh"]
                             ].to_dict("records"),
        survivors_with_confluence=st2.loc[st2["bh_survivor_q10"],
                                          ["tool", "strategy", "null_p", "q_bh"]
                                          ].to_dict("records"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rebuilt_rows_with_p", "published_rows_with_p",
                                          "rows_per_tool", "p_identical_to_published_sorted",
                                          "mc_p_rows", "v1_plus1_check")}, indent=1))
    b = res["bh"]
    print({k: b[k] for k in ("m", "survivors_all", "survivors_by", "h8_rows", "survivors_h8",
                             "min_q_all", "published_min_q_all", "p_min",
                             "rows_needed_at_p_min", "rows_at_p_min")})
    print(pd.DataFrame(b["primaries"]).to_string())
    print(pd.DataFrame(b["per_tool"]).T.to_string())
    print(st.sort_values("null_p").head(8).to_string())
    print("with confluence survivors:", res["strategy_family_posthoc"]["survivors_with_confluence"])


if __name__ == "__main__":
    main()

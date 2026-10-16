#!/usr/bin/env python3
"""
audit_confluence -- final auditor's look-ahead and data-snooping check of the confluence test
(rank/PREREG.md sections 1-4; outputs rank/confluence_weekly_crude.csv and
rank/confluence_results.json).

A. Timing. For every common week: the Friday whose close feeds the Friday components lies on or
   after the report date and strictly before the entry close (first crude_roll day on/after the
   Tuesday after the report week), and the COT publication Friday is before the entry.
B. Components rebuilt FROM RAW FILES with the auditor's own code (no tool library):
     cot      commercials' net (prod/merc + swap) from S/data/cot_weekly.csv, 26-report range
              index incl. the current report, >= 80 -> +1, <= 20 -> -1;
     premium  sign(c1 - c2) as of the Friday (audit_premium_independent.friday_signal);
     dollar   DTWEXBGS from the raw JSON, x(F-7d) / x(F-7d-56d) - 1 (prints at most 4 days
              old), fell -> +1, rose -> -1.
C. TRUNCATION tests for the components whose code is more involved: every input is cut at the
   decision time and the component is recomputed with the tool's own library; the value must
   equal the full-sample value (any difference = information from the future):
     seasonal   crude_spot truncated at the Friday, analog table rebuilt, signal at the Friday;
     valuation  the weekly VI inputs truncated at the Friday row, VI recomputed;
     oi         COT rows and the crude_roll series truncated at the report date.
   The valuation inputs themselves (CL and London gold as of Friday) are checked against
   S/data/prices_daily.csv read independently.
D. Statistics recomputed from the weekly table: the confluence column from the six components,
   the primary long h=8 n / T / excess / hit / exhaustive p (+1), D_X against each component
   and its one-sided circular block-bootstrap p (13-week blocks, B=2000, default_rng(11)).

Run:  python3 -I audit_confluence.py   (writes ROOT/work2/audit_final/audit_confluence.json)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json                       # noqa: E402
import os                         # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent                   # code (audit_premium_independent)
PKG = Path(__file__).resolve().parents[2]                # package dir (battery/<step>/../..)
ROOT = Path(os.environ.get("LARRY_COT_ROOT") or PKG)     # data/ ext/ ext2/ work/ work2/
RANK = ROOT / "work2" / "rank"                           # outputs of the rank step
sys.path.insert(0, str(HERE))
import audit_premium_independent as ap   # noqa: E402  (auditor's raw loaders)

COMPS = ["cot", "premium", "seasonal", "valuation", "oi", "dollar"]
OUT = ROOT / "work2" / "audit_final" / "audit_confluence.json"


def friday_of_week(d: pd.Series) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(d)
    return d + pd.to_timedelta(4 - d.weekday.to_numpy(), unit="D")


# ---------------------------------------------------------------------------------- B. raw
def cot_raw(report_dates: pd.DatetimeIndex) -> np.ndarray:
    """v1 Larry COT index N=26 of crude commercials (F+O disaggregated), +1/-1/0/NaN."""
    c = pd.read_csv(ROOT / "data" / "cot_weekly.csv", parse_dates=["report_date"])
    c = c[c["market"] == "crude"].sort_values("report_date").reset_index(drop=True)
    net = (c["prod_merc_long"] + c["swap_long"]) - (c["prod_merc_short"] + c["swap_short"])
    lo, hi = net.rolling(26, min_periods=26).min(), net.rolling(26, min_periods=26).max()
    idx = 100.0 * (net - lo) / (hi - lo).where(hi > lo)
    sig = np.where(idx >= 80, 1.0, np.where(idx <= 20, -1.0, 0.0))
    sig = np.where(idx.notna(), sig, np.nan)
    return pd.Series(sig, index=c["report_date"]).reindex(report_dates).to_numpy()


def dollar_raw(fridays: pd.DatetimeIndex) -> np.ndarray:
    s = ap.load_eia_json("DTWEXBGS")
    s = s[s > 0]

    def asof(d):
        pos = np.searchsorted(s.index.values, d.values, side="right") - 1
        out = np.full(len(d), np.nan)
        ok = pos >= 0
        age = (d.values[ok] - s.index.values[pos[ok]]) / np.timedelta64(1, "D")
        good = np.flatnonzero(ok)[age <= 4]
        out[good] = s.to_numpy()[pos[good]]
        return out
    a = asof(fridays - pd.Timedelta(days=7))
    b = asof(fridays - pd.Timedelta(days=7 + 56))
    chg = a / b - 1.0
    return np.where(np.isnan(chg), np.nan, -np.sign(chg) + 0.0)


# ----------------------------------------------------------------------- C. truncation tests
def truncation_tests(wk: pd.DataFrame) -> dict:
    """Recompute seasonal / valuation / oi with inputs cut at the decision time."""
    sys.path.insert(0, str(PKG / "battery" / "rank"))
    import rank_lib as rl                 # tool libraries, read-only reuse
    fh, se, vl, ol = rl.fh, rl.se, rl.vl, rl.ol
    out = {}
    fri = friday_of_week(wk["report_date"])

    # seasonal: crude_spot cut at the Friday, L=15, h=8, 70/30 (tool_seasonality primary)
    spot = fh.load_series("crude_spot")
    full_tab = se.est_table(spot, "crude_spot", (8,))
    full = se.seasonal_signal(pd.DatetimeIndex(fri), full_tab, 8, 15, 70, "week")
    trunc = np.full(len(fri), np.nan)
    for i, f in enumerate(fri):
        tab = se.est_table(spot[spot.index <= f], "crude_spot_cut", (8,))
        trunc[i] = se.seasonal_signal(pd.DatetimeIndex([f]), tab, 8, 15, 70, "week")[0]
    out["seasonal"] = dict(weeks=len(fri), mismatches_vs_truncated=int(np.sum(
        ~((trunc == full) | (np.isnan(trunc) & np.isnan(full))))),
        mismatches_vs_weekly_file=int(np.sum(full != wk["seasonal"].to_numpy())))

    # valuation: VI(crude vs gold) recomputed on the weekly inputs cut at each Friday row
    vfri = vl.friday_grid()
    m = vl.market_inputs(vfri)["crude"]
    c = vl.comparator_inputs(vfri)["gold"]
    vi_full = vl.valuation_index(m, c, (2, 22), 156)["vi"]
    rows = vfri.get_indexer(pd.DatetimeIndex(fri))
    mism = 0
    for i in rows:
        vi_cut = vl.valuation_index(m[:i + 1], c[:i + 1], (2, 22), 156)["vi"][-1]
        same = (vi_cut == vi_full[i]) or (np.isnan(vi_cut) and np.isnan(vi_full[i]))
        mism += int(not same)
    sig_full = vl.setups(vi_full, 20.0, 80.0)[rows]
    # inputs vs the raw daily file read independently (CL continuous and London gold)
    px = pd.read_csv(ROOT / "data" / "prices_daily.csv", parse_dates=["date"])
    inp_dev = {}
    for mk, arr in (("crude", m), ("gold", c)):
        s = px[(px["market"] == mk) & (px["close"] > 0)].set_index("date")["close"].sort_index()
        pos = np.searchsorted(s.index.values, vfri[rows].values, side="right") - 1
        raw = s.to_numpy()[pos]
        inp_dev[mk] = float(np.nanmax(np.abs(arr[rows] / raw - 1.0)))
        inp_dev[mk + "_last_input_date_after_friday"] = int(np.sum(s.index.values[pos] >
                                                                   vfri[rows].values))
    out["valuation"] = dict(weeks=len(rows), vi_mismatches_vs_truncated=mism,
                            mismatches_vs_weekly_file=int(np.sum(
                                sig_full != wk["valuation"].to_numpy())),
                            input_max_rel_dev_vs_raw=inp_dev)

    # oi: COT rows and crude_roll cut at the report date
    fo = ol.load_fo_cot()["crude"]
    roll = fh.load_series("crude_roll")
    feats = ol.build_features(fo, roll)
    sig_all = pd.Series(ol.rule_primary(feats), index=pd.DatetimeIndex(fo["report_date"]))
    full_oi = sig_all.reindex(pd.DatetimeIndex(wk["report_date"])).to_numpy()
    trunc_oi = np.full(len(wk), np.nan)
    for i, d in enumerate(pd.DatetimeIndex(wk["report_date"])):
        rep = fo[fo["report_date"] <= d].reset_index(drop=True)
        f = ol.build_features(rep, roll[roll.index <= d])
        trunc_oi[i] = ol.rule_primary(f)[-1]
    out["oi"] = dict(weeks=len(wk), mismatches_vs_truncated=int(np.sum(
        ~((trunc_oi == full_oi) | (np.isnan(trunc_oi) & np.isnan(full_oi))))),
        mismatches_vs_weekly_file=int(np.sum(full_oi != wk["oi"].to_numpy())))
    return out


# ------------------------------------------------------------------------------ D. statistics
def long_excess(sig: np.ndarray, r: np.ndarray) -> float:
    m = sig == 1
    return float(r[m].mean() - r.mean()) if m.any() else np.nan


def exact_p(sig: np.ndarray, r: np.ndarray, min_shift: int = 26) -> float:
    T = len(r)
    pos = np.flatnonzero(sig == 1)
    obs = r[pos].mean()
    offs = np.arange(min_shift, T - min_shift + 1)
    sh = r[(pos[None, :] + offs[:, None]) % T].mean(axis=1)
    return float((1 + np.sum(sh >= obs)) / (1 + len(offs)))


def boot_diff_p(a: np.ndarray, b: np.ndarray, r: np.ndarray, block: int = 13, B: int = 2000,
                seed: int = 11) -> float:
    """One-sided p of 'a beats b': (1 + #{D* <= 0}) / (1 + #valid), circular blocks."""
    T = len(r)
    nb = int(np.ceil(T / block))
    starts = np.random.default_rng(seed).integers(0, T, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(B, -1)[:, :T]
    d = np.array([long_excess(a[i], r[i]) - long_excess(b[i], r[i]) for i in idx])
    d = d[~np.isnan(d)]
    return float((1 + np.sum(d <= 0)) / (1 + len(d)))


def main() -> None:
    wk = pd.read_csv(RANK / "confluence_weekly_crude.csv", parse_dates=["report_date"],
                     float_precision="round_trip")
    pub = json.load(open(RANK / "confluence_results.json", encoding="utf-8"))
    res: dict = {}

    # A. timing
    rep = pd.DatetimeIndex(wk["report_date"])
    fri = friday_of_week(wk["report_date"])
    raw = ap.load_contracts()
    clean, _ = ap.drop_placeholder_rows(raw)
    v = clean[clean["c1"].notna() & clean["c2"].notna()]
    tgt = rep - pd.to_timedelta(rep.weekday.to_numpy(), unit="D") + pd.Timedelta(days=8)
    entry = v.index[np.searchsorted(v.index.values, tgt.values, side="left")]
    res["timing"] = dict(weeks=len(wk), friday_ge_report=bool((fri >= rep).all()),
                         friday_lt_entry=bool((fri < entry).all()),
                         min_days_friday_to_entry=int((entry - fri).days.min()),
                         report_weekdays=pd.Series(rep.weekday).value_counts().to_dict())

    # B. raw recomputation
    sig_prem = ap.friday_signal(v, fri)
    sig_cot = cot_raw(rep)
    sig_usd = dollar_raw(fri)
    res["raw_recomputation_mismatches"] = {
        "premium": int(np.sum(sig_prem != wk["premium"].to_numpy())),
        "cot": int(np.sum(sig_cot != wk["cot"].to_numpy())),
        "dollar": int(np.sum(sig_usd != wk["dollar"].to_numpy()))}

    # C. truncation
    res["truncation_tests"] = truncation_tests(wk)

    # D. statistics
    A = wk[COMPS].to_numpy(float)
    L, S = (A == 1).sum(1), (A == -1).sum(1)
    conf = np.where((L >= 2) & (L > S), 1.0, np.where((S >= 2) & (S > L), -1.0, 0.0))
    r = wk["r8"].to_numpy(float)
    res["confluence_column_mismatches"] = int(np.sum(conf != wk["confluence"].to_numpy()))
    prim = dict(n=int((conf == 1).sum()), T=len(r), excess=long_excess(conf, r),
                hit=float(np.mean(r[conf == 1] > 0)),
                perm_p_exact=exact_p(conf, r))
    res["primary_recomputed"] = prim
    res["primary_published"] = {k: pub["primary"][k] for k in ("n", "T", "excess", "hit",
                                                                "perm_p_exact")}
    hb = {}
    for c in COMPS:
        x = wk[c].to_numpy(float)
        d = long_excess(conf, r) - long_excess(x, r)
        p = boot_diff_p(conf, x, r)
        P = pub["H_b_beats_each_component"][c]
        hb[c] = dict(n_long=int((x == 1).sum()), component_excess=long_excess(x, r), D=d,
                     p_boot=p, published_D=P["diff"], published_p=P["p_boot"])
    res["H_b"] = hb
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

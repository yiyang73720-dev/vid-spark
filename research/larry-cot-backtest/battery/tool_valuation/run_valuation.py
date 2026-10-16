#!/usr/bin/env python3
"""
run_valuation -- backtest of Larry Williams' Valuation Index (WillVal) for the SPEC2 battery.

Follows PREREG.md (sha256 in PREREG.sha256, written before this script was run):
  primary   crude vs gold, EMA(2)-EMA(22), N=156 weeks, 20/80, LONG (VI <= 20), h=8 weeks,
            traded on crude_roll, entry at the first close after the Friday
  grid A    every market/comparator pair + composites at the primary parameters, h 1/4/8/13
  grid B    N x detrending x thresholds for every pair and composite, h=8
  grid C    primary on other crude series / crude_spot as valuation input
  grid D    entry-on-cross variant
  grid E    extended history (crude 1985/1989..2024)
  grid F    pooled cells
  OOS       split halves; walk-forward (identity + parameter-selection)
  strategy  S1/S2 with costs, random-timing null, buy & hold
Outputs (this directory): results.json, tests.csv, trades_primary.csv, run_log.txt.

Run:  python3 -I run_valuation.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent         # this tool's code and PREREG.md
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True
import valuation_lib as vl  # noqa: E402
from valuation_lib import fh  # noqa: E402
HERE = fh.WORK2 / "tool_valuation"             # outputs: ROOT/work2/tool_valuation
HERE.mkdir(parents=True, exist_ok=True)

T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


# -------------------------------------------------------------------------------------------
# Set-up
# -------------------------------------------------------------------------------------------
HORIZONS = (1, 4, 8, 13)
SIDES3 = ("long", "short", "net")
MARKETS4 = ("crude", "silver", "soybean", "sugar")
COMPS = ("gold", "bond", "usd")
TRADED = {"crude": "crude_roll", "gold": "gold", "silver": "silver", "soybean": "soybean",
          "sugar": "sugar"}
INPUT_NAME = {"crude": "crude_cl", "gold": "gold", "silver": "silver", "soybean": "soybean",
              "sugar": "sugar"}
COMP_NAME = {"gold": "gold (v1 London spot)", "bond": "ust10_price (DGS10, 6% coupon, 10y)",
             "usd": "usd_broad (DTWEXBGS, read as of Friday-7d)"}
PAIRS = [(m, c) for m in MARKETS4 for c in COMPS] + [("gold", "bond"), ("gold", "usd")]
P = vl.PRIMARY
EMA_KEY = {v: k for k, v in vl.EMAS.items()}            # (2,22) -> "ema2_22", None -> "none"


def cfg_name(ema_key: str, N: int, b: float, s: float) -> str:
    return f"{ema_key}|N{N}|{b:g}/{s:g}"


PRIMARY_CFG = cfg_name("ema2_22", P["N"], P["buy_th"], P["sell_th"])

fri = vl.friday_grid()
MIN = vl.market_inputs(fri)
CIN = vl.comparator_inputs(fri)

# Valuation indices, cached per (market, comparator, ema_key, N)
VI: dict[tuple, np.ndarray] = {}
for (m, c) in PAIRS:
    for ek, ema in vl.EMAS.items():
        for N in vl.LOOKBACKS:
            VI[(m, c, ek, N)] = vl.valuation_index(MIN[m], CIN[c], ema, N)["vi"]
log(f"computed {len(VI)} valuation-index series on {len(fri)} Fridays")

ALL_ROWS: list[pd.DataFrame] = []


def run_eval(sig: np.ndarray, series: str, *, market: str, variant: str, grid: str,
             comparator: str, horizons=HORIZONS, sides=SIDES3, extra: dict | None = None,
             frame_dates=None) -> pd.DataFrame:
    """Signal array on the Friday grid -> harness evaluate rows with our labels."""
    sf = fh.make_signal_frame(fri if frame_dates is None else frame_dates, sig)
    labels = dict(filter="valuation", variant=variant, market=market, is_primary=False,
                  grid=grid, comparator=comparator, **(extra or {}))
    rows = fh.evaluate(sf, series, horizons, timing="price", sides=sides, labels=labels)
    ALL_ROWS.append(rows)
    return rows


def cfg_extra(ema_key, N, b, s, input_name):
    return dict(ema=ema_key, lookback=N, buy_th=b, sell_th=s, input=input_name)


# -------------------------------------------------------------------------------------------
# Grid A: every pair and the composites at the primary parameters
# -------------------------------------------------------------------------------------------
SF: dict[str, pd.DataFrame] = {}          # signal frames kept for OOS / strategy / pooling
for (m, c) in PAIRS:
    vi = VI[(m, c, "ema2_22", P["N"])]
    sig = vl.setups(vi, P["buy_th"], P["sell_th"])
    variant = f"vs_{c}|{PRIMARY_CFG}"
    SF[f"{m}|{c}"] = fh.make_signal_frame(fri, sig)
    run_eval(sig, TRADED[m], market=m, variant=variant, grid="A", comparator=c,
             extra=cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], INPUT_NAME[m]))
log("grid A pairs done")


def composite_signals(m: str, ek: str, N: int, b: float, s: float) -> dict[str, np.ndarray]:
    vis = [VI[(m, c, ek, N)] for c in COMPS]
    return {"vote2of3": vl.composite_vote(vis, b, s, 2),
            "avg3": vl.setups(vl.composite_avg(vis), b, s)}


for m in MARKETS4:
    for kind, sig in composite_signals(m, "ema2_22", P["N"], P["buy_th"], P["sell_th"]).items():
        SF[f"{m}|{kind}"] = fh.make_signal_frame(fri, sig)
        run_eval(sig, TRADED[m], market=m, variant=f"{kind}|{PRIMARY_CFG}", grid="A",
                 comparator=kind,
                 extra=cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], INPUT_NAME[m]))
gsig = vl.composite_vote([VI[("gold", "bond", "ema2_22", P["N"])],
                          VI[("gold", "usd", "ema2_22", P["N"])]], P["buy_th"], P["sell_th"], 2)
SF["gold|both2"] = fh.make_signal_frame(fri, gsig)
run_eval(gsig, "gold", market="gold", variant=f"both2of2|{PRIMARY_CFG}", grid="A",
         comparator="both2of2",
         extra=cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], "gold"))
log("grid A composites done")

# -------------------------------------------------------------------------------------------
# Grid B: N x detrending x thresholds, h = 8 (the primary parameter set is in grid A)
# -------------------------------------------------------------------------------------------
nB = 0
for ek in vl.EMAS:
    for N in vl.LOOKBACKS:
        for (b, s) in vl.THRESHOLDS:
            cfg = cfg_name(ek, N, b, s)
            if cfg == PRIMARY_CFG:
                continue
            ex = cfg_extra(ek, N, b, s, None)
            for (m, c) in PAIRS:
                sig = vl.setups(VI[(m, c, ek, N)], b, s)
                run_eval(sig, TRADED[m], market=m, variant=f"vs_{c}|{cfg}", grid="B",
                         comparator=c, horizons=(8,), extra={**ex, "input": INPUT_NAME[m]})
                nB += 1
            for m in MARKETS4:
                for kind, sig in composite_signals(m, ek, N, b, s).items():
                    run_eval(sig, TRADED[m], market=m, variant=f"{kind}|{cfg}", grid="B",
                             comparator=kind, horizons=(8,), extra={**ex, "input": INPUT_NAME[m]})
                    nB += 1
log(f"grid B done ({nB} signal frames)")

# -------------------------------------------------------------------------------------------
# Grid C: the primary rule on other crude series; crude_spot as the valuation input
# -------------------------------------------------------------------------------------------
prim_sig = vl.setups(VI[("crude", "gold", "ema2_22", P["N"])], P["buy_th"], P["sell_th"])
ex_prim = cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], "crude_cl")
for ser in ("crude_roll_lead5", "crude_cl", "crude_spot"):
    run_eval(prim_sig, ser, market="crude", variant=f"vs_gold|{PRIMARY_CFG}|traded_{ser}",
             grid="C", comparator="gold", extra=ex_prim)
spot_in = vl.asof_fresh(fh.load_series("crude_spot"), fri)
vi_spot = vl.valuation_index(spot_in, CIN["gold"], (2, 22), P["N"])["vi"]
run_eval(vl.setups(vi_spot, P["buy_th"], P["sell_th"]), "crude_roll", market="crude",
         variant=f"vs_gold|{PRIMARY_CFG}|input_crude_spot", grid="C", comparator="gold",
         extra=cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], "crude_spot"))
log("grid C done")

# -------------------------------------------------------------------------------------------
# Grid D: entry on the cross back out of the extreme zone
# -------------------------------------------------------------------------------------------
cross_sig = vl.cross_setups(VI[("crude", "gold", "ema2_22", P["N"])], P["buy_th"], P["sell_th"])
run_eval(cross_sig, "crude_roll", market="crude", variant=f"vs_gold|{PRIMARY_CFG}|cross",
         grid="D", comparator="gold", extra=ex_prim)
log("grid D done")

# -------------------------------------------------------------------------------------------
# Grid E: extended history (crude input = RCLC1; gold = pysystemtrade GOLD held contract)
# -------------------------------------------------------------------------------------------
rclc1 = vl.asof_fresh(vl.load_rclc1(), fri)
gold_ps = vl.asof_fresh(vl.load_psys_gold(), fri)
vi_ext_gold = vl.valuation_index(rclc1, gold_ps, (2, 22), P["N"])["vi"]
vi_ext_bond = vl.valuation_index(rclc1, CIN["bond"], (2, 22), P["N"])["vi"]
sig_ext_gold = vl.setups(vi_ext_gold, P["buy_th"], P["sell_th"])
sig_ext_bond = vl.setups(vi_ext_bond, P["buy_th"], P["sell_th"])
ex_ext = cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"], "RCLC1")
run_eval(sig_ext_gold, "crude_roll", market="crude", variant=f"vs_gold_psys|{PRIMARY_CFG}|extended",
         grid="E", comparator="gold_psys", extra=ex_ext)
run_eval(sig_ext_bond, "crude_roll", market="crude", variant=f"vs_bond|{PRIMARY_CFG}|extended",
         grid="E", comparator="bond", extra=ex_ext)
SF["crude|gold_ext"] = fh.make_signal_frame(fri, sig_ext_gold)
ext_split = fh.split_half(SF["crude|gold_ext"], "crude_roll", timing="price", horizon=8,
                          sides=SIDES3)
log("grid E done")

# -------------------------------------------------------------------------------------------
# Grid F: pooled cells across the four markets
# -------------------------------------------------------------------------------------------
pooled = []
for key in ("gold", "bond", "usd", "vote2of3"):
    parts = [(SF[f"{m}|{key}"], TRADED[m], "price") for m in MARKETS4]
    pr = fh.evaluate_pooled(parts, HORIZONS, sides=("long", "short"),
                            labels=dict(filter="valuation", market="pooled",
                                        variant=f"pooled4_{'vs_' + key if key in COMPS else key}"
                                                f"|{PRIMARY_CFG}",
                                        is_primary=False, grid="F", comparator=key,
                                        **cfg_extra("ema2_22", P["N"], P["buy_th"], P["sell_th"],
                                                    "crude_cl+silver+soybean+sugar")))
    pooled.append(pr)
POOLED = pd.concat(pooled, ignore_index=True)
ALL_ROWS.append(POOLED)
log("grid F (pooled) done")

# -------------------------------------------------------------------------------------------
# Assemble the tests table, flag the primary row, BH q-values (information only)
# -------------------------------------------------------------------------------------------
tests = pd.concat(ALL_ROWS, ignore_index=True)
prim_mask = ((tests["grid"] == "A") & (tests["market"] == "crude") & (tests["comparator"] == "gold")
             & (tests["series"] == "crude_roll") & (tests["side"] == "long")
             & (tests["horizon_w"] == 8))
assert prim_mask.sum() == 1, prim_mask.sum()
tests["is_primary"] = prim_mask


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q) for the finite entries of p."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok], kind="mergesort")]
    m = len(order)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


h8 = (tests["horizon_w"] == 8).to_numpy()
tests["bh_q_h8"] = np.nan
tests.loc[h8, "bh_q_h8"] = bh_qvalues(tests.loc[h8, "perm_p"].to_numpy())
PRIMARY_ROW = tests[prim_mask].iloc[0].to_dict()
log(f"PRIMARY: n={PRIMARY_ROW['n']} episodes={PRIMARY_ROW['episodes']} mean={PRIMARY_ROW['mean']:.6f} "
    f"baseline={PRIMARY_ROW['baseline']:.6f} excess={PRIMARY_ROW['excess']:.6f} "
    f"hit={PRIMARY_ROW['hit']:.4f} perm_p={PRIMARY_ROW['perm_p']:.4f} "
    f"perm_p_exact={PRIMARY_ROW['perm_p_exact']:.4f} period={PRIMARY_ROW['period']}")

# -------------------------------------------------------------------------------------------
# OOS: split halves and walk-forward
# -------------------------------------------------------------------------------------------
PSF = SF["crude|gold"]
split = fh.split_half(PSF, "crude_roll", timing="price", horizon=8, sides=SIDES3)
for _, r in split[split["side"] == "long"].iterrows():
    log(f"split {r['half']} (split {r['split_date']}): n={r['n']} excess={r['excess']:.6f} "
        f"perm_p={r['perm_p']:.4f} period={r['period']}")

WF_START = "2008-01-01"
# (1) identity walk-forward: the primary has no fitted parameter
wf_frame = pd.DataFrame({"decision_date": fri, "sig": prim_sig})
wf_id = fh.walk_forward(wf_frame, fit=lambda hist, cutoff: None,
                        apply=lambda params, block: block["sig"].to_numpy(), first_test=WF_START)
wf_id_rows = fh.evaluate(wf_id, "crude_roll", (8,), timing="price", sides=SIDES3,
                         labels=dict(filter="valuation", variant="walk_forward_identity",
                                     market="crude", is_primary=False))

# (2) parameter-selection walk-forward over the 60 crude-vs-gold configurations
CFGS = []
sel = pd.DataFrame({"decision_date": fri})
for ek in vl.EMAS:
    for N in vl.LOOKBACKS:
        for (b, s) in vl.THRESHOLDS:
            name = cfg_name(ek, N, b, s)
            CFGS.append(name)
            sel[name] = vl.setups(VI[("crude", "gold", ek, N)], b, s)
ft = fh.forward_table(fri, "crude_roll", "price", (8,))
sel["r8"] = ft["r8"].to_numpy()
sel["exit8"] = ft["exit_date8"].to_numpy()
SEL_LOG: list[dict] = []
MIN_SIG_WEEKS = 26


def wf_fit(hist: pd.DataFrame, cutoff: pd.Timestamp):
    """Pick the configuration with the highest long-side h=8 excess among weeks whose 8-week
    exit is before the cutoff (no outcome after the cutoff is ever seen)."""
    h = hist[(hist["exit8"] < cutoff) & hist["r8"].notna()]
    best, best_ex, best_n = None, 0.0, 0
    for name in CFGS:
        s = h[name]
        el = s.notna()
        if not el.any():
            continue
        lw = (s == 1.0).to_numpy()
        n = int(lw.sum())
        if n < MIN_SIG_WEEKS:
            continue
        ex = float(h["r8"][lw].mean() - h["r8"][el].mean())
        if ex > best_ex:
            best, best_ex, best_n = name, ex, n
    SEL_LOG.append(dict(cutoff=cutoff.date().isoformat(), chosen=best,
                        past_excess=best_ex if best else None, past_n=best_n if best else None,
                        past_weeks_with_realised_r8=int(len(h))))
    return best


def wf_apply(params, block: pd.DataFrame) -> np.ndarray:
    if params is None:      # no trade: valid (0) wherever the primary index is defined
        return np.where(block[PRIMARY_CFG].notna(), 0.0, np.nan)
    s = block[params]
    return np.where(s.notna(), (s == 1.0).astype(float), np.nan)


wf_sel = fh.walk_forward(sel, wf_fit, wf_apply, first_test=WF_START)
wf_sel_rows = fh.evaluate(wf_sel, "crude_roll", (8,), timing="price", sides=("long",),
                          labels=dict(filter="valuation", variant="walk_forward_selection",
                                      market="crude", is_primary=False))
# static primary over the same 2008+ weeks, for comparison
psf_2008 = PSF.copy()
m08 = (psf_2008["decision_date"] >= pd.Timestamp(WF_START)).to_numpy()
psf_2008["valid"] = psf_2008["valid"].to_numpy() & m08
psf_2008["signal"] = np.where(psf_2008["valid"], psf_2008["signal"], 0).astype(np.int8)
static_2008 = fh.evaluate(psf_2008, "crude_roll", (8,), timing="price", sides=SIDES3,
                          labels=dict(filter="valuation", variant="static_primary_2008plus",
                                      market="crude", is_primary=False))
r = wf_sel_rows.iloc[0]
log(f"WF selection (long, h=8): n={r['n']} excess={r['excess']} perm_p={r['perm_p']} period={r['period']}")
r = static_2008[static_2008["side"] == "long"].iloc[0]
log(f"static primary 2008+ (long, h=8): n={r['n']} excess={r['excess']:.6f} perm_p={r['perm_p']:.4f}")

# -------------------------------------------------------------------------------------------
# Strategy (rule 6)
# -------------------------------------------------------------------------------------------
def sim(sf, series, tag):
    t = time.time()
    st = fh.simulate(sf, series, timing="price",
                     labels=dict(filter="valuation", variant=tag, market="crude"))
    log(f"simulate {tag} on {series}: {time.time() - t:.1f}s")
    return st


ST = sim(PSF, "crude_roll", "primary")
ST_CL = sim(PSF, "crude_cl", "primary_on_crude_cl")
ST_VOTE = sim(SF["crude|vote2of3"], "crude_roll", "crude_vote2of3")
for _, r in ST["metrics"].iterrows():
    log(f"  {r['strategy']:15s} n_trades={r.get('n_trades')} sharpe={r['sharpe']:.3f} "
        f"cagr={r['ann_return']:.4f} maxdd={r['max_dd']:.3f} null_p={r.get('null_p_sharpe')}")
strategy = pd.concat([ST["metrics"], ST_CL["metrics"], ST_VOTE["metrics"]], ignore_index=True)
ST["trades"].to_csv(HERE / "trades_primary.csv", index=False)

# -------------------------------------------------------------------------------------------
# POST-HOC context (added after the first run, NOT pre-registered, not used for the verdict):
# why does the primary look better on spot/unadjusted CL than on the roll-adjusted index?
# For each eligible week, the 8-week crude_roll return is split into the EIA spot return over
# the SAME entry/exit dates and a remainder (roll yield + futures/spot basis change).
# -------------------------------------------------------------------------------------------
spot = fh.load_series("crude_spot")
ftp = fh.forward_table(fri, "crude_roll", "price", (8,))
el = np.isfinite(prim_sig) & ftp["r8"].notna().to_numpy()
ent = vl.asof_fresh(spot, pd.DatetimeIndex(ftp["entry_date"].where(el)), max_age_days=4)
ext = vl.asof_fresh(spot, pd.DatetimeIndex(ftp["exit_date8"].where(el)), max_age_days=4)
spot_r = ext / ent - 1.0
roll_r = ftp["r8"].to_numpy()
ok_dec = el & np.isfinite(spot_r)
decomp = {}
for name, msk in (("all_eligible", ok_dec), ("long_setup", ok_dec & (prim_sig == 1.0)),
                  ("short_setup", ok_dec & (prim_sig == -1.0))):
    decomp[name] = dict(n=int(msk.sum()), mean_crude_roll_r8=float(roll_r[msk].mean()),
                        mean_spot_r8_same_dates=float(spot_r[msk].mean()),
                        mean_roll_and_basis=float((roll_r[msk] - spot_r[msk]).mean()))
log("post-hoc decomposition (h=8): " + "; ".join(
    f"{k}: n={v['n']} roll={v['mean_crude_roll_r8']:.4f} spot={v['mean_spot_r8_same_dates']:.4f} "
    f"roll+basis={v['mean_roll_and_basis']:.4f}" for k, v in decomp.items()))

# -------------------------------------------------------------------------------------------
# AUDIT ADDITION (adversarial verifier, added after the published run; NOT pre-registered and
# NOT used for the verdict): how much do the headline cells depend on single episodes?
#   * per-episode contribution of the primary's long weeks (h=8, crude_roll);
#   * leave-one-episode-out primary;
#   * the key cells with every calendar-2020 decision week removed. The COVID crash and
#     rebound is the only large winning episode of the crude long setups.
# Removing weeks = marking them invalid, exactly as fh.split_half does for the other half:
# they leave the signal, the baseline and T, and the permutation runs on the remaining weeks.
# -------------------------------------------------------------------------------------------
IN2020 = np.asarray((fri >= "2020-01-01") & (fri <= "2020-12-31"))


def drop_weeks(sf: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Signal frame with the weeks in `mask` made invalid (no signal, no baseline weight)."""
    out = sf.copy()
    out["valid"] = out["valid"].to_numpy(bool) & ~np.asarray(mask, bool)
    out["signal"] = np.where(out["valid"], out["signal"], 0).astype(np.int8)
    return out


def long_h8(sf: pd.DataFrame, series: str = "crude_roll") -> dict:
    """Long-side h=8 cell of a signal frame (harness evaluate) as a small dict."""
    r = fh.evaluate(sf, series, (8,), timing="price", sides=("long",)).iloc[0]
    return dict(n=int(r["n"]), episodes=int(r["episodes"]), mean=float(r["mean"]),
                excess=float(r["excess"]), hit=float(r["hit"]), perm_p=float(r["perm_p"]),
                perm_p_exact=float(r["perm_p_exact"]), period=r["period"])


# Episodes of the primary long setup on the full Friday grid (runs of signal == +1).
on = (PSF["signal"].to_numpy() == 1) & PSF["valid"].to_numpy(bool)
r8_prim = ftp["r8"].to_numpy()
base_prim = float(np.nanmean(np.where(np.isfinite(prim_sig), r8_prim, np.nan)))
ep_rows, i = [], 0
while i < len(on):
    if not on[i]:
        i += 1
        continue
    j = i
    while j + 1 < len(on) and on[j + 1]:
        j += 1
    wk = np.arange(i, j + 1)
    cnt = wk[np.isfinite(r8_prim[wk])]
    loo = long_h8(drop_weeks(PSF, np.isin(np.arange(len(on)), wk))) if len(cnt) else None
    ep_rows.append(dict(first=fri[i].date().isoformat(), last=fri[j].date().isoformat(),
                        counted_weeks=int(len(cnt)),
                        mean_r8=float(r8_prim[cnt].mean()) if len(cnt) else None,
                        sum_r8_minus_baseline=float((r8_prim[cnt] - base_prim).sum()),
                        primary_without_this_episode=loo))
    i = j + 1

split_mid = pd.Timestamp(split["split_date"].iloc[0])
h1_mask = np.asarray(fri <= split_mid)
cross_sf = fh.make_signal_frame(fri, cross_sig)
without_2020 = dict(
    primary=long_h8(drop_weeks(PSF, IN2020)),
    primary_H2=long_h8(drop_weeks(PSF, IN2020 | h1_mask)),
    primary_traded_crude_spot=long_h8(drop_weeks(PSF, IN2020), "crude_spot"),
    primary_traded_crude_cl=long_h8(drop_weeks(PSF, IN2020), "crude_cl"),
    crude_vs_usd=long_h8(drop_weeks(SF["crude|usd"], IN2020)),
    crude_vote2of3=long_h8(drop_weeks(SF["crude|vote2of3"], IN2020)),
    cross_up=long_h8(drop_weeks(cross_sf, IN2020)),
    walk_forward_selection=long_h8(drop_weeks(wf_sel, IN2020)),
    extended_vs_gold_psys=long_h8(drop_weeks(SF["crude|gold_ext"], IN2020)),
)
for key in ("usd", "vote2of3"):
    parts = [(drop_weeks(SF[f"{m}|{key}"], IN2020), TRADED[m], "price") for m in MARKETS4]
    pr8 = fh.evaluate_pooled(parts, (8,), sides=("long",)).iloc[0]
    without_2020[f"pooled4_{key}"] = dict(n=int(pr8["n"]), excess=float(pr8["excess"]),
                                          perm_p=float(pr8["perm_p"]))
# The spot/roll decomposition of the long-setup weeks, with and without 2020.
for name, msk in (("long_setup_without_2020", ok_dec & (prim_sig == 1.0) & ~IN2020),
                  ("long_setup_2020_only", ok_dec & (prim_sig == 1.0) & IN2020),
                  ("all_eligible_without_2020", ok_dec & ~IN2020)):
    decomp[name] = dict(n=int(msk.sum()), mean_crude_roll_r8=float(roll_r[msk].mean()),
                        mean_spot_r8_same_dates=float(spot_r[msk].mean()),
                        mean_roll_and_basis=float((roll_r[msk] - spot_r[msk]).mean()))
EPISODE_AUDIT = dict(
    note=("Added by the adversarial audit after the published run (not pre-registered, not "
          "used for the verdict). Removing weeks = marking them invalid (as split_half does)."),
    primary_long_episodes=ep_rows, without_2020=without_2020)
w = without_2020
log("AUDIT without 2020 (long h8): " + "; ".join(
    f"{k}: n={v['n']} excess={v['excess']:.4f} p={v['perm_p']:.4f}" for k, v in w.items()))
big = sorted(ep_rows, key=lambda e: abs(e["sum_r8_minus_baseline"]), reverse=True)[:3]
AUDIT_CAVEAT = (
    "AUDIT (added after the run, not pre-registered): the crude long results are set by three "
    "episodes. Their summed weekly excess contributions are "
    + ", ".join(f"{e['first'][:7]}..{e['last'][:7]} {e['sum_r8_minus_baseline']:+.2f} "
                f"({e['counted_weeks']} wks)" for e in big)
    + ". Every positive crude cell depends on the 2020 COVID crash/rebound. With the 2020 "
    f"decision weeks removed: primary excess {w['primary']['excess']:+.4f} "
    f"(p {w['primary']['perm_p']:.4f}); H2 {w['primary_H2']['excess']:+.4f} "
    f"(p {w['primary_H2']['perm_p']:.4f}); traded on spot "
    f"{w['primary_traded_crude_spot']['excess']:+.4f} (p {w['primary_traded_crude_spot']['perm_p']:.4f}); "
    f"vs usd {w['crude_vs_usd']['excess']:+.4f}; 2-of-3 vote {w['crude_vote2of3']['excess']:+.4f}; "
    f"cross-up {w['cross_up']['excess']:+.4f} (p {w['cross_up']['perm_p']:.4f}); "
    f"selection walk-forward {w['walk_forward_selection']['excess']:+.4f} "
    f"(p {w['walk_forward_selection']['perm_p']:.4f}); pooled usd "
    f"{w['pooled4_usd']['excess']:+.4f}, pooled vote {w['pooled4_vote2of3']['excess']:+.4f}. "
    "Outside 2020, spot crude also fell after long setups "
    f"({decomp['long_setup_without_2020']['mean_spot_r8_same_dates']:+.4f} vs "
    f"{decomp['all_eligible_without_2020']['mean_spot_r8_same_dates']:+.4f} for all weeks). "
    "So the post-hoc 'spot rebound erased by roll cost' reading is a single-episode effect, and "
    "the 2020 spot leg starts from dislocated April-2020 Cushing prints.")

# -------------------------------------------------------------------------------------------
# Context: what does the index say at the end of the data (latest defined Friday)?
# -------------------------------------------------------------------------------------------
latest = []
for (m, c) in PAIRS:
    vi = VI[(m, c, "ema2_22", P["N"])]
    ok = np.flatnonzero(np.isfinite(vi))
    if len(ok):
        j = ok[-1]
        latest.append(dict(market=m, comparator=c, date=fri[j].date().isoformat(),
                           vi=float(vi[j]),
                           setup={1.0: "long (undervalued)", -1.0: "short (overvalued)",
                                  0.0: "none"}[float(vl.setups(vi[j:j + 1], 20, 80)[0])]))

# -------------------------------------------------------------------------------------------
# Grid summaries (robustness reading, never the headline)
# -------------------------------------------------------------------------------------------
pair_rows = tests[tests["grid"].isin(["A", "B"]) & (tests["horizon_w"] == 8)
                  & (tests["market"] != "pooled")]
summary = []
for side in SIDES3:
    g = pair_rows[(pair_rows["side"] == side) & (pair_rows["n"] > 0)]
    summary.append(dict(side=side, cells=int(len(g)),
                        share_excess_pos=float((g["excess"] > 0).mean()),
                        median_excess=float(g["excess"].median()),
                        share_perm_p_le_05=float((g["perm_p"] <= 0.05).mean()),
                        min_bh_q=float(g["bh_q_h8"].min())))
by_pair = (pair_rows[pair_rows["n"] > 0].groupby(["market", "comparator", "side"])
           .agg(cells=("excess", "size"), share_excess_pos=("excess", lambda x: float((x > 0).mean())),
                median_excess=("excess", "median"), min_perm_p=("perm_p", "min"))
           .reset_index())
log("grid summary (h=8, grids A+B): " + "; ".join(
    f"{d['side']}: {d['cells']} cells, {d['share_excess_pos']:.3f} excess>0, "
    f"{d['share_perm_p_le_05']:.3f} p<=.05, min BH q {d['min_bh_q']:.3f}" for d in summary))

# -------------------------------------------------------------------------------------------
# Verdict per the pre-registered criteria (PREREG section 4)
# -------------------------------------------------------------------------------------------
pr = PRIMARY_ROW
sh = split[split["side"] == "long"].set_index("half")
s1lo = ST["metrics"].set_index("strategy").loc["S1_long_only"]
wf_ex = float(wf_sel_rows.iloc[0]["excess"]) if wf_sel_rows.iloc[0]["n"] > 0 else float("nan")
checks = dict(
    primary_excess_pos=bool(pr["excess"] > 0),
    primary_perm_p_le_05=bool(pr["perm_p"] <= 0.05),
    primary_perm_p_le_20=bool(pr["perm_p"] <= 0.20),
    split_H1_excess_pos=bool(sh.loc["H1", "excess"] > 0),
    split_H2_excess_pos=bool(sh.loc["H2", "excess"] > 0),
    walk_forward_selection_excess_pos=bool(wf_ex > 0),
    s1_long_only_sharpe_pos=bool(s1lo["sharpe"] > 0),
    s1_long_only_null_p_le_10=bool(s1lo["null_p_sharpe"] <= 0.10),
    anti_signal=bool(pr["excess"] < 0 and pr["perm_p"] >= 0.95),
)
robust_ok = all(checks[k] for k in ("split_H1_excess_pos", "split_H2_excess_pos",
                                    "walk_forward_selection_excess_pos",
                                    "s1_long_only_sharpe_pos", "s1_long_only_null_p_le_10"))
if checks["primary_excess_pos"] and checks["primary_perm_p_le_05"] and robust_ok:
    verdict = "positive EV, robust"
elif checks["primary_excess_pos"] and checks["primary_perm_p_le_20"]:
    verdict = "positive EV, fragile"
elif checks["anti_signal"]:
    verdict = "negative (anti-signal)"
else:
    verdict = "no edge"
log(f"VERDICT (pre-registered criteria): {verdict}  checks={checks}")

# -------------------------------------------------------------------------------------------
# results.json (SPEC2 rule 8)
# -------------------------------------------------------------------------------------------
prereg_sha = hashlib.sha256((CODE / "PREREG.md").read_bytes()).hexdigest()
vi_prim = VI[("crude", "gold", "ema2_22", P["N"])]
okv = np.isfinite(vi_prim)
chart_vi = {"dates": [d.date().isoformat() for d in fri[okv]],
            "vi": [round(float(v), 3) for v in vi_prim[okv]],
            "setup": [int(v) for v in prim_sig[okv]],
            "buy_th": P["buy_th"], "sell_th": P["sell_th"]}
bars = pair_rows[(pair_rows["grid"] == "A") & pair_rows["side"].isin(["long", "short"])]
chart_bars = [dict(market=r["market"], comparator=r["comparator"], side=r["side"],
                   excess=r["excess"], perm_p=r["perm_p"], n=r["n"]) for _, r in bars.iterrows()]

LARRY_DEF = (
    "Larry Williams' Valuation Index (WillVal, 1990): a market's price relative to a comparator "
    "(gold first, also Treasury bonds and the US dollar), R = 100*market/comparator on weekly "
    "closes; Pal = EMA(2) of R minus EMA(22) of R; VI = 100*(Pal - min)/(max - min) over the "
    "last 3 years (156 weekly values). Low VI = undervalued = bullish (buy filter), high = "
    "overvalued = bearish. Thresholds 20/80 pre-registered (sources cite 85/15 and 75/15). "
    "Larry treats it as a filter, not a timing tool; '2 out of 3' comparators agreeing is his "
    "composite reading. Formula details come from third-party re-implementations (ProRealCode "
    "2017, TradingView tradeviZion, MQL5 22099, fxcodebase 2013, StockCharts ACP), not from a "
    "verbatim passage of Larry's books.")
SOURCES = [
    "ProRealCode forum 'Will Val Indicator' (2017): https://www.prorealcode.com/topic/wiil-val-indicator/ (search extract)",
    "TradingView 'Larry Williams Valuation Index [tradeviZion]': https://www.tradingview.com/script/dKwos20x-Larry-Williams-Valuation-Index-tradeviZion/ (search extract)",
    "MQL5 code base 22099 (WillVal): https://www.mql5.com/en/code/22099 (search extract)",
    "StockCharts ACP 'Larry Williams Stock Trading Starter Pack' (WillVal, 65 periods, 75/15, TLT/GLD/UUP): https://help.stockcharts.com/charts-and-tools/stockchartsacp/stockchartsacp-plug-ins/larry-williams-stock-trading-starter-pack-plug-in (search extract)",
    "fxcodebase 'Spread Oscillator' thread (2013), read in full from github.com/gehtsoft-usa/fxcodebase-backup backup/custom-indicators/s/spread-oscillator/index.md ('gold, T-bond and USD index'; '2 out of 3')",
    "Larry Williams interview, Stocks & Commodities July 1997: http://traders.com/documentation/FEEDbk_docs/1997/07/0797Williams.html (bonds vs gold relationship; search extract)",
    "Larry Williams, Trade Stocks and Commodities with the Insiders (Wiley 2005), gold chapter: gold-vs-dollar valuation index, 'major rallies ... begin with this valuation index at a low level' (search extract)",
    "Data: S/data/prices_daily.csv (v1 crude_cl, gold, silver, soybean, sugar); EIA RCLC1/RCLC2 roll index (fh crude_roll); FRED DGS10 (ust10_price proxy); FRED DTWEXBGS (usd_broad); pysystemtrade GOLD multiple prices via scout curve_daily.csv (grid E only)",
]
CAVEATS = [
    "Larry's exact formula is not available verbatim from his books in this sandbox; the pre-registered version (ratio x100, EMA2-EMA22, 156-week min/max) is the most fully specified third-party reading. Sign convention: short minus long EMA (low = undervalued); the reverse sign is exactly 100-VI, i.e. this study's short side.",
    "Comparators are proxies: gold = London spot (not COMEX futures); bonds = a 10-year 6%-coupon price proxy from DGS10 (Larry uses 30-year T-bond futures); dollar = Fed broad index DTWEXBGS from 2006, read with a 7-day publication lag (Larry uses DXY).",
    "Primary sample is short for a 3-year-normalised weekly oscillator: VI exists from 2004-01 and crude_roll ends 2024-04, so about 20 years with few independent extreme episodes; 8-week forward returns overlap.",
    "Crude valuation input is the unadjusted CL front month (crude_cl), whose level steps at rolls; grid C repeats the primary with EIA spot as the input.",
    "Grid E uses pysystemtrade GOLD held-contract prices (unadjusted; post-2013 rows are not always settlements) and EIA RCLC1 for crude.",
    "Grids B-F are many correlated cells; BH q-values are information only and the headline is the pre-registered primary row.",
    "Strategy P&L uses close-to-close daily returns of the traded index, 0.05% per side, no financing/collateral return, no slippage beyond that.",
    AUDIT_CAVEAT,
]

doc_extra = dict(
    prereg_sha256=prereg_sha,
    verdict=verdict, verdict_checks=checks,
    pooled=POOLED,
    grid_summary_h8=summary, grid_by_pair_h8=by_pair,
    latest_readings=latest,
    post_hoc_roll_decomposition_h8=dict(
        note=("Added after the first run (not pre-registered; not used for the verdict). "
              "crude_roll 8-week return of the primary's weeks split into the EIA Cushing spot "
              "return over the same entry/exit dates and the remainder (roll yield + basis)."),
        **decomp),
    extended_split_half=ext_split,
    episode_sensitivity_audit=EPISODE_AUDIT,
    data_notes=dict(
        friday_grid=f"{fri[0].date()}..{fri[-1].date()} ({len(fri)} Fridays)",
        primary_vi_defined=f"{fri[okv][0].date()}..{fri[okv][-1].date()} ({int(okv.sum())} weeks)",
        staleness_limit_days=vl.MAX_AGE_DAYS, usd_lag_days=vl.USD_LAG_DAYS,
        traded_series={m: TRADED[m] for m in TRADED}, valuation_input=INPUT_NAME),
    charts=dict(equity_primary=fh.weekly_equity(ST["equity"]), vi_crude_vs_gold=chart_vi,
                h8_excess_by_pair=chart_bars),
    run_log=LOG,
)
oos = dict(split_half=split,
           walk_forward=dict(note=("The primary rule has no parameter estimated from data (EMAs and "
                                   "the 156-week min/max are causal), so its walk-forward signal "
                                   "equals the static one ('identity'). The verdict uses the "
                                   "pre-registered parameter-selection walk-forward."),
                             identity=wf_id_rows, selection=wf_sel_rows,
                             selection_log=SEL_LOG, static_primary_same_window=static_2008,
                             first_test=WF_START, min_signal_weeks=MIN_SIG_WEEKS))
tests.to_csv(HERE / "tests.csv", index=False)
fh.write_results(HERE / "results.json", filter="valuation", larry_definition=LARRY_DEF,
                 sources=SOURCES, primary={**PRIMARY_ROW, "traded_series": "crude_roll",
                                           "valuation_input": "crude_cl vs gold (v1 London spot)"},
                 tests=tests, strategy=strategy, oos=oos, caveats=CAVEATS, **doc_extra)
log(f"wrote results.json ({len(tests)} test rows)")
(HERE / "run_log.txt").write_text("\n".join(LOG) + "\n")

#!/usr/bin/env python3
"""
run_premium -- backtest of Larry Williams' premium/discount tool, exactly as pre-registered in
PREREG.md (same directory). Run:

    python3 -I battery/tool_premium/run_premium.py

Writes (ROOT/work2/tool_premium/; ROOT = $LARRY_COT_ROOT, else the package dir):
    results.json          SPEC2 rule-8 contract + chart-ready JSON + context blocks
    tests.csv             every test row (primary, variant grid, other markets, comparator)
    trades_primary.csv    S1/S2 trades of the primary rule on crude_roll
    run_log.txt           what was run and how long it took

Re-running this script rewrites results.json WITHOUT the adversarial-verification block; re-run
verifier/verify_independent.py and then verifier/verifier_checks.py afterwards to add the
"verification" key back (post-hoc checks only; they never change the primary or the verdict).

All evaluation (forward returns, baseline, excess, permutation tests, split halves, walk-forward,
S1/S2 strategy and its random-timing null) is done by the shared harness `fund_harness`; this
script only builds the weekly signals (premium_lib) and assembles the outputs. Every number in
results.json is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True   # never write .pyc files next to the published scripts

CODE = Path(__file__).resolve().parent         # this tool's code and PREREG.md
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import premium_lib as pl  # noqa: E402
fh = pl.fh
WORK = fh.WORK2 / "tool_premium"               # outputs: ROOT/work2/tool_premium
WORK.mkdir(parents=True, exist_ok=True)

H = (1, 4, 8, 13)                       # SPEC2 rule 3 horizons (weeks)
SIDES = ("long", "short", "net")
CRUDE_START, CRUDE_END = "1985-01-04", "2024-03-29"   # Fridays with EIA RCLC1/RCLC2 prints
PST_START = "1970-01-02"
WF_FIRST_TEST = "1995-01-01"
FILTER = "premium"
LOG: list[str] = []
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    LOG.append(line)
    print(line, flush=True)


def run_eval(sf, series, *, variant, market, timing="price", sides=SIDES, horizons=H,
             series_name=None):
    """fh.evaluate with the battery's labels."""
    return fh.evaluate(sf, series, horizons, timing=timing, sides=sides, series_name=series_name,
                       labels=dict(filter=FILTER, variant=variant, market=market,
                                   is_primary=False))


def main() -> None:
    prereg = (CODE / "PREREG.md").read_bytes()
    prereg_sha = hashlib.sha256(prereg).hexdigest()
    log(f"PREREG.md sha256 {prereg_sha}")
    tests: list[pd.DataFrame] = []

    # =========================================================================================
    # A. Crude curve signals (EIA RCLC1..RCLC4), Friday grid 1985-01-04 .. 2024-03-29
    # =========================================================================================
    fri = fh.weekly_decision_dates(CRUDE_START, CRUDE_END)
    p12 = pl.weekly_asof(pl.crude_premium_daily(1, 2), fri)
    p14 = pl.weekly_asof(pl.crude_premium_daily(1, 4), fri)
    p23 = pl.weekly_asof(pl.crude_premium_daily(2, 3), fri)
    crude_signals = {
        "primary": pl.sig_sign(p12),
        "cl1_cl4": pl.sig_sign(p14),
        "cl2_cl3": pl.sig_sign(p23),
        "pct_0.5": pl.sig_threshold(p12, 0.005),
        "pct_1.0": pl.sig_threshold(p12, 0.010),
        "range52": pl.sig_range(p12, 52, 80, 20),
        "steepen4": pl.sig_steepen(p12, 4),
        "cross4": pl.sig_cross(p12, 4),
        "z_exp": pl.sig_zexp(p12, 104, 1.0),
    }
    frames = {v: fh.make_signal_frame(fri, s) for v, s in crude_signals.items()}
    sf0 = frames["primary"]
    sig_share = {v: dict(valid=int(f["valid"].sum()), long=int((f["signal"] == 1).sum()),
                         short=int((f["signal"] == -1).sum()), flat=int(((f["signal"] == 0)
                                                                         & f["valid"]).sum()))
                 for v, f in frames.items()}
    log(f"crude weekly frames built: {len(fri)} Fridays; primary {sig_share['primary']}")

    # Primary + crude variant grid on the roll-adjusted index
    for v, sf in frames.items():
        tests.append(run_eval(sf, "crude_roll", variant=v, market="crude"))
        log(f"evaluated crude {v} on crude_roll")
    # Primary signal on alternative traded series
    for ser in ("crude_roll_lead5", "crude_cl", "crude_spot"):
        tests.append(run_eval(sf0, ser, variant="primary", market="crude"))
        log(f"evaluated crude primary on {ser}")

    # =========================================================================================
    # B. Other markets: pysystemtrade near/far curves
    # =========================================================================================
    fri_all = fh.weekly_decision_dates(PST_START, CRUDE_END)
    pst_info, pst_levels, pst_frames = {}, {}, {}
    v1_series = {"gold": "gold", "silver": "silver", "soybean": "soybean", "sugar": "sugar",
                 "heatoil": "ho_spot"}
    for m in pl.PST_MARKETS:
        level, info = pl.pst_roll_index(m)
        pst_levels[m], pst_info[m] = level, info
        pm = pl.weekly_asof(pl.pst_premium_daily(m), fri_all)
        sf_m = fh.make_signal_frame(fri_all, pl.sig_sign(pm))
        pst_frames[m] = sf_m
        info["weeks_valid"] = int(sf_m["valid"].sum())
        info["weeks_premium"] = int((sf_m["signal"] == 1).sum())
        info["weeks_discount"] = int((sf_m["signal"] == -1).sum())
        tests.append(run_eval(sf_m, level, variant="pst_sign", market=m,
                              series_name=f"pst_roll_{m}"))
        if m in v1_series:
            tests.append(run_eval(sf_m, v1_series[m], variant="pst_sign", market=m))
        log(f"evaluated {m}: {info['weeks_valid']} valid weeks, {info['weeks_premium']} premium")
    pooled = fh.evaluate_pooled([(pst_frames[m], pst_levels[m], "price") for m in pl.PST_MARKETS],
                                H, sides=("long", "short"),
                                labels=dict(filter=FILTER, variant="pst_sign_pooled10",
                                            market="pooled10", is_primary=False))
    pooled["series"] = "pst_roll_(10 markets)"
    tests.append(pooled)
    log("evaluated pooled 10-market cell")

    # =========================================================================================
    # C. Non-Larry comparator: EIA commercial crude stocks vs their 5-year seasonal norm
    # =========================================================================================
    inv = pl.inventory_vs_norm("WCESTUS1", 5, 3)
    dev = inv["dev"].to_numpy()
    inv_sig = np.where(np.isnan(dev), np.nan, np.where(dev < 0, 1.0, np.where(dev > 0, -1.0, 0.0)))
    sf_inv = fh.make_signal_frame(inv.index, inv_sig)
    for ser in ("crude_roll", "crude_spot"):
        tests.append(run_eval(sf_inv, ser, variant="inv_vs_5y_norm", market="crude",
                              timing="eia"))
    log("evaluated inventory comparator")
    # Does it say the same thing as the curve? (same reference Friday)
    dev_f = inv["dev"].reindex(pd.DatetimeIndex(fri)).to_numpy()
    both = ~np.isnan(dev_f) & ~np.isnan(p12)
    inv_bull = dev_f[both] < 0
    prem = p12[both] > 0
    agreement = dict(
        n_common_weeks=int(both.sum()),
        period=f"{pd.DatetimeIndex(fri)[both].min().date()}..{pd.DatetimeIndex(fri)[both].max().date()}",
        share_same_direction=float(np.mean(inv_bull == prem)),
        pearson_dev_vs_premium=float(np.corrcoef(dev_f[both], p12[both])[0, 1]),
        spearman_dev_vs_premium=float(pd.Series(dev_f[both]).corr(pd.Series(p12[both]),
                                                                    method="spearman")),
        table={"premium_and_stocks_below_norm": int(np.sum(prem & inv_bull)),
               "premium_and_stocks_above_norm": int(np.sum(prem & ~inv_bull)),
               "discount_and_stocks_below_norm": int(np.sum(~prem & inv_bull)),
               "discount_and_stocks_above_norm": int(np.sum(~prem & ~inv_bull))},
        note="premium = c1 > c2 (equal counted as not premium); stocks below norm = dev < 0")

    # =========================================================================================
    # D. Out-of-sample: split half and walk-forward (learned side)
    # =========================================================================================
    sh = fh.split_half(sf0, "crude_roll", timing="price", horizon=8, sides=("long", "short", "net"),
                       horizons=H, labels=dict(filter=FILTER, variant="primary", market="crude",
                                               is_primary=False))
    log("split half done")

    ft = fh.forward_table(fri, "crude_roll", "price", (8,))
    wf_frame = pd.DataFrame({"decision_date": pd.DatetimeIndex(fri), "sig": crude_signals["primary"],
                             "r8": ft["r8"].to_numpy(), "exit_date8": ft["exit_date8"].to_numpy()})
    wf_params = []

    def fit(hist: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
        """Learn, from 8-week outcomes already realised before the cutoff, whether each side of
        the premium rule had a positive excess over the all-weeks baseline."""
        h = hist[(hist["exit_date8"] < cutoff) & hist["sig"].notna() & hist["r8"].notna()]
        base = float(h["r8"].mean())
        lx = float(h.loc[h["sig"] == 1, "r8"].mean() - base)
        sx = float(-h.loc[h["sig"] == -1, "r8"].mean() + base)
        prm = dict(cutoff=cutoff.date().isoformat(), n_hist=int(len(h)),
                   last_exit=pd.Timestamp(h["exit_date8"].max()).date().isoformat(),
                   long_excess=lx, short_excess=sx, trade_long=bool(lx > 0),
                   trade_short=bool(sx > 0))
        wf_params.append(prm)
        return prm

    def apply(prm: dict, block: pd.DataFrame):
        s = block["sig"].to_numpy(dtype=float)
        out = np.where((s == 1) & prm["trade_long"], 1.0,
                       np.where((s == -1) & prm["trade_short"], -1.0, 0.0))
        return np.where(np.isnan(s), np.nan, out)

    sf_wf = fh.walk_forward(wf_frame, fit, apply, first_test=WF_FIRST_TEST, refit="YS")
    wf_rows = run_eval(sf_wf, "crude_roll", variant="wf_learned_side", market="crude")
    # static primary over the same 1995+ window, for a like-for-like comparison
    sf_static95 = sf0.copy()
    m95 = (sf_static95["decision_date"] >= pd.Timestamp(WF_FIRST_TEST)).to_numpy()
    sf_static95["valid"] = sf_static95["valid"].to_numpy() & m95
    sf_static95["signal"] = np.where(sf_static95["valid"], sf_static95["signal"], 0).astype(np.int8)
    static95_rows = run_eval(sf_static95, "crude_roll", variant="primary_from_1995", market="crude")
    tests += [wf_rows, static95_rows]
    log("walk-forward done")

    # =========================================================================================
    # E. Strategy (S1/S2, long+short and long-only) with costs and random-timing null
    # =========================================================================================
    strat_main = fh.simulate(sf0, "crude_roll", timing="price",
                             labels=dict(filter=FILTER, variant="primary", market="crude"))
    log("strategy on crude_roll done")
    strat_ctx = []
    for ser in ("crude_roll_lead5", "crude_spot"):
        st = fh.simulate(sf0, ser, timing="price", keep_equity=False,
                         labels=dict(filter=FILTER, variant="primary", market="crude"))
        strat_ctx.append(st["metrics"])
        log(f"strategy on {ser} done")
    strategy = pd.concat([strat_main["metrics"]] + strat_ctx, ignore_index=True)

    # =========================================================================================
    # F. Context: roll yield vs spot price change (same entry/exit dates)
    # =========================================================================================
    spot = fh.load_series("crude_spot")
    ent = pd.DatetimeIndex(ft["entry_date"])
    exi = pd.DatetimeIndex(ft["exit_date8"])
    s_ent = fh.asof_values(spot, ent.fillna(pd.Timestamp("1900-01-01")))
    s_exi = fh.asof_values(spot, exi.fillna(pd.Timestamp("1900-01-01")))
    r_spot = s_exi / s_ent - 1.0
    r_roll = ft["r8"].to_numpy()
    ok = ~np.isnan(r_roll) & ~np.isnan(r_spot) & sf0["valid"].to_numpy() & (ent >= spot.index[0])
    sig = sf0["signal"].to_numpy()
    decomp = {}
    for name, m in (("all_eligible", ok), ("premium_long", ok & (sig == 1)),
                    ("discount_short_unsigned", ok & (sig == -1))):
        decomp[name] = dict(n=int(m.sum()), mean_roll_adjusted=float(r_roll[m].mean()),
                            mean_spot=float(r_spot[m].mean()),
                            mean_roll_minus_spot=float((r_roll[m] - r_spot[m]).mean()))
    a, lng = decomp["all_eligible"], decomp["premium_long"]
    decomp["premium_long_excess"] = dict(
        total_roll_adjusted=lng["mean_roll_adjusted"] - a["mean_roll_adjusted"],
        from_spot_change=lng["mean_spot"] - a["mean_spot"],
        from_roll_and_basis=lng["mean_roll_minus_spot"] - a["mean_roll_minus_spot"])
    decomp["note"] = ("h=8 weeks; returns on crude_roll entry/exit dates; spot = EIA Cushing WTI "
                      "as of the same dates; roll_minus_spot = roll yield + futures/spot basis "
                      "change. Weeks before the first spot print (1986-01-02) are excluded.")
    log("decomposition done")

    # =========================================================================================
    # F2. POST-HOC diagnostics (NOT pre-registered; added after the pre-registered run to
    #     understand it; never used for the verdict):
    #     (a) S1/S2 strategy on the two split halves of the primary frame;
    #     (b) the best grid cell (cross4) on the spot series, to see whether it is price or carry;
    #     (c) primary long-side excess by decade on crude_roll and crude_spot (no p-values).
    # =========================================================================================
    split_date = pd.Timestamp(sh["split_date"].iloc[0])
    roll = fh.load_series("crude_roll")
    first_after = int(np.searchsorted(roll.index.values, np.datetime64(split_date), side="right"))
    roll_h1 = roll.iloc[: min(len(roll), first_after + fh.HOLD + 1)]     # let H1 trades finish
    post_strat = []
    for half, mask, ser in (("H1", sf0["decision_date"] <= split_date, roll_h1),
                            ("H2", sf0["decision_date"] > split_date, roll)):
        part = sf0.copy()
        part["valid"] = part["valid"].to_numpy() & mask.to_numpy()
        part["signal"] = np.where(part["valid"], part["signal"], 0).astype(np.int8)
        st = fh.simulate(part, ser, timing="price", series_name=f"crude_roll_{half}",
                         keep_equity=False,
                         labels=dict(filter=FILTER, variant=f"primary_{half}", market="crude"))
        post_strat.append(st["metrics"].assign(half=half, split_date=split_date.date().isoformat()))
        log(f"post-hoc strategy {half} done")
    cross_spot = run_eval(frames["cross4"], "crude_spot", variant="cross4_posthoc",
                          market="crude")
    dec_rows = []
    years = pd.DatetimeIndex(sf0["decision_date"]).year
    for d0 in range(1980, 2030, 10):
        m = (years >= d0) & (years < d0 + 10)
        if not m.any():
            continue
        part = sf0.copy()
        part["valid"] = part["valid"].to_numpy() & np.asarray(m)
        part["signal"] = np.where(part["valid"], part["signal"], 0).astype(np.int8)
        for ser in ("crude_roll", "crude_spot"):
            rr = fh.evaluate(part, ser, (8,), timing="price", sides=("long",), do_perm=False)
            if int(rr["T"].iloc[0]) == 0:
                continue
            dec_rows.append(dict(decade=f"{d0}s", series=ser, **{k: rr[k].iloc[0] for k in (
                "n", "episodes", "mean", "baseline", "excess", "hit", "T", "period")}))
    post_hoc = dict(
        note="POST-HOC diagnostics, not pre-registered, not used for the verdict",
        strategy_by_half=pd.concat(post_strat, ignore_index=True),
        cross4_on_spot=cross_spot,
        primary_long_h8_by_decade=pd.DataFrame(dec_rows))
    log("post-hoc diagnostics done")

    # =========================================================================================
    # G. Assemble tests, primary flag, BH q-values on the h=8 grid
    # =========================================================================================
    tdf = pd.concat(tests, ignore_index=True)
    tdf["is_primary"] = ((tdf["variant"] == "primary") & (tdf["series"] == "crude_roll")
                         & (tdf["side"] == "long") & (tdf["horizon_w"] == 8))
    if int(tdf["is_primary"].sum()) != 1:
        raise AssertionError("exactly one primary row expected")
    h8 = tdf["horizon_w"] == 8
    tdf["bh_q_h8"] = np.nan
    tdf.loc[h8, "bh_q_h8"] = pl.bh_qvalues(tdf.loc[h8, "perm_p"].to_numpy())
    primary = tdf[tdf["is_primary"]].iloc[0].to_dict()

    # =========================================================================================
    # H. Verdict by the PRE-DECLARED criteria (PREREG.md section 3)
    # =========================================================================================
    h1 = sh[(sh["half"] == "H1") & (sh["side"] == "long") & (sh["horizon_w"] == 8)].iloc[0]
    h2 = sh[(sh["half"] == "H2") & (sh["side"] == "long") & (sh["horizon_w"] == 8)].iloc[0]
    wf8 = wf_rows[(wf_rows["side"] == "long") & (wf_rows["horizon_w"] == 8)].iloc[0]
    s1lo = strat_main["metrics"].set_index("strategy").loc["S1_long_only"]
    crit = dict(
        primary_excess_pos=bool(primary["excess"] > 0),
        primary_perm_p_le_0_05=bool(primary["perm_p"] <= 0.05),
        primary_perm_p_le_0_20=bool(primary["perm_p"] <= 0.20),
        h1_excess_pos=bool(h1["excess"] > 0), h2_excess_pos=bool(h2["excess"] > 0),
        wf_excess_pos=bool(wf8["excess"] > 0) if not np.isnan(wf8["excess"]) else False,
        s1_long_only_sharpe_pos=bool(s1lo["sharpe"] > 0),
        s1_long_only_null_p_le_0_10=bool(s1lo["null_p_sharpe"] <= 0.10),
        anti_perm_p_ge_0_95=bool(primary["perm_p"] >= 0.95))
    robust = all(crit[k] for k in ("primary_excess_pos", "primary_perm_p_le_0_05", "h1_excess_pos",
                                   "h2_excess_pos", "wf_excess_pos", "s1_long_only_sharpe_pos",
                                   "s1_long_only_null_p_le_0_10"))
    if robust:
        verdict = "positive EV, robust"
    elif crit["primary_excess_pos"] and crit["primary_perm_p_le_0_20"]:
        verdict = "positive EV, fragile"
    elif (not crit["primary_excess_pos"]) and crit["anti_perm_p_ge_0_95"]:
        verdict = "negative (anti-signal)"
    else:
        verdict = "no edge"
    log(f"verdict by pre-declared criteria: {verdict}")

    # =========================================================================================
    # I. Charts (chart-ready JSON)
    # =========================================================================================
    wk_p = pd.DataFrame({"date": [d.date().isoformat() for d in fri], "premium_c1c2": p12,
                         "signal": sf0["signal"].to_numpy()})
    grid8 = tdf[h8][["variant", "market", "series", "timing", "side", "n", "episodes", "mean",
                     "baseline", "excess", "hit", "perm_p", "bh_q_h8"]]
    charts = dict(
        equity_primary_crude_roll=fh.weekly_equity(strat_main["equity"]),
        weekly_premium_crude=dict(dates=wk_p["date"].tolist(),
                                  premium_c1c2=[None if np.isnan(x) else round(float(x), 6)
                                                for x in p12],
                                  signal=[int(x) for x in sf0["signal"]]),
        grid_h8=grid8,
        roll_vs_spot=decomp,
        wf_params=wf_params)

    # =========================================================================================
    # J. Write outputs
    # =========================================================================================
    ri = fh.build_crude_roll_index(0)
    roll_val = fh.validate_roll_index(ri)
    larry_def = (
        "Premium/discount (Larry Williams): when the nearby futures contract trades ABOVE the "
        "deferred contract (a 'premium' or inverted market, i.e. backwardation) commercial users "
        "are paying up for immediate supply, which he reads as bullish; a nearby DISCOUNT to the "
        "deferred (normal carrying-charge market, contango) is the non-bullish/bearish state; the "
        "cross from discount to premium, or a rising premium in a falling market, signals a "
        "developing bullish condition. Tested as: long when contract 1 > contract 2 at the Friday "
        "close (primary), short when contract 1 < contract 2.")
    sources = [
        "Masterclass with Larry Williams: COT, Market Cycles & Trading Secrets (auto-generated "
        "transcript), https://sozai.app/transcript/masterclass-larry-williams-cot-market-cycles-"
        "trading-secrets/ -- search-engine extract: 'bullish buy signals work a lot better so when "
        "there's a premium in the front months that that's a bullish signal'; page summary: "
        "premiums as a fundamental tool with bullish implications (page not openable from the "
        "sandbox; wording approximate).",
        "Commodity-education page on Williams-style spread analysis, search-engine extract (URL "
        "not confirmed; candidates ireallytrade.com/futurestrading, solerinvestments.com/Online-"
        "Trading/Larry-Williams-Trading.htm): 'the nearby month will sell for more money than the "
        "distant months. This is known as a premium market'; 'the cross from a normal or "
        "discounted market ... to an inverted or premium market is bullish'; 'Commercial buying "
        "causes this phenomenon because of an urgent need for the commodity'. Attribution to "
        "Larry unverified.",
        "Larry Williams books on fundamentals: Long-Term Secrets to Short-Term Trading (Wiley "
        "1999/2011); Trade Stocks & Commodities with the Insiders (Wiley 2005); The Definitive "
        "Guide to Futures Trading (Windsor 1988-89). No verbatim premium passage or page could be "
        "retrieved; no page-level citation is claimed.",
        "Data: EIA NYMEX WTI contracts RCLC1-RCLC4 (accomplish999 archive), crude_roll index "
        "built by fund_harness; pysystemtrade multiple_prices_csv @ bbe29e19 (scout "
        "curve_daily.csv); EIA WCESTUS1 weekly commercial crude stocks; v1 price series.",
    ]
    deviations = [
        "pysystemtrade roll index: besides the pre-registered bridges (new contract priced on t-1; "
        "else move the switch back up to 10 rows), a second bridge was added BEFORE any return "
        "of these markets was evaluated: if the OLD contract is still priced on the roll day t, "
        "the switch is made at the close of t. It was needed for silver's 2009-2021 Sep/Dec rolls, "
        "where the new contract has no t-1 price in the file. Silver still has "
        "unresolved rolls (roll-day return set to 0); counts are in series_info.",
    ]
    caveats = [
        "Primary signal and crude_roll both end 2024-04-05 (EIA stopped publishing RCLC1-4 in "
        "this archive); no free curve data for 2024-04..2026 were found, so the rule cannot be "
        "checked on the most recent 2.5 years.",
        "Crude spends about half of all weeks in each state, so 'premium weeks' is a regime "
        "(long persistent episodes), not a rare event: the effective number of independent "
        "observations is the episode count, not n.",
        "The roll-adjusted series earns roll yield, and roll yield is mechanically positive in "
        "backwardation: part of any premium edge on crude_roll is carry, which a spot-price "
        "chart would not show (see roll_vs_spot).",
        "Contract-1 prints in the last days before expiry can be dislocated (e.g. 2020-04-20, "
        "dropped as non-positive); cl2_cl3 and crude_roll_lead5 rows are the robustness checks.",
        "pysystemtrade curves for crude/corn/wheat/soybean are a deferred seasonal pair (e.g. "
        "crude Nov vs Dec, corn Sep vs Dec), not Larry's nearby vs next; grain Sep/Dec spreads "
        "straddle the old/new crop boundary. Gold/silver/wheat are almost never at a premium, so "
        "their long side has few episodes. Post-2008..2013 pysystemtrade rows are the last "
        "hourly snapshot of the day, not always the settlement.",
        "Excess is measured against the all-weeks baseline of the same series and window; "
        "overlapping 8-week returns make naive t-stats invalid, so only permutation p-values are "
        "reported. Grid cells are robustness only; BH q-values on the h=8 grid are informational.",
        "Search budget/sandbox: Larry's book text could not be opened; the definition rests on "
        "search-engine extracts of secondary pages (see sources).",
        "crude_cl, soybean, sugar and ho_f1-type v1 series are UNADJUSTED continuous front "
        "months: in backwardation every roll prints a negative gap (expiring high contract -> "
        "lower next contract), so on those series premium weeks are mechanically penalised. "
        "crude_spot (EIA Cushing) and gold/silver (London spot) have no roll gaps and are the "
        "clean 'price only' checks.",
        "The crude_spot strategy rows run their buy & hold over the whole series window "
        "(to 2026-10) while the signal stops in 2024-03 (harness make_ctx convention), so their "
        "B&H comparison is not like-for-like; the crude_roll rows are.",
        "Post-hoc diagnostics (results.post_hoc) were added after the pre-registered run; they "
        "explain the result but do not enter the verdict.",
    ]
    oos = dict(split_half=sh, walk_forward=dict(
        description="learned-side rule: each Jan 1 from 1995 the premium-long (discount-short) "
                    "side is traded only if its 8-week excess over the all-weeks baseline, on "
                    "outcomes realised before the cutoff, was > 0",
        rows=wf_rows, static_primary_same_window=static95_rows, params_by_year=wf_params),
        note="primary rule has no fitted parameter; z_exp variant is walk-forward by construction")
    extra = dict(
        verdict=verdict, verdict_criteria=crit, prereg_sha256=prereg_sha,
        prereg_path=str(CODE / "PREREG.md"), deviations=deviations,
        signal_counts_crude=sig_share, inventory_agreement=agreement,
        roll_vs_spot=decomp, post_hoc=post_hoc, series_info=dict(pst_roll_indexes=pst_info,
                                              crude_roll_validation=roll_val),
        charts=charts, run_seconds=round(time.time() - T0, 1))
    fh.write_results(WORK / "results.json", filter=FILTER, larry_definition=larry_def,
                     sources=sources, primary=primary, tests=tdf, strategy=strategy, oos=oos,
                     caveats=caveats, **extra)
    tdf.to_csv(WORK / "tests.csv", index=False)
    strat_main["trades"].to_csv(WORK / "trades_primary.csv", index=False)
    log("wrote results.json, tests.csv, trades_primary.csv")
    (WORK / "run_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

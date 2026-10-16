#!/usr/bin/env python3
"""
run_open_interest -- backtest of Larry Williams' OPEN INTEREST tool (SPEC2 battery).

Implements PREREG.md (this directory) exactly:
  * PRIMARY: crude, F+O total OI, 4-report OI change in the bottom 10% of its last 3 years
    while price is range-bound (|4-report price change| <= its 3-year median) -> LONG;
    top 10% -> SHORT. COT timing, traded on crude_roll, headline cell = long, h = 8 weeks.
  * Variant grid A-I (robustness only), pooled cells, split-half, walk-forward (WF-1, WF-2),
    S1/S2 strategy with costs and random-timing null, multiple-testing summary.
Writes results.json (SPEC2 rule 8), tests.csv, trades_primary.csv and run_log.txt.

Run:  python3 -I run_open_interest.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent         # this tool's code and PREREG.md
sys.path.insert(0, str(CODE))
import oi_lib as ol          # noqa: E402
from oi_lib import fh        # noqa: E402  shared harness (same module object)
WORK = fh.WORK2 / "tool_open_interest"         # outputs: ROOT/work2/tool_open_interest
WORK.mkdir(parents=True, exist_ok=True)

FILTER = "open_interest"
H = (1, 4, 8, 13)
SIDES3 = ("long", "short", "net")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
_FEATS: dict = {}


def feats(key: tuple, rep: pd.DataFrame, series_name: str) -> pd.DataFrame:
    """Cached feature table for (market, oi_basis, traded series)."""
    k = key + (series_name,)
    if k not in _FEATS:
        _FEATS[k] = ol.build_features(rep, fh.load_series(series_name))
    return _FEATS[k]


TESTS: list[pd.DataFrame] = []


def ev(sf: pd.DataFrame, series: str, variant: str, market: str, *, oi_basis: str = "F+O",
       sides=SIDES3, primary: bool = False) -> pd.DataFrame:
    """Evaluate one signal frame with the harness and store the rows."""
    rows = fh.evaluate(sf, series, H, timing="cot", sides=sides,
                       labels=dict(filter=FILTER, variant=variant, market=market,
                                   is_primary=False))
    rows["oi_basis"] = oi_basis
    if primary:
        m = (rows["side"] == "long") & (rows["horizon_w"] == 8)
        rows.loc[m, "is_primary"] = True
    TESTS.append(rows)
    return rows


def pooled(parts, variant: str) -> pd.DataFrame:
    rows = fh.evaluate_pooled(parts, H, sides=("long", "short"),
                              labels=dict(filter=FILTER, variant=variant, market="pooled5",
                                          is_primary=False))
    rows["oi_basis"] = "F+O"
    TESTS.append(rows)
    return rows


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone), NaN-safe."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def row_dict(df: pd.DataFrame, **match) -> dict:
    m = np.ones(len(df), bool)
    for k, v in match.items():
        m &= (df[k] == v).to_numpy()
    sub = df[m]
    if len(sub) != 1:
        raise ValueError(f"expected 1 row for {match}, got {len(sub)}")
    return sub.iloc[0].to_dict()


# ----------------------------------------------------------------------------------------
# Rules of the grid (name -> function of a feature table)
# ----------------------------------------------------------------------------------------
GRID_C = [(w, p, rf) for w in (1, 2, 4, 13) for p in (5, 10, 20) for rf in (True, False)]


def grid_c_name(w, p, rf) -> str:
    return f"oi{w}w_p{p}_{'range' if rf else 'any'}"


def other_rules() -> list[tuple[str, callable]]:
    """Variants D, E, G, H, I and the composite quadrant rules (F) as (name, rule(f))."""
    r = [("larry30_52", lambda f: ol.rule_larry30(f, 52)),
         ("larry30_26", lambda f: ol.rule_larry30(f, 26)),
         ("liq_bottom_1w", lambda f: ol.rule_liq_bottom(f, 1)),
         ("liq_bottom_4w", lambda f: ol.rule_liq_bottom(f, 4))]
    for w in (4, 13):
        r.append((f"quad_classic_{w}w", lambda f, w=w: ol.rule_quad_classic(f, w)))
        r.append((f"quad_trend_{w}w", lambda f, w=w: ol.rule_quad_trend(f, w)))
    r += [("oi_range26", lambda f: ol.rule_oi_range(f, 26)),
          ("oi_range52", lambda f: ol.rule_oi_range(f, 52)),
          ("hy12m", lambda f: ol.rule_hy(f, False)),
          ("hy12m_sign", lambda f: ol.rule_hy(f, True)),
          ("willco26", ol.rule_willco)]
    return r


# ----------------------------------------------------------------------------------------
# Walk-forward (WF-1): yearly choice among the 24 grid-C rules by past realised excess
# ----------------------------------------------------------------------------------------
def walk_forward_selection(rep: pd.DataFrame, f: pd.DataFrame, series: str,
                           candidates: dict[str, np.ndarray], min_n: int = 15,
                           first_test: str = "2010-01-01"):
    """Each calendar year from first_test: choose the candidate with the highest past LONG
    excess at h=8, computed only on weeks whose 8-week exit is before the cutoff (and against
    the baseline of those same weeks), with >= min_n past long weeks; apply it to the year.
    Returns (signal frame, list of yearly choices)."""
    ft = fh.forward_table(rep["report_date"], series, "cot", (8,))
    frame = pd.DataFrame({"decision_date": rep["report_date"].to_numpy(),
                          "excluded": rep["shutdown"].to_numpy(bool),
                          "r8": ft["r8"].to_numpy(), "exit_date8": ft["exit_date8"].to_numpy()})
    for name, sig in candidates.items():
        frame[f"sig__{name}"] = sig
    choices = []

    def fit(hist: pd.DataFrame, cutoff: pd.Timestamp):
        done = hist["exit_date8"].notna() & (hist["exit_date8"] < cutoff) & ~hist["excluded"]
        best, best_ex, table = None, -np.inf, {}
        for name in candidates:
            s = hist[f"sig__{name}"]
            el = done & s.notna() & hist["r8"].notna()
            if not el.any():
                continue
            r = hist.loc[el, "r8"].to_numpy()
            lo = (s[el] == 1).to_numpy()
            if lo.sum() < min_n:
                continue
            # No past long week (only possible with min_n=0, i.e. the WF-2 sanity run):
            # the excess is unknown, so rank it last but keep it selectable.
            ex = float(r[lo].mean() - r.mean()) if lo.any() else -np.inf
            table[name] = dict(n=int(lo.sum()), excess=ex)
            if best is None or ex > best_ex:
                best, best_ex = name, ex
        choices.append(dict(year=cutoff.year, cutoff=cutoff.date().isoformat(), chosen=best,
                            past_excess=None if best is None else best_ex,
                            past_n=None if best is None else table[best]["n"],
                            n_qualified=len(table)))
        return best

    def apply(best, block: pd.DataFrame):
        if best is None:                      # nothing qualified: valid weeks, no setup
            return np.zeros(len(block))
        return block[f"sig__{best}"].to_numpy()

    sf = fh.walk_forward(frame, fit, apply, first_test=first_test, refit="YS")
    return sf, choices


# ----------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    prereg = CODE / "PREREG.md"
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    log(f"PREREG sha256 {prereg_sha}")

    # ---------------- data ----------------
    fo = ol.load_fo_cot()
    cl_fut, cl_info = ol.load_cftc_cl_futonly(fo["crude"])
    sp_fut, sp_chk = ol.splice_crude_futonly(cl_fut)
    log(f"CFTC_CL futures-only: {cl_info}")
    log(f"spliced crude futures-only: {sp_chk}")

    # ---------------- PRIMARY ----------------
    f_pr = feats(("crude", "F+O"), fo["crude"], "crude_roll")
    sig_pr = ol.rule_primary(f_pr)
    sf_pr = ol.signal_frame(fo["crude"], sig_pr)
    pr_rows = ev(sf_pr, "crude_roll", "primary", "crude", primary=True)
    primary = row_dict(pr_rows, side="long", horizon_w=8)
    log("PRIMARY (crude, F+O OI, crude_roll, long, h=8): "
        + ", ".join(f"{k}={primary[k]}" for k in ("n", "episodes", "mean", "baseline", "excess",
                                                    "hit", "perm_p", "perm_p_exact", "T",
                                                    "period")))

    # ---------------- A: the primary rule elsewhere ----------------
    for s in ("crude_roll_lead5", "crude_cl", "crude_spot"):
        f = feats(("crude", "F+O"), fo["crude"], s)
        ev(ol.signal_frame(fo["crude"], ol.rule_primary(f)), s, "primary", "crude")
    sf_by_market = {"crude": sf_pr}
    for m in ol.MARKETS[1:]:
        f = feats((m, "F+O"), fo[m], ol.TRADED[m])
        sf_by_market[m] = ol.signal_frame(fo[m], ol.rule_primary(f))
        ev(sf_by_market[m], ol.TRADED[m], "primary", m)
    pooled([(sf_by_market[m], ol.TRADED[m], "cot") for m in ol.MARKETS], "primary")

    # ---------------- B: OI basis ----------------
    for s in ("crude_roll", "crude_cl"):
        f = feats(("crude", "fut_CFTC_CL"), cl_fut, s)
        ev(ol.signal_frame(cl_fut, ol.rule_primary(f)), s, "primary", "crude",
           oi_basis="futures_only_CFTC_CL_2010+")
    for s in ("crude_roll", "crude_cl", "crude_spot"):
        f = feats(("crude", "fut_spliced"), sp_fut, s)
        ev(ol.signal_frame(sp_fut, ol.rule_primary(f)), s, "primary", "crude",
           oi_basis="futures_only_spliced_2000+")
    for m in ol.MARKETS[1:]:
        rep = ol.load_legacy_futonly(m)
        f = feats((m, "fut_legacy"), rep, ol.TRADED[m])
        ev(ol.signal_frame(rep, ol.rule_primary(f)), ol.TRADED[m], "primary", m,
           oi_basis="futures_only_legacy_2000-2020")

    # ---------------- C-I: grid on F+O for the 5 markets ----------------
    rules_o = other_rules()
    quad_pool, hy_pool, hys_pool = [], [], []
    for m in ol.MARKETS:
        rep, s = fo[m], ol.TRADED[m]
        f = feats((m, "F+O"), rep, s)
        for (w, p, rf) in GRID_C:
            if (w, p, rf) == (4, 10, True):
                continue                      # = primary, already evaluated as "primary"
            ev(ol.signal_frame(rep, ol.rule_oi_pct(f, w, p, rf)), s, grid_c_name(w, p, rf), m)
        for name, fn in rules_o:
            sf = ol.signal_frame(rep, fn(f))
            ev(sf, s, name, m)
            if name == "quad_classic_4w":
                quad_pool.append((sf, s, "cot"))
            elif name == "hy12m":
                hy_pool.append((sf, s, "cot"))
            elif name == "hy12m_sign":
                hys_pool.append((sf, s, "cot"))
        # F: each quadrant alone, as a long-only and as a short-only setup
        for w in (4, 13):
            q, valid = ol.quadrant(f, w)
            z = np.zeros(len(q), bool)
            for k in (1, 2, 3, 4):
                ev(ol.signal_frame(rep, ol._sig(q == k, z, valid)), s, f"quad{k}_{w}w", m,
                   sides=("long",))
                ev(ol.signal_frame(rep, ol._sig(z, q == k, valid)), s, f"quad{k}_{w}w", m,
                   sides=("short",))
        log(f"grid done for {m} ({time.time() - t0:.0f}s)")
    pooled(quad_pool, "quad_classic_4w")
    pooled(hy_pool, "hy12m")
    pooled(hys_pool, "hy12m_sign")

    tests = pd.concat(TESTS, ignore_index=True)

    # ---------------- multiple testing (h=8, long/short rows) ----------------
    mt = tests[(tests["horizon_w"] == 8) & tests["side"].isin(["long", "short"])].copy()
    mt_q = bh_qvalues(mt["perm_p"].to_numpy())
    tests["bh_q_h8"] = np.nan
    tests.loc[mt.index, "bh_q_h8"] = mt_q
    n_cells = int(mt["perm_p"].notna().sum())
    mtest = dict(cells_h8_long_short=n_cells,
                 n_perm_p_le_0_05=int((mt["perm_p"] <= 0.05).sum()),
                 expected_by_chance_5pct=round(0.05 * n_cells, 1),
                 n_bh_q_le_0_10=int((mt_q <= 0.10).sum()),
                 n_excess_pos=int((mt["excess"] > 0).sum()),
                 note="cells are strongly dependent (overlapping rules/markets); descriptive only")
    log(f"multiple testing: {mtest}")
    primary = row_dict(tests, variant="primary", market="crude", series="crude_roll",
                       side="long", horizon_w=8, oi_basis="F+O")

    # ---------------- OOS: split half ----------------
    sh = fh.split_half(sf_pr, "crude_roll", timing="cot", horizon=8, sides=SIDES3, horizons=H,
                       labels=dict(filter=FILTER, variant="primary", market="crude",
                                   is_primary=False))
    for half in ("H1", "H2"):
        r = row_dict(sh, half=half, side="long", horizon_w=8)
        log(f"split {half} (long h8): n={r['n']} episodes={r['episodes']} mean={r['mean']} "
            f"baseline={r['baseline']} excess={r['excess']} perm_p={r['perm_p']} "
            f"period={r['period']}")

    # ---------------- OOS: walk-forward ----------------
    cands = {grid_c_name(w, p, rf): ol.rule_oi_pct(f_pr, w, p, rf) for (w, p, rf) in GRID_C}
    sf_wf1, wf1_choices = walk_forward_selection(fo["crude"], f_pr, "crude_roll", cands)
    wf1 = fh.evaluate(sf_wf1, "crude_roll", H, timing="cot", sides=SIDES3,
                      labels=dict(filter=FILTER, variant="WF1_yearly_select24",
                                  market="crude", is_primary=False))
    wf1_l8 = row_dict(wf1, side="long", horizon_w=8)
    log(f"WF-1 (long h8): n={wf1_l8['n']} mean={wf1_l8['mean']} baseline={wf1_l8['baseline']} "
        f"excess={wf1_l8['excess']} perm_p={wf1_l8['perm_p']} period={wf1_l8['period']}")
    log("WF-1 choices: " + "; ".join(f"{c['year']}:{c['chosen']}" for c in wf1_choices))
    # WF-2: the primary as the only candidate must reproduce the primary on the test years
    sf_wf2, wf2_choices = walk_forward_selection(fo["crude"], f_pr, "crude_roll",
                                                 {"oi4w_p10_range": sig_pr}, min_n=0)
    test_m = sf_wf2["decision_date"] >= pd.Timestamp("2010-01-01")
    ref = sf_pr.loc[test_m.to_numpy(), "signal"].to_numpy()
    ref_valid = sf_pr.loc[test_m.to_numpy(), "valid"].to_numpy()
    wf2_ok = bool(np.array_equal(sf_wf2.loc[test_m, "signal"].to_numpy(), ref) and
                  np.array_equal(sf_wf2.loc[test_m, "valid"].to_numpy(), ref_valid))
    # Same comparison window for the primary itself (decision dates >= 2010-01-01)
    sf_pr_2010 = sf_pr.copy()
    sf_pr_2010["valid"] = sf_pr_2010["valid"].to_numpy() & test_m.to_numpy()
    pr2010 = fh.evaluate(sf_pr_2010, "crude_roll", (8,), timing="cot", sides=("long",),
                         labels=dict(filter=FILTER, variant="primary_from_2010",
                                     market="crude", is_primary=False))
    log(f"WF-2 sanity (primary as only candidate == primary on test years): {wf2_ok}")
    log(f"primary restricted to 2010+ (long h8): excess={pr2010.iloc[0]['excess']} "
        f"perm_p={pr2010.iloc[0]['perm_p']} n={pr2010.iloc[0]['n']}")

    # ---------------- Strategy ----------------
    st = fh.simulate(sf_pr, "crude_roll", timing="cot",
                     labels=dict(filter=FILTER, variant="primary", market="crude"))
    f_cl = feats(("crude", "F+O"), fo["crude"], "crude_cl")
    sf_cl = ol.signal_frame(fo["crude"], ol.rule_primary(f_cl))
    st_cl = fh.simulate(sf_cl, "crude_cl", timing="cot",
                        labels=dict(filter=FILTER, variant="primary", market="crude"))
    met = st["metrics"]
    for _, r in pd.concat([met, st_cl["metrics"]]).iterrows():
        log(f"strategy {r['series']} {r['strategy']}: n_trades={r.get('n_trades')} "
            f"win={r.get('win_rate')} CAGR={r['ann_return']} sharpe={r['sharpe']} "
            f"maxDD={r['max_dd']} null_p={r.get('null_p_sharpe')}")
    st["trades"].to_csv(WORK / "trades_primary.csv", index=False)

    # ---------------- verdict (pre-declared criteria, primary only) ----------------
    h1 = row_dict(sh, half="H1", side="long", horizon_w=8)
    h2 = row_dict(sh, half="H2", side="long", horizon_w=8)
    s1lo = met[met["strategy"] == "S1_long_only"].iloc[0]
    exc, pp = primary["excess"], primary["perm_p"]
    crit = {
        "primary_excess_pos": bool(exc > 0), "primary_perm_p_le_0_05": bool(pp <= 0.05),
        "primary_perm_p_le_0_20": bool(pp <= 0.20),
        "split_H1_excess_pos": bool(h1["excess"] > 0), "split_H2_excess_pos": bool(h2["excess"] > 0),
        "walk_forward_WF1_excess_pos": bool(wf1_l8["excess"] > 0),
        "S1_long_only_sharpe_pos": bool(s1lo["sharpe"] > 0),
        "S1_long_only_null_p_le_0_10": bool(s1lo["null_p_sharpe"] <= 0.10),
        "negative_rule_excess_neg_and_perm_p_ge_0_95": bool(exc < 0 and pp >= 0.95)}
    robust_all = all(crit[k] for k in ("primary_excess_pos", "primary_perm_p_le_0_05",
                                       "split_H1_excess_pos", "split_H2_excess_pos",
                                       "walk_forward_WF1_excess_pos", "S1_long_only_sharpe_pos",
                                       "S1_long_only_null_p_le_0_10"))
    if np.isnan(exc):
        verdict = "untestable with available data"
    elif robust_all:
        verdict = "positive EV, robust"
    elif exc > 0 and pp <= 0.20:
        verdict = "positive EV, fragile"
    elif exc < 0 and pp >= 0.95:
        verdict = "negative (anti-signal)"
    else:
        verdict = "no edge"
    log(f"criteria: {crit}")
    log(f"VERDICT (pre-declared criteria): {verdict}")

    # ---------------- chart-ready data ----------------
    fr = f_pr.copy()
    fr["signal"] = sig_pr
    weekly_oi = {"dates": [d.date().isoformat() for d in fr["report_date"]],
                 "oi": fr["oi"].tolist(), "dOI4": fr["dOI4"].round(6).tolist(),
                 "pOI4": fr["pOI4"].round(3).tolist(), "pP4": fr["pP4"].round(3).tolist(),
                 "crude_roll": fr["px"].round(6).tolist(), "signal": fr["signal"].tolist()}
    g8 = tests[(tests["horizon_w"] == 8)]
    grid_h8 = g8[["variant", "market", "series", "oi_basis", "side", "n", "episodes", "mean",
                  "baseline", "excess", "hit", "perm_p", "bh_q_h8"]]
    quad = tests[tests["variant"].str.match(r"^quad[1-4]_") & (tests["horizon_w"] == 8)]
    quad_h8 = quad[["variant", "market", "side", "n", "mean", "baseline", "excess", "perm_p"]]
    signal_counts = {}
    for m in ol.MARKETS:
        sf = sf_by_market[m]
        v = sf["valid"].to_numpy()
        signal_counts[m] = dict(valid_weeks=int(v.sum()),
                                long_weeks=int((sf["signal"] == 1).sum()),
                                short_weeks=int((sf["signal"] == -1).sum()))

    larry_def = (
        "Larry Williams reads total open interest as the footprint of commercial short hedging "
        "(How I Made One Million Dollars Last Year Trading Commodities, 1973/1979, per secondary "
        "summaries): in a sideways/trading-range market a LARGE INCREASE in OI is bearish "
        "(commercials adding short hedges) and a LARGE DECREASE is bullish (shorts covering / "
        "public liquidation near bottoms); sources cite '30%' as the size and a student's "
        "summary an 'extreme reduction in 1-2 days'. Operationalised (PREREG): crude, F+O total "
        "OI, 4-report OI change in the bottom 10% of its last 3 years (156 reports) while the "
        "4-report price change is at/below its 3-year median absolute size -> LONG (top 10% -> "
        "SHORT); COT timing; traded on crude_roll; primary cell long, h=8 weeks.")
    sources = [
        "L. Williams, How I Made One Million Dollars Last Year Trading Commodities (Windsor "
        "Books 1973; 3rd ed. 1979) -- OI rule known only via secondary summaries below",
        "financial-spread-betting.com/course/open-interest.html (search extract: 'if open "
        "interest increases greatly in a sideways market, the market is likely to move lower, "
        "as commercial interests are taking out more short contracts, as pointed out by Larry "
        "Williams in his book')",
        "wisertrades.com/summary-of-how-i-made-one-million-dollars-last-year-trading-"
        "commodities-by-larry-r-williams/ (search extract: sharp rise or fall in OI in a trading "
        "range shows the direction commercials expect the breakout to take)",
        "solerinvestments.com/Online-Trading/Larry-Williams-Trading.htm (search extract: '30% "
        "decrease in open interest can be a bullish indication and a 30% increase ... bearish'; "
        "market must be in a trading range)",
        "fxtrading.learncurrencytradingonline.com/best-currency-traders-/73-larry-williams-"
        "trading-strategy.html (search extract: OI increase bearish, decrease bullish)",
        "homeytel.medium.com/open-interest-as-an-indicator-of-bottoms-and-tops-904de3266644 "
        "(Williams student: extreme OI reduction in 1-2 days + bullish indicators = major bottom)",
        "L. Williams, Trade Stocks and Commodities with the Insiders: Secrets of the COT Report "
        "(Wiley 2005), ch. 8-10 incl. 'A New Indicator -- Commercials Versus Total Open Interest' "
        "(table of contents via Wiley-VCH / O'Reilly listings)",
        "ireallytrade.com/newsletters/Larry-Williams-Indicators.pdf (search extract: WILLCO = "
        "commercial net position as % of open interest; 'total Combined Open Interest')",
        "J. Murphy, Technical Analysis of the Financial Markets (1999), ch. 7; Barchart 'Tips on "
        "Technicals: Open Interest' (classic price/OI quadrant)",
        "H. Hong & M. Yogo (2012), What does futures market interest tell us about the "
        "macroeconomy and asset prices?, Journal of Financial Economics 105(3):473-490 (NBER "
        "w16712)",
        "Data: S/data/cot_weekly.csv (CFTC disaggregated F+O, 2006-06-13..2026-09-29); "
        "S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json (crude futures-only "
        "2010-01-05..2026-09-22 used); S/work2/scout/cot_smalltrader_weekly.csv (legacy "
        "futures-only 2000-2020, from github.com/bgithub1/cme_open_interest @a5772c97); traded "
        "series from fund_harness (crude_roll from EIA RCLC1/RCLC2)"]
    caveats = [
        "No primary Larry Williams text could be opened from this sandbox (DNS fails for every "
        "source host); the rule is reconstructed from search-engine extracts of secondary "
        "summaries, which partly conflict (a trend-confirmation reading also circulates).",
        "Larry's literal '30%' cannot be used as the primary: crude F+O total OI never falls 30% "
        "below its 26-week high (lowest ratio 0.705). It is >=30% below its 52-week high in "
        "only 19 of 1060 reports: 1.8%, or 1.9% of the 1009 reports with a full 52-week "
        "window. Gold: 222 reports, 21% or 22.0%. 'Large' is therefore defined relative to "
        "each market's last 3 years (variant larry30_* keeps the literal number).",
        "Weekly COT OI cannot see Larry's '1-2 day' OI collapses; the 1-report change is the "
        "finest resolution available (grid C, W=1).",
        "F+O OI includes delta-adjusted options, whose monthly expiries move total OI; Larry's "
        "1970s rule used futures-only exchange OI. Futures-only crude OI (2010+, and spliced "
        "2000+) is tested as variant B.",
        "The primary window is short (crude_roll ends 2024-04-05; F+O COT starts 2006-06 and the "
        "3-year percentile needs 52 values): about 2007-07..2024-01, with few signal episodes, "
        "so power is low; crude_cl extends the same rule to 2026.",
        "Forward returns over 1-13 weeks overlap; only permutation p-values (circular shift) are "
        "reported, no naive t-stats. The grid is large and its cells are dependent; it is "
        "robustness only and the best cell is never the result.",
        "Pooled cells have no exhaustive permutation p (joint offset space).",
        # Added by the adversarial verifier (adv_verify_oi.py / adv_verify_wf.py; numbers on
        # the same crude_roll series, embedded below as 'adversarial_verification').
        "SINGLE-EPISODE DEPENDENCE (adversarial check): the primary's small positive excess "
        "(+0.0088 over 42 weeks, summed excess +0.371) comes entirely from one 3-week episode, "
        "2021-08-17 (summed excess +0.669). Leaving that episode out gives excess -0.0076. "
        "Leaving out the 2021 signal weeks gives -0.0228. Leaving out the 2015-05-19 episode "
        "(summed excess -0.981) gives +0.0356. The criterion 'primary excess > 0' therefore "
        "rests on one episode.",
        "No long signal week of the primary is held across 2020-04-20 (the dropped -37.63 "
        "print) or across any of the 12 largest daily crude_roll moves since 2007, so roll and "
        "negative-price artifacts do not touch the signal returns. They do enter the baseline: "
        "8 eligible weeks span 2020-04-20.",
        "Overlap check: one observation per episode (its first week, 27 entries, a few still "
        "less than 8 weeks apart) gives excess +0.0112 and hit 0.556, close to the all-weeks "
        "figure. Within-episode overlap does not inflate the result.",
        "WF-1 rests on 8 episodes. One week (2022-01-04, r8 +33.2%) supplies 35% of its summed "
        "excess; without that episode the WF-1 excess is +0.0240.",
        "Strategy: S1 long-only Sharpe 0.174 exceeds buy & hold's 0.104, but random timing does "
        "about as well. Random-timing null: median 0.040, p 0.285 (Monte Carlo); an exhaustive "
        "run over all 825 shifts gives median 0.036, p 0.270. A Sharpe above buy & hold is "
        "therefore no evidence of timing skill.",
    ]
    oos = {"split_half": sh, "walk_forward": {
        "WF0_note": "The primary has no outcome-fitted parameter; its percentile thresholds use "
                    "only the trailing 156 reports up to and including t, so it is walk-forward "
                    "by construction.",
        "WF1_yearly_selection_rows": wf1, "WF1_choices": wf1_choices,
        "WF1_long_h8": wf1_l8,
        "WF2_primary_only_reproduces_primary": wf2_ok,
        "primary_restricted_2010plus_long_h8": pr2010.iloc[0].to_dict()}}
    strategy = pd.concat([met, st_cl["metrics"]], ignore_index=True)

    # POST-HOC diagnostics (posthoc_gold.py, written after seeing the grid) are embedded
    # verbatim when present; they never enter the verdict.
    ph_path = WORK / "posthoc_gold.json"
    post_hoc = ({"note": "POST-HOC, not pre-registered, not used for the verdict; produced by "
                         "posthoc_gold.py after the grid showed a cluster of gold cells",
                 "gold_primary_rule": __import__("json").load(open(ph_path))}
                if ph_path.exists() else None)
    # Adversarial re-derivation (adv_verify_oi.py --harness-ltd: own code, harness LTDs only),
    # embedded verbatim when present; it never changes the verdict.
    av_path = WORK / "adv_verify_oi_harnessltd.json"
    adversarial = ({"note": "independent re-derivation by adv_verify_oi.py (no harness "
                            "evaluation code; crude_roll rebuilt from raw RCLC1/RCLC2 with the "
                            "harness's contract-1 last trading days)",
                    "result": __import__("json").load(open(av_path))}
                   if av_path.exists() else None)

    tests.to_csv(WORK / "tests.csv", index=False)
    fh.write_results(
        WORK / "results.json", filter=FILTER, larry_definition=larry_def, sources=sources,
        primary=primary, tests=tests, strategy=strategy, oos=oos, caveats=caveats,
        verdict=verdict, verdict_criteria=crit, multiple_testing=mtest,
        prereg_sha256=prereg_sha, prereg_path=str(prereg),
        data_checks={"cftc_cl": cl_info, "spliced_futures_only": sp_chk},
        signal_counts_primary_rule=signal_counts, post_hoc=post_hoc,
        adversarial_verification=adversarial,
        series_info=fh.describe_series(["crude_roll", "crude_roll_lead5", "crude_cl",
                                        "crude_spot", "gold", "silver", "soybean", "sugar"]),
        charts={"equity_primary_crude_roll": fh.weekly_equity(st["equity"]),
                "equity_primary_crude_cl": fh.weekly_equity(st_cl["equity"]),
                "weekly_oi_crude": weekly_oi, "grid_h8": grid_h8, "quadrants_h8": quad_h8},
        run_seconds=round(time.time() - t0, 1))
    log(f"wrote results.json ({len(tests)} test rows) in {time.time() - t0:.0f}s")
    (WORK / "run_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

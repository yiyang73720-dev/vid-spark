#!/usr/bin/env python3
"""
run_strategies -- trading-strategy lens of the Larry Williams COT backtest.

Simulates, on daily closes with 0.05% cost per side per unit of position change (rf = 0):
  S1 "pure COT"            first tradable week of each N=26 80/20 setup episode -> enter at the
                           entry_date close, hold 40 trading days, one position at a time.
                           Variants: long+short, long-only.
  S2 "COT + price trigger" after a setup's entry_date, enter at the first close within 20
                           trading days that is above (long) / below (short) its 20-day SMA,
                           hold 40 days, one position at a time. Variants: long+short, long-only.
  S3 "regime"              +1 if the latest tradable idx26 >= 80, -1 if <= 20, else 0, updated
                           only at entry_dates. Variants: long+short, long-only.
per market (crude, gold, silver, soybean, sugar) and for an equal-weight 5-market portfolio
(daily-rebalanced average of the market strategy returns), each compared with buy & hold of the
same market(s) over the same window.

Extras (all computed, nothing estimated):
  * random-timing null: idx26 circularly shifted by K=2000 random offsets (seed 11, offsets on
    [26, T-26] like the engine's permutation test); p = (1 + #{null Sharpe >= real}) / (1 + K).
  * S2 definition sensitivity (window starts the day after entry_date; execution at the next
    close; every setup week opens a window instead of only the episode's first week).

Usage:  python3 -I run_strategies.py [--k 2000] [--seed 11]
Outputs go to ROOT/work/lens_strategy (ROOT = $LARRY_COT_ROOT, else the package dir).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = (Path(os.environ.get("LARRY_COT_ROOT") or HERE.parents[1])
               / "work" / "lens_strategy")      # outputs: ROOT/work/lens_strategy
sys.dont_write_bytecode = True          # keep the engine / lens directories free of .pyc
sys.path.insert(0, str(HERE))
import strat_lib as sl  # noqa: E402
lc = sl.lc

MARKETS = list(sl.lc.MARKETS)


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def union_calendar(ctxs: dict) -> pd.DatetimeIndex:
    """Union of the markets' trading days from the common start date to the common end."""
    start = max(c.dates[c.start] for c in ctxs.values())
    end = min(c.dates[-1] for c in ctxs.values())
    u = ctxs[MARKETS[0]].dates
    for c in ctxs.values():
        u = u.union(c.dates)
    return u[(u >= start) & (u <= end)]


def to_union(net: pd.Series, uni: pd.DatetimeIndex) -> np.ndarray:
    """A market's daily net returns on the union calendar: 0 on days that market did not trade
    (its next close-to-close return covers the gap)."""
    return net.reindex(uni).fillna(0.0).to_numpy()


def portfolio_net(nets: dict, uni: pd.DatetimeIndex) -> pd.Series:
    """Equal-weight portfolio, rebalanced daily: the average of the 5 market returns each day."""
    return pd.Series(np.mean([to_union(nets[m], uni) for m in MARKETS], axis=0), index=uni)


def s2_variant(ctx, idx, long_only: bool, **kw) -> tuple:
    """S2 with a non-default definition (used only for the sensitivity table)."""
    first = kw.pop("first_of_episode", True)
    ev = sl.setup_events(ctx, idx, ("long",) if long_only else ("long", "short"),
                         first_of_episode=first)
    tr = sl.s2_trades(ctx, ev, **kw)
    return sl.trades_to_pos(ctx, tr), tr


def iso(d) -> str:
    return pd.Timestamp(d).date().isoformat()


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=lc.K_PERM, help="null-model draws")
    ap.add_argument("--seed", type=int, default=lc.DEFAULT_SEED)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- data via the reconciled engine ------------------------------------------------
    panel = lc.build_panel()
    prices = lc.load_prices()
    ctxs = {m: sl.make_ctx(m, panel[m], prices[m]) for m in MARKETS}
    for m, c in ctxs.items():
        ep = c.entry_pos[sl.tradable_mask(c)]
        assert np.all(np.diff(ep) > 0), f"{m}: entry rows must be strictly increasing"
    uni = union_calendar(ctxs)

    # ---- real runs -----------------------------------------------------------------------
    runs = {m: sl.run_market(ctxs[m]) for m in MARKETS}
    nets, helds, trades = {}, {}, {}
    for m in MARKETS:
        c = ctxs[m]
        for s, (pos, tr) in runs[m].items():
            nets[(m, s)] = sl.daily_net(c, pos)
            helds[(m, s)] = sl.held_mask(c, pos)
            trades[(m, s)] = sl.trade_returns(c, tr)
        bh = np.zeros(c.n)
        bh[c.start:] = 1.0
        nets[(m, "buy_hold")] = sl.daily_net(c, bh, cost=0.0)
        helds[(m, "buy_hold")] = sl.held_mask(c, bh)

    # Invariants of the simulated trades.
    for (m, s), tr in trades.items():
        c = ctxs[m]
        ent = [t["entry"] for t in tr]
        assert all(t["entry"] >= c.start for t in tr)
        if s.startswith(("S1", "S2")):
            # non-overlapping, strictly separated, entry never before the signal's entry day
            for a, b in zip(tr[:-1], tr[1:]):
                assert b["entry"] > a["exit"], (m, s)
            for t in tr:
                assert t["entry"] >= c.entry_pos[t["signal_report"]]
                assert t["open"] or t["days_held"] == sl.HOLD
            if s.startswith("S1"):
                for t in tr:
                    assert t["entry"] == c.entry_pos[t["signal_report"]]
                    if t["side"] == "long" and not t["open"]:
                        r8 = panel[m]["r8"].iloc[t["signal_report"]]
                        assert t["price_ret"] == r8, (m, s, t["entry_date"])
            if s.startswith("S2"):
                for t in tr:
                    j = t["entry"]
                    ok = (c.close[j] > c.sma[j]) if t["side"] == "long" else (c.close[j] < c.sma[j])
                    assert ok and j - c.entry_pos[t["signal_report"]] < sl.TRIG_WIN
        assert ent == sorted(ent)

    # ---- metrics -------------------------------------------------------------------------
    strategies = list(sl.STRATEGIES)
    rows = []
    port = {}
    for s in strategies + ["buy_hold"]:
        port[s] = portfolio_net({m: nets[(m, s)] for m in MARKETS}, uni)
    for m in MARKETS + ["portfolio"]:
        bh = (sl.perf_metrics(nets[(m, "buy_hold")], helds[(m, "buy_hold")]) if m != "portfolio"
              else sl.perf_metrics(port["buy_hold"]))
        for s in strategies + ["buy_hold"]:
            if m == "portfolio":
                pm = sl.perf_metrics(port[s])
                pm["time_in_mkt"] = float(np.mean([np.mean(helds[(mm, s)]) for mm in MARKETS]))
                tr = [t for mm in MARKETS for t in trades.get((mm, s), [])]
            else:
                pm = sl.perf_metrics(nets[(m, s)], helds[(m, s)])
                tr = trades.get((m, s), [])
            ts = sl.trade_stats(tr) if s != "buy_hold" else {}
            rows.append(dict(market=m, strategy=s, **ts, **pm,
                             bh_ann_return=bh["ann_return"], bh_ann_vol=bh["ann_vol"],
                             bh_sharpe=bh["sharpe"], bh_max_dd=bh["max_dd"],
                             bh_total_return=bh["total_return"]))
    metrics = pd.DataFrame(rows)

    # ---- random-timing null (circularly shifted idx26) --------------------------------------
    rng = np.random.default_rng(args.seed)
    T = {m: int(ctxs[m].idx_defined.sum()) for m in MARKETS}
    offs = {m: rng.integers(lc.MIN_SHIFT, T[m] - lc.MIN_SHIFT, size=args.k, endpoint=True)
            for m in MARKETS}
    years = {m: (ctxs[m].dates[-1] - ctxs[m].dates[ctxs[m].start]).days / 365.25
             for m in MARKETS}
    uyears = (uni[-1] - uni[0]).days / 365.25
    upos = {m: uni.get_indexer(ctxs[m].dates[ctxs[m].start:]) for m in MARKETS}
    for m in MARKETS:
        assert (upos[m] >= 0).all() and len(np.unique(upos[m])) == len(upos[m])
    null = {(m, s): np.empty(args.k) for m in MARKETS + ["portfolio"] for s in strategies}
    for i in range(args.k):
        acc = {s: np.zeros(len(uni)) for s in strategies}
        for m in MARKETS:
            c = ctxs[m]
            res = sl.run_market(c, sl.shift_idx(c, int(offs[m][i])))
            for s, (pos, _) in res.items():
                net = sl.daily_net(c, pos).to_numpy()
                null[(m, s)][i] = sl.sharpe_only(net, years[m])
                acc[s][upos[m]] += net / len(MARKETS)
        for s in strategies:
            null[("portfolio", s)][i] = sl.sharpe_only(acc[s], uyears)
    np.savez_compressed(out / "null_sharpes.npz",
                        **{f"{m}__{s}": v for (m, s), v in null.items()})
    nm, npv, n05, n95 = [], [], [], []
    for _, r in metrics.iterrows():
        key = (r["market"], r["strategy"])
        if key in null:
            v = null[key]
            nm.append(float(np.median(v)))
            n05.append(float(np.quantile(v, 0.05)))
            n95.append(float(np.quantile(v, 0.95)))
            npv.append((1 + int(np.sum(v >= r["sharpe"]))) / (1 + args.k))
        else:
            nm.append(np.nan); npv.append(np.nan); n05.append(np.nan); n95.append(np.nan)
    metrics["null_sharpe_median"] = nm
    metrics["null_sharpe_p05"] = n05
    metrics["null_sharpe_p95"] = n95
    metrics["null_p_sharpe"] = npv
    metrics.to_csv(out / "metrics.csv", index=False)

    # ---- trade lists ---------------------------------------------------------------------
    trow = []
    for (m, s), tr in trades.items():
        f = panel[m]
        for t in tr:
            j = t["signal_report"]
            trow.append(dict(
                market=m, strategy=s, side=t["side"],
                signal_report_date=iso(f["report_date"].iloc[j]) if j >= 0 else "",
                signal_idx26=float(f["idx26"].iloc[j]) if j >= 0 else np.nan,
                episode_start_delayed_by_shutdown=t["delayed_by_shutdown"],
                setup_entry_date=iso(f["entry_date"].iloc[j]) if j >= 0 else "",
                trigger_date=iso(ctxs[m].dates[t["trigger"]]),
                entry_date=iso(t["entry_date"]), entry_price=t["entry_price"],
                exit_date=iso(t["exit_date"]), exit_price=t["exit_price"],
                days_held=t["days_held"], open_at_end=t["open"],
                price_ret=t["price_ret"], net_ret=t["net_ret"]))
    trades_df = pd.DataFrame(trow)
    trades_df.to_csv(out / "trades_all.csv", index=False)
    crude_tr = trades_df[(trades_df.market == "crude")
                         & trades_df.strategy.str.startswith(("S1", "S2"))]
    crude_tr.to_csv(out / "crude_trades_S1_S2.csv", index=False)

    # ---- S2 sensitivity --------------------------------------------------------------------
    variants = {
        "S2_main(lag0,exec0,episode)": dict(),
        "S2_window_starts_next_day": dict(lag=1),
        "S2_execute_next_close": dict(exec_delay=1),
        "S2_every_setup_week": dict(first_of_episode=False),
    }
    srows = []
    for vname, kw in variants.items():
        for lo in (False, True):
            vnets, vtr = {}, []
            for m in MARKETS:
                c = ctxs[m]
                pos, tr = s2_variant(c, c.idx, lo, **dict(kw))
                vnets[m] = sl.daily_net(c, pos)
                tr = sl.trade_returns(c, tr)
                vtr += tr
                srows.append(dict(variant=vname, side_mode="long_only" if lo else "long_short",
                                  market=m, **sl.trade_stats(tr),
                                  **sl.perf_metrics(vnets[m], sl.held_mask(c, pos))))
            srows.append(dict(variant=vname, side_mode="long_only" if lo else "long_short",
                              market="portfolio", **sl.trade_stats(vtr),
                              **sl.perf_metrics(portfolio_net(vnets, uni))))
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "s2_sensitivity.csv", index=False)

    # ---- crude on the EIA Cushing spot series (no futures roll gaps) -------------------------
    # Same COT signals and timing rule, entry/exit/P&L on spot closes. Truncated at the futures
    # end date so the window is comparable. Spot is not directly tradable; robustness only.
    spot = lc.load_wti_spot()
    spot = spot[spot.index <= ctxs["crude"].dates[-1]]
    fspot = lc.build_panel(markets=["crude"], price_overrides={"crude": spot})["crude"]
    cspot = sl.make_ctx("crude", fspot, spot)
    srun = sl.run_market(cspot)
    sprow = []
    for s, (pos, tr) in srun.items():
        tr = sl.trade_returns(cspot, tr)
        sprow.append(dict(market="crude_wti_spot", strategy=s, **sl.trade_stats(tr),
                          **sl.perf_metrics(sl.daily_net(cspot, pos), sl.held_mask(cspot, pos))))
    bh = np.zeros(cspot.n)
    bh[cspot.start:] = 1.0
    sprow.append(dict(market="crude_wti_spot", strategy="buy_hold",
                      **sl.perf_metrics(sl.daily_net(cspot, bh, cost=0.0),
                                        sl.held_mask(cspot, bh))))
    pd.DataFrame(sprow).to_csv(out / "crude_spot_sensitivity.csv", index=False)

    # ---- equity curves: daily CSV + weekly chart JSON --------------------------------------
    eq_cols = {}
    for s in strategies + ["buy_hold"]:
        eq_cols[f"crude__{s}"] = (1.0 + nets[("crude", s)]).cumprod()
        eq_cols[f"portfolio__{s}"] = (1.0 + port[s]).cumprod()
    eq = pd.DataFrame(eq_cols)            # union calendar; crude has NaN on non-crude days
    eq.index.name = "date"
    eq.to_csv(out / "equity_daily.csv")
    # Weekly sample = the last trading day (union calendar) of each Saturday-Friday week, keeping
    # its real date; crude is forward-filled over days only the other markets traded.
    eqf = eq.ffill()
    wk = eqf.groupby(eqf.index.to_period("W-FRI")).tail(1)
    chart_series = ["S1_long_short", "S2_long_short", "S3_long_short", "buy_hold",
                    "S1_long_only", "S2_long_only", "S3_long_only"]

    def block(prefix):
        return {"dates": [d.date().isoformat() for d in wk.index],
                "series": {s: [round(float(v), 6) for v in wk[f"{prefix}__{s}"]]
                           for s in chart_series}}
    mk = {}
    for s in ("S1_long_short", "S2_long_short", "S1_long_only", "S2_long_only"):
        mk[s] = [dict(side=t["side"], entry_date=iso(t["entry_date"]),
                      exit_date=iso(t["exit_date"]), entry_price=round(t["entry_price"], 4),
                      exit_price=round(t["exit_price"], 4), net_ret=round(t["net_ret"], 6),
                      open=bool(t["open"])) for t in trades[("crude", s)]]
    charts = {
        "meta": {
            "description": "Weekly-sampled (last trading day of each Mon-Fri week) equity "
                           "curves, equity = 1 just before the first close of the window. "
                           "Costs 0.05% per side per unit of position change; rf = 0; "
                           "buy_hold has no cost. Portfolio = equal-weight 5 markets, "
                           "daily-rebalanced average of market strategy returns.",
            "markets": MARKETS, "index": "idx26 (commercials), thresholds 80/20",
            "hold_days": sl.HOLD, "trigger_window_days": sl.TRIG_WIN, "sma_days": sl.SMA_N,
            "cost_per_side": sl.COST,
            "window_start": iso(uni[0]), "window_end": iso(uni[-1]),
            "series_labels": {
                "S1_long_short": "S1 pure COT (long+short)", "S1_long_only": "S1 pure COT (long only)",
                "S2_long_short": "S2 COT + 20d-SMA trigger (long+short)",
                "S2_long_only": "S2 COT + 20d-SMA trigger (long only)",
                "S3_long_short": "S3 regime (long+short)", "S3_long_only": "S3 regime (long only)",
                "buy_hold": "Buy & hold"},
        },
        "crude": {**block("crude"), "trade_markers": mk},
        "portfolio": block("portfolio"),
        "summary": [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
             for k, v in r.items()
             if k in ("market", "strategy", "n_trades", "win_rate", "avg_trade", "ann_return",
                      "ann_vol", "sharpe", "max_dd", "time_in_mkt", "null_p_sharpe")}
            for r in metrics[metrics.market.isin(["crude", "portfolio"])].to_dict("records")],
    }
    (out / "charts.json").write_text(json.dumps(charts, indent=1))

    # ---- console summary ---------------------------------------------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["market", "strategy", "n_trades", "n_open", "win_rate", "avg_trade", "ann_return",
            "ann_vol", "sharpe", "max_dd", "time_in_mkt", "bh_ann_return", "bh_sharpe",
            "bh_max_dd", "null_sharpe_median", "null_p_sharpe"]
    print(metrics[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"window {iso(uni[0])} .. {iso(uni[-1])}; null K={args.k} seed={args.seed}; "
          f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

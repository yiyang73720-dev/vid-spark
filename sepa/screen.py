"""Combine the four SEPA pillars for each ticker in a data directory.

T  Trend        mechanical: 8-point Trend Template + Weinstein Stage 2
E  Earnings     mechanical when fundamentals JSON is present, else "no_data"
A  Catalyst     never mechanical: left as "research_required" for a human or agent
S  Entry        mechanical: VCP pivot status
"""

import os
from datetime import date

from . import data as data_mod
from .config import Config
from .fundamentals import analyze_fundamentals
from .indicators import rnd, weekly
from .rs import percentile_ratings, rs_line, weighted_performance
from .trend import trend_template, weinstein_stage
from .vcp import analyze_entry

STALE_DAYS = 5


def trend_pillar(trend, stage):
    if stage.get("stage") != 2 or trend["failed"]:
        return "fail"
    if trend["passes_all"]:
        return "pass"
    return "unverified"  # only unknowns left (usually the RS rating)


def earnings_pillar(fund):
    if fund is None:
        return "no_data"
    return {True: "pass", False: "fail"}.get(fund["core_pass"], "incomplete")


def evaluate(ticker, bars, fundamentals, rs_rating, rs_score, bench, cfg, asof=None):
    trend = trend_template(bars, rs_rating, cfg.trend)
    stage = weinstein_stage(bars, cfg.stage)
    entry = analyze_entry(bars, cfg.vcp)
    fund = analyze_fundamentals(fundamentals, cfg.fundamentals, asof) if fundamentals else None
    pillars = {
        "trend": trend_pillar(trend, stage),
        "earnings": earnings_pillar(fund),
        "catalyst": "research_required",
        "entry": entry["status"],
    }
    return {
        "ticker": ticker,
        "last_date": bars.dates[-1],
        "price": rnd(bars.close[-1]),
        "pillars": pillars,
        "candidate": pillars["trend"] in ("pass", "unverified"),
        "trend": trend,
        "stage": stage,
        "rs": {
            "rating": rs_rating,
            "weighted_performance": rnd(rs_score),
            "line": rs_line(bars, bench, cfg.rs.rs_line_bars),
        },
        "fundamentals": fund,
        "entry": entry,
        "weekly": weekly(bars, 30),
    }


def summary_row(result):
    entry = result["entry"]
    fund = result["fundamentals"] or {}
    history = fund.get("history") or [{}]
    return {
        "ticker": result["ticker"],
        "last_date": result["last_date"],
        "price": result["price"],
        "stage": result["stage"].get("stage"),
        "trend_passed": result["trend"]["passed"],
        "trend_failed": result["trend"]["failed"],
        "rs_rating": result["rs"]["rating"],
        "pillars": result["pillars"],
        "candidate": result["candidate"],
        "eps_yoy": history[-1].get("eps_yoy"),
        "revenue_yoy": history[-1].get("revenue_yoy"),
        "entry_status": entry["status"],
        "pivot": entry.get("pivot"),
        "extension": entry.get("extension"),
        "below_pivot": entry.get("below_pivot"),
        "stop": entry.get("stop"),
        "stale": result.get("stale", False),
    }


def scan(data_dir, tickers=None, benchmark="SPY", asof=None, rs_overrides=None, cfg=None,
         log=lambda msg: None):
    """Evaluate ``tickers`` (default: every CSV in the data dir except the benchmark).

    RS ratings rank every price file in the data directory, so a broader
    universe gives a more meaningful criterion 8.
    """
    cfg = cfg or Config()
    rs_overrides = {k.upper(): v for k, v in (rs_overrides or {}).items()}
    benchmark = (benchmark or "").upper() or None
    universe = data_mod.list_tickers(data_dir)
    if not universe:
        raise SystemExit(f"no price CSVs found in {data_mod.prices_dir(data_dir)}")

    loaded, errors = {}, {}
    for ticker in universe:
        try:
            bars = data_mod.load_prices_csv(data_mod.price_path(data_dir, ticker))
        except (OSError, ValueError) as exc:
            errors[ticker] = str(exc)
            continue
        if asof:
            bars = bars.until(asof)
        if len(bars):
            loaded[ticker] = bars
        else:
            errors[ticker] = "no bars on or before the as-of date"

    bench = loaded.get(benchmark) if benchmark else None
    rs_universe = {t: b for t, b in loaded.items() if t != benchmark}
    scores = {t: weighted_performance(b.close, cfg.rs.weights) for t, b in rs_universe.items()}
    ratings = percentile_ratings(scores, cfg.rs.min_universe)
    ratings.update(rs_overrides)

    wanted = [t.upper() for t in tickers] if tickers else sorted(rs_universe)
    last_date = max(b.dates[-1] for b in loaded.values()) if loaded else None
    results = []
    for ticker in wanted:
        if ticker not in loaded:
            errors.setdefault(ticker, "no price data")
            continue
        bars = loaded[ticker]
        fundamentals = data_mod.load_fundamentals(data_mod.fundamentals_path(data_dir, ticker))
        result = evaluate(ticker, bars, fundamentals, ratings.get(ticker), scores.get(ticker),
                          bench, cfg, asof)
        lag = (date.fromisoformat(last_date) - date.fromisoformat(bars.dates[-1])).days
        result["stale"] = lag > STALE_DAYS
        results.append(result)
        log(f"{ticker}: stage {result['stage'].get('stage')}, "
            f"TT {result['trend']['passed']}/8, entry {result['entry']['status']}")

    market = None
    if bench is not None:
        market = {
            "benchmark": benchmark,
            "stage": weinstein_stage(bench, cfg.stage),
            "trend": trend_template(bench, None, cfg.trend)["passes_price_criteria"],
        }

    ranked = sum(1 for v in scores.values() if v is not None)
    return {
        "asof": asof or last_date,
        "data_dir": os.path.abspath(data_dir),
        "rs_universe_size": ranked,
        "rs_note": (
            "" if ranked >= cfg.rs.min_universe else
            f"only {ranked} stocks ranked (< {cfg.rs.min_universe}): RS ratings unknown, "
            "criterion 8 needs --rs overrides or a broader universe"
        ),
        "market": market,
        "config": cfg.to_dict(),
        "results": results,
        "errors": errors,
    }

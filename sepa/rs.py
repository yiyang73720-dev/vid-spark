"""Relative Strength: IBD-style percentile rating and the RS line vs a benchmark."""

from bisect import bisect_left, bisect_right

from .indicators import rnd


def weighted_performance(closes, weights):
    """0.4*3m + 0.2*6m + 0.2*9m + 0.2*12m price change (IBD-style), or None."""
    n = len(closes)
    total = 0.0
    for lag, weight in weights:
        if n <= lag or closes[-1 - lag] <= 0:
            return None
        total += weight * (closes[-1] / closes[-1 - lag] - 1)
    return total


def percentile_ratings(scores, min_universe):
    """Map {ticker: score} to {ticker: 1..99} percentile ratings.

    Ties share the average rank. Returns None ratings for every ticker when
    fewer than ``min_universe`` scores exist, since a percentile of a handful
    of hand-picked stocks is not a relative strength rating.
    """
    ranked = sorted(v for v in scores.values() if v is not None)
    count = len(ranked)
    if count < max(2, min_universe):
        return {ticker: None for ticker in scores}
    out = {}
    for ticker, score in scores.items():
        if score is None:
            out[ticker] = None
            continue
        below = bisect_left(ranked, score)
        ties = bisect_right(ranked, score) - below
        position = (below + (ties - 1) / 2) / (count - 1)
        out[ticker] = int(min(99, max(1, round(1 + 98 * position))))
    return out


def rs_line(bars, bench, window):
    """Stock / benchmark close ratio on shared dates, and whether it sits at a new high."""
    if bench is None or not len(bench):
        return None
    bench_close = dict(zip(bench.dates, bench.close))
    ratio = [c / bench_close[d] for d, c in zip(bars.dates, bars.close) if d in bench_close]
    if len(ratio) < 2:
        return None
    recent = ratio[-window:]
    peak = max(recent)
    return {
        "bars": len(recent),
        "at_new_high": ratio[-1] >= peak,
        "below_high": rnd(1 - ratio[-1] / peak),
        "change_3m": rnd(ratio[-1] / ratio[-64] - 1) if len(ratio) > 63 else None,
    }

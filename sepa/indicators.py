"""Small, dependency-free series helpers."""

from datetime import date


def sma(values, n):
    """Simple moving average; entries before the first full window are None."""
    out = [None] * len(values)
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def argmax(values, start, end):
    """Index of the first maximum of values[start:end]."""
    best = start
    for i in range(start + 1, end):
        if values[i] > values[best]:
            best = i
    return best


def argmin(values, start, end):
    """Index of the first minimum of values[start:end]."""
    best = start
    for i in range(start + 1, end):
        if values[i] < values[best]:
            best = i
    return best


def true_range_pct(bars, i):
    """True range of bar i as a fraction of the prior close."""
    if i == 0:
        return (bars.high[0] - bars.low[0]) / bars.close[0]
    prev = bars.close[i - 1]
    return (max(bars.high[i], prev) - min(bars.low[i], prev)) / prev


def weekly(bars, weeks=None):
    """Aggregate daily bars into ISO weeks: list of dicts, oldest first."""
    out = []
    key = None
    for i, day in enumerate(bars.dates):
        iso = date.fromisoformat(day).isocalendar()
        week_key = (iso[0], iso[1])
        if week_key != key:
            key = week_key
            out.append({
                "week_end": day, "open": bars.open[i], "high": bars.high[i],
                "low": bars.low[i], "close": bars.close[i], "volume": bars.volume[i],
            })
        else:
            row = out[-1]
            row["week_end"] = day
            row["high"] = max(row["high"], bars.high[i])
            row["low"] = min(row["low"], bars.low[i])
            row["close"] = bars.close[i]
            row["volume"] += bars.volume[i]
    return out[-weeks:] if weeks else out


def rnd(value, digits=4):
    return None if value is None else round(value, digits)

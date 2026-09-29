"""Price and fundamentals loading.

Data directory layout::

    <data-dir>/prices/<TICKER>.csv         daily OHLCV, oldest or newest first
    <data-dir>/fundamentals/<TICKER>.json  quarterly results (optional)

CSV headers are matched case-insensitively, so exports from Yahoo Finance,
Stooq, TradingView or ``python3 -m sepa fetch`` all load unchanged.
"""

import csv
import json
import os
from dataclasses import dataclass
from datetime import date

_DATE_KEYS = ("date", "datetime", "timestamp", "time")
_CLOSE_KEYS = ("close", "adj close", "adjclose", "adj_close")


@dataclass
class Bars:
    dates: list
    open: list
    high: list
    low: list
    close: list
    volume: list

    def __len__(self):
        return len(self.close)

    def head(self, end):
        """The first ``end`` bars (bars[0:end])."""
        return Bars(
            self.dates[:end], self.open[:end], self.high[:end],
            self.low[:end], self.close[:end], self.volume[:end],
        )

    def until(self, asof):
        """Bars dated on or before ``asof`` (ISO date string)."""
        end = 0
        while end < len(self.dates) and self.dates[end] <= asof:
            end += 1
        return self.head(end)


def parse_date(value):
    """ISO date string (YYYY-MM-DD) from common date spellings, or None."""
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        text = text[:10]
    elif len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    else:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _number(value):
    try:
        out = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN -> None


def load_prices_csv(path):
    """Load a daily OHLCV CSV into Bars sorted oldest -> newest."""
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty CSV")
        keys = {name.strip().lower().replace("_", " "): name for name in reader.fieldnames}

        def pick(options):
            for option in options:
                if option.replace("_", " ") in keys:
                    return keys[option.replace("_", " ")]
            return None

        date_key = pick(_DATE_KEYS)
        close_key = pick(_CLOSE_KEYS)
        if date_key is None or close_key is None:
            raise ValueError(f"{path}: need a Date and a Close column, got {reader.fieldnames}")
        open_key, high_key, low_key = pick(("open",)), pick(("high",)), pick(("low",))
        vol_key = pick(("volume", "vol"))

        for row in reader:
            day = parse_date(row.get(date_key))
            close = _number(row.get(close_key))
            if day is None or close is None or close <= 0:
                continue
            o = _number(row.get(open_key)) if open_key else None
            h = _number(row.get(high_key)) if high_key else None
            lo = _number(row.get(low_key)) if low_key else None
            v = _number(row.get(vol_key)) if vol_key else None
            o = o if o and o > 0 else close
            h = max(h if h and h > 0 else close, o, close)
            lo = min(lo if lo and lo > 0 else close, o, close)
            rows[day] = (o, h, lo, close, v if v and v > 0 else 0.0)

    days = sorted(rows)
    return Bars(
        days,
        [rows[d][0] for d in days],
        [rows[d][1] for d in days],
        [rows[d][2] for d in days],
        [rows[d][3] for d in days],
        [rows[d][4] for d in days],
    )


def load_fundamentals(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def prices_dir(data_dir):
    return os.path.join(data_dir, "prices")


def fundamentals_dir(data_dir):
    return os.path.join(data_dir, "fundamentals")


def list_tickers(data_dir):
    folder = prices_dir(data_dir)
    if not os.path.isdir(folder):
        return []
    return sorted(
        name[:-4].upper() for name in os.listdir(folder) if name.lower().endswith(".csv")
    )


def price_path(data_dir, ticker):
    folder = prices_dir(data_dir)
    exact = os.path.join(folder, f"{ticker}.csv")
    if os.path.exists(exact):
        return exact
    for name in os.listdir(folder) if os.path.isdir(folder) else []:
        if name.lower() == f"{ticker.lower()}.csv":
            return os.path.join(folder, name)
    return exact


def fundamentals_path(data_dir, ticker):
    return os.path.join(fundamentals_dir(data_dir), f"{ticker}.json")

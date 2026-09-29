"""Optional downloader: daily prices and quarterly fundamentals via yfinance.

Needs ``pip install yfinance`` and network access to Yahoo Finance. Everything
else in the package works from CSV/JSON you provide yourself, so this module
is only imported by ``python3 -m sepa fetch``.
"""

import json
import math
import os
from datetime import date

from .data import fundamentals_dir, fundamentals_path, prices_dir

_EPS_ROWS = ("Diluted EPS", "Basic EPS")
_ROWS = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "operating_income": ("Operating Income",),
    "gross_profit": ("Gross Profit",),
}


def _clean(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _row(frame, names, column):
    for name in names:
        if name in frame.index:
            return _clean(frame.at[name, column])
    return None


def prices_to_rows(history):
    """yfinance ``Ticker.history`` DataFrame -> list of CSV rows (oldest first)."""
    rows = []
    for stamp, rec in history.iterrows():
        close = _clean(rec.get("Close"))
        if close is None:
            continue
        rows.append([
            stamp.date().isoformat(), _clean(rec.get("Open")), _clean(rec.get("High")),
            _clean(rec.get("Low")), close, _clean(rec.get("Volume")) or 0,
        ])
    return rows


def fundamentals_from_frames(income, earnings):
    """Merge ``quarterly_income_stmt`` and ``get_earnings_dates`` into the fundamentals schema.

    Income statement columns give period ends, revenue and margins (usually
    only 4-6 quarters). Earnings dates give reported EPS and the consensus
    estimate for more quarters; each report is matched to the latest period
    that ended within 100 days before it.
    """
    quarters = {}
    if income is not None and not income.empty:
        for column in income.columns:
            end = column.date().isoformat()
            q = {"period_end": end, "eps": _row(income, _EPS_ROWS, column)}
            for key, names in _ROWS.items():
                q[key] = _row(income, names, column)
            quarters[end] = q

    if earnings is not None and not earnings.empty:
        ends = sorted(quarters)
        for stamp, rec in earnings.iterrows():
            reported = _clean(rec.get("Reported EPS"))
            if reported is None:
                continue  # future report date
            day = stamp.date()
            match = [e for e in ends if 0 < (day - _as_date(e)).days <= 100]
            key = match[-1] if match else None
            if key is None:
                # No income-statement column: key the quarter by its report date.
                key = day.isoformat()
                quarters.setdefault(key, {"period_end": None})
            q = quarters[key]
            q["reported"] = day.isoformat()
            q["eps"] = reported  # the reported (adjusted) EPS analysts compare to estimates
            q["eps_estimate"] = _clean(rec.get("EPS Estimate"))
    return [quarters[k] for k in sorted(quarters)]


def _as_date(text):
    return date.fromisoformat(text)


def fetch(tickers, data_dir, period="2y", with_fundamentals=True, log=print):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("fetch needs yfinance: pip install yfinance") from exc

    os.makedirs(prices_dir(data_dir), exist_ok=True)
    os.makedirs(fundamentals_dir(data_dir), exist_ok=True)
    failed = {}
    for ticker in tickers:
        tk = yf.Ticker(ticker)
        try:
            history = tk.history(period=period, interval="1d", auto_adjust=False, actions=False)
            rows = prices_to_rows(history)
            if not rows:
                raise ValueError("no price rows returned")
            with open(os.path.join(prices_dir(data_dir), f"{ticker}.csv"), "w") as fh:
                fh.write("Date,Open,High,Low,Close,Volume\n")
                for row in rows:
                    fh.write(",".join("" if v is None else str(v) for v in row) + "\n")
        except Exception as exc:  # network, delisted ticker, API change
            failed[ticker] = f"prices: {exc}"
            log(f"{ticker}: prices failed ({exc})")
            continue

        if with_fundamentals:
            income = earnings = None
            try:
                income = tk.quarterly_income_stmt
            except Exception as exc:
                log(f"{ticker}: income statement unavailable ({exc})")
            try:
                earnings = tk.get_earnings_dates(limit=16)
            except Exception as exc:
                log(f"{ticker}: earnings dates unavailable ({exc})")
            quarters = fundamentals_from_frames(income, earnings)
            if quarters:
                with open(fundamentals_path(data_dir, ticker), "w") as fh:
                    json.dump({"ticker": ticker, "source": "yfinance", "quarters": quarters},
                              fh, indent=1)
        log(f"{ticker}: {len(rows)} bars")
    return failed

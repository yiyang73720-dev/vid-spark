"""Optional downloader: daily prices and quarterly fundamentals via yfinance.

Needs ``pip install yfinance`` and network access to Yahoo Finance. Everything
else in the package works from CSV/JSON you provide yourself, so this module
is only imported by ``python3 -m sepa fetch``.
"""

import json
import math
import os
from datetime import date, timedelta

from .data import fundamentals_dir, fundamentals_path, prices_dir
from .indicators import median

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
    """First non-missing value among the alternative row labels."""
    for name in names:
        if name in frame.index:
            value = _clean(frame.at[name, column])
            if value is not None:
                return value
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


def fundamentals_from_frames(income, earnings, today=None):
    """Merge ``quarterly_income_stmt`` and ``get_earnings_dates`` into the fundamentals schema.

    Income statement columns give period ends, revenue and margins (Yahoo
    serves about 5 quarters, so revenue acceleration usually needs another
    source). Earnings dates give reported EPS and the consensus estimate for
    more quarters; each past report is matched to the latest period that ended
    within 100 days before it. A report with no matching column gets an
    estimated period end (report date minus the median reporting lag) so the
    year-ago match still lines up.

    EPS uses one basis for every quarter, so YoY growth never compares GAAP
    with adjusted figures: Yahoo's reported (adjusted) EPS when any is
    available, otherwise GAAP diluted EPS. Returns
    ``{"eps_basis": ..., "quarters": [...]}``.
    """
    today = today or date.today()
    quarters = {}
    if income is not None and not income.empty:
        for column in income.columns:
            end = column.date().isoformat()
            q = {"period_end": end, "eps_gaap": _row(income, _EPS_ROWS, column)}
            for key, names in _ROWS.items():
                q[key] = _row(income, names, column)
            quarters[end] = q

    reports = []
    if earnings is not None and not earnings.empty:
        for stamp, rec in earnings.iterrows():
            day = stamp.date()
            if day > today:
                continue  # upcoming report
            reports.append((day, _clean(rec.get("Reported EPS")), _clean(rec.get("EPS Estimate"))))
    reports.sort()

    ends = sorted(quarters)
    lags, unmatched = [], []
    for day, eps, estimate in reports:
        match = [e for e in ends if 0 < (day - date.fromisoformat(e)).days <= 100]
        if not match:
            unmatched.append((day, eps, estimate))
            continue
        lags.append((day - date.fromisoformat(match[-1])).days)
        quarters[match[-1]].update(reported=day.isoformat(), eps_reported=eps, eps_estimate=estimate)
    lag = int(median(lags)) if lags else 35
    for day, eps, estimate in unmatched:
        end = (day - timedelta(days=lag)).isoformat()
        q = quarters.setdefault(end, {"period_end": end, "period_end_estimated": True})
        q.update(reported=day.isoformat(), eps_reported=eps, eps_estimate=estimate)

    basis = ("reported" if any(q.get("eps_reported") is not None for q in quarters.values())
             else "gaap_diluted")
    for q in quarters.values():
        q["eps"] = q.get("eps_reported") if basis == "reported" else q.get("eps_gaap")
    return {"eps_basis": basis, "quarters": [quarters[k] for k in sorted(quarters)]}


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
            merged = fundamentals_from_frames(income, earnings)
            if merged["quarters"]:
                with open(fundamentals_path(data_dir, ticker), "w") as fh:
                    json.dump({"ticker": ticker, "source": "yfinance", **merged}, fh, indent=1)
        log(f"{ticker}: {len(rows)} bars")
    return failed

"""Pillar E: quarterly earnings and revenue acceleration, margins, surprises.

Fundamentals JSON (``<data-dir>/fundamentals/<TICKER>.json``)::

    {
      "quarters": [
        {"period_end": "2025-06-30", "reported": "2025-07-30",
         "eps": 1.23, "eps_estimate": 1.10, "revenue": 1.2e9,
         "net_income": 3.1e8, "operating_income": 3.9e8, "gross_profit": 7.0e8}
      ],
      "guidance": "raised",         # optional: raised | maintained | lowered
      "guidance_date": "2025-07-30" # when that guidance was given (needed with --asof)
    }

Only ``period_end`` (or ``reported``) and ``eps`` are required; checks whose
inputs are missing come back as ``pass: None``.

With ``asof`` a quarter counts only once it was public: on its ``reported``
date, or, when that is missing, ``report_lag_days`` after ``period_end``.
Guidance counts only when its ``guidance_date`` is on or before ``asof``.
"""

from datetime import date, timedelta

from .indicators import rnd

_MARGIN_KEYS = ("net_income", "operating_income", "gross_profit")


def _iso(text):
    try:
        return date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def _day(quarter):
    return _iso(quarter.get("period_end")) or _iso(quarter.get("reported"))


def _public(quarter, cfg):
    """The date a quarter's results became public (or a conservative estimate)."""
    reported = _iso(quarter.get("reported"))
    if reported:
        return reported
    end = _iso(quarter.get("period_end"))
    return end + timedelta(days=cfg.report_lag_days) if end else None


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def growth(current, base):
    """(growth, note). Growth is None when the base is zero/negative."""
    if current is None or base is None:
        return None, "missing"
    if base > 0:
        return current / base - 1, ""
    if current > 0:
        return None, "turnaround (loss to profit)"
    return None, "n.m. (loss in both quarters)"


def _margin(quarter, key):
    revenue = _num(quarter.get("revenue"))
    value = _num(quarter.get(key))
    return value / revenue if value is not None and revenue else None


def _year_ago(quarters, i, cfg):
    """Index of the quarter that ended about one year before quarters[i]."""
    target = _day(quarters[i])
    best, best_gap = None, None
    for j in range(i):
        days = (target - _day(quarters[j])).days
        if cfg.yoy_min_days <= days <= cfg.yoy_max_days:
            gap = abs(days - 365)
            if best_gap is None or gap < best_gap:
                best, best_gap = j, gap
    return best


def _check(cid, name, passed, detail=""):
    return {"id": cid, "name": name, "pass": passed, "detail": detail}


def analyze_fundamentals(data, cfg, asof=None):
    quarters = [q for q in ((data or {}).get("quarters") or []) if isinstance(q, dict) and _day(q)]
    if asof:
        cutoff = _iso(asof)
        quarters = [q for q in quarters if _public(q, cfg) <= cutoff]
    quarters.sort(key=_day)
    if not quarters:
        return {"core_pass": None, "checks": [], "history": [], "missing": ["quarters"]}

    history = []
    for i, q in enumerate(quarters):
        j = _year_ago(quarters, i, cfg)
        prev = quarters[j] if j is not None else {}
        eps_g, eps_note = growth(_num(q.get("eps")), _num(prev.get("eps")))
        rev_g, rev_note = growth(_num(q.get("revenue")), _num(prev.get("revenue")))
        margin_key = next(
            (k for k in _MARGIN_KEYS if _margin(q, k) is not None and _margin(prev, k) is not None),
            None,
        )
        history.append({
            "period_end": _day(q).isoformat(),
            "eps": _num(q.get("eps")),
            "eps_yoy": rnd(eps_g),
            "eps_note": eps_note if j is not None else "no year-ago quarter",
            "revenue": _num(q.get("revenue")),
            "revenue_yoy": rnd(rev_g),
            "margin_type": margin_key,
            "margin": rnd(_margin(q, margin_key)) if margin_key else None,
            "margin_year_ago": rnd(_margin(prev, margin_key)) if margin_key else None,
            "eps_estimate": _num(q.get("eps_estimate")),
        })

    last = history[-1]
    prev = history[-2] if len(history) > 1 else {}

    eps_g = last["eps_yoy"]
    e1 = _check(
        "E1", f"Latest quarter EPS up >= {cfg.min_eps_growth:.0%} YoY",
        (eps_g >= cfg.min_eps_growth) if eps_g is not None else None,
        f"EPS YoY {eps_g:+.1%}" if eps_g is not None else last["eps_note"],
    )

    streak = 0
    for k in range(len(history) - 1, 0, -1):
        a, b = history[k]["eps_yoy"], history[k - 1]["eps_yoy"]
        if a is not None and b is not None and a > b:
            streak += 1
        else:
            break
    prev_eps_g = prev.get("eps_yoy")
    e2 = _check(
        "E2", "EPS growth accelerating (latest YoY > prior quarter's YoY)",
        (eps_g > prev_eps_g) if eps_g is not None and prev_eps_g is not None else None,
        (f"{prev_eps_g:+.1%} -> {eps_g:+.1%}; accelerating {streak} quarter(s) in a row"
         if eps_g is not None and prev_eps_g is not None else "need two YoY comparisons"),
    )

    rev_g, prev_rev_g = last["revenue_yoy"], prev.get("revenue_yoy")
    e3 = _check(
        "E3", "Revenue growing and accelerating",
        (rev_g > prev_rev_g and rev_g >= cfg.min_revenue_growth)
        if rev_g is not None and prev_rev_g is not None else None,
        (f"{prev_rev_g:+.1%} -> {rev_g:+.1%}"
         if rev_g is not None and prev_rev_g is not None else "need two revenue YoY comparisons"),
    )

    m_now, m_ago = last["margin"], last["margin_year_ago"]
    e4 = _check(
        "E4", "Profit margin expanding vs a year ago",
        (m_now > m_ago) if m_now is not None and m_ago is not None else None,
        (f"{last['margin_type']} margin {m_ago:.1%} -> {m_now:.1%}"
         if m_now is not None and m_ago is not None else "need income and revenue"),
    )

    est = last["eps_estimate"]
    e5 = _check(
        "E5", "Beat the consensus EPS estimate",
        (last["eps"] > est) if est is not None and last["eps"] is not None else None,
        f"EPS {last['eps']} vs estimate {est}" if est is not None else "no estimate supplied",
    )

    guidance = str((data or {}).get("guidance") or "").lower() or None
    guidance_note = guidance or "not supplied"
    if guidance and asof:
        given = _iso((data or {}).get("guidance_date"))
        if given is None or given > _iso(asof):
            guidance, guidance_note = None, f"{guidance_note} (ignored: no guidance_date on or before {asof})"
    e6 = _check(
        "E6", "Guidance raised",
        {"raised": True, "lowered": False, "maintained": False}.get(guidance),
        guidance_note,
    )

    core = [e1, e2, e3, e4]
    if any(c["pass"] is False for c in core):
        core_pass = False
    elif all(c["pass"] is True for c in core):
        core_pass = True
    else:
        core_pass = None
    return {
        "core_pass": core_pass,
        "strong_growth": eps_g is not None and eps_g >= cfg.preferred_eps_growth,
        "checks": [e1, e2, e3, e4, e5, e6],
        "history": history[-8:],
        "missing": [c["id"] for c in core if c["pass"] is None],
    }

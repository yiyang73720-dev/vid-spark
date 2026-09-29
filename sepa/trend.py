"""Pillar T: Minervini's 8-point Trend Template and Weinstein stage analysis."""

from .indicators import rnd, sma


def _criterion(cid, name, passed, value=None, threshold=None, detail=""):
    return {
        "id": cid, "name": name, "pass": passed,
        "value": value, "threshold": threshold, "detail": detail,
    }


def ma_up_months(ma, month_bars, max_months=12):
    """Consecutive most-recent months in which ``ma`` rose (each month vs the one before)."""
    months = 0
    last = len(ma) - 1
    for j in range(1, max_months + 1):
        now_i, prev_i = last - month_bars * (j - 1), last - month_bars * j
        if prev_i < 0 or ma[prev_i] is None or ma[now_i] is None:
            break
        if ma[now_i] > ma[prev_i]:
            months += 1
        else:
            break
    return months


def trend_template(bars, rs_rating, cfg):
    """Evaluate the 8 Trend Template criteria on the last bar.

    Each criterion's ``pass`` is True/False, or None when the data needed to
    judge it is missing (too little history, no RS rating). ``passes_all`` is
    True only when all 8 are True.
    """
    closes = bars.close
    n = len(closes)
    price = closes[-1] if n else None
    ma_s, ma_m, ma_l = sma(closes, cfg.ma_short), sma(closes, cfg.ma_mid), sma(closes, cfg.ma_long)
    s, m, l = (ma_s[-1], ma_m[-1], ma_l[-1]) if n else (None, None, None)

    window = min(n, cfg.week52_bars)
    highs = bars.high if cfg.use_intraday_extremes else closes
    lows = bars.low if cfg.use_intraday_extremes else closes
    hi52 = max(highs[n - window:]) if n else None
    lo52 = min(lows[n - window:]) if n else None

    up_months = ma_up_months(ma_l, cfg.month_bars)
    has_slope = n > cfg.ma_long - 1 + cfg.month_bars
    notes = []
    if n < cfg.week52_bars:
        notes.append(f"only {n} bars: 52-week high/low use the available history")

    def both(a, b):
        return a is not None and b is not None

    c = []
    c.append(_criterion(
        1, "Price above the 150-day and 200-day MA",
        (price > m and price > l) if both(m, l) else None,
        value={"price": rnd(price), "ma150": rnd(m), "ma200": rnd(l)},
    ))
    c.append(_criterion(
        2, "150-day MA above the 200-day MA",
        (m > l) if both(m, l) else None,
        value={"ma150": rnd(m), "ma200": rnd(l)},
    ))
    c.append(_criterion(
        3, "200-day MA trending up for at least 1 month",
        (up_months >= cfg.min_ma_long_up_months) if has_slope else None,
        value={"up_months": up_months if has_slope else None},
        threshold=cfg.min_ma_long_up_months,
        detail=(
            f"rising {up_months} consecutive month(s); Minervini prefers "
            f"{cfg.preferred_ma_long_up_months}+" if has_slope else "not enough history"
        ),
    ))
    c.append(_criterion(
        4, "50-day MA above the 150-day and 200-day MA",
        (s > m and s > l) if both(s, m) and l is not None else None,
        value={"ma50": rnd(s), "ma150": rnd(m), "ma200": rnd(l)},
    ))
    c.append(_criterion(
        5, "Price above the 50-day MA",
        (price > s) if both(price, s) else None,
        value={"price": rnd(price), "ma50": rnd(s)},
    ))
    above_low = price / lo52 - 1 if both(price, lo52) else None
    c.append(_criterion(
        6, "Price at least 30% above the 52-week low",
        (above_low >= cfg.min_above_low) if above_low is not None else None,
        value={"low52": rnd(lo52), "above_low": rnd(above_low)},
        threshold=cfg.min_above_low,
    ))
    below_high = 1 - price / hi52 if both(price, hi52) else None
    c.append(_criterion(
        7, "Price within 25% of the 52-week high",
        (below_high <= cfg.max_below_high) if below_high is not None else None,
        value={"high52": rnd(hi52), "below_high": rnd(below_high)},
        threshold=cfg.max_below_high,
    ))
    c.append(_criterion(
        8, "Relative Strength rating at least 70",
        (rs_rating >= cfg.min_rs_rating) if rs_rating is not None else None,
        value={"rs_rating": rs_rating},
        threshold=cfg.min_rs_rating,
        detail="" if rs_rating is not None else (
            "RS rating unknown: rank a larger universe or pass --rs TICKER=NN"
        ),
    ))

    price_criteria = [x["pass"] for x in c[:7]]
    return {
        "criteria": c,
        "passed": sum(1 for x in c if x["pass"] is True),
        "failed": [x["id"] for x in c if x["pass"] is False],
        "unknown": [x["id"] for x in c if x["pass"] is None],
        "passes_price_criteria": all(p is True for p in price_criteria),
        "passes_all": all(x["pass"] is True for x in c),
        "notes": notes,
    }


def weinstein_stage(bars, cfg):
    """Classify the Weinstein stage (1 basing, 2 advancing, 3 topping, 4 declining).

    Uses the 30-week (150-day) MA: its slope over ``slope_bars`` and where price
    sits relative to it. When the two disagree or the MA is flat, the MA's move
    over the preceding ``prior_bars`` separates a base after a decline (Stage 1)
    from a top after an advance (Stage 3).
    """
    closes = bars.close
    n = len(closes)
    ma = sma(closes, cfg.ma)
    if n < cfg.ma + cfg.slope_bars:
        return {"stage": None, "label": "insufficient history", "ma_slope": None}

    price, now, before = closes[-1], ma[-1], ma[-1 - cfg.slope_bars]
    slope = now / before - 1
    rising, falling = slope > cfg.flat_band, slope < -cfg.flat_band

    prior_end = n - 1 - cfg.slope_bars
    prior_start = prior_end - cfg.prior_bars
    prior = None
    if prior_start >= 0 and ma[prior_start] is not None:
        prior = ma[prior_end] / ma[prior_start] - 1

    if rising and price > now:
        stage, label = 2, "Stage 2 - advancing (price above rising 30-week MA)"
    elif falling and price < now:
        stage, label = 4, "Stage 4 - declining (price below falling 30-week MA)"
    elif prior is not None and prior >= cfg.prior_advance:
        stage, label = 3, "Stage 3 - topping (30-week MA stalling after an advance)"
    else:
        stage, label = 1, "Stage 1 - basing (30-week MA flat after a decline or range)"

    # How long the current Stage 2 conditions have held (price above a rising MA).
    days = 0
    i = n - 1
    while i - cfg.slope_bars >= 0 and ma[i - cfg.slope_bars] is not None:
        if closes[i] > ma[i] and ma[i] / ma[i - cfg.slope_bars] - 1 > cfg.flat_band:
            days += 1
            i -= 1
        else:
            break

    return {
        "stage": stage,
        "label": label,
        "ma150": rnd(now),
        "ma_slope": rnd(slope),
        "prior_ma_change": rnd(prior),
        "price_vs_ma": rnd(price / now - 1),
        "stage2_days": days if stage == 2 else 0,
    }

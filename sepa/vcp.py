"""Pillar S: Volatility Contraction Pattern (VCP) detection and pivot entry status.

A base starts at the highest high of the look-back window. Its first
contraction runs from that high to the lowest low that follows. Swings after
that low are found with a percentage zigzag whose threshold scales with the
stock's volatility, and every swing high -> swing low leg is one further
contraction. A leg that is deeper than the one before it but starts from a
*higher* high is a dip in the rally off the lows, not a contraction, and is
dropped. The pivot is the high of the final contraction; its low is the stop.

Entry status (``analyze_entry``):
    forming               valid VCP, price more than ``near_pivot`` below the pivot
    near_pivot            valid VCP, price within ``near_pivot`` of the pivot: set a buy-stop
    breakout              closed above the pivot on >= breakout_volume_ratio x average
                          volume and still within ``max_chase`` of it: buyable
    breakout_unconfirmed  closed above the pivot on light volume
    extended              more than ``max_chase`` above the pivot: do not chase
    failed_breakout       broke out, then closed back below the pivot
    stopped_out           broke out, then closed below the final contraction low
    no_base               no valid VCP (see ``reasons``)
"""

from .indicators import argmax, argmin, mean, median, rnd, true_range_pct


def _zigzag(high, low, start, end, threshold):
    """Swing pivots over bars[start:end], starting from a swing low at ``start``.

    Returns (pivots, direction, extreme): pivots are confirmed (index, "H"|"L")
    pairs; ``direction`` is the leg in progress at ``end`` ("up" or "down") and
    ``extreme`` the index of that leg's running high or low.
    """
    # A bar that sets a new extreme never also confirms the reversal: daily bars
    # don't say whether the high or the low came first, so a single wide bar
    # must not create a swing on its own.
    pivots = [(start, "L")]
    direction, extreme = "up", start
    for i in range(start + 1, end):
        if direction == "up":
            if low[i] < low[pivots[-1][0]]:  # undercut the swing low: move it
                pivots[-1] = (i, "L")
                extreme = i
            elif high[i] >= high[extreme]:
                extreme = i
            elif low[i] <= high[extreme] * (1 - threshold):
                pivots.append((extreme, "H"))
                direction, extreme = "down", i
        else:
            if high[i] > high[pivots[-1][0]]:  # exceeded the swing high: move it
                pivots[-1] = (i, "H")
                extreme = i
            elif low[i] <= low[extreme]:
                extreme = i
            elif high[i] >= low[extreme] * (1 + threshold):
                pivots.append((extreme, "L"))
                direction, extreme = "up", i
    return pivots, direction, extreme


def _depth(high, low, leg):
    return (high[leg[0]] - low[leg[1]]) / high[leg[0]]


def detect_base(bars, end, cfg):
    """Look for a VCP in bars[0:end]. Always returns a dict with ``valid``."""
    high, low, close, vol = bars.high, bars.low, bars.close, bars.volume
    if end < cfg.min_base_bars + 2:
        return {"valid": False, "reasons": ["not enough history"]}

    start = max(0, end - cfg.max_base_bars)
    h0 = argmax(high, start, end)
    base_bars = end - h0
    if base_bars < cfg.min_base_bars:
        return {
            "valid": False,
            "reasons": [f"at or near highs: {base_bars} bar(s) since the high, "
                        f"a base needs {cfg.min_base_bars}+"],
            "base_high": rnd(high[h0]),
            "base_high_date": bars.dates[h0],
        }

    l1 = argmin(low, h0 + 1, end)
    tr = median(true_range_pct(bars, i) for i in range(max(1, h0), end))
    threshold = min(cfg.zigzag_max, max(cfg.zigzag_min, cfg.zigzag_atr_mult * (tr or 0)))
    pivots, direction, extreme = _zigzag(high, low, l1, end, threshold)

    legs = [(h0, l1, False)]
    for k in range(1, len(pivots) - 1):
        if pivots[k][1] == "H" and pivots[k + 1][1] == "L":
            legs.append((pivots[k][0], pivots[k + 1][0], False))
    if direction == "down":
        legs.append((pivots[-1][0], extreme, True))

    # Drop rally dips: a deeper leg that starts from a higher high absorbs the one before it.
    kept = [legs[0]]
    for leg in legs[1:]:
        while (len(kept) > 1 and _depth(high, low, leg) > _depth(high, low, kept[-1])
               and high[leg[0]] > high[kept[-1][0]]):
            kept.pop()
        kept.append(leg)

    contractions = []
    for hi_i, lo_i, still_open in kept:
        contractions.append({
            "high_date": bars.dates[hi_i], "high": rnd(high[hi_i]),
            "low_date": bars.dates[lo_i], "low": rnd(low[lo_i]),
            "depth": rnd(_depth(high, low, (hi_i, lo_i))),
            "bars": lo_i - hi_i,
            "avg_volume": rnd(mean(vol[hi_i:lo_i + 1]), 0),
            "open": still_open,
        })

    depths = [_depth(high, low, (a, b)) for a, b, _ in kept]
    pivot_i, stop_i = kept[-1][0], kept[-1][1]
    pivot, stop = high[pivot_i], low[stop_i]

    prior_from = max(0, h0 - cfg.prior_advance_bars)
    prior_adv = high[h0] / min(low[prior_from:h0]) - 1 if h0 - prior_from >= 20 else None

    avg_vol = mean(vol[max(0, end - cfg.volume_avg_bars):end])
    first_vol, final_vol = contractions[0]["avg_volume"], contractions[-1]["avg_volume"]
    recent = vol[max(h0, end - 10):end]
    dry_days = sum(1 for v in recent if avg_vol and v < cfg.dry_up_ratio * avg_vol)

    count = len(kept)
    checks = [
        ("base_length", cfg.min_base_bars <= base_bars <= cfg.max_base_bars,
         f"{base_bars} bars ({base_bars / 5:.1f} weeks)"),
        ("prior_advance", None if prior_adv is None else prior_adv >= cfg.min_prior_advance,
         f"{prior_adv:+.0%} into the base high" if prior_adv is not None else "not enough history"),
        ("contraction_count", cfg.min_contractions <= count <= cfg.max_contractions,
         f"{count} contraction(s)"),
        ("contractions_shrinking",
         all(depths[k] <= depths[k - 1] + cfg.depth_tolerance for k in range(1, count)),
         " -> ".join(f"{d:.1%}" for d in depths)),
        ("first_depth_ok", depths[0] <= cfg.max_first_depth, f"first contraction {depths[0]:.1%}"),
        ("final_tight", depths[-1] <= cfg.max_final_depth, f"final contraction {depths[-1]:.1%}"),
        ("pivot_near_high", pivot >= high[h0] * (1 - cfg.max_pivot_below_high),
         f"pivot {1 - pivot / high[h0]:.1%} below the base high"),
    ]
    if cfg.require_volume_contraction:
        checks.append((
            "volume_contracting",
            final_vol < first_vol if first_vol and final_vol is not None else None,
            f"avg volume first {first_vol:,.0f} -> final {final_vol:,.0f}"
            if first_vol and final_vol is not None else "no volume data",
        ))
    checks = [{"name": name, "pass": ok, "detail": detail} for name, ok, detail in checks]
    reasons = [f"{c['name']}: {c['detail']}" for c in checks if c["pass"] is False]

    return {
        "valid": not reasons,
        "reasons": reasons,
        "checks": checks,
        "base_start": bars.dates[h0],
        "base_high": rnd(high[h0]),
        "base_bars": base_bars,
        "base_depth": rnd(depths[0]),
        "zigzag_threshold": rnd(threshold),
        "contractions": contractions,
        "pivot": rnd(pivot),
        "pivot_date": bars.dates[pivot_i],
        "pivot_index": pivot_i,
        "stop": rnd(stop),
        "stop_date": bars.dates[stop_i],
        "stop_pct": rnd(1 - stop / pivot),
        "volume": {
            "avg50": rnd(avg_vol, 0),
            "final_vs_avg50": rnd(final_vol / avg_vol) if avg_vol and final_vol is not None else None,
            "dry_up_days_last10": dry_days,
        },
    }


def _recent_failed_breakout(bars, cfg, pivot):
    """An earlier, lower pivot that price closed above and then lost, within the scan window."""
    n = len(bars)
    close = bars.close
    for end in range(n - 1, max(cfg.min_base_bars + 2, n - cfg.breakout_scan_bars) - 1, -1):
        base = detect_base(bars, end, cfg)
        if not base["valid"] or base["pivot"] >= pivot:
            continue
        crossed = next((i for i in range(base["pivot_index"] + 1, n) if close[i] > base["pivot"]),
                       None)
        if crossed is not None and close[-1] < base["pivot"]:
            return {"pivot": base["pivot"], "date": bars.dates[crossed]}
    return None


def analyze_entry(bars, cfg):
    """Current VCP / pivot entry status for the latest bar."""
    n = len(bars)
    close, vol = bars.close, bars.volume
    latest = detect_base(bars, n, cfg)

    base, base_end = None, None
    for end in range(n, max(cfg.min_base_bars + 2, n - cfg.breakout_scan_bars) - 1, -1):
        found = latest if end == n else detect_base(bars, end, cfg)
        if found["valid"]:
            base, base_end = found, end
            break
    if base is None:
        return {"status": "no_base", "actionable": False, "reasons": latest.get("reasons", []),
                "base": latest}

    pivot = base["pivot"]
    breakout = next((i for i in range(base["pivot_index"] + 1, n) if close[i] > pivot), None)
    price = close[-1]
    out = {
        "base": base,
        "pivot": pivot,
        "buy_range": [rnd(pivot), rnd(pivot * (1 + cfg.max_chase))],
        "stop": base["stop"],
        "risk_pct": base["stop_pct"],
        "risk_ok": base["stop_pct"] <= cfg.max_stop,
        "price": rnd(price),
    }

    if breakout is None:
        if base_end != n:
            out.update(status="no_base", actionable=False,
                       reasons=[f"base valid through {bars.dates[base_end - 1]} "
                                f"but no longer: {'; '.join(latest.get('reasons', []))}"])
            return out
        below = 1 - price / pivot
        status = "near_pivot" if below <= cfg.near_pivot else "forming"
        out.update(status=status, actionable=status == "near_pivot", below_pivot=rnd(below),
                   reasons=[])
        failed = _recent_failed_breakout(bars, cfg, pivot)
        if failed:
            # A breakout that reversed deeply enough re-forms the base with its high as
            # the new pivot; keep the failure visible rather than calling it a clean setup.
            out["warnings"] = [
                f"failed breakout: closed above the earlier pivot {failed['pivot']} on "
                f"{failed['date']} and is back below it"
            ]
        return out

    avg = mean(vol[max(0, breakout - cfg.volume_avg_bars):breakout])
    vol_ratio = vol[breakout] / avg if avg else None
    extension = price / pivot - 1
    if price < base["stop"]:
        status = "stopped_out"
    elif price < pivot:
        status = "failed_breakout"
    elif extension > cfg.max_chase:
        status = "extended"
    elif vol_ratio is None or vol_ratio < cfg.breakout_volume_ratio:
        status = "breakout_unconfirmed"
    else:
        status = "breakout"
    out.update(
        status=status,
        actionable=status == "breakout",
        breakout_date=bars.dates[breakout],
        days_since_breakout=n - 1 - breakout,
        breakout_volume_ratio=rnd(vol_ratio, 2),
        extension=rnd(extension),
        reasons=[],
    )
    return out

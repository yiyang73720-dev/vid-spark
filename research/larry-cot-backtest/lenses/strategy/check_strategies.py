#!/usr/bin/env python3
"""
check_strategies -- independent re-implementation used to verify run_strategies.py.

It does NOT import larry_cot or strat_lib. Starting from the raw CSVs in S/data it recomputes
comm_net, idx26, the shutdown flags, the entry-day rule, the S1/S2/S3 long+short and long-only
trades, the daily equity of every market and of the equal-weight portfolio with plain Python
loops, and then compares them with the files written by run_strategies.py:
  * the trade lists (side, entry date, exit date) must be identical,
  * total return, CAGR, Sharpe, max drawdown and time in market must agree to 1e-9,
  * the crude and portfolio daily equity curves must agree to 1e-9.

Usage:  python3 -I check_strategies.py
"""
import bisect
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import pandas as pd  # noqa: E402

ROOT = Path(os.environ.get("LARRY_COT_ROOT")      # root holding data/ and work/
            or Path(__file__).resolve().parents[2])
HERE = ROOT / "work" / "lens_strategy"          # run_strategies.py outputs
MARKETS = ["crude", "gold", "silver", "soybean", "sugar"]
SHUTDOWN = [("2013-10-01", "2013-10-31"), ("2018-12-18", "2019-02-26"),
            ("2025-09-30", "2026-01-20")]
N, BUY, SELL, HOLD, WIN, SMA, COST = 26, 80, 20, 40, 20, 20, 0.0005
TOL = 1e-9


def load():
    cot = pd.read_csv(ROOT / "data" / "cot_weekly.csv", float_precision="round_trip")
    px = pd.read_csv(ROOT / "data" / "prices_daily.csv", float_precision="round_trip")
    data = {}
    for m in MARKETS:
        c = cot[cot.market == m].sort_values("report_date")
        comm = [int(round(a + b - c_ - d)) for a, b, c_, d in zip(
            c.prod_merc_long, c.swap_long, c.prod_merc_short, c.swap_short)]
        rdates = [pd.Timestamp(d) for d in c.report_date]
        p = px[(px.market == m)].sort_values("date")
        p = p[p.close.notna() & (p.close > 0)]
        data[m] = dict(comm=comm, rdates=rdates, dates=[pd.Timestamp(d) for d in p.date],
                       close=[float(v) for v in p.close])
    return data


def prepare(d):
    """idx26, shutdown flag, entry row per report; SMA20 per day."""
    comm, rd, dates = d["comm"], d["rdates"], d["dates"]
    idx = []
    for i in range(len(comm)):
        if i < N - 1:
            idx.append(float("nan"))
            continue
        w = comm[i - N + 1:i + 1]
        lo, hi = min(w), max(w)
        idx.append(float("nan") if hi == lo else 100.0 * (comm[i] - lo) / (hi - lo))
    shut = [any(pd.Timestamp(a) <= r <= pd.Timestamp(b) for a, b in SHUTDOWN) for r in rd]
    entry = []
    for r in rd:
        tgt = r - pd.Timedelta(days=r.weekday()) + pd.Timedelta(days=8)
        k = bisect.bisect_left(dates, tgt)
        entry.append(k if k < len(dates) else -1)
    close = d["close"]
    sma = [float("nan")] * len(close)
    for t in range(SMA - 1, len(close)):
        sma[t] = sum(close[t - SMA + 1:t + 1]) / SMA
    first = next(i for i in range(len(idx)) if not math.isnan(idx[i]) and entry[i] >= 0)
    d.update(idx=idx, shut=shut, entry=entry, sma=sma, start=entry[first])


def setup(v, side):
    if math.isnan(v):
        return False
    return v >= BUY if side == "long" else v <= SELL


def events(d, sides):
    """First tradable week of each setup episode (episodes over the full report sequence)."""
    ev = []
    for side in sides:
        prev, in_ep, done = False, False, False
        for i, v in enumerate(d["idx"]):
            s = setup(v, side)
            if s and not prev:                      # a new episode starts
                in_ep, done = True, False
            if not s:
                in_ep = False
            if in_ep and not done and not d["shut"][i] and d["entry"][i] >= 0:
                ev.append((d["entry"][i], side))
                done = True
            prev = s
    return sorted(ev)


def s1(d, ev):
    out, busy, n = [], -1, len(d["close"])
    for pos, side in ev:
        if pos <= busy:
            continue
        x = pos + HOLD
        out.append((side, pos, min(x, n - 1), x > n - 1))
        busy = min(x, n - 1)
    return out


def s2(d, ev):
    out, busy, n, c, sma = [], -1, len(d["close"]), d["close"], d["sma"]
    for i, (pos, side) in enumerate(ev):
        if pos <= busy:
            continue
        nxt = ev[i + 1][0] if i + 1 < len(ev) else n
        for t in range(pos, min(pos + WIN - 1, n - 1, nxt - 1) + 1):
            ok = (c[t] > sma[t]) if side == "long" else (c[t] < sma[t])
            if ok:                                  # NaN SMA -> comparisons are False
                x = t + HOLD
                out.append((side, t, min(x, n - 1), x > n - 1))
                busy = min(x, n - 1)
                break
    return out


def pos_from_trades(d, trades):
    p = [0.0] * len(d["close"])
    for side, e, x, op in trades:
        for t in range(e, len(p) if op else x):
            p[t] = 1.0 if side == "long" else -1.0
    return p


def s3_pos(d, long_only):
    p, cur, n = [0.0] * len(d["close"]), 0.0, len(d["close"])
    upd = {}
    for i, v in enumerate(d["idx"]):
        if d["shut"][i] or d["entry"][i] < 0:
            continue
        tgt = 1.0 if setup(v, "long") else (-1.0 if setup(v, "short") else 0.0)
        upd[d["entry"][i]] = max(tgt, 0.0) if long_only else tgt
    for t in range(n):
        if t in upd:
            cur = upd[t]
        p[t] = cur if t >= d["start"] else 0.0
    return p


def runs_to_trades(d, p):
    out, t, n = [], d["start"], len(p)
    while t < n:
        if p[t] == 0:
            t += 1
            continue
        j = t
        while j < n and p[j] == p[t]:
            j += 1
        out.append(("long" if p[t] > 0 else "short", t, min(j, n - 1), j >= n))
        t = j
    return out


def daily(d, p, cost=COST):
    """{date: net return} over [start, end]."""
    c, s, out, prev = d["close"], d["start"], {}, 0.0
    for t in range(s, len(c)):
        r = c[t] / c[t - 1] - 1.0 if t > s else 0.0
        out[d["dates"][t]] = (1.0 + prev * r) * (1.0 - cost * abs(p[t] - prev)) - 1.0
        prev = p[t]
    return out


def metrics(rets, held=None):
    dates = sorted(rets)
    eq, e, peak, mdd = [], 1.0, 1.0, 0.0
    for dt in dates:
        e *= 1.0 + rets[dt]
        eq.append(e)
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1.0)
    yrs = (dates[-1] - dates[0]).days / 365.25
    r = [rets[dt] for dt in dates[1:]]
    mu = sum(r) / len(r)
    sd = math.sqrt(sum((x - mu) ** 2 for x in r) / (len(r) - 1))
    ppy = len(r) / yrs
    out = dict(total_return=eq[-1] - 1.0, ann_return=eq[-1] ** (1 / yrs) - 1.0,
               ann_vol=sd * math.sqrt(ppy), sharpe=mu * ppy / (sd * math.sqrt(ppy)),
               max_dd=mdd)
    if held is not None:
        out["time_in_mkt"] = held
    return out, dict(zip(dates, eq))


def main():
    data = load()
    for d in data.values():
        prepare(d)
    met = pd.read_csv(HERE / "metrics.csv")
    trd = pd.read_csv(HERE / "trades_all.csv")
    eqd = pd.read_csv(HERE / "equity_daily.csv", parse_dates=["date"]).set_index("date")
    nets, n_checked, worst = {}, 0, 0.0
    strategies = {}
    for m, d in data.items():
        ev_ls, ev_l = events(d, ("long", "short")), events(d, ("long",))
        p3, p3l = s3_pos(d, False), s3_pos(d, True)
        strategies[m] = {
            "S1_long_short": s1(d, ev_ls), "S1_long_only": s1(d, ev_l),
            "S2_long_short": s2(d, ev_ls), "S2_long_only": s2(d, ev_l),
            "S3_long_short": runs_to_trades(d, p3), "S3_long_only": runs_to_trades(d, p3l)}
        for s, tr in strategies[m].items():
            p = p3 if s == "S3_long_short" else p3l if s == "S3_long_only" else \
                pos_from_trades(d, tr)
            nets[(m, s)] = daily(d, p)
            # 1) identical trade lists
            got = trd[(trd.market == m) & (trd.strategy == s)]
            mine = [(sd, d["dates"][e].date().isoformat(), d["dates"][x].date().isoformat(), op)
                    for sd, e, x, op in tr]
            theirs = list(zip(got.side, got.entry_date, got.exit_date, got.open_at_end))
            assert mine == theirs, (m, s)
            # 2) metrics
            held = sum(1 for t in range(d["start"], len(p) - 1) if p[t] != 0) / \
                (len(p) - 1 - d["start"])
            mm, eq = metrics(nets[(m, s)], held)
            row = met[(met.market == m) & (met.strategy == s)].iloc[0]
            for k, v in mm.items():
                worst = max(worst, abs(v - row[k]))
                assert abs(v - row[k]) < TOL, (m, s, k, v, row[k])
                n_checked += 1
            if m == "crude":
                col = eqd[f"crude__{s}"].dropna()
                for dt, v in eq.items():
                    assert abs(col.loc[dt] - v) < TOL, (s, dt)
        bh = [1.0 if t >= d["start"] else 0.0 for t in range(len(d["close"]))]
        nets[(m, "buy_hold")] = daily(d, bh, cost=0.0)
        mm, _ = metrics(nets[(m, "buy_hold")])
        row = met[(met.market == m) & (met.strategy == "buy_hold")].iloc[0]
        for k, v in mm.items():
            assert abs(v - row[k]) < TOL, (m, "buy_hold", k)
            n_checked += 1
    # 3) portfolio: union calendar from the common start, 0 return on a market's non-trading days
    start = max(data[m]["dates"][data[m]["start"]] for m in MARKETS)
    end = min(data[m]["dates"][-1] for m in MARKETS)
    uni = sorted({dt for m in MARKETS for dt in data[m]["dates"] if start <= dt <= end})
    for s in list(strategies["crude"]) + ["buy_hold"]:
        port = {dt: sum(nets[(m, s)].get(dt, 0.0) for m in MARKETS) / len(MARKETS)
                for dt in uni}
        mm, eq = metrics(port)
        row = met[(met.market == "portfolio") & (met.strategy == s)].iloc[0]
        for k, v in mm.items():
            worst = max(worst, abs(v - row[k]))
            assert abs(v - row[k]) < TOL, ("portfolio", s, k, v, row[k])
            n_checked += 1
        col = eqd[f"portfolio__{s}"]
        for dt, v in eq.items():
            assert abs(col.loc[dt] - v) < TOL, ("portfolio", s, dt)
    print(f"OK: trade lists identical for {len(MARKETS)} markets x 6 strategies; "
          f"{n_checked} metric values agree (max abs diff {worst:.2e}); crude and portfolio "
          f"daily equity curves agree to {TOL}.")


if __name__ == "__main__":
    main()

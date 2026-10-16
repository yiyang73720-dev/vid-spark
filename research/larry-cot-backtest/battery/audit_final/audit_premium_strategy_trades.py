#!/usr/bin/env python3
"""
audit_premium_strategy_trades -- POST-HOC diagnostics by the final auditor (not pre-registered,
never promoted to a result). Question: the premium signal's LONG+SHORT strategies (S1/S2,
SPEC2 rule 6) beat buy & hold and random timing, even though the pre-registered LONG weekly
excess is about zero after 2004. Is that robust, or is it a few crash trades?

Uses the auditor's independent roll index and strategy code (audit_premium_independent.py):
  * trade list of S1/S2 long+short (and long-only) on the full window and from 2010,
  * share of total log-growth contributed by the largest trades,
  * Sharpe and random-timing null p after removing every trade that is open during the 2008,
    2014-15 or 2020 crude crashes (applied to the real AND to every null path, so the null
    stays comparable).

Run:  python3 -I audit_premium_strategy_trades.py
      (writes ROOT/work2/audit_final/audit_premium_strategy_trades.json)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json                       # noqa: E402
from pathlib import Path          # noqa: E402

import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_premium_independent as ap   # noqa: E402  (the auditor's own index and S1/S2)

CRASHES = {"2008": ("2008-07-01", "2008-12-31"), "2014-15": ("2014-07-01", "2016-02-29"),
           "2020": ("2020-01-01", "2020-05-31")}


def trades_from_pos(pos: np.ndarray, ret: np.ndarray, dates: pd.DatetimeIndex) -> list[dict]:
    """Maximal runs of the same non-zero position = trades; net return compounds the daily
    P&L including 0.05% on entry and exit."""
    out, t, n = [], 0, len(pos)
    while t < n:
        if pos[t] == 0:
            t += 1
            continue
        s, j = pos[t], t
        while j < n and pos[j] == s:
            j += 1
        x = min(j, n - 1)
        growth = np.prod(1.0 + s * ret[t + 1:x + 1])
        out.append(dict(side="long" if s > 0 else "short", entry=dates[t].date().isoformat(),
                        exit=dates[x].date().isoformat(), entry_row=t, exit_row=x,
                        net=float((1 - ap.COST) * growth * (1 - ap.COST) - 1.0)))
        t = j
    return out


def mask_trades(pos: np.ndarray, dates: pd.DatetimeIndex, windows) -> np.ndarray:
    """Zero every trade (run of the same position) that overlaps any of the date windows."""
    p = pos.copy()
    n, t = len(p), 0
    while t < n:
        if p[t] == 0:
            t += 1
            continue
        s, j = p[t], t
        while j < n and p[j] == s:
            j += 1
        d0, d1 = dates[t], dates[min(j, n - 1)]
        if any(d0 <= pd.Timestamp(b) and d1 >= pd.Timestamp(a) for a, b in windows):
            p[t:j] = 0.0
        t = j
    return p


def analyse(sig_w, entry_row, px, label: str) -> dict:
    """Trades, concentration and crash-excluded Sharpe (with a matching null) per strategy."""
    close = px.to_numpy()
    ret = np.concatenate(([0.0], close[1:] / close[:-1] - 1.0))
    sma = pd.Series(close).rolling(ap.SMA_N, min_periods=ap.SMA_N).mean().to_numpy()
    dates = px.index
    valid = ~np.isnan(sig_w)
    start = int(entry_row[np.flatnonzero(valid & (entry_row >= 0))[0]])
    windows = list(CRASHES.values())

    def run(sig):
        res = {}
        for mode, sides in (("long_only", ("long",)), ("long_short", ("long", "short"))):
            ev = ap.episode_events(sig, entry_row, sides)
            for name, s2 in (("S1", False), ("S2", True)):
                res[f"{name}_{mode}"] = ap.strategy_position(ev, close, sma, s2)
        return res

    real = run(sig_w)
    vidx = np.flatnonzero(valid)
    offs = np.random.default_rng(ap.SEED).integers(ap.MIN_SHIFT, len(vidx) - ap.MIN_SHIFT,
                                                   size=ap.K_MC, endpoint=True)
    null_full = {k: [] for k in real}
    null_ex = {k: [] for k in real}
    for o in offs:
        sh = sig_w.copy()
        sh[vidx] = np.roll(sig_w[vidx], int(o))
        for k, pos in run(sh).items():
            null_full[k].append(ap.sharpe_of(pos, ret, start, dates)["sharpe"])
            null_ex[k].append(ap.sharpe_of(mask_trades(pos, dates, windows), ret, start,
                                           dates)["sharpe"])
    bh = np.zeros(len(close))
    bh[start:] = 1.0
    out = dict(window=f"{dates[start].date()}..{dates[-1].date()}",
               buy_hold=ap.sharpe_of(bh, ret, start, dates, cost=0.0))
    for k, pos in real.items():
        tr = trades_from_pos(pos, ret, dates)
        logg = np.log1p([t["net"] for t in tr])
        top = sorted(tr, key=lambda t: -t["net"])[:5]
        full = ap.sharpe_of(pos, ret, start, dates)
        ex = ap.sharpe_of(mask_trades(pos, dates, windows), ret, start, dates)
        nf, ne = np.asarray(null_full[k]), np.asarray(null_ex[k])
        out[k] = dict(
            n_trades=len(tr), n_short=sum(t["side"] == "short" for t in tr),
            sharpe=full["sharpe"], cagr=full["cagr"],
            null_p=float((1 + np.sum(nf >= full["sharpe"])) / (1 + len(nf))),
            total_log_growth=float(logg.sum()),
            top5_trades=[{kk: t[kk] for kk in ("side", "entry", "exit", "net")} for t in top],
            top3_share_of_log_growth=float(np.sort(logg)[::-1][:3].sum() / logg.sum())
            if logg.sum() > 0 else None,
            ex_crash_trades=dict(sharpe=ex["sharpe"], cagr=ex["cagr"],
                                 null_p=float((1 + np.sum(ne >= ex["sharpe"])) / (1 + len(ne))),
                                 windows=CRASHES))
    out["label"] = label
    return out


def main() -> None:
    raw = ap.load_contracts()
    clean, _ = ap.drop_placeholder_rows(raw)
    v = clean[clean["c1"].notna() & clean["c2"].notna()]
    roll_days, _ = ap.rule_roll_days(clean, v)
    px = ap.build_roll_index(v, roll_days)
    fridays = pd.date_range(ap.FIRST_FRIDAY, ap.LAST_FRIDAY, freq="W-FRI")
    sig_w = ap.friday_signal(v, fridays)
    entry_row, _ = ap.forward_return(px, fridays)
    res = dict(
        note="POST-HOC auditor diagnostics on the auditor's independent crude_roll index; "
             "not pre-registered and not a result",
        full=analyse(sig_w, entry_row, px, "1985-2024"),
        from2010=analyse(np.where(np.asarray(fridays >= "2010-01-01"), sig_w, np.nan),
                         entry_row, px, "signal restricted to 2010+"))
    ap.OUT.parent.mkdir(parents=True, exist_ok=True)     # ROOT/work2/audit_final
    (ap.OUT.parent / "audit_premium_strategy_trades.json").write_text(json.dumps(res, indent=1))
    for part in ("full", "from2010"):
        r = res[part]
        print(part, r["window"], "B&H", round(r["buy_hold"]["sharpe"], 3))
        for k in ("S1_long_only", "S2_long_only", "S1_long_short", "S2_long_short"):
            x = r[k]
            print(f"  {k}: trades {x['n_trades']} (short {x['n_short']}) sharpe "
                  f"{x['sharpe']:.3f} p {x['null_p']:.4f} | top3 share "
                  f"{x['top3_share_of_log_growth']} | ex-crash sharpe "
                  f"{x['ex_crash_trades']['sharpe']:.3f} p {x['ex_crash_trades']['null_p']:.4f}")
            print("     top:", [(t['side'], t['entry'], round(t['net'], 3))
                                for t in x["top5_trades"]])


if __name__ == "__main__":
    main()

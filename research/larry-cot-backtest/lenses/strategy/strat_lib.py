#!/usr/bin/env python3
"""
strat_lib -- trading-strategy simulation primitives for the Larry Williams COT lens.

Everything that is COT-specific (index, entry timing, shutdown flags, episodes) comes from the
reconciled engine work/engine/larry_cot.py, which is imported, not re-implemented. This module
only turns the engine's per-report signals into daily positions and daily P&L.

Conventions (all choices are marked "LENS-CHOICE")
--------------------------------------------------
* Daily grid = the market's cleaned close series (lc.load_prices: closes <= 0 dropped).
* pos[t] = exposure held from close t to close t+1, set AT close t (as a fraction of equity,
  +1 = 100% long, -1 = 100% short). The exposure is rebalanced daily (constant-notional),
  which is what "daily close-to-close P&L" means.
* net return of day t:  (1 + pos[t-1] * r[t]) * (1 - COST * |pos[t] - pos[t-1]|) - 1,
  with r[t] = close[t] / close[t-1] - 1 and COST = 0.0005 per side per unit of change.
  A flip from +1 to -1 therefore costs 2 * COST. rf = 0.
* Backtest window of a market: from the entry_date of its first report with a defined idx26
  (start, equity = 1 just before that close) to the last close in the data. Buy & hold over the
  same window is pos = +1 from the start close, with no cost (pure price path).
* A trade's net return = (1-COST) * prod(1 + side * r[t], t = entry+1..exit) * (1-COST) - 1,
  the exit cost applying only to closed trades. For S1/S2 (trades never touch each other) this is
  exactly the trade's contribution to the equity curve. The plain price return
  side * (close[exit]/close[entry] - 1) is also reported. For long trades it equals the engine's
  forward return r8, because the hold is 40 trading days = 5*8.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_DIR = Path(__file__).resolve().parents[2] / "engine"     # engine code (package/engine)
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
import larry_cot as lc  # noqa: E402  (reconciled engine)

COST = 0.0005          # 0.05% per side per unit of position change
HOLD = 40              # trading days held by S1 and S2 (= the 8-week horizon of the headline)
TRIG_WIN = 20          # S2: price trigger must occur within this many trading days
SMA_N = 20             # S2: 20-day simple moving average of closes (including today's close)
IDX_COL, BUY, SELL = "idx26", 80, 20   # S1/S2/S3 use N=26 with the 80/20 thresholds


# ----------------------------------------------------------------------------------------
# Per-market context
# ----------------------------------------------------------------------------------------
@dataclass
class MarketCtx:
    """Everything one market's simulations need, aligned on its daily close grid."""
    market: str
    dates: pd.DatetimeIndex          # trading days (cleaned close series)
    close: np.ndarray                # closes
    ret: np.ndarray                  # r[t] = close[t]/close[t-1]-1, r[0] = 0
    sma: np.ndarray                  # 20-day SMA (NaN for t < 19)
    start: int                       # first day of the backtest window (row in dates)
    report_date: np.ndarray          # per report (datetime64)
    entry_pos: np.ndarray            # per report: row of entry_date, -1 if none
    shutdown: np.ndarray             # per report: report in a shutdown window
    idx: np.ndarray                  # per report: idx26 (NaN when undefined)
    idx_defined: np.ndarray = field(default=None)   # per report: idx26 is not NaN

    @property
    def n(self) -> int:
        return len(self.close)


def make_ctx(market: str, frame: pd.DataFrame, prices: pd.Series,
             idx_col: str = IDX_COL) -> MarketCtx:
    """Build the context from the engine's per-report frame and the cleaned close series that
    the frame was built against (entry_pos indexes into it)."""
    close = prices.to_numpy(dtype=float)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    sma = pd.Series(close).rolling(SMA_N, min_periods=SMA_N).mean().to_numpy()
    idx = frame[idx_col].to_numpy(dtype=float)
    ep = frame["entry_pos"].to_numpy(dtype=np.int64)
    defined = ~np.isnan(idx)
    first = int(np.flatnonzero(defined & (ep >= 0))[0])
    return MarketCtx(market=market, dates=prices.index, close=close, ret=ret, sma=sma,
                     start=int(ep[first]),
                     report_date=frame["report_date"].to_numpy(),
                     entry_pos=ep, shutdown=frame["shutdown"].to_numpy(dtype=bool),
                     idx=idx, idx_defined=defined)


def tradable_mask(ctx: MarketCtx) -> np.ndarray:
    """Report weeks a trader could act on in real time: outside the shutdown windows, with an
    entry day inside the price data (the engine's timing rule)."""
    return (~ctx.shutdown) & (ctx.entry_pos >= 0)


# ----------------------------------------------------------------------------------------
# Setup events (the weekly COT information turned into dated events)
# ----------------------------------------------------------------------------------------
def setup_events(ctx: MarketCtx, idx: np.ndarray, sides=("long", "short"),
                 first_of_episode: bool = True) -> list[dict]:
    """Return the dated setup events, sorted by entry row.

    first_of_episode=True (S1 and S2): one event per setup episode, at the FIRST TRADABLE
    week of the episode. Episodes are maximal runs on the full report sequence
    (lc.episode_ids, the engine's "full" mode). LENS-CHOICE: when the episode starts inside a
    shutdown window, the first week after the window that still belongs to the episode is used,
    because that is the first time it could be acted on. An episode lying entirely inside a
    shutdown window yields no event.
    first_of_episode=False (S2 variant): every tradable setup week is an event."""
    trad = tradable_mask(ctx)
    ev = []
    for side in sides:
        setup = lc.setup_mask(idx, side, BUY, SELL)
        if first_of_episode:
            ids = lc.episode_ids(setup)
            for e in np.unique(ids[ids > 0]):
                rows = np.flatnonzero((ids == e) & trad)
                if len(rows) == 0:
                    continue
                j = int(rows[0])
                first_j = int(np.flatnonzero(ids == e)[0])
                ev.append(dict(side=side, pos=int(ctx.entry_pos[j]), report=j,
                               episode_first_report=first_j,
                               delayed_by_shutdown=bool(j != first_j)))
        else:
            for j in np.flatnonzero(setup & trad):
                ev.append(dict(side=side, pos=int(ctx.entry_pos[j]), report=int(j),
                               episode_first_report=int(j), delayed_by_shutdown=False))
    ev.sort(key=lambda d: (d["pos"], d["report"]))
    return ev


# ----------------------------------------------------------------------------------------
# Strategies -> list of trades (S1, S2) or a position path (S3)
# ----------------------------------------------------------------------------------------
def s1_trades(ctx: MarketCtx, events: list[dict], hold: int = HOLD) -> list[dict]:
    """S1 "pure COT": enter at the event's entry close, hold `hold` trading days, one position
    at a time. LENS-CHOICE: an event whose entry day is on or before the current trade's exit
    day counts as "while in a trade" and is ignored (no same-close exit-and-re-entry)."""
    trades, busy_until = [], -1
    for e in events:
        if e["pos"] <= busy_until:
            continue
        entry = e["pos"]
        exit_ = entry + hold
        is_open = exit_ > ctx.n - 1                       # still open at the end of the data
        exit_ = min(exit_, ctx.n - 1)
        trades.append(dict(side=e["side"], entry=entry, exit=exit_, open=is_open,
                           signal_report=e["report"], trigger=entry,
                           delayed_by_shutdown=e["delayed_by_shutdown"]))
        busy_until = exit_
    return trades


def s2_trades(ctx: MarketCtx, events: list[dict], hold: int = HOLD, win: int = TRIG_WIN,
              lag: int = 0, exec_delay: int = 0) -> list[dict]:
    """S2 "COT + price trigger": after a setup's entry day, wait up to `win` trading days for a
    close above (long) / below (short) its 20-day SMA, enter, hold `hold` days.

    LENS-CHOICES:
      * the window is rows [pos+lag, pos+lag+win-1]; lag=0 (main) lets the entry day itself
        trigger, because its close is the first one the trader may act on; lag=1 is a variant.
      * entry at the triggering close (exec_delay=0, the spec's wording) or at the next close
        (exec_delay=1, a no-same-bar-execution variant).
      * one position at a time: an event arriving while a trade is open is ignored; a new event
        arriving while an earlier setup is still WAITING replaces it (the latest COT read wins),
        so the old setup can only trigger strictly before the new event's entry day.
    """
    up = ctx.close > ctx.sma           # NaN SMA compares False
    dn = ctx.close < ctx.sma
    trades, busy_until = [], -1
    for i, e in enumerate(events):
        if e["pos"] <= busy_until:
            continue
        lo = e["pos"] + lag
        hi = min(e["pos"] + lag + win - 1, ctx.n - 1)
        nxt = events[i + 1]["pos"] if i + 1 < len(events) else ctx.n
        hi = min(hi, nxt - 1)                             # superseded by the next setup
        if hi < lo:
            continue
        cond = up if e["side"] == "long" else dn
        hit = np.flatnonzero(cond[lo:hi + 1])
        if len(hit) == 0:
            continue
        trig = lo + int(hit[0])
        entry = trig + exec_delay
        if entry > ctx.n - 1 or entry <= busy_until:
            continue
        exit_ = entry + hold
        is_open = exit_ > ctx.n - 1
        exit_ = min(exit_, ctx.n - 1)
        trades.append(dict(side=e["side"], entry=entry, exit=exit_, open=is_open,
                           signal_report=e["report"], trigger=trig,
                           delayed_by_shutdown=e["delayed_by_shutdown"]))
        busy_until = exit_
    return trades


def trades_to_pos(ctx: MarketCtx, trades: list[dict]) -> np.ndarray:
    """Position path of non-overlapping trades: side from the entry close up to (not including)
    the exit close; an open trade is held to the last close."""
    pos = np.zeros(ctx.n)
    for t in trades:
        sgn = 1.0 if t["side"] == "long" else -1.0
        end = ctx.n if t["open"] else t["exit"]
        pos[t["entry"]:end] = sgn
    return pos


def s3_pos(ctx: MarketCtx, idx: np.ndarray, long_only: bool = False) -> np.ndarray:
    """S3 "regime": at each tradable report's entry close the position becomes +1 if its idx26
    >= 80, -1 if <= 20, else 0 (NaN idx -> 0); it is held until the next update.
    LENS-CHOICE: shutdown-window reports are not tradable, so they do NOT update the position
    (the latest tradable report stays in force until the first report after the window)."""
    lo, sh = lc.signal_masks(idx, BUY, SELL)
    tgt = np.where(lo, 1.0, np.where(sh, -1.0, 0.0))
    if long_only:
        tgt = np.maximum(tgt, 0.0)
    trad = tradable_mask(ctx)
    upd = np.full(ctx.n, np.nan)
    upd[ctx.entry_pos[trad]] = tgt[trad]     # entry rows are strictly increasing (checked)
    pos = pd.Series(upd).ffill().fillna(0.0).to_numpy().copy()   # writable copy (pandas CoW)
    pos[:ctx.start] = 0.0
    return pos


def pos_to_trades(ctx: MarketCtx, pos: np.ndarray) -> list[dict]:
    """Split a position path into trades: maximal runs of the same non-zero sign (a flip ends one
    trade and starts the next on the same close). Used for S3's trade statistics."""
    trades, t, n = [], ctx.start, ctx.n
    while t < n:
        if pos[t] == 0:
            t += 1
            continue
        s, j = pos[t], t
        while j < n and pos[j] == s:
            j += 1
        # held over closes t..j-1, closed at close j (if j < n)
        trades.append(dict(side="long" if s > 0 else "short", entry=t,
                           exit=min(j, n - 1), open=j >= n, signal_report=-1, trigger=t,
                           delayed_by_shutdown=False))
        t = j
    return trades


# ----------------------------------------------------------------------------------------
# P&L, trade returns, metrics
# ----------------------------------------------------------------------------------------
def daily_net(ctx: MarketCtx, pos: np.ndarray, cost: float = COST) -> pd.Series:
    """Net daily strategy returns over the backtest window [start, n-1]. The value on the start
    day is only the entry cost (no position was held into it)."""
    p_prev = np.concatenate(([0.0], pos[:-1]))
    p_prev[:ctx.start + 1] = 0.0          # nothing held before the window starts
    dpos = np.abs(pos - p_prev)
    dpos[:ctx.start] = 0.0
    net = (1.0 + p_prev * ctx.ret) * (1.0 - cost * dpos) - 1.0
    return pd.Series(net[ctx.start:], index=ctx.dates[ctx.start:])


def held_mask(ctx: MarketCtx, pos: np.ndarray) -> np.ndarray:
    """For each day of the window after the start: was a position held INTO that day."""
    return pos[ctx.start:-1] != 0


def trade_returns(ctx: MarketCtx, trades: list[dict], cost: float = COST) -> list[dict]:
    """Add net (compounded, after costs) and plain price returns to each trade."""
    out = []
    for t in trades:
        sgn = 1.0 if t["side"] == "long" else -1.0
        e, x = t["entry"], t["exit"]
        growth = np.prod(1.0 + sgn * ctx.ret[e + 1:x + 1])
        net = (1.0 - cost) * growth * (1.0 if t["open"] else (1.0 - cost)) - 1.0
        price_ret = sgn * (ctx.close[x] / ctx.close[e] - 1.0)
        out.append({**t, "entry_date": ctx.dates[e], "exit_date": ctx.dates[x],
                    "entry_price": float(ctx.close[e]), "exit_price": float(ctx.close[x]),
                    "days_held": int(x - e), "price_ret": float(price_ret),
                    "net_ret": float(net)})
    return out


def perf_metrics(net: pd.Series, held: np.ndarray | None = None) -> dict:
    """Performance of a daily net-return series that starts at the window's first close.

    ann_return = CAGR of the equity curve over calendar time (365.25-day years).
    ann_vol    = sample std of daily returns after the start day * sqrt(obs per year), where
                 obs per year = (number of daily returns) / (calendar years).
    sharpe     = (mean daily return * obs per year) / ann_vol   (rf = 0).
    max_dd     = worst peak-to-trough decline of the equity curve (equity = 1 before day 0).
    time_in_mkt = share of days (after the start day) entered holding a non-zero position."""
    eq = np.cumprod(1.0 + net.to_numpy())
    years = (net.index[-1] - net.index[0]).days / 365.25
    r = net.to_numpy()[1:]
    ppy = len(r) / years
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy))
    peak = np.maximum.accumulate(np.concatenate(([1.0], eq)))
    dd = np.concatenate(([1.0], eq)) / peak - 1.0
    out = dict(start=net.index[0].date().isoformat(), end=net.index[-1].date().isoformat(),
               years=years, total_return=float(eq[-1] - 1.0),
               ann_return=float(eq[-1] ** (1.0 / years) - 1.0), ann_vol=vol,
               sharpe=float(np.mean(r) * ppy / vol) if vol > 0 else float("nan"),
               max_dd=float(dd.min()))
    if held is not None:
        out["time_in_mkt"] = float(np.mean(held))
    return out


def trade_stats(tr: list[dict]) -> dict:
    """Closed-trade statistics (open trades are only counted in n_open).
    win_rate / avg_trade / median_trade use the NET trade return (daily-rebalanced exposure,
    after costs: what the equity curve earns). avg_price_ret / win_rate_price use the plain
    signed price return side*(exit/entry-1) before costs, i.e. a fixed number of contracts
    sized at 1x equity at entry (the canonical forward-return definition)."""
    closed = [t for t in tr if not t["open"]]
    r = np.array([t["net_ret"] for t in closed])
    p = np.array([t["price_ret"] for t in closed])
    nan = float("nan")
    return dict(n_trades=len(closed), n_open=len(tr) - len(closed),
                n_long=sum(t["side"] == "long" for t in closed),
                n_short=sum(t["side"] == "short" for t in closed),
                win_rate=float(np.mean(r > 0)) if len(r) else nan,
                avg_trade=float(np.mean(r)) if len(r) else nan,
                median_trade=float(np.median(r)) if len(r) else nan,
                avg_price_ret=float(np.mean(p)) if len(p) else nan,
                win_rate_price=float(np.mean(p > 0)) if len(p) else nan)


def sharpe_only(net: np.ndarray, years: float) -> float:
    """Fast Sharpe for the null distribution; identical formula to perf_metrics."""
    r = net[1:]
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(len(r) / years)) if sd > 0 else float("nan")


# ----------------------------------------------------------------------------------------
# One full run of every strategy for a market (used by the driver and by the null test)
# ----------------------------------------------------------------------------------------
STRATEGIES = ("S1_long_short", "S1_long_only", "S2_long_short", "S2_long_only",
              "S3_long_short", "S3_long_only")


def run_market(ctx: MarketCtx, idx: np.ndarray | None = None,
               s2_kwargs: dict | None = None) -> dict:
    """Return {strategy: (pos array, trades list)} for the six main strategies.
    idx defaults to the market's real idx26; the null test passes a circularly shifted one."""
    idx = ctx.idx if idx is None else idx
    s2_kwargs = s2_kwargs or {}
    ev_ls = setup_events(ctx, idx, ("long", "short"))
    ev_l = [e for e in ev_ls if e["side"] == "long"]
    out = {}
    for name, ev in (("long_short", ev_ls), ("long_only", ev_l)):
        t1 = s1_trades(ctx, ev)
        out[f"S1_{name}"] = (trades_to_pos(ctx, t1), t1)
        t2 = s2_trades(ctx, ev, **s2_kwargs)
        out[f"S2_{name}"] = (trades_to_pos(ctx, t2), t2)
    for name, lo in (("long_short", False), ("long_only", True)):
        p3 = s3_pos(ctx, idx, long_only=lo)
        out[f"S3_{name}"] = (p3, pos_to_trades(ctx, p3))
    return out


def shift_idx(ctx: MarketCtx, k: int) -> np.ndarray:
    """Null-model index: idx26 circularly shifted by k positions over the reports where it is
    defined (np.roll(+k), as in the engine's permutation test). Timing, shutdown flags and prices
    stay real, so the run structure of the setups and the exposure profile are preserved but their
    alignment with prices is random."""
    out = ctx.idx.copy()
    d = np.flatnonzero(ctx.idx_defined)
    out[d] = np.roll(ctx.idx[d], k)
    return out

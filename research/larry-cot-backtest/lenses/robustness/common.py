"""
common.py -- shared helpers for the robustness & data-snooping lens.

Everything statistical comes from the reconciled engine (work/engine/larry_cot.py); this
module only adds:
  * path constants and the engine import,
  * builders for the robustness *variants* of the per-market frames
      - comm_net/oi index           (lens item f)
      - Monday-after-release entry  (lens item g)
  * a grid summariser (how many cells pass p<0.05, Benjamini-Hochberg survivors, ...),
  * a NaN-safe JSON writer.

All scripts in this directory are meant to be run as
    python3 -I <script>.py
(-I = isolated mode, so the script directory is NOT on sys.path; we add it explicitly).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------
# Paths and engine import
# ------------------------------------------------------------------------------------------
# ROOT holds data/, ext/, work/: $LARRY_COT_ROOT, else the package dir (lenses/robustness/../..).
S = Path(os.environ.get("LARRY_COT_ROOT") or Path(__file__).resolve().parents[2])
ENGINE_DIR = S / "work" / "engine"                  # engine outputs (canonical.csv, pooled.csv)
LENS_DIR = S / "work" / "lens_robustness"           # this lens' outputs
LENS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))   # engine code
import larry_cot as lc  # noqa: E402  (reconciled engine; never re-implemented here)

SEED = lc.DEFAULT_SEED          # 11 -- same seed as the canonical run
K = lc.K_PERM                   # 2000 permutation draws
PRIMARY = dict(lookback=26, buy_th=80, sell_th=20, horizon_w=8)   # pre-registered cell
KEY_COLS = ["market", "lookback", "buy_th", "sell_th", "side", "horizon_w"]


# ------------------------------------------------------------------------------------------
# Multiple-testing helpers
# ------------------------------------------------------------------------------------------
def bh_reject(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure. Returns a boolean mask of rejected hypotheses.
    NaN p-values are treated as 1 (never rejected) but still count in m (conservative)."""
    p = np.asarray(p, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.flatnonzero(passed))         # largest rank i with p_(i) <= i q / m
        out[order[: kmax + 1]] = True
    return out


def bh_adjusted(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values): min over j>=i of p_(j) * m / j, capped at 1."""
    p = np.asarray(p, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def grid_summary(tab: pd.DataFrame, p_col: str = "perm_p", alpha: float = 0.05,
                 q: float = 0.10) -> dict:
    """Headline counts for a grid of cells: how many have p < alpha with positive excess,
    how many would be expected under the null (alpha * m), and BH(q) survivors."""
    p = tab[p_col].to_numpy(dtype=float)
    ex = tab["excess"].to_numpy(dtype=float)
    hit = (p < alpha) & (ex > 0)
    rej = bh_reject(p, q)
    return dict(n_cells=int(len(tab)),
                n_p_lt_alpha=int(np.sum(p < alpha)),
                n_p_lt_alpha_and_pos_excess=int(hit.sum()),
                expected_under_null=float(alpha * len(tab)),
                n_pos_excess=int(np.sum(ex > 0)),
                bh_q=q, n_bh_survivors=int(rej.sum()),
                min_p=float(np.nanmin(p)))


# ------------------------------------------------------------------------------------------
# Variant frames
# ------------------------------------------------------------------------------------------
def add_oi_normalised_index(panel: dict, lookbacks=lc.LOOKBACKS, prefix: str = "oiidx") -> dict:
    """Lens (f): add columns {prefix}{N} = Larry index computed on comm_net / oi instead of
    comm_net. Returns a new dict of copied frames (the input panel is not modified)."""
    out = {}
    for m, f in panel.items():
        g = f.copy()
        ratio = g["comm_net"].to_numpy(dtype=float) / g["oi"].to_numpy(dtype=float)
        for n in lookbacks:
            g[f"{prefix}{n}"] = lc.larry_index(ratio, n)
        out[m] = g
    return out


def monday_entry_panel(panel: dict, prices: dict, horizons_w=lc.HORIZONS_W) -> dict:
    """Lens (g): re-time every entry to the first trading day on or after the MONDAY after the
    release Friday, i.e. report_date - weekday + 7 days (Mon=0), instead of the spec's Tuesday
    (report_date - weekday + 8). Entry price / forward returns are recomputed with the engine's
    forward_returns on the same cleaned price series.

    Caveat (sensitivity only): in holiday weeks the CFTC release slips to Monday 15:30 ET,
    after the futures close, so a Monday close is NOT always tradable on the information."""
    out = {}
    for m, f in panel.items():
        g = f.copy()
        px = prices[m]
        d = pd.DatetimeIndex(g["report_date"])
        tgt = d - pd.to_timedelta(d.weekday, unit="D") + pd.Timedelta(days=7)
        pos = np.searchsorted(px.index.values, tgt.values, side="left")
        pos = np.where(pos < len(px), pos, -1).astype(np.int64)
        g["target_date"] = tgt
        g["entry_pos"] = pos
        has = pos >= 0
        ed = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        ed[has] = px.index.values[pos[has]]
        g["entry_date"] = pd.to_datetime(ed)
        g["entry_price"] = np.where(has, px.to_numpy()[np.where(has, pos, 0)], np.nan)
        lc.add_forward_returns(g, px, horizons_w=horizons_w)
        out[m] = g
    return out


def subperiod_panel(panel: dict, start: str | None, end: str | None) -> dict:
    """Rows whose report_date lies in [start, end] (inclusive; None = open). The indices are
    the full-history ones (a 26..156-report rolling window, so no look-ahead), and forward
    returns are attached to the signal date, so an 8-week return may end after `end`."""
    out = {}
    for m, f in panel.items():
        msk = np.ones(len(f), dtype=bool)
        if start is not None:
            msk &= (f["report_date"] >= pd.Timestamp(start)).to_numpy()
        if end is not None:
            msk &= (f["report_date"] <= pd.Timestamp(end)).to_numpy()
        out[m] = f.loc[msk].reset_index(drop=True)
    return out


def primary_rows(panel: dict, idx_col: str = "idx26", ret_col: str = "r8",
                 include_shutdown: bool = False, label: str = "", exact_p: bool = True) -> list:
    """Primary cell (80/20, both sides) for every market plus the 5-market pooled cell."""
    rows = []
    for sd in lc.SIDES:
        for m in lc.MARKETS:
            st = lc.cell_stats(panel[m], idx_col, 80, 20, sd, ret_col,
                               include_shutdown=include_shutdown, exact_p=exact_p)
            rows.append(dict(variant=label, market=m, side=sd, **st))
        pc = lc.pooled_cell(panel, idx_col, 80, 20, sd, ret_col,
                            include_shutdown=include_shutdown)
        rows.append(dict(variant=label, market="pooled5", side=sd,
                         n_signal_weeks=pc["n_signal_weeks"], n_episodes=pc["n_episodes"],
                         mean_ret=pc["mean_ret"], median_ret=pc["median_ret"],
                         hit_rate=pc["hit_rate"], excess=pc["excess_vs_own_baseline"],
                         equal_weight_mkt_excess=pc["equal_weight_mkt_excess"],
                         n_mkts_positive_excess=pc["n_mkts_positive_excess"],
                         perm_p=pc["perm_p"]))
    return rows


# ------------------------------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------------------------------
def _clean(o):
    """Recursively convert numpy scalars and NaN/inf to JSON-safe values (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    return o


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(_clean(obj), indent=1))

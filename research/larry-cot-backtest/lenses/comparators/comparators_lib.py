#!/usr/bin/env python3
"""
comparators_lib -- helpers for the "comparators and incremental value" lens of the
Larry Williams COT backtest (SPEC.md v1).

Everything here builds on the reconciled engine (work/engine/larry_cot.py). Nothing in the
engine is re-implemented: index, timing, eligibility, episodes, the permutation offsets and
the one-cell statistics all come from `larry_cot`. This module only adds:

  * extra per-report columns on the engine panel
      mmc_idx26   larry_index(-mm_net, 26) = 100 - mm_idx26 mathematically. With it the engine's
                  cell_stats(..., 80, 20, side) gives the *contrarian* managed-money cells
                  (long when mm_idx26 <= 20, short when mm_idx26 >= 80) with exact-integer
                  threshold semantics (checked by `check_mm_masks`).
      trend_up    1.0 if the entry-day close > its 200-trading-day SMA (window ends on and
                  includes the entry day), 0.0 if not, NaN if the SMA is undefined or there is
                  no entry. Computed on the same cleaned price series the panel was built on.
      trend_up_lag same, but using the close and SMA of the trading day BEFORE entry
                  (strictly no-look-ahead variant).
      trend_idx   100*trend_up, set to NaN wherever idx26 is NaN, so trend cells use exactly
                  the same eligible weeks (and therefore the same baseline) as the idx26 cells.
                  cell_stats(f, "trend_idx", 50, 50, side, ...) = trend-up (long) /
                  trend-down (short) weeks.
  * the always-in-the-market trend statistic and its circular-shift permutation p,
  * "agreement subsets" (Larry setup AND/NOT trend) with a conditional permutation test that
    shifts the Larry setup while holding the trend state fixed,
  * OLS helpers with Newey-West (single market) and Driscoll-Kraay / date-clustered (panel) SEs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_CODE = Path(__file__).resolve().parents[2] / "engine"   # engine code (larry_cot.py)
if str(ENGINE_CODE) not in sys.path:
    sys.path.insert(0, str(ENGINE_CODE))
import larry_cot as lc  # noqa: E402

ENGINE_DIR = lc.DEFAULT_ROOT / "work" / "engine"              # engine outputs (canonical.csv)

SMA_N = 200          # trend filter length in trading days


# ----------------------------------------------------------------------------------------
# Panel augmentation
# ----------------------------------------------------------------------------------------
def add_comparator_columns(panel: dict, prices: dict, sma_n: int = SMA_N) -> dict:
    """Add mmc_idx26, trend_up, trend_up_lag, trend_idx, sma200_entry to every market frame
    (in place). `prices` must be the same cleaned series the panel was built on, because
    entry_pos indexes into it."""
    for m, f in panel.items():
        # Contrarian-oriented managed-money index (see module docstring).
        f["mmc_idx26"] = lc.larry_index(-f["mm_net"].to_numpy(dtype=float), 26)

        c = prices[m]
        cv = c.to_numpy(dtype=float)
        sma = c.rolling(sma_n, min_periods=sma_n).mean().to_numpy()
        pos = f["entry_pos"].to_numpy(dtype=np.int64)

        def _trend_at(p: np.ndarray) -> np.ndarray:
            out = np.full(len(p), np.nan)
            ok = p >= 0
            s = sma[p[ok]]
            out[ok] = np.where(np.isnan(s), np.nan, (cv[p[ok]] > s).astype(float))
            return out

        f["sma200_entry"] = np.where(pos >= 0, sma[np.where(pos >= 0, pos, 0)], np.nan)
        f["trend_up"] = _trend_at(pos)
        # Variant: decide on the previous trading day's close vs its SMA (pos-1 must exist).
        f["trend_up_lag"] = _trend_at(np.where(pos >= 1, pos - 1, -1))
        f["trend_idx"] = np.where(f["idx26"].isna(), np.nan, 100.0 * f["trend_up"])
        f["trend_idx_lag"] = np.where(f["idx26"].isna(), np.nan, 100.0 * f["trend_up_lag"])
    return panel


def check_mm_masks(panel: dict) -> list[str]:
    """Assert that float threshold comparisons on mm_idx26 / mmc_idx26 equal the exact-integer
    comparisons on mm_net, so that 'contrarian long' == (mm_idx26 <= 20) exactly etc."""
    facts = []
    for m, f in panel.items():
        mm = f["mm_net"].to_numpy(dtype=np.int64)
        for bt, st in lc.THRESHOLDS:
            # trend-following MM: long mm_idx >= bt, short mm_idx <= st
            lo, sh = lc.signal_masks(f["mm_idx26"].to_numpy(), bt, st)
            assert (lo == lc.exact_threshold_mask(mm, 26, bt, True)).all(), (m, bt)
            assert (sh == lc.exact_threshold_mask(mm, 26, st, False)).all(), (m, st)
            # contrarian MM: long when mm_idx <= st (i.e. mmc >= 100-st = bt), short mm_idx >= bt
            clo, csh = lc.signal_masks(f["mmc_idx26"].to_numpy(), bt, st)
            assert (clo == lc.exact_threshold_mask(mm, 26, st, False)).all(), (m, "c", bt)
            assert (csh == lc.exact_threshold_mask(mm, 26, bt, True)).all(), (m, "c", st)
        # mmc and mm have the same NaN pattern
        assert (f["mmc_idx26"].isna() == f["mm_idx26"].isna()).all(), m
    facts.append("mm_idx26/mmc_idx26 float thresholds == exact-integer thresholds on mm_net "
                 "for all markets and threshold pairs; contrarian masks verified")
    return facts


# ----------------------------------------------------------------------------------------
# Always-in trend statistic
# ----------------------------------------------------------------------------------------
def direction_perm_p(d: np.ndarray, r: np.ndarray, observed: float, k: int = lc.K_PERM,
                     seed: int = lc.DEFAULT_SEED) -> tuple[float, float]:
    """Circular-shift p for an always-in strategy with direction d (+1/-1) on the T eligible
    weeks: shifted statistic = mean(np.roll(d, off) * r). Offsets are the engine's
    perm_offsets (closed interval [26, T-26], fresh RNG). Returns (MC p, exact p)."""
    T = len(d)
    offs = lc.perm_offsets(T, k, seed)
    idx = (np.arange(T)[None, :] - offs[:, None]) % T          # np.roll(d, off)[i] = d[i-off]
    sm = (d[idx] * r[None, :]).mean(axis=1)
    p_mc = (1 + int(np.sum(sm >= observed))) / (1 + k)
    all_offs = np.arange(lc.MIN_SHIFT, T - lc.MIN_SHIFT + 1)
    idx2 = (np.arange(T)[None, :] - all_offs[:, None]) % T
    sm2 = (d[idx2] * r[None, :]).mean(axis=1)
    return p_mc, float(np.mean(sm2 >= observed))


def trend_always_in(frame: pd.DataFrame, ret_col: str, trend_col: str = "trend_idx",
                    k: int = lc.K_PERM, seed: int = lc.DEFAULT_SEED) -> dict:
    """Long when trend up, short when trend down, every eligible week (idx26 universe)."""
    elig = lc.eligible_mask(frame, trend_col, ret_col)
    up = frame[trend_col].to_numpy(dtype=float)[elig] >= 50
    r = frame[ret_col].to_numpy(dtype=float)[elig]
    d = np.where(up, 1.0, -1.0)
    sr = d * r
    obs = float(sr.mean())
    p_mc, p_ex = direction_perm_p(d, r, obs, k, seed)
    return dict(n_weeks=int(len(sr)), share_up=float(up.mean()), mean_ret=obs,
                median_ret=float(np.median(sr)), hit_rate=float((sr > 0).mean()),
                baseline_mean=float(r.mean()),
                # E[d*r] if direction were independent of returns: mean(d)*mean(r)
                excess_vs_independent=obs - float(d.mean()) * float(r.mean()),
                perm_p=p_mc, perm_p_exact=p_ex)


def trend_always_in_pooled(panel: dict, markets, ret_col: str, trend_col: str = "trend_idx",
                           k: int = lc.K_PERM, seed: int = lc.DEFAULT_SEED) -> dict:
    """Always-in trend pooled over markets (every market-week weighted equally). Permutation:
    one default_rng(seed); each market gets its own independent offset vector, drawn in market
    order (same scheme as lc.pooled_cell); statistic = pooled mean signed return."""
    rng = np.random.default_rng(seed)
    srs, ex, sums, n_tot = [], [], np.zeros(k), 0
    for m in markets:
        f = panel[m]
        elig = lc.eligible_mask(f, trend_col, ret_col)
        d = np.where(f[trend_col].to_numpy(dtype=float)[elig] >= 50, 1.0, -1.0)
        r = f[ret_col].to_numpy(dtype=float)[elig]
        T = len(d)
        srs.append(d * r)
        ex.append(d * r - d.mean() * r.mean())
        offs = rng.integers(lc.MIN_SHIFT, T - lc.MIN_SHIFT, size=k, endpoint=True)
        idx = (np.arange(T)[None, :] - offs[:, None]) % T
        sums += (d[idx] * r[None, :]).sum(axis=1)
        n_tot += T
    sr = np.concatenate(srs)
    obs = float(sr.mean())
    return dict(n_weeks=int(n_tot), mean_ret=obs, median_ret=float(np.median(sr)),
                hit_rate=float((sr > 0).mean()),
                excess_vs_independent=float(np.concatenate(ex).mean()),
                perm_p=(1 + int(np.sum(sums / n_tot >= obs))) / (1 + k))


# ----------------------------------------------------------------------------------------
# Agreement subsets with a conditional permutation test
# ----------------------------------------------------------------------------------------
def _agreement_arrays(frame, idx_col, buy, sell, ret_col, trend_col):
    """Arrays on the T eligible weeks (idx & trend & return defined, outside shutdown)."""
    elig = lc.eligible_mask(frame, idx_col, ret_col) & frame[trend_col].notna().to_numpy()
    lo, sh = lc.signal_masks(frame[idx_col].to_numpy(), buy, sell)
    up = frame[trend_col].to_numpy(dtype=float)
    r = frame[ret_col].to_numpy(dtype=float)
    r_e, up_e = r[elig], up[elig]
    # mean raw r by trend state (within this market, eligible weeks)
    state_mean = {1.0: float(r_e[up_e == 1.0].mean()), 0.0: float(r_e[up_e == 0.0].mean())}
    return dict(elig=elig, lo_full=lo, sh_full=sh, up_full=up, lo=lo[elig], sh=sh[elig],
                up=up_e, r=r_e, baseline=float(r_e.mean()), state_mean=state_mean)


def _subset_weeks(a, comps, lo=None, sh=None):
    """Signed returns, per-week benchmarks and a mask for a subset defined by components
    [(side, trend_state), ...] on eligible weeks. `lo`/`sh` may be shifted setups."""
    lo = a["lo"] if lo is None else lo
    sh = a["sh"] if sh is None else sh
    mask = np.zeros(len(a["r"]), dtype=bool)
    sign = np.zeros(len(a["r"]))
    for side, state in comps:
        s = (lo if side == "long" else sh) & (a["up"] == state)
        mask |= s
        sign[s] = 1.0 if side == "long" else -1.0      # long/short setups are disjoint
    return mask, sign


def subset_stats_markets(parts: dict, comps, k: int = lc.K_PERM,
                         seed: int = lc.DEFAULT_SEED) -> dict:
    """Stats of an agreement subset for one market (len(parts)==1) or pooled over markets.

    excess_vs_baseline : mean(signed r - sign*market baseline), as in the canonical table.
    excess_vs_same_trend_state : mean(signed r - sign*mean r of ALL eligible weeks in the same
        trend state of that market) -- what the commercial setup adds once the trend state is
        known.
    perm_p : conditional circular-shift test. The long and short setup series are shifted
        together by the same offset (engine offsets, np.roll(+k)); the trend state is held
        fixed; the statistic is the (pooled) mean signed return over shifted-setup weeks that
        fall in the required trend state. One default_rng(seed) per call; for pooled cells each
        market gets its own independent offset vector, drawn in market order (same scheme as
        lc.pooled_cell). p = (1 + #{shifted >= observed}) / (1 + #valid shifts).
    """
    rng = np.random.default_rng(seed)
    sr_all, ex_all, exs_all, n_epi = [], [], [], 0
    sums = np.zeros(k)
    cnts = np.zeros(k)
    for m, a in parts.items():
        mask, sign = _subset_weeks(a, comps)
        sr = sign[mask] * a["r"][mask]
        sr_all.append(sr)
        ex_all.append(sr - sign[mask] * a["baseline"])
        st_mean = np.where(a["up"][mask] == 1.0, a["state_mean"][1.0], a["state_mean"][0.0])
        exs_all.append(sr - sign[mask] * st_mean)
        # episodes: maximal runs of the subset condition on the full report sequence (as in
        # the engine's "full" mode: shutdown weeks / missing returns do not break a run)
        full = np.zeros(len(a["elig"]), dtype=bool)
        for side, state in comps:
            full |= (a["lo_full"] if side == "long" else a["sh_full"]) & (a["up_full"] == state)
        n_epi += lc.count_episodes(full, a["elig"], "full")
        # permutation
        T = len(a["r"])
        offs = rng.integers(lc.MIN_SHIFT, T - lc.MIN_SHIFT, size=k, endpoint=True)
        sgn_r_long, sgn_r_short = a["r"], -a["r"]
        for side, state in comps:
            pos = np.flatnonzero(a["lo"] if side == "long" else a["sh"])
            if not len(pos):
                continue
            newpos = (pos[None, :] + offs[:, None]) % T
            ok = a["up"][newpos] == state
            vals = (sgn_r_long if side == "long" else sgn_r_short)[newpos]
            sums += np.where(ok, vals, 0.0).sum(axis=1)
            cnts += ok.sum(axis=1)
    sr = np.concatenate(sr_all)
    n = int(len(sr))
    out = dict(n_signal_weeks=n, n_episodes=int(n_epi))
    if n == 0:
        return {**out, "mean_ret": np.nan, "median_ret": np.nan, "hit_rate": np.nan,
                "excess_vs_baseline": np.nan, "excess_vs_same_trend_state": np.nan,
                "perm_p": np.nan, "perm_valid_shifts": 0}
    obs = float(sr.mean())
    valid = cnts > 0
    shifted = sums[valid] / cnts[valid]
    p = (1 + int(np.sum(shifted >= obs))) / (1 + int(valid.sum()))
    return {**out, "mean_ret": obs, "median_ret": float(np.median(sr)),
            "hit_rate": float((sr > 0).mean()),
            "excess_vs_baseline": float(np.concatenate(ex_all).mean()),
            "excess_vs_same_trend_state": float(np.concatenate(exs_all).mean()),
            "perm_p": p, "perm_valid_shifts": int(valid.sum())}


def agreement_parts(panel: dict, markets, idx_col="idx26", buy=80, sell=20, ret_col="r8",
                    trend_col="trend_up") -> dict:
    return {m: _agreement_arrays(panel[m], idx_col, buy, sell, ret_col, trend_col)
            for m in markets}


# ----------------------------------------------------------------------------------------
# Regression helpers
# ----------------------------------------------------------------------------------------
def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise over the given (regression) sample: mean 0, sd 1 (ddof=0)."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def regression_frame(panel: dict, markets, ret_col="r8", trend_col="trend_up") -> pd.DataFrame:
    """Stacked regression sample: weeks with idx26, mm_idx26, trend and the forward return
    defined, outside the shutdown windows. Predictors are standardised WITHIN each market over
    that market's regression sample. time_code = position of report_date in the common
    report calendar (gaps such as shutdown weeks are kept as gaps)."""
    cal = sorted(set().union(*[set(panel[m]["report_date"]) for m in markets]))
    code = {d: i for i, d in enumerate(cal)}
    rows = []
    for m in markets:
        f = panel[m]
        ok = (lc.eligible_mask(f, "idx26", ret_col) & f["mm_idx26"].notna().to_numpy()
              & f[trend_col].notna().to_numpy())
        g = f.loc[ok, ["report_date", "idx26", "mm_idx26", trend_col, ret_col]].copy()
        g = g.rename(columns={trend_col: "trend", ret_col: "y"})
        g["market"] = m
        g["z_idx26"] = zscore(g["idx26"].to_numpy())
        g["z_mm_idx26"] = zscore(g["mm_idx26"].to_numpy())
        # discrete signals (N=26, 80/20)
        g["s_comm"] = np.where(g["idx26"] >= 80, 1.0, np.where(g["idx26"] <= 20, -1.0, 0.0))
        g["s_mm_trendfollow"] = np.where(g["mm_idx26"] >= 80, 1.0,
                                         np.where(g["mm_idx26"] <= 20, -1.0, 0.0))
        g["y_volstd"] = g["y"] / g["y"].std()       # return in units of the market's r sd
        g["time_code"] = g["report_date"].map(code).astype(int)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def ols_table(res, names) -> dict:
    """Flatten a statsmodels result into {var: {coef, se, t, p}} plus fit info."""
    out = {}
    for i, nm in enumerate(names):
        out[nm] = dict(coef=float(res.params[i]), se=float(res.bse[i]),
                       t=float(res.tvalues[i]), p=float(res.pvalues[i]))
    return dict(coefs=out, nobs=int(res.nobs), r2=float(res.rsquared),
                r2_adj=float(res.rsquared_adj), cov_type=str(res.cov_type))


def vif(X: np.ndarray, names) -> dict:
    """Variance inflation factors of the non-constant columns of X (X without constant)."""
    out = {}
    for j, nm in enumerate(names):
        y = X[:, j]
        Z = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        r2 = 1 - resid.var() / y.var()
        out[nm] = float(1.0 / (1.0 - r2))
    return out

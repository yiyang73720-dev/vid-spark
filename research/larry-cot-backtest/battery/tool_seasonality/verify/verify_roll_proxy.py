#!/usr/bin/env python3
"""
verify_roll_proxy -- does the tradability verdict depend on the author's pysystemtrade indices,
and how much of the sugar primary edge is a roll-day artifact?

Findings this script documents:
  * The pysystemtrade indices used for the tradability check (seas_data.psys_roll_index) hold the
    pysystemtrade PRICE contract. Per csvconfig/rollconfig.csv that contract is rolled 60 days
    before expiry for HEATOIL, GASOILINE and SUGAR11 (i.e. a deferred contract, not the front
    month), and SOYBEAN holds only November. Their daily returns are also partly misaligned by
    one day against EIA settlements after ~2008.
  * Independent, settlement-based FRONT-month proxies are built here:
      - heating oil / RBOB: EIA contract-1 series (HO_F1, RBOB_F1) with the return of the first
        trading day of each month set to 0 (NYMEX HO/RB expire on the last business day of the
        month before delivery, so EIA's contract 1 changes identity on that day);
      - sugar: the v1 SB continuous series with the first trading day of Mar/May/Jul/Oct set to 0
        (front-month roll days of the H/K/N/V cycle; their mean |return| is reported).
  * The seasonal rule (verify_primary code, estimation = evaluation) is re-run on the proxies,
    the pooled tradability cell is recomputed with them, and the PRIMARY is recomputed with the
    sugar roll days removed.

Run:  python3 -I verify_roll_proxy.py      (writes ROOT/work2/tool_seasonality/verify/verify_roll_proxy.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import verify_primary as vp  # noqa: E402
import seas_data as sd       # noqa: E402  (author's indices: data only)

fh = vp.fh
TRAD_KEYS = ("crude", "heatoil", "gasoline", "gold", "silver", "soybean", "sugar")


def zero_days(raw: pd.Series, mask: np.ndarray, name: str) -> pd.Series:
    """Index rebuilt from `raw` with the daily returns on `mask` days set to 0."""
    r = raw.pct_change().fillna(0.0)
    r[mask] = 0.0
    return (100.0 * (1.0 + r).cumprod()).rename(name)


def first_trading_day_of_month(idx: pd.DatetimeIndex, months=None) -> np.ndarray:
    m = idx.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    if months is not None:
        first &= np.isin(idx.month, months)
    return first


def ho_front() -> pd.Series:
    raw = fh.load_series("ho_f1")
    return zero_days(raw, first_trading_day_of_month(raw.index), "ho_f1_zeroroll")


def rbob_front() -> pd.Series:
    raw = fh.load_series("rbob_f1")
    return zero_days(raw, first_trading_day_of_month(raw.index), "rbob_f1_zeroroll")


def sugar_front() -> pd.Series:
    raw = fh.load_series("sugar")
    return zero_days(raw, first_trading_day_of_month(raw.index, [3, 5, 7, 10]), "sugar_zeroroll")


def frame(ser: pd.Series, L: int) -> pd.DataFrame:
    """Eligible weeks of the seasonal rule with estimation = evaluation = ser."""
    sg = vp.seasonal_signal(vp.forward(ser, vp.GRID_EXT), L)
    df = sg.join(vp.forward(ser, vp.GRID))
    return df[df["sig"].notna() & df["r"].notna()]


def cell(frames: dict) -> dict:
    P = vp.pooled(frames)
    P["perm_p_seed2024"] = vp.perm_independent(frames)
    idx = pd.DatetimeIndex(np.concatenate([f.index.values for f in frames.values()]))
    P["period"] = f"{idx.min().date()}..{idx.max().date()}"
    return P


def main():
    out = {}
    # 1. Sugar roll days are real jumps in the v1 series
    s = fh.load_series("sugar")
    r = s.pct_change().abs()
    mk = first_trading_day_of_month(s.index, [3, 5, 7, 10])
    out["sugar_v1_mean_abs_ret"] = dict(assumed_roll_days=float(r[mk].mean()),
                                        other_days=float(r[~mk][1:].mean()))
    # 2. Agreement of the front proxies with the author's deferred psys indices (8-week returns)
    for key, prox in (("heatoil", ho_front()), ("gasoline", rbob_front())):
        a = vp.forward(prox, vp.GRID)["r"]
        b = vp.forward(sd.tradable_series(key), vp.GRID)["r"]
        j = pd.concat([a.rename("proxy"), b.rename("psys")], axis=1).dropna()
        j = j[j.index >= "2006-01-01"]
        out[f"{key}_8w_return_corr_front_proxy_vs_psys_from_2006"] = dict(
            n=int(len(j)), corr=float(j["proxy"].corr(j["psys"])))
    # 3. Single-market tradable cells on the front proxies
    out["heatoil_front_L15"] = cell({"ho": frame(ho_front(), 15)})
    out["heatoil_front_L10"] = cell({"ho": frame(ho_front(), 10)})
    out["gasoline_rbob_front_L10"] = cell({"rb": frame(rbob_front(), 10)})
    out["sugar_front_L15"] = cell({"sb": frame(sugar_front(), 15)})
    out["sugar_front_L10"] = cell({"sb": frame(sugar_front(), 10)})
    # 4. Pooled tradability cell (L = 15) with front proxies swapped in
    base = {k: frame(sd.tradable_series(k), 15) for k in TRAD_KEYS}
    v1 = dict(base, heatoil=frame(ho_front(), 15))
    v2 = dict(v1, sugar=frame(sugar_front(), 15))
    out["pooled_tradable_psys_as_author"] = cell(base)
    out["pooled_tradable_ho_front"] = cell(v1)
    out["pooled_tradable_ho_front_sugar_front"] = cell(v2)
    # 5. PRIMARY with the sugar roll days removed (sugar L = 10 as pre-registered)
    prim = {k: vp.market_frame(n, L) for k, n, L in vp.MARKETS}
    out["primary_as_registered"] = cell(prim)
    p2 = dict(prim, sugar=frame(sugar_front(), 10))
    out["primary_sugar_rolls_zeroed"] = cell(p2)
    p3 = {k: f for k, f in p2.items() if k != "gasoline"}
    out["primary_sugar_rolls_zeroed_without_gasoline"] = cell(p3)
    vp.HERE.mkdir(parents=True, exist_ok=True)       # outputs: ROOT/work2/tool_seasonality/verify
    (vp.HERE / "verify_roll_proxy.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

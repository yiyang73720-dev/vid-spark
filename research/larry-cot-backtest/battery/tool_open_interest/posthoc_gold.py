#!/usr/bin/env python3
"""
posthoc_gold -- POST-HOC diagnostic, NOT pre-registered, NOT used for the verdict.

Why: in the robustness grid the primary OI rule applied to GOLD (F+O OI, v1 London spot) gave
the strongest cell (long, h=8, perm_p 0.0005), and 14 of the 28 h=8 cells with perm_p <= 0.05
are gold "large OI decline -> long" variants. This script asks whether that gold cell survives
the same out-of-sample and strategy checks the primary gets (split half, S1/S2 with costs and
random-timing null), and how it looks on the longer futures-only legacy OI (2000-2020). It was
written AFTER seeing the grid, so any number here is exploratory.

Run:  python3 -I posthoc_gold.py   -> posthoc_gold.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))   # oi_lib
import oi_lib as ol          # noqa: E402
from oi_lib import fh        # noqa: E402
WORK = fh.WORK2 / "tool_open_interest"         # outputs: ROOT/work2/tool_open_interest
WORK.mkdir(parents=True, exist_ok=True)

LAB = dict(filter="open_interest", variant="primary_rule_gold_POSTHOC", market="gold",
           is_primary=False)


def main() -> None:
    fo = ol.load_fo_cot()
    f = ol.build_features(fo["gold"], fh.load_series("gold"))
    sf = ol.signal_frame(fo["gold"], ol.rule_primary(f))
    halves = fh.split_half(sf, "gold", timing="cot", horizon=8, sides=("long",),
                           horizons=(4, 8, 13), labels=LAB)
    st = fh.simulate(sf, "gold", timing="cot", labels=LAB)
    rep = ol.load_legacy_futonly("gold")
    fl = ol.build_features(rep, fh.load_series("gold"))
    sfl = ol.signal_frame(rep, ol.rule_primary(fl))
    halves_leg = fh.split_half(sfl, "gold", timing="cot", horizon=8, sides=("long",),
                               horizons=(8,), labels={**LAB, "variant": LAB["variant"] +
                                                      "_legacy_futonly"})
    out = {"note": "POST-HOC, not pre-registered, not used for the verdict",
           "split_half_FO": halves, "split_half_legacy_futures_only_2000_2020": halves_leg,
           "strategy_FO": st["metrics"]}
    (WORK / "posthoc_gold.json").write_text(json.dumps(fh.to_jsonable(out), indent=1))
    for _, r in halves[halves["horizon_w"] == 8].iterrows():
        print(f"F+O {r['half']} long h8: n={r['n']} ep={r['episodes']} excess={r['excess']} "
              f"perm_p={r['perm_p']} period={r['period']}")
    for _, r in halves_leg.iterrows():
        print(f"legacy {r['half']} long h8: n={r['n']} ep={r['episodes']} excess={r['excess']} "
              f"perm_p={r['perm_p']} period={r['period']}")
    for _, r in st["metrics"].iterrows():
        print(f"{r['strategy']}: trades={r.get('n_trades')} CAGR={r['ann_return']} "
              f"sharpe={r['sharpe']} maxDD={r['max_dd']} null_p={r.get('null_p_sharpe')}")


if __name__ == "__main__":
    main()

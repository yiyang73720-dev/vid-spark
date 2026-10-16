#!/usr/bin/env python3
"""
verify_rebuild -- check that rank_lib rebuilds every tool's PRE-REGISTERED primary exactly
(n, mean, excess, hit, Monte-Carlo perm_p, T) before the ranking / confluence code uses the
rebuilt signals. Only already-published numbers are reproduced here; no new test is run.

Run: python3 -I verify_rebuild.py      -> writes ROOT/work2/rank/verify_rebuild.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json                         # noqa: E402
from pathlib import Path            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))   # rank_lib

import pandas as pd                 # noqa: E402

import rank_lib as rl               # noqa: E402

fh = rl.fh


def published_primaries() -> dict:
    """Published primary rows: v1 canonical.csv for COT, results.json 'primary' for the rest."""
    can = pd.read_csv(rl.ROOT / "work" / "engine" / "canonical.csv", float_precision="round_trip")
    c = can[(can.market == "crude") & (can.lookback == 26) & (can.buy_th == 80)
            & (can.side == "long") & (can.horizon_w == 8)].iloc[0]
    out = {"cot": dict(n=c["n_signal_weeks"], mean=c["mean_ret"], excess=c["excess"],
                       hit=c["hit_rate"], perm_p=c["perm_p"])}
    for t in ("premium", "seasonality", "valuation", "open_interest", "intermarket"):
        p = rl.load_results(t)["primary"]
        out[t] = {k: p.get(k) for k in ("n", "mean", "excess", "hit", "perm_p", "T")}
    return out


def main() -> None:
    pub = published_primaries()
    res = rl.verify_primaries(pub)
    for t, v in res.items():
        print(f"{t:14s} match={v['match']} diffs={v['diffs']} rebuilt n={v['rebuilt']['n']} "
              f"excess={v['rebuilt']['excess']:.6f} perm_p={v['rebuilt']['perm_p']}")
    rl.RANK.mkdir(parents=True, exist_ok=True)
    (rl.RANK / "verify_rebuild.json").write_text(
        json.dumps(fh.to_jsonable(dict(published=pub, rebuilt=res)), indent=1))
    if not all(v["match"] for v in res.values()):
        raise SystemExit("a rebuilt primary differs from the published one")
    print("ALL PRIMARIES REBUILT EXACTLY")


if __name__ == "__main__":
    main()

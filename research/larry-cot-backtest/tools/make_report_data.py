"""Collect the verified backtest outputs into one compact JSON for the report page.

Usage: python3 -I make_report_data.py [WORK_DIR [OUT_JSON]]
  WORK_DIR defaults to ROOT/work, OUT_JSON to ROOT/work/report/data.json
  (ROOT = $LARRY_COT_ROOT, else the package dir).
"""
import json, os, sys
import pandas as pd

ROOT = os.environ.get("LARRY_COT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "work")  # ROOT/work
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(W, "report", "data.json")
os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
out = {}

# --- crude 2025-2026 timeline
t = json.load(open(f"{W}/lens_crude2026/timeline.json"))
out["daily"] = [[r["date"], None if r["cl_close"] is None else round(r["cl_close"], 2),
                 None if r["spot_close"] is None else round(r["spot_close"], 2)] for r in t["daily"]]
out["weekly"] = [[r["report_date"], r["release_nominal"], round(r["idx26"], 2), round(r["mm_idx26"], 2),
                  int(r["long_setup_80"]), int(r["short_setup_20"]), int(r["shutdown_delayed"]),
                  r["comm_net"]] for r in t["weekly"]]
out["hypothetical"] = {k: t["hypothetical"][k] for k in ("report_date", "comm_net", "idx26")}

# --- primary cell (N26, 80/20, h=8) per market and pooled
can = pd.read_csv(f"{W}/engine/canonical.csv")
pc = can[(can.lookback == 26) & (can.buy_th == 80) & (can.horizon_w == 8)]
bars = []
for _, r in pc.iterrows():
    bars.append(dict(market=r.market, side=r.side, n=int(r.n_signal_weeks), ep=int(r.n_episodes),
                     mean=r.mean_ret, base=(r.baseline_mean if r.side == "long" else -r.baseline_mean),
                     excess=r.excess, hit=r.hit_rate, p=r.perm_p))
pool = pd.read_csv(f"{W}/engine/pooled.csv")
pp = pool[(pool.lookback == 26) & (pool.buy_th == 80) & (pool.horizon_w == 8)]
for _, r in pp.iterrows():
    bars.append(dict(market="pooled", side=r.side, n=int(r.n_signal_weeks), ep=int(r.n_episodes),
                     mean=r.mean_ret, base=r.mean_ret - r.excess_vs_own_baseline,
                     excess=r.excess_vs_own_baseline, hit=r.hit_rate, p=r.perm_p))
out["bars"] = bars

# --- equity curves (weekly sampled)
c = json.load(open(f"{W}/lens_strategy/charts.json"))
keep = ["S1_long_short", "S2_long_short", "buy_hold"]
out["equity"] = {m: {"dates": c[m]["dates"], **{k: [round(v, 4) for v in c[m]["series"][k]] for k in keep}}
                 for m in ("crude", "portfolio")}
met = pd.read_csv(f"{W}/lens_strategy/metrics.csv")
rows = []
for _, r in met[met.market.isin(["crude", "portfolio"])].iterrows():
    rows.append(dict(market=r.market, strategy=r.strategy, trades=(None if pd.isna(r.n_trades) else int(r.n_trades)), win=(None if pd.isna(r.win_rate) else r.win_rate),
                     cagr=r.ann_return, sharpe=r.sharpe, mdd=r.max_dd, p=(None if pd.isna(r.null_p_sharpe) else r.null_p_sharpe),
                     bh_cagr=r.bh_ann_return, bh_sharpe=r.bh_sharpe, bh_mdd=r.bh_max_dd))
out["metrics"] = rows

# --- robustness heatmap (excess per cell)
h = json.load(open(f"{W}/lens_robustness/heatmaps.json"))
out["heat"] = [[r["market"], r["side"], r["lookback"], r["threshold"], r["horizon_w"],
                round(r["excess"], 5), round(r["perm_p"], 4), r["n_signal_weeks"]] for r in h["records"]]

json.dump(out, open(OUT_JSON, "w"), separators=(",", ":"), default=float)
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})

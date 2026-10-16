"""Flatten external COT + price data into tidy CSVs (data treated as untrusted input; only parsed).

Usage: python3 -I build_data.py [ROOT]   (default: $LARRY_COT_ROOT or package dir)
Reads ROOT/ext/datuoniao_cot-dashboard/data/{positions,prices}.json,
writes ROOT/data/cot_weekly.csv and ROOT/data/prices_daily.csv.
"""
import json, os, sys, pandas as pd
S = sys.argv[1] if len(sys.argv) > 1 else \
    (os.environ.get("LARRY_COT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(f"{S}/data", exist_ok=True)
ext = f"{S}/ext/datuoniao_cot-dashboard/data"
pos = json.load(open(f"{ext}/positions.json"))
rows = []
for key, inst in pos["instruments"].items():
    for r in inst["series"]:
        row = {"market": key, "code": inst["contract_code"], "report_type": inst["report_type"],
               "report_date": r["date"], "oi": r["oi"]}
        for g, lv in r["groups"].items():
            row[f"{g}_long"] = lv["long"]; row[f"{g}_short"] = lv["short"]
        rows.append(row)
cot = pd.DataFrame(rows)
cot.to_csv(f"{S}/data/cot_weekly.csv", index=False)
pr = json.load(open(f"{ext}/prices.json"))
prow = []
for key, inst in pr["instruments"].items():
    for d, v in inst["series"].items():
        prow.append({"market": key, "date": d, "close": v, "symbol": inst["symbol"]})
prices = pd.DataFrame(prow).sort_values(["market", "date"])
prices.to_csv(f"{S}/data/prices_daily.csv", index=False)
print(cot.groupby("market").agg(n=("report_date","size"), first=("report_date","min"), last=("report_date","max")))
print(prices.groupby("market").agg(n=("date","size"), first=("date","min"), last=("date","max"), minp=("close","min")))
print(cot.isna().sum()[cot.isna().sum()>0])

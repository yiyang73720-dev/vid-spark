"""Command line: ``python3 -m sepa {scan,fetch} ...``."""

import argparse
import json
import os
import sys

from .config import Config
from .screen import scan, summary_row


def _tickers(values, path):
    out = []
    for value in values or []:
        out += [t.strip().upper() for t in value.replace(",", " ").split() if t.strip()]
    if path:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0]
                out += [t.strip().upper() for t in line.replace(",", " ").split() if t.strip()]
    return list(dict.fromkeys(out))


def _rs_overrides(values):
    out = {}
    for value in values or []:
        ticker, _, rating = value.partition("=")
        out[ticker.strip().upper()] = float(rating)
    return out


def _pct(value):
    return "" if value is None else f"{value:+.1%}"


def print_table(report, out=None):
    out = out or sys.stdout
    market = report.get("market")
    if market:
        out.write(f"Market ({market['benchmark']}): {market['stage']['label']}; "
                  f"trend template price criteria {'met' if market['trend'] else 'NOT met'}\n")
    if report["rs_note"]:
        out.write(f"note: {report['rs_note']}\n")
    header = f"{'ticker':<7}{'stage':>6}{'TT':>6}{'RS':>5}  {'T':<11}{'E':<11}{'S':<21}{'pivot':>10}{'vs pivot':>10}"
    out.write(header + "\n" + "-" * len(header) + "\n")
    for result in report["results"]:
        row = summary_row(result)
        vs = row["extension"] if row["extension"] is not None else (
            -row["below_pivot"] if row["below_pivot"] is not None else None)
        out.write(
            f"{row['ticker']:<7}{str(row['stage'] or '-'):>6}{row['trend_passed']:>4}/8"
            f"{str(int(row['rs_rating'])) if row['rs_rating'] is not None else '-':>5}  "
            f"{row['pillars']['trend']:<11}{row['pillars']['earnings']:<11}"
            f"{row['entry_status']:<21}{row['pivot'] if row['pivot'] is not None else '-':>10}"
            f"{_pct(vs):>10}\n"
        )
    for ticker, error in sorted(report["errors"].items()):
        out.write(f"{ticker}: {error}\n")


def write_reports(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for result in report["results"]:
        with open(os.path.join(out_dir, f"{result['ticker']}.json"), "w") as fh:
            json.dump(result, fh, indent=1)
    summary = {k: v for k, v in report.items() if k not in ("results", "config")}
    summary["tickers"] = [summary_row(r) for r in report["results"]]
    summary["candidates"] = [r["ticker"] for r in report["results"] if r["candidate"]]
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m sepa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="run the mechanical SEPA checks")
    p_scan.add_argument("--data-dir", default="sepa-data")
    p_scan.add_argument("--tickers", nargs="*", help="tickers to evaluate (default: all)")
    p_scan.add_argument("--tickers-file")
    p_scan.add_argument("--benchmark", default="SPY")
    p_scan.add_argument("--asof", help="evaluate as of this date (YYYY-MM-DD)")
    p_scan.add_argument("--rs", nargs="*", metavar="TICKER=RATING",
                        help="RS ratings from IBD/MarketSmith, overriding the computed ones")
    p_scan.add_argument("--config", help="JSON file overriding thresholds (see sepa/config.py)")
    p_scan.add_argument("--json", action="store_true", help="print the full JSON report")
    p_scan.add_argument("--out-dir", help="write <TICKER>.json reports and summary.json here")
    p_scan.add_argument("--candidates-only", action="store_true",
                        help="only print tickers that pass the trend pillar")

    p_fetch = sub.add_parser("fetch", help="download prices + fundamentals with yfinance")
    p_fetch.add_argument("--data-dir", default="sepa-data")
    p_fetch.add_argument("--tickers", nargs="*")
    p_fetch.add_argument("--tickers-file")
    p_fetch.add_argument("--benchmark", default="SPY")
    p_fetch.add_argument("--period", default="2y")
    p_fetch.add_argument("--no-fundamentals", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        from .fetch import fetch

        tickers = _tickers(args.tickers, args.tickers_file)
        if args.benchmark and args.benchmark.upper() not in tickers:
            tickers.append(args.benchmark.upper())
        if not tickers:
            parser.error("give --tickers or --tickers-file")
        failed = fetch(tickers, args.data_dir, args.period, not args.no_fundamentals,
                       log=lambda msg: print(msg, file=sys.stderr))
        print(json.dumps({"fetched": len(tickers) - len(failed), "failed": failed}))
        return 1 if len(failed) == len(tickers) else 0

    cfg = Config()
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = Config.from_dict(json.load(fh))
    tickers = _tickers(args.tickers, args.tickers_file) or None
    report = scan(args.data_dir, tickers, args.benchmark, args.asof,
                  _rs_overrides(args.rs), cfg)
    if args.candidates_only:
        report["results"] = [r for r in report["results"] if r["candidate"]]
    if args.out_dir:
        summary = write_reports(report, args.out_dir)
        if not args.json:
            print(json.dumps(summary, indent=1))
            return 0
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print_table(report)
    return 0

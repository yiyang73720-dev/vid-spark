"""Tests for the SEPA engine on synthetic price paths. Run: python3 -m unittest discover tests"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import date, timedelta

from sepa.cli import main
from sepa.config import Config
from sepa.data import Bars, load_prices_csv
from sepa.fundamentals import analyze_fundamentals
from sepa.indicators import sma, weekly
from sepa.rs import percentile_ratings, rs_line, weighted_performance
from sepa.trend import trend_template, weinstein_stage
from sepa.vcp import analyze_entry, detect_base

CFG = Config()


def trading_days(count, start=date(2023, 1, 2)):
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def build(segments, start=50.0, spread=0.005):
    """Bars from (bar_count, target_close, volume) segments, linear between waypoints."""
    closes, volumes = [start], [1_000_000.0]
    for count, target, volume in segments:
        begin = closes[-1]
        for k in range(1, count + 1):
            closes.append(begin + (target - begin) * k / count)
            volumes.append(float(volume))
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * (1 + spread) for o, c in zip(opens, closes)]
    lows = [min(o, c) * (1 - spread) for o, c in zip(opens, closes)]
    return Bars(trading_days(len(closes)), opens, highs, lows, closes, volumes)


UPTREND = [(250, 100, 1_000_000)]
# A shallower base after a longer advance keeps the 50-day MA above the 150-day,
# so the whole Trend Template holds while the VCP sets up.
LONG_UPTREND = [(300, 100, 1_000_000)]
SHALLOW_VCP = [
    (10, 85, 1_400_000),  # ~15%
    (10, 97, 1_000_000),
    (6, 90, 800_000),  # ~8%
    (6, 96, 700_000),
    (4, 93, 500_000),  # ~4%
    (3, 95.5, 400_000),
]
VCP_BASE = [
    (15, 75, 1_500_000),  # contraction 1: ~25%
    (15, 95, 1_000_000),
    (8, 85, 800_000),  # contraction 2: ~11%
    (8, 93, 700_000),
    (5, 89, 500_000),  # contraction 3: ~5%
    (4, 92, 400_000),
]


class IndicatorTests(unittest.TestCase):
    def test_sma(self):
        self.assertEqual(sma([1, 2, 3, 4, 5], 3), [None, None, 2.0, 3.0, 4.0])

    def test_weekly_aggregation(self):
        bars = build([(9, 60, 1000)])
        weeks = weekly(bars)
        self.assertEqual(sum(w["volume"] for w in weeks), sum(bars.volume))
        self.assertEqual(weeks[-1]["close"], bars.close[-1])


class TrendTemplateTests(unittest.TestCase):
    def test_steady_uptrend_passes_all_eight(self):
        bars = build([(300, 100, 1_000_000)])
        result = trend_template(bars, 85, CFG.trend)
        self.assertTrue(result["passes_all"], result)
        self.assertEqual(result["passed"], 8)
        stage = weinstein_stage(bars, CFG.stage)
        self.assertEqual(stage["stage"], 2)
        self.assertGreater(stage["stage2_days"], 50)

    def test_missing_rs_is_unknown_not_pass(self):
        bars = build([(300, 100, 1_000_000)])
        result = trend_template(bars, None, CFG.trend)
        self.assertFalse(result["passes_all"])
        self.assertTrue(result["passes_price_criteria"])
        self.assertEqual(result["unknown"], [8])

    def test_low_rs_fails_criterion_8(self):
        bars = build([(300, 100, 1_000_000)])
        self.assertEqual(trend_template(bars, 60, CFG.trend)["failed"], [8])

    def test_downtrend_is_stage_4_and_fails(self):
        bars = build([(300, 30, 1_000_000)], start=100)
        result = trend_template(bars, 20, CFG.trend)
        self.assertEqual(set(result["failed"]), {1, 2, 3, 4, 5, 6, 7, 8})
        self.assertEqual(weinstein_stage(bars, CFG.stage)["stage"], 4)

    def test_deep_pullback_fails_within_25pct_of_high(self):
        bars = build([(280, 100, 1_000_000), (20, 70, 1_000_000)])
        self.assertIn(7, trend_template(bars, 80, CFG.trend)["failed"])

    def test_insufficient_history_is_unknown(self):
        bars = build([(120, 80, 1_000_000)])
        result = trend_template(bars, 90, CFG.trend)
        self.assertIn(1, result["unknown"])
        self.assertIn(3, result["unknown"])
        self.assertFalse(result["passes_all"])
        self.assertIsNone(weinstein_stage(bars, CFG.stage)["stage"])

    def test_ma200_up_months_counts_consecutive_months(self):
        bars = build([(300, 100, 1_000_000)])
        up = trend_template(bars, 90, CFG.trend)["criteria"][2]["value"]["up_months"]
        self.assertGreaterEqual(up, 4)

    def test_stage_1_after_decline_then_flat(self):
        bars = build([(200, 40, 1_000_000), (200, 41, 1_000_000)], start=80)
        self.assertEqual(weinstein_stage(bars, CFG.stage)["stage"], 1)

    def test_stage_3_after_advance_then_flat(self):
        bars = build([(250, 100, 1_000_000), (120, 99, 1_000_000), (5, 96, 1_000_000)], start=40)
        self.assertEqual(weinstein_stage(bars, CFG.stage)["stage"], 3)


class RelativeStrengthTests(unittest.TestCase):
    def test_weighted_performance_formula(self):
        closes = [100.0] * 253
        closes[-1 - 63] = 50.0  # 3-month return +100%
        self.assertAlmostEqual(weighted_performance(closes, CFG.rs.weights), 0.4)

    def test_weighted_performance_needs_a_year(self):
        self.assertIsNone(weighted_performance([1.0] * 200, CFG.rs.weights))

    def test_percentiles(self):
        ratings = percentile_ratings({"A": 0.1, "B": 0.2, "C": 0.3, "D": None}, 2)
        self.assertEqual((ratings["A"], ratings["B"], ratings["C"]), (1, 50, 99))
        self.assertIsNone(ratings["D"])

    def test_small_universe_gives_no_rating(self):
        self.assertEqual(percentile_ratings({"A": 0.1, "B": 0.2}, 20), {"A": None, "B": None})

    def test_rs_line_new_high(self):
        stock = build([(300, 100, 1)])
        bench = build([(300, 60, 1)])
        line = rs_line(stock, bench, 252)
        self.assertTrue(line["at_new_high"])


def quarter_ends(count, start=date(2023, 3, 31)):
    out, year, month = [], start.year, start.month
    for _ in range(count):
        day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
        out.append(date(year, month, day).isoformat())
        month += 3
        if month > 12:
            month, year = 3, year + 1
    return out


def fundamentals(eps, revenue, net_income, estimates=None, guidance=None):
    ends = quarter_ends(len(eps))
    quarters = []
    for i, end in enumerate(ends):
        quarters.append({
            "period_end": end, "eps": eps[i], "revenue": revenue[i], "net_income": net_income[i],
            "eps_estimate": (estimates or [None] * len(eps))[i],
        })
    return {"quarters": quarters, "guidance": guidance}


class FundamentalsTests(unittest.TestCase):
    def test_accelerating_growth_passes(self):
        data = fundamentals(
            eps=[1.0, 1.0, 1.0, 1.0, 1.2, 1.25, 1.35, 1.5],
            revenue=[100, 100, 100, 100, 110, 112, 118, 125],
            net_income=[10, 10, 10, 10, 12, 12.5, 13.5, 15],
            estimates=[None] * 7 + [1.4], guidance="raised",
        )
        result = analyze_fundamentals(data, CFG.fundamentals)
        self.assertTrue(result["core_pass"], result)
        checks = {c["id"]: c["pass"] for c in result["checks"]}
        self.assertEqual(checks, {"E1": True, "E2": True, "E3": True, "E4": True, "E5": True, "E6": True})
        self.assertIn("3 quarter(s)", result["checks"][1]["detail"])

    def test_decelerating_growth_fails(self):
        data = fundamentals(
            eps=[1.0, 1.0, 1.0, 1.0, 1.5, 1.4, 1.3, 1.25],
            revenue=[100] * 4 + [130, 125, 120, 115],
            net_income=[10] * 4 + [15, 14, 13, 12.5],
        )
        result = analyze_fundamentals(data, CFG.fundamentals)
        self.assertFalse(result["core_pass"])
        self.assertFalse(result["checks"][1]["pass"])

    def test_turnaround_is_not_a_growth_number(self):
        data = fundamentals(
            eps=[-0.5, -0.4, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            revenue=[100] * 4 + [140, 145, 150, 160], net_income=[-5] * 4 + [5] * 4,
        )
        result = analyze_fundamentals(data, CFG.fundamentals)
        self.assertIsNone(result["checks"][0]["pass"])
        self.assertIn("turnaround", result["checks"][0]["detail"])
        self.assertIsNone(result["core_pass"])

    def test_missing_year_ago_quarter(self):
        data = fundamentals(eps=[1, 2], revenue=[1, 2], net_income=[1, 2])
        result = analyze_fundamentals(data, CFG.fundamentals)
        self.assertIsNone(result["core_pass"])
        self.assertEqual(result["missing"], ["E1", "E2", "E3", "E4"])

    def test_asof_hides_later_quarters(self):
        data = fundamentals(
            eps=[1.0] * 4 + [1.2, 1.25, 1.35, 1.5], revenue=[100] * 8, net_income=[10] * 8,
        )
        result = analyze_fundamentals(data, CFG.fundamentals, asof=quarter_ends(8)[6])
        self.assertEqual(result["history"][-1]["period_end"], quarter_ends(8)[6])


class VCPTests(unittest.TestCase):
    def test_textbook_vcp_near_pivot(self):
        bars = build(UPTREND + VCP_BASE)
        entry = analyze_entry(bars, CFG.vcp)
        base = entry["base"]
        self.assertTrue(base["valid"], base["reasons"])
        depths = [c["depth"] for c in base["contractions"]]
        self.assertEqual(len(depths), 3, depths)
        self.assertEqual(depths, sorted(depths, reverse=True))
        self.assertAlmostEqual(base["pivot"], 93 * 1.005, places=2)
        self.assertEqual(entry["status"], "near_pivot")
        self.assertTrue(entry["actionable"])
        self.assertTrue(entry["risk_ok"])
        self.assertLess(base["volume"]["final_vs_avg50"], 1)

    def test_breakout_on_volume(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 3_000_000)])
        entry = analyze_entry(bars, CFG.vcp)
        self.assertEqual(entry["status"], "breakout", entry)
        self.assertGreaterEqual(entry["breakout_volume_ratio"], 1.4)
        self.assertEqual(entry["days_since_breakout"], 0)

    def test_breakout_on_light_volume_is_unconfirmed(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 500_000)])
        self.assertEqual(analyze_entry(bars, CFG.vcp)["status"], "breakout_unconfirmed")

    def test_extended_is_not_actionable(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 3_000_000), (3, 99.5, 2_000_000)])
        entry = analyze_entry(bars, CFG.vcp)
        self.assertEqual(entry["status"], "extended")
        self.assertFalse(entry["actionable"])

    def test_run_to_new_highs_still_reports_extended(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 3_000_000), (6, 110, 2_000_000)])
        entry = analyze_entry(bars, CFG.vcp)
        self.assertEqual(entry["status"], "extended", entry)
        self.assertGreater(entry["extension"], 0.15)

    def test_shallow_failed_breakout(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 3_000_000), (2, 93, 900_000)])
        self.assertEqual(analyze_entry(bars, CFG.vcp)["status"], "failed_breakout")

    def test_deep_failed_breakout_is_flagged(self):
        bars = build(UPTREND + VCP_BASE + [(1, 95, 3_000_000), (3, 90, 900_000)])
        entry = analyze_entry(bars, CFG.vcp)
        self.assertIn(entry["status"], ("failed_breakout", "forming", "near_pivot"))
        if entry["status"] != "failed_breakout":
            self.assertTrue(any("failed breakout" in w for w in entry.get("warnings", [])), entry)

    def test_lower_low_merges_into_one_correction(self):
        # 100 -> 90, 98 -> 80: the undercut makes it one 20% correction, not a VCP.
        base = [(15, 90, 1_000_000), (10, 98, 1_000_000), (15, 80, 1_500_000), (10, 95, 1_000_000)]
        entry = analyze_entry(build(UPTREND + base), CFG.vcp)
        self.assertEqual(entry["status"], "no_base")
        self.assertTrue(any("contraction_count" in r for r in entry["reasons"]), entry)

    def test_expanding_contraction_is_not_a_vcp(self):
        # 25% -> 6% -> 12%: the last pullback widens from a lower high.
        base = [(15, 75, 1_000_000), (15, 95, 1_000_000), (5, 90, 800_000), (5, 93, 800_000),
                (8, 82, 900_000), (5, 88, 700_000)]
        entry = analyze_entry(build(UPTREND + base), CFG.vcp)
        self.assertEqual(entry["status"], "no_base")
        self.assertTrue(any("contractions_shrinking" in r for r in entry["reasons"]), entry)

    def test_single_correction_is_not_a_vcp(self):
        entry = analyze_entry(build(UPTREND + [(20, 80, 1_000_000), (20, 97, 900_000)]), CFG.vcp)
        self.assertEqual(entry["status"], "no_base")

    def test_at_new_highs_there_is_no_base(self):
        base = detect_base(build(UPTREND), 251, CFG.vcp)
        self.assertFalse(base["valid"])
        self.assertIn("at or near highs", base["reasons"][0])

    def test_rally_dip_is_not_counted_as_contraction(self):
        # 100 -> 75, rally to 88 with a 3.5% dip at 85, then 95 -> 86 -> 93 -> 89 -> 92.
        base = [
            (15, 75, 1_500_000), (8, 88, 1_000_000), (3, 85, 1_000_000), (8, 95, 1_000_000),
            (8, 86, 800_000), (8, 93, 700_000), (5, 89, 500_000), (4, 92, 400_000),
        ]
        entry = analyze_entry(build(UPTREND + base), CFG.vcp)
        self.assertTrue(entry["base"]["valid"], entry["base"].get("reasons"))
        self.assertEqual(len(entry["base"]["contractions"]), 3)


def write_csv(path, bars):
    with open(path, "w") as fh:
        fh.write("Date,Open,High,Low,Close,Adj Close,Volume\n")
        for row in zip(bars.dates, bars.open, bars.high, bars.low, bars.close, bars.close, bars.volume):
            fh.write(",".join(str(x) for x in row) + "\n")


class DataAndCliTests(unittest.TestCase):
    def test_csv_loader_handles_order_nulls_and_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "X.csv")
            with open(path, "w") as fh:
                fh.write("date,OPEN,High,low,Close,volume\n")
                fh.write("2024-01-03,2,3,1,2.5,100\n")
                fh.write("2024-01-02,1,2,0.5,1.5,100\n")
                fh.write("2024-01-04,null,null,null,null,null\n")
                fh.write("20240105,3,4,2,3.5,\n")
            bars = load_prices_csv(path)
        self.assertEqual(bars.dates, ["2024-01-02", "2024-01-03", "2024-01-05"])
        self.assertEqual(bars.close, [1.5, 2.5, 3.5])
        self.assertEqual(bars.volume[-1], 0.0)

    def test_scan_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "prices"))
            os.makedirs(os.path.join(tmp, "fundamentals"))
            write_csv(os.path.join(tmp, "prices", "VCP.csv"), build(LONG_UPTREND + SHALLOW_VCP, start=40))
            write_csv(os.path.join(tmp, "prices", "DOWN.csv"), build([(300, 30, 1_000_000)], start=100))
            write_csv(os.path.join(tmp, "prices", "SPY.csv"), build([(300, 60, 1_000_000)]))
            with open(os.path.join(tmp, "fundamentals", "VCP.json"), "w") as fh:
                json.dump(fundamentals(
                    eps=[1.0] * 4 + [1.2, 1.25, 1.35, 1.5], revenue=[100] * 4 + [110, 112, 118, 125],
                    net_income=[10] * 4 + [12, 12.5, 13.5, 15]), fh)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["scan", "--data-dir", tmp, "--json", "--rs", "VCP=92"]), 0)
            report = json.loads(out.getvalue())
            by_ticker = {r["ticker"]: r for r in report["results"]}
            self.assertEqual(set(by_ticker), {"VCP", "DOWN"})
            self.assertEqual(by_ticker["VCP"]["pillars"]["trend"], "pass")
            self.assertEqual(by_ticker["VCP"]["pillars"]["earnings"], "pass")
            self.assertEqual(by_ticker["VCP"]["pillars"]["entry"], "near_pivot")
            self.assertEqual(by_ticker["DOWN"]["pillars"]["trend"], "fail")
            self.assertEqual(by_ticker["DOWN"]["pillars"]["earnings"], "no_data")
            self.assertEqual(report["market"]["stage"]["stage"], 2)
            self.assertIn("RS ratings unknown", report["rs_note"])

            out_dir = os.path.join(tmp, "reports")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["scan", "--data-dir", tmp, "--out-dir", out_dir, "--rs", "VCP=92"])
            with open(os.path.join(out_dir, "summary.json")) as fh:
                summary = json.load(fh)
            self.assertEqual(summary["candidates"], ["VCP"])
            self.assertTrue(os.path.exists(os.path.join(out_dir, "VCP.json")))

            table = io.StringIO()
            with contextlib.redirect_stdout(table):
                main(["scan", "--data-dir", tmp])
            self.assertIn("VCP", table.getvalue())

    def test_config_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"trend": {"nope": 1}})
        self.assertEqual(Config.from_dict({"trend": {"min_above_low": 0.25}}).trend.min_above_low, 0.25)


if __name__ == "__main__":
    unittest.main()

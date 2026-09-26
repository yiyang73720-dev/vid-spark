"""Tests for ff_usd_news.py — run with: python3 -m unittest test_ff_usd_news.py"""
import contextlib
import io
import json
import re
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ff_usd_news as ff

HERE = Path(__file__).resolve().parent

# Shaped like https://nfs.faireconomy.media/ff_calendar_thisweek.json (made-up values).
SAMPLE_JSON = r"""[
{"title":"Bank Holiday","country":"USD","date":"2026-10-12T00:00:00-04:00","impact":"Holiday","forecast":"","previous":""},
{"title":"German Prelim CPI m\/m","country":"EUR","date":"2026-10-13T08:00:00-04:00","impact":"High","forecast":"0.2%","previous":"0.1%"},
{"title":"Core CPI m\/m","country":"USD","date":"2026-10-14T08:30:00-04:00","impact":"High","forecast":"0.3%","previous":"0.2%"},
{"title":"CPI y\/y","country":"USD","date":"2026-10-14T08:30:00-04:00","impact":"High","forecast":"2.9%","previous":"2.8%"},
{"title":"Fed's Beige Book","country":"USD","date":"2026-10-14T14:00:00-04:00","impact":"Low","forecast":"","previous":""},
{"title":"Crude Oil Inventories","country":"USD","date":"2026-10-14T10:30:00-04:00","impact":"Low","forecast":"","previous":"-1.2M"}
]"""

SAMPLE_XML = """<?xml version="1.0" encoding="windows-1252"?>
<weeklyevents>
<event><title>Non-Farm Employment Change</title><country>USD</country><date><![CDATA[11-06-2026]]></date><time><![CDATA[1:30pm]]></time><impact><![CDATA[High]]></impact><forecast><![CDATA[120K]]></forecast><previous><![CDATA[95K]]></previous><url>x</url></event>
<event><title>Bank Holiday</title><country>USD</country><date><![CDATA[11-11-2026]]></date><time><![CDATA[All Day]]></time><impact><![CDATA[Holiday]]></impact><forecast></forecast><previous></previous></event>
<event><title>Treasury Sec Speaks</title><country>USD</country><date><![CDATA[11-10-2026]]></date><time><![CDATA[Tentative]]></time><impact><![CDATA[Medium]]></impact><forecast></forecast><previous></previous></event>
</weeklyevents>"""


def pine_strings(block: str) -> list:
    """Undo pine_literal() for every d.push('...') line of a rendered block."""
    out = []
    for m in re.finditer(r"d\.push\('((?:\\.|[^'\\])*)'\)", block):
        out.append(re.sub(r"\\(.)", r"\1", m.group(1)))
    return out


class TimeTests(unittest.TestCase):
    def test_new_york_dst_edges_2026(self):
        # DST starts 2026-03-08 02:00 and ends 2026-11-01 02:00 New York time.
        self.assertEqual(ff.ny_offset(datetime(2026, 3, 8, 1, 59)), timedelta(hours=-5))
        self.assertEqual(ff.ny_offset(datetime(2026, 3, 8, 3, 0)), timedelta(hours=-4))
        self.assertEqual(ff.ny_offset(datetime(2026, 10, 31, 23, 0)), timedelta(hours=-4))
        self.assertEqual(ff.ny_offset(datetime(2026, 11, 1, 1, 30)), timedelta(hours=-5))

    def test_to_ny_keeps_the_instant(self):
        for utc in (datetime(2026, 3, 8, 6, 59), datetime(2026, 3, 8, 7, 0), datetime(2026, 11, 1, 5, 59), datetime(2026, 11, 1, 6, 0), datetime(2026, 7, 3, 12, 30)):
            utc = utc.replace(tzinfo=timezone.utc)
            ny = ff.to_ny(utc)
            self.assertEqual(ny, utc)
        self.assertEqual(ff.to_ny(datetime(2026, 7, 3, 12, 30, tzinfo=timezone.utc)).isoformat(), "2026-07-03T08:30:00-04:00")
        self.assertEqual(ff.to_ny(datetime(2026, 12, 4, 13, 30, tzinfo=timezone.utc)).isoformat(), "2026-12-04T08:30:00-05:00")

    def test_parse_iso_variants(self):
        want = datetime(2026, 10, 14, 12, 30, tzinfo=timezone.utc)
        for s in ("2026-10-14T08:30:00-04:00", "2026-10-14T12:30:00Z", "2026-10-14T12:30:00+0000", "2026-10-14T08:30:00"):
            self.assertEqual(ff.parse_iso(s), want, s)


class ParseTests(unittest.TestCase):
    def test_json_feed(self):
        events = ff.parse_feed(SAMPLE_JSON.encode())
        self.assertEqual(len(events), 6)
        holiday = events[0]
        self.assertTrue(holiday.all_day)
        self.assertEqual(holiday.impact, "Holiday")
        cpi = events[2]
        self.assertEqual(cpi.title, "Core CPI m/m")
        self.assertEqual(cpi.when.isoformat(), "2026-10-14T08:30:00-04:00")
        self.assertFalse(cpi.all_day)

    def test_xml_feed_times_are_gmt(self):
        events = ff.parse_feed(SAMPLE_XML.encode("utf-8"))
        nfp, holiday, tentative = events
        self.assertEqual(nfp.when.astimezone(timezone.utc), datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc))
        self.assertEqual(nfp.when.isoformat(), "2026-11-06T08:30:00-05:00")
        self.assertTrue(holiday.all_day)
        self.assertTrue(tentative.all_day)
        self.assertEqual(tentative.when.isoformat(), "2026-11-10T00:00:00-05:00")

    def test_html_block_page_is_an_error(self):
        with self.assertRaises(ff.FeedError):
            ff.parse_feed(b"<!DOCTYPE html><html><body>Request Denied</body></html>")


class MergeTests(unittest.TestCase):
    def ev(self, iso, title="X"):
        return ff.Event(title, "USD", ff.parse_iso(iso), "High")

    def test_fresh_week_replaces_its_dates_only(self):
        old = [self.ev("2026-10-07T08:30:00-04:00", "last week"), self.ev("2026-10-14T08:30:00-04:00", "cancelled")]
        fresh = [self.ev("2026-10-13T10:00:00-04:00", "new"), self.ev("2026-10-16T10:00:00-04:00", "new2")]
        titles = [e.title for e in ff.merge(old, fresh)]
        self.assertEqual(titles, ["last week", "new", "new2"])

    def test_merge_without_fresh_keeps_stored(self):
        old = [self.ev("2026-10-07T08:30:00-04:00")]
        self.assertEqual(ff.merge(old, []), old)


class RenderTests(unittest.TestCase):
    def test_block_round_trips_through_pine_literals(self):
        events = ff.parse_feed(SAMPLE_JSON.encode())
        block = ff.render_block(events, ["USD"], "10-12 09:00")
        self.assertTrue(block.startswith(ff.BEGIN_MARK))
        self.assertTrue(block.endswith(ff.END_MARK))
        self.assertIn("const string FF_UPDATED = '10-12 09:00'", block)
        decoded = [json.loads(s) for s in pine_strings(block)]
        self.assertEqual([d["title"] for d in decoded], [e.title for e in events])
        self.assertIn("Fed's Beige Book", [d["title"] for d in decoded])
        self.assertTrue(decoded[0]["allDay"])
        self.assertNotIn("allDay", decoded[2])

    def test_update_pine_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pine = Path(tmp) / "x.pine"
            shutil.copy(HERE / "USD_News_ForexFactory.pine", pine)
            block = ff.render_block(ff.parse_feed(SAMPLE_JSON.encode()), ["USD"], "now")
            ff.update_pine(pine, block)
            once = pine.read_text(encoding="utf-8")
            ff.update_pine(pine, block)
            self.assertEqual(once, pine.read_text(encoding="utf-8"))
            self.assertEqual(once.count(ff.BEGIN_MARK), 1)
            self.assertIn("indicator(", once)
            self.assertIn("type NewsEvent", once)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = ff.main(list(args))
        return code, buf.getvalue()

    def test_file_to_pine(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "week.json"
            feed.write_text(SAMPLE_JSON, encoding="utf-8")
            pine = Path(tmp) / "x.pine"
            shutil.copy(HERE / "USD_News_ForexFactory.pine", pine)
            code, out = self.run_cli("--file", str(feed), "--pine", str(pine), "--no-cache", "--now", "2026-10-12T01:00:00Z", "--tz", "UTC")
            self.assertEqual(code, 0)
            self.assertIn("Core CPI m/m", out)
            decoded = [json.loads(s) for s in pine_strings(pine.read_text(encoding="utf-8"))]
            self.assertEqual({d["country"] for d in decoded}, {"USD"})
            self.assertEqual(len(decoded), 5)
            self.assertIn("FF_UPDATED = '10-12 01:00'", pine.read_text(encoding="utf-8"))

    def test_print_json_filters_currency_and_impact(self):
        with tempfile.TemporaryDirectory() as tmp:
            feed = Path(tmp) / "week.json"
            feed.write_text(SAMPLE_JSON, encoding="utf-8")
            code, out = self.run_cli("--file", str(feed), "--no-cache", "--print-json", "--impact", "High", "--now", "2026-10-12T01:00:00Z")
            self.assertEqual(code, 0)
            titles = [d["title"] for d in json.loads(out)]
            self.assertEqual(titles, ["Core CPI m/m", "CPI y/y"])

    def test_cache_accumulates_weeks_and_prunes_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            week1 = Path(tmp) / "w1.json"
            week1.write_text(json.dumps([{"title": "Retail Sales m/m", "country": "USD", "date": "2026-10-06T08:30:00-04:00", "impact": "High", "forecast": "", "previous": ""}]), encoding="utf-8")
            week2 = Path(tmp) / "w2.json"
            week2.write_text(SAMPLE_JSON, encoding="utf-8")
            self.run_cli("--file", str(week1), "--cache", str(cache), "--no-pine", "--now", "2026-10-05T00:00:00Z")
            code, out = self.run_cli("--file", str(week2), "--cache", str(cache), "--print-json", "--now", "2026-10-12T00:00:00Z")
            self.assertEqual(code, 0)
            self.assertIn("Retail Sales m/m", [d["title"] for d in json.loads(out)])
            code, out = self.run_cli("--file", str(week2), "--cache", str(cache), "--print-json", "--keep-days", "3", "--now", "2026-10-12T00:00:00Z")
            self.assertNotIn("Retail Sales m/m", [d["title"] for d in json.loads(out)])

    def test_no_data_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = self.run_cli("--file", str(Path(tmp) / "missing.json"), "--no-cache", "--no-pine")
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()

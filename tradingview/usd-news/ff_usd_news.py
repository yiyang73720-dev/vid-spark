#!/usr/bin/env python3
"""Pull USD events from the Forex Factory calendar feed into the TradingView indicator.

TradingView's Pine Script cannot download data, so this script fetches Forex
Factory's weekly export feed, keeps the currencies you trade (USD by default)
and rewrites the FF DATA block of USD_News_ForexFactory.pine.  Paste the
updated file into the Pine Editor and save; every chart using it updates.

    python3 ff_usd_news.py                    # this week -> update the .pine next to this script
    python3 ff_usd_news.py --next-week        # also try next week's feed
    python3 ff_usd_news.py --file week.json   # a feed saved from the browser (JSON or XML)
    python3 ff_usd_news.py --print-json       # compact JSON for the indicator's paste box
    python3 ff_usd_news.py --currency USD,CNY --tz Asia/Shanghai

Only the Python standard library is used (Python 3.7+).
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import re
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PINE = HERE / "USD_News_ForexFactory.pine"
DEFAULT_CACHE = HERE / ".ff_cache.json"

FEEDS = {
    "this": [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    ],
    "next": [
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
        "https://cdn-nfs.faireconomy.media/ff_calendar_nextweek.json",
    ],
}
USER_AGENT = "Mozilla/5.0 (ff_usd_news.py; TradingView USD news indicator)"
BEGIN_MARK = "// @@FF-DATA-BEGIN@@"
END_MARK = "// @@FF-DATA-END@@"
IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1}
IMPACT_ZH = {"High": "高", "Medium": "中", "Low": "低", "Holiday": "假日"}
IMPACT_ALIASES = {
    "high": "High", "h": "High", "高": "High",
    "medium": "Medium", "med": "Medium", "m": "Medium", "中": "Medium",
    "low": "Low", "l": "Low", "低": "Low",
    "holiday": "Holiday", "假日": "Holiday",
}
WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class FeedError(Exception):
    pass


def note(msg: str) -> None:
    """Progress and warnings go to stderr so --print-json output stays clean."""
    print(msg, file=sys.stderr)


# ── Time helpers ────────────────────────────────────────────────────────────
def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def ny_offset(local: datetime) -> timedelta:
    """UTC offset of New York for a naive New York wall-clock time (US rules since 2007)."""
    start = datetime.combine(_nth_sunday(local.year, 3, 2), datetime.min.time()) + timedelta(hours=2)
    end = datetime.combine(_nth_sunday(local.year, 11, 1), datetime.min.time()) + timedelta(hours=1)
    return timedelta(hours=-4) if start <= local < end else timedelta(hours=-5)


def to_ny(dt: datetime) -> datetime:
    """Aware datetime -> the same instant as New York wall-clock time with a fixed offset."""
    utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    # DST runs from 02:00 EST (07:00 UTC) on the 2nd Sunday of March to 02:00 EDT (06:00 UTC) on the 1st Sunday of November.
    start = datetime.combine(_nth_sunday(utc.year, 3, 2), datetime.min.time()) + timedelta(hours=7)
    end = datetime.combine(_nth_sunday(utc.year, 11, 1), datetime.min.time()) + timedelta(hours=6)
    off = timedelta(hours=-4) if start <= utc < end else timedelta(hours=-5)
    return (utc + off).replace(tzinfo=timezone(off))


def ny_local(naive: datetime) -> datetime:
    return naive.replace(tzinfo=timezone(ny_offset(naive)))


def parse_iso(text: str) -> datetime:
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)  # +0000 -> +00:00 (Python < 3.11)
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else ny_local(dt)


def display_tz(name: str | None):
    """tzinfo for printing; None means the computer's own time zone."""
    if not name:
        return None
    if name.upper() in ("UTC", "GMT"):
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+; Windows also needs `pip install tzdata`

        return ZoneInfo(name)
    except Exception:
        note(f"! 无法识别时区 {name}，改用本机时区")
        return None


# ── Events ──────────────────────────────────────────────────────────────────
class Event:
    __slots__ = ("title", "country", "when", "impact", "forecast", "previous", "actual", "all_day")

    def __init__(self, title, country, when, impact="", forecast="", previous="", actual="", all_day=False):
        self.title = title.strip()
        self.country = country.strip().upper()
        self.when = to_ny(when)
        self.impact = normalize_impact(impact)
        self.forecast = (forecast or "").strip()
        self.previous = (previous or "").strip()
        self.actual = (actual or "").strip()
        self.all_day = bool(all_day) or self.impact == "Holiday"

    @property
    def key(self):
        return (int(self.when.timestamp()), self.country, self.title)

    def to_json(self) -> dict:
        d = {
            "title": self.title,
            "country": self.country,
            "date": self.when.isoformat(),
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
        }
        if self.actual:
            d["actual"] = self.actual
        if self.all_day:
            d["allDay"] = True
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Event":
        return cls(
            str(d.get("title") or ""),
            str(d.get("country") or ""),
            parse_iso(str(d["date"])),
            str(d.get("impact") or ""),
            str(d.get("forecast") or ""),
            str(d.get("previous") or ""),
            str(d.get("actual") or ""),
            bool(d.get("allDay") or d.get("tentative")),
        )


def normalize_impact(text: str) -> str:
    t = (text or "").strip().lower()
    for name in ("High", "Medium", "Low", "Holiday"):
        if t == name.lower():
            return name
    return "Holiday" if t in ("non-economic", "none", "") else (text or "").strip()


# ── Feed parsing ────────────────────────────────────────────────────────────
def parse_feed(raw: bytes | str, xml_tz: timezone = timezone.utc) -> list:
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype") or head.startswith("<html"):
        raise FeedError("收到的是网页而不是数据（可能被 Forex Factory 限流或拦截）")
    if head.startswith("[") or head.startswith("{"):
        return parse_json_feed(json.loads(text))
    if head.startswith("<"):
        # Bytes let ElementTree honour the declared encoding (Forex Factory uses windows-1252).
        return parse_xml_feed(raw.lstrip() if isinstance(raw, bytes) else text.lstrip(), xml_tz)
    raise FeedError("无法识别的数据格式（既不是 JSON 也不是 XML）")


def parse_json_feed(data) -> list:
    if isinstance(data, dict):  # tolerate {"events": [...]} style wrappers
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out = []
    for d in data:
        if isinstance(d, dict) and d.get("title") and d.get("date"):
            try:
                ev = Event.from_json(d)
            except ValueError:
                continue
            # Forex Factory puts all-day and tentative items at New York midnight.
            if (ev.when.hour, ev.when.minute) == (0, 0) and "T00:00" in str(d["date"]):
                ev.all_day = True
            out.append(ev)
    return out


def parse_xml_feed(text, xml_tz=timezone.utc) -> list:
    """ff_calendar_thisweek.xml: dates are MM-DD-YYYY, times like 8:30am in GMT."""
    out = []
    for node in ET.fromstring(text).iter("event"):
        get = lambda tag: (node.findtext(tag) or "").strip()  # noqa: E731
        title, day, clock = get("title"), get("date"), get("time")
        if not title or not day:
            continue
        try:
            d = datetime.strptime(day, "%m-%d-%Y")
        except ValueError:
            continue
        m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([ap]m)", clock.lower())
        if m:
            hour = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
            when = d.replace(hour=hour, minute=int(m.group(2)), tzinfo=xml_tz)
            all_day = False
        else:  # "All Day", "Tentative", "Day 1", ""
            when, all_day = ny_local(d), True
        out.append(Event(title, get("country"), when, get("impact"), get("forecast"), get("previous"), "", all_day))
    return out


# ── Fetching & cache ────────────────────────────────────────────────────────
def load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(path: Path, cache: dict) -> None:
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        note(f"! 缓存写入失败：{e}")


def fetch(url: str, timeout: float = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/xml,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        hint = "（请求太频繁，Forex Factory 限流了，过几分钟再试）" if e.code in (403, 429) else ""
        raise FeedError(f"HTTP {e.code}{hint}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise FeedError(f"网络错误：{getattr(e, 'reason', None) or type(e).__name__}") from e


def fetch_week(which: str, cache: dict, now: datetime, min_interval: int, force: bool) -> list:
    """Events of one feed week; reuses a copy fetched in the last `min_interval` seconds."""
    feeds = cache.setdefault("feeds", {})
    saved = feeds.get(which)
    if saved and not force:
        age = (now - parse_iso(saved["fetched"])).total_seconds()
        if 0 <= age < min_interval:
            note(f"· {which} week: 使用 {int(age)} 秒前下载的数据（避免触发限流，--force 可强制重新下载）")
            return parse_feed(saved["body"])
    errors = []
    for url in FEEDS[which]:
        try:
            body = fetch(url)
            events = parse_feed(body)
        except (FeedError, ValueError, ET.ParseError) as e:
            errors.append(f"{url}: {e}")
            continue
        feeds[which] = {"fetched": now.isoformat(), "url": url, "body": body.decode("utf-8-sig", errors="replace")}
        note(f"· {which} week: {url} → {len(events)} 条（所有货币）")
        return events
    raise FeedError("; ".join(errors))


def merge(stored: list, fresh: list) -> list:
    """Fresh feed data replaces everything stored for the New York dates it covers."""
    if not fresh:
        return stored
    days = {e.when.date() for e in fresh}
    lo, hi = min(days), max(days)
    kept = [e for e in stored if not (lo <= e.when.date() <= hi)]
    merged = {e.key: e for e in kept}
    for e in fresh:
        merged[e.key] = e
    return sorted(merged.values(), key=lambda e: e.when)  # stable: keeps the feed's order within a release time


# ── Output ──────────────────────────────────────────────────────────────────
def pine_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\r", " ").replace("\n", " ") + "'"


def render_block(events: list, currencies: list, updated: str) -> str:
    """The FF DATA block: FF_UPDATED, FF_CURRENCIES and one d.push('<json>') per event."""
    lines = [
        f"{BEGIN_MARK} — written by ff_usd_news.py; everything up to FF-DATA-END is replaced on each run.",
    ]
    if events:
        first, last = events[0].when.date(), events[-1].when.date()
        lines.append(f"// {len(events)} events · {','.join(currencies)} · {first} → {last} (New York dates)")
    lines += [
        f"const string FF_UPDATED = {pine_literal(updated)}",
        f"const string FF_CURRENCIES = {pine_literal(','.join(currencies))}",
        "ffEmbeddedData() =>",
        "    array<string> d = array.new<string>()",
    ]
    for e in events:
        lines.append("    d.push(" + pine_literal(json.dumps(e.to_json(), ensure_ascii=False, separators=(",", ":"))) + ")")
    lines += ["    d", END_MARK]
    return "\n".join(lines)


def update_pine(path: Path, block: str) -> None:
    target = path.resolve()  # follow a symlink instead of replacing it
    raw = target.read_bytes()
    try:
        src = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise FeedError(f"{path.name} 不是 UTF-8 编码，请用 UTF-8 重新保存后再运行") from e
    crlf = b"\r\n" in raw
    src = src.replace("\r\n", "\n")
    start = src.find(BEGIN_MARK)
    end = src.find(END_MARK)
    if start < 0 or end < start:
        raise FeedError(f"{path.name} 里找不到 {BEGIN_MARK} / {END_MARK} 标记")
    new = src[:start] + block + src[end + len(END_MARK):]
    if crlf:
        new = new.replace("\n", "\r\n")
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".ff_usd_news-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        with contextlib.suppress(OSError):
            os.chmod(tmp, stat.S_IMODE(os.stat(target).st_mode))
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def print_summary(events: list, tz, now: datetime) -> None:
    ny_today = to_ny(now).date()
    upcoming = [e for e in events if e.when >= now - timedelta(hours=1) or (e.all_day and e.when.date() >= ny_today)]
    if not upcoming:
        print("  （没有未来的事件）")
    current = None
    for e in upcoming:
        local = e.when if e.all_day else e.when.astimezone(tz)  # all-day items belong to their New York date
        day = f"{local:%m-%d} {WEEKDAY_ZH[local.weekday()]}"
        if day != current:
            print(f"\n  {day}")
            current = day
        clock = " 全天" if e.all_day else f"{local:%H:%M}"
        extra = "  ".join(x for x in (f"预测 {e.forecast}" if e.forecast else "", f"前值 {e.previous}" if e.previous else "") if x)
        print(f"    {clock}  [{IMPACT_ZH.get(e.impact, e.impact)}] {e.country} {e.title}  {extra}".rstrip())


# ── CLI ─────────────────────────────────────────────────────────────────────
def impact_arg(text: str) -> set:
    out = set()
    for tok in text.split(","):
        if tok.strip():
            name = IMPACT_ALIASES.get(tok.strip().lower())
            if not name:
                raise argparse.ArgumentTypeError(f"不认识的影响级别：{tok.strip()}（可用 High/Medium/Low/Holiday、H/M/L 或 高/中/低/假日）")
            out.add(name)
    return out


def xml_tz_arg(text: str):
    t = text.strip()
    if t.upper() in ("UTC", "GMT", "Z"):
        return timezone.utc
    m = re.fullmatch(r"(?:UTC|GMT)?\s*([+-]?)(\d{1,2})(?::?(\d{2}))?", t, re.I)
    if m and int(m.group(2)) <= 14 and int(m.group(3) or 0) <= 59:
        sign = -1 if m.group(1) == "-" else 1
        return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)))
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(t)
    except Exception:
        raise argparse.ArgumentTypeError(f"无法识别的时区：{text}（例如 UTC、+8、--xml-tz=-05:00 或 America/New_York）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Forex Factory USD 新闻 → TradingView 指标数据")
    ap.add_argument("--currency", default="USD", help="逗号分隔的货币，默认 USD；ALL = 全部")
    ap.add_argument("--next-week", action="store_true", help="同时尝试下载下周的数据（Forex Factory 不一定提供）")
    ap.add_argument("--file", action="append", default=[], help="使用本地保存的 JSON/XML 文件（可多次指定），不联网")
    ap.add_argument("--xml-tz", default=timezone.utc, type=xml_tz_arg, help="XML 文件里时间的时区，默认 UTC（Forex Factory XML 用 GMT）；负偏移写成 --xml-tz=-05:00")
    ap.add_argument("--pine", default=str(DEFAULT_PINE), help="要更新的 .pine 文件（默认同目录的 USD_News_ForexFactory.pine）")
    ap.add_argument("--no-pine", action="store_true", help="不修改 .pine 文件")
    ap.add_argument("--print-json", action="store_true", help="输出精简 JSON，可直接粘贴到指标的「Forex Factory JSON」输入框")
    ap.add_argument("--keep-days", type=int, default=14, help="保留多少天以前的历史事件（默认 14）")
    ap.add_argument("--impact", default=set(IMPACT_ZH), type=impact_arg, help="写入哪些影响级别，逗号分隔：High/Medium/Low/Holiday（或 H/M/L、高/中/低/假日），默认全部")
    ap.add_argument("--tz", default=None, help="打印时间用的时区，例如 Asia/Shanghai；默认本机时区")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="缓存文件（累积多周数据、避免重复下载）")
    ap.add_argument("--no-cache", action="store_true", help="不读写缓存")
    ap.add_argument("--min-interval", type=int, default=300, help="两次联网下载的最短间隔秒数（默认 300，防止被限流）")
    ap.add_argument("--force", action="store_true", help="忽略 --min-interval，强制重新下载")
    ap.add_argument("--now", help=argparse.SUPPRESS)  # tests: pretend the current time
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    now = parse_iso(args.now) if args.now else datetime.now(timezone.utc)
    currencies = [c.strip().upper() for c in args.currency.split(",") if c.strip()]
    impacts = args.impact
    cache_path = Path(args.cache)
    cache = {} if args.no_cache else load_cache(cache_path)

    stored = []
    for d in cache.get("events", []):
        try:
            stored.append(Event.from_json(d))
        except (KeyError, ValueError):
            pass

    fresh_sets = []
    if args.file:
        for f in args.file:
            try:
                fresh_sets.append(parse_feed(Path(f).read_bytes(), args.xml_tz))
                note(f"· {f} → {len(fresh_sets[-1])} 条（所有货币）")
            except (OSError, FeedError, ValueError, ET.ParseError) as e:
                note(f"! 读取 {f} 失败：{e}")
    else:
        for which in ["this"] + (["next"] if args.next_week else []):
            try:
                fresh_sets.append(fetch_week(which, cache, now, args.min_interval, args.force))
            except FeedError as e:
                level = "!" if which == "this" else "·"
                note(f"{level} {which} week 下载失败：{e}")

    if not fresh_sets and not stored:
        print(
            "\n没有拿到任何数据。可以在浏览器打开\n  " + FEEDS["this"][0]
            + "\n另存为 week.json，然后运行：python3 ff_usd_news.py --file week.json",
            file=sys.stderr,
        )
        return 1

    events = stored
    for fresh in fresh_sets:
        events = merge(events, fresh)
    events = [e for e in events if e.when >= now - timedelta(days=args.keep_days)]
    got_fresh = bool(fresh_sets)
    if got_fresh:
        cache["updated"] = now.isoformat()
    if not args.no_cache:
        cache["events"] = [e.to_json() for e in events]
        save_cache(cache_path, cache)
    # "Updated at" is when data was last downloaded, not when this run happened.
    stamp = now if got_fresh else (parse_iso(cache["updated"]) if cache.get("updated") else None)
    stale = 0 if got_fresh else 2
    if stale:
        note("! 这次没有拿到新数据，下面用的是缓存里的旧数据")

    wanted = [
        e for e in events
        if ("ALL" in currencies or e.country in currencies) and (e.impact in impacts or e.impact not in IMPACT_RANK and "Holiday" in impacts)
    ]
    tz = display_tz(args.tz)
    updated = stamp.astimezone(tz).strftime("%m-%d %H:%M") if stamp else ""

    if args.print_json:
        print(json.dumps([e.to_json() for e in wanted], ensure_ascii=False, separators=(",", ":")))
        return stale

    print(f"\n{','.join(currencies)} 事件 {len(wanted)} 条（时间为 {args.tz or '本机时区'}）：")
    print_summary(wanted, tz, now)

    if not args.no_pine:
        pine = Path(args.pine)
        try:
            update_pine(pine, render_block(wanted, currencies, updated))
        except (OSError, FeedError) as e:
            note(f"\n! 更新 {pine} 失败：{e}")
            return 1
        print(f"\n✓ 已写入 {pine}")
        print("  打开 TradingView → Pine 编辑器 → 粘贴整个文件 → 保存。图表上的指标会自动更新，")
        print("  但已有的警报不会：请删除本指标的旧警报并重新创建。")
        if currencies != ["USD"]:
            print(f"  指标设置里的「货币」要改成 {','.join(currencies)}，否则这些事件不会显示。")
    return stale


if __name__ == "__main__":
    sys.exit(main())

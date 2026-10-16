# PREREG: Larry Williams "open interest" tool (tool_open_interest)

Written 2026-10-16, BEFORE any backtest of this tool was run. No forward return of any series has
been computed or looked at for this tool. The inputs were inspected only for coverage: first and
last dates, gaps, the futures-only vs F+O basis, the distribution of weekly OI changes, and how many
setup weeks the candidate primary rule creates per market. No outcome was involved in any of
that. `results.json` stores the sha256 of this file (also in `PREREG.sha256`).

## 1. Larry's definition, with sources and how sure we are of each

Larry Williams reads **total open interest (OI) as a footprint of commercial hedging**. Commercial
hedgers do most of the short selling. OI can only rise if new short positions are opened and can
only fall if shorts are covered. So:

* **In a sideways / trading-range market, a LARGE INCREASE in OI is bearish.** Commercials are
  putting on more short hedges, and the breakout is expected to be down.
* **A LARGE DECREASE in OI is bullish.** Shorts are covering, the public is liquidating, and these
  declines mark bottoms.
* The secondary sources give a **"30%"** change as the size that counts.
* A short-term version by a Williams student says: an **extreme reduction in OI over 1-2 days**,
  backed by other bullish indicators, marks major bottoms; an extreme increase, backed by
  bearish indicators, marks tops.
* In his COT book (2005) he adds OI-normalised COT tools: "Commercials versus total open
  interest", later sold as **WILLCO** (commercial net position as a % of OI, as a 26-week index).

Sources. The sandbox cannot open any of these pages: every fetch fails with DNS `ENOTFOUND`. All
wording below comes from search-engine extracts, so **no verbatim page-level quote from Larry's
own text is claimed**.

1. L. Williams, *How I Made One Million Dollars Last Year Trading Commodities* (Windsor Books,
   1973; 3rd ed. 1979). This is the original OI rule. The book's own text was not retrieved. It
   is known through these summaries:
   * financial-spread-betting.com/course/open-interest.html: "if open interest increases
     greatly in a sideways market, the market is likely to move lower, as commercial interests
     are taking out more short contracts, as pointed out by Larry Williams in his book".
   * wisertrades.com, summary of the book: look for "a sharp rise or fall in open interest when
     prices are in a trading range"; this "indicates the direction the commercials expect the
     breakout to take". Commercial hedgers do most of the short selling. OI "can only decline if
     the short sellers are closing out their positions, and a rise is only possible if the
     amount of short selling is also on the rise".
   * solerinvestments.com/Online-Trading/Larry-Williams-Trading.htm (a Williams education page):
     "a 30% decrease in open interest can be a bullish indication and a 30% increase in open
     interest can be a bearish indication". The market must be in a trading range. OI is
     usually "an indication of Commercial short selling as they hedge their positions".
   * fxtrading.learncurrencytradingonline.com (Larry Williams trading strategy page): "If open
     interest increases, it is a sign of bearishness"; "A decrease in open interest indicates
     bullish stance is being taken by the commercial traders."
2. homeytel.medium.com, "OPEN INTEREST . . . as an indicator of bottoms and tops". The author
   calls Williams a mentor. Williams "looks for an extreme reduction in open interest in a 1–2 day
   period with support of other bullish indicators for major bottoms", and an extreme increase
   with bearish indicators for tops. This is a student's summary, not Larry's text.
3. L. Williams, *Trade Stocks and Commodities with the Insiders: Secrets of the COT Report*
   (Wiley, 2005). Per the table of contents (Wiley-VCH and O'Reilly listings): Ch. 8 "The
   Breakthrough: Getting Inside Volume and Open Interest", Ch. 9 "Opening Up on Open Interest",
   Ch. 10 "A Unified Theory of COT Data", which includes "A New Indicator—Commercials Versus Total
   Open Interest". A search extract of the book's reasoning reads: "when open interest is rising
   and commercials are the ones adding by taking short positions, a market peak should be close
   at hand, and vice versa". Not verified verbatim.
4. ireallytrade.com/newsletters/Larry-Williams-Indicators.pdf (Larry's indicator manual), per
   search extract. WILLCO "shows the total Commercial Net Position as a percentage of Open
   Interest". COT NetPositions are read "relative to total Combined Open Interest", i.e. the
   futures+options combined report. POIV combines price, OI and volume.
5. The classic price/OI quadrant is generic technical-analysis lore, not Larry-specific. See for
   example J. Murphy, *Technical Analysis of the Financial Markets* (1999), ch. 7, and Barchart
   "Tips on Technicals: Open Interest" ("when prices are rising, open interest should also be
   rising"). Its readings:
   * price up + OI up = strong;
   * price up + OI down = short covering, weak;
   * price down + OI up = new shorts, strong down;
   * price down + OI down = long liquidation, weak down / near a bottom.
6. Academic comparator: H. Hong and M. Yogo (2012), "What does futures market interest tell us
   about the macroeconomy and asset prices?", *Journal of Financial Economics* 105(3), 473-490
   (NBER w16712). Commodity OI growth, using the 12-month change, predicts commodity returns
   POSITIVELY. That is a trend/pro-cyclical reading, the opposite sign of Larry's contrarian
   reading.

Conflicts we know of (recorded, not resolved by looking at data):

* A third-party "Larry Williams playbook" page (forex.in.rs) says he wants OI to expand on
  breakouts and contract on pullbacks. That is a trend-confirmation reading.
* A student note (coconote.app) says an OI increase driven by commercials is bullish and one
  driven by small speculators is bearish. That needs the trader breakdown; WILLCO (variant I)
  is the testable piece of it.

**Operational reading used here (rule 1 above, the 1973 book rule, the one with the most
sources):** a large OI decline while price is range-bound is **long**; a large OI increase while
price is range-bound is **short**. "Large" cannot be Larry's literal 30%. In the F+O data,
crude total OI never falls 30% below its 26-week high, and is ≥30% below its 52-week high in only
1.8% of weeks, against 21% for gold. A fixed 30% would therefore mean very different things in
different markets. So "large" is defined against each market's own recent history, in Larry's
habit of 26-week/3-year windows. The literal 30% is kept as variant D. The headline idea behind
the tool is "OI declines = public/hedger liquidation near bottoms", so the **PRIMARY side is
LONG**.

## 2. The ONE primary rule

* **Market:** WTI crude oil.
* **OI input:** total open interest, F+O combined (`S/data/cot_weekly.csv`, column `oi`, CFTC
  disaggregated F+O, as in Larry's "total Combined Open Interest"), 2006-06-13 to 2026-09-29.
  One row per report.
* **Price input:** the close of the traded series as of the report date (Tuesday): the last
  close on or before it. It is NaN if that close is more than 4 calendar days old (series ended
  or a gap).
* **Statistics at report t:** W = 4 reports.
  * `dOI4 = OI_t / OI_{t-4} - 1` and `dP4 = P_t / P_{t-4} - 1`.
  * Both are undefined if report t-4 is more than 35 calendar days before report t (gap guard).
  * `pOI = pct156(dOI4)` and `pP = pct156(|dP4|)`, where `pct156(x)_t` is 100 × the share of the
    last ≤156 report values of x (3 years, Larry's long COT window), INCLUDING the current one,
    that are ≤ x_t. It needs at least 52 defined values (`fh.expanding_percentile(x, 52, 156)`)
    and uses only data known at t.
* **Setup:**
  * **long (+1)** if `pOI <= 10` (OI fell more than in ~90% of the last 3 years' 4-week windows)
    **and** `pP <= 50` (price moved less than its 3-year median 4-week move, i.e. a trading
    range);
  * **short (-1)** if `pOI >= 90` and `pP <= 50`;
  * 0 otherwise. Invalid (NaN) when pOI or pP is undefined.
* **Timing:** `cot` (v1 rule): decision_date = report_date; entry = first close on/after the
  Tuesday of the following week. Reports in the v1 government-shutdown windows are `excluded`.
* **Traded series:** `crude_roll`, the roll-adjusted front-month excess-return index (the
  realisable return, SPEC2 rule 2a). The price input P is the same series. The window is
  therefore about 2007-07 to 2024-01 (crude_roll ends 2024-04-05).
* **PRIMARY CELL: crude, LONG side, h = 8 weeks.**
* **Statistics** (shared harness, SPEC2 rules 3-4): n, episodes, mean signed return, baseline,
  excess, hit, Monte-Carlo circular-shift perm p (K=2000, seed 11) and the exhaustive
  +1-corrected perm p.

## 3. Pre-declared verdict criteria (applied to the primary rule only)

These are identical to the other battery tools:

* **positive EV, robust**: all of:
  * primary excess > 0 and perm_p ≤ 0.05;
  * excess > 0 in both split halves;
  * walk-forward (section 5) excess > 0;
  * S1 long-only after 0.05%/side costs has Sharpe > 0 and random-timing null p ≤ 0.10.
* **positive EV, fragile**: primary excess > 0 and perm_p ≤ 0.20, but at least one robustness
  condition fails.
* **negative (anti-signal)**: primary excess < 0 and perm_p ≥ 0.95.
* **no edge**: anything else.
* **untestable**: the primary cannot be computed.

## 4. Variant grid (robustness only; never promoted to "the result")

All variants run at h ∈ {1, 4, 8, 13}, on sides long, short and net (net = long and short setups
pooled, signed), and use COT timing with shutdown exclusion, unless stated otherwise.
Markets: crude (traded on `crude_roll`), gold, silver, soybean and sugar (traded on their v1
series). The price input is always the traded series itself.

**A. The primary rule elsewhere.**
* Crude on `crude_roll_lead5`, on `crude_cl` (v1 CL continuous, runs to 2026-10) and on
  `crude_spot`.
* Gold, silver, soybean and sugar on their v1 series.
* **Pooled across the 5 markets** (crude on crude_roll) with `fh.evaluate_pooled`, long and
  short, all h. This is the first secondary, as in v1.

**B. OI basis (crude).**
* (i) Futures-only OI from `CFTC_CL.json` (2010-01-05..2026-09-22; the 2026-09-29 row is
  mixed-basis and dropped), on crude_roll and on crude_cl.
* (ii) Spliced futures-only crude OI, 2000-01-04..2026-09-22. It takes the scout's legacy file
  (`scout/cot_smalltrader_weekly.csv`, deduplicated) before 2010-01-05 and CFTC_CL from
  2010-01-05. The two agree on all 550 common dates, which will be re-checked. 2007 is missing,
  so the gap guard makes changes across it undefined. Traded on crude_roll, crude_cl and
  crude_spot.
* (iii) Legacy futures-only OI (2000-2020) for gold, silver, soybean and sugar, on their v1
  series.

**C. Parameter sensitivity of the rule** (F+O, all 5 markets):
* OI window W ∈ {1, 2, 4, 13} reports;
* OI percentile cut p ∈ {5, 10, 20}: long pOI ≤ p, short pOI ≥ 100 − p;
* range filter on (`pct156(|dP_W|) ≤ 50`) or off.
* That gives 24 rules, named `oi{W}w_p{p}_{range|any}`. The primary is `oi4w_p10_range`.

**D. Larry's literal 30%** (F+O, 5 markets), each with the range filter (W = 4):
* `larry30_52`: long if OI_t ≤ 0.70 × max(OI over the last 52 reports incl. t); short if
  OI_t ≥ 1.30 × min(OI over the last 52).
* `larry30_26`: the same with 26 reports.

**E. Liquidation bottom / blow-off top** (Williams student's version; W ∈ {1, 4}):
* `liq_bottom_{W}w`: long if pOI ≤ 10 and dP_W < 0;
* short if pOI ≥ 90 and dP_W > 0.

**F. Price/OI quadrants** (W ∈ {4, 13}, F+O, 5 markets).
* The quadrants:
  * Q1 = dP_W > 0 & dOI_W > 0;
  * Q2 = dP_W > 0 & dOI_W < 0;
  * Q3 = dP_W < 0 & dOI_W > 0;
  * Q4 = dP_W < 0 & dOI_W < 0.
* `quad{q}_{W}w` is each quadrant as a long-only setup (side long: are the quadrant weeks
  better than average?) and as a short-only setup (side short).
* `quad_classic_{W}w`: signal = sign(dP_W) × sign(dOI_W), i.e. +1 in Q1/Q4 and −1 in Q2/Q3
  (textbook reading).
* `quad_trend_{W}w`: +1 in Q1, −1 in Q3, 0 otherwise.
* `quad_classic_4w` is also pooled across the 5 markets.

**G. OI vs its own range** (N ∈ {26, 52}):
* `idx = 100 (OI − min_N)/(max_N − min_N)` over the last N reports incl. t.
* `oi_range{N}`: long if idx ≤ 20 (OI near its low after liquidation), short if idx ≥ 80 (OI
  near its high, crowded/heavy hedging). This is Larry's contrarian reading. The
  trend reading is the opposite sign: its excess is the negative of this one.

**H. Hong & Yogo (2012) comparator:**
* `g = ln(OI_t / OI_{t-52})` in contracts, with the gap guard (t-52 within 372 days).
* `hy12m`: long if pct156(g) > 50 (OI growth above its trailing 3-year median, i.e. demeaned
  growth > 0), short otherwise.
* `hy12m_sign`: long if g > 0, short if g < 0.
* Both are also pooled across the 5 markets.
* HY's sign is pro-cyclical: high growth means long.

**I. WILLCO (Larry, *Insiders* ch. 10):**
* `w = comm_net / OI` (commercials = producer/merchant + swap dealers, as in v1);
* `willco26 = 100 (w − min26)/(max26 − min26)` over the last 26 reports incl. t;
* long ≥ 80, short ≤ 20. F+O data.

## 5. Out-of-sample (SPEC2 rule 5)

* **Split half:** `fh.split_half` of the primary frame on crude_roll at h = 8, at the midpoint
  date of its eligible weeks. Long is the criterion; short and net are reported.
* **Walk-forward:**
  * WF-0: the primary has no parameter fitted to outcomes. Its only data-estimated statistics
    (the percentile thresholds) use the trailing 156 reports up to and including t, all known
    at t. So the primary is walk-forward by construction, and this is stated, not re-run.
  * **WF-1 (reported as THE walk-forward result):** each calendar year from 2010-01-01, the
    long-side rule is chosen among the 24 rules of grid C (crude, F+O, crude_roll). It is the
    one with the highest past excess at h = 8. The excess is computed only on weeks whose
    8-week exit date is before the cutoff, against the baseline of those same weeks, and a
    rule needs ≥ 15 such past signal weeks. If no rule qualifies, there is no signal. The
    chosen rule is applied to the next year (`fh.walk_forward`, refit "YS"). It is evaluated on
    crude_roll, long, all h.
  * WF-2: WF-1's selection with the primary forced as the only candidate. This is a sanity
    check and must equal the primary over the test years.

## 6. Strategy (SPEC2 rule 6)

* `fh.simulate(primary frame, "crude_roll", timing="cot")` with the defaults:
  * S1: enter on the first week of each episode and hold 40 trading days;
  * S2: wait ≤ 20 days for the close to cross its 20-day SMA;
  * each in long+short and long-only form;
  * costs 0.05% per side, compared with buy & hold over the same window;
  * random-timing null with K = 2000.
* Also simulated on crude_cl (to 2026), as robustness.

## 7. Multiple testing and reporting

* The headline is the primary row, whatever the grid shows.
* For the h = 8 long and short rows of the grid, report the number of cells, the number with
  perm_p ≤ 0.05 (and how many 5% of cells would give by chance) and Benjamini-Hochberg q-values.
* The grid cells are strongly dependent (overlapping rules, markets and horizons), so these
  numbers are descriptive.
* No naive t-statistics are reported.

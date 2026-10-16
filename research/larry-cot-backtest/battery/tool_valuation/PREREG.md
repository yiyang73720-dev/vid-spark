# PREREG: Larry Williams "Valuation Index" (WillVal), tool_valuation

Written 2026-10-16T08:55Z, BEFORE any backtest of this tool was run. Nothing below was chosen after
looking at a forward return. Inputs were inspected only for coverage (first and last dates, series
lengths), never against outcomes. `results.json` stores the sha256 of this file.

## 1. Larry's definition (with sources, and how sure we are of each)

Larry Williams' **Valuation Index** ("WillVal", 1990) asks whether a market is cheap or expensive
**relative to another market**, not relative to its own past. He compares a market with **gold**
(his base "money"), **Treasury bonds** and the **US dollar**. The result is a 0-100 oscillator: a low
reading means **undervalued** (bullish), a high reading means **overvalued** (bearish). He presents it
as a **filter, not a timing tool**: an undervalued market is a candidate for longs, and a separate
trigger times the entry.

Sources. This sandbox cannot open most web pages: DNS fails for tradingview.com, mql5.com,
prorealcode.com, stockcharts.com and esignal.com. Every source except (5) is known only from
search-engine extracts. Source (5) was read in full from its GitHub backup.

1. **ProRealCode forum, "Will Val Indicator" (2017)**, https://www.prorealcode.com/topic/wiil-val-indicator/ .
   Extract: "Divide the price of any commodity by the price of Gold ... and multiply this by 100 to
   normalize. ... construct either a 2 week or 2 period exponential average of this spread. Also
   construct a 22-week average or 22-period exponential of the spread. Subtract the 2 period averages
   from the 22 period averages this produce the Pal ... this week's Pal - lowest pal of the last 3 years
   divided by highest pal of the last 3 years - lowest pal of the last 3 years, times 100". The code
   runs on a weekly spread chart, uses ProBuilder `average[2,1]` and `average[22,1]` (EMAs), and adds
   that "it does not make much difference which mathematical formula you use".
2. **TradingView, "Larry Williams Valuation Index [tradeviZion]"**,
   https://www.tradingview.com/script/dKwos20x-Larry-Williams-Valuation-Index-tradeviZion/ (also
   /script/QiwEE23w/). The extracts say:
   * it is an "interpretation of Larry Williams' famous WillVal (Valuation) Index, originally
     developed in 1990";
   * ratio = close / close of a correlated security × 100;
   * valuation = "difference between the short-term EMA and the long-term EMA" of that ratio,
     normalised to 0-100 over "Valuation Length (Default: 156)" ("approximately three years ... as
     recommended by Larry Williams"; 156 bars are 3 years of **weekly** bars);
   * comparators: dollar (DXY), gold (GC1!), Treasury bonds (ZB1!);
   * thresholds 85 (overvalued) and 15 (undervalued);
   * "Larry Williams originally used the weekly timeframe".
3. **MQL5 code base 22099** (WillVal indicator), https://www.mql5.com/en/code/22099 . The extract
   says: ratio of the instrument's close to a reference instrument's close; difference of two EMAs of
   the ratio; then 100·(value − min)/(max − min) over a lookback; user-set overbought and oversold
   levels.
4. **StockCharts ACP, "Larry Williams Stock Trading Starter Pack" plug-in (WillVal)**,
   https://help.stockcharts.com/charts-and-tools/stockchartsacp/stockchartsacp-plug-ins/larry-williams-stock-trading-starter-pack-plug-in .
   The extract says:
   * developed in 1990 by Larry Williams; compares a security with a benchmark (TLT, GLD or UUP);
   * default look-back 65 periods; works on daily or weekly charts;
   * overvalued line 75, undervalued line 15;
   * "when it crosses back above that lower line, that is a buy signal".
   No formula is given.
5. **fxcodebase forum, "Spread Oscillator" thread (2013)**, read in full from the GitHub backup
   `gehtsoft-usa/fxcodebase-backup`, file `backup/custom-indicators/s/spread-oscillator/index.md`
   (forum topic f=17&t=36328). It defines:
   * Ratio = (first / second instrument) × 100;
   * oscillator = difference of a long and a short EMA of the ratio (5 and 15 periods), with a
     request to normalise it "for 3 year ... like constructing of COT index".
   It quotes Larry's practice:
   * "Larry Williams suggest all instruments comparing to gold ... the currency will be against
     3 other stuff: gold, T-bond and USD index";
   * "Larry W said about the concept of valuation. It different from oversold/overbought ... when
     you see EUR over valuated to gold and USDindex (2 out of 3), preparing to short EUR".
6. Larry's own words, from search extracts only:
   * Stocks & Commodities interview, July 1997 (traders.com/documentation/FEEDbk_docs/1997/07/0797Williams.html):
     "if I were to consider buying bonds, I want the relationship between bonds and, let's say, gold,
     to be giving me the signal";
   * *Trade Stocks and Commodities with the Insiders* (Wiley 2005), gold chapter: a "crude method"
     for judging when gold is overvalued relative to the US dollar; "major rallies in this metal,
     though not all, begin with this valuation index at a low level";
   * ireallytrade.com "Investors Daily Edge" interview: "the dollar was overvalued vs Gold in my
     Valuation Index".
   No verbatim formula or page number from Larry's books could be retrieved.

**Where the sources disagree, and what is pre-registered.**

| item | sources | pre-registered primary | variants |
|---|---|---|---|
| input | ratio market/comparator × 100: (1), (2), (3), (5) | ratio × 100 (the ×100 cancels in the 0-100 scaling) | none |
| detrending | EMA(2) − EMA(22): (1); short − long EMA: (2), (3); EMA(5)/EMA(15): (5) | **EMA(2) − EMA(22)**, the only fully specified version | EMA(5) − EMA(15); `none` = 0-100 range of the raw ratio |
| normalisation | min/max ("stochastic") over the lookback: (1), (2), (3), (5) | 100·(Pal − min)/(max − min) | none |
| lookback | 3 years weekly = 156: (1), (2), (5); 65 periods: (4) | **156 weekly values**, including the current one | 52, 65, 104, 260 |
| bar size | weekly: (1), (2) | **weekly**, Friday closes | none |
| thresholds | 85/15: (2); 75/15: (4); Larry's own 0-100 COT index uses 80/20 | **20 / 80** | 25/75, 15/85, 15/75 |
| comparator | gold first: (1), (5), (6); gold, bonds, dollar: (2), (4), (5) | **gold** | 10y Treasury price proxy, broad dollar, 3-comparator composite |
| composite | "2 out of 3": (5); three benchmarks: (2) | 2-of-3 vote (secondary) | average of the three indices |

**Sign convention.** Source (1)'s wording "subtract the 2 period averages from the 22 period
averages" literally gives Pal = EMA22 − EMA2. Sources (2) and (3) give short − long. The only
reading consistent with every source's interpretation is short − long: low = undervalued, so the
index must FALL when the market cheapens against the comparator. The pre-registered formula is
therefore **Pal = EMA2(R) − EMA22(R)**. Reversing the sign maps the index to exactly 100 − VI, so the
other convention's "long" rule is this study's **short** rule. Both sides are reported, so both
conventions are covered.

## 2. Formula used (all variants share it)

For a market M and a comparator C, on the Friday grid `fh.weekly_decision_dates`:

* `M_t`, `C_t` = last close on or before Friday t (`asof`). If that close is more than 4 calendar days
  old, the week is undefined.
  * **Broad dollar (DTWEXBGS):** published weekly on Mondays, so it is read as of t − 7 days, again
    with a 4-day staleness limit relative to t − 7.
  * **10y bond proxy (DGS10):** a market yield observable at the close, so no lag.
* `R_t = 100 · M_t / C_t`; undefined weeks are dropped before smoothing.
* `Pal_t = EMA_s(R)_t − EMA_l(R)_t`.
  * EMA: alpha = 2/(n+1), recursive, seeded with the first R.
  * Pal is defined from the l-th R observation on.
  * Variant `none`: Pal = R.
* `VI_t = 100 · (Pal_t − min) / (max − min)`.
  * min and max are taken over the last N defined Pal values, including the current one.
  * VI is undefined until N values exist, and when max = min.
* Setups:
  * **long (undervalued)** when VI ≤ buy_th (20);
  * **short (overvalued)** when VI ≥ sell_th (80);
  * otherwise none.
  * "Episode" = maximal run of the same setup (harness definition).

## 3. The ONE primary rule

* **Market / comparator:** WTI crude oil **vs gold**.
  * Crude valuation input: `crude_cl`, the NYMEX CL front month (v1 series; this is what Larry
    charts).
  * Gold input: `gold`, the v1 London spot series.
* **Parameters:** EMA(2) − EMA(22), N = 156 weeks, thresholds 20/80, weekly Friday closes.
* **Side:** **LONG** (VI ≤ 20, "crude undervalued vs gold"). Larry presents valuation mainly as a
  "buy when undervalued" filter (sources 4 and 6). The short and net sides are secondary.
* **Traded series:** `crude_roll`, the roll-adjusted front-month excess-return index (SPEC2 rule
  2a). Timing `price`: entry at the first close strictly after the Friday.
* **Horizon:** **h = 8 weeks**.
* **Expected sample:**
  * gold and crude_cl start in 2000-08, so the first VI is about 2004-01;
  * crude_roll ends 2024-04-05.
* **Statistics** (shared harness): n, episodes, mean, baseline, excess, hit, Monte-Carlo
  circular-shift perm_p (K=2000, seed 11), exhaustive perm_p_exact (+1 correction).

**Secondary headline (pre-registered, not the verdict):** the same rule pooled across crude, silver,
soybean and sugar vs gold, long side, h = 8 (`fh.evaluate_pooled`). Each market is traded on its own
series: crude on `crude_roll`, the others on their v1 series.

## 4. Pre-declared verdict criteria (applied to the primary rule only)

These are identical to the other tools of the battery.

* **positive EV, robust:** all of the following hold:
  * primary excess > 0 and perm_p ≤ 0.05;
  * excess > 0 in both split halves;
  * walk-forward (section 6) excess > 0;
  * S1 long-only, after 0.05%/side costs, has Sharpe > 0 and random-timing null p ≤ 0.10.
* **positive EV, fragile:** primary excess > 0 and perm_p ≤ 0.20, but at least one robustness
  condition fails.
* **negative (anti-signal):** primary excess < 0 and perm_p ≥ 0.95.
* **no edge:** anything else.
* **untestable:** the primary cannot be computed.

## 5. Variant grid (robustness only; never promoted to "the result")

**A. Pair grid at the primary parameters** (EMA 2/22, N=156, 20/80). Horizons h ∈ {1, 4, 8, 13};
sides long, short and net.

* Markets crude, silver, soybean and sugar, each vs:
  * gold;
  * `ust10_price`, the 10y Treasury price proxy;
  * `usd_broad`, the broad dollar, lagged as above.
* Gold vs `ust10_price` and gold vs `usd_broad`.
* 3-comparator composite for each of the four markets:
  * `vote2of3`: long if at least 2 of the 3 VIs ≤ 20; short if at least 2 are ≥ 80; valid only
    when all three are defined;
  * `avg3`: mean of the three VIs, then the same 20/80 thresholds.
* Gold composite: long if both its VIs ≤ 20, short if both ≥ 80.
* Traded series:
  * crude: `crude_roll`;
  * silver, soybean, sugar and gold: their v1 series.

**B. Parameter grid**, every pair and both composites, h = 8 only, sides long, short and net:

* N ∈ {52, 65, 104, 156, 260};
* detrending ∈ {EMA 2/22, EMA 5/15, none};
* thresholds (buy/sell) ∈ {20/80, 25/75, 15/85, 15/75}.

**C. The primary rule on other crude series**, all h and sides:

* traded on `crude_roll_lead5`, `crude_cl` (signals 2004..2026) and `crude_spot` (2004..2026);
* valuation input `crude_spot` instead of `crude_cl`, traded on `crude_roll`.

**D. Entry-on-cross variant (StockCharts' "buy when it crosses back above the lower line"),** for
crude vs gold at primary parameters: long in the week VI crosses from ≤ 20 to > 20; short in the week
it crosses from ≥ 80 to < 80.

**E. Extended history (crude only, 1985/1989..2024), traded on `crude_roll`:**

* crude input = EIA contract-1 prints (RCLC1);
* gold input = the pysystemtrade GOLD held-contract price from the scout's `curve_daily.csv`
  (1975..2024; unadjusted, so small roll steps; the hourly era after 2013 is not always the
  settlement);
* bond input = `ust10_price`;
* crude vs gold and crude vs bond, primary parameters, all h and sides, plus a split-half of
  crude vs gold.

**F. Pooled cells** (`fh.evaluate_pooled`):

* the four markets vs gold, vs bond, vs dollar and `vote2of3`;
* long and short, h ∈ {1, 4, 8, 13}.

## 6. Out-of-sample (SPEC2 rule 5)

* **Split half:** `fh.split_half` of the primary frame on `crude_roll`, h = 8. Long is the criterion;
  short and net are context.
* **Walk-forward.**
  * The primary rule has no parameter estimated from data. Its EMAs and min/max use only past
    values, so its walk-forward signal is identical to the static one. This is reported, evaluated
    with `fh.walk_forward` (identity fit) from 2008-01-01.
  * The pre-registered walk-forward test is a **parameter-selection** rule for crude vs gold.
    * Each 1 January from 2008, the fit chooses among the 60 configurations of grid B (N ×
      detrending × thresholds). It picks the one with the highest long-side h=8 excess on
      `crude_roll`.
    * The fit uses only decision weeks whose 8-week exit date is before the cutoff, and needs at
      least 26 signal weeks.
    * If no configuration qualifies, or the best excess is ≤ 0, there is no trade that year.
    * The chosen configuration's long setups are traded in that year.
    * Evaluated at h = 8 (long) on `crude_roll` and compared with the static primary over the same
      2008+ weeks.
  * The verdict's "walk-forward excess > 0" condition uses this selection rule.

## 7. Strategy (SPEC2 rule 6)

* `fh.simulate(primary frame, "crude_roll", timing="price")` with defaults:
  * S1 and S2, each long+short and long-only;
  * hold 40 days, cost 0.05% per side;
  * null of 2000 circular shifts, seed 11;
  * vs buy & hold over the same window.
* Context:
  * the same on `crude_cl` (2004..2026);
  * crude `vote2of3` on `crude_roll`.

## 8. Multiple testing

* Every grid row is reported.
* Benjamini-Hochberg q-values across all h = 8 rows (on perm_p) are information only.
* The headline is the primary row, whatever the grid shows.

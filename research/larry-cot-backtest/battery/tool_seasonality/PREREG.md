# PRE-REGISTRATION: Larry Williams SEASONALITY (fundamentals battery, SPEC2)

Written on 2026-10-16, BEFORE any seasonal signal was computed or backtested. Before writing it, the
only data work done was quality checks of the input series: coverage, and the size of roll-day jumps
in the new roll-adjusted futures indices (`seas_data.validate_psys_index`). No seasonal tendency,
signal, forward-return statistic or strategy result had been computed.

## 1. Larry's tool, his own words, and sources

**What Larry says about seasonals.** These points come from WebSearch result summaries. The pages
themselves could not be opened, because WebFetch is DNS-blocked in this session.

- Seasonality was one of his first subjects. His first book was *Sure Thing Commodity Trading: How
  Seasonal Factors Influence Commodity Prices* (with Michelle Noseworthy, Windsor Books). In MoneyShow
  interviews he says he "wrote the first book ever on seasonality ... way back in 1973". Sources:
  futures.ireallytrade.com/aboutlarry/ ; williamspercentr.com/about-larry ;
  moneyshow.com/articles/videotranstr-27772.
- A seasonal is "a pattern averaged over the twelve months" of what usually happens in a year. It
  shows where a market tends to bottom and where it tends to rally.
- He checks "congruency": does this year's price action agree with the seasonal shape?
- Seasonals change over time. His example is "sell in May", which worked better after 1950.
- Seasonals are *suggestive, not mandatory*. "A seasonal confirms a trade ... I won't take a trade
  just because of a seasonal pattern."
- He likes commercials (COT) to be bullish at a seasonal low.
- Sources for the four points above: moneyshow.com/articles/videotranstr-28669 ;
  moneyshow.com/articles/videotranstr-27772 ;
  ninjatrader.com/futures/blogs/larry-williams-on-the-seasonality-of-the-futures-markets/.
- **Williams True Seasonal indicator** (StockCharts Larry Williams plug-in).
  - The seasonal value for a given year uses *only data prior to that year* ("if you wish to
    determine the seasonal data for 2010, only the data up to 2010 would be employed").
  - The default lookback is 1 year, "but that only gives you one data point for each day of the year,
    so Williams strongly recommends increasing that number".
  - Sources: help.stockcharts.com/charts-and-tools/stockchartsacp-plug-ins/larry-williams-stock-trading-starter-pack-plug-in ;
    articles.stockcharts.com/article/articles-chartwatchers-2023-09-optimizing-your-stock-selectio-527/.
- **Industry convention for seasonal "windows" (context only, not Larry).** Moore Research Center
  (MRCI) scans "the last 15 years ... recurrent, with a minimum reliability of 80%, during similar
  time windows". Source: mrci.com/web/help-pages/frequently-asked-questions/70-general/2204-from-our-editor-trade-selection-process.html.
  The task's 15-year / 70% rule follows this convention.

**What I could not verify.** I could not confirm the exact construction steps from Larry's books, and
no source I found gives a single fixed lookback. This test is therefore the task's mechanical,
walk-forward version of "trade the seasonal tendency":

- Each estimate uses ONLY prior years, as the True Seasonal does.
- The lookback is multi-year.
- A window is traded only when the past years agree strongly (a hit-rate threshold).
- It tests the tool **alone**. Larry himself says he would use it only as confirmation (with COT or
  other inputs).

## 2. Signal definition (walk-forward by construction)

**Decision dates and analog dates**

- Decision dates: every Friday d (fh.weekly_decision_dates, 1986-01-03 .. 2026-10-16). Timing is
  `price`: entry at the first close strictly after d, exit 5h trading days later on the evaluation
  series (fh.forward_table).
- For lookback L years and horizon h weeks, the analog of d in year k (k = 1..L) is the **Friday
  nearest (±3 days, unique) to the same calendar date k years earlier**. Feb 29 maps to Feb 28.
- This is the "week-of-year" bucket, aligned to the calendar date instead of an ISO week number, so
  there is no week-53 ambiguity.

**The seasonal tendency at d**

- R_k = the h-week forward return of the analog Friday on the ESTIMATION series, with the same entry
  and exit rule as the evaluation.
- hit = #{R_k > 0} / L and mean = mean(R_k).
- The estimate is undefined (NaN, so no signal) unless all L analog returns exist.

**Leak guard**

- Every analog's exit date must be **strictly before d**. Since h ≤ 13 weeks is less than 1 year, this
  always holds. The code asserts it rather than silently filtering.

**Signal**

- +1 (long setup) if hit ≥ th and mean > 0.
- −1 (short setup) if hit ≤ 1 − th and mean < 0.
- 0 otherwise.
- The comparisons use integer counts: 100·#pos ≥ th%·L, and 100·#pos ≤ (100 − th%)·L.

**Month-of-year version (variant)**

- The bucket is the calendar month M of d.
- For each of the L prior years, the year's value is the mean of R over all Fridays of month M in
  that year. It is NaN unless all of those Fridays have returns.
- hit and mean are then computed across the L yearly values. The same thresholds apply, and the same
  exit-before-d assertion.

**No-look-ahead checks (must pass, else results are void)**

- (a) **Independent re-implementation through `fh.walk_forward` with weekly refit.** Here `fit` sees
  only `forward_table` rows with `exit_date{h} < cutoff`. Its signal must equal the vectorised
  signal exactly, for the primary configuration of every market.
- (b) **Truncation test.** For 12 cut dates X, recompute the signals from estimation series
  truncated at X. Every signal with decision date ≤ X must be unchanged.

## 3. Markets and series (rule 2)

| market | estimation series (analog returns) | evaluation series (primary) | other evaluation / robustness series |
|---|---|---|---|
| crude | `crude_spot` (EIA Cushing spot, 1986) | `crude_spot` | `crude_cl` (CL continuous) and `crude_roll` (est = spot), as the task asks |
| heating oil | `ho_spot` (1986-06) | `ho_spot` | |
| gasoline | `gas_spot` (1986-06) | `gas_spot` | |
| gold | `gold` (v1 London spot, 2000-08) | `gold` | |
| silver | `silver` (v1, 2000-08) | `silver` | |
| soybean | `soybean` (v1 ZS continuous, 2000-09) | `soybean` | |
| sugar | `sugar` (v1 SB continuous, 2000-03) | `sugar` | |

**Lookback "at least 15 years where available, else 10".**

- The primary uses **L = 15** for crude, heating oil and gasoline, whose estimation series start in
  1986. First signals come about 2001.
- It uses **L = 10** for gold, silver, soybean and sugar, whose v1 series start in 2000. First
  signals come about 2010.

**Tradability confirmation (pre-registered, part of the verdict).**

- Why it is needed: unadjusted continuous futures (v1 soybean "103.ZS00Y", sugar "SB=F", crude
  "CL00Y") jump on roll days at almost fixed calendar dates. A seasonal estimator can learn those
  jumps and then "confirm" them on the same series.
- Spot energy series also carry calendar effects that a futures trader cannot capture. The NY Harbor
  gasoline spot assessment follows the seasonal volatility-grade specification. Spot seasonality is
  largely priced into the futures curve.
- The rule is therefore re-run with **estimation and evaluation on roll-adjusted futures
  excess-return indices**:
  - crude: `crude_roll` (EIA RCLC1/RCLC2, 1985-2024-04).
  - heating oil, gasoline, gold, silver, soybean, sugar: contract-consistent indices built in
    `seas_data.py` from the pysystemtrade multiple-prices files (1970-1985 .. 2024-03-28).
  - Settings: L = 15 for all 7 markets, th = 70, h = 8, week bucket, pooled net. It is reported over
    each market's full window and over 2001-01-01..2024-03-28.

**Extended-history variant** (gold, silver, soybean, sugar only):

- L = 15, from 2000 onward.
- Each analog return comes from the v1 series when the analog's entry and exit lie inside it, and
  otherwise from the roll-adjusted pysystemtrade index.
- Evaluation is on the v1 series.

## 4. THE ONE PRIMARY RULE

> **Seasonal walk-forward, all 7 markets pooled, NET side (long and short setups together, signed by
> setup), h = 8 weeks, th = 70% (long: hit ≥ 70% and mean > 0; short: hit ≤ 30% and mean < 0),
> L = 15 (crude, HO, gasoline) / 10 (gold, silver, soybean, sugar), week-of-year bucket, estimation
> series = evaluation series as in the table (crude on `crude_spot`).**

**Headline statistics**

- Pooled n, episodes, mean signed 8-week return, and excess. Excess is the mean of (signed r − setup
  sign × that market's own baseline).
- Hit rate.
- Pooled circular-shift perm p. Each market's ±1 setup vector is shifted over its own eligible weeks.
  One `default_rng(11)` per pooled cell, and each market draws K = 2000 offsets on [26, T−26] in the
  table's order. This follows `fh.evaluate_pooled`, extended to the net side.
- Per market: `fh.evaluate` rows (long, short, net; h = 1, 4, 8, 13), with perm_p (MC) and
  perm_p_exact.

**Pre-registered supplementary diagnostic: the off-calendar permutation p.** This is the exhaustive
shift test with all offsets within ±3 weeks of a multiple of 52.18 weeks removed. A shift by about
one year keeps a seasonal signal aligned with the calendar, so the standard test is conservative for
a seasonal tool. This p is reported next to perm_p and does not replace it.

## 5. Out-of-sample (rule 5)

- **Walk-forward.** The primary is itself walk-forward: every tendency is estimated from prior years
  only. Its per-calendar-year pooled net excess is reported.
- **Split-half.**
  - Per market: `fh.split_half` (sides long, short, net; h = 8), split at the market's own midpoint.
  - Pooled: one common split date, the midpoint of the union of the eligible decision dates of the 7
    markets. Each half is evaluated with its own baselines and permutations.

## 6. Strategy (rule 6)

- For each market: `fh.simulate` on its evaluation series with the primary signal. That gives S1 and
  S2, each long+short and long-only, with hold 40, win 20, cost 0.05% per side, null K = 2000 and
  seed 11.
- **Equal-weight portfolio of the 7 markets**, rebalanced daily:
  - Each day, the mean of the daily net returns of the markets whose window has started. A market
    with no close on a date contributes 0 that day and the multi-day return on its next close.
  - Compared with the equal-weight buy & hold portfolio of the same markets and windows.
  - Its null shifts every market's signal with the same offsets `fh.simulate` draws. The portfolio
    Sharpe of each draw is compared with the real one.

## 7. Variant grid (robustness only; never promoted to "the result")

- **Main grid:** th ∈ {60, 65, 70, 75} × L ∈ {10, 15, 20} × h ∈ {4, 8, 13} (analog horizon = evaluated
  horizon) × bucket ∈ {week, month}. That is 72 cells per market, sides long/short/net, on:
  - each market's primary evaluation series;
  - crude estimated on spot and evaluated on `crude_cl` and on `crude_roll`;
  - the tradable roll-adjusted series (estimation = evaluation).
- Pooled net is reported for every grid cell on the primary series and on the tradable series.
- **Crude extras:** estimation = evaluation = `crude_roll`, and `crude_roll_lead5`.
- **Extended-history L = 15** for the four 2000-start markets.
- **Multiple testing:**
  - Holm adjustment across the 7 per-market primary net p-values.
  - For the grid: the share of cells with positive excess, and the share with perm_p ≤ 0.05.
    Under no edge, about 5% of cells would reach that by chance, and the cells are highly
    correlated.

## 8. Verdict criteria (fixed now)

**"positive EV, robust"** requires ALL of:

1. Primary pooled net excess > 0 with perm_p ≤ 0.05.
2. Pooled excess > 0 in both split halves.
3. Tradability confirmation (full window) pooled net excess > 0 with perm_p ≤ 0.10.
4. The equal-weight S1 long+short portfolio has Sharpe > 0 after costs and null p ≤ 0.10.

**"positive EV, fragile"**: primary pooled net excess > 0 and perm_p ≤ 0.10, but at least one of 2-4
fails.

**"no edge"**: primary perm_p > 0.10, unless the anti-signal condition below holds.

**"negative (anti-signal)"**: primary excess < 0 and the lower-tail pooled shift p ≤ 0.05. That p
is (1 + #{shifted mean ≤ observed}) / (1 + K), computed on the same Monte-Carlo offsets as the
pooled perm_p. The pooled cell has no exhaustive version because its offsets are joint across
markets.

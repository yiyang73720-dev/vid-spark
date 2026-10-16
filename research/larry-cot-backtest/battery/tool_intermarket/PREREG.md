# PREREG: Larry Williams intermarket relationships (tool_intermarket)

Written 2026-10-16T09:10Z, BEFORE any backtest of this tool was run. Nothing below was chosen after
looking at a forward return. Inputs were inspected only for coverage (first and last dates, how often
the dollar index prints), never against outcomes. `results.json` stores the sha256 of this file.

## 1. Larry's definition (with sources, and how sure we are of each)

Larry Williams treats some markets as **leading indicators** of others. Rather than read one market
alone (he calls that "one-dimensional"), he looks at the short-term trend of the market that
**influences** the one he trades:

* **bonds influence stocks**;
* **gold influences bonds**, inversely: rising gold means inflation, which is bad for bonds;
* the **dollar has a strong relationship to gold**;
* **inflation follows gold**.

Sources. This sandbox cannot open the pages: DNS or the proxy fails for traders.com, tradingview.com,
usethinkscript.com, medium.com, stockcharts.com, ireallytrade.com, archive.org and wiley.com. Every
text below therefore comes from search-engine extracts, and no page number is claimed as checked.

1. **Stocks & Commodities interview, July 1997**
   (traders.com/Documentation/FEEDbk_docs/1997/07/0797Williams.html). Extracts:
   * "if I were to consider buying bonds, I want the relationship between bonds and, let's say, gold,
     to be giving me the signal", not the bond price alone;
   * a buy bond signal should be confirmed by a similar signal in Eurodollars;
   * municipal bonds tend to get weak at major highs in the bond market.
2. ***Long-Term Secrets to Short-Term Trading*** (Wiley 1999; 2nd ed. 2011), chapter "Will-Spread and
   the S&P 500 Stock Index" (title from a table-of-contents extract). Quoted in a forum post
   (usethinkscript.com thread 6049, which cites p. 170 of the 2nd edition): Willspread "measures the
   flow of price between the primary market we are trading and a secondary market that influences the
   primary. As you know, Bonds influence stocks, and Gold influences Bonds."
   The formula comes from third-party code only:
   * spread = (primary / secondary) × 100, then EMA(5) − EMA(20) of the spread (MQL5 code base 8528;
     TradingView script Rak6df4K; traderyam.blogspot.com 2016; usethinkscript uses simple averages);
   * a MetaStock forum post gives 7/11 EMAs;
   * reading: crossing zero upward means the rising trend is likely to continue; below zero means a
     downtrend. One description adds a next-bar breakout trigger.
3. **Greatest Swing Value T-bond system** (TradingView chart idea 0uyx2uHR, a third-party
   reconstruction of Larry's bond system, backtest 2002-2020). Its rules include a **gold filter:
   gold lower than it was 20 days earlier** for bond buys. The author says the gold filter on short
   trades "significantly improves" the result, and concludes "the times of this system have passed".
   This is the only explicit intermarket rule with a parameter that could be found. It trades bonds,
   not a commodity.
4. **Investors Daily Edge interview** (ireallytrade.com/investorsdailyedge.html, undated): "the U.S.
   Dollar shows a strong relationship to Gold and at this time the DX is overvalued vs Gold in my
   Valuation Index". This is a LEVEL (valuation) relation, which tool_valuation tests.
5. **StockCharts ChartWatchers, March 2023**, "Trading Legend Larry Williams' 2023 Market Forecast":
   while most hold the popular view that gold responds to inflation, "he sees it in reverse—that
   inflation tends to follow gold prices".
6. **Mail & Guardian, 2006** (mg.co.za/?p=319359): Larry names US interest rates as the main driver of
   gold. During a period of sustained rate hikes, gold rallies, because it is still seen as an
   inflation hedge.
7. **emini-watch.com, "Trading with Larry Williams"** (third-party eyewitness). When trading equities
   he looks at the bond market, and when trading bonds he looks at the gold market. He uses "patterns
   in the short-term trend in these markets against each other", not long-term correlations. The
   author doubts intermarket relationships "can be relied on for the long, long term".
8. ***The Definitive Guide to Futures Trading*** (Windsor 1988-89). A listing says it covers
   "intermarket analysis". The content could not be retrieved.

**What Larry does and does not state.**

* Explicit, with a parameter: gold → bonds, inverse, using gold's 20-day change (source 3, a
  reconstruction).
* Explicit, without a parameter for commodities: bonds → stocks (2), gold ↔ dollar (4),
  gold → inflation (5), rates up → gold up (6).
* No source gives Larry's own intermarket timing rule for crude, silver, soybeans or sugar.

So the PRIMARY rule is the battery's suggested dollar-change rule, applied to **gold**, the commodity
Larry explicitly ties to the dollar. Larry's literal gold → bonds rule is tested as a pre-registered
secondary (section 5B).

**Sign conventions (pre-registered, all from the sources above).**

| leader | sign | reason |
|---|---|---|
| broad dollar | inverse | (4); standard dollar-commodity relation |
| 10y bond price | inverse | (1), (3), (6): rising gold and rising rates go together, so falling bond prices are bullish for gold and inflation assets |
| gold | direct, for the other commodities | (5): inflation follows gold |

Both sides are always reported. The opposite sign convention's long rule has exactly minus the
short-side excess of these rows, so nothing is hidden by the choice.

**Relation to tool_valuation.** tool_valuation tests the LEVEL of the market/comparator ratio (cheap
or dear against gold, bonds and the dollar over 3 years). This tool tests the short-term CHANGE of
the leading market, i.e. "follow the influencing market's trend". Willspread (5C) is a ratio
oscillator too, but a short-term (5/20-day) one.

## 2. Formulas

**Decision grid.** All Fridays from `fh.weekly_decision_dates`. Timing `price`: entry at the first
close strictly after the Friday.

**As-of reading.** `x(t)` is the last observation on or before date t. If that observation is more
than 4 calendar days older than t, `x(t)` is undefined (NaN).

**Leader change over L weeks at decision Friday t:**

    chg_L(t) = x(t - lag) / x(t - lag - 7L days) - 1        (NaN if either reading is undefined)

The leaders are:

* **broad dollar** `usd_broad` (DTWEXBGS). The H.10 release comes out on Mondays and covers the
  previous week, so at a Friday close only data through the previous Friday is public. **lag = 7
  days.**
* **10y bond price proxy** `ust10_price`: the price of a 6% 10-year bond at DGS10, a market yield
  observable at the close. lag = 0.
* **gold** `gold`: the London fix, set before the NY close. lag = 0.

**Signal for market M with leader X and lookback L.** `chg_L(t)` uses the leader's sign:

* inverse (dollar, bond): long if chg < 0, short if chg > 0;
* direct (gold): long if chg > 0, short if chg < 0;
* 0 if chg is exactly 0; NaN, meaning the week is undefined, if chg is undefined.

## 3. The ONE primary rule

* **Market:** gold, traded on `gold` (v1 London spot series).
* **Leader:** broad dollar `usd_broad`, read with the 7-day publication lag.
* **Rule:** `chg_8 = usd(t-7) / usd(t-63) - 1`.
  * **LONG gold when chg_8 < 0** (the dollar fell over the 8 weeks to the last published value).
  * The short side (chg_8 > 0) and the net side are secondary.
* **Horizon:** h = 8 weeks; entry at the first gold close after the decision Friday.
* **Expected sample:** about 2006-03 to 2026-08 (DTWEXBGS starts 2006-01-02).
* **Statistics** (shared harness): n, episodes, mean, baseline, excess, hit, Monte-Carlo
  circular-shift perm_p (K=2000, seed 11), and the exhaustive perm_p_exact (+1 correction).

**Secondary headline (pre-registered, not the verdict).** The same dollar rule, long side, h = 8,
pooled across the 5 markets with `fh.evaluate_pooled`:

* crude on `crude_roll` (ends 2024-04);
* gold, silver, soybean and sugar on their v1 series.

## 4. Pre-declared verdict criteria (applied to the primary rule only)

These are identical to the other tools of the battery.

* **positive EV, robust:** all of the following hold:
  * primary excess > 0 and perm_p <= 0.05;
  * excess > 0 in both split halves;
  * walk-forward (section 6) excess > 0;
  * S1 long-only, after 0.05%/side costs, has Sharpe > 0 and random-timing null p <= 0.10.
* **positive EV, fragile:** primary excess > 0 and perm_p <= 0.20, but at least one robustness
  condition fails.
* **negative (anti-signal):** primary excess < 0 and perm_p >= 0.95.
* **no edge:** anything else.
* **untestable:** the primary cannot be computed.

## 5. Variant grid (robustness only; never promoted to "the result")

Unless stated otherwise, every variant is run at h ∈ {1, 4, 8, 13} on the long, short and net sides,
with price timing.

Traded series:

* crude: `crude_roll`;
* gold, silver, soybean, sugar: their v1 series.

**A. Leader-change grid ("leader trend leads the market").**

* Leaders: dollar (inverse, lag 7), bond (inverse), gold (direct).
* Lookbacks L ∈ {4, 8, 13} weeks.
* Markets: crude, gold, silver, soybean, sugar. Gold is not used as the leader of gold.
* Context: the same grid for crude traded on `crude_cl` (2000-2026).

**B. Larry's literal gold → bonds rule (source 3).**

* Rule: long bonds if gold's last close on or before Friday is BELOW its close 20 gold trading days
  earlier; short if above.
* Traded on `ust10_price`, the bond price proxy; `ust10_tr` as robustness.
* Gold input:
  * the v1 `gold` series (2000-2026);
  * extended: the pysystemtrade gold roll-adjusted index (1975-2024), built by
    `tool_premium/premium_lib.pst_roll_index("gold")`. It is a futures excess-return index, so its
    20-day change excludes roll jumps.
* Split-half (h = 8, long) on both versions.

**C. Willspread (source 2).**

* For each market M and secondary X, on common daily dates: `S = 100 · M / X`,
  `WS = EMA5(S) − EMA20(S)`. EMA: alpha = 2/(n+1), recursive, seeded with the first value, defined
  from the 20th common day.
* Read as of the Friday with the 4-day staleness rule. With the dollar as X, it is read as of
  Friday − 7 days, because both legs must be public.
* Long M if WS > 0 (M gaining on its influencing market: the "zero-line up" reading); short if
  WS < 0.
* Secondaries:
  * gold, for crude, silver, soybean and sugar;
  * the bond price;
  * the dollar.
* Robustness at h = 8 only: EMA 7/11 (the MetaStock parameters).
* The opposite reading (long when the secondary gains) is the short side.

**D. Composite vote (L = 8).**

* Crude, silver, soybean and sugar: long when at least 2 of the 3 leader signals (dollar, bond,
  gold) are long; short when at least 2 are short. Valid only when all three are defined.
* Gold: long when dollar and bond are both long; short when both are short; 0 otherwise.

**E. Extended history (pysystemtrade roll-adjusted indexes, built by `premium_lib.pst_roll_index`).**

* Each market is traded on its own pst index:
  * crude 1990, gold 1975, silver 1970, soybean 1985, sugar 1970, all to 2024-03.
* Leaders:
  * bond (`ust10_price`, 1962+), inverse;
  * gold (pst gold index), direct, for non-gold markets.
* L ∈ {4, 8, 13}. The dollar has no history before 2006, so it is not used here.
* Split-half (h = 8, long) of bond-inverse L = 8 for gold.

**F. Control: own trend.**

* For each market: long if its own 8-week change at the Friday close is > 0, short if < 0.
* Context statistics:
  * the share of weeks where the primary dollar signal and gold's own 8-week trend agree;
  * the contemporaneous correlation of the dollar's 8-week change with gold's 8-week change over the
    same window (the "relationship exists, but does it lead?" question). This uses no forward
    returns.

**G. Pooled cells** (`fh.evaluate_pooled`), long and short, h ∈ {1, 4, 8, 13}:

* each (leader, L) of grid A across its markets: 5 markets for the dollar and bond, 4 for gold;
* the vote composite.

**H. Primary rule, other traded gold series.** The pst gold roll index (futures, 2006-2024).

## 6. Out-of-sample (SPEC2 rule 5)

**Split half.** `fh.split_half` of the primary frame on `gold`, h = 8. Long is the criterion; short
and net are context.

**Walk-forward.**

* The primary has no estimated parameter: its changes use only past values. Its walk-forward signal
  is therefore identical to the static one. This is reported, evaluated with `fh.walk_forward`
  (identity fit) from 2011-01-01.
* The pre-registered walk-forward test (the one the verdict uses) is a **leader/lookback/sign
  selection** rule for gold:
  * Each 1 January from 2011-01-01, the fit chooses among 12 configurations: leader {dollar, bond} ×
    L {4, 8, 13} × sign {pre-registered, reversed}.
  * It picks the configuration with the highest long-side h = 8 excess on `gold`.
  * The fit uses only decision weeks whose 8-week exit date is before the cutoff, and needs at least
    26 long-signal weeks.
  * If no configuration qualifies, or the best excess is <= 0, there is no trade that year (signal 0,
    week still valid).
  * Evaluated at h = 8 (long) on `gold` and compared with the static primary over the same 2011+
    weeks.

## 7. Strategy (SPEC2 rule 6)

* `fh.simulate(primary frame, "gold", timing="price")` with defaults:
  * S1 and S2, each long+short and long-only;
  * hold 40 days, cost 0.05%/side;
  * null of 2000 circular shifts, seed 11;
  * vs buy & hold over the same window.
* Context: the same dollar-L8 rule simulated on:
  * crude (`crude_roll`);
  * silver, soybean and sugar (v1 series);
  * Larry's gold → bonds rule on `ust10_price`.

## 8. Multiple testing

* Every grid row is reported.
* Benjamini-Hochberg q-values across all non-pooled h = 8 rows (on perm_p) are information only.
* The headline is the primary row, whatever the grid shows.

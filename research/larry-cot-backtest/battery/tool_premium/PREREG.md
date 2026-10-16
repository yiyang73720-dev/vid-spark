# PREREG — Larry Williams "premium / discount" tool (tool_premium)

Written 2026-10-16T08:40Z, BEFORE any backtest of this tool was run. Nothing below was chosen after
looking at a forward return. Inputs were only inspected for coverage (dates, counts, how contract
rolls can be bridged), never against outcomes. `results.json` stores the sha256 of this file.

## 1. Larry's definition (with sources, and how sure we are of each)

Larry Williams treats the relation between the **nearby** futures contract and the **deferred**
contracts as a fundamental tool:

* nearby price **above** the deferred = a **"premium" market** (inverted / backwardated). Commercial
  users are paying up for immediate delivery (an "urgent need" for the commodity), which is
  **bullish**;
* nearby **below** the deferred = a normal, **"discount"** / carrying-charge market (contango), the
  non-bullish (bearish) state;
* the **cross** from a discount market to a premium market is bullish, and an **increasing premium
  in a declining market** says a bullish condition may be developing.

Sources and their limits (this sandbox cannot open the pages themselves; DNS for those sites
fails, so every text below comes from search-engine extracts):

1. *Masterclass with Larry Williams: COT, Market Cycles & Trading Secrets* (transcript,
   https://sozai.app/transcript/masterclass-larry-williams-cot-market-cycles-trading-secrets/).
   The search snippet of the auto-generated transcript reads: "bullish buy signals work a lot
   better so when there's a premium in the front months that that's a bullish signal". The page
   summary says he "explains the concept of premiums in commodity markets and their bullish
   implications" and treats "COT reports and market premiums as fundamental tools".
   Auto-transcribed, so the wording is approximate.
2. A commodity-education page on Williams-style spread analysis that a search engine summarised
   (the exact URL could not be confirmed; candidate pages were ireallytrade.com/futurestrading and
   solerinvestments.com/Online-Trading/Larry-Williams-Trading.htm). Extracts: "the nearby month will
   sell for more money than the distant months. This is known as a 'premium' market"; "the 'cross'
   from a normal or 'discounted' market (for the nearby or spot month) to an inverted or 'premium'
   market is bullish"; "Commercial buying causes this phenomenon because of an urgent need for the
   commodity"; "a declining market may also have an increasing premium in the spread, which can
   indicate that a bullish condition may be developing". **Attribution to Larry himself is
   unverified.**
3. Books where Larry covers fundamentals: *Long-Term Secrets to Short-Term Trading* (Wiley 1999,
   2nd ed. 2011), *Trade Stocks & Commodities with the Insiders: Secrets of the COT Report*
   (Wiley 2005), *The Definitive Guide to Futures Trading* (Windsor 1988-89), and his Inner Circle
   workshop material. I could NOT retrieve a verbatim premium passage or page number from any of
   them. The table-of-contents extracts found (Wiley catalog PDF, Insiders TOC) show no chapter
   titled "premium". So no page-level citation is claimed.
4. Terminology warning: some exam glossaries (e.g. CertFuel Series 3) call the *normal*
   (contango) market a "premium market", meaning the deferred months are at a premium. Larry's
   usage, and this test, is the opposite: premium = **nearby over deferred**.

Operational reading used here: **premium = contract 1 settles above contract 2 → bullish (long);
discount = contract 1 below contract 2 → bearish (short).** Larry presents the premium mainly as a
bullish condition and filter. So the PRIMARY rule is the long side; the long/short version is the
first secondary.

## 2. The ONE primary rule

* **Market:** WTI crude oil.
* **Signal input:** EIA NYMEX settlement prints of contract 1 (RCLC1) and contract 2 (RCLC2), on the
  index days of `fh.build_crude_roll_index(0)` (both contracts print, positive, not a stale holiday
  row). Premium `p = c1 / c2 - 1`.
* **Decision dates:** every Friday from 1985-01-04 to 2024-03-29. The signal is read as of the
  Friday close: the last index day on or before the Friday. If that print is more than 4 calendar
  days old, the week is undefined.
* **Setup:** long (+1) when `c1 > c2` (premium); short (-1) when `c1 < c2` (discount); 0 when
  equal. Primary = **LONG side** (premium weeks).
* **Traded series:** `crude_roll`, the roll-adjusted front-month excess-return index (holds contract 1
  through its last trading day, so it includes roll yield). **Timing:** `price`, meaning entry at
  the first close strictly after the Friday.
* **Horizon:** **h = 8 weeks** (40 trading days).
* **Statistics** (shared harness, SPEC2 rules 3-4): n, episodes, mean signed return, baseline = mean
  over all eligible weeks, excess, hit, circular-shift perm p (K=2000, seed 11) and the exhaustive
  +1-corrected p.

## 3. Pre-declared verdict criteria (applied to the primary rule only)

* **positive EV, robust**: all of the following hold.
  * primary excess > 0 and perm_p <= 0.05;
  * excess > 0 in both split halves;
  * walk-forward (section 5) excess > 0;
  * S1 long-only, after 0.05%/side costs, has Sharpe > 0 and random-timing null p <= 0.10.
* **positive EV, fragile**: primary excess > 0 and perm_p <= 0.20, but at least one of the robustness
  conditions fails.
* **negative (anti-signal)**: primary excess < 0 and perm_p >= 0.95 (the signal reliably picks
  worse-than-random weeks).
* **no edge**: anything else.
* **untestable**: the primary cannot be computed.

## 4. Variant grid (robustness only; never promoted to "the result")

All variants are run at h ∈ {1, 4, 8, 13} and on the long, short and net sides unless stated
otherwise. They use the same Friday grid and price timing unless stated otherwise.

Crude curve definitions, traded on `crude_roll`:

| id | definition |
|---|---|
| `primary` | sign(c1 - c2) (the rows other than long/h=8 are part of the grid) |
| `cl1_cl4` | sign(c1 - c4) |
| `cl2_cl3` | sign(c2 - c3), which avoids contract-1 expiry dislocations |
| `pct_0.5` / `pct_1.0` | spread scaled by price: long if p >= +0.5% / +1.0%, short if p <= -0.5% / -1.0%, else 0 |
| `range52` | Larry-style 0-100 index of p over the last 52 weekly values incl. the current one: 100·(p-min)/(max-min); long >= 80, short <= 20 |
| `steepen4` | spread change: long if p(t) - p(t-4w) > 0, short if < 0 |
| `cross4` | fresh cross: long if p(t) > 0 and p was <= 0 in any of the previous 4 weeks; short if p(t) < 0 and p was >= 0 in any of the previous 4 weeks |
| `z_exp` | p z-scored against its expanding mean/std of past + current weekly values (min 104 weeks); long if z >= +1, short if z <= -1 |

The primary signal on alternative traded series:

* `crude_roll_lead5`: rolls 5 business days before expiry.
* `crude_cl`: v1 CL continuous, unadjusted, signal window 2000-08..2024-03.
* `crude_spot`: EIA Cushing spot, 1986..2024-03.

Other markets, from the scout's pysystemtrade curves (`curve_daily.csv`):

* Signal = sign(near/far - 1) as of Friday, with the same 4-day staleness rule, for crude (Nov vs
  Dec), gold, silver, soybean, sugar, corn, wheat, copper, natgas and heatoil.
* Traded on a roll-adjusted index of the pysystemtrade held (PRICE) contract, built here. On a roll
  day the return is P_new(t)/P_new(t-1) - 1, with P_new(t-1) taken from the FORWARD or CARRY column
  of t-1. If neither carries the new contract, the switch is moved back to the latest earlier day
  (within 10 rows) where the new contract has a price, and the unbridged count is reported.
* Also traded, where they exist, on the v1 series (gold, silver, soybean and sugar) and on ho_spot
  for heatoil.
* Pooled cell: long side across the 10 markets on their pysystemtrade roll indexes
  (`fh.evaluate_pooled`).

Non-Larry comparator, crude commercial inventories (EIA WCESTUS1, excl. SPR):

* Deviation `dev = inv / norm - 1`. `norm` is the mean of the inventory on the week-ending dates
  nearest to (week_ending - k years), for k = 1..5, each within ±3 days. All 5 are required, and
  only past data are used.
* Setup: long if dev < 0 (stocks below the 5-year norm), short if dev > 0.
* Timing `eia`: decision = week-ending Friday, entry on or after the following Thursday.
* Traded on `crude_roll`; context runs on `crude_spot` through 2026.
* "Says the same thing?": on common Fridays, report the share of weeks where sign(-dev) equals
  sign(p), the correlation of dev and p, and the 2x2 table.

## 5. Out-of-sample (SPEC2 rule 5)

* **Split half:** `fh.split_half` of the primary frame on `crude_roll`, h=8, long side (net side as
  context).
* **Walk-forward:** the primary rule has no fitted parameter, so its walk-forward signal equals the
  static one. The pre-registered walk-forward is a **learned-side** rule, built with
  `fh.walk_forward` and annual refits from 1995-01-01:
  * the fit sees only decision weeks whose 8-week exit date is before the cutoff;
  * the premium-long side is traded in the next year only if, in that past sample, its 8-week
    excess on `crude_roll` was > 0; the discount-short side likewise;
  * evaluated at h=8 and compared with the static primary over the same 1995+ window.
* `z_exp` (expanding moments) is walk-forward by construction.

## 6. Strategy (SPEC2 rule 6)

* `fh.simulate(primary frame, "crude_roll", timing="price")`, default parameters:
  * S1 and S2, each long+short and long-only;
  * hold 40 days, cost 0.05%/side;
  * null of 2000 circular shifts, seed 11;
  * versus buy & hold over the same window.
* Context: the same on `crude_roll_lead5` and `crude_spot` (signal window ends 2024-03).

## 7. Context: roll yield versus spot price change

* For the primary weeks (long, short, all eligible), h=8, measure the forward return on `crude_roll`
  and, over the same entry and exit dates, the spot return (EIA Cushing, as of each date).
* Roll/basis component = roll-adjusted return - spot return.
* Report the means and how much of the primary's excess comes from each part.

## 8. Multiple testing

* All grid rows are reported.
* Benjamini-Hochberg q-values are computed across every h=8 row of the grid (perm_p) as
  information only.
* The headline is the primary row, whatever the grid shows.

# Data scout inventory (spec v2): curve, small-trader COT and sentiment data

Scope: free public data (GitHub, raw.githubusercontent.com) for Larry Williams' premium/discount
test (nearby vs deferred contract), small-trader (non-reportable) COT positions, and advisory
sentiment. About 20 searches were used: GitHub code and repo search through the MCP tool, plus
WebSearch. The `gh search` CLI returns 403 in this session. Everything downloaded is untrusted:
it was fetched shallow and pinned to one commit, parsed only with `python3 -I`, and never executed.

All numbers below come from `scout_check.py`. The tables are inserted verbatim from
`inventory_tables.md` and `scout_summary.json`, both written by that script.

> Packaging note: `S` is the root `$LARRY_COT_ROOT` (default: the package directory). The scout
> downloaded more than the battery reads; `tools/fetch_data_battery.py` checks out, at the same
> pinned commits, only the files used: 11 `multiple_prices_csv` files (the 10 below plus
> `GASOILINE.csv` for the seasonality test), `csvconfig/rollconfig.csv`, `LICENSE`, and the two
> bgithub1 CSVs. `scout_check.py` writes its outputs to `S/work2/scout/`.

## What was downloaded

| dir under `S/ext2/` | source | pinned commit (commit date) | what | license |
|---|---|---|---|---|
| `robcarver17_pysystemtrade/` | github.com/robcarver17/pysystemtrade | `bbe29e19dab73271570c94b4ee7e2f182d4b9ef1` (2026-10-08) | sparse checkout of `data/futures/multiple_prices_csv/*.csv` (252 instruments), `csvconfig/` and `roll_calendars_csv/`. Daily prices of three contracts per instrument: PRICE (the contract held), CARRY (an adjacent contract) and FORWARD, each with its YYYYMMDD contract id | GPL-3 |
| `bgithub1_cme_open_interest/` | github.com/bgithub1/cme_open_interest | `a5772c97fcb58f06ed8c945fcfec1b7e6ddc5263` (2021-07-13) | only `cot_history.csv` (CFTC legacy, all markets, sha256 `84b93b9b…`) and `cot_new_history.csv` (CFTC disaggregated, sha256 `830e7018…`) checked out | no LICENSE file; the content is CFTC public-domain data |

Derived, tidy files written by `scout_check.py` (in `S/work2/scout/`):
- `curve_daily.csv`: market, date, near_contract, near, far_contract, far, price/carry/forward
  contract and price, close_stamp_2300. One row per weekday, for crude, gold, silver, soybean,
  sugar, corn, wheat, copper, natgas and heatoil.
- `cot_smalltrader_weekly.csv`: legacy futures-only COT for the same 10 markets, 2000 to 2020,
  deduplicated. Columns are oi, noncomm long/short/spread, comm long/short, reportable long/short
  and **nonrept_long/short**.

## 1. Two contract months for the premium/discount test: FOUND (pysystemtrade), with caveats

- **Distinct contracts.** On every weekday kept, NEAR and FAR are different contract ids. The only
  exception is natgas: 23 days in Apr–May 2020 where CARRY equals PRICE. Those days are blanked.
  Exact price equality is rare (0.1%–2.9% of days). The daily log-return correlation is 0.91–0.99,
  which is high but not 1. One-leg staleness is low. These are real, separate contract series,
  not a duplicated continuous series.
- **External check (crude, heating oil).** For crude, the legs match the EIA NYMEX contract-1..4
  settlements already on disk at a median absolute difference of 0.0. Agreement is 100% within 1%
  in every 5-year period except 2015–2019. For heating oil versus EIA HO contract 1, agreement is
  100% within 1% through 2009 and drops after that (see the table).
- **NEAR is not always the front month.** pysystemtrade holds one seasonal contract for some
  markets:
  - crude: Nov vs Dec
  - corn: Sep vs Dec
  - wheat: Sep vs Dec
  - soybean: Sep vs Nov
  
  For these four, NEAR is a median of 5–7 months from delivery. The test there is an
  adjacent-month spread deeper on the curve, not Larry's literal "nearby vs next".

  Gold, silver, sugar, copper, natgas and heatoil are close to front/second, with NEAR a median
  of 1–2 months from delivery. The gap between the two legs also varies (sugar: 2, 3 or 5 months),
  so use the sign of the spread or a per-month normalised spread.
- **Ends 2024-03-28.** This is the last date for every instrument in this snapshot, about the
  same end as the EIA RCLC crude curve (2024-04-05). No free source found here covers
  2024-04 to 2026.
- **Hourly era.** Hourly rows start between 2008-12 and 2013-10, depending on the market. In
  that era the script keeps the last row of each date and drops weekend dates, and those prices
  are not always exchange settlements. Agreement with EIA degrades after 2014 for crude
  (2015–2019 only) and after 2009 for heating oil. Downstream tests should flag post-2012 results
  or report them as a sub-period.
- **Premium is rare in some markets.** The nearby premium is almost never present in gold (0.9%
  of days), wheat (1.7%) and silver (5.0%). A "premium" signal in those markets gives very few
  episodes.
- **Crude.** The EIA RCLC1–4 curve on disk (1983–2024, true contracts 1–4) remains the better
  crude source. CRUDE_W is a cross-check only.

### Futures curve: NEAR vs FAR contract (pysystemtrade multiple prices)

Coverage (one row per weekday; NEAR/FAR both present):

| market | file | first date (both legs) | last date | days both legs | weeks (W-FRI) | hourly rows from | weekend dates dropped | same-contract rows blanked | CARRY NaN days |
|---|---|---|---|---|---|---|---|---|---|
| crude | CRUDE_W.csv | 1990-10-16 | 2024-03-28 | 8164 | 1740 | 2013-10-08 | 215 | 0 | 230 |
| gold | GOLD.csv | 1975-04-01 | 2024-03-28 | 12132 | 2557 | 2013-10-07 | 87 | 0 | 163 |
| silver | SILVER.csv | 1970-06-15 | 2024-03-28 | 8764 | 2079 | 2008-12-01 | 646 | 0 | 4802 |
| soybean | SOYBEAN.csv | 1985-09-20 | 2024-03-28 | 9605 | 2007 | 2013-10-15 | 2 | 0 | 79 |
| sugar | SUGAR11.csv | 1970-02-16 | 2024-03-28 | 13223 | 2813 | 2012-05-01 | 0 | 0 | 125 |
| corn | CORN.csv | 1972-10-18 | 2024-03-28 | 11782 | 2495 | 2013-10-13 | 101 | 0 | 1136 |
| wheat | WHEAT.csv | 1973-12-19 | 2024-03-28 | 12137 | 2552 | 2011-09-26 | 95 | 0 | 501 |
| copper | COPPER.csv | 1995-09-01 | 2024-03-28 | 7219 | 1492 | 2012-12-03 | 403 | 0 | 0 |
| natgas | GAS_US.csv | 1990-07-26 | 2024-03-28 | 6347 | 1434 | 2013-10-21 | 86 | 23 | 2014 |
| heatoil | HEATOIL.csv | 1980-02-19 | 2024-03-28 | 11119 | 2302 | 2012-02-01 | 485 | 0 | 1 |

Distinctness (NEAR = nearer of CARRY/PRICE by contract id; stale = one leg unchanged while the other moved, same-contract consecutive days):

| market | near leg | far-near gap in months (days) | months to near delivery min/med/max | near==far share | corr daily dlog | near-premium share | stale near / far (all) | stale near / far (last 5y) |
|---|---|---|---|---|---|---|---|---|
| crude | CARRY | 1m: 8164 | 1/7.0/14 | 0.019231 | 0.967746 | 0.567859 | 0.006519 / 0.006519 | 0.006506 / 0.004647 |
| gold | PRICE | 2m: 12132 | 0/2.0/4 | 0.002803 | 0.992648 | 0.009067 | 0.005829 / 0.003886 | 0.002595 / 0.002595 |
| silver | CARRY | 1m: 4400, 2m: 3593, 3m: 771 | 0/1.0/3 | 0.007188 | 0.9148 | 0.050434 | 0.007181 / 0.009653 | 0.003783 / 0.007566 |
| soybean | CARRY | 2m: 9605 | 0/6.0/15 | 0.006663 | 0.941645 | 0.600937 | 0.021221 / 0.016203 | 0.010888 / 0.014238 |
| sugar | CARRY | 2m: 7666, 3m: 2197, 5m: 3360 | 0/2.0/6 | 0.010134 | 0.916645 | 0.408455 | 0.016991 / 0.021834 | 0.011638 / 0.009848 |
| corn | CARRY | 3m: 11782 | 0/5.0/13 | 0.010694 | 0.943249 | 0.403157 | 0.029923 / 0.029838 | 0.023766 / 0.012797 |
| wheat | CARRY | 3m: 12137 | 0/6.0/16 | 0.001071 | 0.923738 | 0.016726 | 0.022174 / 0.023995 | 0.012831 / 0.008554 |
| copper | PRICE | 1m: 4778, 2m: 2441 | 0/2.0/6 | 0.029228 | 0.91021 | 0.342153 | 0.013936 / 0.019566 | 0.015188 / 0.02558 |
| natgas | CARRY (0.997291 of rows) | 1m: 4299, 2m: 2048 | 0/1.0/3 | 0.004096 | 0.949005 | 0.216953 | 0.00762 / 0.007454 | 0.001833 / 0.004583 |
| heatoil | CARRY | 1m: 11119 | 1/2.0/4 | 0.003148 | 0.977673 | 0.374494 | 0.003966 / 0.003872 | 0.001645 / 0.0 |

EIA cross-check (weekly Wednesday sample; pysystemtrade leg price vs EIA NYMEX contract-k settlement on the same date; dates within 2 business days of an expiry excluded):

| market | EIA series | n compared | by contract k | median abs rel diff | within 0.25% | within 1% | within 1% by 5y period (n) |
|---|---|---|---|---|---|---|---|
| crude | RCLC1-4 | 699 | k1: 83, k2: 189, k3: 220, k4: 207 | 0.0 | 0.848355 | 0.964235 | 1990-1994: 1.0 (85), 1995-1999: 1.0 (107), 2000-2004: 1.0 (107), 2005-2009: 1.0 (113), 2010-2014: 1.0 (109), 2015-2019: 0.764151 (106), 2020-2024: 1.0 (72) |
| heatoil | HO_F1 | 831 | k1: 831 | 0.000414 | 0.765343 | 0.872443 | 1980-1984: 1.0 (115), 1985-1989: 1.0 (107), 1990-1994: 1.0 (90), 1995-1999: 1.0 (71), 2000-2004: 1.0 (89), 2005-2009: 1.0 (101), 2010-2014: 0.840708 (113), 2015-2019: 0.361111 (108), 2020-2024: 0.486486 (37) |

## 2. Small-trader (non-reportable) COT: FOUND for 2000–2020 (futures-only), with gaps

- **Source.** `bgithub1_cme_open_interest/cot_history.csv` is the CFTC legacy report for all
  10 markets. It covers 2000-01-04 to 2020-07-28 and includes non-reportable long and short
  positions.
- **Futures-only, not combined.** Open interest and non-reportable long match the known
  futures-only crude file on every common date, and never match the project's futures+options
  (F+O) disaggregated open interest. So these data are futures-only, while the v1 COT data are
  F+O combined. Keep this in mind when combining the two.
- **Gaps.** The 2006 block is duplicated in the file: 52 exact duplicate rows per market were
  dropped. **2007 is missing entirely**: there is a 378-day gap from 2006-12-26 to 2008-01-08.
  The 2019-12-24 and 2019-12-31 reports are also missing.
- **Accounting identity.** OI = reportable + non-reportable holds on every row.
- **Second file.** `cot_new_history.csv` (disaggregated, 2010–2020, futures-only) carries the
  same NonRept values on all 550 common dates, so it adds no new history.
- **Timing.** The reports are Tuesday as-of and released Friday, the same as v1. Use the v1
  entry rule.

### Small traders: CFTC legacy COT history (bgithub1/cme_open_interest)

| market | first | last | reports (deduped) | exact dup rows dropped | conflicting dup dates | gaps>8d | OI = rept+nonrept (long) | nonrept long / OI (median) | legacy OI = disagg-file OI | disagg-file OI = project F+O OI | legacy OI = project F+O OI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copper | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 4: 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.108365 | 1.0 | None | None |
| corn | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 3: 2001-12-28->2002-01-08 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.119938 | 1.0 | None | None |
| crude | 2000-01-04 | 2020-07-28 | 1018 | 52 | 0 | 5: 2001-09-04->2001-09-18 (14d); 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.058857 | 1.0 | 0.0 | 0.0 |
| gold | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 4: 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.129925 | 1.0 | 0.0 | 0.0 |
| heatoil | 2000-01-04 | 2020-07-28 | 1018 | 52 | 0 | 5: 2001-09-04->2001-09-18 (14d); 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.140286 | 1.0 | None | None |
| natgas | 2000-01-04 | 2020-07-28 | 1018 | 52 | 0 | 5: 2001-09-04->2001-09-18 (14d); 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.078351 | 1.0 | None | None |
| silver | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 4: 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.208682 | 1.0 | 0.0 | 0.0 |
| soybean | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 3: 2001-12-28->2002-01-08 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.097748 | 1.0 | 0.0 | 0.0 |
| sugar | 2000-01-04 | 2020-07-28 | 1018 | 52 | 0 | 5: 2001-09-04->2001-09-18 (14d); 2001-12-28->2002-01-08 (11d); 2003-02-14->2003-02-25 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.106023 | 1.0 | 0.0 | 0.0 |
| wheat | 2000-01-04 | 2020-07-28 | 1019 | 52 | 0 | 3: 2001-12-28->2002-01-08 (11d); 2006-12-26->2008-01-08 (378d); 2019-12-17->2020-01-07 (21d) | 1.0 | 0.088664 | 1.0 | None | None |

Disaggregated file `cot_new_history.csv` (has NonRept columns too), same for all 10 markets (first market shown): 2010-01-05 to 2020-07-28, 551 reports, gaps>8d: 2019-12-24->2020-01-07 (14d). Per-market values in scout_summary.json.

Crude check against a known FUTURES-ONLY disaggregated file (S/ext/accomplish999 CFTC_CL, 550 common dates): legacy OI equal in 1.0 of dates, legacy non-reportable long equal in 1.0 of dates.

Already on disk (S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json, crude only, disaggregated futures-only): nonrept_long/short on 874 weekly reports, 2010-01-05 to 2026-09-29; gaps>8d: none.

## 3. Already on disk (no download needed)

- `S/ext/accomplish999_crude-oil/data/series/CFTC_CL.json`: crude disaggregated futures-only
  reports, including spread and non-reportable columns. The coverage line above gives the
  numbers. It extends the crude small-trader series to 2026, for crude only.
- `S/ext/accomplish999_crude-oil`: EIA RCLC1–4 (crude contracts 1–4) and HO_F1 (heating oil
  contract 1). These were used above as the external check of the pysystemtrade legs.



## 4. Searched for but NOT found

- **Quandl/Nasdaq CHRIS mirrors** (CME_GC1/2, SI1/2, C1/2, S1/2, HG1/2, NG1/2, ICE_SB1/2): no
  GitHub copies found. Code search for these file names and for the CHRIS CSV header found only
  short S&P CHRIS_CME_ES1/ES2 test fixtures inside QuantConnect Lean forks. Nasdaq Data Link is
  blocked anyway.
- **Small-trader COT after 2020-07-28, for 2007, and futures+options combined:** no free
  multi-year GitHub copy found for the 10 markets. Leads checked and rejected:
  - hikarus211/cot-data: weekly files only from 2026-05.
  - edulongo18/cftc-cot-data-archive: disaggregated xls, 2020 onward. Not downloaded.
  - renessancek/cot_willco_index: latest snapshot only.
  - souravh093/cot_reports_data: a single `annualof.txt` (14.9 MB, legacy F+O). Its year was not
    determined, and it is not a history.
  
  The project's `S/data/cot_weekly.csv` lacks the disaggregated spread columns, so non-reportables
  cannot be derived from it.
- **Advisory/market sentiment (Market Vane Bullish Consensus, Consensus Inc. Bullish Sentiment
  Index): UNAVAILABLE for free.** Market Vane is distributed commercially: it is licensed through
  LSEG (history from 1983, on request) and shipped as a CQG built-in study. Consensus Inc. is a
  paid weekly subscription. GitHub code search for "market vane" / "bullish consensus" in CSV
  files returned 0 results. No proxy was substituted.

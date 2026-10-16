# Larry Williams COT backtest — shared spec (v1)

> Packaging note: this is the spec as given to the implementations. In this package the
> inputs are downloaded by `tools/fetch_data.py` and everything is run by `run_all.sh`
> (see README.md); outputs land in `S/work/`.

All paths below are relative to the root
`S` = `$LARRY_COT_ROOT` (default: this package directory; see README.md).
Python 3 with pandas 3.x / numpy 2.x is installed. You may `pip install scipy statsmodels`
(PyPI is reachable). No other network access exists (cftc.gov, yahoo, fred are blocked) —
do NOT try to download data; use only the files below. Run python with `-I` when reading
files under `S/ext/` (untrusted third-party data). Never execute code from `S/ext/`.

## Data (already prepared — do not modify files in S/data/)

1. `S/data/cot_weekly.csv` — CFTC COT, **Disaggregated, futures+options combined**, weekly,
   2006-06-13 … 2026-09-29 (1060 reports per market). Columns:
   `market, code, report_type, report_date, oi, prod_merc_long, prod_merc_short, swap_long,
   swap_short, m_money_long, m_money_short, other_rept_long, other_rept_short, ...`
   Markets used: `crude` (NYMEX WTI 067651), `gold` (088691), `silver` (084691),
   `soybean` (005602), `sugar` (080732). Ignore `us10y`, `us30y` (TFF report, no commercial
   category) everywhere.
   `report_date` is the "as-of" date of positions (normally Tuesday; 91 Monday-dated and 7
   Wednesday-dated rows exist because of holidays).
2. `S/data/prices_daily.csv` — daily closes `market, date, close, symbol`, ~2000 … 2026-10-05.
   crude = NYMEX CL continuous front month (unadjusted; one non-positive print 2020-04-20 = -37.63);
   gold/silver = London spot; soybean = CBOT ZS continuous; sugar = ICE SB continuous.
   Known artifacts: soybean 2008-09-12/15 (+22%/-21% roll glitch). Leave them in (robustness
   lens will test), but drop any close <= 0 before computing anything.
3. `S/data/wti-daily.csv` — EIA Cushing WTI **spot** daily `Date,Price`, 1986 … 2026-10-14.
   Used as an alternative crude price series (robustness + the 2026 case study). Contains a
   suspicious print 2026-09-25 = 85.23 (CL futures that day = 92.41). Drop non-positive prices.

## Definitions (Larry Williams method)

- **Commercials** (Larry's "insiders"/smart money) = legacy commercial ≈ Producer/Merchant + Swap Dealers:
  `comm_net = (prod_merc_long + swap_long) - (prod_merc_short + swap_short)`.
- **Managed money net** (comparator, the speculators): `mm_net = m_money_long - m_money_short`.
- **Larry COT Index** with lookback N weeks (rolling window over the last N reports, *including the current one*):
  `idx_N = 100 * (comm_net - min_N) / (max_N - min_N)`; undefined (NaN) for the first N-1 reports
  or if max_N == min_N. Lookbacks tested: **N ∈ {26, 52, 156}**. Primary = **26**.
- **Signals**: long setup when `idx_N >= buy_th`; short setup when `idx_N <= sell_th`.
  Threshold pairs tested: **(80,20)** primary, (75,25), (90,10).
- **Episode** = a maximal run of consecutive report weeks with the same setup active.

## No-look-ahead timing (mandatory)

Positions as of `report_date` are published the following Friday 15:30 ET (Monday if a holiday
delays it); crude settles 14:30 ET, so the Friday close is NOT tradable on the information.
Rule: **entry_date = first trading day (in that market's price series) on or after the Tuesday of
the week following report_date** (i.e. `report_date - weekday(report_date) + 1 + 7 days`, weekday Mon=0).
Entry price = close on entry_date.

Government-shutdown publication delays: signals whose `report_date` falls in these windows are
**excluded** (not tradable in real time): 2013-10-01…2013-10-31, 2018-12-18…2019-02-26,
2025-09-30…2026-01-20. (Robustness lens tests including them.)
Those reports still enter the rolling min/max of the index (the data exists in hindsight for later weeks).

## Forward returns

For horizon h weeks (h ∈ {1, 4, 8, 13}) the exit is the close **5·h trading days after entry**
(trading days = rows of that market's cleaned price series). `r = exit/entry - 1`.
If not enough future data, the observation is dropped.
Signed return: long `+r`, short `-r`.

**Baseline** for a (market, horizon): mean of `r` over ALL eligible report weeks (weeks with
non-NaN idx for that lookback, outside shutdown windows, with a valid forward return). For the short
side the signed baseline is `-baseline`. **excess = mean signed return - signed baseline**.
Hit rate = share of signed returns > 0.

**Permutation p-value** (one-sided, "signal timing beats random timing"): circularly shift the
weekly boolean signal series of that (market, N, threshold, side) by K random offsets
(offset uniform in [26, T-26], T = number of eligible weeks), recompute the mean signed return
over shifted-signal weeks, p = (1 + #{shifted >= observed}) / (1 + K). K = 2000.
(Circular shift preserves the clustering of signals and of returns.)

## Canonical output (both independent implementations must produce EXACTLY this file)

`canonical.csv` with one row per (market ∈ 5) × (lookback ∈ 3) × (threshold pair ∈ 3)
× (side ∈ {long, short}) × (horizon_w ∈ {1,4,8,13}) = 360 rows, columns:
`market, lookback, buy_th, sell_th, side, horizon_w, n_signal_weeks, n_episodes, mean_ret,
median_ret, hit_rate, baseline_mean, excess, perm_p`
(returns as decimals, e.g. 0.0123; n_signal_weeks counts weeks with a valid forward return).

Also write `crude_timeseries.csv`: for crude, one row per report:
`report_date, entry_date, comm_net, mm_net, oi, idx26, idx52, idx156, mm_idx26, entry_price`
where `mm_idx26` is the same index formula applied to `mm_net` with N=26.

Pre-registered headline (decided before looking at results): **crude, N=26, 80/20, h=8 weeks**,
plus the same cell pooled across the 5 markets.

# Larry Williams fundamentals battery — shared spec (v2)

Goal: backtest EACH of Larry Williams' fundamental tools separately, with one common, look-ahead-free
evaluation, and find which (if any) has a positive expected value after costs that survives
out-of-sample and multiple-testing checks. COT commercials were already tested (spec v1, results in
`S/work/`); this battery adds the other tools and then ranks all of them together.

> Packaging note: this is the spec as given to the implementations. In this package the code
> lives in `battery/` (v1 code: `engine/larry_cot.py`, `lenses/strategy/strat_lib.py`; v1 spec:
> `SPEC.md`), the inputs are downloaded by `tools/fetch_data.py` (`S/data/`, `S/ext/`) and
> `tools/fetch_data_battery.py` (`S/ext2/`), and everything is run by `battery/run_battery.sh`
> (see `battery/README.md`). Outputs land in `S/work2/<step>/`.

Root: `S` = `$LARRY_COT_ROOT` (default: the package directory; see `battery/README.md`).
New work goes under `S/work2/`. Do NOT modify `S/data/`, `S/ext/`, `S/work/`.
Python 3 + pandas 3 / numpy 2; you may `pip install scipy statsmodels`. Run python with `-I`
whenever reading third-party files under `S/ext*/` (parse only, never execute them).
Network: GitHub / raw.githubusercontent.com / PyPI reachable; cftc.gov, eia.gov, fred, yahoo,
stooq, quandl, huggingface are BLOCKED. WebSearch works (server-side) for documentation.

## Data available

- `S/data/cot_weekly.csv` — CFTC disaggregated F+O, 2006-06-13…2026-09-29, markets crude, gold,
  silver, soybean, sugar (OI + trader groups). See `S/work/SPEC.md`.
- `S/data/prices_daily.csv` — daily closes ~2000…2026-10-05: crude (CL continuous), gold/silver
  (London spot), soybean (ZS continuous), sugar (SB continuous). Drop closes <= 0.
- `S/data/wti-daily.csv` — EIA Cushing WTI spot 1986…2026-10-14.
- `S/ext/accomplish999_crude-oil/data/series/*.json` and `public/downloads/prices.csv` (third-party,
  parse only): EIA NYMEX crude contracts RCLC1..RCLC4 daily 1985…2024-04-05; HO_F1 (heating oil
  contract 1) 1980…2024-04; RBOB_F1 2005…2024-04; HO_SPOT, GAS_SPOT 1986…2026-09; Brent spot RBRTE;
  DGS10 (10y yield) 1962…2026-10; DGS2; DTWEXBGS broad dollar 2006…2026-10; EIA weekly inventories
  (WCESTUS1 commercial crude, CUSHING, distillate, gasoline …) 1982…2026-09.
  Inspect `public/downloads/catalog.csv` for exact ids/units.
- The reconciled v1 engine `S/work/engine/larry_cot.py` (COT index, entry rules, permutation test)
  and strategy lib `S/work/lens_strategy/strat_lib.py` can be imported and reused.

## Common evaluation rules (all filters)

1. **Decision dates** are weekly. A signal may only use information public at its decision time.
   - Price/curve/macro-based signals computed from a Friday close: entry = close of the NEXT trading
     day (normally Monday) in the traded series.
   - COT-based signals (incl. COT open interest): the v1 rule — entry = first trading day on/after
     the Tuesday after the report week (`S/work/SPEC.md`); shutdown windows excluded.
   - Weekly EIA inventory data (if used): released Wednesday 10:30 ET for the week ending prior
     Friday; entry no earlier than the Thursday close.
2. **Traded return series** (forward returns are measured on these):
   - crude: (a) a ROLL-ADJUSTED front-month excess-return index built from RCLC1/RCLC2
     (hold contract 1; on the day contract 1 changes identity, the return is CL1(t)/CL2(t-1)-1),
     1985…2024-04 — the right series for carry/premium tests because it includes roll yield;
     (b) CL continuous (2000…2026-10) and (c) EIA spot (1986…2026-10) as in v1.
   - gold, silver, soybean, sugar: the v1 price series.
   Report which series each number uses.
3. **Forward returns**: h ∈ {1, 4, 8, 13} weeks = 5h trading days after entry; signed (+long, −short).
   Baseline = mean forward return over all eligible decision weeks of that market/series;
   excess = mean signed return − signed baseline; hit = share of signed returns > 0.
4. **Significance**: circular-shift permutation of the weekly signal series (offsets in
   [26, T−26], K=2000, one-sided "beats random timing"); also report the exhaustive version with
   the +1 correction. Overlapping returns make naive t-stats invalid — do not report them as evidence.
5. **Out-of-sample**: every filter must report its PRIMARY rule on two halves of its own sample
   (split at the midpoint date) and a walk-forward result where any parameter or statistic that is
   estimated from data (seasonal tendencies, thresholds, z-score moments) uses only past data.
6. **Costs & strategy**: for the primary rule, simulate S1 (enter on first week of each signal
   episode, hold 40 trading days, one position at a time, 0.05% per side) and S2 (same but wait up to
   20 trading days for close above/below the 20-day SMA), long+short and long-only, vs buy & hold on
   the same series and window. Sharpe, CAGR, max DD, #trades, win rate, random-timing null p.
7. **Pre-registration**: before running any backtest, each filter writes `PREREG.md` in its dir with
   Larry's own definition of the tool (with sources), the ONE primary rule (market(s), parameters,
   side, h=8 weeks), and the full variant grid. Results for the primary rule are the headline; the
   grid is only robustness. Never promote the best grid cell to "the result".
8. **Output contract** (every filter dir): `results.json` with
   `{"filter", "larry_definition", "sources", "primary": {...}, "tests": [ {filter, variant, market,
   series, side, horizon_w, n, episodes, mean, baseline, excess, hit, perm_p, perm_p_exact, period,
   is_primary} ... ], "strategy": [...], "oos": {...}, "caveats": [...]}` plus chart-ready JSON.

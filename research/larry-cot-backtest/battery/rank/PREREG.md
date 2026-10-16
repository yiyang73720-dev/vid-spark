# PREREG — rank step: confluence test, family-wide multiple testing, ranking rule

Written 2026-10-16 ~10:02Z, BEFORE any confluence statistic, any family-wide q-value and any
bootstrap interval was computed. Its sha256 is stored in `PREREG.sha256` and copied into
`results.json`.

What was already known when this was written (stated so the reader can judge it):

- The published results of all six tools (COT v1 and the five SPEC2 tools), including the
  adversarial verifiers' corrections. In particular, the premium primary is positive but not
  significant. The seasonality primary is significant but sits on untradable spot or unadjusted
  series. The other four tools show no edge.
- `verify_rebuild.py` was run once. It rebuilds each tool's primary signal with that tool's own
  library and reproduces the published n, mean, excess, hit, Monte-Carlo perm_p and T exactly.
  It computes nothing new.

Nothing about how the tools combine (agreement rates, confluence returns) had been looked at.

## 1. Confluence components

Each component is the tool's own pre-registered signal construction, with both sides kept:
+1 is a long setup, −1 a short setup, 0 none, NaN undefined. Each is rebuilt with the tool's
library, via `rank_lib.py`.

| component | definition (market m) | read at |
|---|---|---|
| `cot` | v1 Larry COT index of commercials' net (F+O disaggregated), N=26 reports: ≥80 → +1, ≤20 → −1 | COT report date |
| `premium` | crude: sign(RCLC1/RCLC2 − 1) (tool_premium primary). Other markets: sign(near/far − 1) from the pysystemtrade curve (tool_premium `pst_sign`) | Friday of the report week |
| `seasonal` | tool_seasonality primary signal of m: prior-year week-of-year analogs, h=8, 70/30. crude is estimated on crude_spot with L=15; gold/silver/soybean/sugar on their v1 series with L=10 | Friday of the report week |
| `valuation` | tool_valuation VI of m vs London gold (EMA2−EMA22, N=156): ≤20 → +1, ≥80 → −1. Not defined for gold, which is the comparator, so it is absent there | Friday of the report week |
| `oi` | tool_open_interest primary rule on m's F+O OI: 4-report OI change in the bottom/top 10% of the last 156 reports, and |4-report price change| ≤ its 156-report median → +1 / −1 | COT report date |
| `dollar` | tool_intermarket `lead_usd_L8`: DTWEXBGS 8-week change, read as of Friday−7d. Fell → +1, rose → −1 | Friday of the report week |

## 2. Common grid, timing, eligible weeks

- Decision dates are the CFTC report dates of market m (S/data/cot_weekly.csv, the v1 file).
- Timing is `cot`, the v1 rule: entry at the first close on or after the Tuesday of the
  following week. The Friday components use the close of the report week's Friday, which is
  known before that entry. COT is public on that Friday at 15:30 ET (Monday in holiday weeks),
  also before the Tuesday entry. Exit is 5h trading days after entry.
- Reports inside the v1 shutdown windows are excluded.
- A week is eligible when every component that exists for m is defined (non-NaN), it is not
  excluded, and it has a forward return on the traded series.
- **Every comparison between confluence and a single tool uses exactly these eligible weeks.**
  So the baseline and T are identical.

## 3. Confluence rule (PRIMARY)

Let L be the number of components at +1 and S the number at −1.

- **Long setup** if L ≥ 2 and L > S.
- **Short setup** if S ≥ 2 and S > L.
- Otherwise no setup.

**PRIMARY cell:** crude, all six components, LONG side, h = 8 weeks, traded on `crude_roll`
(roll-adjusted front-month excess-return index), timing `cot`.

## 4. Primary hypothesis tests and verdict

- **H-a, beats random timing.** The confluence long h=8 excess has an exhaustive circular-shift
  permutation p (+1 corrected) ≤ 0.05. The Monte-Carlo p (K=2000, seed 11) is reported too.
- **H-b, beats each tool alone.** Take every component X with at least 10 long weeks among the
  eligible weeks. Define D_X = excess(confluence long) − excess(X long) on the same eligible
  weeks. The test is a one-sided circular moving-block bootstrap: 13-week blocks, B=2000,
  `default_rng(11)`, with the weeks resampled jointly for both signals. Its p is
  (1 + #{D* ≤ 0}) / (1 + B), and the requirement is p ≤ 0.05 for every such X.
- **Verdict.**
  - "confluence beats each tool alone" if H-a holds and H-b holds for every X.
  - "confluence better than every single tool but not significantly" if the confluence excess
    is > 0 and larger than every component's excess on the common weeks, but a test fails.
  - Otherwise, "confluence does not beat the tools alone".
- **Also reported for the primary:**
  - Split halves at the midpoint date of the eligible weeks.
  - S1/S2 strategies, long+short and long-only, at 0.05% per side, against buy & hold on
    crude_roll over the same window. `fh.simulate`, timing cot, null_k = 2000.
  - Walk-forward: no parameter of the confluence rule is fitted. Every component uses only
    information available at its decision time: trailing COT/OI windows, prior-year seasonal
    analogs, trailing valuation EMAs and min/max, and no parameters at all for premium and the
    dollar. The walk-forward signal is therefore the static rule ("identity"); this is stated,
    not separately tested.

## 5. Secondary analyses (robustness only; never promoted to "the result")

1. The primary rule on the short and net sides, and at h = 1, 4, 13.
2. Rule variants on crude:
   - k=3 majority (L ≥ 3 and L > S);
   - unanimous k=2 (L ≥ 2 and S = 0);
   - event-only components {cot, seasonal, valuation, oi}, k=2 majority;
   - COT-anchored: cot = +1, and among the other components L_o ≥ 1 and L_o > S_o (short is the
     mirror image).
3. Pooled 5 markets: crude, gold, silver, soybean, sugar. Same rule, using the components that
   exist per market. Traded series are crude_roll, gold, silver, soybean and sugar (the v1
   series). The pooled cell uses `fh.evaluate_pooled` (MC p). Pooled D_X uses a block bootstrap
   over calendar weeks (13-week blocks; all markets' rows of a week move together) with
   per-market baselines.
4. The crude primary traded on crude_roll_lead5 and on crude_cl.
5. Pairwise agreement between components on the common weeks (descriptive).

## 6. Family-wide multiple testing

- **Family.** Every row of `tests` in the five SPEC2 `results.json` files, plus the 360-cell v1
  COT grid. The v1 grid is recomputed with `fh.evaluate` on the v1 series: the harness reproduces
  v1 bit-for-bit, and this also gives the +1-corrected exhaustive p.
- **p used.** perm_p_exact (+1 corrected) where present, else perm_p (Monte-Carlo, e.g. pooled
  cells). Rows with neither are dropped and counted.
- **Main.** Benjamini-Hochberg at q = 0.10 over all rows. Also reported:
  - BH over the h=8 rows only;
  - BH over the six primary rows only;
  - Benjamini-Yekutieli over all rows (valid under any dependence).
- Survivors are reported per tool. The confluence rows from section 5 go in a second family
  (main family + confluence rows); they are not part of the main one.

## 7. Ranking rule

Scorecard of each tool's PRE-REGISTERED primary. Numbers come from `results.json`, using the
verifiers' corrected numbers where they corrected one.

- **R1** primary excess h8 > 0.
- **R2** primary perm p ≤ 0.05: exhaustive +1 p, or the MC p for pooled cells.
- **R3** the primary row survives the main family BH at q ≤ 0.10.
- **R4** both split halves have excess > 0.
- **R5** the pre-registered walk-forward result has excess > 0. For tools with no fitted
  parameter, the walk-forward equals the static rule, so its sample excess is used.
- **R6** after costs: the tool's primary S2 long-only Sharpe is > buy & hold Sharpe on the same
  series and window, AND its random-timing null p ≤ 0.10.
- **R7** tradable: the primary signal's excess is > 0 when measured on a roll-adjusted futures
  series a trader can hold. That is crude_roll, a pysystemtrade roll index, or the tool's own
  tradable check:
  - premium: the primary itself (crude_roll);
  - seasonality: its pre-registered tradable pooled check;
  - valuation and open_interest: the primary itself (crude_roll);
  - intermarket: the primary signal on pst_gold;
  - COT: the v1 primary signal re-evaluated on crude_roll, computed in this step.

**Tiers:**

- **Tier 1, robust positive EV after costs:** R1–R7 all pass.
- **Tier 2, positive EV, fragile, on a tradable series:** R1, R4, R5 and R7 pass, the primary
  p ≤ 0.20, and at least one of R2/R3/R6 fails.
- **Tier 3, positive in-sample but not tradable:** R1 and R2 pass, R7 fails.
- **Tier 4, no edge:** everything else.

Order: tier, then the number of R passes (descending), then the lower bound of the 95%
block-bootstrap interval of the primary excess (descending). The top-ranked tool is named
"least bad / most promising".

## 8. Chart data

- Per tool: primary excess h8 with a 95% interval. The interval comes from a circular
  moving-block bootstrap of the eligible weeks: 13-week blocks, B=2000, seed 11. For the pooled
  seasonality primary the blocks are calendar weeks, with per-market baselines recomputed in
  each resample.
- Per tool: OOS halves, and S2 Sharpe (long-only and long+short) against buy & hold.
- Weekly equity curves of the top-ranked tool's primary strategy against buy & hold.

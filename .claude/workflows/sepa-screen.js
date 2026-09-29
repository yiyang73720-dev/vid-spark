export const meta = {
  name: 'sepa-screen',
  description: 'Minervini SEPA screen: mechanical Trend Template + Stage 2 gate, then earnings, catalyst and VCP entry per candidate, adversarially verified',
  whenToUse: 'Screen US stock tickers for Minervini SEPA buy candidates (Trend, Earnings, Announcement/catalyst, Specific entry point). args: {tickers: ["NVDA", ...], universeFile?: "path of tickers ranked for RS", dataDir?: "sepa-data", benchmark?: "SPY", fetch?: true, asof?: "YYYY-MM-DD", rs?: {"NVDA": 95}, maxCandidates?: 12}',
  phases: [
    { title: 'Mechanical', detail: 'fetch data, python3 -m sepa scan: 8-point Trend Template, Weinstein stage, RS, EPS, VCP' },
    { title: 'Earnings', detail: 'EPS / revenue acceleration, margins, surprise, guidance' },
    { title: 'Catalyst', detail: 'why institutions must buy now' },
    { title: 'Entry', detail: 'VCP pivot, buy range, stop' },
    { title: 'Verify', detail: 'two skeptics try to refute each candidate' },
    { title: 'Report', detail: 'ranked SEPA watchlist' },
  ],
}

const A = args || {}
const toList = v => (Array.isArray(v) ? v : typeof v === 'string' ? v.split(/[\s,]+/) : [])
const tickers = [...new Set(toList(A.tickers).map(t => String(t).trim().toUpperCase()).filter(Boolean))]
const dataDir = A.dataDir || 'sepa-data'
const reportsDir = `${dataDir}/reports`
const benchmark = String(A.benchmark || 'SPY').toUpperCase()
const doFetch = A.fetch !== false
const asof = A.asof || null
const universeFile = A.universeFile || null
const rsOverrides = Object.entries(A.rs || {}).map(([t, r]) => `${String(t).toUpperCase()}=${Number(r)}`)
const maxCandidates = Number(A.maxCandidates) > 0 ? Number(A.maxCandidates) : 12
const asofText = asof ? `as of ${asof} (ignore anything published after that date)` : 'as of the latest data'

if (!tickers.length && !universeFile) {
  return { error: 'Pass args.tickers (e.g. ["NVDA","ANET"]) and/or args.universeFile (a file of tickers, one per line).' }
}

const tickerArgs = tickers.length ? `--tickers ${tickers.join(' ')}` : ''
const fetchCmd = `python3 -m sepa fetch --data-dir ${dataDir} --benchmark ${benchmark} ${tickerArgs}${universeFile ? ` --tickers-file ${universeFile}` : ''}`
const scanCmd = [
  `python3 -m sepa scan --data-dir ${dataDir} --benchmark ${benchmark} --out-dir ${reportsDir}`,
  tickerArgs,
  asof ? `--asof ${asof}` : '',
  rsOverrides.length ? `--rs ${rsOverrides.join(' ')}` : '',
].filter(Boolean).join(' ')

const NUM = { type: ['number', 'null'] }
const STR = { type: ['string', 'null'] }

const MECH_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    error: STR,
    notes: { type: 'array', items: { type: 'string' } },
    asof: STR,
    rs_universe_size: NUM,
    rs_note: STR,
    market: STR,
    errors: { type: 'array', items: { type: 'string' } },
    tickers: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          ticker: { type: 'string' },
          stage: NUM,
          trend_passed: NUM,
          trend_failed: { type: 'array', items: { type: 'number' } },
          rs_rating: NUM,
          pillar_trend: { type: 'string' },
          pillar_earnings: { type: 'string' },
          entry_status: { type: 'string' },
          pivot: NUM,
          extension: NUM,
          below_pivot: NUM,
          stop: NUM,
          candidate: { type: 'boolean' },
        },
        required: ['ticker', 'pillar_trend', 'entry_status', 'candidate'],
      },
    },
  },
  required: ['ok', 'tickers'],
}

const EARN_SCHEMA = {
  type: 'object',
  properties: {
    ticker: { type: 'string' },
    verdict: { enum: ['pass', 'fail', 'insufficient'] },
    quarters: {
      type: 'array',
      items: {
        type: 'object',
        properties: { quarter: { type: 'string' }, eps_yoy: NUM, revenue_yoy: NUM, margin: NUM },
        required: ['quarter'],
      },
    },
    eps_accelerating: { type: ['boolean', 'null'] },
    revenue_accelerating: { type: ['boolean', 'null'] },
    margin_trend: { enum: ['expanding', 'flat', 'contracting', 'unknown'] },
    surprise: STR,
    guidance: { enum: ['raised', 'maintained', 'lowered', 'none', 'unknown'] },
    rationale: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['ticker', 'verdict', 'rationale', 'sources'],
}

const CATALYST_SCHEMA = {
  type: 'object',
  properties: {
    ticker: { type: 'string' },
    verdict: { enum: ['pass', 'fail', 'insufficient'] },
    strength: { enum: ['strong', 'moderate', 'weak', 'none'] },
    why_now: { type: 'string' },
    catalysts: {
      type: 'array',
      items: {
        type: 'object',
        properties: { type: { type: 'string' }, date: STR, summary: { type: 'string' }, source: STR },
        required: ['type', 'summary'],
      },
    },
    risks: { type: 'array', items: { type: 'string' } },
    next_earnings_date: STR,
    rationale: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['ticker', 'verdict', 'strength', 'why_now', 'rationale', 'sources'],
}

const ENTRY_SCHEMA = {
  type: 'object',
  properties: {
    ticker: { type: 'string' },
    engine_status: { type: 'string' },
    agree_with_engine: { type: 'boolean' },
    setup_quality: { enum: ['A', 'B', 'C', 'none'] },
    contractions: { type: 'string' },
    pivot: NUM,
    buy_range_low: NUM,
    buy_range_high: NUM,
    stop: NUM,
    risk_pct: NUM,
    action: { enum: ['buy_now', 'alert_at_pivot', 'wait_for_new_base', 'avoid'] },
    rationale: { type: 'string' },
  },
  required: ['ticker', 'engine_status', 'agree_with_engine', 'setup_quality', 'action', 'rationale'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    pillar: { enum: ['T', 'E', 'A', 'S', 'none'] },
    fatal_issues: { type: 'array', items: { type: 'string' } },
    concerns: { type: 'array', items: { type: 'string' } },
    rationale: { type: 'string' },
  },
  required: ['refuted', 'pillar', 'fatal_issues', 'rationale'],
}

const REPORT_SCHEMA = {
  type: 'object',
  properties: { path: { type: 'string' }, headline: { type: 'string' } },
  required: ['path', 'headline'],
}

// ---------------------------------------------------------------- Mechanical
phase('Mechanical')
const mech = await agent(
  `You run the mechanical stage of a Minervini SEPA stock screen. Run commands and report exactly what they print; never recompute, round differently or adjust any number.

Work from the repository root: the directory containing the \`sepa/\` package (\`python3 -m sepa --help\` must work there).

1. ${doFetch
    ? `Download data:\n   ${fetchCmd}\n   If yfinance is missing, run \`python3 -m pip install yfinance\` once and retry. If the download fails (network, blocked host), check whether ${dataDir}/prices/ already holds CSVs for the tickers; if it does, continue with them and say so in \`notes\`; if not, return ok=false with the error text.`
    : `Do not download anything; use the CSVs already in ${dataDir}/prices/.`}
2. Run the scan:
   ${scanCmd}
   It writes ${reportsDir}/<TICKER>.json and ${reportsDir}/summary.json and prints the summary.
3. Return from summary.json: ok=true, asof, rs_universe_size, rs_note, errors (as "TICKER: message" strings), market (one line: benchmark stage label and whether its trend-template price criteria are met, or null), and for EVERY entry in summary.tickers: ticker, stage, trend_passed, trend_failed, rs_rating, pillar_trend (= pillars.trend), pillar_earnings (= pillars.earnings), entry_status, pivot, extension, below_pivot, stop, candidate.`,
  { label: 'sepa-scan', phase: 'Mechanical', schema: MECH_SCHEMA, effort: 'low' },
)

if (!mech || !mech.ok) {
  return { error: `Mechanical stage failed: ${(mech && mech.error) || 'agent returned nothing'}`, notes: mech && mech.notes }
}

const byTicker = Object.fromEntries(mech.tickers.map(t => [t.ticker, t]))
const failedTrend = mech.tickers.filter(t => !t.candidate)
log(`${mech.tickers.length} screened; ${mech.tickers.length - failedTrend.length} pass the trend pillar (Stage 2 + Trend Template)`)
if (failedTrend.length) {
  log(`Dropped by the Trend Template: ${failedTrend.map(t => `${t.ticker}(${t.trend_passed ?? '?'}/8, stage ${t.stage ?? '?'})`).join(', ')}`)
}
if (mech.rs_note) log(`RS: ${mech.rs_note} -> candidates can reach at most WATCH until RS is verified`)

// Actionable entries first, then the strongest RS.
const entryRank = { breakout: 0, near_pivot: 1, breakout_unconfirmed: 2, forming: 3, extended: 4 }
let candidates = mech.tickers
  .filter(t => t.candidate)
  .sort((a, b) => (entryRank[a.entry_status] ?? 5) - (entryRank[b.entry_status] ?? 5) || (b.rs_rating ?? 0) - (a.rs_rating ?? 0))
  .map(t => t.ticker)
if (candidates.length > maxCandidates) {
  log(`Researching the top ${maxCandidates} of ${candidates.length} candidates; not researched: ${candidates.slice(maxCandidates).join(', ')} (raise args.maxCandidates to include them)`)
  candidates = candidates.slice(0, maxCandidates)
}
if (!candidates.length) {
  return { asof: mech.asof, market: mech.market, rsNote: mech.rs_note, counts: { screened: mech.tickers.length, trendPass: 0 }, results: [], mechanical: mech.tickers }
}

// --------------------------------------------------------- Per-candidate work
const report = t => `${reportsDir}/${t}.json`

const earningsPrompt = t => `Pillar E (Earnings) of Minervini's SEPA method for ${t}, ${asofText}.

Minervini wants quarterly EPS up strongly year over year (about 20-25%+) and ACCELERATING (the latest quarter's YoY growth above the prior quarter's), revenue growing and accelerating with it, expanding profit margins, and ideally an earnings beat and raised guidance.

Start from the mechanical numbers in the \`fundamentals\` field of ${report(t)} (may be null or cover only a few quarters). Verify and complete them with WebSearch/WebFetch from the company's earnings releases, 10-Q/10-K/8-K filings and reputable financial press: the last 3-4 quarters of EPS YoY and revenue YoY growth, margin trend, the latest result vs consensus, and any guidance change. Prefer the EPS basis the company and analysts use (adjusted, if that is what consensus tracks) and say which.

verdict:
- pass: EPS YoY >= ~20% and accelerating (or a very strong, clearly accelerating turnaround), revenue accelerating or strong, margins not contracting
- fail: EPS or revenue decelerating materially, EPS growth under ~20%, or margins contracting
- insufficient: you could not verify the numbers
Never invent numbers: use null for anything you could not find, and cite every source URL.`

const catalystPrompt = t => `Pillar A (Announcement / Catalyst) of Minervini's SEPA method for ${t}, ${asofText}.

Answer: why must institutions buy this stock NOW? Minervini skips a pretty chart that has no reason for institutional demand behind it.

Research the last 3-6 months with WebSearch/WebFetch: new products, new markets, raised guidance, regulatory approvals (FDA etc.), major contracts, an industry inflection, index inclusion, M&A, new management. Also list negatives: dilutive offerings, guidance cuts, investigations, lock-up expiries, and binary events within the next ~2 weeks (earnings date, FDA decision) that make a new entry a gamble.

strength: strong = specific, recent, material driver of future earnings; moderate = real but incremental or already well known; weak/none = no identifiable driver or only hype.
verdict: pass = strong or moderate with no disqualifying negative; fail = weak/none or a disqualifying negative; insufficient = could not research.
Cite every source URL; do not speculate beyond what sources say.`

const entryPrompt = t => `Pillar S (Specific entry point) of Minervini's SEPA method for ${t}.

Read ${report(t)}: \`entry\` (the engine's VCP read: status, base.contractions, pivot, buy_range, stop, risk_pct, breakout data, warnings), \`trend\`, \`stage\`, \`rs\` and \`weekly\` (the last 30 weekly bars).

Minervini buys almost only the pivot breakout from the final contraction of a volatility contraction pattern: each pullback smaller than the one before (e.g. 25% -> 12% -> 6% -> 3%), volume drying up in the tightest area, then a close above the pivot on volume at least ~40-50% above average. He does not chase more than ~5% above the pivot, puts the stop at the low of the final contraction, wants that risk around 7-8% (never above 10%), and prefers early bases (1st-2nd base in a Stage 2 advance) over late ones.

Check the engine's mechanical read against the weekly bars. You may disagree with it, but argue only from numbers in the file; do not invent prices.
action:
- buy_now: broke out of a sound VCP on volume and is still inside the buy range, stop within ~8-10%
- alert_at_pivot: sound VCP just under the pivot; set a buy-stop at the pivot
- wait_for_new_base: extended past the buy range, or the base is still forming / too loose
- avoid: failed breakout, broken base, or stop too wide`

const LENSES = [
  {
    key: 'fundamental',
    focus: `Attack pillars E and A. Are the earnings numbers right and truly accelerating, or flattered by one-offs, easy comparisons, share buybacks or a change in EPS basis? Is the catalyst real, recent and material, or stale and priced in? Look for red flags: dilution, customer concentration, accounting questions, guidance risk, an earnings report or binary event days away. Re-check the key claims with WebSearch.`,
  },
  {
    key: 'technical',
    focus: `Attack pillars T and S using ${report('TICKER')} (weekly bars, trend, stage, entry). Is this an early Stage 2 advance or a late-stage (3rd-4th base), climactic or wide-and-loose chart? Is the VCP genuine (contractions shrinking, volume drying up) or an engine artifact? Is the entry chaseable and is the stop risk acceptable? Also weigh the market backdrop: ${mech.market || 'benchmark stage unknown'}.`,
  },
]

const skepticPrompt = (c, lens) => `You are a skeptical risk manager reviewing a Minervini SEPA buy candidate, ${c.ticker}, ${asofText}. ${lens.focus.replace('TICKER', c.ticker)}

The case for it:
- Mechanical trend: ${JSON.stringify(c.mech)}
- Earnings review: ${JSON.stringify(c.earnings)}
- Catalyst review: ${JSON.stringify(c.catalyst)}
- Entry review: ${JSON.stringify(c.entry)}

Set refuted=true only for a concrete, material flaw that would make Minervini pass on this trade, name the pillar it breaks and state it in fatal_issues. Incomplete evidence or minor worries go in concerns with refuted=false.`

function tierOf(c) {
  const E = c.earnings && c.earnings.verdict
  const Acat = c.catalyst && c.catalyst.verdict
  const S = c.entry && c.entry.action
  const refutes = (c.skeptics || []).filter(s => s && s.refuted).length
  if (E === 'fail' || Acat === 'fail' || refutes >= 2) return 'REJECT'
  const clean = c.mech.pillar_trend === 'pass' && E === 'pass' && Acat === 'pass' && refutes === 0
  if (clean && S === 'buy_now') return 'BUY'
  if (clean && S === 'alert_at_pivot') return 'READY'
  return 'WATCH'
}

const results = await pipeline(
  candidates,
  t => parallel([
    () => agent(earningsPrompt(t), { label: `earnings:${t}`, phase: 'Earnings', schema: EARN_SCHEMA }),
    () => agent(catalystPrompt(t), { label: `catalyst:${t}`, phase: 'Catalyst', schema: CATALYST_SCHEMA }),
    () => agent(entryPrompt(t), { label: `entry:${t}`, phase: 'Entry', schema: ENTRY_SCHEMA }),
  ]).then(([earnings, catalyst, entry]) => ({ ticker: t, mech: byTicker[t], earnings, catalyst, entry })),
  c => {
    if ((c.earnings && c.earnings.verdict === 'fail') || (c.catalyst && c.catalyst.verdict === 'fail')) {
      return { ...c, skeptics: [], tier: 'REJECT' }
    }
    return parallel(LENSES.map(lens => () =>
      agent(skepticPrompt(c, lens), { label: `verify:${lens.key}:${c.ticker}`, phase: 'Verify', schema: VERDICT_SCHEMA }),
    )).then(skeptics => {
      const done = { ...c, skeptics: skeptics.map((s, i) => (s ? { lens: LENSES[i].key, ...s } : null)) }
      return { ...done, tier: tierOf(done) }
    })
  },
)

const ORDER = { BUY: 0, READY: 1, WATCH: 2, REJECT: 3 }
const finished = results.filter(Boolean).sort((a, b) => ORDER[a.tier] - ORDER[b.tier])
const lost = candidates.filter(t => !finished.some(r => r.ticker === t))
if (lost.length) log(`No result (agent failed or skipped) for: ${lost.join(', ')}`)
const counts = { screened: mech.tickers.length, trendPass: mech.tickers.length - failedTrend.length, researched: finished.length }
for (const r of finished) counts[r.tier] = (counts[r.tier] || 0) + 1
log(`Tiers: ${Object.keys(ORDER).map(k => `${k} ${counts[k] || 0}`).join(', ')}`)

// -------------------------------------------------------------------- Report
phase('Report')
const written = await agent(
  `Write a Minervini SEPA screening report in Chinese (keep tickers, numbers and technical terms such as VCP, pivot, Stage 2 in English) to ${reportsDir}/sepa-report.md and return its path and a one-line headline.

Use only the data below; do not add new research or numbers.
Market backdrop: ${mech.market || 'unknown'}. As-of: ${mech.asof || 'latest'}. RS note: ${mech.rs_note || 'none'}.
Screened ${counts.screened}; ${counts.trendPass} passed the Trend Template + Stage 2 gate.
Dropped mechanically: ${JSON.stringify(failedTrend.map(t => ({ ticker: t.ticker, trend_passed: t.trend_passed, failed: t.trend_failed, stage: t.stage })))}
Not researched (cap): ${JSON.stringify(mech.tickers.filter(t => t.candidate && !candidates.includes(t.ticker)).map(t => t.ticker))}
Researched candidates (already ranked BUY > READY > WATCH > REJECT):
${JSON.stringify(finished)}

Structure:
1. 市场环境 (market backdrop and what it means for new buys)
2. 结论总表: one row per researched candidate - tier, ticker, T/E/A/S verdicts, pivot, buy range, stop, risk %
3. One section per candidate in tier order: 趋势 (T), 基本面 (E), 催化剂 (A), 入场点 (S), skeptic objections, and the concrete plan (buy now within range / buy-stop at pivot / wait for a new base / pass). Keep source links.
4. 被机械筛除 (tickers dropped by the Trend Template, with the failed criteria numbers)
5. A short disclaimer: mechanical screen plus research notes, not investment advice; verify prices before trading.`,
  { label: 'report', phase: 'Report', schema: REPORT_SCHEMA },
)

return {
  asof: mech.asof,
  market: mech.market,
  rsNote: mech.rs_note,
  counts,
  report: written && written.path,
  headline: written && written.headline,
  results: finished.map(r => ({
    ticker: r.ticker,
    tier: r.tier,
    trend: r.mech.pillar_trend,
    earnings: r.earnings && r.earnings.verdict,
    catalyst: r.catalyst && r.catalyst.verdict,
    entry: r.entry && r.entry.action,
    pivot: r.entry && r.entry.pivot,
    buyRange: r.entry ? [r.entry.buy_range_low, r.entry.buy_range_high] : null,
    stop: r.entry && r.entry.stop,
    refutedBy: (r.skeptics || []).filter(s => s && s.refuted).map(s => `${s.lens}: ${s.fatal_issues.join('; ')}`),
  })),
}

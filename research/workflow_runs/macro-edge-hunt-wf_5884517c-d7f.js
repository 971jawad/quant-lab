export const meta = {
  name: 'macro-edge-hunt',
  description: 'Hunt for macro/policy/news-driven edge from 5 independent analyst lenses, then adversarially verify anything that survives',
  phases: [
    { title: 'Discover', detail: 'data availability + 5 independent hypothesis lenses' },
    { title: 'Refute', detail: 'adversarial verification of every candidate' },
    { title: 'Synthesize', detail: 'completeness critic + final ranking' },
  ],
}

const CTX = `You are working in the quant-lab repo at "C:/Users/Ayyan/scrapper for jobs/quant-lab".
Use the Bash tool with: cd "/c/Users/Ayyan/scrapper for jobs/quant-lab" && <cmd>

CONTEXT — what has ALREADY been tested and FAILED (do not re-propose these):
- Contemporaneous macro regressions: gold~real yields (corr -0.31), indices~VIX (-0.62). LAGGED corr ~0.00.
- 735 lagged macro tests just run across 38 features x 7 markets x 3 horizons (1/5/20d):
  net liquidity (Fed BS - TGA - RRP), Fed balance sheet change, RRP, TGA, reserves,
  yield curve 2s10s & 3m10y, real yield, breakevens, 5y5y forward, term premium,
  HY/IG credit spreads, NFCI, STLFSI, VIX, US-JP carry differential, BOJ balance sheet,
  policy rate change, initial claims, payrolls, unemployment, industrial production,
  sentiment, CPI, core PCE, federal debt, dollar index.
  RESULT: 28 nominal p<0.05 vs 37 expected by chance; 0 of 735 survive Benjamini-Hochberg FDR.
- Also already dead: pre-FOMC drift (failed holdout), turn-of-month, NFP-day holds,
  FX carry, VIX term-structure gating, gold/silver ratio, session drifts, candlesticks (122 tests, 0 survive FDR),
  ICT/SMC, all intraday timeframes, ML/AI on price and on candle geometry, volume gating.
- Markets available (daily OHLC, 2010-2026): XAUUSD gold, MNQ nasdaq, ES sp500, EURUSD, USDJPY, WTIUSD oil, JPXJPY nikkei.
- Data on disk: data/external/macro/*.csv (31 FRED series), data/external/cot/ (CFTC 1999-2026),
  data/external/fomc_dates.csv, data/daily/*.csv (price), data/external/etf_*.csv.
- Discipline that ANY proposal must satisfy: no lookahead (respect publication lags —
  see qlab/macro.py PUB_LAG), dev window only (< 2022-07-01), t-stats at the signal's
  NATIVE frequency, overlapping-return t deflated by sqrt(horizon), costs applied,
  and it must survive multiple-testing correction across everything tried.

Be concrete and skeptical. A hypothesis is only worth proposing if it is (a) mechanically
testable with data that is actually free and obtainable, (b) NOT a restatement of something
above, and (c) has a real economic reason to predict rather than merely explain.`

const HYPO_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    hypotheses: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          claim: { type: 'string', description: 'the precise predictive claim' },
          mechanism: { type: 'string', description: 'why it should predict, economically' },
          data_needed: { type: 'string' },
          data_is_free_and_available: { type: 'boolean' },
          how_to_test: { type: 'string' },
          novelty_vs_tested: { type: 'string', description: 'why this is not already covered' },
          prior_confidence: { type: 'number', description: '0-1 that it survives FDR' },
        },
        required: ['name', 'claim', 'mechanism', 'data_needed', 'data_is_free_and_available', 'how_to_test', 'novelty_vs_tested', 'prior_confidence'],
      },
    },
  },
  required: ['lens', 'hypotheses'],
}

const LENSES = [
  { key: 'flows', prompt: `LENS: FLOWS AND PLUMBING. Think about mechanical money movement that must happen regardless of anyone's opinion: Treasury buyback operations and their announcements, Treasury refunding announcements (QRA) and coupon-vs-bill issuance mix, TGA rebuild/drawdown episodes, debt-ceiling resolutions, MOVE/collateral scarcity, primary dealer positioning, index rebalance dates, options expiry / gamma, quarter-end and year-end funding. Which of these creates a FORCED, PREDICTABLE flow that a daily-horizon strategy could front-run?` },
  { key: 'japan', prompt: `LENS: JAPAN AND THE CARRY TRADE. BOJ policy meeting dates and their surprises, yield-curve-control band changes, JGB yield moves, MoF FX intervention episodes, Japanese fiscal-year-end (March 31) repatriation, GPIF rebalancing, the 2024 carry unwind. What is mechanically testable about carry-trade stress and its spillover into global risk assets, given we have USDJPY, JPXJPY, and US/JP rates?` },
  { key: 'nonlinear', prompt: `LENS: NON-LINEARITY AND REGIME INTERACTION. Everything tested so far was linear (rank correlation, tertile splits, sign of a z-score). Propose hypotheses where the macro variable only predicts CONDITIONALLY: at extremes, at second-derivative inflections, in interaction with another variable, after a threshold crossing, or asymmetrically (works on the downside only). Be specific about the functional form.` },
  { key: 'news', prompt: `LENS: NEWS, ANNOUNCEMENTS AND SURPRISE. Distinguish the DATA from the SURPRISE. Which economic releases have a free, obtainable consensus/expectation series so a surprise can be computed? What about revision direction, release-day volatility patterns, or the timing structure around releases? Also: central-bank speech calendars, credit-rating actions on sovereigns, and geopolitical event proxies. Only propose what is genuinely obtainable for free.` },
  { key: 'contrarian', prompt: `LENS: THE CONTRARIAN / OUTSIDE-THE-BOX. Everything above assumes macro predicts returns. Propose instead: macro predicting VOLATILITY, CORRELATION, or DRAWDOWN RISK rather than direction (which would be tradeable via position sizing, not entries). Macro as a REGIME FILTER for when the existing trend book should be scaled up or down. Cross-asset lead-lag. Anything genuinely unconventional that the previous 15 research cycles would have missed.` },
]

phase('Discover')
log('Fanning out 5 independent analyst lenses plus a data-availability scout')

const discovery = await parallel([
  ...LENSES.map(L => () => agent(
    `${CTX}\n\n${L.prompt}\n\nPropose 4-6 hypotheses. Be ruthless about novelty and about whether the data is genuinely free. Set prior_confidence honestly — most macro hypotheses fail, and calibration matters more than optimism.`,
    { label: `lens:${L.key}`, phase: 'Discover', schema: HYPO_SCHEMA })),
  () => agent(
    `${CTX}\n\nDATA SCOUT. Determine, by actually checking with curl/WebFetch where possible, whether these are obtainable FREE and in machine-readable form, and give the exact URL/endpoint plus the date coverage:
1. US Treasury BUYBACK operation announcements and results (treasurydirect.gov / treasury.gov APIs)
2. Treasury Quarterly Refunding Announcement (QRA) dates and issuance sizes
3. BOJ monetary policy meeting DATES and decisions (boj.or.jp)
4. ECB / BOE policy meeting dates
5. Economic-release CONSENSUS forecasts (so surprise = actual - consensus) — is ANY free source real?
6. Treasury General Account daily balance (fiscal.treasury.gov)
7. MOVE index or any free bond-vol proxy
8. Sovereign credit rating action dates
Report honestly which are genuinely free/scriptable and which are paywalled. Do not guess URLs — verify.`,
    { label: 'data-scout', phase: 'Discover', schema: {
      type: 'object',
      properties: {
        sources: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, obtainable_free: { type: 'boolean' },
          url_or_endpoint: { type: 'string' }, coverage: { type: 'string' },
          format: { type: 'string' }, verified_how: { type: 'string' }, notes: { type: 'string' },
        }, required: ['name', 'obtainable_free', 'url_or_endpoint', 'coverage', 'format', 'verified_how', 'notes'] } },
      },
      required: ['sources'],
    } }),
])

const lensResults = discovery.slice(0, LENSES.length).filter(Boolean)
const scout = discovery[LENSES.length]
const allHypos = lensResults.flatMap(r => (r.hypotheses || []).map(h => ({ ...h, lens: r.lens })))
log(`${allHypos.length} hypotheses proposed across ${lensResults.length} lenses`)

// keep only those the proposer believes are testable with free data
const viable = allHypos.filter(h => h.data_is_free_and_available)
log(`${viable.length} claim to use free, obtainable data — sending each to adversarial refutation`)

phase('Refute')
const VERDICT = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['REFUTED', 'SURVIVES'] },
    reason: { type: 'string' },
    fatal_flaw: { type: 'string', description: 'lookahead, unavailable data, already-tested, sample too small, or none' },
    realistic_prior: { type: 'number' },
  },
  required: ['verdict', 'reason', 'fatal_flaw', 'realistic_prior'],
}

const judged = await pipeline(
  viable,
  h => agent(
    `${CTX}\n\nADVERSARIAL REVIEW. Your job is to REFUTE this hypothesis, not to like it. Default to REFUTED unless it genuinely survives every check.\n\nHYPOTHESIS: ${h.name}\nCLAIM: ${h.claim}\nMECHANISM: ${h.mechanism}\nDATA: ${h.data_needed}\nTEST: ${h.how_to_test}\nCLAIMED NOVELTY: ${h.novelty_vs_tested}\n\nCheck ruthlessly, and where you can, CHECK AGAINST THE ACTUAL DATA ON DISK with Bash:\n1. LOOKAHEAD — does the test require data before it was published? (see qlab/macro.py PUB_LAG)\n2. DATA — is it genuinely free and machine-readable, with enough history (need 400+ observations in 2010-2022)?\n3. ALREADY TESTED — is it a restatement of something in the failed list, or of the 735 tests just run?\n4. SAMPLE SIZE — how many independent events would this actually generate? Under ~50 it cannot clear multiplicity.\n5. MULTIPLICITY — given ~5500 configurations already tried in this project, could this plausibly survive FDR?\n6. ECONOMIC LOGIC — is the mechanism a genuine reason to PREDICT, or just to co-move?\n\nIf you can cheaply test the core claim with data already on disk, DO IT and report the number.`,
    { label: `refute:${h.name.slice(0, 26)}`, phase: 'Refute', schema: VERDICT })
      .then(v => ({ hypothesis: h, ...(v || { verdict: 'REFUTED', reason: 'verifier failed', fatal_flaw: 'none', realistic_prior: 0 }) }))
)

const survivors = judged.filter(Boolean).filter(j => j.verdict === 'SURVIVES')
log(`${survivors.length} of ${viable.length} survived adversarial review`)

phase('Synthesize')
const critic = await agent(
  `${CTX}\n\nCOMPLETENESS CRITIC AND FINAL SYNTHESIS.\n\nHypotheses proposed: ${JSON.stringify(allHypos.map(h => ({ n: h.name, lens: h.lens, conf: h.prior_confidence })))}\n\nAdversarial verdicts: ${JSON.stringify(judged.filter(Boolean).map(j => ({ n: j.hypothesis.name, v: j.verdict, flaw: j.fatal_flaw, prior: j.realistic_prior })))}\n\nData scout findings: ${JSON.stringify(scout)}\n\nAnswer:\n1. Which SURVIVING hypotheses are worth the implementation cost, ranked, with the single most decisive test for each and the realistic probability it survives FDR?\n2. What MODALITY is still unexamined after 16 research cycles — a data type, a target variable, a time structure nobody proposed?\n3. Given that 735 macro tests produced FEWER nominal hits than chance, what is the honest prior that ANY macro-driven daily-horizon edge exists in this dataset?\n4. Is there a better USE of macro data than prediction — e.g. risk scaling or regime gating — and what is the strongest version of that idea?\nBe concise and decisive. Say clearly if the answer is "stop".`,
  { label: 'synthesis', phase: 'Synthesize', schema: {
    type: 'object',
    properties: {
      ranked_survivors: { type: 'array', items: { type: 'object', properties: {
        name: { type: 'string' }, decisive_test: { type: 'string' },
        p_survives_fdr: { type: 'number' }, worth_building: { type: 'boolean' },
      }, required: ['name', 'decisive_test', 'p_survives_fdr', 'worth_building'] } },
      unexamined_modality: { type: 'string' },
      honest_prior_macro_edge: { type: 'number' },
      better_use_of_macro: { type: 'string' },
      recommendation: { type: 'string', enum: ['BUILD', 'TEST_CHEAPLY_FIRST', 'STOP'] },
      rationale: { type: 'string' },
    },
    required: ['ranked_survivors', 'unexamined_modality', 'honest_prior_macro_edge', 'better_use_of_macro', 'recommendation', 'rationale'],
  } })

return {
  n_hypotheses: allHypos.length,
  n_viable: viable.length,
  n_survived: survivors.length,
  survivors: survivors.map(s => ({ name: s.hypothesis.name, lens: s.hypothesis.lens, claim: s.hypothesis.claim, reason: s.reason, prior: s.realistic_prior })),
  refuted: judged.filter(Boolean).filter(j => j.verdict === 'REFUTED').map(j => ({ name: j.hypothesis.name, flaw: j.fatal_flaw })),
  data_sources: scout,
  synthesis: critic,
}

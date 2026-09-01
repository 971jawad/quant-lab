# Prop-edge research program — final report

Instruments: XAUUSD (gold), MNQ (Nasdaq, NSXUSD series), EURUSD.
Timeframes: 15m, 1h, 1d. Data: 16y verified HistData 15m (resampled up).
Discipline: dev window 2010→2022-06 for ALL iteration; holdout 2022-07→2026-06
touched exactly once by a frozen survivor set; 4,272 configurations logged in
`ledger.jsonl`; walk-forward + embargo everywhere; conservative costs; prop
guardrails (3%/day cap, 5% trailing-DD tracking) in every simulation.

## Program phases and what each found

**1. Trial-and-error simulation loops (3 timeframes).** 204 models evaluated
on dev across rules/drift/ML/AI phases; two refinement rounds with declared
hypotheses (learnings.md). Timeframe gradient monotone: 1d ≫ 1h ≫ 15m — the
edge-per-trade shrinks with frequency, the cost-per-trade doesn't.
15m: 0/63 at t≥2 (best +0.35). 1h: 0/81 (best MNQ orb 1.59). 1d: 3 survivors,
all one theme.

**2. Full metrics suite.** `qlab/metrics.py` — every requested metric
(compounded return, CAGR, max DD, Calmar, vol, Sharpe, Sortino, win rate,
avg win/loss, payoff, PF, expectancy, median, 5th-pct, worst trade, recovery
time, % underwater, consecutive losses, rolling expectancy/Sharpe trend,
outlier-trimmed expectancy, deflated Sharpe).

**3. ML strategy (external factors).** Gradient boosting + cross-asset context
features (EURUSD/SPX returns & vol onto each instrument): 0/18 positive at 1h.
Earlier 75-model 15m sweep: also negative. Verdict: no ML edge net of costs.

**4. AI leg (creative).** MLP neural family with the same cross-features:
all negative (best t −0.19). The learning is consistent: at intraday cost
levels there is nothing for the models to find.

**5. World-trader research → mechanical tests.** Implemented from published
prop playbooks and 23y seasonality studies: Asia-range breakout, London-trap
fade, NY opening-range breakout, session drifts (Asia hold, Friday hold),
trend-pullback, vol-squeeze, RSI mean reversion, plus the earlier ICT/SMC
fib-zone family. Triage: session drifts are cost-sized (gross sign confirmed,
net dead); London fade dead BOTH directions; ORB weak-negative (matches the
literature's own backtests); trend-pullback on the equity index is the only
externally documented premium that also shows up here.

**6. Meta-analysis.** Deflated Sharpe with the true trial count; fragility
flags declared pre-holdout (mrev_C flagged for sign instability across risk
styles — it then collapsed on holdout exactly as flagged, t 0.15).

**7. Ensemble.** Equal-weight book of the frozen survivors' holdout returns.

**8. Synthesis** — this document.

## The one theme that survived: slow Nasdaq momentum

Frozen survivors (dev t≥2, PF≥1.15, n≥50, external economic story required):

| model | dev t | holdout n | holdout exp R | PF | Sharpe | Calmar | maxDD | t |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MNQ_1d_pull_A | 2.27 | 57 | +0.357 | 1.55 | 0.66 | 0.46 | −7.3% | 1.43 |
| MNQ_1d_pull_C | 2.20 | 105 | +0.323 | 1.59 | 0.86 | 0.47 | −3.2% | **2.26** |
| MNQ_1d_mrev_C | 2.11 | 20 | +0.047 | 1.07 | 0.07 | 0.04 | −2.4% | 0.15 |

Ensemble book, holdout only (1,137 trading days, 2022-07 → 2026-03):
CAGR 1.6% at 2.0% vol → **Sharpe 0.82, Sortino 0.85, Calmar 0.45**, max DD
−3.65%, 77% of days underwater, longest recovery 354 days, rolling 1y Sharpe
positive in 91% of windows, second-half Sharpe (0.94) > first-half (0.73) —
the edge is not deteriorating within holdout. Zero trailing-DD breaches: it
survives prop rules. At 3× notional leverage (still inside a 5% trailing
limit given the −3.65% base DD): ~5% CAGR at ~6% vol — passing-a-challenge
territory only in slow motion.

Independent corroboration: the pre-existing lowfreq study (different
construction — vol-targeted daily TSMOM, no event stops) ranks trend_NQ as
its best leg (t 2.44, DSR 0.914) and its diversified momentum book at DSR
0.943. Two constructions, one premium: **slow equity-index momentum**.

## The honest statistics

- Deflated Sharpe, holdout-only frame (dev = selection, holdout = one-shot
  confirmation of 4 models): **DSR = 0.78** — a 78% probability the true
  Sharpe exceeds zero. Suggestive; below the 0.95 bar.
- Deflated Sharpe, ledger-wide frame (all 4,272 configs as attempts):
  DSR = 0.02 — under maximum skepticism this is what the luckiest of 4,272
  draws looks like.
- Trade-level caveats: holdout expectancy is lumpy (pull_C thirds: +0.01,
  +0.94, +0.02 — the 2023-24 Nasdaq rally carried it) and outlier-dependent
  (pull_A exp falls 0.36→0.10 without its top 5 trades; pull_C 0.32→0.24).

## Verdict

- **No intraday edge exists in this data at retail/prop cost levels.** Not in
  rules, sessions, ICT, ML, or AI. 15m/1h directional trading is structurally
  cost-dominated: this is the third independent program (75-model sweep,
  ICT family, this one) to reach the same conclusion.
- **One defensible, thin, slow edge: long-biased Nasdaq daily momentum**
  (trend-pullback entries or vol-targeted TSMOM), Sharpe ≈ 0.6–0.9 net,
  Calmar ≈ 0.45, DSR 0.78–0.94 depending on construction and frame. It is
  real-ish, small, lumpy, and requires years — not a challenge-passing
  intraday machine, and not gold or EURUSD: gold's version (trend_XAUUSD,
  DSR 0.889) is adjacent but weaker; EURUSD has nothing.
- The honest way to trade it under prop rules is small size, months-long
  horizon, and acceptance of ~77% of days spent under water.

**Nothing here is trading advice. Past (even out-of-sample) performance does
not guarantee future results.**

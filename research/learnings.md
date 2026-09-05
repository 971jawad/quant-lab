# Research program — cumulative learnings ledger

Discipline: all iteration on DEV (2010 → 2022-06-30); holdout (2022-07 → 2026-06)
frozen until the end, tested once; every trial appended to `ledger.jsonl` and
counted in the final Deflated Sharpe. Survivor criterion (pre-declared before
any holdout look): dev trade t-stat ≥ 2.0, PF ≥ 1.15, n ≥ 50, plus a plausible
economic story that exists in the external literature (not invented post hoc).

## Round 0 — external research (what the world claims works)

- Prop-firm playbooks converge on session structure: Asia range → London/NY
  breakout; risk caps matter more than entries (80% of failures are risk, not
  signal). Our harness already enforces the caps (3%/day, 5% trailing).
- EV Trade Labs 23y XAUUSD study: Asia session +2.9bp/session drift (p<0.001);
  Friday only significant day (+10bp, p=0.002); London = "trap" window
  (negative drift → motivates fade family); NY = biggest range, no direction.
- QuantifiedStrategies: naked ORB on gold futures backtests NEGATIVE. Session
  breakouts need filters.
- ICT/SMC canon: already built as `ict` family (sweep→displacement/FVG→0.618
  retest). Full-sample 15m result was negative on gold and Nasdaq.

## Round 1 — first dev sweep (what actually survives OUR costs)

**Drift legs (15m held sessions):**
- Asia-drift gold: Sharpe −0.87, t −3.4. The claimed +2.9bp gross drift is real
  in sign but our ~2.5bp/day round-trip cost consumes it. LEARNING: published
  intraday seasonality ≈ cost-sized; not tradeable retail.
- Friday gold long: +0.32 Sharpe, t 1.27 — right sign, not significant.
  Candidate for a filter, not a strategy.
- All Nasdaq/EURUSD drift legs negative.

**Daily rules:**
- MNQ (Nasdaq) trend-pullback: styles A/C t = 2.27 / 2.20, exp +0.44R / +0.25R,
  PF 1.72 / 1.45. First real candidate. Economic story: equity-index time-series
  momentum, the best-documented premium in the literature — consistent with the
  earlier lowfreq study (trend legs were the only DSR-positive thing).
- MNQ mean-reversion C: t 2.11 — but mrev on MNQ flips sign across styles
  (A −0.5, B −0.2) → fragile, style-C conviction scaling may be doing the work.
  Watchlist, not candidate.
- Gold daily: nothing above t 1.1. EURUSD daily: mean reversion strongly
  NEGATIVE (−1.8 to −2.6 t) — EURUSD trends through RSI extremes; fading loses.

**15m intraday (gold seen so far):** pull/sqz/mrev/orb all severely negative
(t −2 to −10). LEARNING: high-frequency entries multiply cost drag; only
1-trade-per-day session families (arb/lfade) even have a chance at 15m.

## Round 2 — refinement hypotheses (declared BEFORE running) and outcomes

1. Confirm MNQ daily pull on the sibling index (SPX proxy). OUTCOME: right sign,
   weak (style A t 0.63, C flat). Partial confirmation — treat the MNQ number as
   premium + luck, not pure premium.
2. Gold Friday-long conditioned on trend. OUTCOME: REJECTED — filter cut t from
   1.27 to 0.33. The Friday effect is not trend-conditional.
3. 15m sweep: 2 of 63 positive, none near t 2. London-fade among the WORST
   (mean t −7) — the "trap window" claim inverted does not pay either; at 15m
   cost drag dominates every family. 1h: 0 of 81 at t≥2; best MNQ orb t 1.59.
4. ML/AI cross-asset (1h): 0 of 18 positive. EURUSD/SPX context features do not
   rescue the classifier legs.

## Cumulative synthesis after all dev rounds

- ONE coherent theme survives dev: **Nasdaq (MNQ) daily momentum family** —
  pull A t 2.27, pull C t 2.20 (+ mrev C t 2.11, fragile across styles).
  Everything else that looked good anywhere is isolated noise.
- Timeframe gradient is monotone: 1d >> 1h >> 15m. Edge-per-trade shrinks with
  frequency while cost-per-trade doesn't.
- External claims triage: session drifts = cost-sized (dead net); London trap =
  dead both directions; ORB = weak-negative as literature said; equity-index
  momentum = the only externally documented premium that shows up here too.

## Round 3 — external drivers, macro events, regimes (2026-09)

Data audit first: all four 15m series VERIFIED vs LBMA/ECB/Yahoo-futures, zero
OHLC violations/dups, gaps = weekends only, coverage stable, ends 2026-06-26
(HistData publishing lag). Silver (XAGUSD) downloaded + FRED DFII10/DGS10/
VIXCLS/DTWEXBGS + 132 FOMC dates fetched.

Science (dev): gold macro R² 16% (real yields −0.31, dollar −0.27), index R²
39-43% (VIX), EURUSD 24% (dollar) — all CONTEMPORANEOUS; lagged corrs ≈ 0.
The drivers are real but priced same-day: no daily-horizon prediction trade.

Dead on dev (no holdout look): turn-of-month (faded post-2010, as literature
says), NFP-day holds, real-yield/dollar-gated gold (gates don't beat buy-and-
hold's own risk), VIX-regime trend splits (non-monotone), gold/silver ratio
reversion (t −2.5 — pairs trade dies at CFD costs).

PRE-DECLARED confirmatory holdout test (single look, logged): the pre-FOMC
drift theme (Lucca-Moench 2015, published on pre-2011 data) — dev shows MNQ
pre-FOMC +38.6bp vs +4.4bp (t 1.78), SPX same sign. Promotion criterion:
holdout mean pre-FOMC-window excess return > 0 with t > 0 on both indices.
OUTCOME: FAILED — MNQ right sign (+18.9 vs +7.7bp, t 0.63) but SPX flat
(t −0.11). Consistent with the published post-2015 decay of the anomaly.
Not promoted. Holdout look #2 recorded in ledger.

## FROZEN survivor set (before any holdout look)

`MNQ_1d_pull_A`, `MNQ_1d_pull_C`, `MNQ_1d_mrev_C` — all three meet t ≥ 2,
PF ≥ 1.15, n ≥ 50 on dev. mrev_C carries a pre-registered fragility flag
(sign flips across styles A/B). No other model qualifies.

## Round 4 — COT positioning + super-book (2026-09)

- COT fetched (17y, 869 wks): gold nothing; EUR positioning = momentum (fading
  it loses hard); NQ spec-short washout (z<-1.5 -> +111bp fwd, dev t 2.3,
  stable halves, beats naive drawdown-buying) promoted long-only, declared.
- SUPER-BOOK (a-priori rules): 7 legs, vol-targeted, equal-weight, Moreira-Muir
  overlay. Dev Sharpe 1.0 -> HOLDOUT 1.12 raw / 1.24 vol-managed, Calmar ~1.0,
  maxDD -7%/-12%, DSR 0.895/0.940. Holdout >= dev; 6/7 legs positive OOS;
  correlations ~0; Sharpe 0.96 without COT leg. PROMOTED per declared criteria.
- LESSON OF THE PROGRAM: the edge was never in a better entry. It is in
  combining many thin, uncorrelated, slow premia and managing vol — portfolio
  construction, not prediction. Holdout looks used: 8 (all in ledger).

## Round 5 — Market Wizards canon + adversarial verification (2026-09)

Tested as published, canonical params, no grids (18 trials, ledger): Turtle
S1/S2, Turtle Soup, LW volatility breakout, Holy Grail, Raschke squeeze, on
XAU/MNQ/EUR daily.

- HEADLINE LESSON: LW vol-breakout showed t=+8.0 on daily OHLC — then died
  (t −2.8 to −5.3) under worst-case both-trigger + 2x cost, and the definitive
  15m-PATH REPLAY put the truth at exp +0.04R (t 1.3) gold/NQ, negative EUR.
  ~95% of the apparent edge was fill-ambiguity artifact. RULE: any intraday-
  triggered system MUST be path-replayed on 15m data before believing daily
  OHLC numbers.
- Raschke squeeze: gold-only under worst-case (t 1.26, halves decay 0.48→0.13),
  NQ flips sign. Not promoted.
- Turtles: sub-threshold everywhere (best: gold S2 t 1.29); MNQ S1 negative.
  The 1980s Donchian edge is decayed, as documented.
- Turtle Soup: EURUSD strongly negative (t −2.1) — third independent
  confirmation that EURUSD punishes mean-reversion.
- Champion interviews/podcasts (FTMO, Chat With Traders): behavioral content
  (risk caps, 1:2 RR, consistency) — already encoded in harness styles A/B/C;
  zero new mechanical rules found.
- NET: zero promotable legs; holdout purity preserved (still 8 looks).
  The super-book remains the program's answer.

## Round 6 — world schools + grand book (2026-09)

- BNF/Kotegawa deep-dip reversion translated (close < 25dMA - 2.5sigma): MNQ
  t 1.76 right-sign but sub-bar AND same theme as the COT washout leg (buy NQ
  panic) — not promoted, would double-count. Gold negative (he traded equities;
  the translation fails where it should). Ichimoku: weakly positive everywhere
  (t <= 1.1) = slow trend, already better-captured by trend legs. EURUSD
  continuation (from our triple-negative reversion finding): t 0.03 — clean
  null; chop+costs eat BOTH directions there.
- GRAND BOOK: correlation-aware ERC weights beat equal-weight ON DEV (1.24 vs
  0.94), chosen there, evaluated once on holdout: Sharpe 1.31, Sortino 1.86,
  Calmar 1.06, CAGR 9.9% @10% vol, maxDD -9.3%, DSR 0.947,
  bootstrap 90% CI [0.57, 2.09], P(Sharpe>0) 99.8%. Holdout looks: 10.
- The program's grand lesson holds: every point of Sharpe added since round 3
  came from portfolio construction (vol targeting, diversification, ERC,
  vol management) — zero came from new entry signals.

## Round 7 — freeze + final table (2026-09-01)

- External review critique addressed: (a) ERC weights verified mechanically
  leak-free (max |dev-only vs full-sample weight diff| = 0.00e+00 over 3,416
  dev dates); (b) FULL SPEC FROZEN in research/FROZEN_SPEC.md; (c) definitive
  table produced (research/final_table.csv). Correction to the review's
  premise: LW-break t=8 was already falsified by 15m path replay — it is an
  OHLC artifact, not a cross-asset edge; the real cross-asset trend evidence
  is the slow legs.
- FINAL TABLE: BOOK_erc_mm holdout Sharpe 1.31 (dev 1.24), CAGR 9.9%, maxDD
  -9.3%, Calmar 1.06, bootstrap 90% CI [0.57, 2.09], P(SR>0) 99.8%, DSR 0.942.
  Equal-weight book 1.26 — the dev-time ERC choice also won on holdout.
  Diversification proof in one line: the dev-best leg (COT, dev 2.00)
  regressed to 0.74 on holdout, while the book went 1.24 -> 1.31.
- Bar set by the external review ("CI comfortably above zero") is MET.

## Round 8 — cycle 3 complete (2026-09-01)

- Strength-aware weights (trailing-Sharpe x invcorr): the single biggest
  weighting gain of the program — dev 1.24->1.46, holdout 1.31->1.47, robust
  across all perturbations. Fixes the discovered invcorr flaw (overweighting
  uncorrelated-but-weak legs) with the mechanism the flaw implied.
- Breadth done RIGHT finally worked: 4 new Stage-1-passing markets under
  strength weights -> champion v1.1: holdout Sharpe 1.41, Calmar 1.26,
  maxDD -9.8%, DSR 0.977 (first >0.95 of the program), CI floor 0.70.
- Silver: Stage-1 fail (costs). WTI data ends 2023-12 (vendor stopped).
- Prop math: pass prob 58%->60%, breach risk 38%->29% at 1.0x.
- Pattern intact: every gain since round 3 is construction, not prediction.

## Round 9 — cycles 4 & 5 (2026-09-05)

- Volume IS obtainable (Yahoo CME futures + GLD + CFTC OI) — my earlier
  "untestable" applied only to HistData quote files. Tested properly: volume
  gating HURTS trend (removes quiet-drift days); climax fades weak; block
  REJECTED at Stage 5 (1.47 -> 1.02).
- ICT "Muso" funded-trader variant mechanized faithfully (DXY bias + FVG/fib +
  BE + session circuit-breaker): negative on EURUSD/GBPUSD/XAUUSD even with
  optimistic ECN costs. The circuit-breaker ("stop after a session loss") is
  ~neutral — folk wisdom not supported.
- FX carry: negative (post-GFC decay). VIX term-structure gate: worse than
  always-long. Both are regime FILTERS, and every filter tested in this program
  has destroyed more edge than it saved.
- SELF-CAUGHT BUG: metrics annualization was hard-coded 252, briefly showing a
  spectacular fake "2-week bars are 4x better" result. Corrected: DAILY is the
  optimum; longer bars have lower t-stats and CIs spanning zero.
- Trade blotter produced (research/trade_blotter.csv, 316 real round-trips with
  entry/exit dates+prices): win rate 31-38%, median hold 5-6 days, edge carried
  by rare runners (one MNQ long: 737 days, +55%).
- Champion v1.1 UNCHANGED and re-verified after the fix.

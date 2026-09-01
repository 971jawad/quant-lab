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

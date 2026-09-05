# FROZEN SPECIFICATION — Grand Book v1.0 (frozen 2026-09-01)

Nothing below may change without opening a new research cycle (new dev-only
iteration, new declared criteria, holdout looks budgeted and ledgered).

## Asset universe
XAUUSD, NSXUSD (traded as MNQ), SPXUSD (traded as ES/MES), EURUSD — HistData
15m quote bars, institutionally cross-verified (LBMA/ECB/Yahoo-futures).

## Legs (7, all walk-forward leak-free; definitions in code, commit-pinned)
1-4. trend_{ES,NQ,XAUUSD,EURUSD}: daily TS-momentum, lookback {40,80,160,240}
     selected per anchored fold on training Sharpe only; vol-targeted 10%,
     60d trailing vol, leverage clip 3. (`run_lowfreq.py`)
5. xsec_ALL: weekly-horizon cross-asset momentum long/short, same fold rules.
6. MNQ_pull_C: daily trend-pullback event system, walk-forward per
   `run_research.py` TF 1d style C (grid + risk chosen in-fold only).
7. COT_NQ_washout: long NQ 1 week when net-spec positioning 2y z < −1.5;
   as-of Tuesday, applied from the Monday after Friday release.

## Costs (per side: half-spread + slippage + commission, price points)
XAUUSD 0.125+0.10+0 | MNQ 0.25+0.50+0.31 | ES 0.125+0.25+0.06 |
EURUSD 0.00003+0.00002+0.00004. Drift/weekly legs charge a full round trip
per entry. No cost parameter may be revised downward, ever.

## Portfolio construction (all a-priori rules)
- Leg vol-targeting: 10% ann, trailing 60d, leverage clip [0,3], shift(1).
- Weights: inverse-sum-|trailing-120d-correlation| (ERC proxy), normalized,
  shift(1). Chosen over equal-weight ON DEV ONLY (1.24 vs 0.94); verified
  mechanically leak-free (max |Δw| = 0 on 3,416 dev dates).
- Book overlay: Moreira-Muir vol management, 10% target, trailing 20d,
  leverage clip [0,2], shift(1).
- Rebalance: daily. No position limits beyond the leverage clips.

## Evaluation protocol
- Dev: 2010-01 → 2022-06-30. Holdout: 2022-07-01 → data end (2026-06).
- Trials ledger: research/ledger.jsonl (4,300+ configs; every future trial
  must be appended). Holdout looks to date: 11 (incl. the final table).
- Metrics: full suite (qlab/metrics.py); DSR deflated at n = holdout looks
  for holdout-frame claims; stationary block bootstrap (L=20, B=2000) CI.

## Falsified along the way (may not be resurrected without new evidence)
Intraday families at 15m/1h (all), session drifts, London fade, ORB,
ICT/SMC fib-zone, ML/AI classifiers (incl. cross-asset features), turn-of-
month, pre-FOMC (post-2015), gold/silver reversion, Turtle S1/S2, Turtle
Soup, LW vol-breakout (OHLC artifact — path-replay mandatory rule), Holy
Grail, Raschke squeeze, Ichimoku, BNF-dip (theme duplicate), EURUSD
continuation-at-extremes.

## Next-cycle hypothesis queue (dev-only until declared)
VWAP/TWAP-deviation intraday reversion (approximation only — no volume in
quote data; true volume-profile/TPO is UNTESTABLE on this dataset), energy/
bond CFD legs for diversification, paper forward-validation via run_live.py.

## Amendment 1 (2026-09-01) — admission test + weighting label

- Weighting method correctly relabeled: **inverse-correlation heuristic**
  (weight_i ∝ 1/Σ|corr_ij|), NOT true ERC. Three-way comparison ran (look #12):
  invcorr dev 1.24 / holdout 1.31 (DSR 0.937); equal 0.94/1.26; TRUE ERC
  (iterative risk-contribution, 120d cov, monthly) 1.16/0.92. Invcorr retained
  — wins both windows.
- STRATEGY ADMISSION TEST (all 5 stages required):
  1 individual: exp>0, PF>1, Sharpe>0 (dev)
  2 statistical: bootstrap CI, DSR, t-stat reported
  3 robustness: small perturbations of entry/exit/params/costs must not
    collapse the result; intraday-triggered systems MUST be 15m path-replayed
  4 diversification: corr vs existing book reported
  5 portfolio contribution (DOMINANCE-AWARE, amended after the twapmr case):
    must improve >=1 of {Sharpe, Calmar, maxDD, tail} on dev WITHOUT degrading
    any other of those by more than 10%. An uncorrelated leg with weak
    standalone expectancy is dilution, not diversification — inverse-corr
    weighting overweights exactly such legs.
- Cycle-2 verdicts: multispeed trend REJECTED (corr 0.69, dilutive; per-fold
  lookback selection already adapts speed). TWAP-MR REJECTED under the amended
  rule (Sharpe -25%, Calmar -33% for a 0.9pt DD gain). Regime gate: standalone
  book weaker than baseline; not admitted. BASELINE BOOK UNCHANGED and champion:
  holdout Sharpe 1.31, Calmar 1.06, DSR 0.937. Holdout looks: 12.

## Amendment 2 (2026-09-01) — implementation + prop sizing (cycle 3, partial)

- Rebalance rule: update allocation weights only when any |w - w_prev| > 0.02
  (dev-identical performance, 54% less allocation churn). Implementation-level
  change, applied from the next holdout look onward.
- Prop-challenge sizing (Monte Carlo, block bootstrap on the champion book,
  1y horizon, 8% target, 5% trailing DD): P(pass) peaks ≈58% at 1.0x the 10%
  vol target; leverage beyond 1x strictly reduces pass probability. Raising
  book Sharpe (breadth) is the only lever that raises the ceiling.
- Breadth block (cycle 3): silver FAILS Stage 1 individually (dev Sharpe
  −0.21; costs eat the trend) -> excluded from the block by existing spec.
  WTI/DAX/Nikkei/USDJPY pending download; block decision deferred.

## Amendment 3 (2026-09-01) — cycle 3 complete: strength weights + breadth ADMITTED

- WEIGHTS: w_i ∝ (max(trailing-756d Sharpe, 0) + 0.1) x invcorr_i, normalized,
  shift(1). Stage-3 robust (all 9 perturbations of lb/floor beat old champion
  on dev). Declared singly, not gridded.
- UNIVERSE: + WTIUSD (ends 2023-12, HistData discontinued), GRXEUR, JPXJPY,
  USDJPY trend legs (all Stage-1 pass, corr 0.07-0.25 vs book). XAGUSD
  excluded (Stage-1 fail). Champion book now 11 legs / 8 markets.
- HOLDOUT LOOK #13 (confirmation): strength_baseline 1.47/1.15/DSR 0.970;
  strength_breadth 1.41/1.26/-9.8%/DSR 0.977, CI [0.70, 2.12]. Both beat the
  old champion (1.31) OOS; no dev->holdout decay. NEW CHAMPION v1.1 =
  strength_breadth (Calmar/DD/CI-floor priority under prop constraints).
- DSR > 0.95 achieved for the first time (0.977).
- Prop MC updated: P(pass 8%/5%-trail, 1y) ~60% at 1.0x, breach risk 38%->29%.
- Holdout looks: 13. Series: research/strength_breadth_daily.csv.

## Amendment 4 (2026-09-05) — cycle 4/5: volume, ICT-Muso, carry, VIX-TS, timeframes

TOOLING BUG FOUND AND FIXED: `full_metrics` hard-coded 252-period annualization.
Harmless for every daily series (the champion and all book legs -> UNAFFECTED,
re-verified: Sharpe 1.41 / Calmar 1.26 / DSR 0.977), but it inflated any
non-daily Sharpe (2-week by 3.1x) and briefly produced a false "longer bars are
better" result. Now takes `periods_per_year`. Tell that exposed it: net Sharpe
printed ABOVE gross Sharpe, which is impossible.

- VOLUME/AUCTION (real volume sourced: Yahoo CME futures NQ/ES/CL/6E verified
  magnitudes, GLD proxy for gold since Yahoo GC=F volume is broken at 189
  contracts/day, plus CFTC OI). 13 legs. Wyckoff/VSA participation gating
  DESTROYS trend edge (4/5 voltrend legs negative) — sitting out quiet days
  removes the drift trend-following harvests. 5 legs pass Stage 1 (Sharpe
  0.1-0.3); block corr 0.28; Stage 5 REJECT (book 1.47 -> 1.02).
- ICT "Muso" variant (DXY-bias + FVG/fib + BE + session circuit-breaker),
  15m, EURUSD/GBPUSD/XAUUSD (GBPUSD downloaded). NEGATIVE everywhere:
  std cost -0.084 to -0.128 R; even at optimistic retail-ECN cost -0.035 to
  -0.058 R. Session circuit-breaker ABLATION: removing it changes expectancy by
  only -0.002 to -0.013 R, i.e. the breaker is ~neutral-to-slightly-harmful —
  post-loss trades were not worse than average.
- FX CARRY (largest documented FX factor, first test here): all three pairs
  NEGATIVE on dev (-0.22 to -0.48). Consistent with post-GFC carry decay.
- VIX TERM STRUCTURE regime gate: underperforms simply staying long
  (ES 0.47 vs 0.53; MNQ 0.61 vs 0.71). Same lesson as volume gating.
- TIMEFRAME EXTENSION past daily (2d/3d/1w/2w), correctly annualized:
  daily is the SWEET SPOT — net Sharpe 0.427 (t 1.61, bootstrap CI floor +0.01)
  vs 1w 0.303 (t 0.99) and 2w 0.228 (t 0.69). t-stats decline monotonically with
  bar size; only daily's CI clears zero. Gradient rises 15m->1d then flattens.
- Holdout looks: still 13 (nothing new was promoted; all dev-only).

## Amendment 5 (2026-09-05) — cycle 6: bonds/ETF breadth REJECTED by holdout

- Added clean ETF series (TLT, IEF, HYG, EEM, EFA, VNQ, DBC) — chosen over
  futures deliberately: Yahoo '=F' front-month carries unadjusted roll gaps
  that manufacture fake trends. Stage 1 pass: TLT .20, IEF .29, HYG .26,
  DBC .73; fail: EEM, EFA, VNQ. Block corr vs champion +0.185.
- Dev Stage 5 said ADMIT (Sharpe 1.47->1.49, Sortino 1.81->1.86, Calmar -5%,
  inside the 10% tolerance). Look #14 taken because the rule was pre-declared
  and declining a passed test is rule-shopping.
- HOLDOUT VERDICT: REJECT. +bonds 1.29 (Calmar 1.15, DSR .956); +all ETF 1.26
  (1.08, .946) vs CHAMPION 1.41 (1.26, .974). A +1.4% dev gain became -8.5%
  OOS. CHAMPION v1.1 STANDS UNCHANGED.
- RULE UPGRADE (learned from this, applies to all FUTURE candidates): Stage 5
  now requires a MATERIAL dev improvement — >=5% on the primary metric — not
  merely any improvement inside tolerance. Marginal dev gains are noise and
  burn holdout looks. The fixed-income hole is real in theory but adding it
  did not pay here.
- Holdout looks: 14.

## Amendment 6 (2026-09-05) — cycle 7/8: attribution, DATA CORRUPTION FIX, eras

DATA INTEGRITY FAILURE FOUND AND FIXED (the important one):
- GRXEUR ("DAX") is CONTAMINATED: HistData spliced EURO STOXX 50 levels
  (~3,000-4,400) into 2020-06-15 -> 2023-12-01 = 896 days = 21% of the series,
  then jumped back to true DAX (16,756). Produced a leg with 64% ann vol vs a
  10% target, one +222% day, kurtosis 2,836, and NEGATIVE alpha (-3.9%/yr) with
  an absurd 5.24 beta to the naive book. The corrupt window OVERLAPS the holdout.
- It slipped past cycle-3 verification because that check only counted >5% daily
  moves; 2 bad bars in 4,258 tripped nothing. Detected instead by leg-level alpha
  attribution — a diagnostic, not a data check, which is why attribution matters.
- FIX: GRXEUR leg REMOVED. CHAMPION v1.2 = 10 legs / 7 markets.
  CORRECTED holdout: Sharpe 1.36 (was 1.41), Sortino 1.85, Calmar 1.07 (was
  1.26), maxDD -9.76%, CAGR 10.43%, DSR 0.942, CI90 [0.53, 2.16]. The corrupt
  leg had been FLATTERING the published numbers. v1.2 is the honest champion.
- NEW PERMANENT CHECK: level-continuity (quarterly median ratio outside
  [0.60, 1.67]) added to run_breadth.verify. Rescan of all 10 series: only
  GRXEUR contaminated (2 breaks). WTIUSD flags 1 break = the REAL 2020 oil
  crash ($51->$30) -> known false positive; commodity crashes need human review.

ATTRIBUTION (cycle 7) — what we are actually paid for:
- vs THE DUMB BENCHMARK (naive 12m TSMOM, equal weight, zero fitting):
  champion dev 1.47 vs 0.44; holdout 1.41 vs 0.71. The machinery earns its keep.
- FACTOR REGRESSION (equity/duration/commodity/naive-trend): holdout ALPHA
  +7.42%/yr, t = +2.87 SIGNIFICANT, R^2 45.6%; dev alpha +7.16%, t +5.29.
  Alpha is stable across windows. Naive-trend beta 0.41 -> ~40% of returns are
  generic trend beta anyone can buy cheaply; the remainder is genuine.
- CRISIS: on the 20 worst equity days (mean -2.97%) the book returns -0.03%
  -> effectively neutral, not a hedge but not a hidden long either.
- LEG-LEVEL ALPHA (orthogonal to naive trend): COT_NQ_washout +9.4%/yr t 6.28
  (beta -0.01, R^2 0%) and MNQ_pull_C +4.5% t 3.14 (beta 0.01) are the TRUE
  alpha legs; trend_NQ t 2.13 and trend_XAUUSD t 2.05 genuine; ES/xsec/WTI/JPX
  are pure trend beta; EURUSD/USDJPY legs are noise. FUTURE RESEARCH SHOULD
  TARGET THE ORTHOGONAL EVENT/POSITIONING FAMILY, not more trend variants.
- Holdout looks: 14 (attribution + the DAX fix are diagnostics/data repair,
  not strategy searches).

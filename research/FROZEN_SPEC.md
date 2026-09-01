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

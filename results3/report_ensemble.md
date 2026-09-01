# RULE + ML + AI ensemble — walk-forward, meta-analysis, live paper

All performance is out-of-sample (walk-forward test windows the models never saw during selection or fitting). Costs, next-bar-open fills, and the prop-firm daily/trailing guardrails are the same as the base study.

## The new AI (neural-net) leg — OOS by instrument/style

| model       |   n_trades |   win_rate |   avg_R |   t_stat |   profit_factor |   sharpe |   calmar |   max_dd_pct |   total_return_pct |
|:------------|-----------:|-----------:|--------:|---------:|----------------:|---------:|---------:|-------------:|-------------------:|
| MES_ai_A    |       7534 |     0.313  | -0.132  |    -7.43 |           0.822 |    -1.87 |    -0.4  |       -99.96 |             -99.96 |
| MES_ai_B    |       5109 |     0.2161 | -0.1481 |    -5.53 |           0.832 |    -1.29 |    -0.26 |       -99.08 |             -99    |
| MES_ai_C    |       9745 |     0.3274 | -0.1364 |    -9.3  |           0.822 |    -2.01 |    -0.24 |       -98.55 |             -98.55 |
| ES_ai_A     |       7534 |     0.3155 | -0.1032 |    -5.81 |           0.857 |    -1.47 |    -0.34 |       -99.82 |             -99.82 |
| ES_ai_B     |       5109 |     0.2161 | -0.1216 |    -4.54 |           0.859 |    -1.15 |    -0.28 |       -99.47 |             -99.34 |
| ES_ai_C     |       9747 |     0.3279 | -0.1061 |    -7.26 |           0.858 |    -1.55 |    -0.16 |       -91    |             -90.99 |
| MNQ_ai_A    |       7508 |     0.3229 | -0.0738 |    -4.14 |           0.895 |    -1.06 |    -0.26 |       -99.47 |             -99.05 |
| MNQ_ai_B    |       5514 |     0.2484 | -0.0193 |    -0.75 |           0.976 |    -0.29 |    -0.08 |       -83.67 |             -66.57 |
| MNQ_ai_C    |       9859 |     0.3338 | -0.0889 |    -6.15 |           0.879 |    -1.5  |    -0.17 |       -93.76 |             -93.11 |
| XAUUSD_ai_A |       8477 |     0.3133 | -0.0922 |    -5.56 |           0.869 |    -1.35 |    -0.32 |       -99.88 |             -99.83 |
| XAUUSD_ai_B |       5983 |     0.2359 | -0.0723 |    -2.99 |           0.911 |    -0.34 |    -0.11 |       -89.58 |             -83.09 |
| XAUUSD_ai_C |      10657 |     0.3432 | -0.1209 |    -9.36 |           0.829 |    -1.98 |    -0.22 |       -98.25 |             -98.14 |
| EURUSD_ai_A |       9299 |     0.3031 | -0.1447 |    -9.03 |           0.807 |    -2.15 |    -0.47 |      -100    |            -100    |
| EURUSD_ai_B |       6538 |     0.2453 | -0.1327 |    -6    |           0.841 |    -1.18 |    -0.23 |       -99.13 |             -98.85 |
| EURUSD_ai_C |      11665 |     0.3322 | -0.1554 |   -12.17 |           0.792 |    -2.8  |    -0.27 |       -99.49 |             -99.47 |

## Ensemble — selective (cash when no approach has positive trailing edge) vs always-on, per instrument and the global book

| scope   | mode      |   n_taken |   win_rate |   avg_R |   t_stat |   profit_factor |   sharpe |   calmar |   max_dd_pct |   total_return_pct |
|:--------|:----------|----------:|-----------:|--------:|---------:|----------------:|---------:|---------:|-------------:|-------------------:|
| MES     | selective |        87 |     0.2989 | -0.2663 |    -1.96 |           0.648 |    -0.69 |    -0.12 |        -4.96 |              -4.96 |
| MES     | always_on |      9342 |     0.2266 | -0.1098 |    -5.57 |           0.873 |    -1.41 |    -0.13 |       -83.72 |             -81.69 |
| ES      | selective |       206 |     0.3641 | -0.1281 |    -1.47 |           0.81  |    -0.38 |    -0.07 |        -6.95 |              -5.39 |
| ES      | always_on |      9846 |     0.2394 | -0.0941 |    -5.06 |           0.886 |    -1.38 |    -0.14 |       -87.74 |             -85.65 |
| MNQ     | selective |        97 |     0.299  | -0.2237 |    -1.56 |           0.721 |    -0.81 |    -0.28 |       -13.26 |              -8.51 |
| MNQ     | always_on |      9559 |     0.251  |  0.0054 |     0.27 |           1.007 |    -0.27 |    -0.04 |       -60.2  |             -31.58 |
| XAUUSD  | selective |        58 |     0.2069 | -0.3307 |    -1.66 |           0.612 |    -0.34 |    -0.08 |       -11.13 |             -11.13 |
| XAUUSD  | always_on |     11528 |     0.2375 | -0.0735 |    -4.25 |           0.909 |    -0.76 |    -0.12 |       -82.64 |             -79.25 |
| EURUSD  | selective |       114 |     0.3947 | -0.0731 |    -0.62 |           0.884 |    -0.22 |    -0.04 |        -5.94 |              -3.75 |
| EURUSD  | always_on |     12760 |     0.2468 | -0.1487 |    -9.52 |           0.821 |    -1.91 |    -0.18 |       -95.38 |             -94.97 |
| GLOBAL  | selective |       562 |     0.3327 | -0.1758 |    -3.17 |           0.755 |    -1.89 |    -0.54 |       -30.75 |             -29.68 |

*Selective* re-picks each approach's best variant every fold on trailing (already-closed) trades only, keeps those with positive trailing avg-R, inverse-variance weights them, and holds cash otherwise. This is the only honest lever an ensemble of weak legs owns: choosing when to stand aside.

## Meta-analysis — is any of it genuine edge?

- **Universe searched**: 115 strategies over 5085 aligned days.
- **Deflated Sharpe** (best = `LF_clue_gold_trend`): annual Sharpe 0.944 vs an expected-max-under-luck of 0.1819 across the universe. **P(true SR > 0) = 0.0** (need ≥ 0.95).
- **White's Reality Check** (best = `LF_clue_buyhold_NQ`): family-wise **p = 0.0555** (need < 0.05).
- **PBO / CSCV**: probability of backtest overfitting = **0.3571** over 252 splits.

**Verdict: NO genuine edge: best result is consistent with luck after correcting for the number of strategies tried.**

## Rule-based conditional-edge search (time / vol / session / …)

Pooled 478871 OOS trades, picked the best condition per metric on the first 60% by date, verified on the untouched last 40%.

**no condition held out-of-sample at t>=2 - the pooled edge is not hiding in these metrics (consistent with the no-edge finding).**

## Live (paper) walk-forward

`python run_live.py --scope GLOBAL --replay` replays the ensemble's most recent OOS window as a dated blotter with running equity; `--watch` books paper fills on genuinely new bars you append. **No broker, no real orders — simulation only.**

---
*Not trading advice. Out-of-sample performance does not guarantee future results.*

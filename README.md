# quant-lab

Walk-forward research framework for prop-firm futures/CFD trading:
5 strategy families x 3 risk styles x 5 instruments (MES, ES, MNQ, XAUUSD,
EURUSD), built and evaluated with strict no-lookahead discipline.

## The honest headline

**On 13-15 years of out-of-sample 15-minute data, none of the 75 model
variants has a positive edge after realistic costs.** Results on short
histories (~2.4y hourly) that looked promising did not survive the larger
sample. A pre-registered meta-labeling filter (Lopez de Prado) did not flip
any negative family positive. Details in `results2/report.md` and
`results2/meta_labeling.json`.

This repo's value is the machine, not a magic signal: verified data pipeline,
leak-free walk-forward, adversarially reviewed backtester, honest statistics.

## Extension: RULE + ML + AI ensemble + genuine-edge meta-analysis (2026)

A later build adds a fourth **AI** leg and combines all three approaches:

- `qlab/strategies.py::_nn_fit` — **AI leg**: a neural net (sklearn `MLPClassifier`)
  refit per fold through the *same* embargoed walk-forward as the tree ML leg.
  Runs as family `ai`; generate it with `python run_ai.py`.
- `qlab/ensemble.py` + `run_ensemble.py` — leak-free **ensemble**: each fold picks
  every approach's best variant on trailing (already-closed) trades only, keeps
  those with positive trailing avg-R, inverse-variance weights them, and holds
  **cash** when none qualifies. Per-instrument books + a global cross-market book.
- `run_meta_analysis.py` — **genuine-edge** stats corrected for the number of
  strategies tried: Deflated Sharpe (Bailey/LdP), White's Reality Check
  (studentized stationary bootstrap), and PBO/CSCV.
- `run_conditional.py` — **rule-based conditional-edge** search: does avg-R turn
  positive under a time/session/volatility/trend condition? Best bucket picked on
  the first 60% by date, verified on the untouched last 40%, plus BH-FDR.
- `run_live.py` — **walk-forward live, PAPER ONLY** (`--replay` / `--watch`). No
  broker, no real orders. Metrics now also report **Calmar** alongside Sharpe.
- `make_ensemble_report.py` — assembles it all into `results3/report_ensemble.md`.

Reproduce: `python run_ai.py && python run_ensemble.py && python run_meta_analysis.py
&& python run_conditional.py && python make_ensemble_report.py`.

## Extension: multi-timeframe prop-edge research program (2026-09)

`run_research.py` + `run_final_holdout.py`: 204 models (session breakouts,
London fade, ORB, trend-pullback, squeeze, mean reversion, ICT fib-zone,
session/day-of-week drifts, ML/AI with cross-asset features) × {XAUUSD, MNQ,
EURUSD} × {15m, 1h, 1d}, iterated on a dev window (2010→2022-06) with a
trials ledger (4,272 configs) and tested ONCE on a frozen holdout
(2022-07→2026-06). Full metrics suite in `qlab/metrics.py`.

**Result** (`research/final_report.md`): no intraday edge at realistic costs —
again. One thin slow edge survived: **Nasdaq daily trend-pullback momentum**,
holdout Sharpe 0.82 / Calmar 0.45 / max DD −3.7% (ensemble book), DSR 0.78
holdout-frame (0.94 for the corroborating lowfreq momentum book). Real-ish,
small, lumpy; not an intraday challenge machine. The `ict` family
(sweep→displacement→0.618 retest, session liquidity pools, Heikin-Ashi and
break-even/structural-TP variants) is negative over 16y on gold and Nasdaq
(`results_ict/report.md`).

## Layout

- `qlab/` - engine: features, strategies (SMC/ML/TA), backtester, walk-forward
- `run_data.py` / `run_data_15m.py` - data download + multi-institution
  cross-verification (Yahoo, HistData vs Nasdaq/LBMA/ECB)
- `run_all.py --profile {1h,15m}` - full model sweep -> results*/, weights*/
- `run_meta.py` - pre-registered meta-labeling stage-2 filter
- `run_risk_matrix.py` - prop-firm contract sizing tables
- `weights/`, `weights2/` - deployment manifests + fitted models (hourly / 15m)
- `results/`, `results2/` - OOS metrics, fold logs, reports

## Reproduce

```
pip install pandas numpy scikit-learn yfinance requests tabulate
python run_data.py          # hourly data + verification
python run_data_15m.py      # 16y of minute data from HistData (~15 min)
python run_all.py --profile 15m
python run_meta.py
python make_report.py results2
```

## Execution assumptions

Signal at bar close -> market entry next bar open; stop-before-target intrabar;
spread + slippage per side + commissions; 3% daily loss cap; 5% trailing-DD
tracking. See `weights/README.md` for the full deployment contract.

**Nothing here is trading advice. Past (even out-of-sample) performance does
not guarantee future results.**

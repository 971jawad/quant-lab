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

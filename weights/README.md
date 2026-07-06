# Deployment weight files

One JSON manifest per model: `{INSTRUMENT}_{STRATEGY}_{STYLE}.json`
(ML models additionally ship a `.pkl` with the fitted scikit-learn classifier).

## Naming

- **Instruments**: MES, ES, MNQ, XAUUSD, EURUSD
- **Strategies**
  - `smc` — ICT/SMC synthesis: previous-day liquidity sweep + displacement +
    fair-value-gap confluence, optional killzone/HTF-bias filters
  - `ml` — GradientBoosting classifier, uniform sample weights
  - `ml_err` — same, but each walk-forward refit up-weights the samples the
    *previous* model got wrong ("learn from every mistake")
  - `ml_rec` — same, with exponential recency weighting (~5-month half-life)
  - `ta` — classic TA vote ensemble: trend + breakout + RSI extreme +
    engulfing-at-structure + pin/rejection bar
- **Styles**
  - `A` — fixed 0.75% risk per trade, 3:1 reward:risk, one position at a time
  - `B` — risk% and R:R selected per fold by in-sample grid search
  - `C` — up to 3 concurrent positions, conviction-scaled risk

## Manifest fields

- `config.params` — strategy parameters chosen on the full training history
- `config.rr / time_exit / max_concurrent / risk_pct` — execution + risk config
- `costs` — spread / slippage / commission (price points) assumed per side
- `walkforward_oos` — **the honest numbers**: out-of-sample metrics from the
  anchored walk-forward. If `t_stat` is below ~2, the edge is NOT statistically
  significant — treat the model as unproven regardless of the return number.
- `risk_guardrails` — daily loss cap (3%) and trailing drawdown (5%) used in
  simulation. Tighten to your prop firm's actual rules.

## Execution contract (must match your deployment)

1. Build 1h bars in UTC; compute features exactly as `qlab/features.py`.
2. Evaluate the signal at bar close; enter at the NEXT bar's open, market order.
3. Stop-loss at `config`-derived structure/ATR level; target = entry + RR x stop
   distance; time-exit after `time_exit` bars.
4. Size = account x risk_pct / (stop distance x point value), floored to whole
   contracts (futures) or 0.01 lots (CFD/FX).
5. Halt new entries for the day when down `daily_loss_cap` from the day's start.

## Warnings (read before risking money)

- Walk-forward OOS covers ~13 months (futures) / ~21 months (FX) of hourly
  data. That is a THIN sample for hourly strategies. No model here showed
  t-stat > 2; deploy in sim/replay first, at minimum size.
- Past performance, including out-of-sample performance, does not guarantee
  future results. Regime change breaks edges without notice.
- ML `.pkl` files: load with the same scikit-learn major version (1.7.x).

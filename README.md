# quant-lab

Systematic trading research on gold, equity indices, FX and oil — built to find
out whether a tradeable edge exists, and to be honest when it doesn't.

**📊 Live dashboard: [971jawad.github.io/quant-lab](https://971jawad.github.io/quant-lab/)**

> **Research signals only.** This repository publishes what the models compute.
> It is not connected to any broker, places no orders, and is not investment
> advice. Past performance — including out-of-sample — does not guarantee
> future results.

---

## The headline

Across **4,572 configurations** in **23 research phases**, the median model
scored a Sharpe of **−0.35** and only **25% were positive**. Almost everything
the trading world believes in did not survive contact with realistic costs.

What did survive is a diversified daily book — the **ENSEMBLER**:

| Window | Sharpe | Sortino | CAGR | Max DD | Calmar |
|---|--:|--:|--:|--:|--:|
| Dev (2010 → 2022-06) | 1.47 | — | 9.17% | −8.83% | 1.04 |
| **Holdout (2022-07 → 2026)** | **1.39** | — | **10.59%** | −10.53% | 1.01 |
| Full sample | 1.43 | — | 9.60% | −10.53% | 0.91 |

It beats a naive 12-month momentum benchmark 2:1 out-of-sample and carries
**+7.4%/yr alpha (t = 2.87)** after controlling for equity, duration, commodity
and generic-trend factors.

## The honest caveats — read these

- **~40% of returns are generic trend beta** available cheaply elsewhere.
- The Nasdaq positioning leg **fails backward validation** on 1999–2009
  (t = 0.43 Nasdaq, −1.35 S&P). It is a post-GFC regime effect, not a market
  law. Without it the book scores Sharpe **1.17**.
- Under the most conservative trial accounting (all 4,572 configs), the holdout
  Sharpe (1.36) **does not clear** the expected-maximum-luck bar (1.51). It
  clears at ~250 independent mechanisms and on the full 19-year sample.
- **PBO = 0.02** — performance is consistent across sub-periods, so this is not
  classic curve-fitting. The edge is *real but regime-dependent*.
- You will be **underwater ~88% of days** and win only **31–38%** of trades.

## What was tested and rejected

Intraday (15m/1h, all families) · ICT/SMC fib-zone · the "Muso" funded-trader
setup · ML and neural nets with cross-asset features · volume/auction gating ·
FX carry · VIX term-structure gating · Turtle & Turtle Soup · Larry Williams
volatility breakout · Holy Grail · Raschke squeeze · Ichimoku · BNF dip ·
bonds/ETF breadth · dedicated short models · pre-FOMC drift · turn-of-month ·
session drifts · gold/silver reversion · the all-schools voting Committee.

Three findings worth the whole exercise:

1. **Timeframe is monotone.** 15m: 3% of models positive. 1h: 12%. **1d: 50%.**
   Longer than daily is worse again. Cost drag dominates everything fast.
2. **The edge is long-only.** Split by side across 7 markets: long **+484%**,
   short **−108%**. Crowded shorts get squeezed; crowded longs just keep drifting.
3. **Combining everything loses.** The Committee — all eight schools voting —
   scored 0.29 versus 0.45 for trend alone. Combination belongs at the
   *portfolio* level with evidence-based weights, never at the signal level.

## Method

- **Dev / holdout split.** All iteration on 2010→2022-06. The holdout was frozen
  and every look is counted in `research/ledger.jsonl`.
- **No lookahead.** Features use trailing windows only; signals read at the daily
  close, executed at the next open; session levels use the previous completed
  session; COT positioning applies 6 days after its as-of date.
- **Conservative costs** — spread + slippage + commission on every trade, never
  revised downward.
- **Multiple-testing corrections** — Deflated Sharpe, expected-maximum-null
  Sharpe, PBO/CSCV, stationary block bootstrap.
- **A 5-stage admission test** for every candidate leg (`research/FROZEN_SPEC.md`).

Three data/methodology bugs were caught and are documented in full, two of which
had *flattered* results: a vendor series splicing Euro Stoxx 50 into 21% of the
DAX history, silent CFTC contract renames that voided a holdout test, and a
√5 t-statistic inflation from weekly→daily spreading.

## Layout

```
qlab/            engine: features, strategies, backtester, walk-forward, metrics
run_research.py  the main dev/holdout sweep across families and timeframes
run_ensembler.py THE PRODUCTION BUILD — positions, risk engine, operating rules
run_attribution.py  what are we actually paid for (vs naive benchmark + factors)
run_shorter.py   long/short attribution and dedicated short models
run_committee.py the all-schools voting ensemble ("intuition", mechanized)
run_capstone_meta.py  multiple-testing verdict over the whole program
build_site.py    generates the dashboard payload
research/        FROZEN_SPEC.md, learnings.md, ledger.jsonl, all result artifacts
docs/            the GitHub Pages dashboard
```

## Reproduce

```bash
pip install pandas numpy scikit-learn scipy yfinance requests
python run_data_15m.py        # 16y of 15-minute data + cross-verification
python run_research.py --phase rules --tf 1d
python run_ensembler.py       # positions + performance
python build_site.py          # refresh the dashboard
```

## Data

15-minute bars from HistData (2010→2026), cross-verified against **LBMA** (gold,
r = 0.93 over 4,131 days), **ECB** (EURUSD), and Yahoo futures (indices, r = 0.98).
Plus FRED macro series, CFTC Commitments of Traders back to 1999, and Yahoo
futures/ETF volume.

**Storage design.** The models are daily, so the repository commits the compact
daily OHLC series (`data/daily/`, 1.4 MB) rather than the 270 MB 15-minute
archive. Those daily bars are **bit-identical** to bars derived from the archive —
verified every run, all 34,124 bars, max difference exactly 0.0 — so nothing the
live system trades is approximated. The 15-minute archive is only needed to
re-run the *intraday* research (which was tested exhaustively and rejected) and
is regenerated locally with `python run_data_15m.py`.

Fresh prices arrive through `run_live_update.py`, which rebuilds ET-day bars from
Yahoo **hourly** data (naive daily bars break FX return alignment: EURUSD
correlation 0.07 → 0.99 after the fix) and refuses to append if the overlap
correlation degrades.

## Licence

MIT for the code. The research conclusions are offered as-is, and the most
valuable thing in this repository is the list of things that **didn't** work.

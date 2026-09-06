# SESSION STATE — resume checkpoint

Last updated: 2026-09-06. Read this first if you are picking the project up cold.

---

## 1. Where everything lives

| What | Where |
|---|---|
| Repo (own git repo) | `C:\Users\Ayyan\scrapper for jobs\quant-lab` |
| GitHub | https://github.com/971jawad/quant-lab (public, account `971jawad`, ssh) |
| Live dashboard | https://971jawad.github.io/quant-lab/ (Pages from `master:/docs`) |
| Frozen spec + all amendments | `research/FROZEN_SPEC.md` |
| Round-by-round narrative | `research/learnings.md` |
| Trials ledger (~5,500 configs) | `research/ledger.jsonl` |

Automation: `.github/workflows/refresh.yml` runs weekdays 23:00 UTC —
live prices → signals → trade details → audit → decay monitor → dashboard →
commit. Verified green end-to-end.

---

## 2. The champion (do not change without the admission protocol)

**CHAMPION v1.2** — 10 legs / 7 markets, daily bars, long-only on drift markets.

| Window | Sharpe | Calmar | Max DD | Notes |
|---|--:|--:|--:|---|
| DEV 2010→2022-06 | 1.47 | 1.04 | −8.8% | in-sample, all iteration here |
| HOLDOUT 2022-07→2026-06 | **1.39** | 1.02 | −10.5% | frozen, 17 looks |
| **LIVE 2026-06-28→now** | **−2.25** | — | −3.8% | 60 days, −2.19% |

- Production build: `run_ensembler.py`. Series: `research/ensembler_daily.csv`.
- Legs: trend on XAUUSD/MNQ/ES/EURUSD/USDJPY/WTIUSD/JPXJPY (MNQ, ES, JPXJPY
  **long-only**), plus `xsec_ALL`, `MNQ_pull_C`, `COT_NQ_washout`.
- Weights: (trailing 756d Sharpe, floored at 0) × inverse-correlation, 2% drift
  rebalance, Moreira-Muir vol overlay.
- Execution: signal from close of day D → fill at open of D+1 (00:00 ET).

**Live decay monitor says WATCH, not WARNING**: −2.19% sits at the 12th
percentile of what an intact edge produces over 60 days (5th pct = −3.91%).
Needs ~2 years to separate skill from noise.

---

## 3. Hard-won methodology rules (each cost a real bug)

1. **Publication lags.** FRED timestamps are AS-OF dates, not release dates. See
   `qlab/macro.py::PUB_LAG`. Using as-of dates manufactures fake edge.
2. **Native-frequency t-stats.** Never compute t on a weekly return spread across
   5 daily rows — it inflates t by √5. Warning is in `qlab/metrics.py`.
3. **Annualization.** `full_metrics(periods_per_year=...)` — a hard-coded 252 on
   weekly bars once produced a fake "2-week bars are 4× better" result.
4. **Vendor data integrity.** HistData spliced Euro Stoxx 50 into 21% of the DAX
   series; CFTC silently renamed contracts in 2022 and voided a holdout test.
   `run_breadth.verify()` now includes a level-continuity check.
5. **Overlapping returns.** Deflate t by √horizon.
6. **Full-sample diagnostics are contaminated.** The SHORTER long/short table was
   full-sample; using it to pick a strategy (e.g. "invert EURUSD") fails — the
   sign flips between dev and holdout.

---

## 4. Everything tested and REJECTED (16 families, ~5,500 configs)

Intraday 15m/1h (all families) · ICT/SMC fib-zone · "Muso" funded-trader setup ·
ML + neural nets on price and on candle geometry · volume/auction gating ·
FX carry · VIX term-structure gating · Turtle & Turtle Soup · Larry Williams
volatility breakout (t=8 was an OHLC fill artifact) · Holy Grail · Raschke
squeeze · Ichimoku · BNF dip · bonds/ETF breadth · dedicated short models ·
pre-FOMC drift · turn-of-month · session drifts · gold/silver reversion ·
the all-schools voting Committee · **candlesticks (122 tests, 0 survive FDR)** ·
**macro/policy/liquidity (735 tests, 0 survive FDR)**.

Three structural findings worth more than any strategy:
1. **Timeframe is monotone**: 15m 3% of models positive, 1h 12%, **1d 50%**.
   Longer than daily is worse again.
2. **The edge is long-only**: long +484% vs short −108% across 7 markets.
3. **Combining everything loses**: the Committee scored 0.29 vs 0.45 for trend
   alone. Combination belongs at the portfolio level, never the signal level.

---

## 5. IN FLIGHT — the macro-edge-hunt workflow

A 5-lens hypothesis hunt + adversarial refutation was launched and may not have
been read before the session ended.

- Run ID: `wf_5884517c-d7f`
- Script: `.../workflows/scripts/macro-edge-hunt-wf_5884517c-d7f.js`
- Transcript: `.../subagents/workflows/wf_5884517c-d7f/` (see `journal.jsonl`)

**To recover its results without re-running:** read `journal.jsonl` in the
transcript dir — each line holds an agent's actual return value. To resume:
`Workflow({scriptPath: "<script path above>", resumeFromRunId: "wf_5884517c-d7f"})`
— unchanged agent calls return cached instantly.

Lenses: flows/plumbing (Treasury buybacks, QRA, TGA), Japan/carry, non-linearity
and regime interaction, news/surprise, contrarian (macro for vol/risk-sizing
rather than direction), plus a data-availability scout.

---

## 6. What to do next (in order of expected value)

1. **Read the workflow results** and act on any hypothesis that survived
   adversarial refutation. Expect most to be refuted.
2. **Forward validation** — the only untainted evidence left. Every month of
   live data is worth more than another sweep of this sample. Do not spend more
   holdout looks (17 used).
3. If adding anything: it must pass the 5-stage admission test in
   FROZEN_SPEC.md, including a **material ≥5% dev improvement** before a holdout
   look is spent.

**Do NOT**: re-run intraday research, auto-reoptimize on live data, or promote a
leg that fails its own pre-declared bar because the holdout happened to look
good. All three have been tried and are documented failures.

---

## 7. Honest bottom line

The champion is at the **boundary of statistical detectability**: PBO 0.02
(consistent, not curve-fitted) but the COT leg **fails backward validation** on
1999–2009, so the edge is real-but-regime-dependent, not a market law. Without
that leg the book scores Sharpe 1.17. Under the harshest trial accounting the
holdout Sharpe does not clear the luck bar; it clears at ~250 independent
mechanisms and on the full 19-year sample.

Sizing: pass probability for an 8%/5%-trailing challenge **peaks at 1.0×** (60.5%)
and falls above it while blow-up risk keeps climbing. On LIVE data, 1.5× and 2.0×
have **already breached** the 5% trailing limit within 60 days.

**Never wire this to a broker for automatic order execution.** The system
publishes signals only.

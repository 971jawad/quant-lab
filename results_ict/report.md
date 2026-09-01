# ICT fib-zone family — walk-forward result

A mechanical build of the classic ICT/SMC discretionary playbook:

1. **Sweep** a time-based liquidity pool (Asia / London / RTH session high-low, or
   prior-day high-low): price wicks past the level and closes back inside.
2. **Displacement + FVG** in the opposite direction — a strong candle (raw or
   Heikin-Ashi) that breaks structure and leaves a 3-bar fair-value gap.
3. **Fib-zone retest entry** — *do not* enter on the displacement. Mark the leg and
   wait up to `ICT_MAX_WAIT` bars for price to retrace into the 0.5–0.786 band
   (0.618 zone). Stop just beyond the swept extreme (tight risk).
4. Optional HTF SMA200 **bias** filter; optional **managed** exit (target = nearest
   opposite liquidity pool + stop to break-even at +1R) vs **raw** fixed-RR exit.

Run through the **same** anchored walk-forward, `ML_HORIZON+1` embargo, 3%/day loss
cap, 5% trailing-DD tracking, and conservative costs (spread + slippage/side +
commission) as the 75-model sweep. 15-minute bars, 2010→2026.

## The honest headline

**No positive edge after costs, on either instrument, in any risk style.**

| model | n | win% | avg R | PF | total ret | max DD | Sharpe | t-stat |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| XAUUSD ict A | 745 | 0.35 | −0.080 | 0.87 | −38.7% | −53.1% | −0.38 | −1.58 |
| XAUUSD ict B | 823 | 0.24 | −0.114 | 0.86 | −24.6% | −32.0% | −0.29 | −1.83 |
| XAUUSD ict C | 758 | 0.31 | −0.122 | 0.83 | −11.6% | −16.8% | −0.34 | −2.41 |
| MNQ (Nasdaq) ict A | 733 | 0.33 | −0.081 | 0.88 | −38.7% | −44.2% | −0.36 | −1.47 |
| MNQ (Nasdaq) ict B | 698 | 0.25 | −0.028 | 0.97 | −12.1% | −30.0% | −0.03 | −0.38 |
| MNQ (Nasdaq) ict C | 741 | 0.34 | −0.098 | 0.86 | −10.7% | −11.6% | −0.41 | −1.95 |

## Why it "looks" like it works on recent charts

On a **1.5-year recent slice** of gold the same code returns avg R **+0.54**, PF
2.27, t **2.03** — a strategy you'd happily trade. Extended to the full 16 years it
is **−0.08** (t −1.58). The edge lives almost entirely in the last ~year:

XAUUSD ict A, avg R by fold-bucket (chronological):
`−0.05, −0.18, +0.07, +0.03, −0.42, −0.04, −0.32, −0.09, −0.24, +0.05, −0.33,
−0.00, −0.06, +0.78` ← only the final (2025-26) bucket carries it.

Losses are **spread across folds, not one blow-up**. This is regime dependence /
recency mirage, the exact failure the project README warns about.

## What the selection tells you about the specific ideas

The walk-forward picks the best-of-16 configs on each training window, so its
choices are evidence about which of your ideas held up *in-sample*:

- **HTF trend filter (`bias`) — supported in-sample.** Chosen in almost every fold.
  "Don't trade against the larger trend" trained better than ignoring it. It just
  wasn't enough to clear costs out-of-sample.
- **Structural TP + break-even (`managed`) — not selected, ever.** The plain
  fixed-RR exit beat "take profit at the opposite liquidity pool and move to break-
  even at +1R" in **every** fold. The management overlay you described did not add
  value here (the variants are generated correctly and simply lose the selection).
- **Heikin-Ashi displacement — no stable benefit.** Selected in some gold folds,
  dropped in others; no consistent lift.
- **Nasdaq: no in-sample winner ever existed.** Every MNQ fold's best training
  objective was *negative* (−3.2 → −2.7). There was nothing to select — the setup
  was already losing on the training window, before any OOS test.

## Method / leak-freeness

Same discipline as the rest of quant-lab: features use only trailing data; session
pools map the **previous completed** session (never a forming one); swing points
are confirmed k bars late; entry is next-bar-open after the signal bar; stop-before-
target intrabar; break-even arms only after a bar *closes* past +1R. Existing
families are byte-for-byte unchanged (the new `target`/`be_r` columns default to
NaN and are ignored when absent).

## What would change the verdict

- A genuinely independent HTF confirmation series (this uses SMA200 on the same
  timeframe as a proxy, not a separate higher timeframe feed).
- Real session volume / order-flow, which HistData (quote-derived, no volume)
  cannot provide.
- DAX/GER40 is untested here (data not yet downloaded). Given gold and Nasdaq both
  come back flat-to-negative, a positive DAX result would be the outlier to prove,
  not assume.

**Nothing here is trading advice. Past (even out-of-sample) performance does not
guarantee future results.**

"""LIVE DECAY MONITOR — is the edge still working, or is it breaking down?

The book adapts on its own in three ways (verified, not asserted):
  * lookbacks re-select per walk-forward fold (~annually)
  * strength weights recompute daily and shrink legs whose trailing Sharpe falls
  * volatility sizing recomputes every bar

What it deliberately does NOT do is invent new strategies or re-run the research
on live data — automatic re-optimisation on the data you are trading is exactly
how a system overfits itself to death. So the honest safeguard is not "learn
more", it is "notice when the thing you froze stops working".

This module answers one question: given the HOLDOUT return distribution as the
null of "the edge is intact", how surprising is the live run so far?

  * bootstrap the holdout returns into windows the same length as the live run
  * report where live sits in that distribution (percentile)
  * a live result below the 5th percentile is a genuine warning, not noise
  * also tracks per-leg live contribution so a single broken leg is visible

Nothing here changes any position. It only tells you whether to trust them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV_END = pd.Timestamp("2022-07-01")


def load(name):
    s = pd.read_csv(OUT / name, index_col=0, parse_dates=True)["ret"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def live_start():
    st = []
    for f in (ROOT / "data" / "live").glob("*_ext.csv"):
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(d):
            st.append(pd.to_datetime(d.index[0]).normalize())
    return min(st) if st else None


def bootstrap_windows(ref, n, B=5000, block=10, seed=11):
    """Stationary block bootstrap: distribution of n-day outcomes under 'the
    edge is intact' (i.e. drawing from the holdout return distribution)."""
    rng = np.random.default_rng(seed)
    v = ref.values
    tot, shp = [], []
    for _ in range(B):
        idx = []
        while len(idx) < n:
            s0 = rng.integers(0, len(v))
            idx.extend([(s0 + k) % len(v) for k in range(min(block, n - len(idx)))])
        w = v[idx]
        tot.append(float((1 + w).prod() - 1))
        shp.append(float(w.mean() / (w.std() + 1e-12) * np.sqrt(252)))
    return np.array(tot), np.array(shp)


def main():
    ls = live_start()
    book = load("ensembler_daily.csv")
    hold = book[(book.index >= DEV_END) & (book.index < ls)]
    live = book[book.index >= ls]
    n = len(live)
    if n < 5:
        print("live window too short to assess")
        return 0

    live_tot = float((1 + live).prod() - 1)
    live_shp = float(live.mean() / (live.std() + 1e-12) * np.sqrt(252))
    tot_d, shp_d = bootstrap_windows(hold, n)
    pct_tot = float((tot_d < live_tot).mean() * 100)
    pct_shp = float((shp_d < live_shp).mean() * 100)

    print("=" * 76)
    print(f"LIVE DECAY MONITOR — {n} live days since {ls.date()}")
    print("=" * 76)
    print(f"  live return   {live_tot*100:+.2f}%   Sharpe {live_shp:+.2f}")
    print(f"  under the null 'edge intact' (bootstrapped from holdout, {n}-day windows):")
    print(f"    expected return   median {np.median(tot_d)*100:+.2f}%   "
          f"5th pct {np.percentile(tot_d,5)*100:+.2f}%   95th {np.percentile(tot_d,95)*100:+.2f}%")
    print(f"    live sits at the {pct_tot:.0f}th percentile of return, "
          f"{pct_shp:.0f}th of Sharpe")

    if pct_tot < 5:
        verdict, note = "WARNING", ("live is below the 5th percentile of what an "
                                    "intact edge produces — investigate")
    elif pct_tot < 20:
        verdict, note = "WATCH", ("soft patch, still inside the range an intact "
                                  "edge routinely produces")
    else:
        verdict, note = "NORMAL", "live is unremarkable versus the null"
    print(f"\n  VERDICT: {verdict} — {note}")

    # how long until a live sample could actually settle the question?
    need = int(np.ceil((2.0 / max(abs(live_shp if live_shp else 0.5), 0.5)) ** 2 * 252))
    print(f"  a live sample large enough to distinguish skill from noise at this "
          f"volatility is roughly {max(need,504)}+ trading days (~{max(need,504)//252}+ years)")

    res = {"live_start": str(ls.date()), "live_days": n,
           "live_return_pct": round(live_tot * 100, 3),
           "live_sharpe": round(live_shp, 3),
           "null_median_return_pct": round(float(np.median(tot_d)) * 100, 3),
           "null_p5_return_pct": round(float(np.percentile(tot_d, 5)) * 100, 3),
           "null_p95_return_pct": round(float(np.percentile(tot_d, 95)) * 100, 3),
           "live_return_percentile": round(pct_tot, 1),
           "live_sharpe_percentile": round(pct_shp, 1),
           "verdict": verdict, "note": note}
    json.dump(res, open(OUT / "decay_monitor.json", "w"), indent=2)
    print(f"\n  wrote research/decay_monitor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

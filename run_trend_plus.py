"""Disciplined improvement sweep on the one promising leg (daily trend).

Rule of the house: 'improving' a backtest by trial-and-error is how fake edge is
born. So every variant here is (a) economically motivated - not a parameter
fish, (b) evaluated on the SAME walk-forward OOS returns, and (c) charged to a
TRIAL LEDGER whose size deflates the final Sharpe. The more we try, the higher
the bar the winner must clear. A variant that only wins by a hair after
deflation did not really win.

Enhancements (each a known trend-following refinement, not a knob):
  V0 base_blend  - average the sign across ALL lookbacks (no per-fold selection,
                   so no selection-overfit). The robust baseline.
  V1 tanh        - smooth response: position ~ tanh(k * z-score of trend) instead
                   of a binary flip -> less whipsaw, lower turnover, less cost.
  V2 skip12_1    - 12-month momentum skipping the last month (avoids short-term
                   reversal); the classic academic MOM.
  V3 combo       - tanh response on a blend of {multi-lookback, skip12_1}.
Book construction: inverse-vol weight instruments, then a portfolio constant-vol
overlay to a fixed annual target. Compared head-to-head with the plain
equal-weight momentum_BOOK from run_lowfreq.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_lowfreq import (to_daily, perf, make_folds, DATA_MAP, COST_PTS,
                         TARGET_VOL, VOL_WIN, LOOKBACKS, WARMUP)

ROOT = Path(__file__).parent
OUT = ROOT / "results3"
DAILY_OUT = OUT / "trend_plus_daily"
BLEND_LB = [20, 60, 120, 250]


def _vol(r):
    return r.rolling(VOL_WIN).std()


def _size(sig_raw, r, c, inst):
    """Vol-target a raw signal in [-cap,cap]; charge turnover cost; no lookahead."""
    pos = (sig_raw * (TARGET_VOL / np.sqrt(252)) / _vol(r)).clip(-3, 3).shift(1)
    cost = COST_PTS[inst] / c
    turn = pos.diff().abs().fillna(pos.abs())
    return pos * r - turn * cost


def variant_returns(d: pd.DataFrame, inst: str, kind: str) -> pd.Series:
    c = d["close"]
    r = c.pct_change()
    if kind == "base_blend":
        sig = np.sign(pd.concat([c.pct_change(L) for L in BLEND_LB], axis=1)).mean(axis=1)
    elif kind == "tanh":
        zs = []
        for L in BLEND_LB:
            m = c.pct_change(L)
            zs.append(m / m.rolling(VOL_WIN).std())
        z = pd.concat(zs, axis=1).mean(axis=1)
        sig = np.tanh(1.5 * z)
    elif kind == "skip12_1":
        m = c.shift(21) / c.shift(273) - 1          # 12m ago -> 1m ago
        sig = np.tanh(1.5 * m / m.rolling(VOL_WIN).std())
    elif kind == "combo":
        blendz = pd.concat([(c.pct_change(L) / c.pct_change(L).rolling(VOL_WIN).std())
                            for L in BLEND_LB], axis=1).mean(axis=1)
        m = c.shift(21) / c.shift(273) - 1
        skipz = m / m.rolling(VOL_WIN).std()
        sig = np.tanh(1.2 * (0.5 * blendz + 0.5 * skipz))
    else:
        raise ValueError(kind)
    return _size(sig, r, c, inst).rename(inst)


def book(daily: dict, kind: str, const_vol: bool = True) -> pd.Series:
    """Inverse-vol weight instruments, then optional portfolio constant-vol."""
    legs = {inst: variant_returns(d, inst, kind) for inst, d in daily.items()}
    L = pd.DataFrame(legs)
    iv = 1.0 / L.rolling(VOL_WIN).std()
    w = iv.div(iv.sum(axis=1), axis=0).shift(1)
    port = (w * L).sum(axis=1)
    if const_vol:
        rv = port.rolling(VOL_WIN).std()
        lev = (TARGET_VOL / np.sqrt(252) / rv).clip(0, 3).shift(1)
        port = port * lev
    return port.rename(f"book_{kind}")


def main():
    OUT.mkdir(exist_ok=True)
    DAILY_OUT.mkdir(exist_ok=True)
    daily = {inst: to_daily(series) for inst, series in DATA_MAP.items()}
    variants = ["base_blend", "tanh", "skip12_1", "combo"]
    # TRIAL LEDGER: base lowfreq grid (4 lookbacks) already spent + these variants
    n_trials = 4 + len(variants)

    rows = []
    print(f"Trial ledger: {n_trials} configurations (deflation uses this).\n")
    baseline = None
    base_path = OUT / "lowfreq_daily" / "momentum_BOOK.csv"
    if base_path.exists():
        b = pd.read_csv(base_path, index_col=0)
        s = b[b.columns[0]]; s.index = pd.to_datetime(s.index, utc=True)
        baseline = s
        m = perf(s, n_trials)
        m["variant"] = "momentum_BOOK (baseline)"
        rows.append(m)
        print(f"  {'baseline':22} sharpe={m.get('sharpe'):>6} calmar={m.get('calmar')} "
              f"maxDD={m.get('max_dd_pct')}% t={m.get('t_stat')} DSR={m.get('DSR_prob_true_SR_gt_0')}")

    best = None
    for kind in variants:
        pr = book(daily, kind).dropna()
        pr.to_csv(DAILY_OUT / f"book_{kind}.csv", header=["ret"])
        m = perf(pr, n_trials)
        m["variant"] = kind
        rows.append(m)
        tag = "  <-- new best" if (best is None or m.get("sharpe", -9) > best) else ""
        if best is None or m.get("sharpe", -9) > best:
            best = m.get("sharpe", -9)
        print(f"  {kind:22} sharpe={m.get('sharpe'):>6} calmar={m.get('calmar')} "
              f"maxDD={m.get('max_dd_pct')}% t={m.get('t_stat')} "
              f"DSR={m.get('DSR_prob_true_SR_gt_0')}{tag}")

    pd.DataFrame(rows).to_csv(OUT / "trend_plus_summary.csv", index=False)
    winner = max(rows, key=lambda r: r.get("DSR_prob_true_SR_gt_0", 0))
    improved = (baseline is not None and winner["variant"] != "momentum_BOOK (baseline)"
                and winner.get("DSR_prob_true_SR_gt_0", 0) >
                rows[0].get("DSR_prob_true_SR_gt_0", 0))
    verdict = (f"best after deflation: {winner['variant']} "
               f"(sharpe={winner.get('sharpe')}, DSR={winner.get('DSR_prob_true_SR_gt_0')}). "
               + ("Enhancements improved the deflated result - but "
                  if improved else "Enhancements did NOT beat the baseline once deflated - ")
               + ("still " if winner.get('DSR_prob_true_SR_gt_0', 0) <= 0.95 else "")
               + ("below the 0.95 proven-edge bar; forward-test before trusting."
                  if winner.get('DSR_prob_true_SR_gt_0', 0) <= 0.95
                  else "clears the 0.95 bar even after deflation."))
    with open(OUT / "trend_plus_summary.json", "w") as fh:
        json.dump({"n_trials": n_trials, "rows": rows, "verdict": verdict}, fh,
                  indent=2, default=str)
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

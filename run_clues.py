"""Clue hunt - more economically-grounded probes for edge, each OOS + deflated.

CRITICAL LABEL on every clue:
  BETA  = harvesting a risk premium (esp. the equity premium). Will often look
          strong, but it is NOT beating the market - it IS the market's drift
          with risk control. Worth real money; not 'edge/alpha'.
  ALPHA = an attempt at market-beating, relative-value return. Rarer, thinner.

Probes (all testable with the existing OHLC, near-zero extra parameters):
  equity_trend_long   BETA  long ES/NQ when close > 200d MA else flat (Faber/GTAA)
  buyhold             BETA  always-long benchmark (shows how much is just beta)
  gold_trend_long     BETA  same MA filter on gold
  intraday_long       ??    long the open->close session (mirror of the failed
                            overnight drift test)
  turn_of_month       seas  long only last day + first 3 days of each month
Outputs daily series into results3/lowfreq_daily/ (prefix clue_) so
run_meta_analysis folds them into the family-wise Reality Check automatically.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_lowfreq import to_daily, perf, DATA_MAP, COST_PTS, TARGET_VOL, VOL_WIN

ROOT = Path(__file__).parent
OUT = ROOT / "results3"
DAILY_OUT = OUT / "lowfreq_daily"
MA = 200


def _cost(inst, c):
    return COST_PTS[inst] / c


def equity_trend_long(d, inst):
    c = d["close"]; r = c.pct_change()
    filt = (c > c.rolling(MA).mean()).astype(float)          # long/flat
    pos = (filt * (TARGET_VOL / np.sqrt(252)) / r.rolling(VOL_WIN).std()).clip(0, 3).shift(1)
    turn = pos.diff().abs().fillna(pos.abs())
    return (pos * r - turn * _cost(inst, c)).rename(inst)


def buyhold(d, inst):
    c = d["close"]; r = c.pct_change()
    pos = ((TARGET_VOL / np.sqrt(252)) / r.rolling(VOL_WIN).std()).clip(0, 3).shift(1)
    return (pos * r).rename(inst)


def intraday_long(d, inst):
    c, o = d["close"], d["open"]
    sess = c / o - 1
    return (sess - _cost(inst, c)).rename(inst).dropna()


def turn_of_month(d, inst):
    c = d["close"]; r = c.pct_change()
    idx = pd.DatetimeIndex(d.index)
    mkey = idx.to_period("M")
    is_last = pd.Series(mkey, index=idx).groupby(mkey).transform(
        lambda s: s.index == s.index.max()).values
    dom_rank = pd.Series(idx, index=idx).groupby(mkey).cumcount().values  # 0-based
    tom = (dom_rank <= 2) | is_last                         # first 3 + last day
    pos = pd.Series(np.where(tom, 1.0, 0.0), index=idx)
    pos = (pos * (TARGET_VOL / np.sqrt(252)) / r.rolling(VOL_WIN).std()).clip(0, 3).shift(1)
    turn = pos.diff().abs().fillna(pos.abs())
    return (pos * r - turn * _cost(inst, c)).rename(inst)


def main():
    OUT.mkdir(exist_ok=True); DAILY_OUT.mkdir(exist_ok=True)
    daily = {inst: to_daily(series) for inst, series in DATA_MAP.items()}
    rows = []
    n_trials = 12  # honest running tally across the momentum + improvement + clue work

    def record(name, sret, kind, label):
        sret = sret.dropna()
        if len(sret) < 60:
            return
        sret.to_csv(DAILY_OUT / f"clue_{name}.csv", header=["ret"])
        m = perf(sret, n_trials); m.update({"clue": name, "family": kind, "label": label})
        rows.append(m)
        print(f"  [{label:5}] {name:20} sharpe={m.get('sharpe'):>6} "
              f"calmar={m.get('calmar')} maxDD={m.get('max_dd_pct')}% "
              f"t={m.get('t_stat')} DSR={m.get('DSR_prob_true_SR_gt_0')}", flush=True)

    print("== equity-premium capture (BETA - not alpha) ==")
    et = {}
    for inst in ("ES", "NQ"):
        s = equity_trend_long(daily[inst], inst); et[inst] = s
        record(f"eqtrend_{inst}", s, "equity_trend", "BETA")
        record(f"buyhold_{inst}", buyhold(daily[inst], inst), "buyhold", "BETA")
    record("gold_trend", equity_trend_long(daily["XAUUSD"], "XAUUSD"), "equity_trend", "BETA")
    record("eqtrend_BOOK", pd.DataFrame(et).mean(axis=1), "equity_trend", "BETA")

    print("\n== intraday session drift (mirror of failed overnight test) ==")
    for inst in ("ES", "NQ"):
        record(f"intraday_{inst}", intraday_long(daily[inst], inst), "intraday", "drift")

    print("\n== turn-of-month seasonality ==")
    for inst in ("ES", "NQ"):
        record(f"tom_{inst}", turn_of_month(daily[inst], inst), "seasonality", "seas")

    pd.DataFrame(rows).to_csv(OUT / "clues_summary.csv", index=False)
    # separate the honest read: best BETA vs best ALPHA/drift/seas
    beta = [r for r in rows if r["label"] == "BETA"]
    other = [r for r in rows if r["label"] != "BETA"]
    best_beta = max(beta, key=lambda r: r.get("sharpe", -9)) if beta else None
    best_other = max(other, key=lambda r: r.get("sharpe", -9)) if other else None
    lines = []
    if best_beta:
        lines.append(f"best BETA: {best_beta['clue']} sharpe={best_beta['sharpe']} "
                     f"(equity-premium capture - real money, but this is the market's "
                     f"own drift with risk control, NOT alpha)")
    if best_other:
        lines.append(f"best non-beta: {best_other['clue']} ({best_other['family']}) "
                     f"sharpe={best_other['sharpe']} t={best_other['t_stat']} "
                     f"DSR={best_other['DSR_prob_true_SR_gt_0']}")
    verdict = " | ".join(lines)
    with open(OUT / "clues_summary.json", "w") as fh:
        json.dump({"n_trials": n_trials, "rows": rows, "verdict": verdict}, fh,
                  indent=2, default=str)
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

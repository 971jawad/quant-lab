"""FINAL phase: aggregate dev results, apply the PRE-DECLARED survivor
criterion, evaluate survivors ONCE on the holdout (2022-07 -> 2026-06), build
the survivor ensemble book, and write the honest report.

Survivor criterion (declared in research/learnings.md before any holdout look):
  dev trade t-stat >= 2.0, profit factor >= 1.15, n_trades >= 50, and the
  family must carry an economic story from the external literature.

Deflation: n_trials = total configs ever tried, from research/ledger.jsonl.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.propstrats as P
import qlab.strategies as S
import qlab.walkforward as WF
from qlab.metrics import deflated_sharpe, full_metrics, summarize_line
from run_research import (COSTS, DEV_END, SERIES, TF_CFG, eval_wf,
                          ledger_trials, prep)

ROOT = Path(__file__).parent
OUT = ROOT / "research"

T_MIN, PF_MIN, N_MIN = 2.0, 1.15, 50


def dev_table() -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(OUT / "summary_*dev*.csv")):
        rows.append(pd.read_csv(path))
    df = pd.concat(rows, ignore_index=True)
    df["t"] = df["t_stat_trades"].fillna(df.get("t_stat_daily"))
    return df


def pick_survivors(df: pd.DataFrame) -> pd.DataFrame:
    ok = (df["t"] >= T_MIN) & (df["profit_factor"].fillna(0) >= PF_MIN) & \
         (df["n_trades"].fillna(0) >= N_MIN)
    return df[ok].sort_values("t", ascending=False)


def main():
    P.register(WF, S)
    n_trials = ledger_trials()
    df = dev_table()
    surv = pick_survivors(df)
    print(f"dev trials in ledger: {n_trials} configs")
    print(f"dev models evaluated: {len(df)}; survivors: {len(surv)}")
    for _, r in surv.iterrows():
        print(f"  {r['model']}: t={r['t']:.2f} PF={r['profit_factor']} "
              f"n={int(r['n_trades'])} exp={r['expectancy_R']}")

    results = []
    daily_book = {}
    for _, r in surv.iterrows():
        inst, tf, fam, style = r["instrument"], r["tf"], r["family"], r["style"]
        cfg = TF_CFG[tf]
        WF.set_scale(**cfg["scale"])
        P.set_tf(cfg["bars_per_day"])
        bars, feats = prep(inst, tf, final=True)
        m, res = eval_wf(bars, feats, fam, style, COSTS[inst],
                         f"{SERIES[inst]}_{tf}_final", final=True,
                         n_trials=n_trials)
        tag = f"{r['model']}"
        m.update({"model": tag, "window": "holdout"})
        results.append(m)
        print("HOLDOUT " + summarize_line(tag, m))
        # holdout daily returns for the ensemble book
        tr = res.oos_trades
        if not tr.empty:
            ts = bars.index[tr["entry_i"].values.astype(int)]
            tr = tr[ts >= DEV_END].reset_index(drop=True)
        if not tr.empty:
            from qlab.backtest import simulate
            et = bars.index.tz_convert("America/New_York").date
            sim = simulate(tr, et, float(tr["risk_pct"].iloc[-1]),
                           bool(res.deploy.get("conviction_scale", False)),
                           WF.DAILY_CAP, WF.TRAIL_LIMIT, want_daily=True)
            dr = sim.pop("daily_returns", None)
            if dr is not None:
                daily_book[tag] = dr

    book_m = {}
    if daily_book:
        book = pd.DataFrame(daily_book).fillna(0.0).mean(axis=1)
        book_m = full_metrics(None, book, n_trials=n_trials)
        book_m["DSR_prob_true_SR_gt_0"] = round(deflated_sharpe(book, n_trials), 4)
        print("HOLDOUT " + summarize_line("ENSEMBLE_BOOK", book_m))
        book.to_csv(OUT / "holdout_book_daily.csv", header=["ret"])

    with open(OUT / "final_holdout.json", "w") as fh:
        json.dump({"n_trials_deflation": n_trials,
                   "criterion": {"t_min": T_MIN, "pf_min": PF_MIN, "n_min": N_MIN},
                   "survivors_dev": surv.to_dict("records"),
                   "holdout": results, "ensemble_book": book_m},
                  fh, indent=2, default=str)
    pd.DataFrame(results).to_csv(OUT / "summary_final_holdout.csv", index=False)
    print(f"\nwrote research/final_holdout.json")


if __name__ == "__main__":
    main()

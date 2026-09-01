"""CFTC COT positioning study (weekly, 2010-2026).

Signals (both declared BEFORE testing, both logged to the ledger):
  H-A contrarian: z-score (2y) of net speculative positioning / OI at an
      extreme -> fade it (crowded trade reverses).
  H-B flow: 4-week CHANGE in net positioning -> follow it.

No-lookahead: report is as-of Tuesday, published Friday afternoon; positions
apply from the NEXT Monday (>=6 calendar days after as-of). Weekly rebalance.
Dev window only unless --final.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.metrics import full_metrics, summarize_line
from run_external import daily_close

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01")
MAP = {"GOLD": "XAUUSD", "EUR": "EURUSD", "NQ": "NSXUSD"}
COST_RT = {"GOLD": 0.45, "EUR": 0.00016, "NQ": 1.62}


def weekly_panel(sym: str, price_series: str):
    cot = pd.read_csv(ROOT / "data/external/cot_weekly.csv", parse_dates=["date"])
    c = cot[cot["sym"] == sym].set_index("date").sort_index()
    net = (c["nc_long"] - c["nc_short"]) / c["oi"]
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    flow = net.diff(4)
    px = daily_close(price_series)
    # position applies from the Monday AFTER release (as-of Tue + 6 days)
    eff = c.index + pd.Timedelta(days=6)
    idx = px.index.searchsorted(eff)
    idx = np.clip(idx, 0, len(px) - 1)
    entry_px = px.iloc[idx]
    # forward return: entry Monday -> next report's entry Monday (~1 week)
    fwd = entry_px.shift(-1).values / entry_px.values - 1
    return pd.DataFrame({"net": net.values, "z": z.values, "flow": flow.values,
                         "fwd": fwd, "entry": entry_px.values},
                        index=c.index).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    rows = []
    for sym, series in MAP.items():
        p = weekly_panel(sym, series)
        p = p[p.index >= DEV_END] if args.final else p[p.index < DEV_END]
        ic_z = p["z"].corr(p["fwd"], method="spearman")
        ic_f = p["flow"].corr(p["fwd"], method="spearman")
        hi, lo = p[p["z"] > 1.5], p[p["z"] < -1.5]
        mid = p[p["z"].abs() <= 1.5]
        t_hi = stats.ttest_ind(hi["fwd"], mid["fwd"], equal_var=False)[0] if len(hi) > 5 else np.nan
        t_lo = stats.ttest_ind(lo["fwd"], mid["fwd"], equal_var=False)[0] if len(lo) > 5 else np.nan
        print(f"{sym}: n={len(p)} IC(z->fwd)={ic_z:+.3f} IC(flow->fwd)={ic_f:+.3f} | "
              f"z>1.5 n={len(hi)} mean={np.nanmean(hi['fwd'])*1e4:+.0f}bp (t={t_hi:+.2f}) | "
              f"z<-1.5 n={len(lo)} mean={np.nanmean(lo['fwd'])*1e4:+.0f}bp (t={t_lo:+.2f}) | "
              f"mid={np.nanmean(mid['fwd'])*1e4:+.0f}bp")
        # strategy versions: weekly position, cost per side-change
        for name, sig in (("contra", -np.sign(p["z"]).where(p["z"].abs() > 1.5, 0.0)),
                          ("flow", np.sign(p["flow"]))):
            pos = pd.Series(sig, index=p.index).fillna(0.0)
            gross = pos * p["fwd"]
            turn = pos.diff().abs().fillna(pos.abs())
            costs = turn * (COST_RT[sym] / p["entry"])
            wk = (gross - costs)
            # convert weekly to pseudo-daily for the metrics engine (x5 spread)
            daily = wk.repeat(5) / 5
            daily.index = pd.date_range(p.index[0], periods=len(daily), freq="B")
            m = full_metrics(None, daily)
            tag = f"COT_{sym}_{name}"
            m["model"] = tag
            rows.append(m)
            print("  " + summarize_line(tag, m))
            if not args.final:
                with open(OUT / "ledger.jsonl", "a") as fh:
                    fh.write(json.dumps({"phase": "cot", "family": tag,
                                         "n_configs": 2,
                                         "dev_sharpe": m.get("sharpe")}) + "\n")
    tag = "holdout" if args.final else "dev"
    pd.DataFrame(rows).to_csv(OUT / f"summary_cot_{tag}.csv", index=False)


if __name__ == "__main__":
    main()

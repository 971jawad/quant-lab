"""Walk-forward the ICT fib-zone family (S.ict_signals) on the same leak-free
harness, prop guardrails, and conservative costs as the 75-model sweep.

Instruments default to gold (XAUUSD) and Nasdaq (NSXUSD) - the CFDs the setup is
designed for - on 16y of 15-minute data. Add DAX once grxeur is downloaded and
verified (see run_data_15m.py / --pairs grxeur). Results land in results_ict/.

    python run_ict.py                 # full 15m walk-forward, gold + nasdaq
    python run_ict.py --smoke         # fast end-to-end sanity check on a slice
    python run_ict.py --instruments XAUUSD
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import qlab.walkforward as WF
from qlab.features import build_features
from qlab.walkforward import run_wf, oos_metrics
from run_all import COSTS, PROFILES, load_series

ROOT = Path(__file__).parent
DATA_MAP = PROFILES["15m"]["data_map"]        # XAUUSD->XAUUSD, MNQ->NSXUSD, ...
DEFAULT_INSTRUMENTS = ["XAUUSD", "MNQ"]        # gold + nasdaq (MNQ uses NSXUSD)
STYLES = ["A", "B", "C"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", nargs="+", default=DEFAULT_INSTRUMENTS)
    ap.add_argument("--styles", nargs="+", default=STYLES)
    ap.add_argument("--smoke", action="store_true",
                    help="truncate to a recent slice for a quick sanity run")
    ap.add_argument("--out", default="results_ict")
    args = ap.parse_args()

    prof = PROFILES["15m"]
    if args.smoke:
        # ~1.5y of 15m bars: enough for a couple of folds, seconds to run
        WF.set_scale(mult=4, min_train=12000, test_len=6000, ml_max_train=None)
    else:
        WF.set_scale(**prof["scale"])

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    bars_cache, feats_cache = {}, {}
    metrics, folds = [], {}
    t0 = time.time()

    for inst in args.instruments:
        series = DATA_MAP[inst]
        if series not in bars_cache:
            df = load_series(series, prof["suffix"])
            if args.smoke:
                df = df.iloc[-30000:]
            bars_cache[series] = df
            feats_cache[series] = build_features(df)
            print(f"loaded {series}: {len(df)} bars, "
                  f"folds={len(WF.make_folds(len(df)))}", flush=True)
        bars, feats = bars_cache[series], feats_cache[series]
        costs = COSTS[inst]["costs"]

        for style in args.styles:
            tag = f"{inst}_ict_{style}"
            res = run_wf(bars, feats, "ict", style, costs, cache_key=series)
            m = oos_metrics(res, bars)
            m.update({"model": tag, "instrument": inst, "strategy": "ict",
                      "style": style})
            metrics.append(m)
            folds[tag] = res.fold_log
            res.oos_trades.to_csv(out / f"trades_{tag}.csv", index=False)
            print(f"[{time.time()-t0:7.1f}s] {tag:>16}: "
                  f"n={m.get('n_trades',0):4d} taken={m.get('n_taken',0):4d} "
                  f"wr={m.get('win_rate',0):.2f} avgR={m.get('avg_R',0):+.3f} "
                  f"PF={m.get('profit_factor',0):.2f} "
                  f"ret={m.get('total_return_pct',0):+.1f}% "
                  f"dd={m.get('max_dd_pct',0):.1f}% t={m.get('t_stat',0)} "
                  f"sharpe={m.get('sharpe',0)} calmar={m.get('calmar')}",
                  flush=True)

    pd.DataFrame(metrics).to_csv(out / "summary.csv", index=False)
    with open(out / "fold_logs.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.0f}s -> {out}/summary.csv")


if __name__ == "__main__":
    main()

"""Run the full 9-model x 5-instrument walk-forward and export weight files.

Models: {smc, ml, ta} x styles {A: fixed 0.75%% risk @ 3:1, B: in-sample
risk/RR grid search, C: multi-trade conviction-scaled}. Instruments map to
verified data series; MES/ES share the ES series but carry their own cost
models and point values.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.walkforward as WF
from qlab.backtest import Costs
from qlab.features import build_features, ML_FEATURES
from qlab.walkforward import run_wf, oos_metrics, deploy_final

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# costs in instrument price points (spread, slippage/side, commission/side)
COSTS = {
    "MES":    {"point_value": 5.0, "tick": 0.25, "costs": Costs(0.25, 0.25, 0.13)},
    "ES":     {"point_value": 50.0, "tick": 0.25, "costs": Costs(0.25, 0.25, 0.06)},
    "MNQ":    {"point_value": 2.0, "tick": 0.25, "costs": Costs(0.50, 0.50, 0.31)},
    "XAUUSD": {"point_value": 100.0, "tick": 0.10, "costs": Costs(0.25, 0.10, 0.0)},
    "EURUSD": {"point_value": 100000.0, "tick": 0.00001,
               "costs": Costs(0.00006, 0.00002, 0.00004)},
}
PROFILES = {
    "1h": {  # ~2.4y Yahoo hourly (v1)
        "suffix": "1h", "results": "results", "weights": "weights",
        "data_map": {"MES": "ES", "ES": "ES", "MNQ": "NQ",
                     "XAUUSD": "GC", "EURUSD": "EURUSD"},
        "scale": None,
    },
    "15m": {  # ~16y HistData 15-minute (v2). min_train ~3y, test ~1y,
              # ML fit window rolling-capped at ~6.5y of bars.
        "suffix": "15m", "results": "results2", "weights": "weights2",
        "data_map": {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
                     "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"},
        "scale": {"mult": 4, "min_train": 70000, "test_len": 23000,
                  "ml_max_train": 150000},
    },
}
STRATEGIES = ["smc", "ml", "ml_err", "ml_rec", "ta"]
STYLES = ["A", "B", "C"]


def load_series(name: str, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}_{suffix}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="1h")
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    if prof["scale"]:
        WF.set_scale(**prof["scale"])
    RESULTS, WEIGHTS = ROOT / prof["results"], ROOT / prof["weights"]
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    INSTRUMENTS = {inst: {**COSTS[inst], "data": prof["data_map"][inst]}
                   for inst in COSTS}
    feats_cache, bars_cache = {}, {}
    all_metrics, all_folds = [], {}
    t0 = time.time()

    for inst, spec in INSTRUMENTS.items():
        key = spec["data"]
        if key not in bars_cache:
            bars_cache[key] = load_series(key, prof["suffix"])
            feats_cache[key] = build_features(bars_cache[key])
            print(f"loaded {key}: {len(bars_cache[key])} bars, "
                  f"folds={len(WF.make_folds(len(bars_cache[key])))}", flush=True)
        bars, feats = bars_cache[key], feats_cache[key]

        for strat in STRATEGIES:
            for style in STYLES:
                tag = f"{inst}_{strat}_{style}"
                res = run_wf(bars, feats, strat, style, spec["costs"], cache_key=key)
                m = oos_metrics(res, bars)
                m.update({"model": tag, "instrument": inst, "strategy": strat,
                          "style": style})
                all_metrics.append(m)
                all_folds[tag] = res.fold_log
                res.oos_trades.to_csv(RESULTS / f"trades_{tag}.csv", index=False)
                print(f"[{time.time()-t0:7.1f}s] {tag:>18}: "
                      f"n={m.get('n_trades',0):4d} wr={m.get('win_rate',0):.2f} "
                      f"avgR={m.get('avg_R',0):+.3f} PF={m.get('profit_factor',0):.2f} "
                      f"ret={m.get('total_return_pct',0):+.1f}% "
                      f"dd={m.get('max_dd_pct',0):.1f}% t={m.get('t_stat',0)}",
                      flush=True)

                # deployable weight file (fit on ALL data; not used in metrics)
                cfg, model = deploy_final(bars, feats, strat, style, spec["costs"], cache_key=key)
                manifest = {
                    "model_id": tag, "version": 1,
                    "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "instrument": inst, "data_series": key,
                    "point_value_usd": spec["point_value"], "tick": spec["tick"],
                    "costs": vars(spec["costs"]),
                    "strategy": strat, "style": style,
                    "config": cfg,
                    "signal_timeframe": prof["suffix"],
                    "execution": "signal at bar close -> market entry next bar open; "
                                 "stop-first intrabar assumption; targets are limits",
                    "risk_guardrails": {"daily_loss_cap": 0.03,
                                        "trailing_dd_limit": 0.05},
                    "walkforward_oos": {k: m.get(k) for k in
                                        ("n_trades", "win_rate", "avg_R", "t_stat",
                                         "profit_factor", "total_return_pct",
                                         "ann_return_pct", "sharpe", "max_dd_pct",
                                         "trailing_dd_breaches", "oos_days")},
                }
                if strat.startswith("ml") and model is not None:
                    pkl = WEIGHTS / f"{tag}.pkl"
                    with open(pkl, "wb") as fh:
                        pickle.dump({"model": model, "features": ML_FEATURES}, fh)
                    manifest["ml"] = {
                        "model_file": pkl.name,
                        "algorithm": type(model).__name__,
                        "features": ML_FEATURES,
                    }
                    if hasattr(model, "feature_importances_"):
                        manifest["ml"]["feature_importances"] = {
                            f: round(float(w), 5) for f, w in
                            sorted(zip(ML_FEATURES, model.feature_importances_),
                                   key=lambda x: -x[1])}
                with open(WEIGHTS / f"{tag}.json", "w") as fh:
                    json.dump(manifest, fh, indent=2, default=str)

    pd.DataFrame(all_metrics).to_csv(RESULTS / "summary.csv", index=False)
    with open(RESULTS / "fold_logs.json", "w") as fh:
        json.dump(all_folds, fh, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.0f}s. {len(all_metrics)} models -> "
          f"results/summary.csv, weights/*.json")


if __name__ == "__main__":
    main()

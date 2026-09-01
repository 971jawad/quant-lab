"""Generate the AI (neural-net) leg's walk-forward OOS trades on the 16y 15m
data, writing into results2/ alongside the existing 75 models. Mirrors
run_all.py's 15m profile exactly (same folds, costs, scale) so the AI trades are
directly comparable and drop straight into the ensemble.

Outputs per (instrument, style):
  results2/trades_{inst}_ai_{style}.csv   - OOS trades
  weights2/{inst}_ai_{style}.json/.pkl    - deploy manifest + fitted pipeline
  results2/summary_ai.csv                 - metrics rows (NOT merged into the
                                            audited summary.csv; make_report can
                                            union them)
"""
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import qlab.walkforward as WF
from qlab.backtest import Costs
from qlab.features import build_features, ML_FEATURES
from qlab.walkforward import run_wf, oos_metrics, deploy_final

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).parent
DATA, RESULTS, WEIGHTS = ROOT / "data", ROOT / "results2", ROOT / "weights2"

COSTS = {
    "MES":    {"point_value": 5.0, "tick": 0.25, "costs": Costs(0.25, 0.25, 0.13)},
    "ES":     {"point_value": 50.0, "tick": 0.25, "costs": Costs(0.25, 0.25, 0.06)},
    "MNQ":    {"point_value": 2.0, "tick": 0.25, "costs": Costs(0.50, 0.50, 0.31)},
    "XAUUSD": {"point_value": 100.0, "tick": 0.10, "costs": Costs(0.25, 0.10, 0.0)},
    "EURUSD": {"point_value": 100000.0, "tick": 0.00001,
               "costs": Costs(0.00006, 0.00002, 0.00004)},
}
DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
# Same folds as the base 15m study (min_train/test_len unchanged) so the AI OOS
# trades line up exactly; the AI leg's own training window is capped tighter
# (90k bars ~ 4y) to keep neural refits fast without changing fold boundaries.
SCALE = {"mult": 4, "min_train": 70000, "test_len": 23000, "ml_max_train": 90000}
STYLES = ["A", "B", "C"]


def load_series(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}_15m.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", nargs="*", default=list(COSTS))
    args = ap.parse_args()
    WF.set_scale(**SCALE)
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)

    bars_cache, feats_cache = {}, {}
    rows = []
    t0 = time.time()
    for inst in args.instruments:
        key = DATA_MAP[inst]
        if key not in bars_cache:
            bars_cache[key] = load_series(key)
            feats_cache[key] = build_features(bars_cache[key])
            print(f"loaded {key}: {len(bars_cache[key])} bars, "
                  f"folds={len(WF.make_folds(len(bars_cache[key])))}", flush=True)
        bars, feats = bars_cache[key], feats_cache[key]
        for style in STYLES:
            tag = f"{inst}_ai_{style}"
            res = run_wf(bars, feats, "ai", style, COSTS[inst]["costs"], cache_key=key)
            m = oos_metrics(res, bars)
            m.update({"model": tag, "instrument": inst, "strategy": "ai", "style": style})
            rows.append(m)
            res.oos_trades.to_csv(RESULTS / f"trades_{tag}.csv", index=False)
            print(f"[{time.time()-t0:7.1f}s] {tag:>16}: n={m.get('n_trades',0):5d} "
                  f"wr={m.get('win_rate',0):.2f} avgR={m.get('avg_R',0):+.3f} "
                  f"PF={m.get('profit_factor',0):.2f} ret={m.get('total_return_pct',0):+.1f}% "
                  f"sharpe={m.get('sharpe',0)} calmar={m.get('calmar')} "
                  f"dd={m.get('max_dd_pct',0):.1f}% t={m.get('t_stat',0)}", flush=True)

            cfg, model = deploy_final(bars, feats, "ai", style, COSTS[inst]["costs"], cache_key=key)
            manifest = {
                "model_id": tag, "version": 1,
                "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "instrument": inst, "data_series": key,
                "point_value_usd": COSTS[inst]["point_value"], "tick": COSTS[inst]["tick"],
                "costs": vars(COSTS[inst]["costs"]),
                "strategy": "ai", "style": style, "config": cfg,
                "signal_timeframe": "15m",
                "risk_guardrails": {"daily_loss_cap": 0.03, "trailing_dd_limit": 0.05},
                "walkforward_oos": {k: m.get(k) for k in
                                    ("n_trades", "win_rate", "avg_R", "t_stat",
                                     "profit_factor", "total_return_pct", "ann_return_pct",
                                     "sharpe", "calmar", "max_dd_pct", "oos_days")},
            }
            if model is not None:
                pkl = WEIGHTS / f"{tag}.pkl"
                with open(pkl, "wb") as fh:
                    pickle.dump({"model": model, "features": ML_FEATURES}, fh)
                manifest["ml"] = {"model_file": pkl.name,
                                  "algorithm": type(model).__name__, "features": ML_FEATURES}
            with open(WEIGHTS / f"{tag}.json", "w") as fh:
                json.dump(manifest, fh, indent=2, default=str)

    pd.DataFrame(rows).to_csv(RESULTS / "summary_ai.csv", index=False)
    print(f"\nDone in {time.time()-t0:.0f}s. {len(rows)} AI models -> results2/", flush=True)


if __name__ == "__main__":
    main()

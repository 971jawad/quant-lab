"""Meta-labeling (Lopez de Prado) - Stage 2 trade filter, PRE-REGISTERED SPEC.

For every model with enough out-of-sample trades, walk forward over the same
folds: train a FIXED meta-classifier on the pre-trade features of all PRIOR
folds' OOS trades (labels: did the trade make money?), then keep only the
current fold's trades whose meta-score clears the top-40%% threshold - with
the threshold computed on the TRAINING trades, never the current fold.

Locked before running (no grids, no peeking, results reported either way):
  - meta-model: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
    min_samples_leaf=50, random_state=7) - never tuned
  - features: ML_FEATURES at the signal bar + direction + conviction +
    stop distance in ATR units
  - activation: >= 200 prior-fold trades, else the fold passes through unfiltered
  - keep rule: meta-score >= 60th percentile of training-trade scores (top 40%)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import qlab.walkforward as WF
from qlab.features import build_features, ML_FEATURES

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results2"

DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
MIN_TRAIN_TRADES = 200
KEEP_PCT = 60          # threshold percentile -> keep top 40%


def meta_features(feats: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    X = feats[ML_FEATURES].iloc[tr["sig_i"].values].reset_index(drop=True)
    X["dir"] = tr["dir"].values
    X["conviction"] = tr["conviction"].values
    X["stop_atr"] = tr["dist"].values / feats["atr"].iloc[tr["sig_i"].values].values
    return X


def run_meta(tag: str, feats: pd.DataFrame, folds: list) -> dict | None:
    f = RESULTS / f"trades_{tag}.csv"
    if not f.exists():
        return None
    tr = pd.read_csv(f)
    if len(tr) < MIN_TRAIN_TRADES + 50 or "dist" not in tr.columns:
        return None
    tr = tr.sort_values("sig_i").reset_index(drop=True)
    X = meta_features(feats, tr)
    y = (tr["R"] > 0).astype(int)
    keep = np.zeros(len(tr), dtype=bool)
    filtered_folds = 0
    for train_end, test_end in folds:
        test_m = (tr["sig_i"] >= train_end) & (tr["sig_i"] < test_end)
        if not test_m.any():
            continue
        # prior-fold trades whose OUTCOME was known before this fold began
        train_m = tr["exit_i"] < train_end
        if train_m.sum() < MIN_TRAIN_TRADES:
            keep[test_m.values] = True          # not enough history: pass through
            continue
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                           min_samples_leaf=50, random_state=7)
        ok = X[train_m].notna().all(axis=1)
        m.fit(X[train_m][ok], y[train_m][ok])
        thresh = np.percentile(m.predict_proba(X[train_m][ok])[:, 1], KEEP_PCT)
        scores = m.predict_proba(X[test_m].fillna(0))[:, 1]
        keep[test_m.values] = scores >= thresh
        filtered_folds += 1
    base, kept = tr["R"].values, tr["R"].values[keep]
    if len(kept) < 20 or filtered_folds == 0:
        return None

    def stats(r):
        return {"n": int(len(r)), "avg_R": round(float(r.mean()), 4),
                "win_rate": round(float((r > 0).mean()), 3),
                "t_stat": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(len(r))), 2)}
    return {"model": tag, "filtered_folds": filtered_folds,
            "all_trades": stats(base), "meta_kept": stats(kept),
            "kept_share": round(float(keep.mean()), 3),
            "delta_avg_R": round(float(kept.mean() - base.mean()), 4)}


def main():
    WF.set_scale(mult=4, min_train=70000, test_len=23000, ml_max_train=150000)
    feats_cache, folds_cache = {}, {}
    out = []
    for inst, series in DATA_MAP.items():
        if series not in feats_cache:
            bars = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0)
            bars.index = pd.to_datetime(bars.index, utc=True)
            feats_cache[series] = build_features(bars)
            folds_cache[series] = WF.make_folds(len(bars))
        for strat in ("smc", "ml", "ml_err", "ml_rec", "ta"):
            for style in ("A", "B", "C"):
                r = run_meta(f"{inst}_{strat}_{style}", feats_cache[series],
                             folds_cache[series])
                if r:
                    out.append(r)
                    d = r["delta_avg_R"]
                    print(f"{r['model']:>20}: all n={r['all_trades']['n']:5d} "
                          f"avgR={r['all_trades']['avg_R']:+.3f} t={r['all_trades']['t_stat']:+5.2f}"
                          f" | kept n={r['meta_kept']['n']:5d} "
                          f"avgR={r['meta_kept']['avg_R']:+.3f} t={r['meta_kept']['t_stat']:+5.2f}"
                          f" | dR={d:+.3f}", flush=True)
    with open(RESULTS / "meta_labeling.json", "w") as fh:
        json.dump(out, fh, indent=2)
    if out:
        deltas = [r["delta_avg_R"] for r in out]
        improved = sum(1 for d in deltas if d > 0)
        print(f"\n{improved}/{len(out)} models improved by meta-filter; "
              f"median dR={np.median(deltas):+.4f}")


if __name__ == "__main__":
    main()

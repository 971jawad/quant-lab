"""Rule-based edge discovery = CONDITIONAL EXPECTANCY analysis, done honestly.

For every out-of-sample trade we attach the metrics known at its signal bar
(session/killzone, hour, day-of-week, volatility regime, trend, Donchian
position, RSI band, direction, distance to prior-day levels) and ask: is there a
CONDITION under which average-R is reliably positive?

Anti-snooping protocol (this is the point):
  * Pool all OOS trades, split by DATE into train (first 60%) and test (last 40%).
  * Bucket edges (terciles) are computed on TRAIN only, then applied to TEST.
  * For each metric we pick the best bucket by TRAIN t-stat (needs positive
    train avg-R and a minimum sample), then report how it did on the untouched
    TEST slice. A condition that only works in train is snooping, not edge.
  * Separately, Benjamini-Hochberg FDR across every bucket flags which, if any,
    survive multiple-testing correction on the full sample.

Output: results3/conditional_edge.csv  (+ printed survivors)
Hypothesis-generating only; anything promising must still clear
run_meta_analysis.py's family-wise correction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import qlab.walkforward as WF
from qlab.features import build_features

ROOT = Path(__file__).parent
DATA, RES2, OUT = ROOT / "data", ROOT / "results2", ROOT / "results3"
DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
SCALE = {"mult": 4, "min_train": 70000, "test_len": 23000, "ml_max_train": 150000}
FAMS = ["smc", "ta", "ml", "ml_err", "ml_rec", "ai"]
STYLES = ["A", "B", "C"]
MIN_TRAIN, MIN_TEST = 60, 30


def session(h):
    if 2 <= h < 5:
        return "London_KZ"
    if 7 <= h < 11:
        return "NY_KZ"
    if h < 2 or h >= 20:
        return "Asia"
    return "Off"


def tercile_labels(x, edges):
    return pd.cut(x, [-np.inf, edges[0], edges[1], np.inf], labels=["lo", "mid", "hi"])


def attach(feats: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    si = tr["sig_i"].astype(int).values
    f = feats.iloc[si].reset_index(drop=True)
    d = pd.DataFrame({
        "R": tr["R"].astype(float).values,
        "dir": np.where(tr["dir"].values > 0, "long", "short"),
        "hour": f["hour_et"].values,
        "session": [session(int(h)) for h in f["hour_et"].values],
        "dow": f["dow"].values,
        "killzone": np.where(f["killzone"].values, "in_KZ", "out_KZ"),
        "trend": np.where(f["sma200_dist"].values > 0, "up", "down"),
        "vol_ratio": f["vol_ratio"].values,
        "atr_pct": f["atr_pct"].values,
        "don_pos": f["don_pos"].values,
        "rsi": f["rsi"].values,
    })
    d["rsi_band"] = np.where(d["rsi"] < 30, "<30", np.where(d["rsi"] > 70, ">70", "30-70"))
    return d


def tstat(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std() == 0:
        return 0.0, 1.0
    t = r.mean() / r.std() * np.sqrt(len(r))
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(r) - 1))
    return float(t), float(p)


CATS = ["session", "dir", "dow", "killzone", "trend", "rsi_band"]
CONTS = ["vol_ratio", "atr_pct", "don_pos"]


def main():
    WF.set_scale(**SCALE)
    OUT.mkdir(exist_ok=True)
    feats_cache = {}
    all_rows = []
    for inst, series in DATA_MAP.items():
        if series not in feats_cache:
            df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            feats_cache[series] = (build_features(df), df.index)
        feats, idx = feats_cache[series]
        chunks = []
        for fam in FAMS:
            for st in STYLES:
                p = RES2 / f"trades_{inst}_{fam}_{st}.csv"
                if not p.exists():
                    continue
                tr = pd.read_csv(p)
                if tr.empty:
                    continue
                d = attach(feats, tr)
                d["ts"] = idx[tr["sig_i"].astype(int).values]
                d["inst"] = inst
                chunks.append(d)
        if chunks:
            all_rows.append(pd.concat(chunks, ignore_index=True))
    if not all_rows:
        print("No OOS trades found. Run run_all.py / run_ai.py first.")
        return
    D = pd.concat(all_rows, ignore_index=True).sort_values("ts").reset_index(drop=True)

    # date split
    cut = D["ts"].quantile(0.6)
    tr, te = D[D["ts"] < cut], D[D["ts"] >= cut]
    base_tr_t, _ = tstat(tr["R"])
    base_te_t, _ = tstat(te["R"])
    print(f"pooled OOS trades: {len(D)} (train {len(tr)}, test {len(te)}); "
          f"baseline avgR train={tr['R'].mean():+.4f} (t={base_tr_t:.2f}) "
          f"test={te['R'].mean():+.4f} (t={base_te_t:.2f})")

    # continuous -> terciles by TRAIN edges
    for c in CONTS:
        e = [tr[c].quantile(1 / 3), tr[c].quantile(2 / 3)]
        D[c + "_b"] = tercile_labels(D[c], e)
        tr, te = D[D["ts"] < cut], D[D["ts"] >= cut]

    rows = []
    metrics = CATS + [c + "_b" for c in CONTS]
    for m in metrics:
        for lvl, g in D.groupby(m, observed=True):
            gtr = g[g["ts"] < cut]["R"]
            gte = g[g["ts"] >= cut]["R"]
            if len(gtr) < MIN_TRAIN:
                continue
            t_tr, p_tr = tstat(gtr)
            t_te, p_te = tstat(gte) if len(gte) >= MIN_TEST else (np.nan, np.nan)
            rows.append({"metric": m, "bucket": str(lvl),
                         "n_train": len(gtr), "avgR_train": round(gtr.mean(), 4),
                         "t_train": round(t_tr, 2), "p_train": round(p_tr, 4),
                         "n_test": len(gte), "avgR_test": round(gte.mean(), 4) if len(gte) else np.nan,
                         "t_test": round(t_te, 2) if not np.isnan(t_te) else np.nan,
                         "p_test": round(p_te, 4) if not np.isnan(p_te) else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "conditional_edge.csv", index=False)

    # per-metric best bucket by TRAIN t (positive train edge), then TEST result
    print("\nBest condition per metric (picked on TRAIN, verified on TEST):")
    survivors = []
    for m in metrics:
        sub = R[(R["metric"] == m) & (R["avgR_train"] > 0)]
        if sub.empty:
            continue
        best = sub.loc[sub["t_train"].idxmax()]
        held = (not np.isnan(best["t_test"])) and best["avgR_test"] > 0 and best["t_test"] > 0
        flag = "HELD OOS" if held else "faded"
        print(f"  {m:12} = {best['bucket']:>10}: train avgR={best['avgR_train']:+.4f} "
              f"t={best['t_train']:+.2f} (n={best['n_train']})  ->  "
              f"test avgR={best['avgR_test']:+.4f} t={best['t_test']:+.2f} "
              f"(n={best['n_test']})  [{flag}]")
        if held and best["t_test"] >= 2.0:
            survivors.append(best.to_dict())

    # BH-FDR across ALL buckets on the FULL sample
    full = []
    for m in metrics:
        for lvl, g in D.groupby(m, observed=True):
            if len(g) < MIN_TRAIN + MIN_TEST:
                continue
            t, p = tstat(g["R"])
            full.append((f"{m}={lvl}", len(g), g["R"].mean(), t, p))
    fdf = pd.DataFrame(full, columns=["cond", "n", "avgR", "t", "p"]).sort_values("p")
    if len(fdf):
        mcnt = len(fdf)
        fdf["bh_thresh"] = (np.arange(1, mcnt + 1) / mcnt) * 0.10
        fdf["fdr_pass"] = fdf["p"].values <= fdf["bh_thresh"].values
        fdf.to_csv(OUT / "conditional_fdr.csv", index=False)
        npass = int(fdf["fdr_pass"].sum())
        print(f"\nBH-FDR (q=0.10) across {mcnt} conditions: {npass} pass.")
        if npass:
            print(fdf[fdf["fdr_pass"]][["cond", "n", "avgR", "t", "p"]].to_string(index=False))

    verdict = ("candidate condition(s) HELD out-of-sample with t>=2 - worth "
               "encoding as a rule and re-testing forward"
               if survivors else
               "no condition held out-of-sample at t>=2 - the pooled edge is not "
               "hiding in these metrics (consistent with the no-edge finding)")
    print(f"\nVERDICT: {verdict}")
    with open(OUT / "conditional_summary.json", "w") as fh:
        json.dump({"n_trades": int(len(D)), "survivors": survivors,
                   "verdict": verdict}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()

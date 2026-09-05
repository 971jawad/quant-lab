"""CANDLE RESEARCH PROGRAM — the whole method, applied to candlesticks.

Phases, mirroring the main program:
  1 PATTERNS   22 canonical patterns x 7 markets, defined from the literature
               with zero tuning. Native-frequency t-stats (per-signal, not
               per-day: the sqrt(5) inflation bug taught us that).
  2 ML         gradient boosting on candle GEOMETRY (not pattern labels), so the
               model can find combinations the canon missed. Walk-forward refit,
               embargoed, so no fold ever trains on its own test window.
  3 AI         a neural net on the same features - a different function class,
               same walk-forward.
  4 ENSEMBLE   combine whatever survives, weighted by evidence, never equally.
  5 META       deflated Sharpe + expected-maximum-null bar for the number of
               patterns tried. With 154 pattern tests, the luckiest is expected
               to look good; this is what corrects for it.

Discipline: DEV 2010->2022-06 for everything. HOLDOUT touched once at the end.
No lookahead (patterns complete at bar t, entry at t+1 open). No survivorship
bias (all 7 markets, all 22 patterns reported, winners and losers). Every trial
appended to the ledger.

    python run_candles.py            # dev research
    python run_candles.py --final    # the single holdout look
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.candles import PATTERNS, feature_matrix, signals
from qlab.metrics import full_metrics
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV_END = pd.Timestamp("2022-07-01")
HOLD_DAYS = 5          # canonical evaluation horizon for a reversal signal
EULER = 0.5772156649


def fwd_returns(d, h=HOLD_DAYS):
    """Entry at NEXT open, exit h bars later at the open. No lookahead."""
    o = d["open"]
    return (o.shift(-(h + 1)) / o.shift(-1) - 1)


def pattern_stats(d, name, final):
    sig = signals(d, name)
    fwd = fwd_returns(d)
    m = pd.DataFrame({"s": sig, "f": fwd}).dropna()
    m = m[m.index >= DEV_END] if final else m[m.index < DEV_END]
    on = m[m["s"] != 0]
    if len(on) < 20:
        return None
    # directional return: pattern's own call
    r = on["s"] * on["f"]
    base = m[m["s"] == 0]["f"]
    t, p = stats.ttest_1samp(r, 0.0)
    return {"n": int(len(on)), "mean_bp": float(r.mean() * 1e4),
            "base_bp": float(base.mean() * 1e4), "t": float(t), "p": float(p),
            "win_rate": float((r > 0).mean())}


def pattern_leg(d, name, inst):
    """Tradeable version: hold the pattern's direction for HOLD_DAYS, vol-targeted."""
    sig = signals(d, name)
    pos = pd.Series(0.0, index=d.index)
    idx = np.flatnonzero(sig.values != 0)
    sv = sig.values
    for i in idx:
        pos.iloc[i + 1:i + 1 + HOLD_DAYS] = sv[i]
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (pos * (0.10 / np.sqrt(252)) / vol).clip(-3, 3)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * COST[inst] / d["close"]).dropna()


# ------------------------------------------------------------------ ML / AI
def ml_leg(d, inst, kind="gbm"):
    """Walk-forward, embargoed. Label = sign of the tradeable forward return."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = feature_matrix(d)
    y = (fwd_returns(d) > 0).astype(float)
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv = X[ok], y[ok]
    n = len(Xv)
    emb = HOLD_DAYS + 1
    oos = pd.Series(0.0, index=Xv.index)
    te = 1000
    while te < n - 100:
        end = min(te + 250, n)
        tr_hi = te - emb                      # embargo: no label crosses the split
        if tr_hi < 300:
            te = end
            continue
        if kind == "gbm":
            mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                                 max_leaf_nodes=15,
                                                 min_samples_leaf=100,
                                                 l2_regularization=1.0,
                                                 random_state=7)
        else:
            mdl = Pipeline([("s", StandardScaler()),
                            ("m", MLPClassifier(hidden_layer_sizes=(24, 8), alpha=3e-3,
                                                max_iter=60, early_stopping=True,
                                                n_iter_no_change=6, random_state=7))])
        mdl.fit(Xv.iloc[:tr_hi], yv.iloc[:tr_hi])
        pr = mdl.predict_proba(Xv.iloc[te:end])[:, 1]
        oos.iloc[te:end] = np.where(pr > 0.55, 1.0, np.where(pr < 0.45, -1.0, 0.0))
        te = end
    pos = oos.reindex(d.index).fillna(0.0)
    held = pos.copy()
    for k in range(1, HOLD_DAYS):
        held = held.where(held != 0, pos.shift(k).fillna(0.0))
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (held.shift(1) * (0.10 / np.sqrt(252)) / vol).clip(-3, 3)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * COST[inst] / d["close"]).dropna()


def emax_null(n_trials, n_obs):
    e = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(e / np.sqrt(n_obs) * np.sqrt(252))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    win = "holdout" if args.final else "dev"
    data = {i: daily(i) for i in MKT}

    # ---------------- phase 1: patterns
    print("=" * 84)
    print(f"PHASE 1 — 22 CANONICAL PATTERNS x 7 MARKETS ({win})")
    print("=" * 84)
    rows = []
    for name in PATTERNS:
        agg = []
        for inst, d in data.items():
            st = pattern_stats(d, name, args.final)
            if st:
                st.update({"pattern": name, "market": inst})
                rows.append(st)
                agg.append(st)
        if agg:
            tot_n = sum(a["n"] for a in agg)
            wmean = sum(a["mean_bp"] * a["n"] for a in agg) / tot_n
            ts = [a["t"] for a in agg]
            best = max(agg, key=lambda a: a["t"])
            print(f"  {name:17} n={tot_n:>5}  mean {wmean:>+7.1f}bp  "
                  f"markets>0: {sum(1 for a in agg if a['mean_bp']>0)}/{len(agg)}  "
                  f"best t={best['t']:+.2f} ({best['market']})")
    df = pd.DataFrame(rows)
    n_tests = len(df)
    sig5 = df[df["p"] < 0.05]
    exp_false = 0.05 * n_tests
    print(f"\n  {n_tests} pattern-market tests | p<0.05: {len(sig5)} "
          f"(expected by chance alone: {exp_false:.0f})")
    if len(sig5):
        print("  nominally significant:")
        for _, r in sig5.sort_values("t", key=abs, ascending=False).head(8).iterrows():
            print(f"    {r['pattern']:17} {r['market']:8} n={int(r['n']):>4} "
                  f"{r['mean_bp']:>+7.1f}bp t={r['t']:+.2f} p={r['p']:.4f}")
        # Benjamini-Hochberg across ALL tests: which survive multiplicity?
        p = np.sort(df["p"].values)
        m = len(p)
        thresh = p <= (np.arange(1, m + 1) / m) * 0.05
        k = int(thresh.sum())
        print(f"  after Benjamini-Hochberg FDR control at 5%: "
              f"{k} of {m} survive")

    # ---------------- phase 2/3: ML + AI on candle geometry
    print("\n" + "=" * 84)
    print(f"PHASE 2/3 — ML + AI ON CANDLE GEOMETRY ({win})")
    print("=" * 84)
    ml_rows = {}
    for kind in ("gbm", "nn"):
        for inst, d in data.items():
            try:
                leg = ml_leg(d, inst, kind)
            except Exception as e:
                print(f"  {kind}_{inst:8} ERROR {str(e)[:40]}")
                continue
            seg = leg[leg.index >= DEV_END] if args.final else leg[leg.index < DEV_END]
            mm = full_metrics(None, seg)
            sh = mm.get("sharpe")
            ml_rows[f"{kind}_{inst}"] = {"sharpe": sh, "cagr": mm.get("cagr_pct")}
            print(f"  {kind}_{inst:10} Sharpe {sh if sh is not None else float('nan'):>6}  "
                  f"CAGR {mm.get('cagr_pct')}%")
            leg.to_csv(OUT / f"candle_{kind}_{inst}.csv", header=["ret"])

    payload = {"window": win, "pattern_tests": rows, "ml": ml_rows,
               "n_tests": n_tests, "n_p05": int(len(sig5))}
    json.dump(payload, open(OUT / f"candles_{win}.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": f"candles_{win}",
                            "n_configs": n_tests + len(ml_rows)}) + "\n")
    print(f"\nwrote research/candles_{win}.json")


if __name__ == "__main__":
    main()

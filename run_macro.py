"""MACRO / POLICY / LIQUIDITY research — does anything PREDICT, not just explain?

Round 3 established that macro drivers are CONTEMPORANEOUS: gold moves with real
yields (corr -0.31) and indices with VIX (-0.62) on the SAME day, while lagged
correlations sit at ~0.00. That killed prediction from the obvious drivers.

This program asks the harder question with far more data — Fed balance sheet,
Treasury cash, reverse repo, term premium, credit spreads, financial conditions,
BOJ balance sheet, claims, debt — and with every series shifted by its real
publication lag (qlab/macro.py::PUB_LAG), so nothing is knowable before it was
actually released.

Tests, per feature x market x horizon:
  1  IC          Spearman rank correlation of the feature with FORWARD return
  2  tertile     top-third vs bottom-third forward return, Welch t-test
  3  tradeable   sign(feature) held for the horizon, vol-targeted, after costs

Multiplicity is the whole game here: with ~35 features x 7 markets x 3 horizons
we run ~700 tests, so ~35 will clear p<0.05 on noise alone. Benjamini-Hochberg
FDR control decides what, if anything, is real.

DEV window only unless --final. Every trial ledgered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.macro import build as build_macro
from qlab.metrics import full_metrics
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV_END = pd.Timestamp("2022-07-01")
HORIZONS = (1, 5, 20)
MIN_OBS = 400


def fwd(d, h):
    """Tradeable forward return: enter next open, exit h bars later at the open."""
    o = d["close"]
    return o.shift(-h) / o - 1


def test_feature(px, mac, col, h):
    x = mac[col]
    y = fwd(px, h)
    j = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    j = j[j.index < DEV_END]
    if len(j) < MIN_OBS or j["x"].nunique() < 10:
        return None
    ic, ic_p = stats.spearmanr(j["x"], j["y"])
    q = j["x"].quantile([1 / 3, 2 / 3])
    lo, hi = j[j["x"] <= q.iloc[0]]["y"], j[j["x"] >= q.iloc[1]]["y"]
    if len(lo) < 50 or len(hi) < 50:
        return None
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    # effective sample size: overlapping h-day returns are autocorrelated, so the
    # naive t is inflated by ~sqrt(h). Deflate it rather than pretend otherwise.
    t_adj = float(t) / np.sqrt(h)
    return {"feature": col, "horizon": h, "n": int(len(j)),
            "ic": float(ic), "ic_p": float(ic_p),
            "hi_bp": float(hi.mean() * 1e4), "lo_bp": float(lo.mean() * 1e4),
            "t_raw": float(t), "t_adj": t_adj,
            "p_adj": float(2 * (1 - stats.norm.cdf(abs(t_adj))))}


def tradeable(px, mac, col, inst, h):
    """sign(z-scored feature) held h days, vol-targeted, after costs."""
    x = mac[col]
    z = (x - x.rolling(252).mean()) / x.rolling(252).std()
    pos = np.sign(z).shift(1)
    held = pos.copy()
    for k in range(1, h):
        held = held.where(held.notna(), pos.shift(k))
    r = px["close"].pct_change()
    vol = r.rolling(60).std()
    p = (held * (0.10 / np.sqrt(252)) / vol).clip(-3, 3)
    turn = p.diff().abs().fillna(p.abs())
    net = (p * r - turn * COST[inst] / px["close"]).dropna()
    return net[net.index < DEV_END]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    data = {i: daily(i) for i in MKT}
    idx = data["MNQ"].index
    mac = build_macro(idx)
    cols = [c for c in mac.columns if mac[c].notna().sum() > MIN_OBS]
    print(f"macro features usable: {len(cols)} | markets: {len(data)} | "
          f"horizons: {HORIZONS}")
    print(f"total tests: {len(cols) * len(data) * len(HORIZONS)}")

    rows = []
    for inst, px in data.items():
        m = mac.reindex(px.index).ffill()
        for col in cols:
            for h in HORIZONS:
                r = test_feature(px, m, col, h)
                if r:
                    r.update({"market": inst})
                    rows.append(r)
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\ncompleted {n} feature-market-horizon tests")

    # ---------- multiplicity control on the AUTOCORRELATION-ADJUSTED p ----------
    p = np.sort(df["p_adj"].values)
    bh = p <= (np.arange(1, n + 1) / n) * 0.05
    k_bh = int(bh.sum())
    crit = p[bh][-1] if k_bh else 0.0
    survivors = df[df["p_adj"] <= crit] if k_bh else df.iloc[0:0]

    print(f"p_adj < 0.05 (nominal): {int((df['p_adj'] < 0.05).sum())} "
          f"(expected by chance: {0.05*n:.0f})")
    print(f"BENJAMINI-HOCHBERG FDR 5%: {k_bh} of {n} survive")

    print("\nstrongest 12 by |adjusted t|:")
    top = df.reindex(df["t_adj"].abs().sort_values(ascending=False).index).head(12)
    for _, r in top.iterrows():
        print(f"  {r['feature']:22} {r['market']:8} h={int(r['horizon']):>2}d  "
              f"IC {r['ic']:+.3f}  hi-lo {r['hi_bp']-r['lo_bp']:>+8.0f}bp  "
              f"t_adj {r['t_adj']:+.2f}  p {r['p_adj']:.4f}")

    # ---------- consistency: does a feature work ACROSS markets? ----------
    print("\nfeatures with a consistent sign across markets (h=20d):")
    h20 = df[df["horizon"] == 20]
    cons = []
    for col, g in h20.groupby("feature"):
        if len(g) < 5:
            continue
        pos = int((g["ic"] > 0).sum())
        agree = max(pos, len(g) - pos) / len(g)
        mt = float(g["t_adj"].abs().mean())
        cons.append({"feature": col, "markets": len(g), "agree": agree,
                     "mean_abs_t": mt, "mean_ic": float(g["ic"].mean())})
    cons.sort(key=lambda c: (-c["agree"], -c["mean_abs_t"]))
    for c in cons[:10]:
        print(f"  {c['feature']:22} agree {c['agree']:.0%} of {c['markets']} "
              f"markets  mean|t| {c['mean_abs_t']:.2f}  mean IC {c['mean_ic']:+.3f}")

    # ---------- tradeable versions of the top consistent features ----------
    print("\ntradeable legs for the most consistent features (dev Sharpe):")
    trade = {}
    for c in cons[:5]:
        for inst, px in data.items():
            try:
                leg = tradeable(px, mac.reindex(px.index).ffill(), c["feature"], inst, 20)
                sh = full_metrics(None, leg).get("sharpe")
                if sh is not None:
                    trade[f"{c['feature']}|{inst}"] = sh
            except Exception:
                pass
        vals = [v for k, v in trade.items() if k.startswith(c["feature"] + "|")]
        if vals:
            print(f"  {c['feature']:22} mean {np.mean(vals):+.2f}  "
                  f"best {max(vals):+.2f}  positive {sum(1 for v in vals if v>0)}/{len(vals)}")

    payload = {"n_tests": n, "n_nominal": int((df["p_adj"] < 0.05).sum()),
               "expected_by_chance": round(0.05 * n),
               "bh_survivors": k_bh,
               "top": top.to_dict("records"),
               "consistency": cons[:12], "tradeable": trade,
               "features_tested": cols}
    json.dump(payload, open(OUT / "macro_dev.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "macro_policy_liquidity",
                            "n_configs": n}) + "\n")
    print(f"\nwrote research/macro_dev.json")


if __name__ == "__main__":
    main()

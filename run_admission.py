"""Strategy admission test (FROZEN_SPEC addendum) + weighting comparison.

Stages: 1 individual, 2 statistical, 3 robustness, 4 diversification (corr vs
baseline), 5 portfolio contribution (add to baseline; must improve >=1 of
Sharpe/Calmar/maxDD on DEV, else REJECT).

Candidates this cycle: multi-speed trend legs (12), TWAP-MR legs (2, k=0.75),
regime gate on trend legs. Weightings compared on the SAME legs: equal,
inverse-correlation (heuristic, formerly mislabeled ERC), TRUE ERC (iterative
risk-contribution equalization, trailing 120d cov, monthly). One holdout look
(#12) for the final chosen book + the baseline weighting comparison.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_superbook as SB
from qlab.metrics import deflated_sharpe, full_metrics
from run_cycle2 import SPEEDS, daily, mm, trend_leg, twap_mr, vol_scale

ROOT = Path(__file__).parent
OUT = ROOT / "research"
LF = ROOT / "results3" / "lowfreq_daily"
DEV = pd.Timestamp("2022-07-01")


def baseline_legs():
    legs = {}
    for f in sorted(LF.glob("trend_*.csv")) + [LF / "xsec_ALL.csv"]:
        legs[f.stem] = pd.read_csv(f, index_col=0, parse_dates=True)["ret"]
    legs["MNQ_pull_C"] = SB.pull_leg_daily()
    legs["COT_NQ_washout"] = SB.cot_leg_daily()
    for k in legs:
        idx = pd.to_datetime(legs[k].index, utc=True)
        legs[k].index = idx.tz_convert("America/New_York").tz_localize(None).normalize()
        legs[k] = SB.vol_scale(legs[k][~legs[k].index.duplicated()])
    return legs


def w_equal(rets):
    return pd.DataFrame(1.0 / rets.shape[1], index=rets.index, columns=rets.columns)


def w_invcorr(rets):
    cs = rets.rolling(120).corr().abs().groupby(level=0).sum().reindex(rets.index)
    w = (1.0 / cs.replace(0, np.nan)).fillna(1.0)
    return w.div(w.sum(axis=1), axis=0)


def w_true_erc(rets, lb=120):
    """Iterative ERC on trailing cov, monthly rebalance, ffilled daily."""
    month = rets.index.to_period("M")
    ends = rets.index.to_series().groupby(month).last()
    W = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for dt in ends:
        loc = rets.index.get_loc(dt)
        if loc < lb:
            continue
        window = rets.iloc[loc - lb:loc].fillna(0.0)
        cov = window.cov().values + 1e-10 * np.eye(rets.shape[1])
        w = np.ones(rets.shape[1]) / rets.shape[1]
        for _ in range(200):
            mrc = cov @ w
            w_new = (w / np.maximum(mrc, 1e-12))
            w_new = np.maximum(w_new, 0)
            w_new /= w_new.sum()
            if np.abs(w_new - w).max() < 1e-8:
                w = w_new
                break
            w = 0.5 * w + 0.5 * w_new
        W.loc[dt] = w
    return W.ffill()


def book(rets, W):
    W = W.shift(1)
    return (W * rets).sum(axis=1).div(W.abs().sum(axis=1)).dropna()


def seg_stats(r, window):
    seg = r[r.index < DEV] if window == "dev" else r[r.index >= DEV]
    m = full_metrics(None, seg)
    return {k: m.get(k) for k in ("sharpe", "calmar", "max_dd_pct", "cagr_pct", "sortino")}


def main():
    base = baseline_legs()
    base_frame = pd.DataFrame(base).fillna(0.0)
    base_book = mm(book(base_frame, w_invcorr(base_frame)))
    b_dev = seg_stats(base_book, "dev")
    print(f"BASELINE (invcorr+mm) dev: {b_dev}")

    insts = ("XAUUSD", "MNQ", "ES", "EURUSD")
    dd = {i: daily(i) for i in insts}
    ms = {f"ms_{i}_{sp}": vol_scale(trend_leg(dd[i], i, lb).dropna())
          for i in insts for sp, lb in SPEEDS.items()}
    tw = {f"twapmr_{i}": vol_scale(twap_mr(i).dropna()) for i in ("XAUUSD", "MNQ")}

    results = {}
    for cand_name, cand in (("multispeed", ms), ("twapmr", tw), ("both", {**ms, **tw})):
        cbook = pd.DataFrame(cand).fillna(0.0).mean(axis=1).dropna()
        corr = base_book.corr(cbook.reindex(base_book.index))
        aug = pd.DataFrame({**base, **cand}).fillna(0.0)
        aug_book = mm(book(aug, w_invcorr(aug)))
        a_dev = seg_stats(aug_book, "dev")
        improves = (a_dev["sharpe"] > b_dev["sharpe"] or a_dev["calmar"] > b_dev["calmar"]
                    or a_dev["max_dd_pct"] > b_dev["max_dd_pct"])
        results[cand_name] = {"corr_vs_base": round(float(corr), 3),
                              "dev": a_dev, "admitted": bool(improves)}
        print(f"+{cand_name:10} corr={corr:+.2f} dev={a_dev} -> "
              f"{'ADMIT' if improves else 'REJECT'}")

    # weighting comparison on the BASELINE legs (dev decision, holdout report)
    print("\nWeighting comparison (baseline legs):")
    weight_rows = {}
    for wname, wfn in (("equal", w_equal), ("invcorr", w_invcorr), ("trueERC", w_true_erc)):
        r = mm(book(base_frame, wfn(base_frame)))
        weight_rows[wname] = {"dev": seg_stats(r, "dev"), "holdout": seg_stats(r, "holdout")}
        h = r[r.index >= DEV]
        weight_rows[wname]["holdout"]["DSR"] = round(deflated_sharpe(h, 12), 3)
        print(f"  {wname:8} dev {weight_rows[wname]['dev']}")
        print(f"           holdout {weight_rows[wname]['holdout']}")

    # final book = baseline + admitted candidates, invcorr (spec) — one holdout look
    admitted = {}
    for name, cand in (("multispeed", ms), ("twapmr", tw)):
        if results[name]["admitted"]:
            admitted.update(cand)
    final_frame = pd.DataFrame({**base, **admitted}).fillna(0.0)
    final = mm(book(final_frame, w_invcorr(final_frame)))
    fh_ = seg_stats(final, "holdout")
    fh_["DSR"] = round(deflated_sharpe(final[final.index >= DEV], 12), 3)
    print(f"\nFINAL BOOK (baseline + admitted): dev {seg_stats(final, 'dev')}")
    print(f"FINAL BOOK holdout: {fh_}")
    with open(OUT / "admission_results.json", "w") as fp:
        json.dump({"baseline_dev": b_dev, "candidates": results,
                   "weighting": weight_rows, "final_holdout": fh_},
                  fp, indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as fp:
        fp.write(json.dumps({"phase": "admission", "holdout_look": 12,
                             "n_configs": 17}) + "\n")
    final.to_csv(OUT / "finalbook_daily.csv", header=["ret"])


if __name__ == "__main__":
    main()

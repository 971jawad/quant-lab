"""Cycle 12b — apply the DECLARED economic prior: no shorting where a structural
risk premium exists.

Prior stated before testing (not data-mined): equity indices carry a documented
equity risk premium => a systematic short in them fights a known positive drift.
Cycle-12 attribution measured that drag directly: -108% total short P&L, of
which the equity indices are the worst offenders. FX and commodities have no
such drift and keep both sides.

Restriction: trend legs on MNQ / ES / JPXJPY are LONG-ONLY. Everything else in
the champion is untouched: same walk-forward lookback selection, same vol
targeting, same strength weights, same Moreira-Muir overlay.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_superbook as SB
from qlab.metrics import deflated_sharpe, full_metrics
from run_admission import baseline_legs, book, mm, seg_stats, w_invcorr
from run_breadth import new_trend_legs
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV = pd.Timestamp("2022-07-01")
DRIFT = ("MNQ", "ES", "JPXJPY")
LOOKBACKS = [40, 80, 160, 240]
WARMUP, MIN_TRAIN, TEST_LEN = 260, 750, 250


def wf_trend_leg(inst, long_only):
    """Same anchored walk-forward as the frozen architecture: lookback chosen
    per fold on TRAINING Sharpe only."""
    d = daily(inst)
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    cand = {}
    for L in LOOKBACKS:
        sig = np.sign(d["close"].pct_change(L))
        if long_only:
            sig = sig.clip(lower=0)
        pos = (sig * (0.10 / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
        turn = pos.diff().abs().fillna(pos.abs())
        cand[L] = (pos * r - turn * COST[inst] / d["close"])
    n, oos, te = len(d), [], WARMUP + MIN_TRAIN
    while te < n - 100:
        end = min(te + TEST_LEN, n)
        best, bs = None, -np.inf
        for L, s in cand.items():
            tr = s.iloc[WARMUP:te].dropna()
            sh = tr.mean() / (tr.std() + 1e-12) if len(tr) > 30 else -np.inf
            if sh > bs:
                bs, best = sh, L
        oos.append(cand[best].iloc[te:end])
        te = end
    return pd.concat(oos).dropna() if oos else pd.Series(dtype=float)


def w_strength(rets, lb=756, floor=0.1):
    mu = rets.rolling(lb).mean()
    sd = rets.rolling(lb).std()
    sh = (mu / (sd + 1e-12) * np.sqrt(252)).clip(lower=0.0)
    w = (sh.fillna(0.0) + floor) * w_invcorr(rets)
    return w.div(w.sum(axis=1), axis=0)


def boot(r, B=2000, L=20):
    rng = np.random.default_rng(7)
    n, v, sh = len(r), r.values, []
    for _ in range(B):
        i = []
        while len(i) < n:
            s0 = rng.integers(0, n)
            i.extend([(s0 + k) % n for k in range(min(L, n - len(i)))])
        s = v[i]
        sh.append(s.mean() / (s.std() + 1e-12) * np.sqrt(252))
    sh = np.array(sh)
    return round(np.percentile(sh, 5), 2), round(np.percentile(sh, 95), 2)


def main():
    base = baseline_legs()
    br = new_trend_legs(final=True)
    br.pop("trend_XAGUSD", None)
    br.pop("trend_GRXEUR", None)
    champ = {**base, **br}
    # non-trend legs are untouched
    keep = {k: v for k, v in champ.items() if not k.startswith("trend_")}

    variants = {}
    for name, restrict in (("v1.2 (long/short)", False), ("v1.4 (long-only on drift)", True)):
        legs = dict(keep)
        for inst in MKT:
            lo = restrict and inst in DRIFT
            legs[f"trend_{inst}"] = SB.vol_scale(wf_trend_leg(inst, lo))
        fr = pd.DataFrame(legs).fillna(0.0)
        r = mm(book(fr, w_strength(fr)))
        variants[name] = r
        for w in ("dev", "holdout"):
            seg = r[r.index < DEV] if w == "dev" else r[r.index >= DEV]
            m = full_metrics(None, seg)
            res = {k: m.get(k) for k in ("sharpe", "sortino", "calmar", "max_dd_pct", "cagr_pct")}
            if w == "holdout":
                lo_, hi_ = boot(seg)
                res["CI90"] = f"[{lo_},{hi_}]"
                res["DSR"] = round(deflated_sharpe(seg, 17), 3)
            print(f"{name:28} {w:8} {res}")
        if restrict:
            r.to_csv(OUT / "champion_v14_daily.csv", header=["ret"])
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "longonly_drift", "holdout_look": 17,
                            "n_configs": 2}) + "\n")


if __name__ == "__main__":
    main()

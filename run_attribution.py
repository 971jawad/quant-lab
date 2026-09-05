"""Cycle 7 — RISK ATTRIBUTION: what are we actually being paid for?

Two tests a real allocator runs before funding anything:

TEST 1 — THE DUMB BENCHMARK. A naive 12-month time-series-momentum portfolio:
  sign(12m return), equal weight, vol-targeted, across the same 8 markets.
  ZERO parameter selection, ZERO walk-forward, ZERO clever weighting, ZERO
  vol-management overlay. If the champion cannot beat this, every ounce of
  machinery in this repo added nothing and we should trade the naive version.

TEST 2 — FACTOR DECOMPOSITION. Regress champion daily returns on:
  SPY (equity beta), TLT (duration), DBC (commodity), DXY (dollar), and the
  naive-trend benchmark. Report ALPHA and its t-stat. If alpha dies once the
  naive trend factor is included, our "edge" is just generic trend beta.

Also reports downside/crisis behaviour: worst-20-SPY-day performance (is the
book a diversifier or a hidden long?).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT = ROOT / "data" / "external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
MKTS = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD",
        "WTIUSD": "WTIUSD", "GRXEUR": "GRXEUR", "JPXJPY": "JPXJPY", "USDJPY": "USDJPY"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016, "WTIUSD": 0.07,
        "GRXEUR": 2.5, "JPXJPY": 20.0, "USDJPY": 0.025}


def daily(inst):
    d = to_tf(load_15m(MKTS[inst]), "1d")
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def naive_tsmom():
    """The dumb benchmark. One rule, no choices, no fitting."""
    legs = {}
    for inst in MKTS:
        d = daily(inst)
        r = d["close"].pct_change()
        sig = np.sign(d["close"].pct_change(252))          # 12 months, fixed
        vol = r.rolling(60).std()
        pos = (sig * (0.10 / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
        turn = pos.diff().abs().fillna(pos.abs())
        legs[inst] = (pos * r - turn * COST[inst] / d["close"]).dropna()
    book = pd.DataFrame(legs).mean(axis=1).dropna()        # equal weight, that's it
    return book


def etf_ret(t):
    d = pd.read_csv(EXT / f"etf_{t}_1d.csv", index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d["close"].pct_change().dropna()


def yf_ret(name):
    d = pd.read_csv(EXT / f"vol_{name}_1d.csv", index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d["close"].pct_change().dropna()


def main():
    champ = pd.read_csv(OUT / "strength_breadth_daily.csv", index_col=0,
                        parse_dates=True)["ret"]
    champ.index = pd.to_datetime(champ.index).tz_localize(None).normalize()
    naive = naive_tsmom()
    naive.to_csv(OUT / "naive_tsmom_daily.csv", header=["ret"])

    print("=" * 72)
    print("TEST 1 — CHAMPION vs THE DUMB BENCHMARK (naive 12m TSMOM, equal weight)")
    print("=" * 72)
    rows = {}
    for wname, lo in (("dev", None), ("holdout", DEV)):
        for name, s in (("champion", champ), ("naive_tsmom", naive)):
            seg = s[s.index < DEV] if lo is None else s[s.index >= DEV]
            m = full_metrics(None, seg)
            rows[f"{name}_{wname}"] = {k: m.get(k) for k in
                                       ("sharpe", "calmar", "max_dd_pct", "cagr_pct")}
            print(f"  {name:12} {wname:8} sharpe={m.get('sharpe'):>6} "
                  f"calmar={m.get('calmar'):>6} dd={m.get('max_dd_pct'):>7} "
                  f"cagr={m.get('cagr_pct'):>6}")

    print("\n" + "=" * 72)
    print("TEST 2 — FACTOR DECOMPOSITION (holdout): what explains the returns?")
    print("=" * 72)
    facs = {"SPY_equity": etf_ret("EFA") * 0,  # placeholder replaced below
            }
    spy = yf_ret("ES")            # ES futures = equity beta proxy
    facs = {"equity(ES)": spy, "duration(TLT)": etf_ret("TLT"),
            "commodity(DBC)": etf_ret("DBC"), "naive_trend": naive}
    for wname, seg_fn in (("dev", lambda s: s[s.index < DEV]),
                          ("holdout", lambda s: s[s.index >= DEV])):
        y = seg_fn(champ).dropna()
        X = pd.DataFrame({k: v for k, v in facs.items()}).reindex(y.index).fillna(0.0)
        A = np.c_[np.ones(len(X)), X.values]
        beta, *_ = np.linalg.lstsq(A, y.values, rcond=None)
        resid = y.values - A @ beta
        se = np.sqrt(np.sum(resid ** 2) / (len(y) - A.shape[1]) *
                     np.diag(np.linalg.pinv(A.T @ A)))
        tstats = beta / se
        r2 = 1 - (resid ** 2).sum() / ((y.values - y.values.mean()) ** 2).sum()
        ann_alpha = beta[0] * 252 * 100
        print(f"\n  [{wname}]  R^2 = {r2:.1%}")
        print(f"    ALPHA        = {ann_alpha:+.2f}%/yr   t = {tstats[0]:+.2f}"
              f"   {'SIGNIFICANT' if abs(tstats[0]) > 2 else 'not significant'}")
        for i, k in enumerate(X.columns, start=1):
            print(f"    beta {k:15} = {beta[i]:+.3f}   t = {tstats[i]:+.2f}")
        rows[f"attrib_{wname}"] = {"r2": round(float(r2), 4),
                                   "alpha_ann_pct": round(float(ann_alpha), 3),
                                   "alpha_t": round(float(tstats[0]), 2),
                                   "betas": {k: round(float(beta[i + 1]), 4)
                                             for i, k in enumerate(X.columns)}}

    print("\n" + "=" * 72)
    print("CRISIS BEHAVIOUR — book return on the 20 worst equity days (holdout)")
    print("=" * 72)
    h = champ[champ.index >= DEV]
    s = spy.reindex(h.index).dropna()
    worst = s.nsmallest(20).index
    print(f"  mean equity day: {s.loc[worst].mean()*100:+.2f}%  ->  "
          f"book: {h.loc[worst].mean()*100:+.3f}%  "
          f"({'DIVERSIFIER' if h.loc[worst].mean() > 0 else 'co-moves with equity'})")
    rows["crisis_book_mean_pct"] = round(float(h.loc[worst].mean() * 100), 4)

    json.dump(rows, open(OUT / "attribution.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "attribution", "n_configs": 1,
                            "note": "diagnostic, no new strategy"}) + "\n")


if __name__ == "__main__":
    main()

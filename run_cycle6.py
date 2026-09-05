"""Cycle 6 — ASSET-CLASS BREADTH via clean ETF series (fixed income first).

Rationale from 9 rounds of evidence: every FILTER and PREDICTOR tested has
failed; every gain has come from BREADTH + construction. The book's biggest
structural hole is fixed income (real CTA books run 25-30% bonds).

ETFs, not futures, deliberately: continuous total-return series with no roll-gap
artifacts (Yahoo '=F' front-month series carry unadjusted roll jumps that
manufacture fake trends). Costs: conservative 5bp round trip (retail ETF spread
+ commission is typically 1-3bp on these; 5bp is punitive).

Legs = the SAME frozen trend architecture (per-fold lookback selection from
{40,80,160,240} on training Sharpe only, vol-targeted 10%, clip 3, shift(1)).
Dev-only; admission per FROZEN_SPEC Amendment 1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_lowfreq as LFQ
import run_superbook as SB
from qlab.metrics import full_metrics
from run_admission import baseline_legs, book, mm, seg_stats, w_invcorr
from run_breadth import new_trend_legs

ROOT = Path(__file__).parent
EXT, OUT = ROOT / "data" / "external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
ETFS = ["TLT", "IEF", "HYG", "EEM", "EFA", "VNQ", "DBC"]
COST_BPS = 5.0


def etf_daily(t):
    d = pd.read_csv(EXT / f"etf_{t}_1d.csv", index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[["open", "high", "low", "close"]]


def w_strength(rets, lb=756, floor=0.1):
    mu = rets.rolling(lb).mean(); sd = rets.rolling(lb).std()
    sh = (mu / (sd + 1e-12) * np.sqrt(252)).clip(lower=0.0)
    w = (sh.fillna(0.0) + floor) * w_invcorr(rets)
    return w.div(w.sum(axis=1), axis=0)


def etf_trend_leg(t, final=False):
    d = etf_daily(t)
    if not final:
        d = d[d.index < DEV]
    LFQ.COST_PTS[t] = 0.0            # cost applied here in bps, not points
    n = len(d)
    r = d["close"].pct_change()
    cand = {}
    for L in (40, 80, 160, 240):
        sig = np.sign(d["close"].pct_change(L))
        vol = r.rolling(60).std()
        pos = (sig * (0.10 / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
        turn = pos.diff().abs().fillna(pos.abs())
        cand[L] = (pos * r - turn * COST_BPS / 1e4).rename(t)
    # anchored walk-forward lookback selection on TRAINING Sharpe only
    oos = []
    te = 260 + 750
    while te < n - 100:
        end = min(te + 250, n)
        best, bs = None, -np.inf
        for L, s in cand.items():
            tr = s.iloc[260:te].dropna()
            sh = tr.mean() / (tr.std() + 1e-12) if len(tr) > 30 else -np.inf
            if sh > bs:
                bs, best = sh, L
        oos.append(cand[best].iloc[te:end])
        te = end
    return pd.concat(oos) if oos else pd.Series(dtype=float)


def main():
    print("STAGE 1 — ETF trend legs (dev):")
    legs = {}
    for t in ETFS:
        s = etf_trend_leg(t).dropna()
        m = full_metrics(None, s)
        sh = m.get("sharpe")
        print(f"  trend_{t:5} sharpe={sh:>6} calmar={m.get('calmar'):>6} "
              f"dd={m.get('max_dd_pct'):>7}  {'PASS' if (sh or -9) > 0 else 'fail'}")
        if (sh or -9) > 0:
            legs[f"trend_{t}"] = SB.vol_scale(s)

    base = baseline_legs()
    br = new_trend_legs(final=False); br.pop("trend_XAGUSD", None)
    champ_legs = {**base, **br}
    cf = pd.DataFrame(champ_legs).fillna(0.0)
    champ = mm(book(cf, w_strength(cf)))
    c = seg_stats(champ, "dev")
    print(f"\nCHAMPION v1.1 dev: {c}")

    if not legs:
        print("no ETF legs passed Stage 1")
        return
    blk = pd.DataFrame(legs).fillna(0.0).mean(axis=1).dropna()
    corr = champ.corr(blk.reindex(champ.index))
    print(f"Stage 4: corr(ETF block, champion) = {corr:+.3f}  [{len(legs)} legs]")

    aug = pd.DataFrame({**champ_legs, **legs}).fillna(0.0)
    a = seg_stats(mm(book(aug, w_strength(aug))), "dev")
    dom = ((a["sharpe"] >= c["sharpe"] or a["calmar"] >= c["calmar"]
            or a["max_dd_pct"] >= c["max_dd_pct"])
           and a["sharpe"] >= 0.9 * c["sharpe"] and a["calmar"] >= 0.9 * c["calmar"])
    print(f"Stage 5: +ETF block dev: {a} -> {'ADMIT' if dom else 'REJECT'}")

    # bonds-only variant (the specific structural hole)
    bonds = {k: v for k, v in legs.items() if k.split('_')[1] in ("TLT", "IEF")}
    if bonds:
        augb = pd.DataFrame({**champ_legs, **bonds}).fillna(0.0)
        ab = seg_stats(mm(book(augb, w_strength(augb))), "dev")
        domb = ((ab["sharpe"] >= c["sharpe"] or ab["calmar"] >= c["calmar"]
                 or ab["max_dd_pct"] >= c["max_dd_pct"])
                and ab["sharpe"] >= 0.9 * c["sharpe"] and ab["calmar"] >= 0.9 * c["calmar"])
        print(f"Stage 5: +BONDS only dev: {ab} -> {'ADMIT' if domb else 'REJECT'}")
        json.dump({"etf_block": a, "bonds_only": ab, "champion": c},
                  open(OUT / "cycle6_dev.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "cycle6_etf", "n_configs": len(ETFS) * 4}) + "\n")


if __name__ == "__main__":
    main()

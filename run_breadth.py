"""Breadth expansion (cycle 3): same FROZEN trend architecture, more markets.

New legs: trend_{XAGUSD, WTIUSD, GRXEUR, JPXJPY, USDJPY} via run_lowfreq's
wf_trend (per-fold lookback selection, vol-targeted 10%, leverage clip 3) and
an EXPANDED xsec across all available markets. Costs conservative, declared
below, never revised downward.

Admission per FROZEN_SPEC Amendment 1: Stage-1 individual dev stats, Stage-4
corr vs baseline, Stage-5 dominance-aware portfolio contribution on DEV.
Admitted legs join the book; ONE holdout look (#13) for the expanded book.

    python run_breadth.py            # verify + dev admission
    python run_breadth.py --final    # the single holdout look
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_lowfreq as LFQ
import run_superbook as SB
from qlab.metrics import deflated_sharpe, full_metrics
from run_admission import baseline_legs, book, mm, w_invcorr

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")

# per-SIDE cost in price points (half-spread + slip + comm), conservative
NEW_COSTS = {"XAGUSD": 0.02, "WTIUSD": 0.035, "GRXEUR": 1.25,
             "JPXJPY": 10.0, "USDJPY": 0.0125}
NEW_SERIES = list(NEW_COSTS)


def verify(name: str) -> dict:
    df = pd.read_csv(DATA / f"{name}_15m.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    yr = df.groupby(df.index.year).size()
    bad = ((df["high"] < df[["open", "close"]].max(axis=1)) |
           (df["low"] > df[["open", "close"]].min(axis=1))).sum()
    r = np.log(df["close"] / df["close"].shift(1))
    # LEVEL-CONTINUITY CHECK (added after the GRXEUR contamination: HistData
    # spliced Euro Stoxx 50 levels into 21% of the DAX series). A genuine index
    # never halves or doubles its level regime; flag any quarter whose median
    # level is <60% or >167% of the adjacent quarter's.
    q = df["close"].resample("QE").median().dropna()
    ratio = (q / q.shift(1)).dropna()
    level_breaks = int(((ratio < 0.6) | (ratio > 1.67)).sum())
    return {"series": name, "bars": len(df), "level_breaks": level_breaks,
            "range": f"{df.index[0].date()}->{df.index[-1].date()}",
            "ohlc_violations": int(bad), "dups": int(df.index.duplicated().sum()),
            "absret_gt5pct": int((r.abs() > 0.05).sum()),
            "median_bars_per_year": int(yr.median())}


def new_trend_legs(final: bool) -> dict:
    LFQ.COST_PTS.update(NEW_COSTS)
    legs = {}
    for name in NEW_SERIES:
        if not (DATA / f"{name}_15m.csv").exists():
            print(f"  {name}: data missing, skipped")
            continue
        LFQ.DATA_MAP[name] = name
        d = LFQ.to_daily(name)
        if not final:
            d = d[d.index < DEV.tz_localize("America/New_York")]
        s, _ = LFQ.wf_trend(d, name)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        legs[f"trend_{name}"] = SB.vol_scale(s[~s.index.duplicated()])
    return legs


def seg(r, window):
    s = r[r.index < DEV] if window == "dev" else r[r.index >= DEV]
    m = full_metrics(None, s)
    return {k: m.get(k) for k in ("sharpe", "calmar", "max_dd_pct", "cagr_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    print("== data verification ==")
    for name in NEW_SERIES:
        if (DATA / f"{name}_15m.csv").exists():
            print(" ", verify(name))

    base = baseline_legs()
    base_frame = pd.DataFrame(base).fillna(0.0)
    base_book = mm(book(base_frame, w_invcorr(base_frame)))
    b = seg(base_book, "dev")
    print(f"\nBASELINE dev: {b}")

    legs = new_trend_legs(final=args.final)
    rows = {}
    for k, v in legs.items():
        rows[k] = seg(v, "dev")
        c = base_book.corr(v.reindex(base_book.index))
        rows[k]["corr_vs_base"] = round(float(c), 3)
        print(f"  {k:16} dev {rows[k]}")

    # Stage 5: add ALL new trend legs as one breadth block (declared: breadth
    # is one decision, not per-leg cherry-picking)
    aug = pd.DataFrame({**base, **legs}).fillna(0.0)
    aug_book = mm(book(aug, w_invcorr(aug)))
    a = seg(aug_book, "dev")
    admit = ((a["sharpe"] >= b["sharpe"] or a["calmar"] >= b["calmar"]
              or a["max_dd_pct"] >= b["max_dd_pct"])
             and a["sharpe"] >= 0.9 * b["sharpe"]
             and a["calmar"] >= 0.9 * b["calmar"])
    print(f"\nBREADTH BOOK dev: {a} -> {'ADMIT' if admit else 'REJECT'}")

    out = {"verify": [verify(n) for n in NEW_SERIES if (DATA / f'{n}_15m.csv').exists()],
           "legs_dev": rows, "baseline_dev": b, "breadth_dev": a,
           "admitted": bool(admit)}
    if args.final and admit:
        h = seg(aug_book, "holdout")
        h["DSR"] = round(deflated_sharpe(aug_book[aug_book.index >= DEV], 13), 3)
        hb = seg(base_book, "holdout")
        print(f"BASELINE holdout: {hb}")
        print(f"BREADTH  holdout: {h}")
        out["baseline_holdout"] = hb
        out["breadth_holdout"] = h
        aug_book.to_csv(OUT / "breadthbook_daily.csv", header=["ret"])
        with open(OUT / "ledger.jsonl", "a") as fp:
            fp.write(json.dumps({"phase": "breadth", "holdout_look": 13,
                                 "n_configs": len(legs) * 4}) + "\n")
    with open(OUT / "breadth_results.json", "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    print("wrote research/breadth_results.json")


if __name__ == "__main__":
    main()

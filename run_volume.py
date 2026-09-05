"""Cycle 4 — VOLUME / AUCTION legs. Real volume, sourced externally.

Volume was previously called untestable; that was true only of HistData quote
files. Sources now on disk: CME front-month futures daily volume via Yahoo
(NQ/ES/CL/6E — magnitudes verified), GLD ETF volume as the gold activity proxy
(Yahoo GC=F volume is broken: median 189 contracts/day), and CFTC open interest
(weekly, already held).

Volume is used as a SIGNAL INPUT only; price and execution stay on the verified
HistData series so costs stay identical to the rest of the book.

Legs (all declared before testing, canonical, no grids):
  voltrend_X   Wyckoff/VSA participation rule: hold the 80d TSMOM position ONLY
               when volume is above its own 20d average (moves on expanding
               participation continue; on drying volume they fail). Else flat.
  volclimax_X  Exhaustion fade: volume > +2.5 sigma (60d) AND |ret| > 2 sigma
               -> take the OPPOSITE side next open, hold 5 days.
  oitrend_X    New-money confirmation (weekly COT): hold the trend position only
               when open interest is rising 4w (new money entering, not
               short-covering). NQ / EUR / GOLD.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics
from run_research import SERIES, load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT = ROOT / "data" / "external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
TARGET_VOL = 0.10
# price series we trade -> external volume source
VOLMAP = {"XAUUSD": "GLD", "MNQ": "NQ", "ES": "ES", "EURUSD": "6E", "WTIUSD": "CL"}
PSER = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD",
        "EURUSD": "EURUSD", "WTIUSD": "WTIUSD"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016, "WTIUSD": 0.07}
COT_SYM = {"MNQ": "NQ", "EURUSD": "EUR", "XAUUSD": "GOLD"}


def price(inst):
    d = to_tf(load_15m(PSER[inst]), "1d")
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def volume(inst, idx):
    df = pd.read_csv(EXT / f"vol_{VOLMAP[inst]}_1d.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df["volume"].reindex(idx).ffill()


def vt(pos, d, inst):
    """position -> net daily returns, vol-targeted, costs charged on turnover."""
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (pos * (TARGET_VOL / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * (COST[inst] / d["close"])).dropna()


def voltrend(inst):
    d = price(inst)
    v = volume(inst, d.index)
    sig = np.sign(d["close"].pct_change(80))
    participating = (v > v.rolling(20).mean()).astype(float)
    return vt(sig * participating, d, inst).rename(f"voltrend_{inst}")


def volclimax(inst):
    d = price(inst)
    v = volume(inst, d.index)
    r = d["close"].pct_change()
    vz = (v - v.rolling(60).mean()) / v.rolling(60).std()
    rz = r / r.rolling(60).std()
    trig = (vz > 2.5) & (rz.abs() > 2.0)
    pos = pd.Series(0.0, index=d.index)
    sign = -np.sign(rz).where(trig, 0.0)
    for i in np.flatnonzero(trig.fillna(False).values):
        pos.iloc[i:i + 5] = sign.iloc[i]
    return vt(pos, d, inst).rename(f"volclimax_{inst}")


def oitrend(inst):
    d = price(inst)
    cot = pd.read_csv(ROOT / "data/external/cot_weekly.csv", parse_dates=["date"])
    c = cot[cot["sym"] == COT_SYM[inst]].set_index("date").sort_index()
    # report as-of Tue, public Fri -> usable from the following Monday
    oi_rising = (c["oi"].diff(4) > 0)
    oi_rising.index = oi_rising.index + pd.Timedelta(days=6)
    gate = oi_rising.reindex(d.index, method="ffill").fillna(False).astype(float)
    sig = np.sign(d["close"].pct_change(80))
    return vt(sig * gate, d, inst).rename(f"oitrend_{inst}")


def main():
    legs = {}
    for inst in VOLMAP:
        legs[f"voltrend_{inst}"] = voltrend(inst)
        legs[f"volclimax_{inst}"] = volclimax(inst)
    for inst in COT_SYM:
        legs[f"oitrend_{inst}"] = oitrend(inst)

    print("STAGE 1 (dev, individual):")
    rows = {}
    for k, v in legs.items():
        seg = v[v.index < DEV].dropna()
        m = full_metrics(None, seg)
        rows[k] = {x: m.get(x) for x in ("sharpe", "calmar", "max_dd_pct", "cagr_pct")}
        ok = (m.get("sharpe") or -9) > 0
        print(f"  {k:22} sharpe={m.get('sharpe'):>6} calmar={m.get('calmar'):>6} "
              f"dd={m.get('max_dd_pct'):>7}  {'PASS' if ok else 'fail'}")
        v.to_csv(OUT / f"volleg_{k}.csv", header=["ret"])
    with open(OUT / "volume_stage1.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "volume", "n_configs": len(legs)}) + "\n")


if __name__ == "__main__":
    main()

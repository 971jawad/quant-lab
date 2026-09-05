"""Cycle 5 — three genuinely untested directions, all declared before testing.

A. TIMEFRAME EXTENSION (user question): the 15m->1h->1d gradient was monotone.
   Does it keep improving past daily? Test the SAME trend architecture on
   2d / 3d / weekly / 2-week bars. Lookbacks scaled to keep the economic
   horizon constant (~40/80/160/240 calendar days everywhere).

B. FX CARRY (largest documented FX factor; never tested here). Signal = 3-month
   interest-rate differential. Rates: US DTB3 (daily), EZ/JP/GB 3m interbank
   (monthly, ffilled). All are public on release -> shift(1) is sufficient.
   Both the per-pair version and a cross-sectional long-high/short-low book.

C. VIX TERM STRUCTURE regime (VIX/VIX3M). Contango (<1) = calm -> hold equity;
   backwardation (>1) = stress -> flat. Applied to ES and MNQ.

Dev window only. No grids. Stage 1 -> Stage 4/5 vs champion in the next step.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT = ROOT / "data" / "external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
TARGET_VOL = 0.10
PSER = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD", "USDJPY": "USDJPY", "WTIUSD": "WTIUSD"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016,
        "GBPUSD": 0.00022, "USDJPY": 0.025, "WTIUSD": 0.07}


def fred(sid):
    df = pd.read_csv(EXT / f"{sid}.csv")
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[sid], errors="coerce").dropna()


def bars(inst, rule):
    d15 = load_15m(PSER[inst]).tz_convert("America/New_York")
    g = d15.resample(rule)
    o = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last()}).dropna()
    o.index = o.index.tz_localize(None).normalize()
    return o


def vt(pos, d, inst, per_year):
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (pos * (TARGET_VOL / np.sqrt(per_year)) / vol).clip(-3, 3).shift(1)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * (COST[inst] / d["close"])).dropna()


# ---------------- A. timeframe extension ----------------
TFS = {"1d": ("1D", 1, 252), "2d": ("2D", 2, 126), "3d": ("3D", 3, 84),
       "1w": ("W-FRI", 5, 52), "2w": ("2W-FRI", 10, 26)}


def timeframe_scan():
    print("A. TIMEFRAME EXTENSION (dev, trend architecture, horizon held constant)")
    rows = []
    for tf, (rule, days, per_year) in TFS.items():
        for inst in ("XAUUSD", "MNQ", "ES", "EURUSD"):
            d = bars(inst, rule)
            d = d[d.index < DEV]
            per_leg = []
            for cal_days in (40, 80, 160, 240):
                lb = max(int(round(cal_days / days)), 2)
                s = vt(np.sign(d["close"].pct_change(lb)), d, inst, per_year)
                m = full_metrics(None, s)
                per_leg.append(m.get("sharpe") or np.nan)
            rows.append({"tf": tf, "inst": inst, "mean_sharpe": np.nanmean(per_leg),
                         "best_sharpe": np.nanmax(per_leg), "bars": len(d)})
    df = pd.DataFrame(rows)
    g = df.groupby("tf").agg(mean_sharpe=("mean_sharpe", "mean"),
                             best_sharpe=("best_sharpe", "max"),
                             pct_pos=("mean_sharpe", lambda s: (s > 0).mean() * 100))
    print(g.reindex(list(TFS)).round(3).to_string())
    df.to_csv(OUT / "timeframe_scan.csv", index=False)
    return df


# ---------------- B. FX carry ----------------
RATE = {"USD": "DTB3", "EUR": "IR3TIB01EZM156N",
        "JPY": "IR3TIB01JPM156N", "GBP": "IR3TIB01GBM156N"}
PAIR_LEGS = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
             "USDJPY": ("USD", "JPY")}


def carry_legs():
    print("\nB. FX CARRY (dev)")
    rates = {k: fred(v) for k, v in RATE.items()}
    out = {}
    for pair, (base, quote) in PAIR_LEGS.items():
        d = bars(pair, "1D")
        rb = rates[base].reindex(d.index, method="ffill")
        rq = rates[quote].reindex(d.index, method="ffill")
        diff = (rb - rq).shift(1)              # public on release; lag anyway
        s = vt(np.sign(diff), d, pair, 252)
        s = s[s.index < DEV]
        m = full_metrics(None, s)
        out[f"carry_{pair}"] = s
        print(f"  carry_{pair:8} sharpe={m.get('sharpe'):>6} calmar={m.get('calmar'):>6} "
              f"dd={m.get('max_dd_pct'):>7}")
    return out


# ---------------- C. VIX term structure ----------------
def vixts_legs():
    print("\nC. VIX TERM STRUCTURE (dev)")
    vix, vix3m = fred("VIXCLS"), fred("VXVCLS")
    out = {}
    for inst in ("ES", "MNQ"):
        d = bars(inst, "1D")
        ratio = (vix / vix3m.reindex(vix.index).ffill()).reindex(d.index).ffill()
        calm = (ratio < 1.0).astype(float).shift(1)      # contango -> hold equity
        s = vt(calm, d, inst, 252)
        s = s[s.index < DEV]
        m = full_metrics(None, s)
        out[f"vixts_{inst}"] = s
        print(f"  vixts_{inst:9} sharpe={m.get('sharpe'):>6} calmar={m.get('calmar'):>6} "
              f"dd={m.get('max_dd_pct'):>7}")
        # benchmark: always-long, same vol target
        bm = vt(pd.Series(1.0, index=d.index), d, inst, 252)
        bm = bm[bm.index < DEV]
        print(f"    (benchmark always-long: sharpe={full_metrics(None, bm).get('sharpe')})")
    return out


if __name__ == "__main__":
    timeframe_scan()
    legs = {**carry_legs(), **vixts_legs()}
    for k, v in legs.items():
        v.to_csv(OUT / f"c5leg_{k}.csv", header=["ret"])
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "cycle5", "n_configs": 20 + len(legs)}) + "\n")

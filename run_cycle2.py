"""Research cycle 2 (dev-only until the single declared holdout look).

New engines, all declared a priori:
  A. MULTI-SPEED TREND (AHL): fast(20)/med(80)/slow(240) daily TSMOM per
     instrument, fixed lookbacks (no per-fold selection at all), vol-targeted.
  B. TWAP-DEVIATION MR: intraday, path-honest on 15m bars. Day's running TWAP
     (cumulative mean of 15m closes, ET day); enter when price deviates
     > 2 x ATR(20d) from TWAP during 8-15 ET; exit at TWAP touch or day close.
     Long and short. TRUE VWAP/volume-profile is untestable (no volume).
  C. REGIME ALLOCATOR: agreement score s = |sign(mom20)+sign(mom80)+
     sign(mom240)| / 3 per instrument. Trend legs scaled by s, MR legs by
     (1 - s). Simple, no ML, no fitted thresholds.

Adoption rule (FROZEN_SPEC addendum): a new engine joins the book only if it
improves the DEV book on >=1 of {Sharpe, Calmar, maxDD, tail}; then ONE
holdout look for the upgraded book (look #12).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics, summarize_line
from run_research import SERIES, load_15m, to_tf

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01")
COST_PTS = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016}
DSER = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD"}
TARGET_VOL = 0.10
SPEEDS = {"fast": 20, "med": 80, "slow": 240}


def daily(inst):
    d = to_tf(load_15m(DSER[inst]), "1d")
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def trend_leg(d, inst, lb):
    r = d["close"].pct_change()
    sig = np.sign(d["close"].pct_change(lb))
    vol = r.rolling(60).std()
    pos = (sig * (TARGET_VOL / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
    cost = COST_PTS[inst] / d["close"]
    turn = pos.diff().abs().fillna(0)
    return (pos * r - turn * cost).rename(f"{inst}_{lb}")


def twap_mr(inst, k=0.75):   # k in daily-ATR units; 2.0 was a scale error (25 trades/16y)
    """Path-honest 15m TWAP-deviation reversion, one entry/side/day."""
    b = load_15m(DSER[inst])
    et = b.index.tz_convert("America/New_York")
    day = pd.Series(et.date, index=b.index)
    c = b["close"]
    twap = c.groupby(day.values).expanding().mean().reset_index(level=0, drop=True)
    twap = twap.reindex(b.index)
    dr = to_tf(load_15m(DSER[inst]), "1d")
    atr = (dr["high"] - dr["low"]).rolling(20).mean().shift(1)
    atr.index = atr.index.tz_convert("America/New_York").tz_localize(None).normalize()
    atr_by_day = pd.Series(pd.to_datetime(pd.Series(day.values)).values).map(
        lambda d_: atr.get(pd.Timestamp(d_), np.nan))
    atr_by_day.index = b.index
    hour = et.hour
    window = (hour >= 8) & (hour < 15)
    dev = (c - twap) / atr_by_day
    cost = COST_PTS[inst]
    rows = {}
    cvals, tvals, dvals, wvals = c.values, twap.values, dev.values, window
    dayvals = day.values
    ret_by_day = {}
    i, n = 0, len(b)
    while i < n:
        d0 = dayvals[i]
        j = i
        pos, entry, pnl, sides = 0, 0.0, 0.0, set()
        while j < n and dayvals[j] == d0:
            if pos == 0 and wvals[j] and not np.isnan(dvals[j]):
                if dvals[j] < -k and "L" not in sides:
                    pos, entry = 1, cvals[j]; sides.add("L")
                elif dvals[j] > k and "S" not in sides:
                    pos, entry = -1, cvals[j]; sides.add("S")
            elif pos != 0:
                if (pos == 1 and cvals[j] >= tvals[j]) or (pos == -1 and cvals[j] <= tvals[j]):
                    pnl += pos * (cvals[j] - entry) - cost
                    pos = 0
            j += 1
        if pos != 0:
            pnl += pos * (cvals[j - 1] - entry) - cost
        if pnl != 0.0 or sides:
            ret_by_day[d0] = pnl / cvals[j - 1]
        i = j
    s = pd.Series(ret_by_day)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().rename(f"{inst}_twapmr")


def vol_scale(r):
    vol = r.rolling(60).std() * np.sqrt(252)
    return (r * (TARGET_VOL / vol).clip(0, 3).shift(1)).dropna()


def mm(r):
    lev = (TARGET_VOL / (r.rolling(20).std() * np.sqrt(252))).clip(0, 2).shift(1)
    return (r * lev).dropna()


def stats_line(name, r, window):
    seg = r[r.index < DEV_END] if window == "dev" else r[r.index >= DEV_END]
    m = full_metrics(None, seg)
    print(summarize_line(f"{name}_{window}", m))
    return m


def main():
    insts = ("XAUUSD", "MNQ", "ES", "EURUSD")
    dd = {i: daily(i) for i in insts}

    # A. multi-speed trend legs
    legs = {}
    for i in insts:
        for sp, lb in SPEEDS.items():
            legs[f"trend_{i}_{sp}"] = trend_leg(dd[i], i, lb)
    # B. TWAP-MR legs (path-honest)
    for i in ("XAUUSD", "MNQ"):
        legs[f"twapmr_{i}"] = twap_mr(i)

    scaled = {k: vol_scale(v.dropna()) for k, v in legs.items()}
    frame = pd.DataFrame(scaled)

    # C. regime allocator weights
    agree = {}
    for i in insts:
        signs = sum(np.sign(dd[i]["close"].pct_change(lb)) for lb in SPEEDS.values())
        agree[i] = (signs.abs() / 3).shift(1)
    w = pd.DataFrame(1.0, index=frame.index, columns=frame.columns)
    for col in frame.columns:
        inst = col.split("_")[1]
        a = agree.get(inst)
        if a is None:
            continue
        a = a.reindex(frame.index).ffill()
        w[col] = a if col.startswith("trend") else (1 - a)

    dev_rows = {}
    print("== dev evaluation ==")
    trend_book = frame[[c for c in frame if c.startswith("trend")]].mean(axis=1).dropna()
    mr_book = frame[[c for c in frame if c.startswith("twapmr")]].mean(axis=1).dropna()
    corr = trend_book.corr(mr_book.reindex(trend_book.index))
    print(f"corr(multi-speed trend book, TWAP-MR book) = {corr:.3f}")
    combo_eq = frame.mean(axis=1).dropna()
    combo_regime = (frame * w).sum(axis=1).div(w.sum(axis=1)).dropna()
    for name, r in [("ms_trend_book", trend_book), ("twapmr_book", mr_book),
                    ("cycle2_eq", mm(combo_eq)), ("cycle2_regime", mm(combo_regime))]:
        dev_rows[name] = stats_line(name, r, "dev")
    combo_eq.to_csv(OUT / "cycle2_eq_daily.csv", header=["ret"])
    combo_regime.to_csv(OUT / "cycle2_regime_daily.csv", header=["ret"])
    with open(OUT / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps({"phase": "cycle2", "n_configs": len(legs) + 2,
                             "note": "multi-speed trend + TWAP-MR + regime gate"}) + "\n")
    print("dev artifacts written; holdout look deferred to explicit --final")


if __name__ == "__main__":
    main()

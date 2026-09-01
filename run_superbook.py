"""SUPER-BOOK: combine every leg that survived the program into one
vol-managed portfolio - the way multi-strat desks actually extract Sharpe.

Legs (all previously built leak-free, full-sample walk-forward daily returns):
  trend_ES / trend_NQ / trend_XAUUSD / trend_EURUSD / xsec_ALL   (lowfreq)
  MNQ_1d_pull_C event leg (regenerated full-sample)
  COT_NQ_washout: long NQ 1 week when net-spec z < -1.5 (declared, long-only)

Combination rules FIXED A PRIORI (nothing optimized):
  1. each leg scaled to 10% ann vol using trailing 60d realized vol (lev cap 3)
  2. equal-weight across legs
  3. Moreira-Muir volatility management on the book: scale by
     (10% / trailing 20d book vol)^2 capped at 2x  [Vol-Managed Portfolios, JF 2017]

Report dev (2010->2022-06) and holdout (2022-07->) separately. Declared holdout
promotion criterion: Sharpe > 0.5, max DD > -15%, DSR(holdout frame) > 0.5.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.propstrats as P
import qlab.strategies as S
import qlab.walkforward as WF
from qlab.backtest import simulate
from qlab.metrics import deflated_sharpe, full_metrics, summarize_line
from run_cot import weekly_panel
from run_research import COSTS, TF_CFG, prep

ROOT = Path(__file__).parent
OUT = ROOT / "research"
LF = ROOT / "results3" / "lowfreq_daily"
DEV_END = pd.Timestamp("2022-07-01")
TARGET_VOL = 0.10
HOLDOUT_LOOKS = 8      # models ever evaluated on holdout, incl this round's 2


def pull_leg_daily() -> pd.Series:
    """Full-sample MNQ 1d pull style-C walk-forward daily returns."""
    P.register(WF, S)
    WF.set_scale(**TF_CFG["1d"]["scale"])
    P.set_tf(1)
    bars, feats = prep("MNQ", "1d", final=True)
    res = WF.run_wf(bars, feats, "pull", "C", COSTS["MNQ"], cache_key="NSX1d_book")
    tr = res.oos_trades
    et = bars.index.tz_convert("America/New_York").date
    m = simulate(tr, et, float(tr["risk_pct"].iloc[-1]), True,
                 WF.DAILY_CAP, WF.TRAIL_LIMIT, want_daily=True)
    return m["daily_returns"]


def cot_leg_daily() -> pd.Series:
    p = weekly_panel("NQ", "NSXUSD")
    sig = (p["z"] < -1.5).astype(float)
    wk = sig * p["fwd"] - sig.diff().abs().fillna(sig) * (1.62 / p["entry"])
    daily = wk.repeat(5) / 5
    daily.index = pd.date_range(p.index[0], periods=len(daily), freq="B")
    return daily


def vol_scale(r: pd.Series) -> pd.Series:
    vol = r.rolling(60).std() * np.sqrt(252)
    lev = (TARGET_VOL / vol).clip(0, 3).shift(1)
    return (r * lev).dropna()


def main():
    legs = {}
    for f in sorted(LF.glob("trend_*.csv")) + [LF / "xsec_ALL.csv"]:
        legs[f.stem] = pd.read_csv(f, index_col=0, parse_dates=True)["ret"]
    legs["MNQ_pull_C"] = pull_leg_daily()
    legs["COT_NQ_washout"] = cot_leg_daily()
    for k in legs:
        idx = pd.to_datetime(legs[k].index, utc=True)
        legs[k].index = idx.tz_convert("America/New_York").tz_localize(None).normalize()
        legs[k] = vol_scale(legs[k][~legs[k].index.duplicated()])
    book = pd.DataFrame(legs)
    print("legs:", {k: len(v.dropna()) for k, v in book.items()})
    raw = book.mean(axis=1, skipna=True).dropna()
    # Moreira-Muir overlay on the combined book
    bvol = raw.rolling(20).std() * np.sqrt(252)
    mm = (TARGET_VOL / bvol).clip(0, 2).shift(1)
    managed = (raw * mm).dropna()

    out = {}
    for name, r in (("superbook_raw", raw), ("superbook_volmanaged", managed)):
        for wname, seg in (("dev", r[r.index < DEV_END]),
                           ("holdout", r[r.index >= DEV_END])):
            m = full_metrics(None, seg)
            if wname == "holdout":
                m["DSR_holdout_frame"] = round(deflated_sharpe(seg, HOLDOUT_LOOKS), 4)
            out[f"{name}_{wname}"] = m
            print(summarize_line(f"{name}_{wname}", m),
                  f"DSR={m.get('DSR_holdout_frame','-')}")
    managed.to_csv(OUT / "superbook_daily.csv", header=["ret"])
    with open(OUT / "superbook.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("wrote research/superbook.json")


if __name__ == "__main__":
    main()

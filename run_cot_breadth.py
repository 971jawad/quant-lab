"""Cycle 9 — replicate the ORTHOGONAL ALPHA mechanism across markets.

Attribution (cycle 7) showed the true alpha legs are the positioning/event
family, not trend: COT_NQ_washout carries +9.4%/yr alpha at t=6.28 with
trend-beta -0.01. It is currently applied to ONE market. The CFTC publishes
positioning for dozens.

HYPOTHESIS (single, declared, unchanged from the NQ leg — no re-tuning):
  when speculative net positioning reaches a 2-year z-score below -1.5
  (crowded short), the contract rallies over the following week.
Applied identically to every market where we hold tradeable price data.
Timing unchanged: report as-of Tuesday, public Friday, position from the
following Monday.
"""
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
COTD, EXT, OUT = ROOT / "data/external/cot", ROOT / "data/external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
Z_THRESH = -1.5          # unchanged from the NQ leg

# COT market name -> (our price source, kind, sign, round-trip cost in price pts)
MARKETS = {
    "GOLD - COMMODITY EXCHANGE INC.":              ("XAUUSD", "hist", +1, 0.45),
    "SILVER - COMMODITY EXCHANGE INC.":            ("XAGUSD", "hist", +1, 0.04),
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": ("WTIUSD", "hist", +1, 0.07),
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": ("SPXUSD", "hist", +1, 0.62),
    # CFTC RENAMED these contracts in 2022; without the aliases the legs silently
    # end at 2022-01-31 and contribute nothing to holdout (caught the hard way).
    "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE": ("SPXUSD", "hist", +1, 0.62),
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE":  ("GBPUSD", "hist", +1, 0.00022),
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":        ("EURUSD", "hist", +1, 0.00016),
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": ("GBPUSD", "hist", +1, 0.00022),
    # yen contract rallies => USDJPY FALLS, hence sign -1
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":   ("USDJPY", "hist", -1, 0.025),
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": ("IEF", "etf", +1, 0.02),
    "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE": ("NSXUSD", "hist", +1, 1.62),
}


def load_cot():
    frames = []
    for y in range(2010, 2027):
        z = zipfile.ZipFile(COTD / f"deacot{y}.zip")
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        nm = df["Market and Exchange Names"].str.strip()
        m = df[nm.isin(MARKETS)].copy()
        m["mkt"] = nm[nm.isin(MARKETS)]
        frames.append(m)
    c = pd.concat(frames)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    c = c.rename(columns={"Open Interest (All)": "oi",
                          "Noncommercial Positions-Long (All)": "ncl",
                          "Noncommercial Positions-Short (All)": "ncs"})
    return c[["date", "mkt", "oi", "ncl", "ncs"]].sort_values(["mkt", "date"])


def price(src, kind):
    if kind == "etf":
        d = pd.read_csv(EXT / f"etf_{src}_1d.csv", index_col=0, parse_dates=True)["close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    d = to_tf(load_15m(src), "1d")["close"]
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def washout_leg(cot, mkt):
    src, kind, sign, cost = MARKETS[mkt]
    same = [k for k, v in MARKETS.items() if v[0] == src]     # fold renamed aliases
    c = cot[cot["mkt"].isin(same)].drop_duplicates("date").set_index("date").sort_index()
    net = (c["ncl"] - c["ncs"]) / c["oi"]
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    px = price(src, kind)
    eff = c.index + pd.Timedelta(days=6)                 # Monday after Friday release
    idx = np.clip(px.index.searchsorted(eff), 0, len(px) - 1)
    entry = px.iloc[idx]
    fwd = entry.shift(-1).values / entry.values - 1      # ~1 week hold
    p = pd.DataFrame({"z": z.values, "fwd": fwd, "entry": entry.values}, index=c.index).dropna()
    pos = (p["z"] < Z_THRESH).astype(float)
    wk = sign * pos * p["fwd"] - pos.diff().abs().fillna(pos) * (cost / p["entry"])
    daily = wk.repeat(5) / 5
    daily.index = pd.date_range(p.index[0], periods=len(daily), freq="B")
    return daily.rename(f"cotwash_{src}"), int(pos.sum()), p


def main():
    cot = load_cot()
    print(f"COT rows loaded: {len(cot)} across {cot['mkt'].nunique()} markets\n")
    print("SINGLE DECLARED HYPOTHESIS (z < -1.5 -> long 1 week), applied unchanged:")
    print(f"{'market':12} {'signals':>8} {'fwd_when_ON':>12} {'fwd_base':>10} {'sharpe':>8} {'t':>7}")
    legs, rows = {}, {}
    seen = set()
    for mkt in MARKETS:
        if MARKETS[mkt][0] in seen:
            continue
        seen.add(MARKETS[mkt][0])
        try:
            leg, n_sig, p = washout_leg(cot, mkt)
        except Exception as e:
            print(f"  {MARKETS[mkt][0]:12} FAILED {e}")
            continue
        src, kind, sign, cost = MARKETS[mkt]
        dev = leg[leg.index < DEV].dropna()
        pd_ = p[p.index < DEV]
        on = pd_[pd_["z"] < Z_THRESH]["fwd"] * sign
        base = pd_[pd_["z"] >= Z_THRESH]["fwd"] * sign
        m = full_metrics(None, dev)
        sh = m.get("sharpe")
        t = dev.mean() / (dev.std() + 1e-12) * np.sqrt(len(dev))
        print(f"  {src:12} {len(on):>8} {on.mean()*1e4:>11.0f}bp {base.mean()*1e4:>9.0f}bp "
              f"{sh:>8} {t:>7.2f}")
        rows[src] = {"n_signals": len(on), "fwd_on_bp": round(float(on.mean() * 1e4), 1),
                     "fwd_base_bp": round(float(base.mean() * 1e4), 1),
                     "dev_sharpe": sh}
        if (sh or -9) > 0:
            legs[f"cotwash_{src}"] = leg
    for k, v in legs.items():
        v.to_csv(OUT / f"c9leg_{k}.csv", header=["ret"])
    json.dump(rows, open(OUT / "cot_breadth_dev.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "cot_breadth", "n_configs": len(MARKETS)}) + "\n")
    print(f"\npassing Stage 1: {len(legs)} / {len(MARKETS)} markets")


if __name__ == "__main__":
    main()

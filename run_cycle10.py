"""Cycle 10 — extend the CONFIRMED alpha mechanism across the equity-index family.

The washout rule (2y net-spec z < -1.5 -> long 1 week) is confirmed in equity
indices: NQ t 6.42, SPX t 3.74 (independent replication), and NOT in FX, metals,
energy or bonds. This cycle applies the SAME UNCHANGED RULE to every liquid
equity-index contract the CFTC publishes, matched to a clean ETF.

PRE-DECLARED PROMOTION CRITERION (fixed before any result was seen):
  a leg is admitted iff its STANDALONE dev t-stat >= 2.0 — the same bar the
  program's original survivor criterion used. NOT chosen by dev book Sharpe,
  which has twice produced holdout failures (cycles 6 and 9-invalid).
Then ONE holdout look on the pre-declared admitted set.

Alias groups handle CFTC contract renames (the bug that voided look #15).
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
COTD, EXT, OUT = ROOT / "data/external/cot", ROOT / "data/external", ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
Z = -1.5
T_BAR = 2.0                       # pre-declared admission bar

# price source -> list of COT market-name aliases across the years
FAMILY = {
    "IWM": (["RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE",
             "E-MINI RUSSELL 2000 INDEX - CHICAGO MERCANTILE EXCHANGE",
             "RUSSELL 2000 MINI - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.02),
    "DIA": (["DJIA Consolidated - CHICAGO BOARD OF TRADE",
             "DJIA x $5 - CHICAGO BOARD OF TRADE"], "etf", 0.02),
    "MDY": (["E-MINI S&P 400 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.03),
    "IWD": (["EMINI RUSSELL 1000 VALUE INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.02),
    "XLF": (["E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLU": (["E-MINI S&P UTILITIES INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLP": (["E-MINI S&P CONSU STAPLES INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLV": (["E-MINI S&P HEALTH CARE INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLE": (["E-MINI S&P ENERGY INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLK": (["E-MINI S&P TECHNOLOGY INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLB": (["E-MINI S&P MATERIALS INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "XLI": (["E-MINI S&P INDUSTRIAL INDEX - CHICAGO MERCANTILE EXCHANGE"], "etf", 0.01),
    "JPXJPY": (["NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE",
                "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE"], "hist", 20.0),
}
ALL_NAMES = {n for v in FAMILY.values() for n in v[0]}


def load_cot():
    fr = []
    for y in range(2010, 2027):
        z = zipfile.ZipFile(COTD / f"deacot{y}.zip")
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        nm = df["Market and Exchange Names"].str.strip()
        m = df[nm.isin(ALL_NAMES)].copy()
        m["mkt"] = nm[nm.isin(ALL_NAMES)]
        fr.append(m)
    c = pd.concat(fr)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    return c.rename(columns={"Open Interest (All)": "oi",
                             "Noncommercial Positions-Long (All)": "ncl",
                             "Noncommercial Positions-Short (All)": "ncs"})


def price(src, kind):
    if kind == "etf":
        d = pd.read_csv(EXT / f"etf_{src}_1d.csv", index_col=0, parse_dates=True)["close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    d = to_tf(load_15m(src), "1d")["close"]
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def leg(cot, src):
    names, kind, cost = FAMILY[src]
    c = (cot[cot["mkt"].isin(names)].drop_duplicates("date")
         .set_index("date").sort_index())
    if len(c) < 200:
        return None, None
    net = (c["ncl"] - c["ncs"]) / c["oi"]
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    px = price(src, kind)
    eff = c.index + pd.Timedelta(days=6)
    idx = np.clip(px.index.searchsorted(eff), 0, len(px) - 1)
    entry = px.iloc[idx]
    fwd = entry.shift(-1).values / entry.values - 1
    p = pd.DataFrame({"z": z.values, "fwd": fwd, "entry": entry.values},
                     index=c.index).dropna()
    pos = (p["z"] < Z).astype(float)
    wk = pos * p["fwd"] - pos.diff().abs().fillna(pos) * (cost / p["entry"])
    daily = wk.repeat(5) / 5
    daily.index = pd.date_range(p.index[0], periods=len(daily), freq="B")
    return daily.rename(f"cotwash_{src}"), p


def main():
    cot = load_cot()
    print(f"COT rows: {len(cot)}   contracts matched: {cot['mkt'].nunique()}\n")
    print(f"UNCHANGED RULE (z<{Z} -> long 1wk). PRE-DECLARED bar: standalone dev t >= {T_BAR}")
    print(f"{'mkt':8} {'sig':>5} {'ON':>8} {'base':>8} {'sharpe':>7} {'t':>6} {'hold_rows':>10}  verdict")
    admitted, rows = {}, {}
    for src in FAMILY:
        out = leg(cot, src)
        if out[0] is None:
            print(f"  {src:8} insufficient COT history")
            continue
        d, p = out
        dev = d[d.index < DEV].dropna()
        hold_rows = int((d.index >= DEV).sum())
        pdv = p[p.index < DEV]
        on = pdv[pdv["z"] < Z]["fwd"]; base = pdv[pdv["z"] >= Z]["fwd"]
        m = full_metrics(None, dev)
        sh = m.get("sharpe")
        if sh is None or len(dev) < 200:
            print(f"  {src:8} insufficient dev history ({len(dev)} obs)")
            continue
        t = dev.mean() / (dev.std() + 1e-12) * np.sqrt(len(dev))
        ok = (t >= T_BAR) and hold_rows > 200
        print(f"  {src:8} {len(on):>5} {on.mean()*1e4:>7.0f}bp {base.mean()*1e4:>7.0f}bp "
              f"{sh:>7} {t:>6.2f} {hold_rows:>10}  {'ADMIT' if ok else '-'}")
        rows[src] = {"n_sig": len(on), "on_bp": round(float(on.mean()*1e4), 1),
                     "dev_sharpe": sh, "dev_t": round(float(t), 2),
                     "holdout_rows": hold_rows, "admitted": bool(ok)}
        if ok:
            admitted[f"cotwash_{src}"] = d
            d.to_csv(OUT / f"c10leg_cotwash_{src}.csv", header=["ret"])
    json.dump(rows, open(OUT / "cycle10_dev.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "cycle10_equity_family", "n_configs": len(FAMILY)}) + "\n")
    print(f"\nADMITTED by pre-declared bar: {sorted(admitted)}")


if __name__ == "__main__":
    main()

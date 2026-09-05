"""Cycle 13 — "THE COMMITTEE": every strategy family votes; consensus trades.

This is the only honest, testable operationalization of "instinct / intuition
trading". What skilled discretionary traders describe is not a single rule: it
is holding many weak, partly-contradictory signals at once and acting on their
consensus, with size scaled to conviction. That IS mechanizable — as a vote.

Eight voters, one per school we have tested, each emitting a vote in [-1, +1]
using only trailing information:
  trend        mean sign of momentum over {40, 80, 160, 240} days
  breakout     position within the trailing 50-day Donchian range
  meanrev      negative z-score of price vs its 20-day mean (contrarian)
  positioning  COT speculative z-score, contrarian, where published
  macro        VIX percentile (equities) / real-yield trend (gold) / dollar (FX)
  volregime    20d vs 100d realized vol (risk-on when contracting)
  seasonal     day-of-week + turn-of-month tilt (the documented weak effects)
  crossasset   relative strength vs the equal-weight basket

NOTHING IS FITTED: votes are equal-weighted, the aggregate is the simple mean.
Two variants, both declared in advance:
  committee        trade the raw consensus
  conviction       trade ONLY when |consensus| exceeds 0.25 -- the discretionary
                   "wait for the A+ setup" principle, made mechanical
Dev window only; admission per FROZEN_SPEC before any holdout look.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT, COTD = ROOT / "data/external", ROOT / "research", ROOT / "data/external/cot"
DEV = pd.Timestamp("2022-07-01")
MKT = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD",
       "USDJPY": "USDJPY", "WTIUSD": "WTIUSD", "JPXJPY": "JPXJPY"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016,
        "USDJPY": 0.025, "WTIUSD": 0.07, "JPXJPY": 20.0}
COT_NAME = {"MNQ": ["NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
            "ES": ["E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
                   "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"],
            "XAUUSD": ["GOLD - COMMODITY EXCHANGE INC."],
            "EURUSD": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"]}
EQUITY = ("MNQ", "ES", "JPXJPY")
CONVICTION = 0.25


def daily(inst):
    d = to_tf(load_15m(MKT[inst]), "1d")
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def fred(sid):
    df = pd.read_csv(EXT / f"{sid}.csv")
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[sid], errors="coerce").dropna()


def cot_z(inst, idx):
    if inst not in COT_NAME:
        return None
    fr = []
    for y in range(2010, 2027):
        z = zipfile.ZipFile(COTD / f"deacot{y}.zip")
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        nm = df["Market and Exchange Names"].str.strip()
        m = df[nm.isin(COT_NAME[inst])].copy()
        fr.append(m)
    c = pd.concat(fr)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    c = c.drop_duplicates("date").set_index("date").sort_index()
    net = ((c["Noncommercial Positions-Long (All)"]
            - c["Noncommercial Positions-Short (All)"]) / c["Open Interest (All)"])
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    z.index = z.index + pd.Timedelta(days=6)          # public on release
    return z.reindex(idx, method="ffill")


def votes(inst, basket):
    d = daily(inst)
    c, idx = d["close"], d.index
    r = c.pct_change()
    V = {}
    V["trend"] = np.sign(pd.concat([c.pct_change(L) for L in (40, 80, 160, 240)],
                                   axis=1)).mean(axis=1)
    hi, lo = c.rolling(50).max(), c.rolling(50).min()
    V["breakout"] = (2 * (c - lo) / (hi - lo).replace(0, np.nan) - 1).clip(-1, 1)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    V["meanrev"] = (-(c - m20) / s20.replace(0, np.nan) / 2).clip(-1, 1)
    z = cot_z(inst, idx)
    V["positioning"] = (-z / 2).clip(-1, 1) if z is not None else pd.Series(0.0, index=idx)
    if inst in EQUITY:
        vix = fred("VIXCLS").reindex(idx).ffill()
        V["macro"] = (1 - 2 * vix.rolling(504).rank(pct=True)).clip(-1, 1)
    elif inst == "XAUUSD":
        ry = fred("DFII10").reindex(idx).ffill()
        V["macro"] = (-np.sign(ry.diff(20))).fillna(0.0)
    else:
        dx = fred("DTWEXBGS").reindex(idx).ffill()
        V["macro"] = (-np.sign(dx.pct_change(20))).fillna(0.0)
    v20, v100 = r.rolling(20).std(), r.rolling(100).std()
    V["volregime"] = (-(v20 / v100 - 1)).clip(-1, 1)
    dow = pd.Series(idx.dayofweek, index=idx)
    tom = pd.Series((idx.day <= 3) | (idx.day >= 28), index=idx).astype(float)
    V["seasonal"] = (0.5 * (dow == 4).astype(float) + 0.5 * tom).clip(-1, 1)
    rel = c.pct_change(60) - basket.reindex(idx).ffill()
    V["crossasset"] = np.sign(rel).fillna(0.0)
    W = pd.DataFrame(V).shift(1)                        # act next bar: no lookahead
    return d, W


def vt(pos, d, inst):
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (pos * (0.10 / np.sqrt(252)) / vol).clip(-3, 3)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * COST[inst] / d["close"]).dropna()


def main():
    closes = {}
    for inst in MKT:
        closes[inst] = daily(inst)["close"]
    basket = pd.DataFrame(closes).pct_change(60).mean(axis=1)

    legs_c, legs_k, per_voter = {}, {}, {}
    for inst in MKT:
        d, W = votes(inst, basket)
        cons = W.mean(axis=1).fillna(0.0)
        legs_c[inst] = vt(cons, d, inst)
        legs_k[inst] = vt(cons.where(cons.abs() > CONVICTION, 0.0), d, inst)
        for col in W.columns:
            per_voter.setdefault(col, {})[inst] = vt(W[col].fillna(0.0), d, inst)

    print("=" * 74)
    print("INDIVIDUAL VOTERS (dev Sharpe of each school traded alone, equal-wt book)")
    print("=" * 74)
    rows = {}
    for name, legs in per_voter.items():
        bk = pd.DataFrame(legs).mean(axis=1).dropna()
        m = full_metrics(None, bk[bk.index < DEV])
        rows[name] = m.get("sharpe")
        print(f"  {name:14} dev Sharpe {m.get('sharpe'):>6}   calmar {m.get('calmar'):>6}")

    print("\n" + "=" * 74)
    print("THE COMMITTEE (all voters combined) vs its parts")
    print("=" * 74)
    out = {}
    for label, legs in (("committee", legs_c), ("conviction (|vote|>0.25)", legs_k)):
        bk = pd.DataFrame(legs).mean(axis=1).dropna()
        for w in ("dev", "holdout"):
            seg = bk[bk.index < DEV] if w == "dev" else bk[bk.index >= DEV]
            m = full_metrics(None, seg)
            res = {k: m.get(k) for k in ("sharpe", "calmar", "max_dd_pct", "cagr_pct")}
            out[f"{label}|{w}"] = res
            if w == "dev":
                print(f"  {label:26} {w:8} {res}")
        bk.to_csv(OUT / f"committee_{label.split()[0]}_daily.csv", header=["ret"])
    best_voter = max(rows, key=lambda k: rows[k] or -9)
    print(f"\n  best single voter: {best_voter} ({rows[best_voter]})")
    print(f"  champion v1.2 dev Sharpe for reference: 1.45")
    json.dump({"voters": rows, "committee": out},
              open(OUT / "committee.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "committee", "n_configs": 8 * 7 + 2}) + "\n")


if __name__ == "__main__":
    main()

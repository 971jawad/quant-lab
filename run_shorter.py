"""Cycle 12 — "SHORTER": the short/downside side, tested in its own right.

Part A — DIAGNOSTIC: split every trend leg's P&L into the part earned while
LONG and the part earned while SHORT. Nobody has asked whether our edge is
symmetric. If shorting contributes nothing, that is a major structural finding.

Part B — DEDICATED SHORT SIGNALS (all declared before testing, no grids, and
t-stats computed at each signal's NATIVE frequency per Amendment 8):
  euphoria   COT net-spec z > +1.5 (crowded LONG) -> short 1 week.
             The exact mirror of the washout that works long-only.
  vix_bw     VIX/VIX3M > 1 (term-structure inversion = stress) -> short
  credit     HYG/IEF 20d ratio falling (credit deteriorating) -> short
  downtrend  below 200d MA AND negative 80d momentum -> short (isolates the
             short half of trend following)
  crashvol   below 200d MA AND 20d realized vol rising vs 100d -> short
             (leverage effect: down + expanding vol = crash regime)

Stated in advance: equity indices carry upward drift, so the short side faces a
permanent headwind. Negative results are the base case; the diagnostic value is
in HOW they fail.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT, COTD = ROOT / "data/external", ROOT / "research", ROOT / "data/external/cot"
DEV = pd.Timestamp("2022-07-01")
MKT = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD",
       "USDJPY": "USDJPY", "WTIUSD": "WTIUSD", "JPXJPY": "JPXJPY"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016,
        "USDJPY": 0.025, "WTIUSD": 0.07, "JPXJPY": 20.0}
COT_EQ = {"MNQ": ["NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
          "ES": ["E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
                 "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"]}
DASH = "-"


def daily(inst):
    """Frozen HistData history, extended with the live Yahoo feed when present.

    The extension is produced by run_live_update.py, which rebuilds ET-day bars
    from Yahoo HOURLY data so the bar convention matches, and refuses to write
    anything if the overlap return-correlation drops below its guard."""
    frozen = ROOT / "data" / "daily" / f"{inst}.csv"
    if frozen.exists():                      # committed daily series (CI path)
        d = pd.read_csv(frozen, index_col=0, parse_dates=True)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    else:                                    # local path: derive from the 15m archive
        d = to_tf(load_15m(MKT[inst]), "1d")
        d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    ext = ROOT / "data" / "live" / f"{inst}_ext.csv"
    if ext.exists():
        e = pd.read_csv(ext, index_col=0, parse_dates=True)
        e.index = pd.to_datetime(e.index).tz_localize(None).normalize()
        e = e[e.index > d.index[-1]]
        if len(e):
            d = pd.concat([d, e[["open", "high", "low", "close"]]])
    return d


def fred(sid):
    df = pd.read_csv(EXT / f"{sid}.csv")
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[sid], errors="coerce").dropna()


def etf(t):
    d = pd.read_csv(EXT / f"etf_{t}_1d.csv", index_col=0, parse_dates=True)["close"]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d


def vt(pos, d, inst):
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    p = (pos * (0.10 / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
    turn = p.diff().abs().fillna(p.abs())
    return (p * r - turn * COST[inst] / d["close"]).dropna(), p


def part_a():
    print("=" * 78)
    print("PART A - is our edge symmetric? Trend-leg P&L split by side (full sample)")
    print("=" * 78)
    print(f"{'leg':18} {'%days long':>11} {'long P&L':>11} {'short P&L':>11}   verdict")
    rows = {}
    for inst in MKT:
        d = daily(inst)
        sig = np.sign(d["close"].pct_change(160))
        ret, pos = vt(sig, d, inst)
        pos = pos.reindex(ret.index)
        lo = float(ret[pos > 0].sum() * 100)
        sh = float(ret[pos < 0].sum() * 100)
        frac = float((pos > 0).mean())
        verdict = "LONG-ONLY edge" if (sh <= 0 < lo) else (
            "symmetric" if (sh > 0 and lo > 0) else "both weak")
        print(f"  trend_{inst:12} {frac:>10.0%} {lo:>+10.1f}% {sh:>+10.1f}%   {verdict}")
        rows[inst] = {"pct_long": round(frac, 3), "long_pnl": round(lo, 2),
                      "short_pnl": round(sh, 2)}
    tot_l = sum(v["long_pnl"] for v in rows.values())
    tot_s = sum(v["short_pnl"] for v in rows.values())
    print(f"\n  TOTAL across 7 markets:  long {tot_l:+.1f}%    short {tot_s:+.1f}%")
    rows["TOTAL"] = {"long": round(tot_l, 1), "short": round(tot_s, 1)}
    return rows


def load_cot_eq():
    fr = []
    alln = {n for v in COT_EQ.values() for n in v}
    for y in range(2010, 2027):
        z = zipfile.ZipFile(COTD / f"deacot{y}.zip")
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        nm = df["Market and Exchange Names"].str.strip()
        m = df[nm.isin(alln)].copy()
        m["mkt"] = nm[nm.isin(alln)]
        fr.append(m)
    c = pd.concat(fr)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    return c.rename(columns={"Open Interest (All)": "oi",
                             "Noncommercial Positions-Long (All)": "ncl",
                             "Noncommercial Positions-Short (All)": "ncs"})


def euphoria(cot, inst):
    """Mirror of the washout: crowded LONG -> short one week. WEEKLY t-stat."""
    c = (cot[cot["mkt"].isin(COT_EQ[inst])].drop_duplicates("date")
         .set_index("date").sort_index())
    net = (c["ncl"] - c["ncs"]) / c["oi"]
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    px = daily(inst)["close"]
    eff = c.index + pd.Timedelta(days=6)
    idx = np.clip(px.index.searchsorted(eff), 0, len(px) - 1)
    entry = px.iloc[idx]
    fwd = entry.shift(-1).values / entry.values - 1
    p = pd.DataFrame({"z": z.values, "fwd": fwd}, index=c.index).dropna()
    p = p[p.index < DEV]
    on = -p[p["z"] > 1.5]["fwd"]
    base = -p[p["z"] <= 1.5]["fwd"]
    if len(on) < 5:
        return None
    t, pv = stats.ttest_ind(on, base, equal_var=False)
    return len(on), float(on.mean() * 1e4), float(base.mean() * 1e4), float(t), float(pv)


def short_rule(inst, kind):
    d = daily(inst)
    c = d["close"]
    ma200 = c.rolling(200).mean()
    mom = c.pct_change(80)
    r = c.pct_change()
    if kind == "downtrend":
        pos = -((c < ma200) & (mom < 0)).astype(float)
    elif kind == "crashvol":
        v20, v100 = r.rolling(20).std(), r.rolling(100).std()
        pos = -((c < ma200) & (v20 > v100)).astype(float)
    elif kind == "vix_bw":
        vix = fred("VIXCLS")
        ratio = vix / fred("VXVCLS").reindex(vix.index).ffill()
        pos = -(ratio.reindex(c.index).ffill() > 1.0).astype(float)
    elif kind == "credit":
        rr = (etf("HYG") / etf("IEF")).reindex(c.index).ffill()
        pos = -(rr.pct_change(20) < 0).astype(float)
    ret, _ = vt(pos.fillna(0.0), d, inst)
    dev = ret[ret.index < DEV]
    m = full_metrics(None, dev)
    t = float(dev.mean() / (dev.std() + 1e-12) * np.sqrt(len(dev))) if len(dev) > 100 else np.nan
    return m.get("sharpe"), t, ret


def part_b():
    print("\n" + "=" * 78)
    print("PART B - dedicated SHORT signals (dev window, native-frequency t-stats)")
    print("=" * 78)
    out = {}
    cot = load_cot_eq()
    print("  euphoria (COT z > +1.5 -> SHORT 1wk), WEEKLY t-stat:")
    for inst in COT_EQ:
        r = euphoria(cot, inst)
        if r:
            n, on, base, t, pv = r
            tag = "EDGE" if (pv < 0.05 and t > 0) else "no edge"
            print(f"    {inst:6} n={n:>3}  short-ret {on:>+7.0f}bp vs base {base:>+6.0f}bp"
                  f"  t={t:>+5.2f} p={pv:.3f}   {tag}")
            out[f"euphoria_{inst}"] = {"n": n, "on_bp": round(on, 1),
                                       "t": round(t, 2), "p": round(pv, 4)}
    print("\n  price / vol / credit SHORT-ONLY rules (dev Sharpe):")
    header = "    " + f"{'rule':11}" + " ".join(f"{i:>9}" for i in MKT)
    print(header)
    for kind in ("downtrend", "crashvol", "vix_bw", "credit"):
        cells = []
        for inst in MKT:
            try:
                sh, t, _ = short_rule(inst, kind)
                cells.append(f"{sh:>9.2f}" if sh is not None else f"{DASH:>9}")
                out[f"{kind}_{inst}"] = {"sharpe": sh, "t": round(t, 2)}
            except Exception:
                cells.append(f"{'err':>9}")
        print(f"    {kind:11}" + " ".join(cells))
    return out


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    json.dump({"side_attribution": a, "short_signals": b},
              open(OUT / "shorter.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "shorter", "n_configs": 7 + 4 * 7 + 2}) + "\n")

"""Cycle 11 — BACKWARD VALIDATION on 1999-2009: data the system has never seen.

The champion was developed on 2010-2022 and confirmed on 2022-2026. The CFTC
publishes positioning back to 1986 and Nasdaq futures began in 1999, so the one
confirmed alpha mechanism — the washout (2y net-spec z < -1.5 -> long 1 week) —
can be tested on an ENTIRELY INDEPENDENT DECADE that played no part in its
discovery. This is as clean as forward validation and available today.

Nothing is tuned. The rule, the threshold, the holding period and the timing
convention are all exactly as frozen. t-stats are computed on WEEKLY
observations (the native frequency) per the Amendment-8 rule.

Covers a violently different regime set: dot-com bust, 2003-07 bull, GFC.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
COTD, EXT, OUT = ROOT / "data/external/cot", ROOT / "data/external", ROOT / "research"
Z = -1.5
NAMES = {
    "NDX": ["NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
            "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE",
            "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
    "GSPC": ["S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
             "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
             "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE"],
}


def load_cot(years):
    fr = []
    for y in years:
        f = COTD / f"deacot{y}.zip"
        if not f.exists():
            continue
        z = zipfile.ZipFile(f)
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        nm = df["Market and Exchange Names"].str.strip()
        alln = {n for v in NAMES.values() for n in v}
        m = df[nm.isin(alln)].copy()
        m["mkt"] = nm[nm.isin(alln)]
        fr.append(m)
    c = pd.concat(fr)
    c["date"] = pd.to_datetime(c["As of Date in Form YYYY-MM-DD"])
    return c.rename(columns={"Open Interest (All)": "oi",
                             "Noncommercial Positions-Long (All)": "ncl",
                             "Noncommercial Positions-Short (All)": "ncs"})


def panel(cot, key):
    c = (cot[cot["mkt"].isin(NAMES[key])].drop_duplicates("date")
         .set_index("date").sort_index())
    net = (c["ncl"] - c["ncs"]) / c["oi"]
    z = (net - net.rolling(104).mean()) / net.rolling(104).std()
    px = pd.read_csv(EXT / f"pre2010_{key}_1d.csv", index_col=0, parse_dates=True)["close"]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    eff = c.index + pd.Timedelta(days=6)
    idx = np.clip(px.index.searchsorted(eff), 0, len(px) - 1)
    entry = px.iloc[idx]
    fwd = entry.shift(-1).values / entry.values - 1
    return pd.DataFrame({"z": z.values, "fwd": fwd}, index=c.index).dropna()


def main():
    cot = load_cot(range(1999, 2011))
    print(f"pre-2010 COT rows: {len(cot)}   span "
          f"{cot['date'].min().date()} -> {cot['date'].max().date()}\n")
    print("BACKWARD VALIDATION 1999-2009 — rule frozen, never tuned on this data")
    print(f"{'mkt':6} {'n_ON':>6} {'ON':>9} {'base':>9} {'t_weekly':>9} {'p':>8}  verdict")
    out = {}
    for key in NAMES:
        p = panel(cot, key)
        p = p[(p.index >= "1999-01-01") & (p.index < "2010-01-01")]
        on = p[p["z"] < Z]["fwd"]
        base = p[p["z"] >= Z]["fwd"]
        if len(on) < 5:
            print(f"  {key:6} too few signals ({len(on)})")
            continue
        t, pv = stats.ttest_ind(on, base, equal_var=False)
        verdict = "CONFIRMED" if (pv < 0.05 and t > 0) else (
            "right sign, ns" if t > 0 else "FAILS")
        print(f"  {key:6} {len(on):>6} {on.mean()*1e4:>8.0f}bp {base.mean()*1e4:>8.0f}bp "
              f"{t:>9.2f} {pv:>8.4f}  {verdict}")
        out[key] = {"n_on": len(on), "on_bp": round(float(on.mean()*1e4), 1),
                    "base_bp": round(float(base.mean()*1e4), 1),
                    "t_weekly": round(float(t), 2), "p": round(float(pv), 4),
                    "verdict": verdict}
    # pooled across the two indices
    if len(out) == 2:
        pn = panel(cot, "NDX"); pg = panel(cot, "GSPC")
        for d in (pn, pg):
            d.drop(d[(d.index < "1999-01-01") | (d.index >= "2010-01-01")].index, inplace=True)
        on = pd.concat([pn[pn["z"] < Z]["fwd"], pg[pg["z"] < Z]["fwd"]])
        bs = pd.concat([pn[pn["z"] >= Z]["fwd"], pg[pg["z"] >= Z]["fwd"]])
        t, pv = stats.ttest_ind(on, bs, equal_var=False)
        print(f"\n  POOLED 1999-2009: ON {on.mean()*1e4:+.0f}bp (n={len(on)}) vs "
              f"base {bs.mean()*1e4:+.0f}bp -> t={t:+.2f}, p={pv:.4f}")
        out["pooled"] = {"t": round(float(t), 2), "p": round(float(pv), 4),
                         "on_bp": round(float(on.mean()*1e4), 1)}
    print("\n  reference — 2010-2022 dev: NQ +111bp vs +26bp, t 2.32 (p~0.02)")
    json.dump(out, open(OUT / "backward_validation.json", "w"), indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "backward_validation_1999_2009",
                            "n_configs": 2, "note": "independent decade"}) + "\n")


if __name__ == "__main__":
    main()

"""Cycle 8 — MACRO ERA DECOMPOSITION: what actually moved these markets, when.

Splits 2010-2026 into economically-defined regimes (not data-mined breakpoints)
and reports, per era: each market's annualized return, the macro state that
defined it (real yields, dollar, VIX, Fed policy), and how the book performed.

The point is understanding, not a signal: this is the "why" layer under the
statistical work.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
EXT, OUT = ROOT / "data" / "external", ROOT / "research"
MKTS = {"Gold": "XAUUSD", "Nasdaq": "NSXUSD", "S&P": "SPXUSD",
        "EURUSD": "EURUSD", "Oil": "WTIUSD", "Nikkei": "JPXJPY", "USDJPY": "USDJPY"}
ERAS = [
    ("2010-01-01", "2011-09-06", "QE2 / euro crisis", "gold peaks $1920 on debt fear"),
    ("2011-09-06", "2015-12-16", "gold bear / taper", "Fed taper, dollar rally"),
    ("2015-12-16", "2020-02-19", "hiking + low-vol", "slow tightening, equity melt-up"),
    ("2020-02-19", "2020-08-06", "COVID crash+stimulus", "unlimited QE, real yields collapse"),
    ("2020-08-06", "2022-01-03", "reflation", "stimulus, negative real yields"),
    ("2022-01-03", "2022-10-12", "inflation shock", "fastest hiking cycle since 1980"),
    ("2022-10-12", "2026-09-01", "AI boom + gold record", "disinflation, CB gold buying"),
]


def daily(series):
    d = to_tf(load_15m(series), "1d")["close"]
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def fred(sid):
    df = pd.read_csv(EXT / f"{sid}.csv")
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[sid], errors="coerce").dropna()


def ann(s, a, b):
    seg = s[(s.index >= a) & (s.index < b)].dropna()
    if len(seg) < 10:
        return np.nan
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    return ((seg.iloc[-1] / seg.iloc[0]) ** (1 / max(yrs, 0.1)) - 1) * 100


def main():
    px = {k: daily(v) for k, v in MKTS.items()}
    ry, dxy, vix = fred("DFII10"), fred("DTWEXBGS"), fred("VIXCLS")
    book = pd.read_csv(OUT / "champion_v12_daily.csv", index_col=0, parse_dates=True)["ret"]
    book.index = pd.to_datetime(book.index).tz_localize(None).normalize()

    print("=" * 118)
    print("MACRO ERAS — annualized % return per market, with the macro state that defined each")
    print("=" * 118)
    hdr = f"{'era':22} {'Gold':>7} {'Nasdaq':>7} {'S&P':>7} {'EUR':>7} {'Oil':>7} {'Nikkei':>7} | {'realY':>6} {'DXY':>6} {'VIX':>5} | {'BOOK':>7}"
    print(hdr)
    rows = []
    for a, b, name, note in ERAS:
        A, B = pd.Timestamp(a), pd.Timestamp(b)
        vals = {k: ann(v, A, B) for k, v in px.items()}
        r_ = ry[(ry.index >= A) & (ry.index < B)]
        d_ = dxy[(dxy.index >= A) & (dxy.index < B)]
        v_ = vix[(vix.index >= A) & (vix.index < B)]
        bk = book[(book.index >= A) & (book.index < B)]
        bk_ann = ((1 + bk).prod() ** (252 / max(len(bk), 1)) - 1) * 100 if len(bk) > 20 else np.nan
        print(f"{name:22} {vals['Gold']:>7.1f} {vals['Nasdaq']:>7.1f} {vals['S&P']:>7.1f} "
              f"{vals['EURUSD']:>7.1f} {vals['Oil']:>7.1f} {vals['Nikkei']:>7.1f} | "
              f"{r_.mean():>6.2f} {d_.mean():>6.1f} {v_.mean():>5.1f} | {bk_ann:>7.1f}")
        print(f"{'':22} -> {note}")
        rows.append({"era": name, **{k: round(float(x), 1) for k, x in vals.items()},
                     "real_yield": round(float(r_.mean()), 2),
                     "dxy": round(float(d_.mean()), 1), "vix": round(float(v_.mean()), 1),
                     "book_ann_pct": round(float(bk_ann), 1) if not np.isnan(bk_ann) else None})

    print("\n" + "=" * 118)
    print("THE ONE RELATIONSHIP THAT EXPLAINS GOLD ACROSS ALL ERAS")
    print("=" * 118)
    g = px["Gold"].resample("QE").last().pct_change().dropna() * 100
    rq = ry.resample("QE").last().diff().dropna()
    j = pd.concat([g, rq], axis=1, keys=["gold_q%", "d_realyield"]).dropna()
    c = j["gold_q%"].corr(j["d_realyield"])
    print(f"  quarterly gold return vs change in 10y REAL yield: corr = {c:+.3f}  (n={len(j)} quarters)")
    lo = j[j["d_realyield"] < -0.2]["gold_q%"].mean()
    hi = j[j["d_realyield"] > 0.2]["gold_q%"].mean()
    print(f"  quarters real yields FELL >0.2pp : gold {lo:+.1f}% avg")
    print(f"  quarters real yields ROSE >0.2pp : gold {hi:+.1f}% avg")
    print(f"  -> gold is a REAL-RATE instrument. The post-COVID boom was real yields")
    print(f"     going deeply negative; the 2011-2015 bear was them rising.")
    print(f"  BUT (already proven in round 3): this is CONTEMPORANEOUS. Lagged real-yield")
    print(f"     changes predict next-day gold with corr ~0.00 -> understanding != edge.")

    json.dump(rows, open(OUT / "eras.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

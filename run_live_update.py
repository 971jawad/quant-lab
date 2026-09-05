"""LIVE PRICE FEED — extends the HistData history with fresh daily bars.

Why this exists: the 15-minute HistData archive is published with a ~2 month lag
and requires a multi-hour scrape, so without this the dashboard would freeze on
stale positions forever. This appends recent bars from Yahoo so the book keeps
rolling on its own.

CRITICAL detail, verified before building: Yahoo's DAILY bars close at a
different time than our ET-day convention, which destroys day-over-day return
alignment (EURUSD correlation 0.07, USDJPY 0.00). Rebuilding ET-day bars from
Yahoo HOURLY data fixes it completely:

    market   naive daily corr    ET-day-from-hourly corr
    XAUUSD        0.761                   0.973
    MNQ           0.868                   0.990
    ES            0.868                   0.989
    EURUSD        0.070                   0.991
    USDJPY        0.002                   0.984
    JPXJPY        0.142                   0.803

So the feed always resamples hourly -> ET day, and REFUSES to append if the
overlap correlation falls below MIN_CORR (a guard against the vendor silently
changing a contract, which has already bitten this project once).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
LIVE = ROOT / "data" / "live"
ET = "America/New_York"
MIN_CORR = 0.70
MIN_OVERLAP = 30

# our instrument -> Yahoo symbol. Futures where possible, cash index for Nikkei.
YMAP = {"XAUUSD": "GC=F", "MNQ": "NQ=F", "ES": "ES=F", "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X", "WTIUSD": "CL=F", "JPXJPY": "^N225"}


def hist_daily(inst):
    """The frozen daily series (committed to the repo, so CI works without the
    multi-hundred-MB 15-minute archive)."""
    f = ROOT / "data" / "daily" / f"{inst}.csv"
    if f.exists():
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d
    from run_research import load_15m, to_tf
    PSER = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD",
            "EURUSD": "EURUSD", "USDJPY": "USDJPY", "WTIUSD": "WTIUSD",
            "JPXJPY": "JPXJPY"}
    d = to_tf(load_15m(PSER[inst]), "1d")
    d.index = d.index.tz_convert(ET).tz_localize(None).normalize()
    return d


def yahoo_et_daily(symbol):
    """Yahoo hourly -> ET-day OHLC, matching the HistData bar convention."""
    d = yf.download(symbol, period="730d", interval="1h",
                    auto_adjust=False, progress=False)
    if d is None or len(d) == 0:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(ET)
    g = d.resample("1D")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()}).dropna()
    out.index = out.index.tz_localize(None).normalize()
    return out


def main():
    LIVE.mkdir(parents=True, exist_ok=True)
    report = []
    for inst, sym in YMAP.items():
        try:
            hist = hist_daily(inst)
        except Exception as e:
            print(f"  {inst:8} SKIP (no base history: {str(e)[:40]})")
            continue
        y = yahoo_et_daily(sym)
        if y is None or y.empty:
            print(f"  {inst:8} SKIP (no Yahoo data for {sym})")
            continue

        j = pd.concat([hist["close"], y["close"]], axis=1,
                      keys=["h", "y"]).dropna()
        if len(j) < MIN_OVERLAP:
            print(f"  {inst:8} REFUSED — overlap only {len(j)} bars")
            report.append({"market": inst, "status": "refused", "reason": "overlap"})
            continue
        corr = float(j["h"].pct_change().corr(j["y"].pct_change()))
        lvl = float(((j["h"] - j["y"]).abs() / j["y"]).median() * 100)
        if not (corr >= MIN_CORR):
            print(f"  {inst:8} REFUSED — return corr {corr:.3f} < {MIN_CORR} "
                  f"(vendor mismatch?)")
            report.append({"market": inst, "status": "refused",
                           "reason": f"corr {corr:.3f}"})
            continue

        # keep only bars strictly newer than the frozen history
        new = y[y.index > hist.index[-1]]
        if new.empty:
            print(f"  {inst:8} up to date (hist ends {hist.index[-1].date()}, "
                  f"corr {corr:.3f})")
            report.append({"market": inst, "status": "current",
                           "hist_end": str(hist.index[-1].date()),
                           "corr": round(corr, 3)})
            continue
        new.to_csv(LIVE / f"{inst}_ext.csv")
        print(f"  {inst:8} +{len(new):>3} bars -> {new.index[-1].date()}  "
              f"(corr {corr:.3f}, level diff {lvl:.2f}%)")
        report.append({"market": inst, "status": "extended", "added": len(new),
                       "hist_end": str(hist.index[-1].date()),
                       "live_end": str(new.index[-1].date()),
                       "corr": round(corr, 3), "level_diff_pct": round(lvl, 3)})

    (ROOT / "research" / "live_feed.json").write_text(
        json.dumps({"checked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "min_corr_guard": MIN_CORR, "markets": report},
                   indent=2, default=str))
    ok = sum(1 for r in report if r["status"] in ("extended", "current"))
    print(f"\n{ok}/{len(YMAP)} markets healthy -> research/live_feed.json")
    return 0 if ok else 1


if __name__ == "__main__":
    print("LIVE PRICE FEED — Yahoo hourly resampled to ET-day\n")
    sys.exit(main())

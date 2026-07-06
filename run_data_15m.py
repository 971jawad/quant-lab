"""Download 16 years of 1-minute data from HistData.com, aggregate to 15m,
and cross-verify against independent institutions (ECB, LBMA, Yahoo daily).

Series: EURUSD (spot), XAUUSD (spot), SPXUSD (S&P500 CFD -> MES/ES proxy),
NSXUSD (Nasdaq100 CFD -> MNQ proxy). HistData is quote-derived (no volume).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qlab.histdata import download_pair, to_15m
from run_data import fetch_ecb_eurusd, fetch_lbma_gold_pm  # reuse verified fetchers

ROOT = Path(__file__).parent
DATA = ROOT / "data"
START_YEAR = 2010
PAIRS = ["eurusd", "xauusd", "spxusd", "nsxusd"]
NAME = {"eurusd": "EURUSD", "xauusd": "XAUUSD", "spxusd": "SPXUSD", "nsxusd": "NSXUSD"}
# Yahoo daily futures series (already on disk, independently verified earlier)
YAHOO_DAILY = {"SPXUSD": "ES_1d.csv", "NSXUSD": "NQ_1d.csv"}


def verify(name: str, m15: pd.DataFrame) -> dict:
    daily = m15["close"].resample("1D").last().dropna()
    daily.index = daily.index.date
    if name == "EURUSD":
        other, src, corr_min, ret_days = fetch_ecb_eurusd(), "ECB", 0.85, 5
    elif name == "XAUUSD":
        other, src, corr_min, ret_days = fetch_lbma_gold_pm(), "LBMA", 0.85, 5
    else:
        y = pd.read_csv(DATA / YAHOO_DAILY[name], index_col=0, parse_dates=True)["close"]
        y.index = pd.to_datetime(y.index, utc=True).date
        other, src, corr_min, ret_days = y, "Yahoo-fut", 0.90, 5
    j = pd.concat([daily, other], axis=1, keys=["hist", src]).dropna()
    if len(j) < 200:
        return {"series": name, "verdict": "INSUFFICIENT OVERLAP", "overlap": len(j)}
    corr = j["hist"].pct_change(ret_days).corr(j[src].pct_change(ret_days))
    diff = ((j["hist"] - j[src]).abs() / j[src]).median()
    ok = corr > corr_min and diff < 0.02
    return {"series": name, "vs": src, "overlap_days": int(len(j)),
            "ret_corr_5d": round(float(corr), 4),
            "median_level_diff_pct": round(float(diff * 100), 3),
            "verdict": "VERIFIED" if ok else "MISMATCH - INVESTIGATE"}


def main():
    report = []
    for pair in PAIRS:
        name = NAME[pair]
        print(f"=== {name} ===", flush=True)
        m1 = download_pair(pair, DATA, start_year=START_YEAR,
                           log=lambda s: print(s, flush=True))
        m15 = to_15m(m1)
        del m1
        m15.to_csv(DATA / f"{name}_15m.csv")
        print(f"  saved {name}_15m.csv: {len(m15)} bars "
              f"{m15.index[0]} -> {m15.index[-1]}", flush=True)
        v = verify(name, m15)
        print(f"  verify: {v}", flush=True)
        report.append(v)
    with open(DATA / "verification_15m.json", "w") as fh:
        json.dump(report, fh, indent=2)
    bad = [v for v in report if v["verdict"] != "VERIFIED"]
    print("\nSUMMARY:", "ALL VERIFIED" if not bad else f"{len(bad)} FAILED")
    sys.exit(2 if bad else 0)


if __name__ == "__main__":
    main()

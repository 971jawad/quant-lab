"""Data pipeline: download OHLCV from Yahoo Finance (primary) and Stooq (verification),
cross-verify the two sources, and save clean CSVs.

Series used (continuous front-month futures / spot FX):
  ES  -> Yahoo ES=F  (E-mini S&P 500; MES is the micro contract on the identical price series)
  NQ  -> Yahoo NQ=F  (E-mini Nasdaq-100; MNQ micro, identical price series)
  GC  -> Yahoo GC=F  (COMEX gold; XAUUSD spot tracks it within carry)
  EURUSD -> Yahoo EURUSD=X
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

YAHOO = {"ES": "ES=F", "NQ": "NQ=F", "GC": "GC=F", "EURUSD": "EURUSD=X"}
# Micro contracts trade in their own order books -> arbitrage-consistency check
MICRO = {"ES": "MES=F", "NQ": "MNQ=F", "GC": "MGC=F"}


def fetch_yahoo(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} {interval}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    # sanity: kill rows with non-positive prices or absurd hi/lo inversion
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["high"] >= df["low"]]
    return df


def _curl(url: str) -> str:
    import subprocess
    out = subprocess.run(
        ["curl.exe", "-s", "-m", "60", url,
         "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
         "-H", "Accept: application/json, text/csv, */*"],
        capture_output=True, text=True, check=True)
    if not out.stdout.strip():
        raise RuntimeError(f"empty response from {url}")
    return out.stdout


def fetch_nasdaq(symbol: str, assetclass: str) -> pd.Series:
    url = (f"https://api.nasdaq.com/api/quote/{symbol}/historical?"
           f"assetclass={assetclass}&fromdate=2023-07-01&todate=2026-12-31&limit=9999")
    payload = json.loads(_curl(url))
    rows = payload["data"]["tradesTable"]["rows"]
    idx, vals = [], []
    for row in rows:
        idx.append(pd.to_datetime(row["date"]).date())
        vals.append(float(row["close"].replace(",", "").replace("$", "")))
    return pd.Series(vals, index=idx).sort_index()


def fetch_ecb_eurusd() -> pd.Series:
    url = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
           "?format=csvdata&startPeriod=2023-07-01")
    df = pd.read_csv(io.StringIO(_curl(url)))
    df = df.dropna(subset=["OBS_VALUE"])
    return pd.Series(df["OBS_VALUE"].astype(float).values,
                     index=pd.to_datetime(df["TIME_PERIOD"]).dt.date).sort_index()


def fetch_lbma_gold_pm() -> pd.Series:
    url = "https://prices.lbma.org.uk/json/gold_pm.json"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json()
    idx, vals = [], []
    for row in rows:
        usd = row["v"][0]
        if usd and usd > 0:
            idx.append(pd.to_datetime(row["d"]).date())
            vals.append(float(usd))
    return pd.Series(vals, index=idx).sort_index()


def cross_verify(name: str, ydf: pd.DataFrame, other: pd.Series, source: str,
                 corr_min: float, diff_max: float | None, ret_days: int = 1) -> dict:
    """Compare daily closes/returns between Yahoo and an independent source.
    Sources are related-but-distinct fixings (cash index vs futures, PM fix vs
    settle, ETF vs future), so thresholds are per-pair. diff_max=None skips the
    level check (e.g. SPY trades at ~1/10 of ES)."""
    y = ydf["close"].copy()
    y.index = y.index.date
    joined = pd.concat([y, other], axis=1, keys=["yahoo", source]).dropna()
    joined = joined.tail(750)  # last ~3y of overlap
    if len(joined) < 100:
        return {"series": name, "overlap_days": len(joined), "verdict": "INSUFFICIENT OVERLAP"}
    # ret_days > 1 neutralizes fixing-time offsets (e.g. ECB 2:15pm CET vs
    # Yahoo 5pm NY on a 24h market): multi-day returns re-correlate if the
    # underlying prices agree.
    ret_corr = joined["yahoo"].pct_change(ret_days).corr(joined[source].pct_change(ret_days))
    ok = ret_corr > corr_min
    out = {"series": name, "vs": source, "overlap_days": int(len(joined)),
           "return_days": ret_days, "return_correlation": round(float(ret_corr), 5)}
    if diff_max is not None:
        med_diff = ((joined["yahoo"] - joined[source]).abs() / joined[source]).median()
        ok = ok and med_diff < diff_max
        out["median_abs_close_diff_pct"] = round(float(med_diff * 100), 4)
    out["verdict"] = "VERIFIED" if ok else "MISMATCH - INVESTIGATE"
    return out


def main():
    report = []
    for name, ysym in YAHOO.items():
        print(f"--- {name} ({ysym}) ---")
        daily = fetch_yahoo(ysym, "1d", "10y")
        hourly = fetch_yahoo(ysym, "1h", "730d")
        daily.to_csv(DATA / f"{name}_1d.csv")
        hourly.to_csv(DATA / f"{name}_1h.csv")
        print(f"  yahoo daily:  {len(daily):>6} bars  {daily.index[0].date()} -> {daily.index[-1].date()}")
        print(f"  yahoo hourly: {len(hourly):>6} bars  {hourly.index[0]} -> {hourly.index[-1]}")

        # 1) independent-source check. Cash-vs-futures basis and fixing-time
        #    offsets mean returns corr ~0.8-0.95, level diff up to ~2%.
        try:
            if name == "ES":  # SPY ETF (Nasdaq feed), returns-only (level is ~1/10)
                v = cross_verify(name, daily, fetch_nasdaq("SPY", "etf"),
                                 "NASDAQ:SPY", 0.90, None)
            elif name == "NQ":
                v = cross_verify(name, daily, fetch_nasdaq("NDX", "index"),
                                 "NASDAQ:NDX", 0.90, 0.02)
            elif name == "GC":
                v = cross_verify(name, daily, fetch_lbma_gold_pm(),
                                 "LBMA:gold_pm", 0.80, 0.02)
            else:  # EURUSD vs ECB 2:15pm CET reference rate (timing offset)
                v = cross_verify(name, daily, fetch_ecb_eurusd(),
                                 "ECB:refrate", 0.85, 0.01, ret_days=5)
        except Exception as e:  # noqa: BLE001
            v = {"series": name, "verdict": f"SECOND-SOURCE FETCH FAILED: {e}"}
        print(f"  independent-source check: {v}")
        report.append(v)

        # 2) micro-contract arbitrage check (separate order book, must track ~exactly)
        if name in MICRO:
            try:
                micro = fetch_yahoo(MICRO[name], "1d", "5y")
                vm = cross_verify(name, daily, micro["close"].rename(None).set_axis(
                    micro.index.date), f"micro:{MICRO[name]}", 0.995, 0.002)
            except Exception as e:  # noqa: BLE001
                vm = {"series": name, "verdict": f"MICRO FETCH FAILED: {e}"}
            print(f"  micro-contract check:     {vm}")
            report.append(vm)
        time.sleep(1)

    with open(DATA / "verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nVERIFICATION SUMMARY")
    for v in report:
        print(f"  {v['series']:>7}: {v['verdict']}")
    if any("VERIFIED" not in v["verdict"] for v in report):
        print("WARNING: not all series verified against second source")
        sys.exit(2)


if __name__ == "__main__":
    main()

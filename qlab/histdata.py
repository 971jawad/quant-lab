"""HistData.com downloader: free 1-minute ASCII bars (EST, no DST).

Flow per file: GET the year/month page, scrape the hidden form token, POST to
get.php with Referer -> ZIP with one CSV: 'YYYYMMDD HHMMSS;O;H;L;C;V'.
"""
import io
import re
import time
import zipfile

import pandas as pd
import requests

BASE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes"
GET = "https://www.histdata.com/get.php"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def available_years(pair: str, s: requests.Session | None = None) -> list[int]:
    s = s or _session()
    r = s.get(f"{BASE}/{pair.lower()}", timeout=60)
    r.raise_for_status()
    years = sorted({int(y) for y in
                    re.findall(rf"/{pair.lower()}/(\d{{4}})", r.text)})
    return years


def _scrape_form(html: str) -> dict:
    out = {}
    for field in ("tk", "date", "datemonth", "platform", "timeframe", "fxpair"):
        m = re.search(rf'id="{field}" value="([^"]*)"', html)
        if not m:
            raise RuntimeError(f"form field {field} not found")
        out[field] = m.group(1)
    return out


def fetch_zip_csv(pair: str, year: int, month: int | None = None,
                  s: requests.Session | None = None) -> pd.DataFrame:
    s = s or _session()
    page = f"{BASE}/{pair.lower()}/{year}" + (f"/{month}" if month else "")
    r = s.get(page, timeout=60)
    r.raise_for_status()
    form = _scrape_form(r.text)
    dl = s.post(GET, data=form, headers={"Referer": page}, timeout=180)
    dl.raise_for_status()
    if not dl.content[:2] == b"PK":
        raise RuntimeError(f"not a zip for {pair} {year}/{month}: {dl.content[:80]!r}")
    zf = zipfile.ZipFile(io.BytesIO(dl.content))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(zf.open(name), sep=";", header=None,
                     names=["ts", "open", "high", "low", "close", "vol"])
    # EST fixed offset (no DST) -> UTC
    idx = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
    df.index = idx.dt.tz_localize("UTC")
    return df[["open", "high", "low", "close"]].sort_index()


def download_pair(pair: str, data_dir, start_year: int | None = None,
                  log=print) -> pd.DataFrame:
    """Download all available M1 history for a pair, return one frame."""
    s = _session()
    years = available_years(pair, s)
    if start_year:
        years = [y for y in years if y >= start_year]
    if not years:
        raise RuntimeError(f"no years found for {pair}")
    cur_year = years[-1]
    parts = []
    for y in years[:-1]:
        for attempt in (1, 2, 3):
            try:
                parts.append(fetch_zip_csv(pair, y, None, s))
                log(f"  {pair} {y}: {len(parts[-1])} rows")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    log(f"  {pair} {y}: FAILED ({e})")
                else:
                    time.sleep(3 * attempt)
        time.sleep(0.5)
    for m in range(1, 13):
        try:
            parts.append(fetch_zip_csv(pair, cur_year, m, s))
            log(f"  {pair} {cur_year}/{m:02d}: {len(parts[-1])} rows")
        except Exception:  # noqa: BLE001
            break                      # months beyond 'now' don't exist
        time.sleep(0.5)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["high"] >= df["low"]]
    return df


def to_15m(m1: pd.DataFrame) -> pd.DataFrame:
    o = m1["open"].resample("15min").first()
    h = m1["high"].resample("15min").max()
    l = m1["low"].resample("15min").min()
    c = m1["close"].resample("15min").last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    out["volume"] = 0.0
    return out

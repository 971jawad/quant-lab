"""Macro / policy / liquidity feature layer — with PUBLICATION LAGS.

The single most important thing in this module is PUB_LAG.

FRED timestamps are AS-OF dates, not release dates. Nonfarm payrolls dated
2026-08-01 is published in early September; the Fed's balance sheet dated
Wednesday is released Thursday afternoon. Using a value on its as-of date is a
lookahead bug that would manufacture edge out of nothing — the exact failure this
whole project is built to avoid. Every series therefore carries an explicit lag
in CALENDAR DAYS, set conservatively (later than the true release, never earlier),
and every feature is shifted by it before it can be used.

Series groups:
  rates      DGS2/10/30, T10Y2Y, T10Y3M, DFII10, T10YIE, T5YIFR, THREEFYTP10
  liquidity  WALCL (Fed balance sheet), WTREGEN (Treasury general account),
             RRPONTSYD (reverse repo), WRESBAL (bank reserves)
             -> NET LIQUIDITY = WALCL - WTREGEN - RRPONTSYD, the macro-trader
                thesis that has never been tested in this repo
  credit     BAMLH0A0HYM2 (high-yield OAS), BAMLC0A0CM (investment-grade OAS)
  condition  NFCI, STLFSI4, VIXCLS
  growth     ICSA (weekly claims), INDPRO, PAYEMS, UNRATE, UMCSENT
  inflation  CPIAUCSL, PCEPILFE
  policy     DFF (fed funds)
  japan      IRLTLT01JPM156N (JGB 10y), JPNASSETS (BOJ balance sheet), DEXJPUS
  fiscal     GFDEBTN (federal debt)
  dollar     DTWEXBGS
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MACRO = ROOT / "data" / "external" / "macro"

# Calendar-day lag from the series' AS-OF date to when a trader could safely act.
# Deliberately conservative: later than the real release in every case.
PUB_LAG = {
    # daily market data - published after that day's close, usable next session
    "DGS2": 1, "DGS10": 1, "DGS30": 1, "T10Y2Y": 1, "T10Y3M": 1,
    "DFII10": 1, "T10YIE": 1, "T5YIFR": 1, "THREEFYTP10": 2,
    "DFF": 1, "VIXCLS": 1, "DTWEXBGS": 4, "DEXJPUS": 4,
    "RRPONTSYD": 1,
    # weekly Fed H.4.1 - as-of Wednesday, released Thursday 16:30 -> Friday safe
    "WALCL": 3, "WRESBAL": 3, "WTREGEN": 3,
    # weekly, released the following week
    "ICSA": 6, "NFCI": 6, "STLFSI4": 6,
    # credit indices - daily but revised; give a couple of days
    "BAMLH0A0HYM2": 2, "BAMLC0A0CM": 2,
    # monthly - published 2-6 weeks after the reference month ends
    "CPIAUCSL": 45, "PCEPILFE": 45, "PAYEMS": 40, "UNRATE": 40,
    "INDPRO": 45, "UMCSENT": 30, "IRLTLT01JPM156N": 45,
    "JPNASSETS": 40, "GFDEBTN": 90,
}


def load(sid: str) -> pd.Series:
    """One FRED series, indexed by AS-OF date (lag not yet applied)."""
    f = MACRO / f"{sid}.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    d.columns = ["date", sid]
    d["date"] = pd.to_datetime(d["date"])
    s = pd.to_numeric(d.set_index("date")[sid], errors="coerce").dropna()
    s.index = s.index.normalize()
    return s


def available(sid: str, idx: pd.DatetimeIndex) -> pd.Series:
    """The series as it would have been KNOWN on each date in idx.

    Shifts the as-of index forward by the publication lag, then forward-fills.
    A value dated 2026-08-01 with a 40-day lag becomes usable 2026-09-10.
    """
    s = load(sid)
    if s.empty:
        return pd.Series(np.nan, index=idx)
    lag = PUB_LAG.get(sid, 45)
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=lag)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def net_liquidity(idx: pd.DatetimeIndex) -> pd.Series:
    """Fed balance sheet minus Treasury cash minus reverse repo, in $bn.

    The 'net liquidity drives risk assets' thesis, popular among macro traders
    and never tested here. WALCL and WTREGEN are $mn; RRPONTSYD is $bn.
    """
    bs = available("WALCL", idx) / 1000.0
    tga = available("WTREGEN", idx) / 1000.0
    rrp = available("RRPONTSYD", idx)
    return (bs - tga - rrp).rename("net_liquidity")


def build(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Every macro feature, each already lagged to its true availability."""
    f = pd.DataFrame(index=idx)

    # --- levels that matter as levels
    f["curve_2s10s"] = available("T10Y2Y", idx)
    f["curve_3m10y"] = available("T10Y3M", idx)
    f["real_yield"] = available("DFII10", idx)
    f["breakeven_10y"] = available("T10YIE", idx)
    f["infl_5y5y"] = available("T5YIFR", idx)
    f["term_premium"] = available("THREEFYTP10", idx)
    f["hy_oas"] = available("BAMLH0A0HYM2", idx)
    f["ig_oas"] = available("BAMLC0A0CM", idx)
    f["nfci"] = available("NFCI", idx)
    f["stlfsi"] = available("STLFSI4", idx)
    f["vix"] = available("VIXCLS", idx)

    # --- liquidity
    nl = net_liquidity(idx)
    f["net_liq"] = nl
    f["net_liq_chg4w"] = nl.diff(20)
    f["net_liq_chg13w"] = nl.diff(65)
    f["fed_bs_chg4w"] = (available("WALCL", idx) / 1000.0).diff(20)
    f["rrp_chg4w"] = available("RRPONTSYD", idx).diff(20)
    f["tga_chg4w"] = (available("WTREGEN", idx) / 1000.0).diff(20)
    f["reserves_chg4w"] = (available("WRESBAL", idx) / 1000.0).diff(20)

    # --- carry / Japan
    us10 = available("DGS10", idx)
    jp10 = available("IRLTLT01JPM156N", idx)
    f["carry_us_jp"] = us10 - jp10
    f["carry_chg13w"] = (us10 - jp10).diff(65)
    f["boj_bs_chg"] = available("JPNASSETS", idx).pct_change(60)
    f["policy_rate"] = available("DFF", idx)
    f["policy_chg13w"] = available("DFF", idx).diff(65)

    # --- growth / inflation momentum
    f["claims_chg4w"] = available("ICSA", idx).pct_change(20)
    f["payems_yoy"] = available("PAYEMS", idx).pct_change(252)
    f["unrate_chg"] = available("UNRATE", idx).diff(126)
    f["indpro_yoy"] = available("INDPRO", idx).pct_change(252)
    f["sentiment_chg"] = available("UMCSENT", idx).pct_change(126)
    f["cpi_yoy"] = available("CPIAUCSL", idx).pct_change(252)
    f["core_pce_yoy"] = available("PCEPILFE", idx).pct_change(252)

    # --- fiscal / dollar
    f["debt_yoy"] = available("GFDEBTN", idx).pct_change(252)
    f["dollar_chg13w"] = available("DTWEXBGS", idx).pct_change(65)

    # --- changes of the key levels (often what matters, not the level)
    for col in ("curve_2s10s", "real_yield", "hy_oas", "term_premium",
                "breakeven_10y", "nfci"):
        f[f"{col}_chg4w"] = f[col].diff(20)

    return f


FEATURES = None   # populated on first build() call by callers that want the list

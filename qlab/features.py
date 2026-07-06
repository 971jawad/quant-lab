"""Feature engineering on hourly bars. Every feature at bar t uses ONLY data
up to and including bar t's close (trailing windows, positive shifts).
Swing points are only usable after their confirmation bar."""
import numpy as np
import pandas as pd

ET = "America/New_York"


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return _ema(tr, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = _ema(d.clip(lower=0), n)
    dn = _ema((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def confirmed_swings(df: pd.DataFrame, k: int = 3) -> tuple[pd.Series, pd.Series]:
    """Last confirmed swing high/low level available at each bar.
    A swing high at bar i needs high[i] strictly greater than the k bars on
    each side, so it is only KNOWN at bar i+k. The returned series place the
    level at the confirmation bar and forward-fill - no lookahead."""
    h, lo = df["high"].values, df["low"].values
    n = len(df)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]) and (h[i] > h[i - k:i]).all() and (h[i] > h[i + 1:i + k + 1]).all():
            sh[i + k] = h[i]                      # confirmed k bars later
        if lo[i] == min(lo[i - k:i + k + 1]) and (lo[i] < lo[i - k:i]).all() and (lo[i] < lo[i + 1:i + k + 1]).all():
            sl[i + k] = lo[i]
    return (pd.Series(sh, index=df.index).ffill(),
            pd.Series(sl, index=df.index).ffill())


def prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous completed ET-calendar-day high/low/close mapped onto each bar."""
    et_date = df.index.tz_convert(ET).date
    g = df.groupby(et_date)
    day = pd.DataFrame({"pdh": g["high"].max(), "pdl": g["low"].min(),
                        "pdc": g["close"].last()})
    day = day.shift(1)  # bar t sees only the PREVIOUS completed day
    out = day.reindex(et_date)
    out.index = df.index
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    f["open"], f["high"], f["low"], f["close"] = df["open"], df["high"], df["low"], c
    f["atr"] = atr(df, 14)
    f["atr_pct"] = f["atr"] / c
    f["rsi"] = rsi(c, 14)
    for k in (1, 3, 6, 24):
        f[f"ret{k}"] = np.log(c / c.shift(k))
    macd = _ema(c, 12) - _ema(c, 26)
    f["macd_hist"] = (macd - _ema(macd, 9)) / f["atr"]
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - m20) / s20.replace(0, np.nan)
    f["sma50_dist"] = c / c.rolling(50).mean() - 1
    f["sma200_dist"] = c / c.rolling(200).mean() - 1
    f["don_hi"] = df["high"].rolling(20).max()
    f["don_lo"] = df["low"].rolling(20).min()
    f["don_hi50"] = df["high"].rolling(50).max()
    f["don_lo50"] = df["low"].rolling(50).min()
    f["don_pos"] = (c - f["don_lo"]) / (f["don_hi"] - f["don_lo"]).replace(0, np.nan)
    f["vol_ratio"] = f["atr"] / f["atr"].rolling(100).mean()
    et = df.index.tz_convert(ET)
    f["hour_et"] = et.hour
    f["dow"] = et.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * et.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * et.hour / 24)
    pdl = prev_day_levels(df)
    f["pdh"], f["pdl"], f["pdc"] = pdl["pdh"], pdl["pdl"], pdl["pdc"]
    f["pd_range_pct"] = (pdl["pdh"] - pdl["pdl"]) / c
    f["dist_pdh"] = (pdl["pdh"] - c) / f["atr"]
    f["dist_pdl"] = (c - pdl["pdl"]) / f["atr"]
    sh, sl = confirmed_swings(df, 3)
    f["swing_hi"], f["swing_lo"] = sh, sl
    # candle anatomy
    body = (df["close"] - df["open"])
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_ratio"] = body.abs() / rng
    f["up_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    f["dn_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    f["candle_dir"] = np.sign(body)
    f["range_atr"] = (df["high"] - df["low"]) / f["atr"]
    # killzones (ICT): London open 02-05 ET, NY 07-11 ET
    f["killzone"] = ((f["hour_et"] >= 2) & (f["hour_et"] < 5)) | \
                    ((f["hour_et"] >= 7) & (f["hour_et"] < 11))
    return f


ML_FEATURES = ["ret1", "ret3", "ret6", "ret24", "rsi", "macd_hist", "bb_z",
               "atr_pct", "vol_ratio", "sma50_dist", "sma200_dist", "don_pos",
               "hour_sin", "hour_cos", "dow", "pd_range_pct", "dist_pdh",
               "dist_pdl", "body_ratio", "range_atr"]

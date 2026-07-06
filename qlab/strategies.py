"""Three strategy families. Each generator returns a DataFrame of candidate
trades: [i, dir, stop, conviction] where i is the SIGNAL bar (all information
from bars <= i). The backtester enters at bar i+1's open.

S1  smc      - ICT/SMC synthesis: prior-day liquidity sweep + displacement +
               fair value gap, optional killzone / HTF-bias filters.
S2  ml       - gradient-boosting classifier on engineered features,
               walk-forward refit per fold (see walkforward.py).
S3  ta_score - classic technical-analysis vote ensemble: trend, breakout,
               RSI extreme, engulfing at structure, pin/rejection bar.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import ML_FEATURES

SWEEP_LOOKBACK = 6
STOP_MIN_ATR, STOP_MAX_ATR = 0.25, 3.0


def set_scale(mult: int) -> None:
    """Rescale STRUCTURAL windows (time-based semantics) for a finer bar size,
    e.g. mult=4 for 15m bars vs the 1h baseline. Indicator periods (RSI/ATR/
    Donchian/SMA) stay in bars, as is standard when changing timeframe."""
    global SWEEP_LOOKBACK, ML_HORIZON, RECENCY_HALF_LIFE
    SWEEP_LOOKBACK = 6 * mult
    ML_HORIZON = 6 * mult
    RECENCY_HALF_LIFE = 2500 * mult

# require_fvg stays fixed True: FVG-after-sweep is the consensus core of the
# SMC canon, and widening the grid only feeds selection overfit (verified:
# OOS degraded when it was tunable)
SMC_GRID = [{"disp_k": dk, "killzone": kz, "bias": b, "require_fvg": True}
            for dk in (1.0, 1.5) for kz in (True, False) for b in (True, False)]

TA_GRID = [{"thresh": t, "don_n": n, "stop_k": sk}
           for t in (2, 3) for n in (20, 50) for sk in (1.0, 1.75)]

ML_GRID = [{"p_thresh": p, "stop_k": sk}
           for p in (0.56, 0.60) for sk in (1.0, 1.75)]

ML_HORIZON = 6  # forward bars for the label


def _stops_ok(dist: pd.Series, a: pd.Series) -> pd.Series:
    return (dist >= STOP_MIN_ATR * a) & (dist <= STOP_MAX_ATR * a)


def smc_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    a = f["atr"]
    lo_min = f["low"].rolling(SWEEP_LOOKBACK).min()
    hi_max = f["high"].rolling(SWEEP_LOOKBACK).max()

    swept_pdl = (lo_min < f["pdl"]) & (f["close"] > f["pdl"])   # raided & reclaimed
    swept_pdh = (hi_max > f["pdh"]) & (f["close"] < f["pdh"])
    disp_up = (f["range_atr"] >= p["disp_k"]) & (f["candle_dir"] > 0) & (f["body_ratio"] >= 0.5)
    disp_dn = (f["range_atr"] >= p["disp_k"]) & (f["candle_dir"] < 0) & (f["body_ratio"] >= 0.5)
    fvg_up = f["low"] > f["high"].shift(2)      # bullish 3-bar imbalance at t
    fvg_dn = f["high"] < f["low"].shift(2)

    long_ = swept_pdl & disp_up
    short = swept_pdh & disp_dn
    if p["require_fvg"]:
        long_, short = long_ & fvg_up, short & fvg_dn
    if p["killzone"]:
        long_, short = long_ & f["killzone"], short & f["killzone"]
    if p["bias"]:
        long_ &= f["sma200_dist"] > 0
        short &= f["sma200_dist"] < 0

    stop_l = lo_min - 0.1 * a
    stop_s = hi_max + 0.1 * a
    long_ &= _stops_ok(f["close"] - stop_l, a)
    short &= _stops_ok(stop_s - f["close"], a)

    out = []
    for mask, d, st in ((long_, 1, stop_l), (short, -1, stop_s)):
        idx = np.flatnonzero(mask.fillna(False).values)
        out.append(pd.DataFrame({"i": idx, "dir": d, "stop": st.values[idx],
                                 "conviction": 1.0}))
    return pd.concat(out).sort_values("i").reset_index(drop=True)


def ta_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    a = f["atr"]
    hi_col = "don_hi" if p["don_n"] == 20 else "don_hi50"
    lo_col = "don_lo" if p["don_n"] == 20 else "don_lo50"
    # shift(1): breakout is vs the PRIOR N-bar extreme, not the bar's own high
    brk = np.where(f["close"] > f[hi_col].shift(1), 1,
                   np.where(f["close"] < f[lo_col].shift(1), -1, 0))
    trend = np.sign(f["sma200_dist"]).fillna(0).values
    rsi_c = np.where(f["rsi"] < 30, 1, np.where(f["rsi"] > 70, -1, 0))

    near_lo = (f["close"] - np.minimum(f["swing_lo"], f[lo_col])).abs() < a
    near_hi = (np.maximum(f["swing_hi"], f[hi_col]) - f["close"]).abs() < a
    engulf_up = ((f["candle_dir"] > 0) & (f["candle_dir"].shift(1) < 0) &
                 (f["open"] <= f["close"].shift(1)) & (f["close"] > f["open"].shift(1)) &
                 (f["body_ratio"] > 0.6) & near_lo)
    engulf_dn = ((f["candle_dir"] < 0) & (f["candle_dir"].shift(1) > 0) &
                 (f["open"] >= f["close"].shift(1)) & (f["close"] < f["open"].shift(1)) &
                 (f["body_ratio"] > 0.6) & near_hi)
    engulf = np.where(engulf_up, 1, np.where(engulf_dn, -1, 0))
    pin_up = (f["dn_wick"] > 0.5) & (f["body_ratio"] < 0.35) & near_lo
    pin_dn = (f["up_wick"] > 0.5) & (f["body_ratio"] < 0.35) & near_hi
    pin = np.where(pin_up, 1, np.where(pin_dn, -1, 0))

    score = brk + trend + rsi_c + engulf + pin
    long_ = score >= p["thresh"]
    short = score <= -p["thresh"]

    stop_l = f["close"] - p["stop_k"] * a
    stop_s = f["close"] + p["stop_k"] * a
    conv = np.clip(np.abs(score) / p["thresh"], 0.5, 1.5)
    out = []
    for mask, d, st in ((long_, 1, stop_l), (short, -1, stop_s)):
        idx = np.flatnonzero(mask & f["atr"].notna().values)
        out.append(pd.DataFrame({"i": idx, "dir": d, "stop": st.values[idx],
                                 "conviction": conv[idx]}))
    return pd.concat(out).sort_values("i").reset_index(drop=True)


# ---------------------------------------------------------------- ML strategy

def ml_labels(f: pd.DataFrame) -> pd.Series:
    """Label at bar t: does the tradeable forward return (open[t+1] ->
    open[t+1+H]) come out positive? Purely future info - used for TRAINING
    targets only, never as a feature."""
    o = f["open"]
    fwd = np.log(o.shift(-(ML_HORIZON + 1)) / o.shift(-1))
    return (fwd > 0).astype(int).where(fwd.notna())


RECENCY_HALF_LIFE = 2500   # bars (~5 months of hourly) for the recency scheme
ERR_ALPHA = 2.0            # error scheme: weight = 1 + alpha * |y - p_teacher|


def ml_fit(f: pd.DataFrame, lo: int, hi: int, scheme: str = "uniform",
           teacher=None) -> HistGradientBoostingClassifier:
    """Fit on bars [lo, hi). Caller must leave an embargo of ML_HORIZON+1 bars
    between hi and any evaluation range so no label window crosses over.

    Training schemes ("learn from experience"):
      uniform  - every bar weighs the same (baseline)
      recency  - exponential decay, half-life RECENCY_HALF_LIFE bars: the
                 model tracks the current regime
      error    - samples the TEACHER model (the previous walk-forward fold's
                 model) got wrong are up-weighted: each fold literally studies
                 its predecessor's mistakes
    """
    X = f[ML_FEATURES].iloc[lo:hi]
    y = ml_labels(f).iloc[lo:hi]
    ok = X.notna().all(axis=1) & y.notna()
    Xo, yo = X[ok], y[ok].astype(int)
    w = None
    if scheme == "recency":
        pos = np.arange(len(X))[ok.values]
        w = 0.5 ** ((len(X) - 1 - pos) / RECENCY_HALF_LIFE)
    elif scheme == "error":
        if teacher is None:             # first fold: study your own draft
            teacher = ml_fit(f, lo, hi, "uniform")
        p = teacher.predict_proba(Xo)[:, 1]
        w = 1.0 + ERR_ALPHA * np.abs(yo.values - p)
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                       max_leaf_nodes=31, min_samples_leaf=200,
                                       l2_regularization=1.0,
                                       early_stopping=False, random_state=7)
    m.fit(Xo, yo, sample_weight=w)
    return m


def ml_signals(f: pd.DataFrame, model, lo: int, hi: int, p: dict) -> pd.DataFrame:
    """Signals for bars [lo, hi) from an already-fit model."""
    X = f[ML_FEATURES].iloc[lo:hi]
    ok = X.notna().all(axis=1)
    prob = np.full(len(X), 0.5)
    if ok.any():
        prob[ok.values] = model.predict_proba(X[ok])[:, 1]
    a = f["atr"].values[lo:hi]
    close = f["close"].values[lo:hi]
    rows = []
    last_i = -10**9
    for j in range(len(X)):
        if np.isnan(a[j]):
            continue
        d = 1 if prob[j] >= p["p_thresh"] else (-1 if prob[j] <= 1 - p["p_thresh"] else 0)
        if d == 0:
            continue
        i = lo + j
        if i - last_i < ML_HORIZON:     # don't restack the same signal every bar
            continue
        last_i = i
        stop = close[j] - d * p["stop_k"] * a[j]
        conv = float(np.clip(0.5 + 4 * abs(prob[j] - 0.5), 0.5, 1.5))
        rows.append((i, d, stop, conv))
    return pd.DataFrame(rows, columns=["i", "dir", "stop", "conviction"])

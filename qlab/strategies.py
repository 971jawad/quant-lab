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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ML_FEATURES

SWEEP_LOOKBACK = 6
STOP_MIN_ATR, STOP_MAX_ATR = 0.25, 3.0


ICT_MAX_WAIT = 12       # bars to wait for the fib retrace after a setup


def set_scale(mult: int) -> None:
    """Rescale STRUCTURAL windows (time-based semantics) for a finer bar size,
    e.g. mult=4 for 15m bars vs the 1h baseline. Indicator periods (RSI/ATR/
    Donchian/SMA) stay in bars, as is standard when changing timeframe."""
    global SWEEP_LOOKBACK, ML_HORIZON, RECENCY_HALF_LIFE, ICT_MAX_WAIT
    SWEEP_LOOKBACK = 6 * mult
    ML_HORIZON = 6 * mult
    RECENCY_HALF_LIFE = 2500 * mult
    ICT_MAX_WAIT = 12 * mult

# require_fvg stays fixed True: FVG-after-sweep is the consensus core of the
# SMC canon, and widening the grid only feeds selection overfit (verified:
# OOS degraded when it was tunable)
SMC_GRID = [{"disp_k": dk, "killzone": kz, "bias": b, "require_fvg": True}
            for dk in (1.0, 1.5) for kz in (True, False) for b in (True, False)]

TA_GRID = [{"thresh": t, "don_n": n, "stop_k": sk}
           for t in (2, 3) for n in (20, 50) for sk in (1.0, 1.75)]

# ICT fib-zone family. disp_k = displacement strength; use_ha swaps the raw
# displacement candle for a Heikin-Ashi one; bias adds the HTF SMA200 trend
# filter; manage bundles the two coherent trade-management philosophies:
#   raw      - fixed RR target, no break-even (let the runner's RR grid decide)
#   managed  - target = nearest opposite liquidity pool, stop to break-even at +1R
ICT_GRID = [{"disp_k": dk, "use_ha": ha, "bias": b, "manage": mg}
            for dk in (1.0, 1.5) for ha in (False, True)
            for b in (True, False) for mg in ("raw", "managed")]
FIB_LO, FIB_HI = 0.5, 0.786      # discount/premium retrace band for the entry
LOW_POOLS = ["asia_lo", "london_lo", "rth_lo", "pdl"]
HIGH_POOLS = ["asia_hi", "london_hi", "rth_hi", "pdh"]

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


def ict_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """ICT/SMC fib-zone setup:

      1. sweep of a time-based liquidity pool (Asia/London/RTH/prior-day extreme):
         price wicks past the level and closes back inside (manipulation),
      2. displacement candle in the opposite direction (raw or Heikin-Ashi) that
         also prints a 3-bar fair-value gap (break of structure + inefficiency),
      3. ENTRY IS NOT TAKEN ON THE DISPLACEMENT. The leg (sweep extreme -> displacement
         extreme) is marked and we wait up to ICT_MAX_WAIT bars for price to retrace
         into the 0.5-0.786 fib band (0.618 zone) - the high-probability discount/
         premium entry. Stop sits just beyond the swept extreme (tight risk).

    Optional HTF SMA200 bias filter. `manage="managed"` sets a structural target
    (nearest opposite liquidity pool) and arms a +1R break-even (columns `target`,
    `be_r`, read by the backtester); `manage="raw"` leaves both NaN so the runner's
    RR grid governs the exit. All references at the signal bar use only past data."""
    a = f["atr"].values
    lo, hi, cl = f["low"].values, f["high"].values, f["close"].values
    rmin = f["low"].rolling(SWEEP_LOOKBACK).min()
    rmax = f["high"].rolling(SWEEP_LOOKBACK).max()

    def pool_sweep(pools, extreme, reclaim_above):
        swept = pd.Series(False, index=f.index)
        cnt = pd.Series(0, index=f.index)
        for name in pools:
            lvl = f[name]
            hit = (extreme < lvl) & (cl > lvl) if reclaim_above else (extreme > lvl) & (cl < lvl)
            hit = hit.fillna(False)
            swept |= hit
            cnt += hit.astype(int)
        return swept, cnt

    swept_lo, nlo = pool_sweep(LOW_POOLS, rmin, True)     # bullish: raid lows, reclaim
    swept_hi, nhi = pool_sweep(HIGH_POOLS, rmax, False)   # bearish: raid highs, reclaim

    if p["use_ha"]:
        d_up = (f["ha_dir"] > 0) & (f["ha_body_ratio"] >= 0.5)
        d_dn = (f["ha_dir"] < 0) & (f["ha_body_ratio"] >= 0.5)
    else:
        d_up = (f["candle_dir"] > 0) & (f["body_ratio"] >= 0.5)
        d_dn = (f["candle_dir"] < 0) & (f["body_ratio"] >= 0.5)
    disp_up = (f["range_atr"] >= p["disp_k"]) & d_up & (f["low"] > f["high"].shift(2))
    disp_dn = (f["range_atr"] >= p["disp_k"]) & d_dn & (f["high"] < f["low"].shift(2))

    setup_long = (swept_lo & disp_up).fillna(False)
    setup_short = (swept_hi & disp_dn).fillna(False)
    if p["bias"]:
        setup_long &= (f["sma200_dist"] > 0).fillna(False)
        setup_short &= (f["sma200_dist"] < 0).fillna(False)

    managed = p["manage"] == "managed"
    high_pool_vals = [f[c].values for c in HIGH_POOLS]
    low_pool_vals = [f[c].values for c in LOW_POOLS]
    n = len(f)
    rmin_v, rmax_v = rmin.values, rmax.values
    sl_v, ss_v = setup_long.values, setup_short.values
    ncf = np.maximum(nlo.values, nhi.values)

    def draw_target(j, entry, d):
        """Nearest opposite liquidity pool at least 1x the stop-distance away."""
        if not managed:
            return np.nan
        pools = high_pool_vals if d == 1 else low_pool_vals
        need = entry + d * (entry - (rmin_v[j] if d == 1 else rmax_v[j]))  # ~+1R
        best = np.nan
        for pv in pools:
            v = pv[j]
            if np.isnan(v):
                continue
            if d == 1 and v >= need and (np.isnan(best) or v < best):
                best = v
            if d == -1 and v <= need and (np.isnan(best) or v > best):
                best = v
        return best

    rows = []
    guard = -1                          # don't start a new scan before this bar
    for t in range(n):
        d = 1 if sl_v[t] else (-1 if ss_v[t] else 0)
        if d == 0 or t <= guard or np.isnan(a[t]):
            continue
        if d == 1:
            leg_lo, leg_hi = rmin_v[t], hi[t]
        else:
            leg_lo, leg_hi = lo[t], rmax_v[t]
        rng = leg_hi - leg_lo
        if not (rng > 0):
            continue
        # long enters on a pullback DOWN into the discount band; short on a rally UP
        zone_near = leg_hi - FIB_LO * rng if d == 1 else leg_lo + FIB_LO * rng
        last = min(t + ICT_MAX_WAIT, n - 1)
        for j in range(t + 1, last + 1):
            if d == 1:
                if lo[j] < leg_lo:                       # structure broken -> cancel
                    break
                touched = lo[j] <= zone_near
            else:
                if hi[j] > leg_hi:
                    break
                touched = hi[j] >= zone_near
            if not touched:
                continue
            stop = (leg_lo - 0.1 * a[j]) if d == 1 else (leg_hi + 0.1 * a[j])
            sdist = (cl[j] - stop) * d
            if not (STOP_MIN_ATR * a[j] <= sdist <= STOP_MAX_ATR * a[j]):
                break
            conv = float(np.clip(0.5 + 0.25 * (ncf[t] - 1), 0.5, 1.5))
            tgt = draw_target(j, cl[j], d)
            be_r = 1.0 if managed else np.nan
            rows.append((j, d, stop, conv, tgt, be_r))
            guard = j                    # serialize armed setups
            break
    out = pd.DataFrame(rows, columns=["i", "dir", "stop", "conviction",
                                      "target", "be_r"])
    return out.sort_values("i").reset_index(drop=True)


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

# --- AI (neural-net) leg -----------------------------------------------------
# A feed-forward MLP on the SAME engineered features as the tree-based ML leg,
# refit per walk-forward fold with the identical embargo. It is deliberately a
# different function class (smooth, distributed representation) from the
# gradient-boosted trees, so the two carry independent errors in the ensemble.
NN_HIDDEN = (32, 8)
NN_ALPHA = 3e-3            # L2 penalty (regularization) - keep it stiff, small data edge
NN_MAX_ITER = 70


def _nn_fit(f: pd.DataFrame, lo: int, hi: int) -> Pipeline:
    """Fit a standardized MLP classifier on bars [lo, hi). Same label and
    NaN-handling contract as ml_fit; MLPClassifier has no sample_weight, so
    this leg is always uniform-weighted. StandardScaler is fit inside the
    pipeline on the training slice only (no leak); early stopping carves its
    validation split from the tail of the training window."""
    X = f[ML_FEATURES].iloc[lo:hi]
    y = ml_labels(f).iloc[lo:hi]
    ok = X.notna().all(axis=1) & y.notna()
    Xo, yo = X[ok], y[ok].astype(int)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=NN_HIDDEN, alpha=NN_ALPHA,
                              activation="relu", solver="adam",
                              batch_size=512, learning_rate_init=1e-3,
                              max_iter=NN_MAX_ITER, early_stopping=True,
                              n_iter_no_change=8, validation_fraction=0.1,
                              random_state=7)),
    ])
    pipe.fit(Xo, yo)
    return pipe


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
      nn       - feed-forward neural net (the AI leg); see _nn_fit. Ignores
                 sample weights (MLP has none), so it trains uniform-weighted.
    """
    if scheme == "nn":
        return _nn_fit(f, lo, hi)
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

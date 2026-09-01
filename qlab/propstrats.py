"""Prop-trader strategy families distilled from the 2026 web research sweep
(session playbooks, ORB literature, XAUUSD seasonality studies). Same signal
contract as qlab.strategies: DataFrame [i, dir, stop, conviction], signal bar i
uses only info <= i's close; the backtester enters at bar i+1's open.

Families (all usable on 15m and 1h bars; PULL/SQZ/MREV also on 1d):
  arb    - Asian-range breakout at/after London open (the classic prop setup)
  lfade  - London "trap" fade: poke through the Asia extreme that closes back
           inside during the London window -> trade the reversal
  orb    - NY opening-range breakout (literature says weak on gold - test it)
  pull   - trend-pullback: above/below long SMA, pull back to a fast EMA,
           enter on the resumption candle
  sqz    - volatility-squeeze breakout: Bollinger-width percentile low, then
           Donchian break
  mrev   - mean-reversion at RSI extremes (optionally trend-filtered)

Return-series drift legs (no stop/target; position * bar returns - costs):
  asia_drift - long during the Asia session (EV Trade Labs: +drift, p<0.001)
  fri_drift  - long all Friday (only day with significant bias in 23y study)
"""
import numpy as np
import pandas as pd

ET = "America/New_York"

ARB_GRID = [{"buf_atr": b, "stop_mode": sm}
            for b in (0.0, 0.25) for sm in ("range", "mid")]
LFADE_GRID = [{"poke_atr": pk} for pk in (0.0, 0.25)]
ORB_GRID = [{"orb_min": nm, "cutoff_h": 13} for nm in (30, 60)]
PULL_GRID = [{"ema_n": em, "stop_k": sk} for em in (20, 50) for sk in (1.0, 1.75)]
SQZ_GRID = [{"sq_pct": q} for q in (0.2, 0.3)]
MREV_GRID = [{"rsi_th": th, "trend_filter": tf_, "stop_k": 1.5}
             for th in (25, 30) for tf_ in (True, False)]

_BARS_PER_DAY = 96          # 15m default; set_tf() adjusts (96/24/1)


def set_tf(bars_per_day: int) -> None:
    global _BARS_PER_DAY
    _BARS_PER_DAY = bars_per_day


def _pack(masks_dirs_stops, f):
    out = []
    for mask, d, st in masks_dirs_stops:
        m = np.asarray(mask) & f["atr"].notna().values
        idx = np.flatnonzero(m)
        stv = st.values if hasattr(st, "values") else st
        out.append(pd.DataFrame({"i": idx, "dir": d, "stop": stv[idx],
                                 "conviction": 1.0}))
    return (pd.concat(out).sort_values("i").reset_index(drop=True)
            if out else pd.DataFrame(columns=["i", "dir", "stop", "conviction"]))


def _first_per_day(mask: np.ndarray, et_date) -> np.ndarray:
    """Keep only the first True per ET calendar day (one entry/day/direction)."""
    s = pd.Series(mask)
    grp = s.groupby(pd.Series(et_date)).cumsum()
    return (mask & (grp.values == 1))


def arb_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Break of the completed Asia range during the London window (2-5 ET).
    Stop: other side of the range ('range') or its midpoint ('mid')."""
    et = f.index.tz_convert(ET)
    lon = (et.hour >= 2) & (et.hour < 5)
    a = f["atr"]
    hi, lo = f["asia_hi"], f["asia_lo"]
    buf = p["buf_atr"] * a
    up = lon & (f["close"] > hi + buf) & (f["close"].shift(1) <= (hi + buf).shift(1))
    dn = lon & (f["close"] < lo - buf) & (f["close"].shift(1) >= (lo - buf).shift(1))
    mid = (hi + lo) / 2
    stop_l = lo if p["stop_mode"] == "range" else mid
    stop_s = hi if p["stop_mode"] == "range" else mid
    ed = et.date
    up = _first_per_day(up.fillna(False).values, ed)
    dn = _first_per_day(dn.fillna(False).values, ed)
    return _pack([(up, 1, stop_l - 0.1 * a), (dn, -1, stop_s + 0.1 * a)], f)


def lfade_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """London-window poke through the Asia extreme that closes back inside ->
    fade it (research: London is gold's trap-laden window)."""
    et = f.index.tz_convert(ET)
    lon = (et.hour >= 2) & (et.hour < 5)
    a = f["atr"]
    poke = p["poke_atr"] * a
    hi, lo = f["asia_hi"], f["asia_lo"]
    sh = lon & (f["high"] > hi + poke) & (f["close"] < hi)      # failed upside raid
    lg = lon & (f["low"] < lo - poke) & (f["close"] > lo)       # failed downside raid
    ed = et.date
    sh = _first_per_day(sh.fillna(False).values, ed)
    lg = _first_per_day(lg.fillna(False).values, ed)
    return _pack([(lg, 1, f["low"] - 0.1 * a), (sh, -1, f["high"] + 0.1 * a)], f)


def orb_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """NY opening-range breakout: range = first orb_bars RTH bars (from 9:30 ET),
    entry on a close beyond it before cutoff_h ET, stop at the other side."""
    et = f.index.tz_convert(ET)
    tmin = et.hour * 60 + et.minute
    rth = (tmin >= 570) & (tmin < 960)              # 9:30-16:00
    ed = pd.Series(et.date, index=f.index)
    rank = pd.Series(rth, index=f.index).groupby(ed.values).cumsum()
    nb = max(p["orb_min"] * _BARS_PER_DAY // 1440, 1)   # minutes -> bars at this tf
    in_or = rth & (rank.values <= nb)
    or_hi = f["high"].where(in_or).groupby(ed.values).cummax()
    or_lo = f["low"].where(in_or).groupby(ed.values).cummin()
    or_hi, or_lo = or_hi.ffill(), or_lo.ffill()     # within day via groupby cummax+ffill
    # invalidate once the day changes (groupby cummax already day-scoped)
    ready = rth & (rank.values > nb) & (et.hour < p["cutoff_h"])
    up = ready & (f["close"] > or_hi) & (f["close"].shift(1) <= or_hi.shift(1))
    dn = ready & (f["close"] < or_lo) & (f["close"].shift(1) >= or_lo.shift(1))
    a = f["atr"]
    up = _first_per_day(pd.Series(up).fillna(False).values, et.date)
    dn = _first_per_day(pd.Series(dn).fillna(False).values, et.date)
    return _pack([(up, 1, or_lo - 0.1 * a), (dn, -1, or_hi + 0.1 * a)], f)


def pull_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Trend + pullback + resumption: above SMA200 -> wait for a touch of the
    fast EMA, enter on the next bullish candle (mirror for shorts)."""
    c = f["close"]
    ema = c.ewm(span=p["ema_n"], adjust=False).mean()
    a = f["atr"]
    up_tr = f["sma200_dist"] > 0
    dn_tr = f["sma200_dist"] < 0
    touched_up = (f["low"] <= ema) & up_tr
    touched_dn = (f["high"] >= ema) & dn_tr
    lg = touched_up.shift(1).fillna(False) & (f["candle_dir"] > 0) & (c > ema) & up_tr
    sh = touched_dn.shift(1).fillna(False) & (f["candle_dir"] < 0) & (c < ema) & dn_tr
    # throttle: one entry per side per day
    ed = f.index.tz_convert(ET).date
    lg = _first_per_day(lg.fillna(False).values, ed)
    sh = _first_per_day(sh.fillna(False).values, ed)
    return _pack([(lg, 1, c - p["stop_k"] * a), (sh, -1, c + p["stop_k"] * a)], f)


def sqz_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Bollinger-width squeeze (bottom q of trailing year) then Donchian break."""
    c = f["close"]
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    width = (4 * s20) / m20
    lookback = max(252 * _BARS_PER_DAY, 200)        # ~1 trading year of bars
    qth = width.rolling(lookback).quantile(p["sq_pct"])
    squeezed = (width <= qth).shift(1).fillna(False)
    up = squeezed & (c > f["don_hi"].shift(1))
    dn = squeezed & (c < f["don_lo"].shift(1))
    a = f["atr"]
    return _pack([(up.fillna(False).values, 1, c - 1.5 * a),
                  (dn.fillna(False).values, -1, c + 1.5 * a)], f)


def mrev_signals(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """RSI-extreme mean reversion; optional with-trend-only filter."""
    lg = f["rsi"] < p["rsi_th"]
    sh = f["rsi"] > 100 - p["rsi_th"]
    if p["trend_filter"]:
        lg &= f["sma200_dist"] > 0
        sh &= f["sma200_dist"] < 0
    # cooldown: don't restack while still stretched - first bar of the condition
    lg &= ~lg.shift(1).fillna(False)
    sh &= ~sh.shift(1).fillna(False)
    a, c = f["atr"], f["close"]
    return _pack([(lg.fillna(False).values, 1, c - p["stop_k"] * a),
                  (sh.fillna(False).values, -1, c + p["stop_k"] * a)], f)


# ------------------------- return-series drift legs -------------------------

def session_drift_returns(bars: pd.DataFrame, cost_pts: float,
                          hours=(20, 24)) -> pd.Series:
    """Daily return of holding long only during the given ET-hour window.
    One round-trip per day -> costs charged once per day traded."""
    et = bars.index.tz_convert(ET)
    in_sess = (et.hour >= hours[0]) & (et.hour < hours[1])
    r = bars["close"].pct_change().where(in_sess, 0.0)
    daily = (1 + r).groupby(et.date).prod() - 1
    traded = pd.Series(in_sess, index=bars.index).groupby(et.date).any()
    cost = cost_pts / bars["close"].groupby(et.date).last()
    out = daily - cost.where(traded, 0.0)
    out.index = pd.to_datetime(out.index)
    return out.dropna()


def dow_drift_returns(bars: pd.DataFrame, cost_pts: float, dow: int = 4) -> pd.Series:
    """Hold long for the whole ET calendar day when weekday == dow (4=Friday)."""
    et = bars.index.tz_convert(ET)
    day_close = bars["close"].groupby(et.date).last()
    day_close.index = pd.to_datetime(day_close.index)
    ret = day_close.pct_change()
    on = ret.index.dayofweek == dow
    cost = cost_pts / day_close
    return (ret.where(on, 0.0) - cost.where(on, 0.0)).dropna()


FAMILIES = {
    "arb": ("ARB_GRID", "arb_signals"),
    "lfade": ("LFADE_GRID", "lfade_signals"),
    "orb": ("ORB_GRID", "orb_signals"),
    "pull": ("PULL_GRID", "pull_signals"),
    "sqz": ("SQZ_GRID", "sqz_signals"),
    "mrev": ("MREV_GRID", "mrev_signals"),
}


def register(WF, S) -> None:
    """Plug every family into qlab.walkforward's rules registry so run_wf can
    drive them exactly like smc/ta/ict."""
    import qlab.propstrats as P
    for fam, (grid, gen) in FAMILIES.items():
        setattr(S, grid, getattr(P, grid))
        setattr(S, gen, getattr(P, gen))
        WF._RULES_GRID[fam] = grid
        WF._RULES_GEN[fam] = gen
    WF.RULES_STRATS = tuple(list(WF.RULES_STRATS) + list(FAMILIES))

"""Canonical candlestick pattern library.

Definitions follow the standard literature (Nison's *Japanese Candlestick
Charting Techniques*, Bulkowski's *Encyclopedia of Candlestick Charts*) rather
than anything tuned to this data. Every threshold that must exist is expressed
relative to TRAILING volatility (shifted ATR) or the bar's own range, never as a
fitted constant, so nothing here is optimised.

NO LOOKAHEAD: a pattern at bar t uses only bars <= t. Trend context uses a
trailing SMA shifted by one bar. The caller enters at bar t+1's OPEN.

Each function returns +1 (bullish), -1 (bearish) or 0, aligned to the bar on
which the pattern COMPLETES.
"""
import numpy as np
import pandas as pd

# Standard proportions from the canon. These are definitional, not tuned:
DOJI_BODY = 0.05        # body <= 5% of range  -> doji
LONG_BODY = 0.60        # body >= 60% of range -> long/marubozu-ish
SMALL_BODY = 0.30       # body <= 30% of range -> small body
LONG_SHADOW = 2.0       # shadow >= 2x body    -> hammer/shooting-star shadow
SHORT_SHADOW = 0.10     # opposite shadow <= 10% of range
NEAR = 0.05             # "equal" within 5% of ATR (tweezers, abandoned baby)


def anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar geometry. All trailing / same-bar only."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    a = pd.DataFrame(index=df.index)
    a["range"] = rng
    a["body"] = body
    a["body_pct"] = body / rng
    a["upper"] = (h - top) / rng
    a["lower"] = (bot - l) / rng
    a["up"] = (c > o).astype(int)
    a["dn"] = (c < o).astype(int)
    a["o"], a["h"], a["l"], a["c"] = o, h, l, c
    a["top"], a["bot"] = top, bot
    # trailing context, shifted so bar t never sees its own future
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    a["atr"] = tr.ewm(span=14, adjust=False).mean().shift(1)
    a["sma20"] = c.rolling(20).mean().shift(1)
    a["trend_up"] = (c.shift(1) > a["sma20"]).astype(int)
    a["trend_dn"] = (c.shift(1) < a["sma20"]).astype(int)
    a["big"] = (rng > a["atr"]).astype(int)
    return a


# ----------------------------------------------------------- single-bar
def doji(a):
    return np.where(a["body_pct"] <= DOJI_BODY, 1, 0) * 0   # neutral by itself


def hammer(a):
    """Long lower shadow, small upper, after a downtrend -> bullish reversal."""
    ok = ((a["lower"] >= 0.5) & (a["upper"] <= SHORT_SHADOW * 2) &
          (a["body_pct"] <= SMALL_BODY) & (a["trend_dn"] == 1))
    return np.where(ok, 1, 0)


def hanging_man(a):
    """Same shape as the hammer but after an uptrend -> bearish."""
    ok = ((a["lower"] >= 0.5) & (a["upper"] <= SHORT_SHADOW * 2) &
          (a["body_pct"] <= SMALL_BODY) & (a["trend_up"] == 1))
    return np.where(ok, -1, 0)


def shooting_star(a):
    ok = ((a["upper"] >= 0.5) & (a["lower"] <= SHORT_SHADOW * 2) &
          (a["body_pct"] <= SMALL_BODY) & (a["trend_up"] == 1))
    return np.where(ok, -1, 0)


def inverted_hammer(a):
    ok = ((a["upper"] >= 0.5) & (a["lower"] <= SHORT_SHADOW * 2) &
          (a["body_pct"] <= SMALL_BODY) & (a["trend_dn"] == 1))
    return np.where(ok, 1, 0)


def marubozu(a):
    """Body fills the bar: strong continuation signal in its own direction."""
    ok = (a["body_pct"] >= 0.90) & (a["big"] == 1)
    return np.where(ok & (a["up"] == 1), 1, np.where(ok & (a["dn"] == 1), -1, 0))


def spinning_top(a):
    return np.where((a["body_pct"] <= SMALL_BODY) & (a["upper"] >= 0.25) &
                    (a["lower"] >= 0.25), 0, 0)          # indecision: no direction


def belt_hold(a):
    """Opens at the extreme and closes strongly the other way."""
    bull = (a["up"] == 1) & (a["lower"] <= 0.02) & (a["body_pct"] >= LONG_BODY) & (a["trend_dn"] == 1)
    bear = (a["dn"] == 1) & (a["upper"] <= 0.02) & (a["body_pct"] >= LONG_BODY) & (a["trend_up"] == 1)
    return np.where(bull, 1, np.where(bear, -1, 0))


# ----------------------------------------------------------- two-bar
def engulfing(a):
    p_top, p_bot = a["top"].shift(1), a["bot"].shift(1)
    bull = ((a["up"] == 1) & (a["dn"].shift(1) == 1) &
            (a["bot"] <= p_bot) & (a["top"] >= p_top) & (a["trend_dn"] == 1))
    bear = ((a["dn"] == 1) & (a["up"].shift(1) == 1) &
            (a["bot"] <= p_bot) & (a["top"] >= p_top) & (a["trend_up"] == 1))
    return np.where(bull, 1, np.where(bear, -1, 0))


def harami(a):
    p_top, p_bot = a["top"].shift(1), a["bot"].shift(1)
    inside = (a["top"] <= p_top) & (a["bot"] >= p_bot)
    bull = inside & (a["dn"].shift(1) == 1) & (a["up"] == 1) & (a["trend_dn"] == 1)
    bear = inside & (a["up"].shift(1) == 1) & (a["dn"] == 1) & (a["trend_up"] == 1)
    return np.where(bull, 1, np.where(bear, -1, 0))


def piercing(a):
    """Bear bar, then a gap-down open closing above the midpoint."""
    mid = (a["o"].shift(1) + a["c"].shift(1)) / 2
    ok = ((a["dn"].shift(1) == 1) & (a["up"] == 1) & (a["o"] < a["c"].shift(1)) &
          (a["c"] > mid) & (a["c"] < a["o"].shift(1)) & (a["trend_dn"] == 1))
    return np.where(ok, 1, 0)


def dark_cloud(a):
    mid = (a["o"].shift(1) + a["c"].shift(1)) / 2
    ok = ((a["up"].shift(1) == 1) & (a["dn"] == 1) & (a["o"] > a["c"].shift(1)) &
          (a["c"] < mid) & (a["c"] > a["o"].shift(1)) & (a["trend_up"] == 1))
    return np.where(ok, -1, 0)


def tweezer(a):
    tol = NEAR * a["atr"]
    bot_eq = (a["l"] - a["l"].shift(1)).abs() <= tol
    top_eq = (a["h"] - a["h"].shift(1)).abs() <= tol
    bull = bot_eq & (a["trend_dn"] == 1) & (a["up"] == 1) & (a["dn"].shift(1) == 1)
    bear = top_eq & (a["trend_up"] == 1) & (a["dn"] == 1) & (a["up"].shift(1) == 1)
    return np.where(bull, 1, np.where(bear, -1, 0))


def kicker(a):
    """Opposite-colour marubozu pair separated by a gap: a violent reversal."""
    gap_up = a["o"] > a["o"].shift(1)
    gap_dn = a["o"] < a["o"].shift(1)
    strong = a["body_pct"] >= LONG_BODY
    bull = strong & (a["up"] == 1) & (a["dn"].shift(1) == 1) & gap_up & (a["o"] > a["c"].shift(1))
    bear = strong & (a["dn"] == 1) & (a["up"].shift(1) == 1) & gap_dn & (a["o"] < a["c"].shift(1))
    return np.where(bull, 1, np.where(bear, -1, 0))


# ----------------------------------------------------------- three-bar
def morning_star(a):
    c1_dn = a["dn"].shift(2) == 1
    c1_long = a["body_pct"].shift(2) >= LONG_BODY
    c2_small = a["body_pct"].shift(1) <= SMALL_BODY
    c2_gap = a["top"].shift(1) < a["c"].shift(2)
    c3 = (a["up"] == 1) & (a["c"] > (a["o"].shift(2) + a["c"].shift(2)) / 2)
    return np.where(c1_dn & c1_long & c2_small & c2_gap & c3 & (a["trend_dn"] == 1), 1, 0)


def evening_star(a):
    c1_up = a["up"].shift(2) == 1
    c1_long = a["body_pct"].shift(2) >= LONG_BODY
    c2_small = a["body_pct"].shift(1) <= SMALL_BODY
    c2_gap = a["bot"].shift(1) > a["c"].shift(2)
    c3 = (a["dn"] == 1) & (a["c"] < (a["o"].shift(2) + a["c"].shift(2)) / 2)
    return np.where(c1_up & c1_long & c2_small & c2_gap & c3 & (a["trend_up"] == 1), -1, 0)


def three_soldiers(a):
    up3 = (a["up"] == 1) & (a["up"].shift(1) == 1) & (a["up"].shift(2) == 1)
    rising = (a["c"] > a["c"].shift(1)) & (a["c"].shift(1) > a["c"].shift(2))
    solid = (a["body_pct"] >= 0.5) & (a["body_pct"].shift(1) >= 0.5)
    opens_in = (a["o"] < a["c"].shift(1)) & (a["o"] > a["o"].shift(1))
    return np.where(up3 & rising & solid & opens_in, 1, 0)


def three_crows(a):
    dn3 = (a["dn"] == 1) & (a["dn"].shift(1) == 1) & (a["dn"].shift(2) == 1)
    falling = (a["c"] < a["c"].shift(1)) & (a["c"].shift(1) < a["c"].shift(2))
    solid = (a["body_pct"] >= 0.5) & (a["body_pct"].shift(1) >= 0.5)
    opens_in = (a["o"] > a["c"].shift(1)) & (a["o"] < a["o"].shift(1))
    return np.where(dn3 & falling & solid & opens_in, -1, 0)


def three_inside(a):
    h = harami(a)
    conf_up = (h_shift(h, 1) == 1) & (a["c"] > a["c"].shift(1)) & (a["up"] == 1)
    conf_dn = (h_shift(h, 1) == -1) & (a["c"] < a["c"].shift(1)) & (a["dn"] == 1)
    return np.where(conf_up, 1, np.where(conf_dn, -1, 0))


def h_shift(arr, n):
    return pd.Series(arr).shift(n).fillna(0).values


def abandoned_baby(a):
    tol = NEAR * a["atr"]
    doji_mid = a["body_pct"].shift(1) <= DOJI_BODY
    bull = (doji_mid & (a["h"].shift(1) < a["l"].shift(2)) & (a["l"] > a["h"].shift(1)) &
            (a["dn"].shift(2) == 1) & (a["up"] == 1))
    bear = (doji_mid & (a["l"].shift(1) > a["h"].shift(2)) & (a["h"] < a["l"].shift(1)) &
            (a["up"].shift(2) == 1) & (a["dn"] == 1))
    return np.where(bull, 1, np.where(bear, -1, 0))


# ----------------------------------------------------------- range / gap
def inside_bar(a):
    ins = (a["h"] < a["h"].shift(1)) & (a["l"] > a["l"].shift(1))
    return np.where(ins & (a["trend_up"] == 1), 1,
                    np.where(ins & (a["trend_dn"] == 1), -1, 0))


def outside_bar(a):
    out = (a["h"] > a["h"].shift(1)) & (a["l"] < a["l"].shift(1))
    return np.where(out & (a["up"] == 1), 1, np.where(out & (a["dn"] == 1), -1, 0))


def nr4(a):
    """Narrowest range of 4 — a compression signal, traded in trend direction."""
    n = a["range"] <= a["range"].rolling(4).min()
    return np.where(n & (a["trend_up"] == 1), 1, np.where(n & (a["trend_dn"] == 1), -1, 0))


def gap_go(a):
    """Gap beyond the prior range that holds into the close."""
    gu = (a["l"] > a["h"].shift(1)) & (a["up"] == 1)
    gd = (a["h"] < a["l"].shift(1)) & (a["dn"] == 1)
    return np.where(gu, 1, np.where(gd, -1, 0))


PATTERNS = {
    "hammer": hammer, "hanging_man": hanging_man, "shooting_star": shooting_star,
    "inverted_hammer": inverted_hammer, "marubozu": marubozu,
    "belt_hold": belt_hold, "engulfing": engulfing, "harami": harami,
    "piercing": piercing, "dark_cloud": dark_cloud, "tweezer": tweezer,
    "kicker": kicker, "morning_star": morning_star, "evening_star": evening_star,
    "three_soldiers": three_soldiers, "three_crows": three_crows,
    "three_inside": three_inside, "abandoned_baby": abandoned_baby,
    "inside_bar": inside_bar, "outside_bar": outside_bar, "nr4": nr4,
    "gap_go": gap_go,
}


def signals(df: pd.DataFrame, name: str) -> pd.Series:
    a = anatomy(df)
    s = pd.Series(PATTERNS[name](a), index=df.index).fillna(0)
    return s.replace([np.inf, -np.inf], 0)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Candle GEOMETRY features for the ML/AI legs - no pattern labels, just the
    raw shape of recent bars, so the model can find combinations the canon missed."""
    a = anatomy(df)
    X = pd.DataFrame(index=df.index)
    for k in range(3):
        X[f"body_pct_{k}"] = a["body_pct"].shift(k)
        X[f"upper_{k}"] = a["upper"].shift(k)
        X[f"lower_{k}"] = a["lower"].shift(k)
        X[f"dir_{k}"] = (a["up"] - a["dn"]).shift(k)
        X[f"range_atr_{k}"] = (a["range"] / a["atr"]).shift(k)
    X["gap"] = ((df["open"] - df["close"].shift(1)) / a["atr"])
    X["close_pos"] = (df["close"] - df["low"]) / a["range"]
    X["trend"] = a["trend_up"] - a["trend_dn"]
    X["atr_pct"] = a["atr"] / df["close"]
    return X.replace([np.inf, -np.inf], np.nan)

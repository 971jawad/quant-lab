"""Market Wizards canon, tested as published - CANONICAL parameters, no grids.

  turtle_s1   20d breakout entry / 10d opposite exit / 2N stop / skip-after-win
  turtle_s2   55d breakout entry / 20d opposite exit / 2N stop
  tsoup       Raschke-Connors Turtle Soup: failed 20d-extreme breakout fade,
              prior extreme >= 4 sessions old, stop beyond today's extreme,
              time exit 4 days
  lw_break    Larry Williams volatility breakout: stop-entry at open +/- 0.6 x
              prior day's range, exit on close (day trade)
  holy_grail  Raschke Holy Grail: ADX(14) > 30, pullback to EMA20, enter on
              break of prior day's extreme, stop at pullback low, exit 10d
  r_squeeze   Raschke volatility squeeze: 6d HV < 0.5 x 100d HV AND (NR4 or
              inside day) -> next-day range breakout both sides, 3-day exit

Daily bars, dev window unless --final. Execution: signal at close -> act next
day; stop-entry fills at max(open, level); stops fill at level or worse (gap).
Position sizing: 1R risk per trade tracked in R-multiples; daily return series
built at 1% risk/trade for the metrics engine. Every system logged to ledger.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.metrics import full_metrics, summarize_line
from run_research import SERIES, load_15m, to_tf

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01", tz="UTC")
COSTS_PTS = {"XAUUSD": 0.45, "MNQ": 1.62, "EURUSD": 0.00016}   # round trip
RISK = 0.01


def wilder_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _trades_to_daily(trades, idx, cost_frac):
    """R-multiples -> daily return series at RISK per trade (PnL on exit day)."""
    eq = pd.Series(0.0, index=idx)
    for exit_i, R in trades:
        eq.iloc[exit_i] += RISK * R - cost_frac[exit_i]
    return eq


def turtle(d, sys2=False):
    """Returns list of (exit_index, R). Canonical Turtle state machine."""
    hi_n, lo_n = (55, 20) if sys2 else (20, 10)
    h, l, o, c = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    ehi = pd.Series(h, d.index).rolling(hi_n).max().shift(1).values
    elo = pd.Series(l, d.index).rolling(hi_n).min().shift(1).values
    xhi = pd.Series(h, d.index).rolling(lo_n).max().shift(1).values
    xlo = pd.Series(l, d.index).rolling(lo_n).min().shift(1).values
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    N = tr.ewm(alpha=1 / 20, adjust=False).mean().shift(1).values
    trades, pos, last_won = [], 0, False
    entry = stop = 0.0
    for i in range(hi_n + 1, len(d)):
        if pos == 0:
            if np.isnan(N[i]) or N[i] <= 0:
                continue
            skip = (not sys2) and last_won            # S1 skip-after-winner
            if h[i] > ehi[i] and not skip:
                entry, pos = max(o[i], ehi[i]), 1
                stop = entry - 2 * N[i]
            elif l[i] < elo[i] and not skip:
                entry, pos = min(o[i], elo[i]), -1
                stop = entry + 2 * N[i]
            elif (h[i] > ehi[i] or l[i] < elo[i]) and skip:
                last_won = False                      # a skipped signal resets
        else:
            risk = abs(entry - stop)
            if pos == 1:
                if l[i] <= stop:
                    px = min(o[i], stop)
                    trades.append((i, (px - entry) / risk)); last_won = px > entry
                    pos = 0
                elif l[i] < xlo[i]:
                    px = min(o[i], xlo[i])
                    trades.append((i, (px - entry) / risk)); last_won = px > entry
                    pos = 0
            else:
                if h[i] >= stop:
                    px = max(o[i], stop)
                    trades.append((i, (entry - px) / risk)); last_won = px < entry
                    pos = 0
                elif h[i] > xhi[i]:
                    px = max(o[i], xhi[i])
                    trades.append((i, (entry - px) / risk)); last_won = px < entry
                    pos = 0
    return trades


def tsoup(d):
    h, l, o, c = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    lo20 = pd.Series(l, d.index).rolling(20).min().shift(1)
    hi20 = pd.Series(h, d.index).rolling(20).max().shift(1)
    argmin = pd.Series(l, d.index).rolling(20).apply(np.argmin, raw=True).shift(1)
    argmax = pd.Series(h, d.index).rolling(20).apply(np.argmax, raw=True).shift(1)
    trades = []
    i = 21
    while i < len(d) - 5:
        # long: new 20d low, prior low >= 4 sessions old, close back above it
        if l[i] < lo20.iloc[i] and (19 - argmin.iloc[i]) >= 4 and c[i] > lo20.iloc[i]:
            entry, stop = c[i], l[i]
            risk = entry - stop
            if risk > 0:
                done = False
                for j in range(i + 1, min(i + 5, len(d))):
                    if l[j] <= stop:
                        trades.append((j, (min(o[j], stop) - entry) / risk)); done = True; break
                if not done:
                    j = min(i + 4, len(d) - 1)
                    trades.append((j, (c[j] - entry) / risk))
                i = j + 1
                continue
        if h[i] > hi20.iloc[i] and (19 - argmax.iloc[i]) >= 4 and c[i] < hi20.iloc[i]:
            entry, stop = c[i], h[i]
            risk = stop - entry
            if risk > 0:
                done = False
                for j in range(i + 1, min(i + 5, len(d))):
                    if h[j] >= stop:
                        trades.append((j, (entry - max(o[j], stop)) / risk)); done = True; break
                if not done:
                    j = min(i + 4, len(d) - 1)
                    trades.append((j, (entry - c[j]) / risk))
                i = j + 1
                continue
        i += 1
    return trades


def lw_break(d, k=0.6):
    h, l, o, c = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    rng = (d["high"] - d["low"]).shift(1).values
    trades = []
    for i in range(2, len(d)):
        if np.isnan(rng[i]) or rng[i] <= 0:
            continue
        up, dn = o[i] + k * rng[i], o[i] - k * rng[i]
        if h[i] >= up:
            entry = max(up, o[i])
            trades.append((i, (c[i] - entry) / (k * rng[i])))
        elif l[i] <= dn:
            entry = min(dn, o[i])
            trades.append((i, (entry - c[i]) / (k * rng[i])))
    return trades


def holy_grail(d):
    adx = wilder_adx(d).shift(1)
    ema20 = d["close"].ewm(span=20, adjust=False).mean()
    h, l, o, c = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    e = ema20.values
    trades = []
    i = 30
    while i < len(d) - 1:
        if adx.iloc[i] > 30 and c[i] > e[i] and l[i] <= e[i]:       # up-trend pullback
            trig, stop = h[i], l[i]
            for j in range(i + 1, min(i + 11, len(d))):
                if h[j] > trig:
                    entry = max(o[j], trig)
                    risk = entry - stop
                    if risk <= 0:
                        break
                    for k2 in range(j, min(j + 11, len(d))):
                        if l[k2] <= stop:
                            trades.append((k2, (min(o[k2], stop) - entry) / risk)); break
                    else:
                        k2 = min(j + 10, len(d) - 1)
                        trades.append((k2, (c[k2] - entry) / risk))
                    i = k2
                    break
                if l[j] <= stop:
                    break
        elif adx.iloc[i] > 30 and c[i] < e[i] and h[i] >= e[i]:     # down-trend rally
            trig, stop = l[i], h[i]
            for j in range(i + 1, min(i + 11, len(d))):
                if l[j] < trig:
                    entry = min(o[j], trig)
                    risk = stop - entry
                    if risk <= 0:
                        break
                    for k2 in range(j, min(j + 11, len(d))):
                        if h[k2] >= stop:
                            trades.append((k2, (entry - max(o[k2], stop)) / risk)); break
                    else:
                        k2 = min(j + 10, len(d) - 1)
                        trades.append((k2, (entry - c[k2]) / risk))
                    i = k2
                    break
                if h[j] >= stop:
                    break
        i += 1
    return trades


def r_squeeze(d):
    r = np.log(d["close"] / d["close"].shift(1))
    hv6 = r.rolling(6).std()
    hv100 = r.rolling(100).std()
    rng = d["high"] - d["low"]
    nr4 = rng <= rng.rolling(4).min()
    inside = (d["high"] < d["high"].shift(1)) & (d["low"] > d["low"].shift(1))
    setup = ((hv6 < 0.5 * hv100) & (nr4 | inside)).shift(1).fillna(False).values
    h, l, o, c = d["high"].values, d["low"].values, d["open"].values, d["close"].values
    ph, pl = d["high"].shift(1).values, d["low"].shift(1).values
    trades = []
    for i in range(101, len(d) - 4):
        if not setup[i]:
            continue
        risk = ph[i] - pl[i]
        if risk <= 0:
            continue
        if h[i] > ph[i]:                                   # upside break
            entry = max(o[i], ph[i])
            j = min(i + 3, len(d) - 1)
            stopped = False
            for k2 in range(i, j + 1):
                if l[k2] <= pl[i]:
                    trades.append((k2, (pl[i] - entry) / risk)); stopped = True; break
            if not stopped:
                trades.append((j, (c[j] - entry) / risk))
        elif l[i] < pl[i]:
            entry = min(o[i], pl[i])
            j = min(i + 3, len(d) - 1)
            stopped = False
            for k2 in range(i, j + 1):
                if h[k2] >= ph[i]:
                    trades.append((k2, (ph[i] - entry) / risk)); stopped = True; break
            if not stopped:
                trades.append((j, (entry - c[j]) / risk))
    return trades


SYSTEMS = {"turtle_s1": lambda d: turtle(d, False),
           "turtle_s2": lambda d: turtle(d, True),
           "tsoup": tsoup, "lw_break": lw_break,
           "holy_grail": holy_grail, "r_squeeze": r_squeeze}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    rows = []
    for inst in ("XAUUSD", "MNQ", "EURUSD"):
        d = to_tf(load_15m(SERIES[inst]), "1d")
        d = d[d.index >= DEV_END] if args.final else d[d.index < DEV_END]
        cost_frac = (COSTS_PTS[inst] / d["close"]).values * 0  # costs inside R below
        for name, fn in SYSTEMS.items():
            trades = fn(d)
            if not trades:
                continue
            # cost in R units: round-trip points / risk-per-trade points is trade
            # specific; conservative flat deduction of cost/ATR20
            atr = (d["high"] - d["low"]).rolling(20).mean()
            Rs = []
            daily = pd.Series(0.0, index=d.index)
            for exit_i, R in trades:
                cost_R = COSTS_PTS[inst] / max(atr.iloc[exit_i], 1e-9)
                Rn = R - cost_R
                Rs.append(Rn)
                daily.iloc[exit_i] += RISK * Rn
            m = full_metrics(np.array(Rs), daily)
            tag = f"{inst}_{name}"
            m["model"] = tag
            rows.append(m)
            print(summarize_line(tag, m))
            if not args.final:
                with open(OUT / "ledger.jsonl", "a") as fh:
                    fh.write(json.dumps({"phase": "wizards", "family": tag,
                                         "n_configs": 1,
                                         "dev_t": m.get("t_stat_trades"),
                                         "dev_sharpe": m.get("sharpe")}) + "\n")
    tag = "holdout" if args.final else "dev"
    pd.DataFrame(rows).to_csv(OUT / f"summary_wizards_{tag}.csv", index=False)
    print(f"wrote research/summary_wizards_{tag}.csv")


if __name__ == "__main__":
    main()

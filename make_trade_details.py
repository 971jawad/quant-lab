"""Actual executable trade detail for every open position.

This book is POSITION-BASED, not an entry/stop/target system: risk is controlled
by volatility sizing and the book-level risk engine, not by a fixed stop price.
But every momentum signal has a computable price at which it FLIPS — for a
sign(close / close[-L] - 1) rule that price is simply close[-L]. That level is
the functional equivalent of a stop, and it is what this script reports.

Produces, per open leg:
  direction, entry date, entry price (the open we would have executed at),
  current price, unrealised %, the flip/reversal level, distance to it,
  and the ATR-based risk unit.

Also emits the last closed round-trips with real entry/exit prices and dates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_ensembler import DRIFT_MARKETS, LOOKBACKS, MIN_TRAIN, TEST_LEN, WARMUP
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"


def chosen_lookback(inst, long_only):
    """Replays the same walk-forward selection the live book uses."""
    d = daily(inst)
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    cand = {}
    for L in LOOKBACKS:
        sig = np.sign(d["close"].pct_change(L))
        if long_only:
            sig = sig.clip(lower=0)
        pos = (sig * (0.10 / np.sqrt(252)) / vol).clip(-3, 3).shift(1)
        turn = pos.diff().abs().fillna(pos.abs())
        cand[L] = (pos * r - turn * COST[inst] / d["close"])
    n, te, best = len(d), WARMUP + MIN_TRAIN, None
    while te < n - 100:
        b, bs = None, -np.inf
        for L, s in cand.items():
            tr = s.iloc[WARMUP:te].dropna()
            sh = tr.mean() / (tr.std() + 1e-12) if len(tr) > 30 else -np.inf
            if sh > bs:
                bs, b = sh, L
        best, te = b, min(te + TEST_LEN, n)
    return d, best


def atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def leg_detail(inst):
    long_only = inst in DRIFT_MARKETS
    d, L = chosen_lookback(inst, long_only)
    c = d["close"]
    sig = np.sign(c.pct_change(L))
    if long_only:
        sig = sig.clip(lower=0)
    sig = sig.shift(1)                      # what we act on today
    cur = float(sig.iloc[-1])
    a = float(atr(d).iloc[-1])
    px_now = float(c.iloc[-1])

    # when did the current direction begin?
    run = sig[sig != cur]
    start_i = (d.index.get_loc(run.index[-1]) + 1) if len(run) else 0
    start_i = min(start_i, len(d) - 1)
    entry_date = d.index[start_i]
    entry_px = float(d["open"].iloc[start_i])   # executed at that session's open

    # the price at which the signal reverses = the close L bars ago
    flip = float(c.iloc[-L]) if len(c) > L else np.nan
    unreal = ((px_now - entry_px) / entry_px * 100 * (cur if cur != 0 else 0))
    dist = ((px_now - flip) / px_now * 100) if flip == flip else np.nan

    return {
        "market": inst,
        "timeframe": "1 day",
        "mode": "LONG-ONLY" if long_only else "long/short",
        "direction": "LONG" if cur > 0 else ("SHORT" if cur < 0 else "FLAT"),
        "lookback_days": int(L),
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "entry_price": round(entry_px, 5),
        "current_price": round(px_now, 5),
        "unrealised_pct": round(float(unreal), 2),
        "days_held": int((d.index[-1] - entry_date).days),
        "flip_level": round(flip, 5) if flip == flip else None,
        "pct_to_flip": round(float(dist), 2) if dist == dist else None,
        "atr14": round(a, 5),
        "round_trip_cost": COST[inst],
    }


def closed_trades(inst, k=6):
    long_only = inst in DRIFT_MARKETS
    d, L = chosen_lookback(inst, long_only)
    c = d["close"]
    sig = np.sign(c.pct_change(L))
    if long_only:
        sig = sig.clip(lower=0)
    pos = sig.shift(1)
    rows, cur, e_i = [], 0.0, None
    for i in range(len(d)):
        p = pos.iloc[i]
        if np.isnan(p):
            continue
        if p != cur:
            if cur != 0 and e_i is not None:
                en, ex = float(d["open"].iloc[e_i]), float(d["open"].iloc[i])
                net = (ex - en) * cur - COST[inst]
                rows.append({"market": inst,
                             "dir": "LONG" if cur > 0 else "SHORT",
                             "entry_date": d.index[e_i].strftime("%Y-%m-%d"),
                             "entry_price": round(en, 5),
                             "exit_date": d.index[i].strftime("%Y-%m-%d"),
                             "exit_price": round(ex, 5),
                             "days": int((d.index[i] - d.index[e_i]).days),
                             "net_pct": round(net / en * 100, 2)})
            cur, e_i = p, (i if p != 0 else None)
    return rows[-k:]


def main():
    open_legs, recent = [], []
    for inst in MKT:
        open_legs.append(leg_detail(inst))
        recent += closed_trades(inst)
    recent.sort(key=lambda r: r["exit_date"], reverse=True)
    recent = recent[:20]

    print("OPEN POSITIONS — daily bars, signal at close, executed next open")
    print(f"{'market':9} {'dir':6} {'entry date':11} {'entry':>10} {'now':>10} "
          f"{'unreal':>8} {'flip level':>11} {'to flip':>8} {'held':>6}")
    for r in open_legs:
        print(f"  {r['market']:7} {r['direction']:6} {r['entry_date']:11} "
              f"{r['entry_price']:>10} {r['current_price']:>10} "
              f"{r['unrealised_pct']:>7}% {str(r['flip_level']):>11} "
              f"{str(r['pct_to_flip']):>7}% {r['days_held']:>5}d")

    print("\nMOST RECENT CLOSED ROUND-TRIPS")
    print(f"{'market':9} {'dir':6} {'entry':>21} {'exit':>21} {'days':>5} {'net':>8}")
    for r in recent[:10]:
        print(f"  {r['market']:7} {r['dir']:6} {r['entry_date']} @{r['entry_price']:>9} "
              f"  {r['exit_date']} @{r['exit_price']:>9} {r['days']:>5} {r['net_pct']:>7}%")

    json.dump({"open": open_legs, "recent": recent},
              open(OUT / "trade_details.json", "w"), indent=2, default=str)
    print(f"\nwrote research/trade_details.json")


if __name__ == "__main__":
    main()

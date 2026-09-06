"""DAILY PICKS — the action list, not just the standing positions.

The dashboard already shows what the book HOLDS. This produces what a trader
would actually DO each day: the diff between yesterday's position and today's,
expressed as concrete actions with the price they execute at.

Because the system reads the signal at the daily close and executes at the NEXT
session's open, a pick generated on date D is executed at D+1's open. That is
exactly how the backtest fills, so the picks log and the equity curve describe
the same trades.

Actions:
  OPEN LONG / OPEN SHORT   position established from flat
  CLOSE                    position closed to flat
  FLIP TO LONG/SHORT       direction reversed (close + open in one action)
  HOLD                     unchanged (reported only for the current day)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_ensembler import DRIFT_MARKETS, LOOKBACKS, MIN_TRAIN, TEST_LEN, WARMUP
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
LOOKBACK_DAYS = 90          # how much pick history to publish


def chosen_lookback(inst, long_only):
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


def label(prev, cur):
    if prev == cur:
        return "HOLD"
    if prev == 0:
        return "OPEN LONG" if cur > 0 else "OPEN SHORT"
    if cur == 0:
        return "CLOSE"
    return "FLIP TO LONG" if cur > 0 else "FLIP TO SHORT"


def main():
    picks, today_rows = [], []
    for inst in MKT:
        long_only = inst in DRIFT_MARKETS
        d, L = chosen_lookback(inst, long_only)
        c = d["close"]
        sig = np.sign(c.pct_change(L))
        if long_only:
            sig = sig.clip(lower=0)
        acted = sig.shift(1).fillna(0.0)          # position held on each day
        # signal computed on today's close -> executed at tomorrow's open
        sig_today = float(sig.iloc[-1])
        cur = float(acted.iloc[-1])

        # historical action log
        chg = acted.diff().fillna(0.0)
        for i in np.flatnonzero(chg.values != 0):
            if i < len(d) - LOOKBACK_DAYS * 3:
                continue
            prev, now = float(acted.iloc[i - 1]), float(acted.iloc[i])
            picks.append({
                "signal_date": d.index[i - 1].strftime("%Y-%m-%d"),
                "execute_date": d.index[i].strftime("%Y-%m-%d"),
                "market": inst,
                "action": label(prev, now),
                "direction": "LONG" if now > 0 else ("SHORT" if now < 0 else "FLAT"),
                "exec_price": round(float(d["open"].iloc[i]), 5),
            })

        # what to do at the NEXT open
        pending = label(cur, sig_today)
        today_rows.append({
            "market": inst,
            "held_now": "LONG" if cur > 0 else ("SHORT" if cur < 0 else "FLAT"),
            "signal_on_last_close": "LONG" if sig_today > 0 else ("SHORT" if sig_today < 0 else "FLAT"),
            "action_next_open": pending,
            "last_close": round(float(c.iloc[-1]), 5),
            "as_of": d.index[-1].strftime("%Y-%m-%d"),
            "mode": "LONG-ONLY" if long_only else "long/short",
            "lookback_days": int(L),
        })

    picks.sort(key=lambda p: p["execute_date"], reverse=True)
    picks = picks[:60]

    print("ACTION AT THE NEXT OPEN (signal read on the last close)")
    print(f"  {'market':9} {'holding':8} {'signal':8} {'action':15} {'last close':>12}")
    for r in today_rows:
        flag = "" if r["action_next_open"] == "HOLD" else "  <<<"
        print(f"  {r['market']:9} {r['held_now']:8} {r['signal_on_last_close']:8} "
              f"{r['action_next_open']:15} {r['last_close']:>12}{flag}")

    acts = [r for r in today_rows if r["action_next_open"] != "HOLD"]
    print(f"\n  {len(acts)} action(s) pending, {len(today_rows) - len(acts)} holds")

    print(f"\nRECENT PICKS (last {min(len(picks), 12)} of {len(picks)} in the log)")
    print(f"  {'signal':11} {'execute':11} {'market':9} {'action':15} {'price':>12}")
    for p in picks[:12]:
        print(f"  {p['signal_date']:11} {p['execute_date']:11} {p['market']:9} "
              f"{p['action']:15} {p['exec_price']:>12}")

    json.dump({"today": today_rows, "log": picks,
               "n_pending": len(acts)},
              open(OUT / "daily_picks.json", "w"), indent=2, default=str)
    print(f"\nwrote research/daily_picks.json")


if __name__ == "__main__":
    main()

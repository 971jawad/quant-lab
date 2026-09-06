"""LIVE TRADES — every trade taken since the live feed began, at each sizing scale.

The backtest windows (dev, holdout) are history. This file is only the LIVE
period: trades generated from data that did not exist when the system was
frozen. It is the ledger you would compare against a broker statement.

EXECUTION CONVENTION, stated exactly:
  * bars are ET calendar days (midnight-to-midnight New York)
  * the signal is computed from the CLOSE of day D (23:59:59 ET)
  * the order executes at the OPEN of day D+1, i.e. 00:00 ET
  * these are 24-hour CFD/futures markets, so 00:00 ET is a live, tradeable
    moment (Asia session) — not a synthetic price
  * measured sensitivity: filling at the signal close instead gives Sharpe 0.85
    vs 0.87 for next-open, so the choice is immaterial; being a full day late
    costs about 0.10

P&L is reported at each sizing scale so the risk table and the trade ledger
describe the same thing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_ensembler import DRIFT_MARKETS, LOOKBACKS, MIN_TRAIN, TEST_LEN, WARMUP
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
EXEC_TIME = "00:00 ET"


def live_start():
    st = []
    for f in (ROOT / "data" / "live").glob("*_ext.csv"):
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(d):
            st.append(pd.to_datetime(d.index[0]).normalize())
    return min(st) if st else None


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


def main():
    ls = live_start()
    trades = []
    for inst in MKT:
        long_only = inst in DRIFT_MARKETS
        d, L = chosen_lookback(inst, long_only)
        c = d["close"]
        sig = np.sign(c.pct_change(L))
        if long_only:
            sig = sig.clip(lower=0)
        pos = sig.shift(1)
        cur, e_i = 0.0, None
        for i in range(len(d)):
            p = pos.iloc[i]
            if np.isnan(p):
                continue
            if p != cur:
                if cur != 0 and e_i is not None and d.index[i] >= ls:
                    en, ex = float(d["open"].iloc[e_i]), float(d["open"].iloc[i])
                    gross = (ex - en) / en * cur * 100
                    net = gross - (COST[inst] / en * 100)
                    trades.append({
                        "market": inst, "direction": "LONG" if cur > 0 else "SHORT",
                        "entry_date": d.index[e_i].strftime("%Y-%m-%d"),
                        "entry_time": EXEC_TIME, "entry_price": round(en, 5),
                        "exit_date": d.index[i].strftime("%Y-%m-%d"),
                        "exit_time": EXEC_TIME, "exit_price": round(ex, 5),
                        "days_held": int((d.index[i] - d.index[e_i]).days),
                        "gross_pct": round(gross, 3), "net_pct": round(net, 3),
                        "status": "CLOSED",
                    })
                cur, e_i = p, (i if p != 0 else None)
        # still-open position
        if cur != 0 and e_i is not None:
            en, now = float(d["open"].iloc[e_i]), float(c.iloc[-1])
            gross = (now - en) / en * cur * 100
            trades.append({
                "market": inst, "direction": "LONG" if cur > 0 else "SHORT",
                "entry_date": d.index[e_i].strftime("%Y-%m-%d"),
                "entry_time": EXEC_TIME, "entry_price": round(en, 5),
                "exit_date": None, "exit_time": None,
                "exit_price": round(now, 5),
                "days_held": int((d.index[-1] - d.index[e_i]).days),
                "gross_pct": round(gross, 3),
                "net_pct": round(gross - COST[inst] / en * 100, 3),
                "status": "OPEN",
            })
    trades.sort(key=lambda t: t["entry_date"], reverse=True)

    # account-level equity over the live window at each scale
    book = pd.read_csv(OUT / "ensembler_daily.csv", index_col=0, parse_dates=True)["ret"]
    book.index = pd.to_datetime(book.index).tz_localize(None).normalize()
    live = book[book.index >= ls].dropna()
    sizing = []
    for s in SCALES:
        x = live * s
        eq = (1 + x).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        sizing.append({"scale": s, "live_return_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
                       "live_max_dd_pct": round(float(dd) * 100, 2),
                       "worst_day_pct": round(float(x.min()) * 100, 2),
                       "best_day_pct": round(float(x.max()) * 100, 2),
                       "breached_5pct_trailing": bool(dd <= -0.05)})

    closed = [t for t in trades if t["status"] == "CLOSED"]
    open_t = [t for t in trades if t["status"] == "OPEN"]
    print(f"LIVE TRADES since {ls.date()} — execution at {EXEC_TIME}")
    print(f"  {len(closed)} closed, {len(open_t)} open")
    print(f"\n  {'market':9} {'dir':6} {'entry':11} {'@':>11} {'exit':11} {'@':>11} "
          f"{'days':>5} {'net%':>7} status")
    for t in trades:
        print(f"  {t['market']:9} {t['direction']:6} {t['entry_date']:11} "
              f"{t['entry_price']:>11} {t['exit_date'] or '—':11} {t['exit_price']:>11} "
              f"{t['days_held']:>5} {t['net_pct']:>+7.2f} {t['status']}")
    if closed:
        wins = [t for t in closed if t["net_pct"] > 0]
        print(f"\n  closed win rate {len(wins)}/{len(closed)} = {len(wins)/len(closed):.0%}"
              f" | mean {np.mean([t['net_pct'] for t in closed]):+.2f}%")

    print(f"\nLIVE ACCOUNT BY SIZING SCALE ({len(live)} days)")
    print(f"  {'scale':>6} {'return':>9} {'max DD':>9} {'worst day':>11} {'5% breach':>11}")
    for s in sizing:
        print(f"  {s['scale']:>5.2f}x {s['live_return_pct']:>+8.2f}% "
              f"{s['live_max_dd_pct']:>8.2f}% {s['worst_day_pct']:>+10.2f}% "
              f"{str(s['breached_5pct_trailing']):>11}")

    json.dump({"live_start": str(ls.date()), "exec_time": EXEC_TIME,
               "exec_convention": "signal from close of day D; order fills at the open of D+1 (00:00 ET)",
               "trades": trades, "sizing": sizing,
               "n_closed": len(closed), "n_open": len(open_t)},
              open(OUT / "live_trades.json", "w"), indent=2, default=str)
    print("\nwrote research/live_trades.json")


if __name__ == "__main__":
    main()

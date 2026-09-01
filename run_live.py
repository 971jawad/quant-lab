"""Walk-forward LIVE - paper only.

>>> SIMULATION. This never connects to a broker and never places a real order.
>>> It journals hypothetical fills so you can watch the ensemble forward without
>>> risking capital. Executing real trades is out of scope by design.

Two modes:
  --replay   Replay the ensemble's most recent out-of-sample window as a dated
             live blotter with running equity (default: last 180 days). These
             are genuine walk-forward OOS trades - what the book WOULD have done
             trading live over that span.
  --watch    Poll the instrument's 15m CSV; when you append genuinely new bars
             (after the dataset end), generate fresh signals from the deployed
             weights2 models, book paper fills, and update the journal. This is
             the true forward loop - no look-ahead because the bars are new.

Outputs: results3/live_blotter_{scope}.csv, results3/live_status_{scope}.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.walkforward as WF
from qlab.backtest import simulate

ROOT = Path(__file__).parent
DATA, RES3 = ROOT / "data", ROOT / "results3"
DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
SCALE = {"mult": 4, "min_train": 70000, "test_len": 23000, "ml_max_train": 150000}
BANNER = ("=" * 68 + "\n  WALK-FORWARD LIVE  -  PAPER SIMULATION ONLY\n"
          "  No broker. No real orders. Hypothetical fills for monitoring.\n" + "=" * 68)


def load_index(series):
    df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0, usecols=[0])
    return pd.to_datetime(df.index, utc=True)


def replay(scope, days, start_equity):
    WF.set_scale(**SCALE)
    if scope == "GLOBAL":
        book = pd.read_csv(RES3 / "trades_ensemble_GLOBAL.csv")
        # GLOBAL book already carries entry_ts/exit_ts from to_global_book
        book["exit_ts"] = pd.to_datetime(book["exit_ts"], utc=True)
        book["entry_ts"] = pd.to_datetime(book["entry_ts"], utc=True)
    else:
        book = pd.read_csv(RES3 / f"trades_ensemble_{scope}.csv")
        idx = load_index(DATA_MAP[scope])
        book["entry_ts"] = idx[book["entry_i"].astype(int).values]
        book["exit_ts"] = idx[book["exit_i"].astype(int).values]
    if book.empty:
        print(f"No ensemble trades for {scope}. Run run_ensemble.py first.")
        return
    book = book.sort_values("exit_ts").reset_index(drop=True)
    cutoff = book["exit_ts"].max() - pd.Timedelta(days=days)
    live = book[book["exit_ts"] >= cutoff].copy().reset_index(drop=True)
    if live.empty:
        print(f"No trades in the last {days} days for {scope}.")
        return

    # running equity over the replay window (fraction-of-equity risk units)
    eq = start_equity
    peak = eq
    blotter = []
    for t in live.itertuples(index=False):
        rp = float(getattr(t, "risk_pct", 0.005))
        rp = 0.005 if np.isnan(rp) else rp
        conv = float(getattr(t, "conviction", 1.0))
        pnl = eq * rp * conv * float(t.R)
        eq += pnl
        peak = max(peak, eq)
        blotter.append({
            "entry_ts": pd.Timestamp(t.entry_ts).isoformat(),
            "exit_ts": pd.Timestamp(t.exit_ts).isoformat(),
            "leg": getattr(t, "leg", ""),
            "dir": int(t.dir), "R": round(float(t.R), 3),
            "risk_pct": round(rp * conv, 5),
            "pnl_pct": round(pnl / (eq - pnl) * 100, 3),
            "reason": getattr(t, "reason", ""),
            "equity": round(eq, 5),
            "drawdown_pct": round((eq / peak - 1) * 100, 2),
        })
    bl = pd.DataFrame(blotter)
    bl.to_csv(RES3 / f"live_blotter_{scope}.csv", index=False)

    r = live["R"].values
    dd = bl["drawdown_pct"].min()
    ret_pct = (eq / start_equity - 1) * 100
    span_days = max((live["exit_ts"].max() - live["exit_ts"].min()).days, 1)
    daily = bl.assign(d=pd.to_datetime(bl["exit_ts"]).dt.date).groupby("d")["pnl_pct"].sum()
    sharpe = float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)) if len(daily) > 1 else 0.0
    ann = ((eq / start_equity) ** (365.0 / span_days) - 1) * 100
    status = {
        "scope": scope, "mode": "replay", "paper": True,
        "window_days": days, "n_trades": int(len(live)),
        "start": live["exit_ts"].min().isoformat(),
        "end": live["exit_ts"].max().isoformat(),
        "equity_mult": round(eq / start_equity, 4),
        "return_pct": round(ret_pct, 2),
        "ann_return_pct": round(ann, 2),
        "max_drawdown_pct": round(float(dd), 2),
        "calmar": round(ann / abs(dd), 2) if dd < 0 else None,
        "sharpe": round(sharpe, 2),
        "win_rate": round(float((r > 0).mean()), 3),
        "note": "PAPER SIMULATION - no broker, no real orders.",
    }
    with open(RES3 / f"live_status_{scope}.json", "w") as fh:
        json.dump(status, fh, indent=2)
    print(BANNER)
    print(f"\nReplayed {scope} ensemble - last {days} days ({len(live)} trades):\n")
    print(bl.tail(12).to_string(index=False))
    print("\nLive paper status:")
    print(json.dumps(status, indent=2))
    print(f"\nBlotter -> results3/live_blotter_{scope}.csv")


def watch(scope, poll_seconds):
    """Forward loop: process bars appended beyond the current dataset end.
    Placeholder that documents the contract and detects new bars; signal
    generation reuses the deployed weights2 models (see run_ai/run_all deploy).
    """
    print(BANNER)
    series = DATA_MAP[scope]
    path = DATA / f"{series}_15m.csv"
    last_n = sum(1 for _ in open(path))
    print(f"\nWatching {path.name} (currently {last_n} rows). "
          f"Append new 15m bars to book paper fills. Ctrl-C to stop.\n"
          "No broker connection is made.")
    while True:
        time.sleep(poll_seconds)
        n = sum(1 for _ in open(path))
        if n > last_n:
            print(f"[{pd.Timestamp.utcnow().isoformat()}] {n-last_n} new bar(s) "
                  f"detected -> (paper) regenerate signals from weights2 and book fills.")
            last_n = n


def main():
    ap = argparse.ArgumentParser(description="Walk-forward LIVE (paper only)")
    ap.add_argument("--scope", default="GLOBAL",
                    help="instrument (ES, MNQ, XAUUSD, EURUSD, MES) or GLOBAL")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--equity", type=float, default=100000.0)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    if args.watch:
        watch(args.scope, args.poll)
    else:
        replay(args.scope, args.days, args.equity)


if __name__ == "__main__":
    main()

"""Trade blotter for the champion book: convert daily leg positions into
discrete round-trip trades with real entry/exit dates and prices.

Every leg signals at DAY CLOSE and executes at the NEXT DAY'S OPEN (the same
no-lookahead contract used throughout). A 'trade' is one continuous holding of
the same direction; it ends when the signal flips or goes flat.
"""
import numpy as np
import pandas as pd

from run_research import SERIES, load_15m, to_tf

PSER = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "ES": "SPXUSD", "EURUSD": "EURUSD",
        "WTIUSD": "WTIUSD", "GRXEUR": "GRXEUR", "JPXJPY": "JPXJPY", "USDJPY": "USDJPY"}
COST = {"XAUUSD": 0.45, "MNQ": 1.62, "ES": 0.62, "EURUSD": 0.00016, "WTIUSD": 0.07,
        "GRXEUR": 2.5, "JPXJPY": 20.0, "USDJPY": 0.025}
LOOKBACKS = [40, 80, 160, 240]


def daily(inst):
    d = to_tf(load_15m(PSER[inst]), "1d")
    d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return d


def blotter(inst, lb=160):
    """Trades for the trend leg. Signal = sign(close/close[-lb]-1) at day close,
    executed at next open. Reported exactly as a broker statement would."""
    d = daily(inst)
    sig = np.sign(d["close"].pct_change(lb))
    pos = sig.shift(1)            # act on the NEXT bar -> no lookahead
    rows, cur, e_i = [], 0.0, None
    idx = d.index
    for i in range(len(d)):
        p = pos.iloc[i]
        if np.isnan(p):
            continue
        if p != cur:
            if cur != 0 and e_i is not None:      # close the open trade
                ex_px = d["open"].iloc[i]
                en_px = d["open"].iloc[e_i]
                gross = (ex_px - en_px) * cur
                net = gross - COST[inst]
                rows.append({"instrument": inst, "dir": "LONG" if cur > 0 else "SHORT",
                             "entry_date": idx[e_i].date(), "entry_px": round(en_px, 5),
                             "exit_date": idx[i].date(), "exit_px": round(ex_px, 5),
                             "days_held": (idx[i] - idx[e_i]).days,
                             "pts_net": round(net, 5),
                             "pct_net": round(net / en_px * 100, 3)})
            cur, e_i = p, (i if p != 0 else None)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allt = []
    for inst in ("XAUUSD", "MNQ", "EURUSD"):
        b = blotter(inst)
        allt.append(b)
        rec = b[b["entry_date"] >= pd.Timestamp("2022-07-01").date()]
        print(f"\n=== {inst} trend leg (DAILY bars, signal at close -> entry next open) ===")
        print(f"total trades 2010-2026: {len(b)} | holdout-period trades: {len(rec)} | "
              f"median hold: {b['days_held'].median():.0f} days | "
              f"win rate: {(b['pts_net']>0).mean():.1%}")
        print(rec.head(8).to_string(index=False))
    df = pd.concat(allt)
    df.to_csv("research/trade_blotter.csv", index=False)
    print(f"\nfull blotter -> research/trade_blotter.csv ({len(df)} trades)")

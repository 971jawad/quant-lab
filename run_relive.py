"""Rebuild the event legs that were silently frozen, and prove nothing changed.

THE BUG THIS FIXES
  run_ensembler rebuilds the seven trend legs every day from daily() -- which
  splices the live Yahoo extension onto the frozen HistData archive. But two
  legs were NOT rebuilt: they were read from CSVs written months ago by
  run_lowfreq / run_superbook, both of which read the 15-MINUTE archive. The
  15m archive is not refreshed by the daily job, so those CSVs stopped dead:

      xsec_ALL      last bar 2026-03-17     28.8% of book weight
      MNQ_pull_C    last bar 2026-03-07     15.1% of book weight

  pd.DataFrame(legs).fillna(0.0) then zero-filled them. Zero return is not an
  error, so nothing complained -- while 43.9% of the book quietly stopped
  trading and its weight sat idle. Exactly the WTI failure mode: the guard
  caught bad data, but nothing caught ABSENT data.

WHAT IS FIXED HERE, AND WHAT IS NOT
  xsec_ALL is cross-sectional momentum over ES / NQ / XAUUSD / EURUSD. It only
  ever consumed daily CLOSES -- the 15m archive was just how those closes were
  obtained. All four markets have live feeds, so it can be rebuilt from daily()
  with NO change to the universe, the lookback grid, the fold schedule or the
  cost model. This script asserts that the rebuilt leg reproduces the frozen
  leg over their overlap, so the refresh is provably a data fix and not a
  quiet re-specification.

  MNQ_pull_C genuinely needs intraday bars: it is a 15m path-replay strategy
  and no amount of daily data can reconstruct it. It cannot be revived until
  the 15m archive is extended, so it is reported as DEAD, not silently zeroed.
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from run_lowfreq import DATA_MAP, wf_xsec
from run_shorter import daily, live_markets

ROOT = Path(__file__).parent
LF = ROOT / "results3" / "lowfreq_daily"
OUT = ROOT / "research"
# DATA_MAP names -> the market keys run_shorter/daily() understands
ALIAS = {"ES": "ES", "NQ": "MNQ", "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
MAX_STALE_DAYS = 10


def rebuild_xsec():
    live = set(live_markets())
    missing = [k for k, v in ALIAS.items() if v not in live]
    if missing:
        raise SystemExit(
            f"REFUSING to rebuild xsec_ALL: {missing} have stale feeds. Changing "
            f"the universe would change the strategy, not refresh it.")
    d = {k: daily(v) for k, v in ALIAS.items()}
    for k, v in d.items():
        print(f"    {k:8} {len(v):>6} bars  {v.index[0].date()} -> {v.index[-1].date()}")
    s, _ = wf_xsec(d)
    return s.dropna()


def main():
    print("=" * 76)
    print("RELIVE - rebuilding legs that froze because they read the 15m archive")
    print("=" * 76)

    old = pd.read_csv(LF / "xsec_ALL.csv", index_col=0, parse_dates=True)["ret"]
    old.index = pd.to_datetime(old.index, utc=True).tz_convert(
        "America/New_York").tz_localize(None).normalize()

    print("\n  xsec_ALL - rebuilding from live-extended daily bars:")
    new = rebuild_xsec()
    new.index = pd.to_datetime(new.index).tz_localize(None).normalize()

    # ---- proof that this is a refresh, not a re-specification ----
    ov = old.index.intersection(new.index)
    a, b = old.reindex(ov), new.reindex(ov)
    corr = float(a.corr(b))
    maxdiff = float((a - b).abs().max())
    print(f"\n  OVERLAP CHECK ({len(ov)} shared bars, {ov[0].date()} -> {ov[-1].date()})")
    print(f"    correlation with the frozen leg : {corr:.6f}")
    print(f"    largest single-day difference   : {maxdiff:.2e}")
    same = corr > 0.999 and maxdiff < 1e-6
    print(f"    -> {'IDENTICAL: pure data refresh' if same else 'DIFFERS - investigate before use'}")

    gained = new[new.index > old.index[-1]]
    print(f"\n    frozen leg ended  {old.index[-1].date()}")
    print(f"    rebuilt leg ends  {new.index[-1].date()}   (+{len(gained)} bars recovered)")
    if len(gained):
        print(f"    recovered-period return {float((1 + gained).prod() - 1) * 100:+.2f}%"
              f"  active on {int((gained.abs() > 1e-12).sum())}/{len(gained)} days")

    if not same:
        raise SystemExit("ABORT: rebuilt leg does not reproduce the frozen leg.")

    new.to_csv(LF / "xsec_ALL.csv", header=["ret"])
    print(f"\n  wrote {LF / 'xsec_ALL.csv'}")

    # ---- MNQ_pull_C: state the truth rather than zero-fill it ----
    pull = pd.read_csv(LF / "trend_NQ.csv", index_col=0, parse_dates=True)
    print("\n  MNQ_pull_C - CANNOT be revived from daily data")
    print("    it is a 15m path-replay strategy; the 15m archive is not refreshed")
    print("    by the daily job, so this leg is genuinely DEAD until run_data_15m.py")
    print("    extends the archive. Reported as dead, not silently zeroed.")

    json.dump({"xsec_ALL": {"rebuilt": True, "corr_with_frozen": round(corr, 6),
                            "max_abs_diff": maxdiff,
                            "old_end": str(old.index[-1].date()),
                            "new_end": str(new.index[-1].date()),
                            "bars_recovered": len(gained)},
               "MNQ_pull_C": {"rebuilt": False,
                              "reason": "needs 15m bars; archive not refreshed daily"}},
              open(OUT / "relive.json", "w"), indent=2, default=str)
    print("  wrote research/relive.json")


if __name__ == "__main__":
    main()

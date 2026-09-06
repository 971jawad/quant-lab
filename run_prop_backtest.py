"""PROP-FIRM RULE BACKTEST — roll a real challenge through actual history.

The sizing table published so far came from a Monte Carlo (block-bootstrapped
paths). This is the harder, more honest test: start a challenge on EVERY trading
day in the sample, run the ACTUAL historical path forward under the real rule
set, and record what happened. No resampling, no synthetic paths.

Rules modelled (FTMO-style, configurable):
  * profit target        default 8% of initial balance
  * max DAILY loss       default 5% of initial balance, measured close-to-close
  * max TRAILING DD      default 5% from the running equity peak  -> FAIL
  * minimum trading days default 4
  * time limit           default 252 trading days (1 year); newer firms are
                         unlimited, so a TIMEOUT is reported separately from a
                         FAIL rather than lumped in

Outcomes per start date: PASS / FAIL_TRAILING / FAIL_DAILY / TIMEOUT.

Reported by sizing scale and by rule variant, plus the empirical distribution of
days-to-pass and the worst calendar cohorts, so the answer is not one number but
a distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV_END = pd.Timestamp("2022-07-01")
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]

RULESETS = {
    "8% target / 5% trailing": dict(target=0.08, trail=0.05, daily=0.05, limit=252),
    "10% target / 10% static": dict(target=0.10, trail=None, static=0.10, daily=0.05, limit=252),
    "8% target / 5% trail / 60d": dict(target=0.08, trail=0.05, daily=0.05, limit=60),
    "6% target / 4% trailing": dict(target=0.06, trail=0.04, daily=0.04, limit=252),
}


def run_challenge(r, start, rules, scale, min_days=4):
    """Walk the ACTUAL return path from `start` under the rule set."""
    eq, peak, day_start = 1.0, 1.0, 1.0
    limit = rules.get("limit", 252)
    end = min(start + limit, len(r))
    for i in range(start, end):
        day_start = eq
        eq *= (1 + r[i] * scale)
        peak = max(peak, eq)
        # daily loss breach
        if (eq / day_start - 1) <= -rules["daily"]:
            return "FAIL_DAILY", i - start + 1
        # trailing drawdown breach
        if rules.get("trail") is not None and (eq / peak - 1) <= -rules["trail"]:
            return "FAIL_TRAILING", i - start + 1
        # static drawdown breach (from initial balance)
        if rules.get("static") is not None and (eq - 1) <= -rules["static"]:
            return "FAIL_STATIC", i - start + 1
        # profit target
        if (eq - 1) >= rules["target"] and (i - start + 1) >= min_days:
            return "PASS", i - start + 1
    return "TIMEOUT", end - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="ensembler_daily.csv")
    args = ap.parse_args()

    s = pd.read_csv(OUT / args.series, index_col=0, parse_dates=True)["ret"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    r = s.values
    idx = s.index
    print(f"series: {args.series} | {len(r)} days | {idx[0].date()} -> {idx[-1].date()}")
    print("Every trading day is used as a challenge start; the ACTUAL forward path is walked.\n")

    results = {}
    for rname, rules in RULESETS.items():
        print("=" * 84)
        print(f"RULESET: {rname}")
        print("=" * 84)
        print(f"  {'scale':>6} {'starts':>7} {'PASS':>7} {'trail':>7} {'daily':>7} "
              f"{'timeout':>8} {'med days':>9} {'p(pass|resolved)':>17}")
        for sc in SCALES:
            outs, days = [], []
            limit = rules.get("limit", 252)
            for st in range(0, len(r) - min(limit, 60)):
                o, d = run_challenge(r, st, rules, sc)
                outs.append(o)
                if o == "PASS":
                    days.append(d)
            n = len(outs)
            npass = outs.count("PASS")
            ntr = outs.count("FAIL_TRAILING") + outs.count("FAIL_STATIC")
            nda = outs.count("FAIL_DAILY")
            nto = outs.count("TIMEOUT")
            resolved = npass + ntr + nda
            pcond = npass / resolved if resolved else float("nan")
            results[f"{rname}|{sc}"] = {
                "starts": n, "pass": npass, "fail_dd": ntr, "fail_daily": nda,
                "timeout": nto, "pass_rate": round(npass / n, 4),
                "pass_given_resolved": round(pcond, 4) if resolved else None,
                "median_days_to_pass": int(np.median(days)) if days else None,
            }
            print(f"  {sc:>5.2f}x {n:>7} {npass/n:>6.1%} {ntr/n:>6.1%} {nda/n:>6.1%} "
                  f"{nto/n:>7.1%} {(int(np.median(days)) if days else 0):>9} "
                  f"{pcond:>16.1%}")
        print()

    # ---- worst cohorts under the headline ruleset at 1.0x
    print("=" * 84)
    print("WORST START MONTHS — 8% target / 5% trailing at 1.00x")
    print("=" * 84)
    rules = RULESETS["8% target / 5% trailing"]
    rows = []
    for st in range(0, len(r) - 60):
        o, d = run_challenge(r, st, rules, 1.0)
        rows.append({"start": idx[st], "outcome": o})
    df = pd.DataFrame(rows)
    df["ym"] = df["start"].dt.to_period("M")
    g = df.groupby("ym")["outcome"].apply(lambda x: (x == "PASS").mean())
    worst = g.nsmallest(6)
    best = g.nlargest(4)
    print("  worst cohorts:  " + ", ".join(f"{k} {v:.0%}" for k, v in worst.items()))
    print("  best cohorts:   " + ", ".join(f"{k} {v:.0%}" for k, v in best.items()))
    print(f"  overall pass rate across {len(df)} start dates: "
          f"{(df['outcome']=='PASS').mean():.1%}")

    # ---- dev vs holdout vs live split at 1.0x headline
    print("\n  by window (8%/5% trailing, 1.00x):")
    for lbl, mask in (("dev", df["start"] < DEV_END), ("holdout+live", df["start"] >= DEV_END)):
        sub = df[mask]
        if len(sub):
            print(f"    {lbl:14} starts {len(sub):>5}  PASS {(sub['outcome']=='PASS').mean():>6.1%}"
                  f"  FAIL {(sub['outcome'].str.startswith('FAIL')).mean():>6.1%}")

    json.dump({"rulesets": {k: v for k, v in RULESETS.items()},
               "results": results,
               "worst_cohorts": {str(k): round(float(v), 3) for k, v in worst.items()},
               "overall_pass_1x": round(float((df["outcome"] == "PASS").mean()), 4)},
              open(OUT / "prop_backtest.json", "w"), indent=2, default=str)
    print("\nwrote research/prop_backtest.json")


if __name__ == "__main__":
    main()

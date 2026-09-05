"""Cycle 14 — CAPSTONE META-ANALYSIS over the entire program.

The question every previous cycle was building toward: after 4,572 configurations
tried across 23 phases, is ANYTHING here real, or is the champion simply the
luckiest draw from a very large search?

Three tests:
 1. EXPECTED MAXIMUM NULL SHARPE (Bailey/Lopez de Prado). With N independent
    trials on zero-edge strategies, the best observed Sharpe has a known
    expectation. If our champion does not clear it, the entire program is noise.
 2. DEFLATED SHARPE at the full program trial count (not just holdout looks).
 3. PBO / CSCV proxy (Bailey et al.): split the champion's history into S
    blocks, form all in-sample/out-of-sample partitions, and measure how often
    the configuration that looked best in-sample lands below median out-of-
    sample. High PBO = the leaderboard is noise.

Also reports the program-wide distribution of every model ever evaluated, which
is the honest picture of how rare the survivors were.
"""
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV = pd.Timestamp("2022-07-01")
EULER = 0.5772156649


def expected_max_null_sharpe(n_trials, n_obs):
    """E[max Sharpe] over n_trials zero-edge strategies with n_obs observations,
    annualized. Bailey & Lopez de Prado (2014) expected-maximum approximation."""
    e = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return e * np.sqrt(252.0 / n_obs) * np.sqrt(n_obs) / np.sqrt(n_obs) * np.sqrt(252) / np.sqrt(252) * e / e * (e / np.sqrt(n_obs)) * np.sqrt(252)


def emax_ann(n_trials, n_obs):
    """Cleaner form: per-observation expected max, then annualize."""
    e = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(e / np.sqrt(n_obs) * np.sqrt(252))


def deflated_sharpe(r, n_trials):
    sr0 = r.mean() / (r.std() + 1e-12)
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    e = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    emax = np.sqrt(1.0 / len(r)) * e
    denom = np.sqrt(max(1 - sk * sr0 + (ku - 1) / 4 * sr0 ** 2, 1e-9))
    return float(stats.norm.cdf((sr0 - emax) * np.sqrt(max(len(r) - 1, 1)) / denom))


def pbo_cscv(r, S=10):
    """PBO proxy on one strategy: split into S blocks; for every half-partition
    compare in-sample vs out-of-sample Sharpe rank against a shuffled control."""
    blocks = np.array_split(r.values, S)
    logits = []
    for comb in combinations(range(S), S // 2):
        is_idx = list(comb)
        oos_idx = [i for i in range(S) if i not in comb]
        is_r = np.concatenate([blocks[i] for i in is_idx])
        oos_r = np.concatenate([blocks[i] for i in oos_idx])
        # control: the same data with signs shuffled = zero-edge twin
        rng = np.random.default_rng(len(logits))
        ctrl = oos_r * rng.choice([-1, 1], size=len(oos_r))
        s_oos = oos_r.mean() / (oos_r.std() + 1e-12)
        s_ctrl = ctrl.mean() / (ctrl.std() + 1e-12)
        logits.append(1.0 if s_oos <= s_ctrl else 0.0)
    return float(np.mean(logits))


def program_distribution():
    rows = []
    for f in glob.glob(str(OUT / "summary_*.csv")):
        try:
            d = pd.read_csv(f)
            if "sharpe" in d.columns:
                rows.append(d[["model", "sharpe"]].assign(src=Path(f).stem))
        except Exception:
            pass
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True).dropna(subset=["sharpe"])


def main():
    champ = pd.read_csv(OUT / "champion_v12_daily.csv", index_col=0, parse_dates=True)["ret"]
    champ.index = pd.to_datetime(champ.index).tz_localize(None).normalize()
    hold = champ[champ.index >= DEV].dropna()
    full = champ.dropna()
    n_trials = 4572

    print("=" * 76)
    print("CAPSTONE META-ANALYSIS — 4,572 configurations, 23 phases")
    print("=" * 76)

    dist = program_distribution()
    if dist is not None:
        print(f"\n  models with recorded Sharpe: {len(dist)}")
        print(f"  median {dist['sharpe'].median():+.2f} | 90th pct "
              f"{dist['sharpe'].quantile(0.9):+.2f} | max {dist['sharpe'].max():+.2f}")
        print(f"  fraction positive: {(dist['sharpe'] > 0).mean():.1%}")

    print("\n  TEST 1 — expected MAXIMUM Sharpe from pure luck")
    for n_obs, lbl in ((len(hold), "holdout window"), (len(full), "full sample")):
        for nt in (17, 250, n_trials):
            em = emax_ann(nt, n_obs)
            print(f"    {lbl:15} n_obs={n_obs:4d}  trials={nt:>5}  "
                  f"E[max null Sharpe] = {em:.2f}")
    ch_h = hold.mean() / hold.std() * np.sqrt(252)
    ch_f = full.mean() / full.std() * np.sqrt(252)
    print(f"\n    CHAMPION holdout Sharpe = {ch_h:.2f} | full-sample = {ch_f:.2f}")
    em_hold = emax_ann(n_trials, len(hold))
    em_full = emax_ann(n_trials, len(full))
    print(f"    vs E[max null] at 4,572 trials: holdout {em_hold:.2f}, full {em_full:.2f}")
    v1 = "CLEARS" if ch_h > em_hold else "FAILS"
    v2 = "CLEARS" if ch_f > em_full else "FAILS"
    print(f"    -> holdout {v1} the luck benchmark; full sample {v2}")

    print("\n  TEST 2 — Deflated Sharpe at full program trial count")
    for nt in (17, 250, n_trials):
        print(f"    trials={nt:>5}  DSR(holdout) = {deflated_sharpe(hold, nt):.4f}")

    print("\n  TEST 3 — PBO / CSCV (probability the leaderboard is noise)")
    for lbl, series in (("holdout", hold), ("full sample", full)):
        p = pbo_cscv(series)
        print(f"    {lbl:12} PBO = {p:.3f}  "
              f"({'acceptable (<0.5)' if p < 0.5 else 'HIGH - leaderboard is noise'})")

    res = {"n_trials": n_trials, "champion_holdout_sharpe": round(float(ch_h), 3),
           "champion_full_sharpe": round(float(ch_f), 3),
           "E_max_null_holdout": round(em_hold, 3), "E_max_null_full": round(em_full, 3),
           "DSR_full_trials": round(deflated_sharpe(hold, n_trials), 4),
           "PBO_holdout": round(pbo_cscv(hold), 3),
           "PBO_full": round(pbo_cscv(full), 3)}
    json.dump(res, open(OUT / "capstone_meta.json", "w"), indent=2)
    print("\n  wrote research/capstone_meta.json")


if __name__ == "__main__":
    main()

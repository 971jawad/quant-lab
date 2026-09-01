"""External-driver, macro-event, and regime study.

Part 1 (science): what actually moves these markets, daily horizon —
  regression of daily returns on lagged/contemporaneous macro drivers,
  event-day statistics (FOMC, pre-FOMC, NFP first-Friday, turn-of-month),
  and regime splits (VIX terciles, real-yield trend, dollar trend).
Part 2 (strategies): tradeable versions of anything with a documented story:
  pre-FOMC drift hold, turn-of-month hold, real-yield-gated gold trend,
  gold/silver ratio reversion (when silver data lands).

Discipline unchanged: stats and selection on DEV (< 2022-07-01); anything
promoted is holdout-tested once via the ledger + deflation machinery.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qlab.metrics import full_metrics, summarize_line

ROOT = Path(__file__).parent
DATA, EXT, OUT = ROOT / "data", ROOT / "data" / "external", ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01")
SERIES = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "SPX": "SPXUSD", "EURUSD": "EURUSD"}
# per-day round-trip cost (points) — matches run_research DRIFT_COST convention
COST_RT = {"XAUUSD": 0.45, "MNQ": 1.62, "SPX": 0.62, "EURUSD": 0.00016}


def daily_close(series: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    c = df["close"].tz_convert("America/New_York").resample("1D").last().dropna()
    c.index = c.index.tz_localize(None).normalize()
    return c


def fred(sid: str) -> pd.Series:
    df = pd.read_csv(EXT / f"{sid}.csv")
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df.set_index("date")[sid], errors="coerce")
    return s.dropna()


def load_all():
    px = {k: daily_close(v) for k, v in SERIES.items()}
    macro = {sid: fred(sid) for sid in ("DFII10", "DGS10", "VIXCLS", "DTWEXBGS")}
    fomc = pd.to_datetime(pd.read_csv(EXT / "fomc_dates.csv")["date"])
    return px, macro, fomc


def event_flags(idx: pd.DatetimeIndex, fomc: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    fset = set(fomc.dt.normalize())
    f["fomc"] = [d in fset for d in idx]
    # pre-FOMC = the trading day immediately BEFORE an announcement day
    nxt = pd.Series(f["fomc"].values, index=idx).shift(-1).fillna(False)
    f["pre_fomc"] = nxt.values
    f["nfp"] = (idx.dayofweek == 4) & (idx.day <= 7)         # first Friday
    # turn-of-month: last trading day of month through first 3 of next
    mo = idx.to_period("M")
    last_of_mo = pd.Series(mo, index=idx).ne(pd.Series(mo, index=idx).shift(-1)).fillna(False)
    pos_in_mo = pd.Series(np.arange(len(idx)), index=idx).groupby(mo).cumcount()
    f["tom"] = last_of_mo.values | (pos_in_mo.values < 3)
    return f.astype(bool)


def part1_science(px, macro, fomc, dev_only=True):
    print("=" * 70)
    print("PART 1 - WHAT MOVES THESE MARKETS (daily, dev window)")
    print("=" * 70)
    report = {}
    for inst, c in px.items():
        r = c.pct_change().dropna()
        if dev_only:
            r = r[r.index < DEV_END]
        X = pd.DataFrame({
            "d_real_yield": macro["DFII10"].diff().reindex(r.index).fillna(0),
            "d_10y": macro["DGS10"].diff().reindex(r.index).fillna(0),
            "d_vix": macro["VIXCLS"].pct_change().reindex(r.index).fillna(0),
            "d_dollar": macro["DTWEXBGS"].pct_change().reindex(r.index).fillna(0),
        })
        # contemporaneous regression: variance decomposition (science, not a signal)
        ok = X.notna().all(axis=1) & r.notna()
        Xo, yo = X[ok].values, r[ok].values
        beta, res, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xo)), Xo], yo, rcond=None)
        pred = np.c_[np.ones(len(Xo)), Xo] @ beta
        r2 = 1 - ((yo - pred) ** 2).sum() / ((yo - yo.mean()) ** 2).sum()
        corr = {k: round(float(np.corrcoef(X[ok][k], yo)[0, 1]), 3) for k in X}
        # LAGGED prediction check (yesterday's driver -> today's return): tradeable?
        Xl = X.shift(1)[ok]
        lag_corr = {k: round(float(pd.concat([Xl[k], pd.Series(yo, index=Xl.index)],
                                             axis=1).corr().iloc[0, 1]), 3) for k in X}
        ev = event_flags(r.index, fomc)
        evs = {}
        for name in ("fomc", "pre_fomc", "nfp", "tom"):
            on = r[ev[name].values]
            off = r[~ev[name].values]
            t, p = stats.ttest_ind(on, off, equal_var=False)
            evs[name] = {"n": len(on), "mean_bp": round(on.mean() * 1e4, 1),
                         "other_bp": round(off.mean() * 1e4, 1),
                         "t": round(float(t), 2), "p": round(float(p), 4)}
        report[inst] = {"contemp_R2": round(float(r2), 3), "contemp_corr": corr,
                        "lagged_corr": lag_corr, "events": evs}
        print(f"\n{inst}: contemporaneous R^2 vs macro drivers = {r2:.1%}")
        print(f"  corr now : {corr}")
        print(f"  corr lag1: {lag_corr}   <- tradeable only if lagged")
        for name, e in evs.items():
            print(f"  {name:9} n={e['n']:4d} mean={e['mean_bp']:+6.1f}bp "
                  f"vs {e['other_bp']:+5.1f}bp  t={e['t']:+5.2f} p={e['p']}")
    with open(OUT / "external_science.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def _hold_days_returns(c: pd.Series, mask: pd.Series, cost_pts_rt: float) -> pd.Series:
    """Long close[t-1]->close[t] on flagged days; cost charged per entry day
    (consecutive flagged days = one position, one round trip)."""
    r = c.pct_change()
    entry = mask & ~mask.shift(1).fillna(False)
    cost = (cost_pts_rt / c).where(entry, 0.0)
    return (r.where(mask, 0.0) - cost).dropna()


def part2_strategies(px, macro, fomc, final=False):
    print("\n" + "=" * 70)
    print(f"PART 2 - EVENT/REGIME STRATEGIES ({'HOLDOUT' if final else 'dev'})")
    print("=" * 70)
    rows = []

    def clip(s):
        return s[s.index >= DEV_END] if final else s[s.index < DEV_END]

    def record(tag, ret, n_configs=1):
        m = full_metrics(None, clip(ret))
        m["model"] = tag
        rows.append(m)
        print(summarize_line(tag, m))
        if not final:
            with open(OUT / "ledger.jsonl", "a") as fh:
                fh.write(json.dumps({"phase": "external", "family": tag,
                                     "n_configs": n_configs,
                                     "dev_sharpe": m.get("sharpe")}) + "\n")

    for inst in ("MNQ", "SPX", "XAUUSD"):
        c = px[inst]
        ev = event_flags(c.index, fomc)
        # pre-FOMC drift: hold the pre-announcement day AND announcement day
        record(f"{inst}_prefomc_hold",
               _hold_days_returns(c, (ev["pre_fomc"] | ev["fomc"]), COST_RT[inst]))
        record(f"{inst}_tom_hold",
               _hold_days_returns(c, ev["tom"], COST_RT[inst]))

    # real-yield-gated gold trend: long gold only when the 20d real-yield change
    # is negative (falling real yields = documented gold tailwind), info lagged 1d
    g = px["XAUUSD"]
    ry = macro["DFII10"].reindex(g.index).ffill()
    gate = (ry.diff(20) < 0).shift(1).fillna(False)
    record("XAUUSD_realyield_gate", _hold_days_returns(g, gate, COST_RT["XAUUSD"]))
    # dollar-gated variant
    dx = macro["DTWEXBGS"].reindex(g.index).ffill()
    gate2 = (dx.pct_change(20) < 0).shift(1).fillna(False)
    record("XAUUSD_dollar_gate", _hold_days_returns(g, gate2, COST_RT["XAUUSD"]))
    # both-gates variant (declared: AND of two documented tailwinds)
    record("XAUUSD_ry_dxy_gate", _hold_days_returns(g, gate & gate2, COST_RT["XAUUSD"]))

    # VIX-regime split of the surviving MNQ momentum theme (TSMOM proxy):
    nq = px["MNQ"]
    mom = nq.pct_change(120).shift(1) > 0
    vix = macro["VIXCLS"].reindex(nq.index).ffill().shift(1)
    lo, hi = vix.quantile(0.33), vix.quantile(0.67)
    for name, m in [("lowvix", vix <= lo), ("midvix", (vix > lo) & (vix < hi)),
                    ("hivix", vix >= hi)]:
        record(f"MNQ_trend_{name}", _hold_days_returns(nq, (mom & m).fillna(False),
                                                       COST_RT["MNQ"]), n_configs=3)

    # gold/silver ratio reversion, if silver landed
    xag = DATA / "XAGUSD_15m.csv"
    if xag.exists():
        s = daily_close("XAGUSD")
        j = pd.concat([px["XAUUSD"], s], axis=1, keys=["g", "s"]).dropna()
        z = np.log(j["g"] / j["s"])
        zz = (z - z.rolling(120).mean()) / z.rolling(120).std()
        pos = (-zz.clip(-2, 2) / 2).shift(1)          # ratio high -> short gold/long silver
        rg, rs = j["g"].pct_change(), j["s"].pct_change()
        pair = pos * (rg - rs) / 2
        turn = pos.diff().abs().fillna(0)
        costs = turn * (COST_RT["XAUUSD"] / j["g"] + 0.03 / j["s"]) / 2
        record("XAU_XAG_ratio_rev", (pair - costs).dropna(), n_configs=1)

    tag = "holdout" if final else "dev"
    pd.DataFrame(rows).to_csv(OUT / f"summary_external_{tag}.csv", index=False)
    print(f"\nwrote research/summary_external_{tag}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-science", action="store_true")
    args = ap.parse_args()
    px, macro, fomc = load_all()
    if not args.skip_science:
        part1_science(px, macro, fomc, dev_only=not args.final)
    part2_strategies(px, macro, fomc, final=args.final)


if __name__ == "__main__":
    main()

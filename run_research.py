"""Multi-timeframe, multi-instrument prop-edge research program.

Discipline (the whole point):
  * DEV window  = data before 2022-07-01. All iteration, learning, and
    refinement happens here (walk-forward inside, so selection is still
    leak-free bar-to-bar - the dev/holdout split guards against ME the
    researcher overfitting by iterating).
  * HOLDOUT     = 2022-07-01 -> end. Touched ONCE by the frozen survivor set
    (--final). Configs continue to be chosen per fold from past data only.
  * LEDGER      = research/ledger.jsonl - every trial ever evaluated, so the
    final Deflated Sharpe is corrected for the true number of attempts.

Phases: rules (arb/lfade/orb/pull/sqz/mrev/ict/smc/ta), drift (session and
day-of-week holds), mlx (ML + AI legs with cross-asset features).

    python run_research.py --phase rules --tf 15m
    python run_research.py --phase drift
    python run_research.py --phase mlx --tf 1h
    python run_research.py --phase rules --tf 15m --final   # holdout, once
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.features as F
import qlab.propstrats as P
import qlab.strategies as S
import qlab.walkforward as WF
from qlab.backtest import Costs, simulate
from qlab.features import build_features
from qlab.metrics import full_metrics, summarize_line

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01", tz="UTC")

COSTS = {
    "XAUUSD": Costs(0.25, 0.10, 0.0),
    "MNQ": Costs(0.50, 0.50, 0.31),
    "EURUSD": Costs(0.00006, 0.00002, 0.00004),
}
SERIES = {"XAUUSD": "XAUUSD", "MNQ": "NSXUSD", "EURUSD": "EURUSD"}
# per-day round-trip cost in points for the drift legs (2 sides)
DRIFT_COST = {k: 2 * (c.spread / 2 + c.slip + c.comm_side) for k, c in COSTS.items()}

TF_CFG = {
    "15m": {"bars_per_day": 96,
            "scale": dict(mult=4, min_train=70000, test_len=23000, ml_max_train=150000)},
    "1h": {"bars_per_day": 24,
           "scale": dict(mult=1, min_train=18000, test_len=6000, ml_max_train=60000)},
    "1d": {"bars_per_day": 1,
           "scale": dict(mult=1, min_train=750, test_len=250, ml_max_train=None)},
}
RULE_FAMILIES = {
    "15m": ["arb", "lfade", "orb", "pull", "sqz", "mrev", "ict"],
    "1h": ["arb", "lfade", "orb", "pull", "sqz", "mrev", "ict", "smc", "ta"],
    "1d": ["pull", "sqz", "mrev", "ta"],
}
STYLES = ["A", "B", "C"]
XCOLS = ["x1_ret1", "x1_ret1d", "x2_ret1", "x2_ret1d", "x1_vol", "x2_vol"]

_raw15 = {}


def load_15m(series: str) -> pd.DataFrame:
    if series not in _raw15:
        df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        _raw15[series] = df[["open", "high", "low", "close"]]
    return _raw15[series]


def to_tf(df15: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "15m":
        return df15
    if tf == "1h":
        g = df15.resample("1h")
    else:  # ET calendar days so session/day semantics match the rest of the lab
        g = df15.tz_convert("America/New_York").resample("1D")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()}).dropna()
    return out.tz_convert("UTC")


# instrument -> compact committed daily file (CI runs without the 15m archive)
_DAILY_ALIAS = {"XAUUSD": "XAUUSD", "MNQ": "MNQ", "ES": "ES", "EURUSD": "EURUSD",
                "USDJPY": "USDJPY", "WTIUSD": "WTIUSD", "JPXJPY": "JPXJPY"}


def _daily_from_repo(inst: str):
    """Committed daily series + live extension, returned UTC-aware like to_tf."""
    alias = _DAILY_ALIAS.get(inst)
    if not alias:
        return None
    f = DATA / "daily" / f"{alias}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index).tz_localize("America/New_York")
    ext = DATA / "live" / f"{alias}_ext.csv"
    if ext.exists():
        e = pd.read_csv(ext, index_col=0, parse_dates=True)
        e.index = pd.to_datetime(e.index).tz_localize("America/New_York")
        e = e[e.index > d.index[-1]]
        if len(e):
            d = pd.concat([d, e[["open", "high", "low", "close"]]])
    return d[["open", "high", "low", "close"]].tz_convert("UTC")


def prep(inst: str, tf: str, final: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = _daily_from_repo(inst) if tf == "1d" else None
    if bars is None:
        bars = to_tf(load_15m(SERIES[inst]), tf)
    if not final:
        bars = bars[bars.index < DEV_END]
    return bars, build_features(bars)


def ledger_append(row: dict) -> None:
    OUT.mkdir(exist_ok=True)
    row["ts"] = pd.Timestamp.now(tz="UTC").isoformat()
    with open(OUT / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def ledger_trials() -> int:
    """Total configs ever evaluated (for Sharpe deflation)."""
    path = OUT / "ledger.jsonl"
    if not path.exists():
        return 1
    tot = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        tot += int(r.get("n_configs", 1))
    return max(tot, 1)


def eval_wf(bars, feats, strat, style, costs, cache_key, final, n_trials):
    """Walk-forward + (holdout-filtered) metrics for one model."""
    res = WF.run_wf(bars, feats, strat, style, costs, cache_key=cache_key)
    tr = res.oos_trades
    if final and not tr.empty:
        ts = bars.index[tr["entry_i"].values.astype(int)]
        tr = tr[ts >= DEV_END].reset_index(drop=True)
    if tr.empty:
        return {"n_trades": 0}, res
    et_date = bars.index.tz_convert("America/New_York").date
    conv = bool(res.deploy.get("conviction_scale", False))
    m = simulate(tr, et_date, float(tr["risk_pct"].iloc[-1]), conv,
                 WF.DAILY_CAP, WF.TRAIL_LIMIT, want_daily=True)
    daily = m.pop("daily_returns", None)
    full = full_metrics(tr["R"].values, daily, n_trials=n_trials)
    full["skipped_by_daily_cap"] = m.get("skipped_by_daily_cap")
    full["trailing_dd_breaches"] = m.get("trailing_dd_breaches")
    return full, res


def grid_size(strat: str) -> int:
    if strat in WF._RULES_GRID:
        return len(getattr(S, WF._RULES_GRID[strat]))
    return len(S.ML_GRID)


def phase_rules(args):
    tf = args.tf
    cfg = TF_CFG[tf]
    WF.set_scale(**cfg["scale"])
    P.set_tf(cfg["bars_per_day"])
    fams = args.families or RULE_FAMILIES[tf]
    rows = []
    t0 = time.time()
    n_tr = ledger_trials()
    for inst in args.instruments:
        bars, feats = prep(inst, tf, args.final)
        print(f"{inst} {tf}: {len(bars)} bars "
              f"{bars.index[0].date()} -> {bars.index[-1].date()}, "
              f"folds={len(WF.make_folds(len(bars)))}", flush=True)
        for fam in fams:
            for style in STYLES:
                tag = f"{inst}_{tf}_{fam}_{style}"
                try:
                    m, res = eval_wf(bars, feats, fam, style, COSTS[inst],
                                     f"{SERIES[inst]}_{tf}", args.final, n_tr)
                except Exception as e:  # noqa: BLE001
                    print(f"  {tag}: ERROR {e}", flush=True)
                    continue
                m.update({"model": tag, "instrument": inst, "tf": tf,
                          "family": fam, "style": style,
                          "window": "holdout" if args.final else "dev"})
                rows.append(m)
                print(f"[{time.time()-t0:6.0f}s] {summarize_line(tag, m)}", flush=True)
                if not args.final:
                    ledger_append({"phase": "rules", "tf": tf, "inst": inst,
                                   "family": fam, "style": style,
                                   "n_configs": grid_size(fam) * (8 if style == "C" else
                                                                  (4 if style == "B" else 1)),
                                   "dev_expectancy_R": m.get("expectancy_R"),
                                   "dev_sharpe": m.get("sharpe"),
                                   "dev_t": m.get("t_stat_trades")})
                if args.final and res is not None:
                    res.oos_trades.to_csv(OUT / f"holdout_trades_{tag}.csv", index=False)
    return rows


def phase_drift(args):
    rows = []
    n_tr = ledger_trials()
    for inst in args.instruments:
        bars = to_tf(load_15m(SERIES[inst]), "15m")
        dev = bars[bars.index < DEV_END]
        use = bars if args.final else dev
        legs = {
            "asia_drift": P.session_drift_returns(use, DRIFT_COST[inst], hours=(20, 24)),
            "fri_drift": P.dow_drift_returns(use, DRIFT_COST[inst], dow=4),
        }
        for name, ret in legs.items():
            if args.final:
                ret = ret[ret.index >= DEV_END.tz_localize(None)]
            m = full_metrics(None, ret, n_trials=n_tr)
            tag = f"{inst}_{name}"
            m.update({"model": tag, "instrument": inst, "tf": "session",
                      "family": name, "style": "-",
                      "window": "holdout" if args.final else "dev"})
            rows.append(m)
            print(summarize_line(tag, m), flush=True)
            if not args.final:
                ledger_append({"phase": "drift", "inst": inst, "family": name,
                               "n_configs": 1, "dev_sharpe": m.get("sharpe")})
            else:
                ret.to_csv(OUT / f"holdout_daily_{tag}.csv", header=["ret"])
    return rows


def add_cross_features(feats: pd.DataFrame, inst: str, tf: str,
                       final: bool) -> pd.DataFrame:
    """Cross-asset context: returns/vol of the OTHER two instruments aligned
    onto this instrument's bars (ffill, so only completed bars are seen)."""
    others = [i for i in SERIES if i != inst]
    f = feats.copy()
    bpd = TF_CFG[tf]["bars_per_day"]
    for k, o in enumerate(others, 1):
        ob = to_tf(load_15m(SERIES[o]), tf)
        if not final:
            ob = ob[ob.index < DEV_END]
        c = ob["close"].reindex(f.index, method="ffill")
        f[f"x{k}_ret1"] = np.log(c / c.shift(1))
        f[f"x{k}_ret1d"] = np.log(c / c.shift(bpd))
        f[f"x{k}_vol"] = f[f"x{k}_ret1"].rolling(5 * bpd).std()
    return f


def phase_mlx(args):
    tf = args.tf
    cfg = TF_CFG[tf]
    WF.set_scale(**cfg["scale"])
    rows = []
    t0 = time.time()
    n_tr = ledger_trials()
    saved = list(F.ML_FEATURES)
    try:
        for c in XCOLS:
            if c not in F.ML_FEATURES:
                F.ML_FEATURES.append(c)
        for inst in args.instruments:
            bars, feats = prep(inst, tf, args.final)
            feats = add_cross_features(feats, inst, tf, args.final)
            print(f"{inst} {tf} mlx: {len(bars)} bars, "
                  f"{len(F.ML_FEATURES)} features", flush=True)
            for fam in ("ml", "ai"):
                for style in STYLES:
                    tag = f"{inst}_{tf}_{fam}x_{style}"
                    m, res = eval_wf(bars, feats, fam, style, COSTS[inst],
                                     f"{SERIES[inst]}_{tf}_x", args.final, n_tr)
                    m.update({"model": tag, "instrument": inst, "tf": tf,
                              "family": fam + "x", "style": style,
                              "window": "holdout" if args.final else "dev"})
                    rows.append(m)
                    print(f"[{time.time()-t0:6.0f}s] {summarize_line(tag, m)}",
                          flush=True)
                    if not args.final:
                        ledger_append({"phase": "mlx", "tf": tf, "inst": inst,
                                       "family": fam + "x", "style": style,
                                       "n_configs": len(S.ML_GRID) * 4,
                                       "dev_expectancy_R": m.get("expectancy_R"),
                                       "dev_sharpe": m.get("sharpe"),
                                       "dev_t": m.get("t_stat_trades")})
                    if args.final and res is not None:
                        res.oos_trades.to_csv(OUT / f"holdout_trades_{tag}.csv",
                                              index=False)
    finally:
        F.ML_FEATURES[:] = saved
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["rules", "drift", "mlx"], required=True)
    ap.add_argument("--tf", choices=["15m", "1h", "1d"], default="15m")
    ap.add_argument("--instruments", nargs="+", default=list(SERIES))
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--final", action="store_true",
                    help="run on FULL history and report holdout-only metrics")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    P.register(WF, S)
    rows = {"rules": phase_rules, "drift": phase_drift, "mlx": phase_mlx}[args.phase](args)
    tag = args.out_tag or f"{args.phase}_{args.tf}_{'holdout' if args.final else 'dev'}"
    pd.DataFrame(rows).to_csv(OUT / f"summary_{tag}.csv", index=False)
    print(f"\nwrote research/summary_{tag}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()

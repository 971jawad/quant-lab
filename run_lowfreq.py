"""Low-frequency, cost-light legs - the corners where documented edge tends to
survive (unlike the intraday directional zoo).

  trend       per-instrument daily time-series momentum, vol-targeted, lookback
              chosen per walk-forward fold on training Sharpe only.
  xsec        cross-asset momentum: each day rank the instruments by trailing
              return, long the strongest / short the weakest, vol-targeted.
  overnight   equity-index close->open drift capture (ES, NQ). No parameter, so
              the whole sample is out-of-sample by construction.

Everything is daily (resampled from the 15m data), so turnover - and therefore
cost drag, the thing that killed the intraday ML legs - is ~100x smaller.

Honesty: these are thin, widely-known risk premia. Metrics are net of modelled
costs; a small lookback grid is corrected with a Deflated Sharpe. Daily PnL is
written to results3/lowfreq_daily/ so run_meta_analysis can fold them into the
family-wise correction. Not investment advice.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "results3"
DAILY_OUT = OUT / "lowfreq_daily"
DATA_MAP = {"ES": "SPXUSD", "NQ": "NSXUSD", "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
# per-side cost in price points -> converted to a return fraction per unit turnover
COST_PTS = {"ES": 0.25 / 2 + 0.25 + 0.06, "NQ": 0.50 / 2 + 0.50 + 0.31,
            "XAUUSD": 0.25 / 2 + 0.10 + 0.0, "EURUSD": 0.00006 / 2 + 0.00002 + 0.00004}
OVERNIGHT_INSTR = ("ES", "NQ")
LOOKBACKS = [40, 80, 160, 240]      # trading-day momentum horizons (the grid)
TARGET_VOL = 0.10                    # annualized vol target for sizing
VOL_WIN = 60
WARMUP, MIN_TRAIN, TEST_LEN = 260, 750, 250   # daily bars (~1y / 3y / 1y)
Phi, Phi_inv, EULER = stats.norm.cdf, stats.norm.ppf, 0.5772156649


# series name -> the compact committed daily file (lets CI run without the
# multi-hundred-MB 15-minute archive, and picks up the live Yahoo extension)
_DAILY_ALIAS = {"NSXUSD": "MNQ", "SPXUSD": "ES", "XAUUSD": "XAUUSD",
                "EURUSD": "EURUSD", "USDJPY": "USDJPY", "WTIUSD": "WTIUSD",
                "JPXJPY": "JPXJPY"}


def to_daily(series: str) -> pd.DataFrame:
    alias = _DAILY_ALIAS.get(series)
    frozen = DATA / "daily" / f"{alias}.csv" if alias else None
    if frozen is not None and frozen.exists():
        d = pd.read_csv(frozen, index_col=0, parse_dates=True)
        d.index = pd.to_datetime(d.index).tz_localize("America/New_York")
        ext = DATA / "live" / f"{alias}_ext.csv"
        if ext.exists():
            e = pd.read_csv(ext, index_col=0, parse_dates=True)
            e.index = pd.to_datetime(e.index).tz_localize("America/New_York")
            e = e[e.index > d.index[-1]]
            if len(e):
                d = pd.concat([d, e[["open", "high", "low", "close"]]])
        return d[["open", "high", "low", "close"]]
    df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()


def perf(sret: pd.Series, n_trials: int = 1) -> dict:
    """Metrics for a daily return series (net). n_trials deflates the Sharpe."""
    sret = sret.dropna()
    if len(sret) < 30:
        return {"n_days": int(len(sret))}
    mu, sd = sret.mean(), sret.std()
    sharpe = mu / (sd + 1e-12) * np.sqrt(252)
    eq = (1 + sret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann = eq.iloc[-1] ** (252 / len(sret)) - 1
    t = mu / (sd + 1e-12) * np.sqrt(len(sret))
    # deflated sharpe (daily units) vs expected max over n_trials, with skew/kurt
    sk, ku = float(stats.skew(sret)), float(stats.kurtosis(sret, fisher=False))
    sr0 = mu / (sd + 1e-12)
    if n_trials > 1:
        emax = np.sqrt(1.0 / len(sret)) * ((1 - EULER) * Phi_inv(1 - 1.0 / n_trials) +
                                           EULER * Phi_inv(1 - 1.0 / (n_trials * np.e)))
    else:
        emax = 0.0
    denom = np.sqrt(max(1 - sk * sr0 + (ku - 1) / 4 * sr0 ** 2, 1e-9))
    dsr = float(Phi((sr0 - emax) * np.sqrt(max(len(sret) - 1, 1)) / denom))
    return {"n_days": int(len(sret)), "ann_return_pct": round(ann * 100, 2),
            "sharpe": round(float(sharpe), 3), "calmar": round(float(ann / abs(dd)), 2) if dd < 0 else None,
            "max_dd_pct": round(float(dd) * 100, 2), "t_stat": round(float(t), 2),
            "DSR_prob_true_SR_gt_0": round(dsr, 4)}


def trend_returns(d: pd.DataFrame, inst: str, lookback: int) -> pd.Series:
    """Vol-targeted daily TS-momentum. Position for day t+1 uses info <= t."""
    r = d["close"].pct_change()
    mom = d["close"].pct_change(lookback)
    vol = r.rolling(VOL_WIN).std()
    sig = np.sign(mom)
    pos = (sig * (TARGET_VOL / np.sqrt(252)) / vol).clip(-3, 3).shift(1)  # no lookahead
    cost = COST_PTS[inst] / d["close"]
    turn = pos.diff().abs().fillna(pos.abs())
    return (pos * r - turn * cost).rename(inst)


def overnight_returns(d: pd.DataFrame, inst: str) -> pd.Series:
    """Long the close->open gap each day (equity-index drift). One crossing/day."""
    gap = d["open"] / d["close"].shift(1) - 1
    cost = COST_PTS[inst] / d["close"]
    return (gap - cost).rename(inst).dropna()


def make_folds(n):
    folds, te = [], WARMUP + MIN_TRAIN
    while te < n - 100:
        folds.append((te, min(te + TEST_LEN, n)))
        te += TEST_LEN
    return folds


def wf_trend(d: pd.DataFrame, inst: str) -> tuple[pd.Series, list]:
    """Anchored WF: pick lookback on training Sharpe, apply to the OOS block."""
    n = len(d)
    oos, log = [], []
    cand = {L: trend_returns(d, inst, L) for L in LOOKBACKS}
    for tr_end, te_end in make_folds(n):
        best, bs = None, -np.inf
        for L in LOOKBACKS:
            s = cand[L].iloc[WARMUP:tr_end].dropna()
            sh = s.mean() / (s.std() + 1e-12) if len(s) > 30 else -np.inf
            if sh > bs:
                bs, best = sh, L
        seg = cand[best].iloc[tr_end:te_end]
        oos.append(seg)
        log.append({"train_end": int(tr_end), "lookback": best})
    return (pd.concat(oos) if oos else pd.Series(dtype=float)), log


def wf_xsec(daily: dict) -> tuple[pd.Series, list]:
    """Cross-sectional momentum across the instruments, vol-targeted portfolio."""
    closes = pd.DataFrame({k: v["close"] for k, v in daily.items()}).dropna()
    rets = closes.pct_change()
    n = len(closes)
    oos, log = [], []
    cand = {}
    for L in LOOKBACKS:
        mom = closes.pct_change(L)
        rank = mom.rank(axis=1)
        k = rank.max(axis=1)
        w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        w = w.where(~(rank.eq(k, axis=0)), 1.0).where(~(rank.eq(1, axis=0)), -1.0)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
        pvol = (w.shift(1) * rets).sum(axis=1).rolling(VOL_WIN).std()
        lev = (TARGET_VOL / np.sqrt(252)) / pvol
        gross = (w.shift(1) * rets).sum(axis=1) * lev.shift(1).clip(0, 5)
        turn = w.diff().abs().sum(axis=1)
        cost = pd.Series([np.mean(list(COST_PTS.values()))] * n, index=closes.index) / closes.mean(axis=1)
        cand[L] = (gross - turn * cost).rename("xsec")
    for tr_end, te_end in make_folds(n):
        best, bs = None, -np.inf
        for L in LOOKBACKS:
            s = cand[L].iloc[WARMUP:tr_end].dropna()
            sh = s.mean() / (s.std() + 1e-12) if len(s) > 30 else -np.inf
            if sh > bs:
                bs, best = sh, L
        oos.append(cand[best].iloc[tr_end:te_end])
        log.append({"train_end": int(tr_end), "lookback": best})
    return (pd.concat(oos) if oos else pd.Series(dtype=float)), log


def main():
    OUT.mkdir(exist_ok=True)
    DAILY_OUT.mkdir(exist_ok=True)
    daily = {inst: to_daily(series) for inst, series in DATA_MAP.items()}
    for inst, d in daily.items():
        print(f"{inst}: {len(d)} daily bars  {d.index[0].date()} -> {d.index[-1].date()}", flush=True)

    rows = []

    def record(name, sret, n_trials):
        sret = sret.dropna()
        if sret.empty:
            return
        sret.to_csv(DAILY_OUT / f"{name}.csv", header=["ret"])
        m = perf(sret, n_trials)
        m["strategy"] = name
        rows.append(m)
        print(f"  {name:16} ann={m.get('ann_return_pct'):>7}%  sharpe={m.get('sharpe'):>6}  "
              f"calmar={m.get('calmar')}  maxDD={m.get('max_dd_pct')}%  t={m.get('t_stat')}  "
              f"DSR={m.get('DSR_prob_true_SR_gt_0')}", flush=True)

    print("\n== trend (per-instrument TS-momentum, walk-forward) ==")
    for inst, d in daily.items():
        s, _ = wf_trend(d, inst)
        record(f"trend_{inst}", s, n_trials=len(LOOKBACKS))

    print("\n== xsec (cross-asset momentum, walk-forward) ==")
    s, _ = wf_xsec(daily)
    record("xsec_ALL", s, n_trials=len(LOOKBACKS))

    print("\n== overnight (equity-index close->open drift, no parameter) ==")
    for inst in OVERNIGHT_INSTR:
        s = overnight_returns(daily[inst], inst)
        record(f"overnight_{inst}", s, n_trials=1)

    # books. The momentum book is the structural trend-following portfolio (all
    # trend legs + xsec, equal weight) - NOT a cherry-pick, it includes every
    # instrument. The ALL book additionally drags in the failed overnight legs.
    legs = {r["strategy"]: pd.read_csv(DAILY_OUT / f"{r['strategy']}.csv",
            index_col=0, parse_dates=True)["ret"] for r in rows}
    mom = [k for k in legs if k.startswith("trend_") or k.startswith("xsec")]
    print("\n== momentum book (equal-weight trend_* + xsec, diversified) ==")
    record("momentum_BOOK", pd.DataFrame({k: legs[k] for k in mom}).mean(axis=1),
           n_trials=len(LOOKBACKS))
    print("\n== all-legs book (includes the failed overnight legs) ==")
    record("lowfreq_ALL", pd.DataFrame(legs).mean(axis=1), n_trials=len(rows))

    pd.DataFrame(rows).to_csv(OUT / "lowfreq_summary.csv", index=False)
    best = max(rows, key=lambda r: r.get("DSR_prob_true_SR_gt_0", 0))
    verdict = (f"best leg {best['strategy']} DSR={best['DSR_prob_true_SR_gt_0']} "
               f"(t={best['t_stat']}, sharpe={best['sharpe']}) -> "
               + ("EDGE candidate (DSR>0.95) worth forward-testing"
                  if best.get("DSR_prob_true_SR_gt_0", 0) > 0.95 else
                  "still not a proven edge after cost + deflation"))
    with open(OUT / "lowfreq_summary.json", "w") as fh:
        json.dump({"rows": rows, "verdict": verdict}, fh, indent=2, default=str)
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()

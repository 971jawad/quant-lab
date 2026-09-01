"""Meta-analysis for GENUINE edge - does any strategy (or the ensemble) beat
luck once we account for how many things we tried?

Three complementary, textbook tests:
  1. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) - deflates the best
     observed Sharpe by the number of trials, its variance, and the return
     skew/kurtosis. Reports P(true SR > 0).
  2. White's Reality Check (2000), studentized, stationary-bootstrap (Politis-
     Romano) - the family-wise p-value for "the single best strategy's mean
     daily PnL > 0" across the WHOLE universe searched. This is the honest
     multiple-testing correction.
  3. PBO via CSCV (Bailey et al. 2017) - probability that the config that looked
     best in-sample is below-median out-of-sample. High PBO = the leaderboard is
     noise.

All inputs are the already-out-of-sample walk-forward trades (results2 + the
ensemble in results3). Daily PnL is per-trade R times its risk fraction, booked
on the ET exit date - additive risk units, aligned across models by date.
"""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import qlab.walkforward as WF

ROOT = Path(__file__).parent
DATA, RES2, RES3, OUT = ROOT / "data", ROOT / "results2", ROOT / "results3", ROOT / "results3"
DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
SCALE = {"mult": 4, "min_train": 70000, "test_len": 23000, "ml_max_train": 150000}
FAMS = ["smc", "ta", "ml", "ml_err", "ml_rec", "ai"]
STYLES = ["A", "B", "C"]
Phi, Phi_inv = stats.norm.cdf, stats.norm.ppf
EULER = 0.5772156649


def daily_pnl(trades: pd.DataFrame, et_date: np.ndarray) -> pd.Series:
    """Additive daily PnL in risk units: sum of R*risk_pct over trades exiting
    that ET date."""
    if trades.empty:
        return pd.Series(dtype=float)
    rp = trades.get("risk_pct")
    rp = rp.astype(float).fillna(0.005) if rp is not None else 0.005
    pnl = trades["R"].astype(float).values * (rp.values if hasattr(rp, "values") else rp)
    d = et_date[trades["exit_i"].astype(int).values]
    return pd.Series(pnl, index=pd.Index(d, name="date")).groupby(level=0).sum()


def deflated_sharpe(sr_hat, sr_trials, n_obs, skew, kurt):
    """PSR against the deflated benchmark E[max SR] over N independent trials."""
    N = max(len(sr_trials), 2)
    var_sr = np.var(sr_trials, ddof=1) if np.std(sr_trials) > 0 else 1e-6
    e_max = np.sqrt(var_sr) * ((1 - EULER) * Phi_inv(1 - 1.0 / N) +
                               EULER * Phi_inv(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2, 1e-9))
    z = (sr_hat - e_max) * np.sqrt(max(n_obs - 1, 1)) / denom
    return float(Phi(z)), float(e_max)


def stationary_bootstrap_idx(T, B, mean_block, rng):
    """Politis-Romano stationary bootstrap row indices, shape (B, T)."""
    p = 1.0 / mean_block
    out = np.empty((B, T), dtype=int)
    for b in range(B):
        idx = np.empty(T, dtype=int)
        t = rng.integers(0, T)
        for i in range(T):
            idx[i] = t
            if rng.random() < p:
                t = rng.integers(0, T)
            else:
                t = (t + 1) % T
        out[b] = idx
    return out


def reality_check(M: pd.DataFrame, B=2000, mean_block=10, seed=7):
    """White's RC studentized. M: dates x models daily PnL (0-filled). Returns
    family-wise p for max_k mean_k > 0, plus the winning model."""
    X = M.values
    T, K = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = np.inf
    stat = np.sqrt(T) * mu / sd
    k_best = int(np.nanargmax(stat))
    V = float(stat[k_best])
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(T, B, mean_block, rng)
    ge = 0
    for b in range(B):
        Xb = X[idx[b]]
        mub = Xb.mean(axis=0)
        Vb = np.sqrt(T) * (mub - mu) / sd     # recentered studentized
        if np.nanmax(Vb) >= V:
            ge += 1
    return {"best_model": M.columns[k_best], "V": round(V, 3),
            "rc_pvalue": round((ge + 1) / (B + 1), 4), "n_models": K, "T": T}


def pbo_cscv(R: pd.DataFrame, S=10):
    """Probability of Backtest Overfitting. R: folds x models matrix of per-fold
    avg_R. Split folds into S groups, every balanced IS/OOS combination; logit of
    the IS-best model's OOS relative rank. PBO = P(logit <= 0)."""
    R = R.dropna(axis=1, how="any")
    n = len(R)
    if n < 4 or R.shape[1] < 4:
        return {"pbo": None, "note": "insufficient folds/models"}
    S = min(S, n - (n % 2 if n % 2 else 0)) if n >= S else n
    if S % 2:
        S -= 1
    groups = np.array_split(np.arange(n), S)
    logits = []
    for isg in combinations(range(S), S // 2):
        is_rows = np.concatenate([groups[g] for g in isg])
        oos_rows = np.concatenate([groups[g] for g in range(S) if g not in isg])
        is_perf = R.iloc[is_rows].mean()
        oos_perf = R.iloc[oos_rows].mean()
        best = is_perf.idxmax()
        ranks = oos_perf.rank()
        rel = ranks[best] / (len(ranks) + 1)
        rel = min(max(rel, 1e-6), 1 - 1e-6)
        logits.append(np.log(rel / (1 - rel)))
    logits = np.array(logits)
    return {"pbo": round(float((logits <= 0).mean()), 4),
            "n_splits": len(logits), "median_logit": round(float(np.median(logits)), 3)}


def main():
    WF.set_scale(**SCALE)
    OUT.mkdir(exist_ok=True)
    idx_cache, fold_cache = {}, {}
    series_daily, per_fold = {}, {}

    def load_idx(series):
        if series not in idx_cache:
            df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0, usecols=[0])
            idx_cache[series] = pd.to_datetime(df.index, utc=True)
            fold_cache[series] = WF.make_folds(len(idx_cache[series]))
        return idx_cache[series], fold_cache[series]

    # ---- gather per-model daily PnL and per-fold avg_R
    for inst, series in DATA_MAP.items():
        idx, folds = load_idx(series)
        et = idx.tz_convert("America/New_York").date
        et = np.array(et)
        for fam in FAMS:
            for st in STYLES:
                tag = f"{inst}_{fam}_{st}"
                p = RES2 / f"trades_{tag}.csv"
                if not p.exists():
                    continue
                tr = pd.read_csv(p)
                if tr.empty:
                    continue
                series_daily[tag] = daily_pnl(tr, et)
                fold_r = []
                for (a, b) in folds:
                    m = (tr["sig_i"] >= a) & (tr["sig_i"] < b)
                    fold_r.append(tr.loc[m, "R"].mean() if m.any() else np.nan)
                per_fold[tag] = fold_r

    # include the ensemble books
    for inst, series in DATA_MAP.items():
        idx, _ = load_idx(series)
        et = np.array(idx.tz_convert("America/New_York").date)
        p = RES3 / f"trades_ensemble_{inst}.csv"
        if p.exists():
            tr = pd.read_csv(p)
            if not tr.empty:
                series_daily[f"ENS_{inst}"] = daily_pnl(tr, et)

    # include the low-frequency legs (already daily return series). Scale-free RC
    # studentizes each column, so mixing risk-unit and return-unit columns is fine.
    lf_dir = RES3 / "lowfreq_daily"
    if lf_dir.exists():
        for p in sorted(lf_dir.glob("*.csv")):
            s = pd.read_csv(p, index_col=0)
            col = s.columns[0]
            ser = s[col]
            ser.index = pd.to_datetime(ser.index, utc=True).date
            series_daily[f"LF_{p.stem}"] = ser.groupby(level=0).sum()

    if not series_daily:
        print("No trade files found. Run run_all.py / run_ai.py / run_ensemble.py first.")
        return

    # ---- align daily PnL matrix
    M = pd.DataFrame(series_daily).sort_index().fillna(0.0)
    M = M.loc[:, M.std() > 0]

    # ---- 1. Deflated Sharpe on the best single model by raw Sharpe
    sr_trials, sr_info = [], {}
    for c in M.columns:
        r = M[c].values
        r = r[r != 0] if (r != 0).sum() > 30 else r
        s = r.mean() / (r.std() + 1e-12)
        sr_trials.append(s)
        sr_info[c] = (s, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))
    best_sr_model = max(sr_info, key=lambda c: sr_info[c][0])
    s_hat, n_obs, sk, ku = sr_info[best_sr_model]
    dsr, e_max = deflated_sharpe(s_hat, sr_trials, n_obs, sk, ku)

    # ---- 2. Reality Check
    rc = reality_check(M)

    # ---- 3. PBO on the model universe (exclude ensembles; align by fold index)
    kmin = min(len(v) for v in per_fold.values())
    Rmat = pd.DataFrame({t: v[:kmin] for t, v in per_fold.items()})
    pbo = pbo_cscv(Rmat)

    out = {
        "universe_size": int(M.shape[1]),
        "aligned_days": int(M.shape[0]),
        "deflated_sharpe": {
            "best_by_sharpe": best_sr_model,
            "daily_sharpe": round(float(s_hat), 4),
            "ann_sharpe": round(float(s_hat) * np.sqrt(252), 3),
            "E_max_sharpe_under_null": round(float(e_max), 4),
            "DSR_prob_true_SR_gt_0": round(dsr, 4),
        },
        "white_reality_check": rc,
        "pbo_cscv": pbo,
        "verdict": None,
    }
    edge = (out["deflated_sharpe"]["DSR_prob_true_SR_gt_0"] > 0.95 and
            rc["rc_pvalue"] < 0.05 and (pbo.get("pbo") or 1) < 0.5)
    out["verdict"] = ("GENUINE EDGE survives multiple-testing correction"
                      if edge else
                      "NO genuine edge: best result is consistent with luck "
                      "after correcting for the number of strategies tried")
    with open(OUT / "meta_analysis.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

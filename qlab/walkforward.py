"""Anchored walk-forward harness.

For every fold: parameters (and for styles B/C, the risk/RR/exec config) are
selected using ONLY the training window, then applied unchanged to the
out-of-sample test window. ML models are refit per fold with an embargo of
ML_HORIZON+1 bars so no training label window touches test data. Trades that
exit after the training boundary are excluded from selection metrics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import Costs, bt_trades, simulate, train_objective
from . import strategies as S

WARMUP = 500
MIN_TRAIN = 6000
TEST_LEN = 1500
MIN_LAST_TEST = 300
ML_MAX_TRAIN = None      # cap on ML training-window bars (None = anchored/full)

RISK_GRID_B = [0.0025, 0.005, 0.0075, 0.01]
RR_GRID_B = [1.5, 2.0, 3.0, 4.0]
STYLE_A = {"rr": 3.0, "risk": 0.0075, "time_exit": 48, "max_concurrent": 1,
           "conviction_scale": False}
EXECS_AB = [(rr, 48, 1) for rr in RR_GRID_B]            # (rr, time_exit, mc)
EXECS_C = [(rr, te, 3) for rr in (1.5, 2.0) for te in (24, 48)]
RISK_GRID_C = [0.002, 0.004]
DAILY_CAP = 0.03
TRAIL_LIMIT = 0.05


def set_scale(mult: int, min_train: int | None = None,
              test_len: int | None = None, ml_max_train: int | None = None) -> None:
    """Rescale bar-count constants for a finer timeframe (mult=4: 15m vs 1h).
    min_train/test_len override the defaults for long histories (e.g. 3y/1y).
    ml_max_train caps ML fit windows (rolling) so 16y fits stay tractable."""
    global WARMUP, MIN_TRAIN, TEST_LEN, MIN_LAST_TEST, ML_MAX_TRAIN
    global STYLE_A, EXECS_AB, EXECS_C
    WARMUP = 500 * mult
    MIN_TRAIN = min_train if min_train else 6000 * mult
    TEST_LEN = test_len if test_len else 1500 * mult
    MIN_LAST_TEST = 300 * mult
    ML_MAX_TRAIN = ml_max_train
    STYLE_A = {**STYLE_A, "time_exit": 48 * mult}
    EXECS_AB = [(rr, 48 * mult, 1) for rr in RR_GRID_B]
    EXECS_C = [(rr, te * mult, 3) for rr in (1.5, 2.0) for te in (24, 48)]
    S.set_scale(mult)


@dataclass
class WFResult:
    oos_trades: pd.DataFrame
    fold_log: list
    deploy: dict            # config chosen on the final training window


# GBM fits depend only on (data series, train range, scheme); cache across
# styles/instruments that share a data series. The error-scheme teacher chain
# is deterministic in fold order, so the key stays valid.
_FIT_CACHE: dict = {}

STRAT_SCHEME = {"ml": "uniform", "ml_err": "error", "ml_rec": "recency"}


def _ml_fit_cached(feats: pd.DataFrame, lo: int, hi: int, key: str,
                   scheme: str = "uniform", teacher=None):
    if ML_MAX_TRAIN:
        lo = max(lo, hi - ML_MAX_TRAIN)   # rolling window at deep-history scale
    k = (key, lo, hi, scheme)
    if key and k in _FIT_CACHE:
        return _FIT_CACHE[k]
    m = S.ml_fit(feats, lo, hi, scheme, teacher)
    if key:
        _FIT_CACHE[k] = m
    return m


def make_folds(n: int) -> list[tuple[int, int]]:
    folds = []
    train_end = WARMUP + MIN_TRAIN
    while train_end < n - MIN_LAST_TEST:
        test_end = min(train_end + TEST_LEN, n)
        folds.append((train_end, test_end))
        train_end = test_end
    return folds


def _slice_train(tr: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Training-selection R series: entered AND exited inside [lo, hi)."""
    m = (tr["sig_i"] >= lo) & (tr["sig_i"] < hi) & (tr["exit_i"] < hi)
    return tr.loc[m, "R"].values


def _slice_test(tr: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return tr[(tr["sig_i"] >= lo) & (tr["sig_i"] < hi)].copy()


def _risk_by_sim(trades: pd.DataFrame, et_date: np.ndarray, risk_grid,
                 conviction_scale: bool) -> float:
    """Choose risk%% on TRAIN trades by drawdown-adjusted return with a heavy
    penalty per trailing-DD breach (prop survival first)."""
    best, best_score = risk_grid[0], -np.inf
    for rk in risk_grid:
        t = trades.copy()
        t["risk_pct"] = rk
        m = simulate(t, et_date, rk, conviction_scale, DAILY_CAP, TRAIL_LIMIT)
        if m.get("n_trades", 0) == 0:
            continue
        score = m["ann_return_pct"] / (1 + abs(m["max_dd_pct"])) * (0.5 ** m["trailing_dd_breaches"])
        if score > best_score:
            best, best_score = rk, score
    return best


def _rules_signal_cache(feats: pd.DataFrame, strat: str) -> dict:
    grid = S.SMC_GRID if strat == "smc" else S.TA_GRID
    gen = S.smc_signals if strat == "smc" else S.ta_signals
    return {tuple(sorted(p.items())): (p, gen(feats, p)) for p in grid}


def run_wf(bars: pd.DataFrame, feats: pd.DataFrame, strat: str, style: str,
           costs: Costs, cache_key: str = "") -> WFResult:
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    et_date = bars.index.tz_convert("America/New_York").date
    n = len(bars)
    folds = make_folds(n)
    execs = EXECS_C if style == "C" else ([(STYLE_A["rr"], 48, 1)] if style == "A" else EXECS_AB)
    conviction = style == "C"
    risk_grid = RISK_GRID_C if style == "C" else (RISK_GRID_B if style == "B" else [STYLE_A["risk"]])

    oos_parts, fold_log = [], []
    deploy = {}

    if strat in ("smc", "ta"):
        sig_cache = _rules_signal_cache(feats, strat)
        bt_cache: dict = {}

        def trades_for(key, sig, ex):
            k = (key, ex)
            if k not in bt_cache:
                rr, te, mc = ex
                bt_cache[k] = bt_trades(o, h, l, c, sig, rr, te, costs, mc)
            return bt_cache[k]

        for train_end, test_end in folds:
            best = None
            for key, (p, sig) in sig_cache.items():
                for ex in execs:
                    tr = trades_for(key, sig, ex)
                    obj = train_objective(_slice_train(tr, WARMUP, train_end))
                    if best is None or obj > best[0]:
                        best = (obj, p, ex, tr)
            obj, p, ex, tr = best
            if not np.isfinite(obj):
                fold_log.append({"train_end": train_end, "skipped": "no viable config"})
                continue
            train_tr = tr[(tr["sig_i"] >= WARMUP) & (tr["sig_i"] < train_end) &
                          (tr["exit_i"] < train_end)]
            risk = _risk_by_sim(train_tr, et_date, risk_grid, conviction)
            test_tr = _slice_test(tr, train_end, test_end)
            test_tr["risk_pct"] = risk
            oos_parts.append(test_tr)
            deploy = {"params": p, "rr": ex[0], "time_exit": ex[1],
                      "max_concurrent": ex[2], "risk_pct": risk,
                      "conviction_scale": conviction, "train_obj": round(obj, 2)}
            fold_log.append({"train_end": train_end, "test_end": test_end,
                             "oos_trades": len(test_tr), **deploy})

    elif strat in STRAT_SCHEME:
        scheme = STRAT_SCHEME[strat]
        prev_full = None                # teacher chain: fold k studies fold k-1
        for train_end, test_end in folds:
            emb = S.ML_HORIZON + 1
            fit_hi = WARMUP + int((train_end - WARMUP) * 0.75) - emb
            val_lo, val_hi = fit_hi + emb, train_end
            m_inner = _ml_fit_cached(feats, WARMUP, fit_hi, cache_key, scheme, prev_full)
            best = None
            for p in S.ML_GRID:
                sig = S.ml_signals(feats, m_inner, val_lo, val_hi, p)
                for ex in execs:
                    rr, te, mc = ex
                    tr = bt_trades(o, h, l, c, sig, rr, te, costs, mc)
                    obj = train_objective(_slice_train(tr, val_lo, val_hi), min_trades=10)
                    if best is None or obj > best[0]:
                        best = (obj, p, ex, tr)
            obj, p, ex, val_tr = best
            if not np.isfinite(obj):
                fold_log.append({"train_end": train_end, "skipped": "no viable config"})
                continue
            val_in = val_tr[(val_tr["sig_i"] >= val_lo) & (val_tr["sig_i"] < val_hi) &
                            (val_tr["exit_i"] < val_hi)]
            risk = _risk_by_sim(val_in, et_date, risk_grid, conviction)
            m_full = _ml_fit_cached(feats, WARMUP, train_end - emb, cache_key, scheme, prev_full)
            prev_full = m_full
            sig = S.ml_signals(feats, m_full, train_end, test_end, p)
            rr, te, mc = ex
            test_tr = bt_trades(o, h, l, c, sig, rr, te, costs, mc)
            test_tr["risk_pct"] = risk
            oos_parts.append(test_tr)
            deploy = {"params": p, "rr": rr, "time_exit": te, "max_concurrent": mc,
                      "risk_pct": risk, "conviction_scale": conviction,
                      "train_obj": round(obj, 2)}
            fold_log.append({"train_end": train_end, "test_end": test_end,
                             "oos_trades": len(test_tr), **deploy})
    else:
        raise ValueError(strat)

    oos = (pd.concat(oos_parts).sort_values("entry_i").reset_index(drop=True)
           if oos_parts else pd.DataFrame(columns=["sig_i", "entry_i", "exit_i",
                                                   "dir", "R", "conviction",
                                                   "reason", "risk_pct"]))
    return WFResult(oos, fold_log, deploy)


def deploy_final(bars: pd.DataFrame, feats: pd.DataFrame, strat: str, style: str,
                 costs: Costs, cache_key: str = ""):
    """Select the go-forward config using ALL available history as the
    training window. Used only to produce weight files - all reported
    performance comes from the walk-forward OOS trades."""
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    et_date = bars.index.tz_convert("America/New_York").date
    n = len(bars)
    execs = EXECS_C if style == "C" else ([(STYLE_A["rr"], 48, 1)] if style == "A" else EXECS_AB)
    conviction = style == "C"
    risk_grid = RISK_GRID_C if style == "C" else (RISK_GRID_B if style == "B" else [STYLE_A["risk"]])

    model = None
    if strat in ("smc", "ta"):
        best = None
        for key, (p, sig) in _rules_signal_cache(feats, strat).items():
            for ex in execs:
                rr, te, mc = ex
                tr = bt_trades(o, h, l, c, sig, rr, te, costs, mc)
                obj = train_objective(_slice_train(tr, WARMUP, n))
                if best is None or obj > best[0]:
                    best = (obj, p, ex, tr)
        obj, p, ex, tr = best
        train_tr = tr[(tr["sig_i"] >= WARMUP) & (tr["exit_i"] < n)]
    else:
        scheme = STRAT_SCHEME[strat]
        emb = S.ML_HORIZON + 1
        # teacher for the deploy fit = last walk-forward full model (if built)
        folds = make_folds(n)
        teacher = None
        if folds:
            t_lo, t_hi = WARMUP, folds[-1][0] - emb
            if ML_MAX_TRAIN:
                t_lo = max(t_lo, t_hi - ML_MAX_TRAIN)
            teacher = _FIT_CACHE.get((cache_key, t_lo, t_hi, scheme))
        fit_hi = WARMUP + int((n - WARMUP) * 0.75) - emb
        val_lo = fit_hi + emb
        m_inner = _ml_fit_cached(feats, WARMUP, fit_hi, cache_key, scheme, teacher)
        best = None
        for p in S.ML_GRID:
            sig = S.ml_signals(feats, m_inner, val_lo, n, p)
            for ex in execs:
                rr, te, mc = ex
                tr = bt_trades(o, h, l, c, sig, rr, te, costs, mc)
                obj = train_objective(_slice_train(tr, val_lo, n), min_trades=10)
                if best is None or obj > best[0]:
                    best = (obj, p, ex, tr)
        obj, p, ex, tr = best
        train_tr = tr[(tr["sig_i"] >= val_lo) & (tr["exit_i"] < n)]
        model = _ml_fit_cached(feats, WARMUP, n - emb, cache_key, scheme, teacher)

    risk = _risk_by_sim(train_tr, et_date, risk_grid, conviction) if len(train_tr) else risk_grid[0]
    cfg = {"params": p, "rr": ex[0], "time_exit": ex[1], "max_concurrent": ex[2],
           "risk_pct": risk, "conviction_scale": conviction,
           "full_train_obj": round(float(obj), 2) if np.isfinite(obj) else None}
    return cfg, model


def oos_metrics(res: WFResult, bars: pd.DataFrame) -> dict:
    et_date = bars.index.tz_convert("America/New_York").date
    tr = res.oos_trades
    if tr.empty:
        return {"n_trades": 0}
    conviction = bool(res.deploy.get("conviction_scale", False))
    # per-fold risk sits on each trade; simulate reads the risk_pct column
    return simulate(tr, et_date, float(tr["risk_pct"].iloc[-1]), conviction,
                    DAILY_CAP, TRAIL_LIMIT)

"""Full performance-metrics suite for the research program.

Two inputs, both optional but at least one required:
  trades_R  - per-trade R multiples in chronological order
  daily_ret - daily portfolio return series (net), tz-naive date index

Every metric the spec asks for: compounded return, CAGR, max DD, Calmar, vol,
Sharpe, Sortino, win rate, avg win/loss, payoff ratio, profit factor,
expectancy, median trade, 5th-percentile trade, worst trade, recovery time,
% time underwater, max consecutive losses, rolling expectancy/Sharpe trend,
and outlier-adjusted return (does the result survive removing the best trades?).
"""
import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def deflated_sharpe(daily_ret: pd.Series, n_trials: int) -> float:
    """Bailey & Lopez de Prado: probability the true Sharpe > 0 after accounting
    for having tried n_trials strategies (expected max of null Sharpes)."""
    r = daily_ret.dropna()
    if len(r) < 60:
        return np.nan
    sr0 = r.mean() / (r.std() + 1e-12)
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    if n_trials > 1:
        emax = np.sqrt(1.0 / len(r)) * (
            (1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
            + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    else:
        emax = 0.0
    denom = np.sqrt(max(1 - sk * sr0 + (ku - 1) / 4 * sr0 ** 2, 1e-9))
    return float(stats.norm.cdf((sr0 - emax) * np.sqrt(max(len(r) - 1, 1)) / denom))


def full_metrics(trades_R=None, daily_ret: pd.Series | None = None,
                 n_trials: int = 1) -> dict:
    out = {}

    # ---------------- equity-curve family (needs daily returns) ----------------
    if daily_ret is not None:
        r = pd.Series(daily_ret).dropna()
        if len(r) >= 30:
            eq = (1 + r).cumprod()
            n = len(r)
            total = float(eq.iloc[-1] - 1)
            cagr = float(eq.iloc[-1] ** (252 / n) - 1)
            peak = eq.cummax()
            dd = eq / peak - 1
            max_dd = float(dd.min())
            out.update({
                "total_return_pct": round(total * 100, 2),
                "cagr_pct": round(cagr * 100, 2),
                "max_dd_pct": round(max_dd * 100, 2),
                "calmar": round(cagr / abs(max_dd), 2) if max_dd < 0 else None,
                "ann_vol_pct": round(float(r.std() * np.sqrt(252)) * 100, 2),
                "sharpe": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)), 2),
                "pct_time_underwater": round(float((dd < 0).mean()) * 100, 1),
                "n_days": n,
            })
            downside = r[r < 0]
            out["sortino"] = round(float(r.mean() / (downside.std() + 1e-12)
                                         * np.sqrt(252)), 2) if len(downside) > 5 else None
            # longest peak-to-recovery stretch, in trading days
            under, longest, cur = dd < -1e-12, 0, 0
            for u in under.values:
                cur = cur + 1 if u else 0
                longest = max(longest, cur)
            out["max_recovery_days"] = int(longest)
            # rolling Sharpe stability: 1y window, report mean/min and last-vs-first half
            if n >= 504:
                roll = r.rolling(252).apply(
                    lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252), raw=True).dropna()
                out["rolling_sharpe_min"] = round(float(roll.min()), 2)
                out["rolling_sharpe_frac_positive"] = round(float((roll > 0).mean()), 2)
            half = n // 2
            s1 = r.iloc[:half].mean() / (r.iloc[:half].std() + 1e-12) * np.sqrt(252)
            s2 = r.iloc[half:].mean() / (r.iloc[half:].std() + 1e-12) * np.sqrt(252)
            out["sharpe_first_half"] = round(float(s1), 2)
            out["sharpe_second_half"] = round(float(s2), 2)
            out["t_stat_daily"] = round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(n)), 2)
            out["DSR_prob_true_SR_gt_0"] = round(deflated_sharpe(r, n_trials), 4)

    # ---------------- trade-level family ----------------
    if trades_R is not None and len(trades_R) > 0:
        R = np.asarray(trades_R, dtype=float)
        R = R[~np.isnan(R)]
        wins, losses = R[R > 0], R[R <= 0]
        out.update({
            "n_trades": int(len(R)),
            "win_rate": round(float((R > 0).mean()), 4),
            "avg_win_R": round(float(wins.mean()), 3) if len(wins) else None,
            "avg_loss_R": round(float(losses.mean()), 3) if len(losses) else None,
            "win_loss_ratio": round(float(wins.mean() / -losses.mean()), 2)
                              if len(wins) and len(losses) and losses.mean() < 0 else None,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 3)
                             if losses.sum() < 0 else None,
            "expectancy_R": round(float(R.mean()), 4),
            "median_trade_R": round(float(np.median(R)), 4),
            "p5_trade_R": round(float(np.percentile(R, 5)), 3),
            "worst_trade_R": round(float(R.min()), 3),
            "t_stat_trades": round(float(R.mean() / (R.std() + 1e-12) * np.sqrt(len(R))), 2),
        })
        # max consecutive losses
        mx = cur = 0
        for x in R:
            cur = cur + 1 if x <= 0 else 0
            mx = max(mx, cur)
        out["max_consecutive_losses"] = int(mx)
        # outlier dependence: expectancy after removing the 5 best trades
        if len(R) > 20:
            trimmed = np.sort(R)[:-5]
            out["expectancy_wo_top5_R"] = round(float(trimmed.mean()), 4)
        # rolling expectancy trend: thirds of the sample, chronological
        if len(R) >= 60:
            k = len(R) // 3
            out["expectancy_thirds_R"] = [round(float(R[:k].mean()), 3),
                                          round(float(R[k:2 * k].mean()), 3),
                                          round(float(R[2 * k:].mean()), 3)]
    return out


def summarize_line(name: str, m: dict) -> str:
    """One printable line for progress logs."""
    return (f"{name:34} n={m.get('n_trades','-'):>5} exp={m.get('expectancy_R','-'):>7} "
            f"PF={m.get('profit_factor','-'):>6} shp={m.get('sharpe','-'):>6} "
            f"cal={m.get('calmar','-'):>6} dd={m.get('max_dd_pct','-'):>7} "
            f"t={m.get('t_stat_trades', m.get('t_stat_daily','-')):>6}")

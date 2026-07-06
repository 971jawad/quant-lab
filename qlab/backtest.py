"""R-multiple event backtester + equity simulation with prop-firm guardrails.

Execution model (deliberately conservative):
  - signal read at bar i close -> entry at bar i+1 OPEN, paying half-spread
    + slippage on the way in and out
  - if the entry open already gapped through the stop, the trade is SKIPPED
  - intrabar, the STOP is assumed to hit before the target (pessimistic)
  - targets are limit orders filled only if the bar STRICTLY exceeds them
  - time exit at the close of the Nth bar if neither level is reached
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    spread: float       # full bid-ask spread in price points
    slip: float         # slippage per side in price points
    comm_side: float    # commission per side expressed in price points


def bt_trades(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
              signals: pd.DataFrame, rr: float, time_exit: int,
              costs: Costs, max_concurrent: int) -> pd.DataFrame:
    n = len(o)
    half = costs.spread / 2 + costs.slip
    rows = []
    open_exits: list[int] = []          # exit bars of accepted trades
    for sig in signals.itertuples(index=False):
        i, d, stop, conv = int(sig.i), int(sig.dir), float(sig.stop), float(sig.conviction)
        e = i + 1
        if e >= n:
            continue
        open_exits = [x for x in open_exits if x > e]
        if len(open_exits) >= max_concurrent:
            continue
        entry = o[e] + d * half
        dist = (entry - stop) * d
        if dist <= 0:                   # gapped through the stop -> stand aside
            continue
        target = entry + d * rr * dist
        exit_i, exit_px, reason = None, None, None
        last = min(e + time_exit, n - 1)
        for j in range(e, last + 1):
            if d == 1:
                if l[j] <= stop:
                    exit_i, exit_px, reason = j, stop - half, "stop"
                    break
                if h[j] > target:
                    exit_i, exit_px, reason = j, target, "target"
                    break
            else:
                if h[j] >= stop:
                    exit_i, exit_px, reason = j, stop + half, "stop"
                    break
                if l[j] < target:
                    exit_i, exit_px, reason = j, target, "target"
                    break
        if exit_i is None:
            exit_i, exit_px, reason = last, c[last] - d * half, "time"
        pnl = (exit_px - entry) * d - 2 * costs.comm_side
        rows.append((i, e, exit_i, d, pnl / dist, conv, reason, dist))
        open_exits.append(exit_i)
    return pd.DataFrame(rows, columns=["sig_i", "entry_i", "exit_i", "dir",
                                       "R", "conviction", "reason", "dist"])


def simulate(trades: pd.DataFrame, et_date: np.ndarray, risk_pct: float,
             conviction_scale: bool = False, daily_cap: float = 0.03,
             trail_limit: float = 0.05, start_eq: float = 1.0) -> dict:
    """Chronological equity sim. Risk is committed at entry (fraction of
    equity THEN), PnL realized at exit. New entries are blocked for the rest
    of an ET day once the realized day loss exceeds daily_cap (prop-firm
    daily loss rule). Trailing-drawdown breaches are counted, not halted,
    so the report shows how often a prop account would have died."""
    if trades.empty:
        return {"n_trades": 0}
    ev = []
    for seq, t in enumerate(trades.itertuples(index=False)):
        # exits of other trades process before new entries on the same bar,
        # but a zero-duration trade must still run entry -> exit
        ev.append(((t.entry_i, 1, seq), 1, t))
        exit_order = 2 if t.exit_i == t.entry_i else 0
        ev.append(((t.exit_i, exit_order, seq), 0, t))
    ev.sort(key=lambda x: x[0])
    ev = [(k[0], kind, t) for k, kind, t in ev]

    eq, peak = start_eq, start_eq
    day, day_start = None, start_eq
    blocked = False
    open_risk: dict[int, float] = {}
    daily_eq: dict[object, float] = {}
    in_breach, breaches, taken = False, 0, 0
    skipped_by_cap = 0
    eq_path = [start_eq]
    taken_R: list[float] = []           # executed trades only

    for bar, kind, t in ev:
        d = et_date[bar]
        if d != day:
            day, day_start, blocked = d, eq, False
        if kind == 1:
            if blocked:
                skipped_by_cap += 1
                continue
            conv = float(t.conviction) if conviction_scale else 1.0
            rk = float(getattr(t, "risk_pct", risk_pct))   # per-fold risk if present
            if np.isnan(rk):
                rk = risk_pct
            open_risk[id(t)] = eq * rk * conv
            taken += 1
        else:
            amt = open_risk.pop(id(t), None)
            if amt is None:
                continue                # entry was blocked -> no position
            eq += amt * float(t.R)
            eq_path.append(eq)
            taken_R.append(float(t.R))
            if (eq - day_start) / day_start <= -daily_cap:
                blocked = True
            peak = max(peak, eq)
            if (eq / peak - 1) <= -trail_limit:
                if not in_breach:
                    breaches += 1
                in_breach = True
            else:
                in_breach = False
        daily_eq[d] = eq

    if not taken_R:
        return {"n_trades": int(len(trades)), "n_taken": 0}
    eq_arr = np.array(eq_path)
    run_max = np.maximum.accumulate(eq_arr)
    max_dd = float(((eq_arr - run_max) / run_max).min())
    # equity marked on EVERY trading day in span (not only event days) so
    # Sharpe/annualization aren't flattered by sparse trading
    ev_days = pd.Series(daily_eq).sort_index()
    all_days = pd.Index(sorted(set(et_date)))
    span = all_days[(all_days >= ev_days.index[0]) & (all_days <= ev_days.index[-1])]
    ds = ev_days.reindex(span).ffill()
    dret = ds.pct_change().dropna()
    n_days = max(len(ds), 2)
    r_all = np.array(taken_R)
    wins = r_all[r_all > 0]
    losses = r_all[r_all <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    return {
        "n_trades": int(len(trades)), "n_taken": taken,
        "skipped_by_daily_cap": skipped_by_cap,
        "win_rate": round(float((r_all > 0).mean()), 4),
        "avg_R": round(float(r_all.mean()), 4),
        "t_stat": round(float(r_all.mean() / (r_all.std() + 1e-12) * np.sqrt(len(r_all))), 2),
        "profit_factor": round(pf, 3),
        "total_return_pct": round((eq / start_eq - 1) * 100, 2),
        "ann_return_pct": round(((eq / start_eq) ** (252 / n_days) - 1) * 100, 2),
        "sharpe": round(float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(252)), 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "worst_day_pct": round(float(dret.min() * 100) if len(dret) else 0.0, 2),
        "trailing_dd_breaches": breaches,
        "oos_days": int(n_days),
    }


def train_objective(r: np.ndarray, min_trades: int = 25) -> float:
    """Trade-level t-stat, the in-sample selection criterion. Requires a
    minimum sample so one lucky outlier can't win the fold."""
    if len(r) < min_trades:
        return -np.inf
    return float(r.mean() / (r.std() + 1e-12) * np.sqrt(len(r)))

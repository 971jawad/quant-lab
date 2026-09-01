"""Build and evaluate the RULE+ML+AI ensemble from the walk-forward OOS trades.

Produces (in results3/):
  ensemble_summary.csv   - per-instrument SELECTIVE vs ALWAYS-ON, plus the GLOBAL
                           cross-instrument book; OOS Sharpe/Calmar/t/DD.
  ensemble_weights.json  - per-fold picks and weights (audit trail).
  trades_ensemble_{inst}.csv, trades_ensemble_GLOBAL.csv
It reads only results2/trades_*.csv (all out-of-sample) - never price data for
signals - so nothing here can leak.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.walkforward as WF
from qlab.ensemble import (APPROACHES, STYLES, build_instrument_ensemble,
                           simulate_instrument, to_global_book)
from qlab.backtest import simulate

ROOT = Path(__file__).parent
DATA, RESULTS, OUT = ROOT / "data", ROOT / "results2", ROOT / "results3"
DATA_MAP = {"MES": "SPXUSD", "ES": "SPXUSD", "MNQ": "NSXUSD",
            "XAUUSD": "XAUUSD", "EURUSD": "EURUSD"}
SCALE = {"mult": 4, "min_train": 70000, "test_len": 23000, "ml_max_train": 150000}
ALL_FAMS = [f for fams in APPROACHES.values() for f in fams]
KEEP = ("n_trades", "n_taken", "win_rate", "avg_R", "t_stat", "profit_factor",
        "total_return_pct", "ann_return_pct", "sharpe", "calmar", "max_dd_pct",
        "trailing_dd_breaches", "oos_days")


def load_index(series: str) -> pd.DatetimeIndex:
    df = pd.read_csv(DATA / f"{series}_15m.csv", index_col=0, usecols=[0])
    return pd.to_datetime(df.index, utc=True)


def main():
    WF.set_scale(**SCALE)
    OUT.mkdir(exist_ok=True)
    idx_cache = {}
    rows, weights_out = [], {}
    global_books = []

    for inst, series in DATA_MAP.items():
        if series not in idx_cache:
            idx_cache[series] = load_index(series)
        idx = idx_cache[series]
        folds = WF.make_folds(len(idx))
        legs = {}
        for fam in ALL_FAMS:
            for st in STYLES:
                tag = f"{inst}_{fam}_{st}"
                p = RESULTS / f"trades_{tag}.csv"
                if p.exists():
                    tr = pd.read_csv(p)
                    if not tr.empty:
                        legs[tag] = tr.sort_values("sig_i").reset_index(drop=True)
        n_ai = sum(1 for t in legs if t.startswith(f"{inst}_ai_"))
        if n_ai == 0:
            print(f"[{inst}] WARNING: no AI legs found yet - run run_ai.py first", flush=True)

        sel = build_instrument_ensemble(inst, legs, folds, selective=True)
        alw = build_instrument_ensemble(inst, legs, folds, selective=False)
        m_sel = simulate_instrument(sel.trades, idx)
        m_alw = simulate_instrument(alw.trades, idx)
        for name, m in (("selective", m_sel), ("always_on", m_alw)):
            row = {"scope": inst, "mode": name, **{k: m.get(k) for k in KEEP}}
            rows.append(row)
        weights_out[inst] = sel.weight_log
        sel.trades.to_csv(OUT / f"trades_ensemble_{inst}.csv", index=False)
        if not sel.trades.empty:
            global_books.append((sel.trades, idx))
        print(f"[{inst}] selective: n={m_sel.get('n_taken',0)} "
              f"t={m_sel.get('t_stat')} sharpe={m_sel.get('sharpe')} "
              f"calmar={m_sel.get('calmar')} ret={m_sel.get('total_return_pct')}% "
              f"dd={m_sel.get('max_dd_pct')}%  |  always-on: "
              f"t={m_alw.get('t_stat')} sharpe={m_alw.get('sharpe')} "
              f"ret={m_alw.get('total_return_pct')}%", flush=True)

    # global cross-instrument book (selective legs only)
    gbook, get_date = to_global_book(global_books)
    if len(gbook):
        risk0 = float(gbook["risk_pct"].iloc[-1])
        gm = simulate(gbook, get_date, risk0, conviction_scale=True,
                      daily_cap=0.03, trail_limit=0.05)
        rows.append({"scope": "GLOBAL", "mode": "selective",
                     **{k: gm.get(k) for k in KEEP}})
        gbook.to_csv(OUT / "trades_ensemble_GLOBAL.csv", index=False)
        print(f"[GLOBAL] n={gm.get('n_taken',0)} t={gm.get('t_stat')} "
              f"sharpe={gm.get('sharpe')} calmar={gm.get('calmar')} "
              f"ret={gm.get('total_return_pct')}% dd={gm.get('max_dd_pct')}%", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "ensemble_summary.csv", index=False)
    with open(OUT / "ensemble_weights.json", "w") as fh:
        json.dump(weights_out, fh, indent=2, default=str)
    print(f"\nWrote {OUT}/ensemble_summary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()

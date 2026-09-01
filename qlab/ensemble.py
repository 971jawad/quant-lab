"""Leak-free ensemble of the three approaches: RULE (smc/ta), ML (ml/ml_err/
ml_rec) and AI (neural net). It combines the per-leg walk-forward OOS trade
streams into one portfolio - it never re-reads price data for signals, only the
already-out-of-sample trades each leg produced.

Why an ensemble can (only) help here
------------------------------------
You cannot average a positive edge out of negative-expectancy legs. The single
honest lever an ensemble owns is CHOOSING WHEN TO STAND ASIDE. So the policy is:

  For each walk-forward fold, look ONLY at trades that had already CLOSED before
  the fold began (realized, knowable history). For each approach pick its own
  best concrete (family,style) by trailing t-stat, keep the approaches whose
  trailing average R is > 0, inverse-variance weight them, and if none clears
  zero the fold is held in CASH.

Every weight for fold k is a function of folds < k only -> no look-ahead. The
first fold has no closed history and is always flat.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import simulate

STYLES = ("A", "B", "C")
APPROACHES = {
    "RULE": ("smc", "ta"),
    "ML": ("ml", "ml_err", "ml_rec"),
    "AI": ("ai",),
}
MIN_TRAIL_TRADES = 30      # trust a leg's trailing record only past this many
DAILY_CAP = 0.03
TRAIL_LIMIT = 0.05


@dataclass
class EnsembleResult:
    trades: pd.DataFrame                 # combined book (instrument bar indices)
    weight_log: list = field(default_factory=list)
    picks: list = field(default_factory=list)


def load_leg(results_dir, tag: str) -> pd.DataFrame | None:
    p = results_dir / f"trades_{tag}.csv"
    if not p.exists():
        return None
    tr = pd.read_csv(p)
    if tr.empty or "exit_i" not in tr.columns:
        return None
    return tr.sort_values("sig_i").reset_index(drop=True)


def _trailing_stats(tr: pd.DataFrame, before_i: int) -> dict | None:
    """Record of trades that ENTERED and fully CLOSED before bar `before_i`."""
    r = tr.loc[tr["exit_i"] < before_i, "R"].values
    if len(r) < MIN_TRAIL_TRADES:
        return None
    sd = r.std()
    if not np.isfinite(sd) or sd <= 0:
        return None
    return {"n": len(r), "avg_R": float(r.mean()), "sd": float(sd),
            "t": float(r.mean() / sd * np.sqrt(len(r))), "inv_var": 1.0 / (sd * sd)}


def build_instrument_ensemble(inst: str, legs: dict, folds: list,
                              selective: bool = True) -> EnsembleResult:
    """legs: {tag: trades_df}. folds: [(train_end, test_end)] from make_folds.
    selective=True -> keep only approaches with trailing avg_R>0, else cash.
    selective=False -> always-on equal-weight benchmark (best style per approach
    by trailing t, no positivity gate)."""
    parts, wlog, picks = [], [], []
    for train_end, test_end in folds:
        chosen = {}                      # approach -> (tag, stats)
        for appr, fams in APPROACHES.items():
            best = None
            for fam in fams:
                for st in STYLES:
                    tag = f"{inst}_{fam}_{st}"
                    tr = legs.get(tag)
                    if tr is None:
                        continue
                    s = _trailing_stats(tr, train_end)
                    if s is None:
                        continue
                    if best is None or s["t"] > best[1]["t"]:
                        best = (tag, s)
            if best is not None:
                chosen[appr] = best

        if selective:
            kept = {a: v for a, v in chosen.items() if v[1]["avg_R"] > 0}
        else:
            kept = chosen
        wl = {"train_end": int(train_end), "test_end": int(test_end)}
        if not kept:
            wl["state"] = "cash"
            wlog.append(wl)
            continue
        wsum = sum(v[1]["inv_var"] for v in kept.values())
        for appr, (tag, s) in kept.items():
            w = s["inv_var"] / wsum
            fam, st = tag[len(inst) + 1:].rsplit("_", 1)
            seg = legs[tag]
            m = (seg["sig_i"] >= train_end) & (seg["sig_i"] < test_end)
            part = seg[m].copy()
            if part.empty:
                continue
            part["risk_pct"] = part["risk_pct"].astype(float) * w
            part["leg"] = f"{appr}:{fam}_{st}"
            parts.append(part)
            wl[appr] = {"pick": f"{fam}_{st}", "w": round(w, 3),
                        "trail_avgR": round(s["avg_R"], 4), "trail_t": round(s["t"], 2)}
            picks.append({"train_end": int(train_end), "approach": appr,
                          "pick": f"{fam}_{st}", "w": round(float(w), 4)})
        wlog.append(wl)

    if parts:
        book = pd.concat(parts).sort_values("entry_i").reset_index(drop=True)
    else:
        book = pd.DataFrame(columns=list(next(iter(legs.values())).columns) + ["leg"])
    return EnsembleResult(book, wlog, picks)


def simulate_instrument(book: pd.DataFrame, series_index: pd.DatetimeIndex) -> dict:
    if book.empty:
        return {"n_trades": 0}
    et_date = series_index.tz_convert("America/New_York").date
    risk0 = float(book["risk_pct"].iloc[-1]) if "risk_pct" in book else 0.005
    return simulate(book, et_date, risk0, conviction_scale=True,
                    daily_cap=DAILY_CAP, trail_limit=TRAIL_LIMIT)


# ------------------------------------------------------- global cross-instrument

def to_global_book(books: list[tuple[pd.DataFrame, pd.DatetimeIndex]]
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge per-instrument books that use their OWN bar indices into one book on
    a shared timeline. Each event timestamp becomes a global integer position, so
    backtest.simulate can run the combined portfolio under one daily loss cap and
    one trailing-DD counter."""
    frames = []
    for book, idx in books:
        if book.empty:
            continue
        b = book.copy()
        b["entry_ts"] = idx[b["entry_i"].values]
        b["exit_ts"] = idx[b["exit_i"].values]
        frames.append(b)
    if not frames:
        return pd.DataFrame(), np.array([])
    allb = pd.concat(frames, ignore_index=True)
    # tz-preserving unique timeline (np.union1d would strip tz and break the map)
    stamps = pd.DatetimeIndex(
        pd.unique(pd.concat([allb["entry_ts"], allb["exit_ts"]]))).sort_values()
    pos = pd.Series(np.arange(len(stamps)), index=stamps)
    allb["entry_i"] = allb["entry_ts"].map(pos).astype(int)
    allb["exit_i"] = allb["exit_ts"].map(pos).astype(int)
    allb = allb.sort_values("entry_i").reset_index(drop=True)
    et_date = np.array(stamps.tz_convert("America/New_York").date)
    return allb, et_date

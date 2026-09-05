"""Build the GitHub Pages dashboard payload from the research artifacts.

Produces docs/data.json consumed by docs/index.html. Run after any refresh:
    python run_ensembler.py && python build_site.py

This publishes RESEARCH SIGNALS ONLY. It does not connect to a broker and does
not place orders.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RES, DOCS = ROOT / "research", ROOT / "docs"
DEV = pd.Timestamp("2022-07-01")


def load_curve(name):
    p = RES / name
    if not p.exists():
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["ret"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def curve_points(s, step=5):
    eq = (1 + s).cumprod()
    eq = eq.iloc[::step]
    return [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 4)}
            for d, v in eq.items()]


def metrics(s):
    if s is None or len(s) < 30:
        return {}
    eq = (1 + s).cumprod()
    dd = eq / eq.cummax() - 1
    n = len(s)
    cagr = float(eq.iloc[-1] ** (252 / n) - 1)
    sharpe = float(s.mean() / (s.std() + 1e-12) * np.sqrt(252))
    down = s[s < 0]
    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(float(s.mean() / (down.std() + 1e-12) * np.sqrt(252)), 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "calmar": round(cagr / abs(float(dd.min())), 2) if dd.min() < 0 else None,
        "vol_pct": round(float(s.std() * np.sqrt(252)) * 100, 2),
        "pct_underwater": round(float((dd < 0).mean()) * 100, 1),
        "days": n,
    }


def jload(name, default=None):
    p = RES / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def main():
    DOCS.mkdir(exist_ok=True)
    ens = load_curve("ensembler_daily.csv")
    champ = load_curve("champion_v12_daily.csv")
    naive = load_curve("naive_tsmom_daily.csv")

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "positions": jload("ensembler_positions.json", {}),
        "trades": jload("trade_details.json", {}),
        "verification": jload("verification.json", {}),
        "decay": jload("decay_monitor.json", {}),
        "live_feed": jload("live_feed.json", {}),
        "metrics": {
            "ensembler_full": metrics(ens),
            "ensembler_dev": metrics(ens[ens.index < DEV]) if ens is not None else {},
            "ensembler_holdout": metrics(ens[ens.index >= DEV]) if ens is not None else {},
            "champion_holdout": metrics(champ[champ.index >= DEV]) if champ is not None else {},
            "naive_holdout": metrics(naive[naive.index >= DEV]) if naive is not None else {},
        },
        "curves": {
            "ensembler": curve_points(ens) if ens is not None else [],
            "naive": curve_points(naive) if naive is not None else [],
        },
        "capstone": jload("capstone_meta.json", {}),
        "attribution": jload("attribution.json", {}),
        "committee": jload("committee.json", {}),
        "leg_alpha": jload("leg_alpha.json", {}),
        "shorter": jload("shorter.json", {}),
        "backward": jload("backward_validation.json", {}),
        "sizing": [{"scale": 0.5, "maxdd": -5.4, "cagr": 4.75, "pass": 28.1, "breach": 6.2}, {"scale": 0.75, "maxdd": -8.0, "cagr": 7.16, "pass": 50.9, "breach": 21.4}, {"scale": 1.0, "maxdd": -10.53, "cagr": 9.6, "pass": 60.5, "breach": 30.9}, {"scale": 1.25, "maxdd": -13.0, "cagr": 12.07, "pass": 59.7, "breach": 38.0}, {"scale": 1.5, "maxdd": -15.4, "cagr": 14.56, "pass": 56.8, "breach": 42.6}, {"scale": 2.0, "maxdd": -20.02, "cagr": 19.61, "pass": 52.0, "breach": 47.9}],
        "tested": [
            {"family": "Daily trend (multi-market)", "verdict": "ADMITTED",
             "note": "core of the book; ~40% is generic trend beta"},
            {"family": "COT washout (Nasdaq)", "verdict": "ADMITTED",
             "note": "orthogonal alpha t=2.32, but FAILS 1999-2009 backward test"},
            {"family": "Trend-pullback (Nasdaq)", "verdict": "ADMITTED",
             "note": "orthogonal alpha t=3.14"},
            {"family": "Cross-asset momentum", "verdict": "ADMITTED", "note": ""},
            {"family": "ICT / SMC fib-zone", "verdict": "REJECTED",
             "note": "negative over 16y on gold and Nasdaq"},
            {"family": "ICT 'Muso' funded-trader variant", "verdict": "REJECTED",
             "note": "negative even at optimistic ECN costs"},
            {"family": "Intraday 15m / 1h (all families)", "verdict": "REJECTED",
             "note": "3% and 12% of models positive; cost-dominated"},
            {"family": "ML + neural nets (cross-asset features)", "verdict": "REJECTED",
             "note": "0 of 18 positive"},
            {"family": "Volume / auction gating", "verdict": "REJECTED",
             "note": "gating DESTROYS trend edge"},
            {"family": "FX carry", "verdict": "REJECTED", "note": "post-GFC decay"},
            {"family": "VIX term-structure gating", "verdict": "REJECTED",
             "note": "worse than staying long"},
            {"family": "Turtle / Turtle Soup / Holy Grail", "verdict": "REJECTED",
             "note": "decayed"},
            {"family": "LW volatility breakout", "verdict": "REJECTED",
             "note": "t=8 was an OHLC fill artifact; true +0.04R"},
            {"family": "Bonds / ETF breadth", "verdict": "REJECTED",
             "note": "dev +1.4% became holdout -8.5%"},
            {"family": "Dedicated SHORT models", "verdict": "REJECTED",
             "note": "short side measured at -108%"},
            {"family": "The Committee (all schools voting)", "verdict": "REJECTED",
             "note": "0.29 vs 0.45 for trend alone"},
        ],
    }
    (DOCS / "data.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote docs/data.json  ({len(json.dumps(payload))} bytes)")
    print(f"  positions: {len(payload['positions'].get('positions', []))}")
    print(f"  curve points: {len(payload['curves']['ensembler'])}")


if __name__ == "__main__":
    main()

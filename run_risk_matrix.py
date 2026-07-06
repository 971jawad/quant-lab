"""Prop-firm risk matrix: contract sizing per instrument, account size and
risk level, derived from current ATR-based stop distances. Also sanity-checks
each combination against typical prop-firm daily loss / trailing drawdown
rules (Topstep/Apex/FTMO-style)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.features import atr

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

INSTRUMENTS = {
    "MES":    {"data": "ES", "point_value": 5.0},
    "ES":     {"data": "ES", "point_value": 50.0},
    "MNQ":    {"data": "NQ", "point_value": 2.0},
    "XAUUSD": {"data": "GC", "point_value": 100.0},   # per 1-lot (100 oz)
    "EURUSD": {"data": "EURUSD", "point_value": 100000.0},  # per standard lot
}
ACCOUNTS = [25_000, 50_000, 100_000]
RISK_LEVELS = [0.005, 0.0075, 0.01]
STOP_ATR = 1.5          # representative structure stop (median across models)

# typical evaluation rules (verify against your specific firm before use)
PROP_RULES = {
    "topstep_50k":  {"daily_loss": 1000, "trailing_dd": 2000},
    "apex_50k":     {"daily_loss": None, "trailing_dd": 2500},
    "ftmo_100k":    {"daily_loss": 5000, "max_dd": 10000},
}


def main():
    rows = []
    for inst, spec in INSTRUMENTS.items():
        df = pd.read_csv(DATA / f"{spec['data']}_1h.csv", index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        a = atr(df, 14).iloc[-250:]            # last ~2 weeks of hourly ATR
        med_atr = float(a.median())
        stop_pts = STOP_ATR * med_atr
        stop_usd_per_ct = stop_pts * spec["point_value"]
        for acct in ACCOUNTS:
            for rk in RISK_LEVELS:
                risk_usd = acct * rk
                contracts = int(risk_usd // stop_usd_per_ct)
                rows.append({
                    "instrument": inst, "account_usd": acct,
                    "risk_pct": rk * 100,
                    "median_atr_1h_pts": round(med_atr, 4),
                    "stop_pts(1.5xATR)": round(stop_pts, 4),
                    "risk_per_contract_usd": round(stop_usd_per_ct, 2),
                    "contracts": contracts,
                    "losses_to_daily_cap_3pct": int(np.ceil(0.03 / rk)),
                    "losses_to_trail_5pct": int(np.ceil(0.05 / rk)),
                })
    m = pd.DataFrame(rows)
    m.to_csv(RESULTS / "risk_matrix.csv", index=False)
    with open(RESULTS / "prop_rules_reference.json", "w") as fh:
        json.dump(PROP_RULES, fh, indent=2)
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()

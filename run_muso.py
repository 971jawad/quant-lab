"""Cycle 4b — the "Muso" ICT variant, mechanized exactly as described.

Source: funded-trader interview (The5ers, $200K). Stated rules:
  * DXY on higher timeframes (D/4H/1H) sets directional BIAS first
  * then drop to 1m/5m for entry
  * entry = Fair Value Gap aligned with a Fibonacci zone
  * 1% risk, ~10 pip stop, move to BREAK-EVEN once +1R
  * instruments: EURUSD, GBPUSD, sometimes Gold
  * SESSION CIRCUIT-BREAKER: one loss in a session (London or NY) -> stop
    trading that session

Two components here are genuinely NEW vs the already-falsified `ict` family:
  (1) DXY-derived bias (we previously used SMA200 on the instrument itself)
  (2) the session circuit-breaker

Honest limits, stated before results:
  - finest bars we hold are 15m, not 1m/5m. A 10-pip stop is NOT representable
    at 15m; we use the structural stop (beyond the swept extreme), which is
    strictly MORE forgiving of noise than a 10-pip stop.
  - synthetic DXY is built from EURUSD/GBPUSD/USDJPY with renormalized ICE
    weights (83% of the real index) using only COMPLETED daily bars.
No parameter grids: his rules as stated. Dev window only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlab.strategies as S
from qlab.backtest import Costs, bt_trades
from qlab.features import build_features
from qlab.metrics import full_metrics
from run_research import load_15m, to_tf

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV = pd.Timestamp("2022-07-01", tz="UTC")
# ICE DXY weights, renormalized over the three legs we can build (83% of index)
DXY_W = {"EURUSD": -0.576, "GBPUSD": -0.119, "USDJPY": 0.136}
COSTS = {"EURUSD": Costs(0.00006, 0.00002, 0.00004),
         "GBPUSD": Costs(0.00008, 0.00003, 0.00004),
         "XAUUSD": Costs(0.25, 0.10, 0.0)}
# optimistic retail-ECN sensitivity (NOT the headline number)
COSTS_TIGHT = {"EURUSD": Costs(0.00002, 0.00001, 0.00002),
               "GBPUSD": Costs(0.00003, 0.00001, 0.00002),
               "XAUUSD": Costs(0.10, 0.05, 0.0)}


def synth_dxy_daily():
    """Synthetic DXY from completed daily closes; geometric, ICE-style."""
    parts = {}
    avail = {k: v for k, v in DXY_W.items()
             if (ROOT / f"data/{k}_15m.csv").exists()}
    print(f"  [DXY built from {list(avail)} = "
          f"{sum(abs(v) for v in avail.values())/0.831:.0%} of the index]")
    for pair, w in avail.items():
        d = to_tf(load_15m(pair), "1d")["close"]
        d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
        parts[pair] = np.log(d) * (w / sum(abs(v) for v in avail.values()))
    dxy = np.exp(pd.DataFrame(parts).dropna().sum(axis=1))
    return dxy.rename("dxy")


def dxy_bias(dxy: pd.Series) -> pd.Series:
    """+1 = dollar WEAK (long EUR/GBP/Gold), -1 = dollar STRONG.
    Uses the previous completed day only -> no lookahead."""
    trend = np.sign(dxy / dxy.rolling(20).mean() - 1)
    return (-trend).shift(1)          # dollar down => long the anti-dollar asset


def muso_signals(f: pd.DataFrame, bias_by_day: pd.Series) -> pd.DataFrame:
    """ict fib-zone entries, but the HTF filter is the DXY bias, not SMA200."""
    p = {"disp_k": 1.0, "use_ha": False, "bias": False, "manage": "managed"}
    sig = S.ict_signals(f, p)
    if sig.empty:
        return sig
    days = f.index[sig["i"].values].tz_convert("America/New_York").normalize().tz_localize(None)
    b = bias_by_day.reindex(days).values
    keep = (np.sign(sig["dir"].values) == np.sign(b)) & ~np.isnan(b)
    return sig[keep].reset_index(drop=True)


def apply_circuit_breaker(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Drop any entry in a (ET-day, session) where an earlier trade already lost."""
    if trades.empty:
        return trades
    et = bars.index.tz_convert("America/New_York")
    ent = et[trades["entry_i"].values]
    sess = np.where((ent.hour >= 2) & (ent.hour < 5), "LDN",
                    np.where((ent.hour >= 7) & (ent.hour < 11), "NY", "OTHER"))
    key = pd.Series([f"{d}_{s}" for d, s in zip(ent.date, sess)])
    keep, blocked = [], set()
    for i, (k, r) in enumerate(zip(key, trades["R"].values)):
        if k in blocked:
            keep.append(False)
            continue
        keep.append(True)
        if r <= 0:
            blocked.add(k)
    return trades[np.array(keep)].reset_index(drop=True)


def run(inst, costs, label, breaker=True):
    bars = to_tf(load_15m(inst), "15m")
    bars = bars[bars.index < DEV]
    f = build_features(bars)
    dxy = synth_dxy_daily()
    bias = dxy_bias(dxy)
    sig = muso_signals(f, bias)
    if sig.empty:
        print(f"  {inst} {label}: no signals")
        return None
    o, h, l, c = (bars[k].values for k in ("open", "high", "low", "close"))
    tr = bt_trades(o, h, l, c, sig, rr=2.0, time_exit=96, costs=costs, max_concurrent=1)
    if breaker:
        tr = apply_circuit_breaker(tr, bars)
    m = full_metrics(tr["R"].values) if len(tr) else {}
    print(f"  {inst:7} {label:16} n={m.get('n_trades',0):4d} "
          f"exp={m.get('expectancy_R','-'):>7} PF={m.get('profit_factor','-'):>6} "
          f"wr={m.get('win_rate','-'):>6} t={m.get('t_stat_trades','-'):>6}")
    return m


if __name__ == "__main__":
    S.set_scale(4)
    res = {}
    print("MUSO variant (DXY bias + FVG/fib + BE + circuit-breaker), 15m, DEV:")
    print(" -- headline: conservative costs --")
    for inst in ("EURUSD", "GBPUSD", "XAUUSD"):
        if not (ROOT / f"data/{inst}_15m.csv").exists():
            print(f"  {inst}: data not present yet")
            continue
        res[f"{inst}_std"] = run(inst, COSTS[inst], "std-cost")
    print(" -- sensitivity: optimistic retail-ECN costs --")
    for inst in ("EURUSD", "GBPUSD", "XAUUSD"):
        if not (ROOT / f"data/{inst}_15m.csv").exists():
            continue
        res[f"{inst}_tight"] = run(inst, COSTS_TIGHT[inst], "tight-cost")
    print(" -- ablation: circuit-breaker OFF (std cost) --")
    for inst in ("EURUSD", "GBPUSD", "XAUUSD"):
        if not (ROOT / f"data/{inst}_15m.csv").exists():
            continue
        res[f"{inst}_nobreak"] = run(inst, COSTS[inst], "no-breaker", breaker=False)
    with open(OUT / "muso_dev.json", "w") as fp:
        json.dump(res, fp, indent=2, default=str)
    with open(OUT / "ledger.jsonl", "a") as fp:
        fp.write(json.dumps({"phase": "muso_ict", "n_configs": 9}) + "\n")

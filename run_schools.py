"""Additional schools of thought — declared, canonical, hypothesis-driven.

  bnf_dip   BNF/Kotegawa deep-panic reversion, translated to CFDs: long when
            close < 25d MA - 2.5 x 25d sigma; exit at MA touch or 5 days.
            Long-only (his documented book was long panic dips).
  ichimoku  Japanese systematic school, canonical 9/26/52: long on TK cross
            above the cloud, short on TK cross below; exit on opposite cross.
  eur_cont  OUR OWN triple-confirmed finding inverted into a positive claim:
            EURUSD continues through extremes. Buy RSI(14)>70 / sell RSI<30,
            hold 10 days. (Mean-reversion there failed 3 independent ways.)

Daily bars, dev window unless --final; every trial ledgered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.features import rsi
from qlab.metrics import full_metrics, summarize_line
from run_research import SERIES, load_15m, to_tf

ROOT = Path(__file__).parent
OUT = ROOT / "research"
DEV_END = pd.Timestamp("2022-07-01", tz="UTC")
COSTS_PTS = {"XAUUSD": 0.45, "MNQ": 1.62, "EURUSD": 0.00016}
RISK = 0.01


def bnf_dip(d):
    c, o, l = d["close"].values, d["open"].values, d["low"].values
    ma = d["close"].rolling(25).mean()
    sd = d["close"].rolling(25).std()
    trig = (d["close"] < ma - 2.5 * sd).values
    mav = ma.values
    trades, i = [], 26
    while i < len(d) - 6:
        if trig[i]:
            entry = o[i + 1]
            stop = entry - 2 * sd.values[i]          # 2-sigma disaster stop
            risk = entry - stop
            done = False
            for j in range(i + 1, min(i + 6, len(d))):
                if l[j] <= stop:
                    trades.append((j, (min(o[j], stop) - entry) / risk)); done = True
                    break
                if c[j] >= mav[j]:                   # mean touched -> take it
                    trades.append((j, (c[j] - entry) / risk)); done = True
                    break
            if not done:
                j = min(i + 5, len(d) - 1)
                trades.append((j, (c[j] - entry) / risk))
            i = j + 1
            continue
        i += 1
    return trades


def ichimoku(d):
    h9 = d["high"].rolling(9).max(); l9 = d["low"].rolling(9).min()
    h26 = d["high"].rolling(26).max(); l26 = d["low"].rolling(26).min()
    h52 = d["high"].rolling(52).max(); l52 = d["low"].rolling(52).min()
    tenkan, kijun = (h9 + l9) / 2, (h26 + l26) / 2
    spanA = ((tenkan + kijun) / 2).shift(26)
    spanB = ((h52 + l52) / 2).shift(26)
    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    cloud_bot = pd.concat([spanA, spanB], axis=1).min(axis=1)
    cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1)) & (d["close"] > cloud_top)
    cross_dn = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1)) & (d["close"] < cloud_bot)
    o, c = d["open"].values, d["close"].values
    atr = (d["high"] - d["low"]).rolling(20).mean().values
    up, dn = cross_up.fillna(False).values, cross_dn.fillna(False).values
    tk_dn = (tenkan < kijun).fillna(False).values
    tk_up = (tenkan > kijun).fillna(False).values
    trades, pos, entry_px, entry_atr = [], 0, 0.0, 1.0
    for i in range(80, len(d) - 1):
        if pos == 1 and tk_dn[i]:
            trades.append((i + 1, (o[i + 1] - entry_px) / entry_atr)); pos = 0
        elif pos == -1 and tk_up[i]:
            trades.append((i + 1, (entry_px - o[i + 1]) / entry_atr)); pos = 0
        if pos == 0:
            if up[i]:
                pos, entry_px, entry_atr = 1, o[i + 1], max(atr[i], 1e-9)
            elif dn[i]:
                pos, entry_px, entry_atr = -1, o[i + 1], max(atr[i], 1e-9)
    return trades


def eur_cont(d):
    r = rsi(d["close"], 14)
    o, c = d["open"].values, d["close"].values
    atr = (d["high"] - d["low"]).rolling(20).mean().values
    hotu = (r > 70) & (r.shift(1) <= 70)
    hotd = (r < 30) & (r.shift(1) >= 30)
    trades, i = [], 21
    sigs = np.where(hotu.fillna(False).values, 1,
                    np.where(hotd.fillna(False).values, -1, 0))
    while i < len(d) - 11:
        s = sigs[i]
        if s != 0:
            entry = o[i + 1]
            j = min(i + 10, len(d) - 1)
            trades.append((j, s * (c[j] - entry) / max(atr[i], 1e-9)))
            i = j
        i += 1
    return trades


SYSTEMS = {"bnf_dip": bnf_dip, "ichimoku": ichimoku, "eur_cont": eur_cont}
APPLIES = {"bnf_dip": ("MNQ", "XAUUSD", "EURUSD"),
           "ichimoku": ("MNQ", "XAUUSD", "EURUSD"),
           "eur_cont": ("EURUSD",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    rows = []
    for name, fn in SYSTEMS.items():
        for inst in APPLIES[name]:
            d = to_tf(load_15m(SERIES[inst]), "1d")
            d = d[d.index >= DEV_END] if args.final else d[d.index < DEV_END]
            trades = fn(d)
            if not trades:
                continue
            atr20 = (d["high"] - d["low"]).rolling(20).mean()
            Rs, daily = [], pd.Series(0.0, index=d.index)
            for exit_i, R in trades:
                Rn = R - COSTS_PTS[inst] / max(atr20.iloc[exit_i], 1e-9)
                Rs.append(Rn)
                daily.iloc[exit_i] += RISK * Rn
            m = full_metrics(np.array(Rs), daily)
            tag = f"{inst}_{name}"
            m["model"] = tag
            rows.append(m)
            print(summarize_line(tag, m))
            if not args.final:
                with open(OUT / "ledger.jsonl", "a") as fh:
                    fh.write(json.dumps({"phase": "schools", "family": tag,
                                         "n_configs": 1,
                                         "dev_t": m.get("t_stat_trades"),
                                         "dev_sharpe": m.get("sharpe")}) + "\n")
    tag = "holdout" if args.final else "dev"
    pd.DataFrame(rows).to_csv(OUT / f"summary_schools_{tag}.csv", index=False)


if __name__ == "__main__":
    main()

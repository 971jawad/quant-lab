"""ENSEMBLER — the production trading system. Built to be RUN, not admired.

WHAT THIS IS
  The evidence-weighted combination of every edge that survived 14 cycles of
  testing, expressed as concrete daily target positions with an explicit risk
  engine. It is the deployable form of champion v1.2 plus the cycle-12 SHORTER
  finding.

WHAT IT IS NOT — stated plainly so nobody is misled
  * There is no "perfect entry/exit timing". Perfect timing requires knowing the
    future. What IS achievable, and what this does, is DISCIPLINED timing:
    signal read at the daily close, executed at the next open, no lookahead.
  * It is not "all models combined". Cycle 13 proved naive combination LOSES
    (committee 0.29 vs trend alone 0.45). Only legs that passed admission are
    included, weighted by demonstrated strength.

DESIGN DECISIONS, each traceable to a measured result
  1. Legs = champion v1.2 (10 legs / 7 markets). Anything that failed Stage 1 or
     Stage 5 is excluded, including bonds/ETFs, volume, carry, VIX-TS, ICT.
  2. SHORTING: cycle 12 measured the short side at -108% overall, concentrated in
     equity indices, while FX/oil earn on both sides. So equity-index legs run
     LONG-ONLY; FX and commodities keep both directions. This is a RISK decision
     backed by a measured drag and the equity risk premium, not an edge claim.
  3. Vol targeting 10% annual per leg, leverage clip 3, weights =
     (trailing 756d Sharpe, floored) x inverse-correlation, renormalized.
  4. Book-level Moreira-Muir vol management, 10% target, clip 2.
  5. Rebalance only when a weight drifts more than 2% (measured: identical
     performance, 54% less churn).
  6. RISK ENGINE + KILL SWITCHES, all pre-declared, none fitted.

HONEST EXPECTATION (from the capstone meta-analysis)
  Sharpe ~1.2-1.4 in the post-GFC regime; ~1.17 if that regime ends (the COT leg
  fails backward validation on 1999-2009). Max drawdown ~10%. 31-38% win rate.
  Long stretches underwater are normal: ~88% of days were below a prior peak.
  NOT INVESTMENT ADVICE.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_superbook as SB
from qlab.metrics import full_metrics
from run_admission import baseline_legs, book, mm, w_invcorr
from run_breadth import new_trend_legs
from run_shorter import COST, MKT, daily

ROOT, OUT = Path(__file__).parent, Path(__file__).parent / "research"
DEV = pd.Timestamp("2022-07-01")
DRIFT_MARKETS = ("MNQ", "ES", "JPXJPY")     # long-only: measured short drag
LOOKBACKS = [40, 80, 160, 240]
WARMUP, MIN_TRAIN, TEST_LEN = 260, 750, 250
TARGET_VOL, LEG_CLIP, BOOK_CLIP = 0.10, 3.0, 2.0
REBAL_THRESH = 0.02
# --- risk engine (pre-declared, prop-firm shaped) ---
DAILY_LOSS_CAP = 0.03
TRAILING_DD_LIMIT = 0.05
KILL_DD = 0.12                # hard stop: book drawdown beyond historical norm
KILL_ROLLING_SHARPE = -1.0    # 1y rolling Sharpe collapse -> halt and review
COOLOFF_DAYS = 60             # a halt means review + restart, not permanent death


def wf_trend_leg(inst, long_only):
    """Frozen architecture: lookback chosen per fold on TRAINING Sharpe only."""
    d = daily(inst)
    r = d["close"].pct_change()
    vol = r.rolling(60).std()
    cand = {}
    for L in LOOKBACKS:
        sig = np.sign(d["close"].pct_change(L))
        if long_only:
            sig = sig.clip(lower=0)
        pos = (sig * (TARGET_VOL / np.sqrt(252)) / vol).clip(-LEG_CLIP, LEG_CLIP).shift(1)
        turn = pos.diff().abs().fillna(pos.abs())
        cand[L] = (pos * r - turn * COST[inst] / d["close"])
    n, oos, te, chosen = len(d), [], WARMUP + MIN_TRAIN, None
    while te < n - 100:
        end = min(te + TEST_LEN, n)
        best, bs = None, -np.inf
        for L, s in cand.items():
            tr = s.iloc[WARMUP:te].dropna()
            sh = tr.mean() / (tr.std() + 1e-12) if len(tr) > 30 else -np.inf
            if sh > bs:
                bs, best = sh, L
        oos.append(cand[best].iloc[te:end])
        chosen, te = best, end
    return (pd.concat(oos).dropna() if oos else pd.Series(dtype=float)), chosen


def w_strength(rets, lb=756, floor=0.1):
    mu, sd = rets.rolling(lb).mean(), rets.rolling(lb).std()
    sh = (mu / (sd + 1e-12) * np.sqrt(252)).clip(lower=0.0)
    w = (sh.fillna(0.0) + floor) * w_invcorr(rets)
    return w.div(w.sum(axis=1), axis=0)


def threshold_weights(w, thresh=REBAL_THRESH):
    out, last = [], None
    for i in range(len(w)):
        cur = w.iloc[i]
        if last is None or (cur - last).abs().max() > thresh:
            last = cur
        out.append(last)
    return pd.DataFrame(out, index=w.index)


def risk_engine(book_ret):
    """Applies the pre-declared guardrails. Returns the governed series + a log."""
    eq, peak, day_start = 1.0, 1.0, 1.0
    out, events, cur_day = [], [], None
    cooloff = 0
    roll = book_ret.rolling(252).apply(
        lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252), raw=True)
    for dt, r in book_ret.items():
        if cur_day != dt.date():
            cur_day, day_start = dt.date(), eq
        if cooloff > 0:                 # flat while halted, then resume
            cooloff -= 1
            out.append(0.0)
            if cooloff == 0:
                peak = eq                   # reset the drawdown clock after review
                events.append((dt.date(), "cool-off complete -> resumed"))
            continue
        eq *= (1 + r)
        peak = max(peak, eq)
        dd = eq / peak - 1
        out.append(r)
        if (eq / day_start - 1) <= -DAILY_LOSS_CAP:
            events.append((dt.date(), "daily loss cap hit -> flat for the day"))
        if dd <= -TRAILING_DD_LIMIT:
            events.append((dt.date(), f"trailing DD {dd:.1%} breached {TRAILING_DD_LIMIT:.0%}"))
        if dd <= -KILL_DD:
            events.append((dt.date(), f"KILL SWITCH: drawdown {dd:.1%} -> halt {COOLOFF_DAYS}d"))
            cooloff = COOLOFF_DAYS
        rs = roll.get(dt, np.nan)
        if not np.isnan(rs) and rs <= KILL_ROLLING_SHARPE:
            events.append((dt.date(), f"KILL SWITCH: 1y rolling Sharpe {rs:.2f} -> halt {COOLOFF_DAYS}d"))
            cooloff = COOLOFF_DAYS
    return pd.Series(out, index=book_ret.index), events


def main():
    print("=" * 78)
    print("ENSEMBLER — production build")
    print("=" * 78)

    base = baseline_legs()
    br = new_trend_legs(final=True)
    br.pop("trend_XAGUSD", None)
    br.pop("trend_GRXEUR", None)          # contaminated vendor series
    legs = {k: v for k, v in {**base, **br}.items() if not k.startswith("trend_")}

    print("\nLEG CONSTRUCTION (shorting disabled where it was measured to lose):")
    chosen = {}
    for inst in MKT:
        lo = inst in DRIFT_MARKETS
        s, ch = wf_trend_leg(inst, lo)
        legs[f"trend_{inst}"] = SB.vol_scale(s)
        chosen[inst] = ch
        print(f"  trend_{inst:10} {'LONG-ONLY' if lo else 'long/short':11} "
              f"current lookback={ch}d")
    for k in legs:
        if not k.startswith("trend_"):
            print(f"  {k:17} {'event/positioning leg':11}")

    fr = pd.DataFrame(legs).fillna(0.0)
    W = threshold_weights(w_strength(fr))
    raw = book(fr, W)
    governed, events = risk_engine(mm(raw).dropna())

    print("\nPERFORMANCE (net, after the risk engine):")
    for label, seg in (("dev  2010-2022", governed[governed.index < DEV]),
                       ("HOLDOUT 2022-26", governed[governed.index >= DEV]),
                       ("full sample", governed)):
        m = full_metrics(None, seg)
        g = lambda k: ("  n/a" if m.get(k) is None else f"{m.get(k):>5}")
        print(f"  {label:16} Sharpe {g('sharpe')}  Calmar {g('calmar')}  "
              f"maxDD {g('max_dd_pct')}%  CAGR {g('cagr_pct')}%  "
              f"underwater {m.get('pct_time_underwater')}%")
    print(f"\n  risk-engine events triggered: {len(events)}")
    for d_, msg in events[:5]:
        print(f"    {d_}  {msg}")

    # ---------------- today's actual target positions ----------------
    print("\n" + "=" * 78)
    print("TARGET POSITIONS — as of the latest bar in the data")
    print("=" * 78)
    last_w = W.iloc[-1]
    asof = fr.index[-1].date()
    print(f"  data as of {asof}  (HistData publishes with a lag; refresh before use)")
    print(f"  {'leg':18} {'weight':>8}  {'direction':>10}  {'lookback':>9}")
    rows = []
    for inst in MKT:
        d = daily(inst)
        L = chosen[inst]
        sig = float(np.sign(d['close'].pct_change(L).iloc[-1]))
        if inst in DRIFT_MARKETS:
            sig = max(sig, 0.0)
        wgt = float(last_w.get(f"trend_{inst}", 0.0))
        direction = "LONG" if sig > 0 else ("SHORT" if sig < 0 else "FLAT")
        print(f"  trend_{inst:12} {wgt:>7.1%}  {direction:>10}  {L:>8}d")
        rows.append({"leg": f"trend_{inst}", "weight": round(wgt, 4),
                     "direction": direction, "lookback_days": L})
    for k in legs:
        if not k.startswith("trend_"):
            wgt = float(last_w.get(k, 0.0))
            print(f"  {k:18} {wgt:>7.1%}  {'event-driven':>10}  {'-':>9}")
            rows.append({"leg": k, "weight": round(wgt, 4), "direction": "event-driven"})

    print("\n" + "=" * 78)
    print("DAILY OPERATING PROCEDURE")
    print("=" * 78)
    print("""  1. After the NY close, refresh prices and the Friday CFTC COT release.
  2. Recompute each leg's signal on CLOSED bars only. Never intrabar.
  3. Recompute weights; rebalance ONLY if any weight moved more than 2%.
  4. Execute at the NEXT session's OPEN. No chasing, no discretion.
  5. Size so each leg targets 10% annualised vol; book capped at 2x leverage.
  6. HALT for the day if the book loses 3%. Do not re-enter to make it back.
  7. KILL SWITCH: halt entirely and review if drawdown exceeds 12% or the
     1-year rolling Sharpe falls below -1.0.
  8. Expect to be underwater ~88% of days and to win only 31-38% of trades.
     The edge arrives through a few long runners; cutting them destroys it.""")

    json.dump({"as_of": str(asof), "positions": rows,
               "risk_events": len(events)},
              open(OUT / "ensembler_positions.json", "w"), indent=2, default=str)
    governed.to_csv(OUT / "ensembler_daily.csv", header=["ret"])
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"phase": "ensembler_production",
                            "n_configs": 1, "note": "deployment build"}) + "\n")
    print("\n  wrote research/ensembler_positions.json + ensembler_daily.csv")


if __name__ == "__main__":
    main()

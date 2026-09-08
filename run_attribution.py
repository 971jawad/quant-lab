"""LIVE ATTRIBUTION - per-leg answer to "what is actually trading, and is it working?"

The book-level number hides everything that matters. -3% live could be one leg
bleeding while six work, or every leg drifting, or -- as it turned out here --
two legs holding 44% of the weight having quietly stopped trading months ago
because their data source froze and pandas dutifully filled the gap with zeros.

This decomposes the live window by leg: weight held, whether the feed is alive,
P&L contributed, and how many days it was actually in the market. A leg with
weight and zero active days is the signature of a dead feed, not a flat signal.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import run_superbook as SB
from run_admission import baseline_legs, book, mm, w_invcorr
from run_breadth import new_trend_legs
from run_ensembler import (DRIFT_MARKETS, wf_trend_leg, w_strength,
                           threshold_weights, risk_engine)
from run_shorter import MKT, daily, live_markets, stale_markets

LIVE = pd.Timestamp("2026-06-28")
base = baseline_legs(); br = new_trend_legs(final=True)
br.pop("trend_XAGUSD", None); br.pop("trend_GRXEUR", None)
legs = {k: v for k, v in {**base, **br}.items() if not k.startswith("trend_")}
for inst in MKT:
    s, _ = wf_trend_leg(inst, inst in DRIFT_MARKETS)
    legs[f"trend_{inst}"] = SB.vol_scale(s)

fr = pd.DataFrame(legs).fillna(0.0)
W = threshold_weights(w_strength(fr))
live_m = set(live_markets())
IDLE = set(json.load(open("research/ensembler_positions.json")).get("idle_weight", {}))

print("=" * 76)
print("EVERY LEG IN THE BOOK - is it running, and what has it done LIVE?")
print("=" * 76)
print(f"{'leg':20} {'kind':14} {'weight':>7} {'status':10} {'live P&L':>9} {'days act':>9}")
rows = []
for leg in fr.columns:
    w = float(W[leg].iloc[-1])
    contrib = (fr[leg] * W[leg]).loc[fr.index >= LIVE]
    raw = fr[leg].loc[fr.index >= LIVE]
    inst = leg.replace("trend_", "")
    kind = "trend/momentum" if leg.startswith("trend_") else "event-driven"
    # The staleness verdict has ONE owner: run_ensembler's guard, recorded in
    # ensembler_positions.json. Re-deriving it here is how MNQ_pull_C got
    # labelled LIVE while holding 18% of the book and never trading.
    status = "STALE-IDLE" if leg in IDLE else "LIVE"
    active = int((raw.abs() > 1e-12).sum())
    print(f"  {leg:18} {kind:14} {w:>6.1%} {status:10} "
          f"{contrib.sum()*100:>+8.2f}% {active:>7}/{len(raw)}")
    rows.append({"leg": leg, "kind": kind, "weight": round(w, 4), "status": status,
                 "live_contrib_pct": round(float(contrib.sum()*100), 3),
                 "days_active": active, "live_days": len(raw)})

tr = [r for r in rows if r["kind"] == "trend/momentum"]
ev = [r for r in rows if r["kind"] == "event-driven"]
print(f"\n  trend legs   {len(tr)}  weight {sum(r['weight'] for r in tr):.1%}"
      f"   live P&L {sum(r['live_contrib_pct'] for r in tr):+.2f}%")
print(f"  event legs   {len(ev)}  weight {sum(r['weight'] for r in ev):.1%}"
      f"   live P&L {sum(r['live_contrib_pct'] for r in ev):+.2f}%")
dead = [r for r in ev if r["days_active"] == 0]
if dead:
    print(f"\n  *** {len(dead)} leg(s) held weight but NEVER FIRED in the live window: "
          + ", ".join(f"{r['leg']} ({r['weight']:.1%})" for r in dead))
json.dump(rows, open("research/live_attribution.json", "w"), indent=2)
print("\nwrote research/live_attribution.json")

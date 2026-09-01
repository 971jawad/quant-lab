"""Fast smoke test for the new AI (neural-net) leg. Runs on a TRUNCATED slice
with small windows so it finishes in ~1-2 min and only proves the wiring +
leak-free path work end to end. Not a performance claim."""
import time

import pandas as pd

import qlab.walkforward as WF
import qlab.strategies as S
from qlab.backtest import Costs
from qlab.features import build_features
from qlab.walkforward import run_wf, oos_metrics

# small windows for a quick multi-fold run on a truncated series
WF.set_scale(mult=4, min_train=40000, test_len=12000, ml_max_train=60000)

t0 = time.time()
df = pd.read_csv("data/SPXUSD_15m.csv", index_col=0)
df.index = pd.to_datetime(df.index, utc=True)
df = df.iloc[:90000]                     # truncate for speed
f = build_features(df)
folds = WF.make_folds(len(df))
print(f"bars: {len(df)}  folds: {len(folds)}  feat {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
one = S.ml_fit(f, len(df) - 60000, len(df) - 100, scheme="nn")
print(f"one MLP fit on ~60k bars: {time.time()-t1:.0f}s  type={type(one).__name__}", flush=True)

t2 = time.time()
r = run_wf(df, f, "ai", "A", Costs(0.25, 0.25, 0.06), cache_key="SPXUSD_smoke")
m = oos_metrics(r, df)
print("ai A:", {k: m.get(k) for k in ("n_trades", "win_rate", "avg_R",
      "profit_factor", "total_return_pct", "sharpe", "calmar", "t_stat")},
      f"{time.time()-t2:.0f}s", flush=True)
print("OK" if not r.oos_trades.empty else "NO TRADES")

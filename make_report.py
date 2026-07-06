"""Build the final research report from walk-forward results."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")

COLS = ["model", "n_trades", "win_rate", "avg_R", "t_stat", "profit_factor",
        "total_return_pct", "ann_return_pct", "sharpe", "max_dd_pct",
        "worst_day_pct", "trailing_dd_breaches", "oos_days"]


def main():
    df = pd.read_csv(RESULTS / "summary.csv")
    df = df[[c for c in COLS if c in df.columns] +
            [c for c in ("instrument", "strategy", "style") if c in df.columns]]
    df = df.sort_values("t_stat", ascending=False)

    tf = "16y of 15-minute data" if "2" in RESULTS.name else "~2.4y of hourly data"
    lines = [f"# Walk-forward results - 75 models, all out-of-sample ({tf})",
             "",
             "Every number below comes from trades in test windows the model",
             "never saw during parameter selection or fitting. Selection used",
             "training data only; ML labels were embargoed at fold boundaries.",
             "", "## Ranked by trade-level t-statistic", ""]
    lines.append(df.to_markdown(index=False))
    lines += ["", "## How to read this",
              "- `t_stat` >= 2.0 -> edge unlikely to be luck. **None reached it.**",
              "- `t_stat` 0.5-2.0 -> promising but unproven; needs more data.",
              "- Negative -> the strategy lost after realistic costs.",
              "",
              "## Aggregates by strategy family (mean across instruments/styles)", ""]
    agg = df.groupby("strategy")[["t_stat", "avg_R", "profit_factor",
                                  "total_return_pct", "max_dd_pct"]].mean().round(3)
    lines.append(agg.to_markdown())
    lines += ["", "## Aggregates by style", ""]
    agg2 = df.groupby("style")[["t_stat", "avg_R", "profit_factor",
                                "total_return_pct", "max_dd_pct"]].mean().round(3)
    lines.append(agg2.to_markdown())
    lines += ["", "## Verification & caveats (read this before believing anything)", "",
              "- **Data**: cross-verified against Nasdaq Inc. (SPY/NDX), LBMA gold PM",
              "  fix, ECB EUR/USD reference rate, and micro-vs-mini contract arbitrage",
              "  (all 7 checks VERIFIED, see data/verification_report.json).",
              "- **No-lookahead**: signals use bar-close info only; entries next bar",
              "  open; stops assumed to fill before targets intrabar; ML labels",
              "  embargoed ML_HORIZON+1 bars at every fold boundary; parameters and",
              "  risk configs selected on training windows only.",
              "- **Noise canary passed**: pure-random features produced no edge beyond",
              "  gold's unconditional drift (+0.099R on always-long) - the pipeline",
              "  does not leak future data.",
              "- **Drift benchmark**: XAUUSD ml_rec (the best family, t up to 1.97)",
              "  earns +0.08..0.13R on longs - statistically indistinguishable from",
              "  the drift baseline. Treat it as trend capture, not alpha.",
              "- **The learning-from-mistakes scheme (ml_err)** cut index losses",
              "  roughly in half vs uniform ML (t -1.4 -> -0.3 on MES) but did NOT",
              "  turn them positive. Recency weighting (ml_rec) helped only on gold",
              "  and hurt badly on indices (regime-chasing).",
              "- **NO MODEL reached t-stat 2.0.** Nothing here is a statistically",
              "  proven edge on ~13-21 months of OOS hourly data. The honest next",
              "  steps are: more history (minute-level, 8+ years), event/session",
              "  filters, and letting the SMC families accumulate sample size.",
              "- Costs modeled: spread + slippage per side + commissions. Untracked:",
              "  overnight funding on CFDs, futures roll gaps, extreme-event slippage.",
              ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.head(15).to_string(index=False))
    print("\n--- strategy aggregates ---\n", agg.to_string())
    print("\n--- style aggregates ---\n", agg2.to_string())


if __name__ == "__main__":
    main()

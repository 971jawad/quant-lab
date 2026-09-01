"""Assemble the ensemble + meta-analysis + conditional-edge results into one
markdown report: results3/report_ensemble.md. Robust to missing pieces (prints
what exists)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RES2, RES3 = ROOT / "results2", ROOT / "results3"


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def md_table(df, cols=None):
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def main():
    out = ["# RULE + ML + AI ensemble — walk-forward, meta-analysis, live paper",
           "",
           "All performance is out-of-sample (walk-forward test windows the models "
           "never saw during selection or fitting). Costs, next-bar-open fills, and "
           "the prop-firm daily/trailing guardrails are the same as the base study.",
           ""]

    # --- per-approach AI leg vs base families
    sA = RES2 / "summary_ai.csv"
    if sA.exists():
        ai = pd.read_csv(sA)
        out += ["## The new AI (neural-net) leg — OOS by instrument/style", "",
                md_table(ai, ["model", "n_trades", "win_rate", "avg_R", "t_stat",
                              "profit_factor", "sharpe", "calmar", "max_dd_pct",
                              "total_return_pct"]), ""]

    # --- ensemble
    es = RES3 / "ensemble_summary.csv"
    if es.exists():
        e = pd.read_csv(es)
        out += ["## Ensemble — selective (cash when no approach has positive "
                "trailing edge) vs always-on, per instrument and the global book", "",
                md_table(e, ["scope", "mode", "n_taken", "win_rate", "avg_R",
                             "t_stat", "profit_factor", "sharpe", "calmar",
                             "max_dd_pct", "total_return_pct"]), "",
                "*Selective* re-picks each approach's best variant every fold on "
                "trailing (already-closed) trades only, keeps those with positive "
                "trailing avg-R, inverse-variance weights them, and holds cash "
                "otherwise. This is the only honest lever an ensemble of weak legs "
                "owns: choosing when to stand aside.", ""]

    # --- meta-analysis
    ma = read_json(RES3 / "meta_analysis.json")
    if ma:
        d = ma["deflated_sharpe"]
        rc = ma["white_reality_check"]
        pbo = ma["pbo_cscv"]
        out += ["## Meta-analysis — is any of it genuine edge?", "",
                f"- **Universe searched**: {ma['universe_size']} strategies over "
                f"{ma['aligned_days']} aligned days.",
                f"- **Deflated Sharpe** (best = `{d['best_by_sharpe']}`): annual "
                f"Sharpe {d['ann_sharpe']} vs an expected-max-under-luck of "
                f"{d['E_max_sharpe_under_null']} across the universe. "
                f"**P(true SR > 0) = {d['DSR_prob_true_SR_gt_0']}** "
                f"(need ≥ 0.95).",
                f"- **White's Reality Check** (best = `{rc['best_model']}`): "
                f"family-wise **p = {rc['rc_pvalue']}** (need < 0.05).",
                f"- **PBO / CSCV**: probability of backtest overfitting = "
                f"**{pbo.get('pbo')}** over {pbo.get('n_splits')} splits.",
                "",
                f"**Verdict: {ma['verdict']}.**", ""]

    # --- conditional edge
    cs = read_json(RES3 / "conditional_summary.json")
    if cs:
        out += ["## Rule-based conditional-edge search (time / vol / session / …)", "",
                f"Pooled {cs['n_trades']} OOS trades, picked the best condition per "
                "metric on the first 60% by date, verified on the untouched last 40%.",
                "", f"**{cs['verdict']}.**", ""]
        if cs.get("survivors"):
            out += ["Conditions that held out-of-sample (t≥2):", "",
                    md_table(pd.DataFrame(cs["survivors"])), ""]

    out += ["## Live (paper) walk-forward", "",
            "`python run_live.py --scope GLOBAL --replay` replays the ensemble's "
            "most recent OOS window as a dated blotter with running equity; "
            "`--watch` books paper fills on genuinely new bars you append. "
            "**No broker, no real orders — simulation only.**", "",
            "---", "*Not trading advice. Out-of-sample performance does not "
            "guarantee future results.*", ""]

    (RES3 / "report_ensemble.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {RES3/'report_ensemble.md'}")


if __name__ == "__main__":
    main()

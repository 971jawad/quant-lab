"""Zip the deployable artifacts: weights, results, risk matrix, report."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "quant_lab_release.zip"


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("weights/*", "weights2/*", "results/summary.csv",
                    "results/report.md", "results/risk_matrix.csv",
                    "results/fold_logs.json", "results/prop_rules_reference.json",
                    "results2/summary.csv", "results2/report.md",
                    "results2/fold_logs.json", "results2/meta_labeling.json",
                    "data/verification_report.json", "data/verification_15m.json",
                    "README.md"):
            for p in ROOT.glob(pat):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

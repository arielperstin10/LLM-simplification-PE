"""
Run all evaluation metrics for a given description.

Usage:
    python -m app.experiments.evaluation.evaluate_run --description "step 2 - RAG top k=3"
    python -m app.experiments.evaluation.evaluate_run  # evaluates all results
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure app is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

METRICS = [
    "app.experiments.evaluation.sari.calculate_sari",
    "app.experiments.evaluation.bleu.calculate_bleu",
    "app.experiments.evaluation.bertscore.calculate_bert",
    "app.experiments.evaluation.fkgl.calculate_fkgl",
    "app.experiments.evaluation.fre.calculate_fre",
    "app.experiments.evaluation.perplexity.calculate_perplexity",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation metrics (SARI, BLEU, BERTScore, FKGL, FRE, Perplexity)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help='Only evaluate results with this description (e.g. "step 2 - RAG top k=3")',
    )
    args = parser.parse_args()

    desc_arg = ["--description", args.description] if args.description else []
    print(f"Running evaluation{' for description: ' + args.description if args.description else ' (all results)'}...\n")

    for i, module in enumerate(METRICS, 1):
        name = module.split(".")[-1].replace("calculate_", "")
        print(f"[{i}/{len(METRICS)}] {name}...")
        result = subprocess.run(
            [sys.executable, "-m", module] + desc_arg,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        if result.returncode != 0:
            print(f"  Failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n✓ All evaluations complete.")


if __name__ == "__main__":
    main()

"""
Aggregate Plan-Simp metrics for the 40 test items and export to CSV.

Uses the same test split as RAG / prompt engineering (random.seed(42)).
Output format matches app/experiments/comparison_models/t5_model/analyze_t5_test_set.py.

Usage:
    python -m app.experiments.comparison_models.plan_simp.analyze_plan_simp_test_set
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from app.db.session import SessionLocal
from app.experiments.RAG.bge.build_embedding_index_test_set import get_test_items
from app.models.plan_simp_evaluation import PlanSimpTextSimplificationEvaluation


def aggregate_plan_simp_test_set_metrics():
    db = SessionLocal()
    try:
        test_items = get_test_items(db)
        test_item_ids = {item[0] for item in test_items}

        if len(test_item_ids) != 40:
            raise ValueError(
                f"Expected 40 test items, got {len(test_item_ids)}. "
                "Ensure get_test_items() returns the correct split."
            )

        rows = (
            db.query(PlanSimpTextSimplificationEvaluation)
            .filter(PlanSimpTextSimplificationEvaluation.item_id.in_(test_item_ids))
            .all()
        )

        if len(rows) < 40:
            print(
                f"Warning: Found {len(rows)} Plan-Simp results for test set (expected 40). "
                "Proceeding with available data."
            )

        def safe_values(field):
            vals = [getattr(r, field) for r in rows if getattr(r, field) is not None]
            return np.array(vals) if vals else np.array([np.nan])

        metrics = {
            "bertscore_f1": safe_values("bertscore_f1"),
            "bleu": safe_values("bleu"),
            "sari": safe_values("sari"),
            "perplexity": safe_values("perplexity"),
            "delta_fkgl": safe_values("delta_fkgl"),
            "fre_delta": safe_values("fre_delta"),
            "fkgl_output": safe_values("fkgl_output"),
            "fre_output": safe_values("fre_output"),
            "lens": safe_values("lens"),
        }

        def mean_std(arr):
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return np.nan, np.nan
            return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0

        row = {
            "model": "plan-simp-pgdyn",
            "description": "Plan-Simp PG-Dyn simplifier (test set)",
            "count": len(rows),
            "BERTScore": mean_std(metrics["bertscore_f1"])[0],
            "BERTScore_std": mean_std(metrics["bertscore_f1"])[1],
            "BLEU": mean_std(metrics["bleu"])[0],
            "BLEU_std": mean_std(metrics["bleu"])[1],
            "SARI": mean_std(metrics["sari"])[0],
            "SARI_std": mean_std(metrics["sari"])[1],
            "Perplexity": mean_std(metrics["perplexity"])[0],
            "Perplexity_std": mean_std(metrics["perplexity"])[1],
            "FKGL_Delta": mean_std(metrics["delta_fkgl"])[0],
            "FKGL_Delta_std": mean_std(metrics["delta_fkgl"])[1],
            "FRE_Delta": mean_std(metrics["fre_delta"])[0],
            "FRE_Delta_std": mean_std(metrics["fre_delta"])[1],
            "FKGL_Output": mean_std(metrics["fkgl_output"])[0],
            "FKGL_Output_std": mean_std(metrics["fkgl_output"])[1],
            "FRE_Output": mean_std(metrics["fre_output"])[0],
            "FRE_Output_std": mean_std(metrics["fre_output"])[1],
            "Entity_Additions_Rate": "",
            "Number_Mismatch_Rate": "",
            "LENS": mean_std(metrics["lens"])[0],
            "LENS_std": mean_std(metrics["lens"])[1],
        }

        return pd.DataFrame([row])
    finally:
        db.close()


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "plan_simp_test_set_analysis.csv"

    print("Aggregating Plan-Simp metrics for 40 test items...")
    df = aggregate_plan_simp_test_set_metrics()
    df.to_csv(output_path, index=False)
    print(f"Exported to {output_path}")
    print(f"  Rows: {len(df)}, Count: {df['count'].iloc[0]}")


if __name__ == "__main__":
    main()

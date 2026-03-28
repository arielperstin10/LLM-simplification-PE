"""
Calculate LENS scores for Plan-Simp test-set rows and store in
plan_simp_text_simplification_evaluation.

Usage:
    python -m app.experiments.comparison_models.plan_simp.calculate_plan_simp_lens
"""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_LIGHTNING_LOG_LEVEL", "ERROR")
os.environ.setdefault("PL_ENABLE_PROGRESS_BAR", "0")
for logger_name in (
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch",
    "pytorch_lightning",
):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

_orig_torch_load = torch.load


def _cpu_torch_load(f, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _cpu_torch_load

from lens import LENS, download_model

from app.db.session import SessionLocal
from app.experiments.RAG.bge.build_embedding_index_test_set import get_test_items
from app.models.dataset import DatasetItem
from app.models.plan_simp_evaluation import PlanSimpTextSimplificationEvaluation

LENS_MODEL_ID = "davidheineman/lens"
LENS_BATCH_SIZE = 32


def calculate_plan_simp_lens_scores():
    db = SessionLocal()

    model_path = download_model(LENS_MODEL_ID)
    lens_metric = LENS(model_path, rescale=True)

    try:
        test_items = get_test_items(db)
        test_item_ids = {item[0] for item in test_items}

        rows = (
            db.query(PlanSimpTextSimplificationEvaluation)
            .filter(PlanSimpTextSimplificationEvaluation.item_id.in_(test_item_ids))
            .all()
        )

        dataset_items = (
            db.query(DatasetItem)
            .filter(DatasetItem.item_id.in_(test_item_ids))
            .all()
        )
        reference_map = {str(d.item_id): d.text_ele for d in dataset_items}

        valid_rows = [
            row
            for row in rows
            if reference_map.get(str(row.item_id)) and row.output_text and row.input_text
        ]

        for batch_start in range(0, len(valid_rows), LENS_BATCH_SIZE):
            batch = valid_rows[batch_start : batch_start + LENS_BATCH_SIZE]
            batch_end = batch_start + len(batch)
            print(f"Processing {batch_start + 1}-{batch_end}/{len(valid_rows)}...")

            complex_texts = [row.input_text for row in batch]
            simplified_texts = [row.output_text for row in batch]
            refs = [[reference_map[str(row.item_id)]] for row in batch]

            try:
                scores = lens_metric.score(
                    complex=complex_texts,
                    simplified=simplified_texts,
                    references=refs,
                    batch_size=LENS_BATCH_SIZE,
                    devices=[],
                )
                for i, row in enumerate(batch):
                    if i < len(scores):
                        row.lens = float(scores[i])
            except Exception:
                import traceback

                traceback.print_exc()

        db.commit()
        print(f"\n✓ LENS scores stored for {len(valid_rows)} Plan-Simp rows.")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    calculate_plan_simp_lens_scores()

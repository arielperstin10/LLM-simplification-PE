"""
Calculate LENS scores for T5 test-set results and store them in the
t5_large_text_simplification_evaluation table.

Mirrors app/experiments/evaluation/lens/calculate_lens.py but targets
T5LargeTextSimplificationEvaluation instead of Evaluation.

Usage:
    python -m app.experiments.comparison_models.t5_model.calculate_t5_lens
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch

# The LENS checkpoint was saved on CUDA — force CPU loading.
_orig_torch_load = torch.load
def _cpu_torch_load(f, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _cpu_torch_load

from lens import LENS, download_model
from app.db.session import SessionLocal
from app.models.t5_evaluation import T5LargeTextSimplificationEvaluation
from app.models.dataset import DatasetItem
from app.experiments.RAG.build_embedding_index_test_set import get_test_items


LENS_MODEL_ID = "davidheineman/lens"


def calculate_t5_lens_scores():
    db = SessionLocal()

    model_path = download_model(LENS_MODEL_ID)
    lens_metric = LENS(model_path, rescale=True)

    try:
        # Only score test-set items (same 40 as used in analysis)
        test_items = get_test_items(db)
        test_item_ids = {item[0] for item in test_items}

        rows = (
            db.query(T5LargeTextSimplificationEvaluation)
            .filter(T5LargeTextSimplificationEvaluation.item_id.in_(test_item_ids))
            .all()
        )

        # Build a lookup: item_id -> reference text (text_ele)
        dataset_items = (
            db.query(DatasetItem)
            .filter(DatasetItem.item_id.in_(test_item_ids))
            .all()
        )
        reference_map = {str(d.item_id): d.text_ele for d in dataset_items}

        processed = 0
        for row in rows:
            processed += 1
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(rows)}...")

            reference = reference_map.get(str(row.item_id))
            if not reference or not row.output_text or not row.input_text:
                continue

            try:
                scores = lens_metric.score(
                    complex=[row.input_text],
                    simplified=[row.output_text],
                    references=[[reference]],
                    batch_size=1,
                    devices=[],
                )
                row.lens = float(scores[0])
            except Exception:
                import traceback
                traceback.print_exc()

        db.commit()
        print(f"\n✓ LENS scores stored for {processed} T5 rows.")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    calculate_t5_lens_scores()

"""
LENS (Learnable Evaluation Metric for Text Simplification) is a reference-based
metric that correlates better with human judgment than SARI or BERTScore.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch

# The LENS checkpoint was saved on CUDA. Patch torch.load to force CPU
# loading since lens-metric doesn't expose a map_location parameter.
_orig_torch_load = torch.load
def _cpu_torch_load(f, *args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")  # force CPU — pytorch_lightning passes map_location=None explicitly
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _cpu_torch_load

from lens import LENS, download_model
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


LENS_MODEL_ID = "davidheineman/lens"


def calculate_lens_scores(description=None):
    db = SessionLocal()

    # Download/load the LENS model checkpoint from HuggingFace (cached after first run)
    # rescale=True gives scores in 0-100 range for better interpretability
    model_path = download_model(LENS_MODEL_ID)
    lens_metric = LENS(model_path, rescale=True)

    try:
        # Get all PromptResults with their corresponding DatasetItems
        query = db.query(
            PromptResult.result_id,
            PromptResult.prompt_version_id,
            PromptResult.input_text,
            PromptResult.output_text,
            DatasetItem.text_ele
        ).join(
            DatasetItem, PromptResult.item_id == DatasetItem.item_id
        ).filter(
            PromptResult.output_text.isnot(None),
            DatasetItem.text_ele.isnot(None)
        )
        if description is not None:
            query = query.filter(PromptResult.description == description)
        results = query.all()

        processed = 0

        for result_id, prompt_version_id, input_text, output_text, text_ele in results:
            processed += 1

            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")

            lens_score = None

            try:
                # LENS score() takes: complex, simplified, references (list of lists)
                # Returns a plain list of scores (not a tuple)
                scores = lens_metric.score(
                    complex=[input_text],
                    simplified=[output_text],
                    references=[[text_ele]],
                    batch_size=1,
                    devices=[]
                )
                lens_score = float(scores[0])
            except Exception:
                import traceback
                traceback.print_exc()

            # Upsert into Evaluation table
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()

            if existing_eval:
                existing_eval.lens = lens_score
            else:
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    lens=lens_score
                )
                db.add(evaluation)

        db.commit()
        print(f"\n✓ Successfully calculated and stored LENS scores:")
        print(f"  - Processed: {processed} results")

    except Exception as e:
        db.rollback()
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LENS scores for PromptResults")
    parser.add_argument("--description", type=str, default=None, help="Only process results with this description")
    args = parser.parse_args()
    calculate_lens_scores(description=args.description)

"""
Calculate SARI scores for PromptResults and store in Evaluation table.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from easse.sari import corpus_sari
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


def calculate_sari_scores():
    db = SessionLocal()
    
    try:
        # Get all PromptResults with their corresponding DatasetItems
        results = db.query(
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
        ).all()
        # Calculate SARI for each individual PromptResult
        processed = 0
        
        for result_id, prompt_version_id, input_text, output_text, text_ele in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            sari_score = None
            
            # Calculate SARI for this single result using corpus_sari with single sentences
            try:
                sari_score = corpus_sari(
                    orig_sents=[input_text],
                    sys_sents=[output_text],
                    refs_sents=[[text_ele]]  
                )
            except Exception as e:
                    import traceback
                    traceback.print_exc()
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                # Update existing evaluation
                existing_eval.sari = sari_score
            else:
                # Create new evaluation
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    sari=sari_score
                )
                db.add(evaluation)
        
        # Commit all changes
        db.commit()
        print(f"\n✓ Successfully calculated and stored SARI scores:")
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
    calculate_sari_scores()

"""
Calculate FRE (Flesch Reading Ease) for PromptResults and store in Evaluation table.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import textstat
from app.db.session import SessionLocal
from app.models.prompt import PromptResult
from app.models.dataset import DatasetItem
from app.models.evaluation import Evaluation


def calculate_fre():
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
            PromptResult.input_text.isnot(None)
        ).all()
        
        # Calculate FRE for each individual PromptResult
        processed = 0
        
        for result_id, prompt_version_id, input_text, output_text, text_ele in results:
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processing {processed}/{len(results)}...")
            
            fre_input = None
            fre_output = None
            fre_delta = None
            
            # Calculate FRE for input text
            try:
                fre_input = textstat.flesch_reading_ease(input_text)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Calculate FRE for output text
            try:
                fre_output = textstat.flesch_reading_ease(output_text)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Calculate FRE delta (output FRE - input FRE)
            # Positive delta means the output is easier to read (higher FRE score) than input
            if fre_input is not None and fre_output is not None:
                fre_delta = fre_output - fre_input
            
            # Check if Evaluation already exists for this result_id
            existing_eval = db.query(Evaluation).filter(
                Evaluation.result_id == result_id
            ).first()
            
            if existing_eval:
                # Update existing evaluation
                existing_eval.fre_input = fre_input
                existing_eval.fre_output = fre_output
                existing_eval.fre_delta = fre_delta
            else:
                # Create new evaluation
                evaluation = Evaluation(
                    prompt_version_id=prompt_version_id,
                    result_id=result_id,
                    fre_input=fre_input,
                    fre_output=fre_output,
                    fre_delta=fre_delta
                )
                db.add(evaluation)
        
        # Commit all changes
        db.commit()
        print(f"\n✓ Successfully calculated and stored FRE scores:")
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
    calculate_fre()


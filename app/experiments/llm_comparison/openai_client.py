import random
import asyncio
from openai import OpenAI
from agents import Agent, Runner, trace
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.prompt import Prompt, PromptVersion, PromptResult
from app.models.dataset import DatasetItem

# Set random seed for reproducibility
random.seed(42)

if settings.OPENAI_API_KEY:
    OpenAI=OpenAI(api_key=settings.OPENAI_API_KEY)
else:
    raise ValueError("OPENAI_API_KEY not set in environment")

# Query: prompt_id connects Prompt and PromptVersion
db = SessionLocal()

try:
    #Prompt Instructions - Get both template_text and prompt_version_id
    zero_shot_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "zeroshot", 
        PromptVersion.is_active == True
    ).first()

    structured_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "structured", 
        PromptVersion.is_active == True
    ).first()

    constraint_prompt = db.query(PromptVersion).join(Prompt, PromptVersion.prompt_id == Prompt.prompt_id).filter(
        Prompt.strategy_type == "constraint", 
        PromptVersion.is_active == True
    ).first()

    # Validate that all prompts exist
    if not zero_shot_prompt:
        raise ValueError("No active zeroshot prompt found in database")
    if not structured_prompt:
        raise ValueError("No active structured prompt found in database")
    if not constraint_prompt:
        raise ValueError("No active constraint prompt found in database")

    InstructionZeroShotPrompt = zero_shot_prompt.template_text
    InstructionStructuredPrompt = structured_prompt.template_text
    InstructionConstraintPrompt = constraint_prompt.template_text

    # Get all items, then randomly select 40 with seed 42
    all_items = db.query(DatasetItem.item_id, DatasetItem.text_ele).filter(DatasetItem.text_ele.isnot(None)).all()
    DatasetItems = random.sample(all_items, min(40, len(all_items)))
finally:
    db.close()

# Create agents with instructions from database
ZeroAgent = Agent(
    name="ZeroAgent",
    instructions=InstructionZeroShotPrompt.replace("{INPUT_TEXT}", ""),
    model="gpt-4o-mini"
)

StructuredAgent = Agent(
    name="StructuredAgent",
    instructions=InstructionStructuredPrompt.replace("{INPUT_TEXT}", ""),
    model="gpt-4o-mini"
)

ConstraintAgent = Agent(
    name="ConstraintAgent",
    instructions=InstructionConstraintPrompt.replace("{INPUT_TEXT}", ""),
    model="gpt-4o-mini",
)

# Iterate and call LLM 3 times for each item (zero-shot, structured, constraint)
async def run_comparison():
    db_session = SessionLocal()
    
    try:
        print(f"Starting comparison with {len(DatasetItems)} items...")
        
        for idx, (item_id, text_ele) in enumerate(DatasetItems, 1):
            print(f"\nProcessing item {idx}/{len(DatasetItems)} (ID: {item_id})...")
            
            with trace(f"LLM Comparison - Item {item_id}"):
                # asyncio.gather guarantees results are returned in the same order as coroutines
                result_zeroshot, result_structured, result_constraint = await asyncio.gather(
                    Runner.run(ZeroAgent, InstructionZeroShotPrompt.replace("{INPUT_TEXT}", text_ele)),
                    Runner.run(StructuredAgent, InstructionStructuredPrompt.replace("{INPUT_TEXT}", text_ele)),
                    Runner.run(ConstraintAgent, InstructionConstraintPrompt.replace("{INPUT_TEXT}", text_ele))
                )
                
                # Extract text from Runner results
                output_zeroshot_text = result_zeroshot.final_output if hasattr(result_zeroshot, 'final_output') else str(result_zeroshot)
                output_structured_text = result_structured.final_output if hasattr(result_structured, 'final_output') else str(result_structured)
                output_constraint_text = result_constraint.final_output if hasattr(result_constraint, 'final_output') else str(result_constraint)
                # Store results in database - all three outputs in one record
                prompt_result = PromptResult(
                    item_id=item_id,
                    prompt_version_id=zero_shot_prompt.prompt_version_id,  
                    input_text=text_ele,
                    output_zeroshot=output_zeroshot_text,
                    output_structured=output_structured_text,
                    output_constraint=output_constraint_text,
                    model_name="gpt-4o-mini"
                )
                db_session.add(prompt_result)
                db_session.commit()     
    except Exception as e:
        db_session.rollback()
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db_session.close()

if __name__ == "__main__":
    asyncio.run(run_comparison())



"""
Run APIO text simplification (zero-shot, few-shot, instruction_induction, optimized) on dataset_items
and store results in apio_text_simplification_evaluation.

Usage:
    python -m app.experiments.comparison_models.APIO.run_apio_text_simplification
    python -m app.experiments.comparison_models.APIO.run_apio_text_simplification --limit 10
    python -m app.experiments.comparison_models.APIO.run_apio_text_simplification --technique optimized
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from uuid import UUID
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure app is on path when run as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem
from app.models.apio_evaluation import APioTextSimplificationEvaluation
from app.experiments.comparison_models.t5_model.metrics import (
    compute_bertscore,
    compute_bleu,
    compute_fkgl,
    compute_fre,
    compute_perplexity,
    compute_sari,
)

# Import llm_utils from APIO package (must be after path setup)
from app.experiments.comparison_models.APIO.llm_utils import llm_single

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# APIO defaults (from config_text_simplification.yaml)
USE_LLM_PROXY = False
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 256
TOP_P = 0.1

# Technique -> glob pattern for prompt JSON (in outputs/gpt-4o-mini_text_simplification/)
TECHNIQUE_PROMPT_PATTERNS = {
    "zero_shot": "zero-shot.json",
    "few_shot": "*few-shot*.json",
    "instruction_induction": "*instruction_induction*.json",
    "optimized": "*optimized*.json",
}

# Exclude subdirs (evaluated_prompts_*, optimized_prompts_debug)
EXCLUDED_SUBDIRS = ("evaluated_prompts_test", "evaluated_prompts_valid", "evaluated_prompts_train", "optimized_prompts_debug")


def _apio_postprocess(text: str) -> str:
    """Clean LLM output (from APIO tasks.py TaskTextSimplification.postprocess_text)."""
    text = text.replace("\n\n", "")
    text = text.strip()
    text = text.strip('"')
    extra_strings = [
        "Answer:",
        "A simplified version of the input sentence is:",
        "Simplified sentence:",
        "Simplified answer:",
        "Simple sentence:",
        "The simplified sentence is:",
        "Simplified input sentence:",
        "Simplified:",
        "Simplification:",
        "Simplified input:",
        "Simplified Sentence:",
        "Simpler Sentence:",
        "Simple answer:",
        "The sentence simplifies to:",
        "Here's a simplified version of the sentence:",
        "The simplified answer would be:",
        "Input sentence simplified:",
        "Certainly! Here's a simplified version of the sentence:",
        "Output simplified sentence:",
        "The simplified sentence could be:",
        "The simple sentence is:",
        "The sentence in simpler terms could be:",
        "Output:",
        "The answer is:",
        "Simplify the input sentence:",
        "Here is the simplified version of the input sentence:",
        "The input sentence simplifies to:",
    ]
    for s in extra_strings:
        text = text.replace(s, "")
    return text.strip()


def get_prompts_dir() -> Path:
    """Resolve path to APIO prompts (outputs/gpt-4o-mini_text_simplification)."""
    return Path(__file__).parent / "outputs" / "gpt-4o-mini_text_simplification"


def load_prompt_templates(prompts_dir: Path, techniques: Optional[List[str]] = None) -> Dict[str, str]:
    """Load PROMPT_TEMPLATE from JSON files. Returns {technique: template}."""
    result = {}
    if techniques is None:
        techniques = list(TECHNIQUE_PROMPT_PATTERNS.keys())

    for tech in techniques:
        pattern = TECHNIQUE_PROMPT_PATTERNS[tech]
        if "*" in pattern:
            candidates = list(prompts_dir.glob(pattern))
            # Exclude subdirs
            candidates = [c for c in candidates if c.parent == prompts_dir]
            if not candidates:
                logger.warning(f"No prompt file for {tech} (pattern: {pattern})")
                continue
            # Prefer deterministic order (e.g. optimized over iter debug)
            candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))
        else:
            p = prompts_dir / pattern
            if not p.exists():
                logger.warning(f"Prompt file not found: {p}")
                continue
            candidates = [p]

        with open(candidates[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        template = data.get("PROMPT_TEMPLATE")
        if not template or "<input_text>" not in template:
            logger.warning(f"Invalid prompt template for {tech}: missing PROMPT_TEMPLATE or <input_text>")
            continue
        result[tech] = template

    return result


def fetch_dataset_items(
    db, limit: Optional[int] = None
) -> List[Tuple[UUID, str, Optional[str]]]:
    """Fetch item_id, text_adv, text_ele from dataset_items where text_adv is not null."""
    query = db.query(
        DatasetItem.item_id,
        DatasetItem.text_adv,
        DatasetItem.text_ele,
    ).filter(DatasetItem.text_adv.isnot(None))
    rows = query.all()
    if limit:
        rows = rows[:limit]
    return [(r.item_id, r.text_adv, r.text_ele) for r in rows]


def exists_for_item_technique(db, item_id: UUID, technique: str) -> bool:
    """Check if a record already exists for (item_id, technique)."""
    return (
        db.query(APioTextSimplificationEvaluation)
        .filter(
            APioTextSimplificationEvaluation.item_id == item_id,
            APioTextSimplificationEvaluation.technique == technique,
        )
        .first()
        is not None
    )


def run_llm_inference(prompt: str) -> str:
    """Call LLM and postprocess output."""
    raw = llm_single(
        use_llm_proxy=USE_LLM_PROXY,
        prompt=prompt,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
    )
    return _apio_postprocess(raw) if raw else ""


def compute_all_metrics(
    input_text: str,
    output_text: str,
    reference_text: Optional[str],
    perplexity_model,
    perplexity_tokenizer,
    device,
) -> dict:
    """Compute all evaluation metrics (same as T5)."""
    sari = compute_sari(input_text, output_text, reference_text)
    bertscore_f1 = compute_bertscore(output_text, reference_text)
    bleu = compute_bleu(output_text, reference_text)
    fkgl_input, fkgl_output, delta_fkgl = compute_fkgl(input_text, output_text)
    fre_input, fre_output, fre_delta = compute_fre(input_text, output_text)
    perplexity = compute_perplexity(output_text, perplexity_model, perplexity_tokenizer, device)
    return {
        "sari": sari,
        "bertscore_f1": bertscore_f1,
        "fkgl_input": fkgl_input,
        "fkgl_output": fkgl_output,
        "delta_fkgl": delta_fkgl,
        "fre_input": fre_input,
        "fre_output": fre_output,
        "fre_delta": fre_delta,
        "bleu": bleu,
        "perplexity": perplexity,
    }


def insert_result(
    db, item_id: UUID, technique: str, input_text: str, output_text: str, metrics: dict
):
    """Insert a result row into apio_text_simplification_evaluation."""
    record = APioTextSimplificationEvaluation(
        item_id=item_id,
        technique=technique,
        input_text=input_text,
        output_text=output_text,
        sari=metrics.get("sari"),
        bertscore_f1=metrics.get("bertscore_f1"),
        fkgl_input=metrics.get("fkgl_input"),
        fkgl_output=metrics.get("fkgl_output"),
        delta_fkgl=metrics.get("delta_fkgl"),
        fre_input=metrics.get("fre_input"),
        fre_output=metrics.get("fre_output"),
        fre_delta=metrics.get("fre_delta"),
        bleu=metrics.get("bleu"),
        perplexity=metrics.get("perplexity"),
    )
    db.add(record)
    db.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Run APIO text simplification on dataset_items and store results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items (for testing)",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        choices=list(TECHNIQUE_PROMPT_PATTERNS.keys()),
        help="Run only this technique (default: all four)",
    )
    args = parser.parse_args()

    prompts_dir = get_prompts_dir()
    if not prompts_dir.exists():
        logger.error(f"Prompts directory not found: {prompts_dir}")
        sys.exit(1)

    techniques = [args.technique] if args.technique else list(TECHNIQUE_PROMPT_PATTERNS.keys())
    prompt_templates = load_prompt_templates(prompts_dir, techniques)
    if not prompt_templates:
        logger.error("No valid prompt templates loaded. Check prompts_dir and JSON files.")
        sys.exit(1)
    logger.info(f"Loaded {len(prompt_templates)} techniques: {list(prompt_templates.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading perplexity model (distilgpt2)")
    perplexity_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    perplexity_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token
    perplexity_model.eval()

    db = SessionLocal()
    try:
        items = fetch_dataset_items(db, limit=args.limit)
        logger.info(f"Found {len(items)} dataset items to process")

        processed = 0
        skipped = 0
        failed = 0

        for idx, (item_id, text_adv, text_ele) in enumerate(items, 1):
            if idx % 10 == 0 or idx == 1:
                logger.info(f"Processing {idx}/{len(items)}...")

            for technique, template in prompt_templates.items():
                if exists_for_item_technique(db, item_id, technique):
                    skipped += 1
                    continue

                try:
                    prompt = template.replace("<input_text>", text_adv)
                    output_text = run_llm_inference(prompt)
                    metrics = compute_all_metrics(
                        text_adv,
                        output_text,
                        text_ele,
                        perplexity_model,
                        perplexity_tokenizer,
                        device,
                    )
                    insert_result(db, item_id, technique, text_adv, output_text, metrics)
                    processed += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"Failed item {item_id} technique {technique}: {e}")

        logger.info(
            f"\nDone. Processed: {processed}, Skipped: {skipped}, Failed: {failed}"
        )
    except Exception as e:
        db.rollback()
        logger.exception(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()

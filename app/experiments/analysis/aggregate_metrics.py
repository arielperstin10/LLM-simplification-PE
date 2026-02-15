"""
Aggregate evaluation metrics for model comparison.
Provides multiple levels of aggregation:
1. Overall by model (across all prompts)
2. By model + strategy (per prompt strategy)
3. Detailed per-item data
"""

import sys
from pathlib import Path
from sqlalchemy import func
import pandas as pd
import numpy as np

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.db.session import SessionLocal
from app.models.evaluation import Evaluation
from app.models.prompt import PromptResult, PromptVersion, Prompt


def aggregate_overall_by_model():
    """
    Aggregate metrics overall by model (ignoring prompt strategy).
    Returns DataFrame with mean, std, min, max, count for each metric.
    """
    db = SessionLocal()
    
    try:
        results = db.query(
            PromptResult.model_name,
            # Quality metrics (higher is better)
            func.avg(Evaluation.bertscore_f1).label('avg_bertscore'),
            func.stddev(Evaluation.bertscore_f1).label('std_bertscore'),
            func.min(Evaluation.bertscore_f1).label('min_bertscore'),
            func.max(Evaluation.bertscore_f1).label('max_bertscore'),
            
            func.avg(Evaluation.bleu).label('avg_bleu'),
            func.stddev(Evaluation.bleu).label('std_bleu'),
            func.min(Evaluation.bleu).label('min_bleu'),
            func.max(Evaluation.bleu).label('max_bleu'),
            
            func.avg(Evaluation.sari).label('avg_sari'),
            func.stddev(Evaluation.sari).label('std_sari'),
            func.min(Evaluation.sari).label('min_sari'),
            func.max(Evaluation.sari).label('max_sari'),
            
            func.avg(Evaluation.perplexity).label('avg_perplexity'),
            func.stddev(Evaluation.perplexity).label('std_perplexity'),
            func.min(Evaluation.perplexity).label('min_perplexity'),
            func.max(Evaluation.perplexity).label('max_perplexity'),
            
            # Readability deltas (FKGL: negative is better, FRE: positive is better)
            func.avg(Evaluation.delta_fkgl).label('avg_delta_fkgl'),
            func.stddev(Evaluation.delta_fkgl).label('std_delta_fkgl'),
            func.min(Evaluation.delta_fkgl).label('min_delta_fkgl'),
            func.max(Evaluation.delta_fkgl).label('max_delta_fkgl'),
            
            func.avg(Evaluation.fre_delta).label('avg_fre_delta'),
            func.stddev(Evaluation.fre_delta).label('std_fre_delta'),
            func.min(Evaluation.fre_delta).label('min_fre_delta'),
            func.max(Evaluation.fre_delta).label('max_fre_delta'),
            
            # Output readability levels
            func.avg(Evaluation.fkgl_output).label('avg_fkgl_output'),
            func.avg(Evaluation.fre_output).label('avg_fre_output'),
            
            # Count
            func.count(Evaluation.evaluation_id).label('count')
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        ).group_by(
            PromptResult.model_name
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'count': r.count,
            # BERTScore
            'bertscore_mean': float(r.avg_bertscore) if r.avg_bertscore else None,
            'bertscore_std': float(r.std_bertscore) if r.std_bertscore else None,
            'bertscore_min': float(r.min_bertscore) if r.min_bertscore else None,
            'bertscore_max': float(r.max_bertscore) if r.max_bertscore else None,
            # BLEU
            'bleu_mean': float(r.avg_bleu) if r.avg_bleu else None,
            'bleu_std': float(r.std_bleu) if r.std_bleu else None,
            'bleu_min': float(r.min_bleu) if r.min_bleu else None,
            'bleu_max': float(r.max_bleu) if r.max_bleu else None,
            # SARI
            'sari_mean': float(r.avg_sari) if r.avg_sari else None,
            'sari_std': float(r.std_sari) if r.std_sari else None,
            'sari_min': float(r.min_sari) if r.min_sari else None,
            'sari_max': float(r.max_sari) if r.max_sari else None,
            # Perplexity
            'perplexity_mean': float(r.avg_perplexity) if r.avg_perplexity else None,
            'perplexity_std': float(r.std_perplexity) if r.std_perplexity else None,
            'perplexity_min': float(r.min_perplexity) if r.min_perplexity else None,
            'perplexity_max': float(r.max_perplexity) if r.max_perplexity else None,
            # FKGL Delta
            'fkgl_delta_mean': float(r.avg_delta_fkgl) if r.avg_delta_fkgl else None,
            'fkgl_delta_std': float(r.std_delta_fkgl) if r.std_delta_fkgl else None,
            'fkgl_delta_min': float(r.min_delta_fkgl) if r.min_delta_fkgl else None,
            'fkgl_delta_max': float(r.max_delta_fkgl) if r.max_delta_fkgl else None,
            # FRE Delta
            'fre_delta_mean': float(r.avg_fre_delta) if r.avg_fre_delta else None,
            'fre_delta_std': float(r.std_fre_delta) if r.std_fre_delta else None,
            'fre_delta_min': float(r.min_fre_delta) if r.min_fre_delta else None,
            'fre_delta_max': float(r.max_fre_delta) if r.max_fre_delta else None,
            # Output readability
            'fkgl_output_mean': float(r.avg_fkgl_output) if r.avg_fkgl_output else None,
            'fre_output_mean': float(r.avg_fre_output) if r.avg_fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def aggregate_by_model_and_strategy():
    """
    Aggregate metrics by model and prompt strategy.
    Returns DataFrame with metrics grouped by model + strategy_type.
    """
    db = SessionLocal()
    
    try:
        results = db.query(
            PromptResult.model_name,
            Prompt.strategy_type,
            # Quality metrics
            func.avg(Evaluation.bertscore_f1).label('avg_bertscore'),
            func.stddev(Evaluation.bertscore_f1).label('std_bertscore'),
            func.avg(Evaluation.bleu).label('avg_bleu'),
            func.stddev(Evaluation.bleu).label('std_bleu'),
            func.avg(Evaluation.sari).label('avg_sari'),
            func.stddev(Evaluation.sari).label('std_sari'),
            func.avg(Evaluation.perplexity).label('avg_perplexity'),
            func.stddev(Evaluation.perplexity).label('std_perplexity'),
            # Readability deltas
            func.avg(Evaluation.delta_fkgl).label('avg_delta_fkgl'),
            func.stddev(Evaluation.delta_fkgl).label('std_delta_fkgl'),
            func.avg(Evaluation.fre_delta).label('avg_fre_delta'),
            func.stddev(Evaluation.fre_delta).label('std_fre_delta'),
            # Output readability
            func.avg(Evaluation.fkgl_output).label('avg_fkgl_output'),
            func.avg(Evaluation.fre_output).label('avg_fre_output'),
            # Count
            func.count(Evaluation.evaluation_id).label('count')
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, Evaluation.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        ).group_by(
            PromptResult.model_name,
            Prompt.strategy_type
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'strategy': r.strategy_type,
            'count': r.count,
            # BERTScore
            'bertscore_mean': float(r.avg_bertscore) if r.avg_bertscore else None,
            'bertscore_std': float(r.std_bertscore) if r.std_bertscore else None,
            # BLEU
            'bleu_mean': float(r.avg_bleu) if r.avg_bleu else None,
            'bleu_std': float(r.std_bleu) if r.std_bleu else None,
            # SARI
            'sari_mean': float(r.avg_sari) if r.avg_sari else None,
            'sari_std': float(r.std_sari) if r.std_sari else None,
            # Perplexity
            'perplexity_mean': float(r.avg_perplexity) if r.avg_perplexity else None,
            'perplexity_std': float(r.std_perplexity) if r.std_perplexity else None,
            # FKGL Delta
            'fkgl_delta_mean': float(r.avg_delta_fkgl) if r.avg_delta_fkgl else None,
            'fkgl_delta_std': float(r.std_delta_fkgl) if r.std_delta_fkgl else None,
            # FRE Delta
            'fre_delta_mean': float(r.avg_fre_delta) if r.avg_fre_delta else None,
            'fre_delta_std': float(r.std_fre_delta) if r.std_fre_delta else None,
            # Output readability
            'fkgl_output_mean': float(r.avg_fkgl_output) if r.avg_fkgl_output else None,
            'fre_output_mean': float(r.avg_fre_output) if r.avg_fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def get_detailed_results():
    """
    Get detailed per-item results for deep analysis.
    Returns DataFrame with all metrics for each individual result.
    """
    db = SessionLocal()
    
    try:
        results = db.query(
            PromptResult.model_name,
            Prompt.strategy_type,
            PromptResult.item_id,
            Evaluation.result_id,
            # Quality metrics
            Evaluation.bertscore_f1,
            Evaluation.bleu,
            Evaluation.sari,
            Evaluation.perplexity,
            # Readability deltas
            Evaluation.delta_fkgl,
            Evaluation.fre_delta,
            # Input/output readability
            Evaluation.fkgl_input,
            Evaluation.fkgl_output,
            Evaluation.fre_input,
            Evaluation.fre_output,
        ).join(
            PromptResult, Evaluation.result_id == PromptResult.result_id
        ).join(
            PromptVersion, Evaluation.prompt_version_id == PromptVersion.prompt_version_id
        ).join(
            Prompt, PromptVersion.prompt_id == Prompt.prompt_id
        ).filter(
            PromptResult.model_name.isnot(None),
            Evaluation.bertscore_f1.isnot(None)
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'model': r.model_name,
            'strategy': r.strategy_type,
            'item_id': str(r.item_id),
            'result_id': str(r.result_id),
            # Quality metrics
            'bertscore': float(r.bertscore_f1) if r.bertscore_f1 else None,
            'bleu': float(r.bleu) if r.bleu else None,
            'sari': float(r.sari) if r.sari else None,
            'perplexity': float(r.perplexity) if r.perplexity else None,
            # Readability deltas
            'fkgl_delta': float(r.delta_fkgl) if r.delta_fkgl else None,
            'fre_delta': float(r.fre_delta) if r.fre_delta else None,
            # Input/output readability
            'fkgl_input': float(r.fkgl_input) if r.fkgl_input else None,
            'fkgl_output': float(r.fkgl_output) if r.fkgl_output else None,
            'fre_input': float(r.fre_input) if r.fre_input else None,
            'fre_output': float(r.fre_output) if r.fre_output else None,
        } for r in results])
        
        return df
        
    finally:
        db.close()


def print_summary_table(df_overall):
    """Print a formatted summary table of overall metrics."""
    print("\n" + "="*100)
    print("OVERALL MODEL COMPARISON SUMMARY")
    print("="*100)
    
    for _, row in df_overall.iterrows():
        print(f"\n{row['model'].upper()}:")
        print(f"  Samples: {row['count']}")
        print(f"\n  Quality Metrics (higher is better):")
        print(f"    BERTScore: {row['bertscore_mean']:.4f} ± {row['bertscore_std']:.4f} "
              f"[{row['bertscore_min']:.4f}, {row['bertscore_max']:.4f}]")
        print(f"    BLEU:      {row['bleu_mean']:.4f} ± {row['bleu_std']:.4f} "
              f"[{row['bleu_min']:.4f}, {row['bleu_max']:.4f}]")
        print(f"    SARI:      {row['sari_mean']:.4f} ± {row['sari_std']:.4f} "
              f"[{row['sari_min']:.4f}, {row['sari_max']:.4f}]")
        print(f"    Perplexity: {row['perplexity_mean']:.4f} ± {row['perplexity_std']:.4f} "
              f"[{row['perplexity_min']:.4f}, {row['perplexity_max']:.4f}] (lower is better)")
        print(f"\n  Readability Improvement:")
        print(f"    FKGL Δ:    {row['fkgl_delta_mean']:.2f} ± {row['fkgl_delta_std']:.2f} "
              f"(negative = simpler, better)")
        print(f"    FRE Δ:     {row['fre_delta_mean']:.2f} ± {row['fre_delta_std']:.2f} "
              f"(positive = easier, better)")
        print(f"\n  Output Readability:")
        print(f"    FKGL:      {row['fkgl_output_mean']:.2f} (lower = simpler)")
        print(f"    FRE:       {row['fre_output_mean']:.2f} (higher = easier)")


def export_to_csv(df, filename, output_dir=None):
    """Export DataFrame to CSV file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "csv"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Exported to {filepath}")
    return filepath


def main():
    """Main function to run all aggregations and exports."""
    print("Starting metric aggregation...")
    
    # 1. Overall aggregation by model
    print("\n1. Aggregating overall metrics by model...")
    df_overall = aggregate_overall_by_model()
    if not df_overall.empty:
        print_summary_table(df_overall)
        export_to_csv(df_overall, "model_comparison_overall.csv")
    else:
        print("⚠ No data found for overall aggregation")
    
    # 2. Aggregation by model + strategy
    print("\n2. Aggregating metrics by model and strategy...")
    df_by_strategy = aggregate_by_model_and_strategy()
    if not df_by_strategy.empty:
        print(f"\n✓ Found {len(df_by_strategy)} model-strategy combinations")
        export_to_csv(df_by_strategy, "model_comparison_by_strategy.csv")
    else:
        print("⚠ No data found for strategy-based aggregation")
    
    # 3. Detailed per-item results
    print("\n3. Extracting detailed per-item results...")
    df_detailed = get_detailed_results()
    if not df_detailed.empty:
        print(f"✓ Found {len(df_detailed)} individual results")
        export_to_csv(df_detailed, "model_comparison_detailed.csv")
    else:
        print("⚠ No detailed data found")
    
    print("\n" + "="*100)
    print("✓ Aggregation complete!")
    print("="*100)


if __name__ == "__main__":
    main()


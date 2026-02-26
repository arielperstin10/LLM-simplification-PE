"""
Generate evaluation results tables grouped by prompt_version_id.

Each table shows:
- Rows: Models (OpenAI, Ollama, etc.)
- Columns: Metrics (BERTScore, BLEU, SARI, etc.)
- One table per prompt_version_id
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.experiments.analysis.aggregate_metrics import (
    aggregate_by_prompt_version,
    print_tables_by_prompt_version,
    export_tables_by_prompt_version
)


def main():
    """Generate and display/export results tables by prompt_version_id."""
    print("="*120)
    print("EVALUATION RESULTS BY PROMPT VERSION")
    print("="*120)
    
    # Generate tables
    tables_dict = aggregate_by_prompt_version()
    
    if not tables_dict:
        print("\n⚠ No evaluation data found.")
        print("Make sure you have:")
        print("  1. Run LLM comparisons to generate prompt results")
        print("  2. Calculated evaluation metrics (BERTScore, BLEU, SARI, etc.)")
        return
    
    print(f"\n✓ Found {len(tables_dict)} prompt version(s) with evaluation data\n")
    
    # Display tables
    print_tables_by_prompt_version(tables_dict)
    
    # Export to CSV
    print("\nExporting tables to CSV...")
    exported_files = export_tables_by_prompt_version(tables_dict)
    print(f"\n✓ Successfully exported {len(exported_files)} table(s)")
    
    print("\n" + "="*120)
    print("Complete!")
    print("="*120)


if __name__ == "__main__":
    main()

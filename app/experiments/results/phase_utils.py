"""Shared phase / experiment labels for results pipeline (DB description + version → phase)."""

from __future__ import annotations

import pandas as pd

from app.experiments.visualization.config import DESCRIPTION_TO_PHASE

EXCLUDED_MODELS = {"sonar-pro"}

PHASE_TO_META = {
    "Simple Prompt v1": ("Simple Prompt v1", "none"),
    "Simple Prompt v2": ("Simple Prompt v2", "none"),
    "OpenAI RAG best of v1/v2": ("OpenAI RAG best of v1/v2", "openai"),
    "OpenAI RAG v3": ("OpenAI RAG v3", "openai"),
    "E5 v3": ("E5 v3", "e5"),
    "BGE v3": ("BGE v3", "bge"),
}


def assign_phase(row: pd.Series) -> str:
    """Match visualization/data_loader._assign_phase; support corrected DB spelling for v3."""
    desc = row.get("description")
    version = row.get("version")

    phase = DESCRIPTION_TO_PHASE.get(desc)
    if phase is not None:
        return phase

    if desc == "step 1 - simple prompt engineering":
        if version == "v1":
            return "Simple Prompt v1"
        if version == "v2":
            return "Simple Prompt v2"

    return "UNKNOWN"


def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase"] = out.apply(assign_phase, axis=1)
    out["experiment_group"] = out["phase"].map(
        lambda p: PHASE_TO_META.get(p, (None, None))[0]
    )
    out["retrieval"] = out["phase"].map(lambda p: PHASE_TO_META.get(p, (None, None))[1])
    return out


def filter_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    return df[~df["model"].isin(EXCLUDED_MODELS)].copy()

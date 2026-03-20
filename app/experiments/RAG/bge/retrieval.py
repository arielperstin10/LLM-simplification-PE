"""
retrieval.py (BGE version)
--------------------------
Retrieves top-k similar items from the train embeddings (dataset_item_embeddings_bge_768).
"""

import uuid
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding import DatasetItemEmbeddingBGE, DatasetItemEmbeddingBGETestSet
from app.models.dataset import DatasetItem


def _get_query_embedding(item_id: uuid.UUID, db: Session) -> List[float]:
    """
    Fetch the precomputed embedding for a test item from dataset_item_embeddings_bge_768_test_set.
    """
    row = (
        db.query(DatasetItemEmbeddingBGETestSet.embedding)
        .filter(DatasetItemEmbeddingBGETestSet.item_id == item_id)
        .first()
    )
    if not row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_bge_768_test_set. "
            "Run build_embedding_index_test_set first."
        )
    emb = row[0]
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return list(emb)


def retrieve_top_k(
    item_id: uuid.UUID,
    k: int,
    db: Session,
) -> List[Tuple[str, str]]:
    """
    Retrieve top-k most similar items from the BGE train embeddings corpus.
    Uses precomputed embeddings.
    """
    query_embedding = _get_query_embedding(item_id, db)

    stmt = (
        select(DatasetItemEmbeddingBGE.text_adv, DatasetItem.text_ele)
        .join(
            DatasetItem,
            DatasetItemEmbeddingBGE.item_id == DatasetItem.item_id,
        )
        .where(DatasetItem.text_ele.isnot(None))
        .order_by(
            DatasetItemEmbeddingBGE.embedding.cosine_distance(query_embedding)
        )
        .limit(k)
    )

    rows = db.execute(stmt).fetchall()
    return [(row[0], row[1]) for row in rows if row[1]]

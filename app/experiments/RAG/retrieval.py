"""
RAG retrieval module.

Retrieves top-k similar items from the train embeddings (dataset_item_embeddings_1536)
using cosine similarity. Uses precomputed embeddings from dataset_item_embeddings_1536_test_set
for the query (no API calls).
"""

import uuid
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.embedding import DatasetItemEmbedding1536, DatasetItemEmbedding1536TestSet
from app.models.dataset import DatasetItem


def _get_query_embedding(item_id: uuid.UUID, db: Session) -> List[float]:
    """
    Fetch the precomputed embedding for a test item from dataset_item_embeddings_1536_test_set.
    """
    row = (
        db.query(DatasetItemEmbedding1536TestSet.embedding)
        .filter(DatasetItemEmbedding1536TestSet.item_id == item_id)
        .first()
    )
    if not row:
        raise ValueError(
            f"No embedding found for item_id={item_id} in dataset_item_embeddings_1536_test_set. "
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
    Retrieve top-k most similar items from the train embeddings corpus.

    Uses the precomputed embedding from dataset_item_embeddings_1536_test_set (no API call).
    Searches dataset_item_embeddings_1536 via cosine similarity.

    Args:
        item_id: Test item ID (embedding looked up from test set table).
        k: Number of similar items to retrieve.
        db: SQLAlchemy session.

    Returns:
        List of (text_adv, text_ele) tuples. Items without text_ele are excluded.
    """
    query_embedding = _get_query_embedding(item_id, db)

    stmt = (
        select(DatasetItemEmbedding1536.text_adv, DatasetItem.text_ele)
        .join(
            DatasetItem,
            DatasetItemEmbedding1536.item_id == DatasetItem.item_id,
        )
        .where(DatasetItem.text_ele.isnot(None))
        .order_by(
            DatasetItemEmbedding1536.embedding.cosine_distance(query_embedding)
        )
        .limit(k)
    )

    rows = db.execute(stmt).fetchall()
    return [(row[0], row[1]) for row in rows if row[1]]

from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from pgvector.sqlalchemy import Vector

from app.db.base import Base


class DatasetItemEmbedding1536(Base):
    """
    ORM model for public.dataset_item_embeddings_1536.

    Stores 1536-dimensional OpenAI embeddings (text-embedding-3-small)
    for each dataset item, used by the RAG pipeline.

    DDL:
        CREATE TABLE public.dataset_item_embeddings_1536 (
            embedding_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            item_id      uuid UNIQUE NOT NULL REFERENCES dataset_items(item_id) ON DELETE CASCADE,
            text_adv     text NOT NULL,
            embedding    vector(1536) NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now()
        );
    """

    __tablename__ = "dataset_item_embeddings_1536"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationship back to the source item
    dataset_item = relationship("DatasetItem", backref="embedding_1536")


class DatasetItemEmbedding1536TestSet(Base):
    """
    ORM model for public.dataset_item_embeddings_1536_test_set.

    Stores 1536-dimensional OpenAI embeddings for the 40 test-set items
    (the complement of the RAG/train split).

    DDL:
        CREATE TABLE public.dataset_item_embeddings_1536_test_set (
            embedding_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            item_id      uuid UNIQUE NOT NULL REFERENCES dataset_items(item_id) ON DELETE CASCADE,
            text_adv     text NOT NULL,
            embedding    vector(1536) NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now()
        );
    """

    __tablename__ = "dataset_item_embeddings_1536_test_set"
    __table_args__ = {"schema": "public"}

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    item_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_items.item_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text_adv = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    dataset_item = relationship("DatasetItem", backref="embedding_1536_test_set")

from sqlalchemy import Column, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.db.base import Base


class Evaluation(Base):
    __tablename__ = "evaluation"

    evaluation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.prompt_version_id", ondelete="CASCADE"), nullable=False)
    sari = Column(Float, nullable=True)
    sari_with_refs = Column(Float, nullable=True)
    bertscore_f1 = Column(Float, nullable=True)
    fkgl = Column(Float, nullable=True)
    fkgl_delta = Column(Float, nullable=True)
    fre = Column(Float, nullable=True)
    fre_delta = Column(Float, nullable=True)
    entity_additions_rate = Column(Float, nullable=True)
    number_mismatch_rate = Column(Float, nullable=True)

    # Relationships
    prompt_version = relationship("PromptVersion", back_populates="evaluations")


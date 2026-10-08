from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, model_validator
from datetime import datetime, timezone

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    POLITICAL_PARTY = "POLITICAL_PARTY"
    GOVERNMENT_BODY = "GOVERNMENT_BODY"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    LAW_STATUTE = "LAW_STATUTE"
    LEGAL_CASE = "LEGAL_CASE"
    FINANCIAL_INSTRUMENT = "FINANCIAL_INSTRUMENT"

class EvidencePolarity(str, Enum):
    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    NEUTRAL = "NEUTRAL"

class DecisionType(str, Enum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    REVIEW = "REVIEW"

class SourceType(str, Enum):
    URL = "URL"
    DOCUMENT = "DOCUMENT"
    API = "API"
    USER = "USER"
    OTHER = "OTHER"

class Source(BaseModel):
    source_id: str
    source_type: SourceType
    source_name: str
    source_uri: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    reliability_metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)

class Record(BaseModel):
    record_id: str
    source_id: str
    entity_type: EntityType
    raw_name: str
    normalized_attributes: Dict[str, Any] = Field(default_factory=dict)
    structured_attributes: Dict[str, Any] = Field(default_factory=dict)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    observed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CandidatePair(BaseModel):
    candidate_id: str
    left_record_id: str
    right_record_id: str
    generation_method: str
    blocking_signals: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_not_self_pair(self):
        if self.left_record_id == self.right_record_id:
            raise ValueError("CandidatePair cannot contain identical record IDs")
        return self

class Evidence(BaseModel):
    evidence_id: str
    evidence_type: str
    polarity: EvidencePolarity
    value_description: str
    source_reference: Optional[str] = None
    record_reference: Optional[str] = None
    provenance: str
    strength: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_provenance(self):
        if not self.provenance.strip():
            raise ValueError("Evidence must have provenance")
        return self

class MatchDecision(BaseModel):
    candidate_pair_id: str
    decision: DecisionType
    score: Optional[float] = None
    threshold: Optional[float] = None
    model_version: str
    evidence_references: List[str] = Field(default_factory=list)
    explanation_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EntityCluster(BaseModel):
    entity_cluster_id: str
    entity_type: EntityType
    member_record_ids: List[str]
    status: str
    quality_confidence_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    derivation_version: str

    @model_validator(mode='after')
    def check_unique_members(self):
        if len(self.member_record_ids) != len(set(self.member_record_ids)):
            raise ValueError("EntityCluster cannot contain duplicate record IDs")
        return self

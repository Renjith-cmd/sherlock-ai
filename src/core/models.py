from enum import Enum
from types import MappingProxyType
from typing import List, Optional, Any, Dict, Mapping, Tuple, Set
from datetime import datetime, date, time, timezone
from uuid import UUID
from pydantic import BaseModel, Field, model_validator, field_validator, model_serializer, ConfigDict

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

class ClusterStatus(str, Enum):
    ACTIVE = "active"
    PROVISIONAL = "provisional"
    MERGED = "merged"
    SPLIT = "split"
    ARCHIVED = "archived"

# ---------------------------------------------------------------------------
# Maintainable Deep Immutability via Python Standard Library Types
# ---------------------------------------------------------------------------

SUPPORTED_PRIMITIVES = (str, int, float, bool, type(None), datetime, date, time, bytes, UUID, Enum)

def freeze(val: Any, seen: Optional[Set[int]] = None) -> Any:
    """
    Recursively converts mutable mappings and sequences to standard immutable types:
    - dict / Mapping / MappingProxyType -> types.MappingProxyType (fresh copy)
    - list / tuple                      -> tuple (immutable sequence)
    - set / frozenset                   -> frozenset (immutable set)

    Guarantees:
    1. Input aliasing protection: Always deep-copies mapping contents, even if
       the input is already a MappingProxyType, ensuring externally retained
       underlying dictionaries cannot bypass immutability.
    2. Cycle protection: Detects cyclic container references and raises ValueError.
    3. Strict payload validation: Unsupported mutable objects (custom classes,
       bytearrays, etc.) raise TypeError to prevent unverified mutable state.
    """
    if isinstance(val, SUPPORTED_PRIMITIVES):
        return val

    if seen is None:
        seen = set()

    val_id = id(val)
    if val_id in seen:
        raise ValueError("Cyclic reference detected in payload; payloads must be acyclic.")

    # Mappings (including dict, Mapping, and MappingProxyType)
    if isinstance(val, (dict, Mapping)):
        seen.add(val_id)
        try:
            frozen_dict = {
                k: freeze(v, seen)
                for k, v in val.items()
            }
            return MappingProxyType(frozen_dict)
        finally:
            seen.remove(val_id)

    # Sequences (list, tuple)
    elif isinstance(val, (list, tuple)):
        seen.add(val_id)
        try:
            return tuple(freeze(v, seen) for v in val)
        finally:
            seen.remove(val_id)

    # Sets (set, frozenset)
    elif isinstance(val, (set, frozenset)):
        seen.add(val_id)
        try:
            return frozenset(freeze(v, seen) for v in val)
        finally:
            seen.remove(val_id)

    raise TypeError(
        f"Unsupported payload value of type {type(val).__name__}: "
        "only JSON-compatible primitives, datetimes, enums, and container types (mappings, sequences, sets) are allowed."
    )

def unfreeze(val: Any) -> Any:
    """
    Recursively converts immutable views back to standard JSON-compatible Python structures:
    - MappingProxyType -> dict
    - tuple            -> list
    - frozenset        -> list
    """
    if isinstance(val, (MappingProxyType, dict, Mapping)):
        return {k: unfreeze(v) for k, v in val.items()}
    elif isinstance(val, (tuple, list)):
        return [unfreeze(v) for v in val]
    elif isinstance(val, (frozenset, set)):
        return [unfreeze(v) for v in sorted(val, key=str)]
    return val

def _validate_non_empty_str(val: str, field_name: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace string")
    return val

class ImmutableBaseModel(BaseModel):
    """
    Base model enforcing:
    1. Top-level field immutability via ConfigDict(frozen=True).
    2. Deep nested immutability via MappingProxyType and tuple conversion.
    3. Input aliasing protection by deep-converting incoming data.
    4. Plain JSON-compatible serialization via @model_serializer(mode="plain").
    """
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _deep_freeze(self) -> "ImmutableBaseModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            frozen_val = freeze(val)
            if frozen_val is not val:
                object.__setattr__(self, field_name, frozen_val)
        return self

    @model_serializer(mode="plain")
    def _serialize_model(self) -> Dict[str, Any]:
        """Ensures dumped dictionaries and JSON strings output clean native containers."""
        return {k: unfreeze(getattr(self, k)) for k in self.__class__.model_fields}

# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class Source(ImmutableBaseModel):
    source_id: str
    source_type: SourceType
    source_name: str
    source_uri: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    reliability_metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "source_name")
    @classmethod
    def _validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

class Record(ImmutableBaseModel):
    record_id: str
    source_id: str
    entity_type: EntityType
    raw_name: str
    normalized_attributes: Dict[str, Any] = Field(default_factory=dict)
    structured_attributes: Dict[str, Any] = Field(default_factory=dict)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    observed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("record_id", "source_id", "raw_name")
    @classmethod
    def _validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

class CandidatePair(ImmutableBaseModel):
    candidate_id: str
    left_record_id: str
    right_record_id: str
    generation_method: str
    blocking_signals: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_pair(cls, data: Any) -> Any:
        if isinstance(data, dict):
            left = data.get("left_record_id")
            right = data.get("right_record_id")
            if isinstance(left, str) and isinstance(right, str):
                if left.strip() > right.strip():
                    data["left_record_id"], data["right_record_id"] = right, left
        return data

    @field_validator("candidate_id", "left_record_id", "right_record_id", "generation_method")
    @classmethod
    def _validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @model_validator(mode="after")
    def check_not_self_pair(self):
        if self.left_record_id == self.right_record_id:
            raise ValueError("CandidatePair cannot contain identical record IDs")
        return self

    @property
    def canonical_pair_key(self) -> str:
        return f"{self.left_record_id}::{self.right_record_id}"

class Evidence(ImmutableBaseModel):
    evidence_id: str
    evidence_type: str
    polarity: EvidencePolarity
    value_description: str
    source_reference: Optional[str] = None
    record_reference: Optional[str] = None
    provenance: str
    strength: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "evidence_type", "value_description")
    @classmethod
    def _validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("strength")
    @classmethod
    def _validate_strength(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Evidence strength must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def check_provenance(self):
        if not self.provenance or not self.provenance.strip():
            raise ValueError("Evidence must have provenance")
        return self

class MatchDecision(ImmutableBaseModel):
    candidate_pair_id: str
    decision: DecisionType
    score: Optional[float] = None
    threshold: Optional[float] = None
    model_version: str
    evidence_references: List[str] = Field(default_factory=list)
    explanation_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("candidate_pair_id", "model_version")
    @classmethod
    def _validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("score", "threshold")
    @classmethod
    def _validate_scores(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v

class EntityCluster(ImmutableBaseModel):
    entity_cluster_id: str
    entity_type: EntityType
    member_record_ids: List[str]
    status: ClusterStatus
    quality_confidence_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    derivation_version: str

    @field_validator("entity_cluster_id", "derivation_version")
    @classmethod
    def _validate_strings(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            for cs in ClusterStatus:
                if cs.value == v_lower:
                    return cs
            raise ValueError(f"Invalid cluster status: {v}. Must be one of {[s.value for s in ClusterStatus]}")
        return v

    @model_validator(mode="after")
    def check_cluster_members(self):
        if not self.member_record_ids or len(self.member_record_ids) == 0:
            raise ValueError("EntityCluster must contain at least one member record ID")
        for mid in self.member_record_ids:
            _validate_non_empty_str(mid, "member_record_id")
        if len(self.member_record_ids) != len(set(self.member_record_ids)):
            raise ValueError("EntityCluster cannot contain duplicate record IDs")
        return self

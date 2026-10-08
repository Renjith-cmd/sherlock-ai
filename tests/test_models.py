import pytest
from pydantic import ValidationError
from src.core.models import (
    Record, CandidatePair, Evidence, EntityCluster, MatchDecision,
    EntityType, EvidencePolarity, DecisionType, SourceType, Source
)

def test_source_creation():
    s = Source(source_id="s1", source_type=SourceType.URL, source_name="Test Source")
    assert s.source_id == "s1"
    assert s.source_type == SourceType.URL
    
def test_record_creation():
    r = Record(
        record_id="r1", 
        source_id="s1", 
        entity_type=EntityType.PERSON, 
        raw_name="Test Person"
    )
    assert r.record_id == "r1"
    assert r.entity_type == EntityType.PERSON
    
def test_candidate_pair_self_rejection():
    with pytest.raises(ValidationError) as exc_info:
        CandidatePair(
            candidate_id="cp1",
            left_record_id="r1",
            right_record_id="r1",
            generation_method="test"
        )
    assert "CandidatePair cannot contain identical record IDs" in str(exc_info.value)
    
def test_candidate_pair_valid():
    cp = CandidatePair(
        candidate_id="cp1",
        left_record_id="r1",
        right_record_id="r2",
        generation_method="test"
    )
    assert cp.left_record_id == "r1"

def test_evidence_provenance_required():
    with pytest.raises(ValidationError) as exc_info:
        Evidence(
            evidence_id="e1",
            evidence_type="name_match",
            polarity=EvidencePolarity.SUPPORTING,
            value_description="Exact match",
            provenance="   "
        )
    assert "Evidence must have provenance" in str(exc_info.value)

def test_entity_cluster_unique_members():
    with pytest.raises(ValidationError) as exc_info:
        EntityCluster(
            entity_cluster_id="c1",
            entity_type=EntityType.PERSON,
            member_record_ids=["r1", "r2", "r1"],
            status="active",
            derivation_version="v1"
        )
    assert "EntityCluster cannot contain duplicate record IDs" in str(exc_info.value)

def test_match_decision_serialization():
    md = MatchDecision(
        candidate_pair_id="cp1",
        decision=DecisionType.REVIEW,
        model_version="1.0",
    )
    serialized = md.model_dump()
    assert serialized["decision"] == "REVIEW"
    
    deserialized = MatchDecision.model_validate(serialized)
    assert deserialized.decision == DecisionType.REVIEW

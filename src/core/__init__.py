# Core package initialization
from src.core.models import (
    EntityType,
    EvidencePolarity,
    DecisionType,
    SourceType,
    ClusterStatus,
    Source,
    Record,
    CandidatePair,
    Evidence,
    MatchDecision,
    EntityCluster,
)
from src.core.evaluation import (
    UnknownPredictionPolicy,
    EvaluationMetrics,
    PairwiseEvaluator,
    CandidateEvaluationMetrics,
    evaluate_candidate_generation,
)
from src.core.normalization import (
    normalize_text,
    strip_punctuation,
    tokenize_name,
    extract_initials,
    name_fingerprint,
    soundex,
    normalize_attribute_value,
)
from src.core.blocking import (
    BlockingRule,
    ExactNameRule,
    NameTokenRule,
    NameFingerprintRule,
    InitialsAcronymRule,
    NameSoundexRule,
    StructuredAttributeRule,
    CandidateGenerator,
    get_default_blocking_rules,
)

__all__ = [
    # Models
    "EntityType",
    "EvidencePolarity",
    "DecisionType",
    "SourceType",
    "ClusterStatus",
    "Source",
    "Record",
    "CandidatePair",
    "Evidence",
    "MatchDecision",
    "EntityCluster",
    # Evaluation
    "UnknownPredictionPolicy",
    "EvaluationMetrics",
    "PairwiseEvaluator",
    "CandidateEvaluationMetrics",
    "evaluate_candidate_generation",
    # Normalization
    "normalize_text",
    "strip_punctuation",
    "tokenize_name",
    "extract_initials",
    "name_fingerprint",
    "soundex",
    "normalize_attribute_value",
    # Blocking
    "BlockingRule",
    "ExactNameRule",
    "NameTokenRule",
    "NameFingerprintRule",
    "InitialsAcronymRule",
    "NameSoundexRule",
    "StructuredAttributeRule",
    "CandidateGenerator",
    "get_default_blocking_rules",
]

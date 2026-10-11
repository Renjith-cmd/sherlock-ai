# Sherlock AI Core Foundation

This directory contains the foundational, non-destructive data architecture and evaluation engine for Sherlock AI.

---

## Architectural Principles & Design Guarantees

### 1. Immutability & Deep Payload Protection
A `Record` represents an exact empirical observation from a source at a specific point in time. If raw data is modified or overwritten during downstream normalization or entity merging, the original audit trail and provenance are permanently lost.

- **Top-Level Immutability:** All domain models (`Record`, `Source`, `CandidatePair`, `Evidence`, `MatchDecision`, `EntityCluster`) inherit from `ImmutableBaseModel`, configured with Pydantic's `frozen=True` and `extra="forbid"`. Any field reassignment raises a validation error.
- **Deep Payload Protection:** Nested dictionaries, lists, and sets within `raw_payload`, `normalized_attributes`, `structured_attributes`, and `metadata` are recursively converted to Python standard library immutable types (`types.MappingProxyType` for read-only mappings, `tuple` for immutable sequences, `frozenset` for sets). Calling mutating operations (such as `.append()`, `__setitem__`, `.pop()`, or `.update()`) raises a `TypeError` or `AttributeError`.
- **Complete Input Aliasing Decoupling:** Passing mutable objects—or read-only `MappingProxyType` wrappers over externally retained dictionaries—performs a recursive deep copy into fresh immutable views. Mutating the original external dictionary post-construction cannot alter the stored model.
- **Strict Payload Type Validation & Cycle Protection:** Payload values are strictly restricted to JSON-compatible primitives (`str`, `int`, `float`, `bool`, `None`), datetimes, dates, times, bytes, UUIDs, enums, and container types (`Mapping`, `Sequence`, `Set`). Unsupported mutable custom objects raise a `TypeError`. Cyclic structures are detected and rejected with a `ValueError`.
- **Serialization Fidelity:** `ImmutableBaseModel` employs `@model_serializer(mode="plain")` to recursively unfreeze internal containers back to standard Python `dict` and `list` structures, ensuring that `model_dump()` and `model_dump_json()` output clean, JSON-compatible types without leaking internal container wrappers.

### 2. Separation of Raw vs. Normalized Attributes
- `raw_name` and `raw_payload` store unmanipulated data as observed from the underlying data source (e.g., verbatim spelling, original casing, unparsed payload strings).
- `normalized_attributes` and `structured_attributes` store derived, standardized data (e.g., lowercase names, parsed dates, standardized tax/registration IDs).
- Normalization operations must produce new derived structures and must never mutate or overwrite raw observation fields.

### 3. Canonical Candidate Pairs
Pairwise entity comparison is symmetric: comparing `(record_A, record_B)` is logically identical to `(record_B, record_A)`.
- `CandidatePair` normalizes its input record IDs during construction such that `left_record_id <= right_record_id`.
- Constructing `CandidatePair` with `(r1, r2)` or `(r2, r1)` produces the exact same canonical pair with identical `canonical_pair_key` (`"r1::r2"`).
- Candidate pairs with identical IDs (`left_record_id == right_record_id`) are rejected.

### 4. Pairwise Evaluator Policies
The `PairwiseEvaluator` assesses predictions against binary ground-truth labels.

#### A. Ungrounded Predictions (Predictions Outside Ground Truth)
A prediction whose candidate pair ID is absent from ground truth has an unknown label. Fabricating a ground-truth label or silently dropping the prediction creates severe evaluation bias.
- **Default Policy (`UnknownPredictionPolicy.RAISE`):** Raises a `ValueError` listing the unknown pair IDs.
- **Opt-in Policy (`UnknownPredictionPolicy.IGNORE`):** Silently ignores ungrounded pairs (use only for partial sub-graph benchmarks).
- **Opt-in Policy (`UnknownPredictionPolicy.PENALIZE_AS_FP`):** Penalizes any predicted MATCH on an ungrounded pair as a False Positive without fabricating ground-truth labels.

#### B. Missing Predictions
Candidate pairs present in ground truth but omitted from predictions represent candidates filtered out or missed by blocking:
- If the omitted pair was an actual match (`is_actual_match=True`), it counts as a **False Negative** (missed match).
- If the omitted pair was a non-match (`is_actual_match=False`), it counts as a **True Negative** (correct rejection).

#### C. Upstream Duplicate Detection
Dictionary inputs (`Dict[str, DecisionType]`) can mask duplicate predictions produced upstream. Use `PairwiseEvaluator.validate_and_build_predictions(decisions)` on sequences of `MatchDecision` or tuples to detect and reject duplicate pair IDs before evaluation.

#### D. The `REVIEW` Decision State
Entity resolution often encounters ambiguous cases requiring human adjudication. The evaluator supports configurable policies:
- **Default (`review_is_match=False, review_is_error=True`):**
  - REVIEW on a true match counts as a False Negative (missed match).
  - REVIEW on a true non-match counts as a False Positive (false alarm alert).
- **`review_is_match=True`:**
  - REVIEW counts as a MATCH (TP on actual match, FP on actual non-match).
- **`review_is_match=False, review_is_error=False`:**
  - REVIEW on a true match counts as FN.
  - REVIEW on a true non-match counts as TN (neutral alert).

---

## Candidate Generation & Blocking (Milestone 2)

Candidate generation is the indexing layer that proposes plausible record pairs for downstream pairwise comparison. It reduces the $O(N^2)$ Cartesian product of record comparisons into a tractable candidate set while retaining known true matches.

### Core Principles
1. **Candidate Generation is Not Matching:** Blocking does not declare two records to be the same identity. It proposes plausible candidates. Unannotated or non-matching candidate pairs are not classified as errors at this stage.
2. **High Recall with Aggressive Pair Reduction:** Candidate generation must maximize the retention of true matches while pruning unpromising comparisons.
3. **Non-Destructive & Immutable:** Input records are never mutated. No record merging or clustering takes place.
4. **Canonical Ordering:** Emits deduplicated `CandidatePair` instances where `left_record_id < right_record_id`.
5. **Deterministic Execution:** Results and blocking signals are strictly identical regardless of the order in which input records are supplied.
6. **Entity Type Compatibility:** By default (`allow_cross_entity_type=False`), records of different entity types (e.g. `ORGANIZATION` vs `LOCATION`) are never paired.

### Supported Blocking Strategies
- **`ExactNameRule`:** Blocks on exact normalized `raw_name` (Unicode NFKC normalized, casefolded, whitespace collapsed).
- **`NameTokenRule`:** Blocks on individual normalized tokens from `raw_name` (default `min_token_length=2`). Captures partial, shared family, or transposed names.
- **`NameFingerprintRule`:** Blocks on order-invariant token fingerprints (e.g. `"Smith, John"` and `"John Smith"` both map to `"fingerprint:john smith"`).
- **`InitialsAcronymRule`:** Blocks on token initials and uppercase acronyms (e.g. `"International Business Machines"` and `"IBM"` both map to `"acronym:ibm"`).
- **`NameSoundexRule`:** Blocks on American Soundex phonetic codes of Latin name tokens (e.g. `"Sarah Connor"` and `"Sara Connor"` both map to `"soundex:S600"`).
- **`StructuredAttributeRule`:** Blocks on exact normalized values of structured attributes (e.g. `dob`, `phone`, `org`, `city`). Connects aliases sharing non-name identifiers (e.g. `"Marshall Mathers"` and `"Eminem"` sharing `city: "Detroit"`).

### Candidate Recall & Reduction Metrics
Evaluated using `evaluate_candidate_generation()`:
- **Candidate Recall:**
  $$\text{candidate recall} = \frac{\text{known positive pairs generated as candidates}}{\text{all known positive pairs}}$$
  Guarded against zero denominators: returns `0.0` if no positive pairs exist.
- **Pair Reduction Ratio:**
  $$\text{reduction ratio} = 1.0 - \frac{\text{generated candidate pairs}}{\text{total possible pairs}}$$
  where $\text{total possible pairs} = \frac{N(N - 1)}{2}$. Measures the percentage of comparisons pruned from downstream evaluation.

### Usage Example
```python
from src.core.models import Record, EntityType
from src.core.blocking import CandidateGenerator, get_default_blocking_rules
from src.core.evaluation import evaluate_candidate_generation

# 1. Define records
records = [
    Record(record_id="r1", source_id="s1", entity_type=EntityType.PERSON, raw_name="Sarah Connor", structured_attributes={"city": "Los Angeles"}),
    Record(record_id="r2", source_id="s1", entity_type=EntityType.PERSON, raw_name="Sara Connor", structured_attributes={"city": "Los Angeles"}),
    Record(record_id="r3", source_id="s2", entity_type=EntityType.PERSON, raw_name="David Bowman", structured_attributes={"city": "New York"}),
]

# 2. Generate candidate pairs
generator = CandidateGenerator(rules=get_default_blocking_rules())
candidates = generator.generate_candidates(records)

# 3. Evaluate candidate generation against ground truth
ground_truth = {"cp_r1_r2": True}
metrics = evaluate_candidate_generation(candidates, ground_truth, total_records=len(records))

print(f"Candidate Recall: {metrics.candidate_recall:.2%}")      # 100.00%
print(f"Reduction Ratio:  {metrics.reduction_ratio:.2%}")       # 66.67%
```

### Known Limitations & Future Improvements
1. **Script-Specific Phonetics:** Soundex is Latin/English-specific; it does not generate phonetic codes for non-Latin scripts (Cyrillic, Arabic, Chinese). Future milestones can incorporate script-aware phonetic algorithms (e.g., Metaphone or Beider-Morse).
2. **Completely Disjoint Observations:** If two records representing the same entity share no name tokens, no phonetics, no initials, and no recorded attributes, deterministic rule-based blocking will not pair them.
3. **Common Name Over-generation:** Highly frequent tokens (e.g. common first names) can produce large blocks. The `max_block_size` parameter provides a safeguard against combinatorial explosions.

---

## Installation & Running Tests

### 1. Requirements
The core domain models require Pydantic v2. Test suites require pytest.
- Production requirements: `requirements.txt`
- Development & test requirements: `requirements-dev.txt`

### 2. Environment Setup
```bash
# In your virtual environment:
pip install -r requirements-dev.txt
```

### 3. Executing Tests
```bash
# Run all core, normalization, blocking, and integrity tests:
.venv/bin/python -m pytest tests/ -v
```

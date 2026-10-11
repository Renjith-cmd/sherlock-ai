from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Sequence, Tuple, Union
from src.core.models import DecisionType, MatchDecision, CandidatePair

class UnknownPredictionPolicy(str, Enum):
    """
    Policy for handling predictions whose candidate pair IDs are not in ground truth.
    
    - RAISE: Default. Raises a ValueError. Ground truth labels for unknown pairs 
      are unknown; fabricating labels or silently ignoring them is unacceptable.
    - IGNORE: Opt-in. Silently skips predictions not found in ground truth.
    - PENALIZE_AS_FP: Opt-in. Penalizes any predicted MATCH (or REVIEW if review_is_match=True)
      for unknown pairs as a False Positive, without fabricating a ground-truth label.
    """
    RAISE = "RAISE"
    IGNORE = "IGNORE"
    PENALIZE_AS_FP = "PENALIZE_AS_FP"

class EvaluationMetrics:
    """
    Standard binary evaluation metrics for pairwise entity resolution.
    
    All division operations are guarded against zero denominators:
    - If TP + FP == 0, precision returns 0.0.
    - If TP + FN == 0, recall returns 0.0.
    - If precision + recall == 0, F1 returns 0.0.
    """
    def __init__(self, tp: int, fp: int, fn: int, true_negatives: int = 0):
        self.true_positives = tp
        self.false_positives = fp
        self.false_negatives = fn
        self.true_negatives = true_negatives

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

class PairwiseEvaluator:
    """
    Evaluates pairwise match decisions against ground truth.

    Missing Predictions Policy:
    Ground-truth pairs omitted from predictions represent candidate pairs that
    were either blocked or not classified. An omitted true match is penalized as
    a False Negative (missed match). An omitted true non-match counts as a True
    Negative (correct non-candidate).

    Ungrounded Predictions Policy:
    Predictions for candidate pairs not present in ground truth have unknown labels.
    By default (policy=RAISE), a ValueError is raised to prevent silent evaluation bias.
    Alternative policies (IGNORE, PENALIZE_AS_FP) must be explicitly configured.

    REVIEW Policy:
    - By default, review_is_match=False and review_is_error=True:
      * REVIEW on an actual match counts as FN (failed to confirm a true match).
      * REVIEW on an actual non-match counts as FP (unnecessary alert/review burden).
    - If review_is_match=True:
      * REVIEW counts as a MATCH (TP on actual match, FP on actual non-match).
    - If review_is_match=False and review_is_error=False:
      * REVIEW on an actual match counts as FN.
      * REVIEW on an actual non-match counts as TN (not penalized as an alert error).
    """
    def __init__(
        self,
        review_is_match: bool = False,
        review_is_error: bool = True,
        unknown_prediction_policy: UnknownPredictionPolicy = UnknownPredictionPolicy.RAISE
    ):
        self.review_is_match = review_is_match
        self.review_is_error = review_is_error
        self.unknown_prediction_policy = unknown_prediction_policy

    @staticmethod
    def validate_and_build_predictions(
        decisions: Iterable[Union[MatchDecision, Tuple[str, DecisionType]]]
    ) -> Dict[str, DecisionType]:
        """
        Validates an iterable of decisions and constructs a prediction dictionary,
        detecting and rejecting duplicate candidate pair IDs before dictionary conversion.
        """
        seen: Dict[str, DecisionType] = {}
        duplicates: List[str] = []
        for item in decisions:
            if isinstance(item, MatchDecision):
                pair_id = item.candidate_pair_id
                decision = item.decision
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pair_id, decision = item
            else:
                raise TypeError(f"Expected MatchDecision or 2-tuple (pair_id, DecisionType), got {type(item)}")

            if pair_id in seen:
                duplicates.append(pair_id)
            seen[pair_id] = decision

        if duplicates:
            unique_dups = sorted(set(duplicates))
            raise ValueError(f"Duplicate predictions detected for candidate pair IDs: {unique_dups}")
        return seen

    def evaluate(
        self,
        predictions: Union[Dict[str, DecisionType], Iterable[Union[MatchDecision, Tuple[str, DecisionType]]]],
        ground_truth: Dict[str, bool]
    ) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth.

        predictions: Dict[candidate_pair_id, DecisionType] or Iterable of MatchDecision/tuples.
        ground_truth: Dict[candidate_pair_id, bool] (True if actual match, False if non-match).
        """
        if not isinstance(predictions, dict):
            pred_dict = self.validate_and_build_predictions(predictions)
        else:
            pred_dict = predictions

        # Check for predictions with unknown ground-truth labels
        unknown_ids = set(pred_dict.keys()) - set(ground_truth.keys())
        if unknown_ids:
            if self.unknown_prediction_policy == UnknownPredictionPolicy.RAISE:
                sorted_unknown = sorted(list(unknown_ids))[:10]
                raise ValueError(
                    f"Found {len(unknown_ids)} prediction(s) outside ground truth: {sorted_unknown}. "
                    "Actual ground-truth labels are unknown. Configure unknown_prediction_policy "
                    "explicitly if evaluating partial benchmarks."
                )

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for pair_id, is_actual_match in ground_truth.items():
            if pair_id not in pred_dict:
                # Pair was omitted by candidate generation or model
                if is_actual_match:
                    fn += 1
                else:
                    tn += 1
                continue

            decision = pred_dict[pair_id]
            is_pred_match = False
            is_review = False

            if decision == DecisionType.MATCH:
                is_pred_match = True
            elif decision == DecisionType.REVIEW:
                is_review = True
                is_pred_match = self.review_is_match

            if is_actual_match:
                if is_pred_match:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_pred_match:
                    fp += 1
                else:
                    if is_review and self.review_is_error:
                        fp += 1
                    else:
                        tn += 1

        # Apply opt-in policy for ungrounded predictions
        if self.unknown_prediction_policy == UnknownPredictionPolicy.PENALIZE_AS_FP:
            for unk_id in unknown_ids:
                decision = pred_dict[unk_id]
                if decision == DecisionType.MATCH or (decision == DecisionType.REVIEW and self.review_is_match):
                    fp += 1

        return EvaluationMetrics(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Candidate Generation Evaluation (Milestone 2)
# ---------------------------------------------------------------------------

class CandidateEvaluationMetrics:
    """
    Focused metrics for evaluating candidate generation and blocking efficiency.

    Distinctions from Match Evaluation:
    - Candidate recall measures the proportion of true identity matches successfully
      retained in the generated candidate set.
    - Candidate generation proposes plausible pairs; an unannotated or false-match
      candidate pair is NOT treated as a match decision error or false positive.
    - The absence of a candidate is NOT a negative identity decision.
    - Zero denominators are explicitly guarded against.
    """

    def __init__(
        self,
        candidate_pairs_count: int,
        total_possible_pairs: int,
        ground_truth_positives_count: int,
        covered_positives_count: int,
        total_compatible_pairs: Optional[int] = None,
    ):
        self.candidate_pairs_count = candidate_pairs_count
        self.total_possible_pairs = total_possible_pairs
        self.ground_truth_positives_count = ground_truth_positives_count
        self.covered_positives_count = covered_positives_count
        self.total_compatible_pairs = total_compatible_pairs

    @property
    def candidate_recall(self) -> float:
        """
        candidate recall = known positive pairs generated / all known positive pairs
        Guarded: returns 0.0 if ground_truth_positives_count == 0.
        """
        if self.ground_truth_positives_count == 0:
            return 0.0
        return self.covered_positives_count / self.ground_truth_positives_count

    @property
    def reduction_ratio(self) -> float:
        """
        reduction ratio = 1.0 - (candidate_pairs / total_possible_pairs)
        Measures the fraction of pairs pruned from pairwise comparison against
        the full Cartesian product baseline (N * (N - 1) // 2).
        Guarded: returns 0.0 if total_possible_pairs == 0.
        """
        if self.total_possible_pairs == 0:
            return 0.0
        return 1.0 - (self.candidate_pairs_count / self.total_possible_pairs)

    @property
    def compatible_reduction_ratio(self) -> float:
        """
        compatible reduction ratio = 1.0 - (candidate_pairs / total_compatible_pairs)
        Measures the fraction of pairs pruned against the entity-type-compatible
        baseline when cross-type matching is disabled.
        Guarded: returns 0.0 if total_compatible_pairs is None or 0.
        """
        if self.total_compatible_pairs is None or self.total_compatible_pairs == 0:
            return 0.0
        return 1.0 - (self.candidate_pairs_count / self.total_compatible_pairs)

    @property
    def pair_reduction(self) -> float:
        """Alias for reduction_ratio."""
        return self.reduction_ratio

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "candidate_pairs_count": self.candidate_pairs_count,
            "total_possible_pairs": self.total_possible_pairs,
            "ground_truth_positives_count": self.ground_truth_positives_count,
            "covered_positives_count": self.covered_positives_count,
            "candidate_recall": self.candidate_recall,
            "reduction_ratio": self.reduction_ratio,
        }
        if self.total_compatible_pairs is not None:
            d["total_compatible_pairs"] = self.total_compatible_pairs
            d["compatible_reduction_ratio"] = self.compatible_reduction_ratio
        return d


def _extract_pair_signatures(pair: Union[CandidatePair, str]) -> Set[str]:
    """Helper returning canonical identifier aliases for a candidate pair."""
    sigs: Set[str] = set()
    if isinstance(pair, CandidatePair):
        sigs.add(pair.candidate_id)
        sigs.add(pair.canonical_pair_key)
        sigs.add(f"cp_{pair.left_record_id}_{pair.right_record_id}")
        sigs.add(f"cp_{pair.right_record_id}_{pair.left_record_id}")
        sigs.add(f"{pair.left_record_id}::{pair.right_record_id}")
        sigs.add(f"{pair.right_record_id}::{pair.left_record_id}")
    elif isinstance(pair, str):
        sigs.add(pair)
        if pair.startswith("cp_"):
            parts = pair[3:].split("_")
            if len(parts) == 2:
                a, b = parts
                sigs.add(f"{a}::{b}")
                sigs.add(f"{b}::{a}")
                sigs.add(f"cp_{b}_{a}")
        elif "::" in pair:
            parts = pair.split("::")
            if len(parts) == 2:
                a, b = parts
                sigs.add(f"cp_{a}_{b}")
                sigs.add(f"cp_{b}_{a}")
                sigs.add(f"{b}::{a}")
    else:
        raise TypeError(f"Expected CandidatePair or str, got {type(pair).__name__}")
    return sigs


def evaluate_candidate_generation(
    candidate_pairs: Iterable[Union[CandidatePair, str]],
    ground_truth: Dict[str, bool],
    total_records: Optional[Union[int, Sequence[Any]]] = None,
    total_compatible_pairs: Optional[int] = None,
) -> CandidateEvaluationMetrics:
    """
    Evaluates candidate generation efficiency and recall against ground truth.

    candidate_pairs: Iterable of CandidatePair objects or pair ID strings.
    ground_truth: Mapping of candidate_pair_id -> bool (True indicates an actual match).
    total_records: Optional total count of input records (int), or a sequence of Record
                   objects. If a sequence of records is provided, calculates both
                   total_possible_pairs (Cartesian product) and total_compatible_pairs
                   (pairs sharing the same entity_type).
    total_compatible_pairs: Optional precomputed count of entity-type-compatible pairs.
    """
    # Materialize candidate pairs and collect recognized signature aliases
    cand_list = list(candidate_pairs)
    candidate_signatures: Set[str] = set()
    for item in cand_list:
        candidate_signatures.update(_extract_pair_signatures(item))

    # Calculate total possible pairs and compatible pairs
    computed_total_possible = 0
    computed_compatible = total_compatible_pairs

    if total_records is not None:
        if isinstance(total_records, int):
            num_records = total_records
            if num_records >= 2:
                computed_total_possible = num_records * (num_records - 1) // 2
        elif isinstance(total_records, Sequence):
            num_records = len(total_records)
            if num_records >= 2:
                computed_total_possible = num_records * (num_records - 1) // 2
                # Calculate entity-type-compatible pairs if records have entity_type
                if computed_compatible is None and all(hasattr(r, "entity_type") for r in total_records):
                    from collections import Counter
                    type_counts = Counter(r.entity_type for r in total_records)
                    computed_compatible = sum(c * (c - 1) // 2 for c in type_counts.values())
        else:
            num_records = int(total_records)
            if num_records >= 2:
                computed_total_possible = num_records * (num_records - 1) // 2

    # Ground truth evaluation
    ground_truth_positives = 0
    covered_positives = 0

    for gt_id, is_actual_match in ground_truth.items():
        if is_actual_match:
            ground_truth_positives += 1
            # Check if this positive pair was generated under any recognized signature
            gt_sigs = _extract_pair_signatures(gt_id)
            if any(sig in candidate_signatures for sig in gt_sigs):
                covered_positives += 1

    return CandidateEvaluationMetrics(
        candidate_pairs_count=len(cand_list),
        total_possible_pairs=computed_total_possible,
        ground_truth_positives_count=ground_truth_positives,
        covered_positives_count=covered_positives,
        total_compatible_pairs=computed_compatible,
    )

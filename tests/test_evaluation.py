import pytest
from src.core.evaluation import PairwiseEvaluator, EvaluationMetrics, UnknownPredictionPolicy
from src.core.models import DecisionType, MatchDecision

# ---------------------------------------------------------------------------
# 8. Existing TP=2, FP=1, FN=1 Mathematical Example
# ---------------------------------------------------------------------------

def test_pairwise_evaluation_mathematics():
    # Setup a scenario where TP=2, FP=1, FN=1, TN=2
    # Total actual matches = TP + FN = 3
    # Total actual non-matches = FP + TN = 3
    ground_truth = {
        "cp_tp1": True,
        "cp_tp2": True,
        "cp_fn1": True,
        "cp_fp1": False,
        "cp_tn1": False,
        "cp_tn2": False
    }
    
    predictions = {
        "cp_tp1": DecisionType.MATCH,
        "cp_tp2": DecisionType.MATCH,
        "cp_fn1": DecisionType.NON_MATCH,
        "cp_fp1": DecisionType.MATCH,
        "cp_tn1": DecisionType.NON_MATCH,
        "cp_tn2": DecisionType.NON_MATCH
    }
    
    evaluator = PairwiseEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 2
    
    # Precision = TP / (TP + FP) = 2 / 3
    assert abs(metrics.precision - 2/3) < 1e-5
    
    # Recall = TP / (TP + FN) = 2 / 3
    assert abs(metrics.recall - 2/3) < 1e-5
    
    # F1 = 2 * (P * R) / (P + R) = 2/3
    assert abs(metrics.f1 - 2/3) < 1e-5

# ---------------------------------------------------------------------------
# 1. Predictions Containing Unknown Pair IDs
# ---------------------------------------------------------------------------

def test_evaluation_unknown_pair_ids_raise_by_default():
    ground_truth = {"cp_known": True}
    predictions = {
        "cp_known": DecisionType.MATCH,
        "cp_unknown_1": DecisionType.MATCH,
        "cp_unknown_2": DecisionType.NON_MATCH
    }
    evaluator = PairwiseEvaluator()
    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(predictions, ground_truth)
    assert "outside ground truth" in str(exc_info.value)
    assert "cp_unknown_1" in str(exc_info.value)

def test_evaluation_unknown_pair_ids_opt_in_policies():
    ground_truth = {"cp_known": True}
    predictions = {
        "cp_known": DecisionType.MATCH,
        "cp_unknown": DecisionType.MATCH
    }
    # Opt-in policy: IGNORE
    eval_ignore = PairwiseEvaluator(unknown_prediction_policy=UnknownPredictionPolicy.IGNORE)
    m_ignore = eval_ignore.evaluate(predictions, ground_truth)
    assert m_ignore.true_positives == 1
    assert m_ignore.false_positives == 0

    # Opt-in policy: PENALIZE_AS_FP
    eval_penalize = PairwiseEvaluator(unknown_prediction_policy=UnknownPredictionPolicy.PENALIZE_AS_FP)
    m_penalize = eval_penalize.evaluate(predictions, ground_truth)
    assert m_penalize.true_positives == 1
    assert m_penalize.false_positives == 1  # The ungrounded MATCH is penalized as FP

# ---------------------------------------------------------------------------
# 2. Missing Predictions for Known Ground-Truth Pairs
# ---------------------------------------------------------------------------

def test_evaluation_missing_predictions_handling():
    ground_truth = {
        "cp_match": True,
        "cp_non_match": False
    }
    # Predictions dictionary completely omits both pairs
    predictions = {}
    evaluator = PairwiseEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)

    # Missing true match must count as False Negative (missed match)
    assert metrics.false_negatives == 1
    # Missing true non-match counts as True Negative (correct rejection)
    assert metrics.true_negatives == 1
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0

# ---------------------------------------------------------------------------
# 3. Empty Ground Truth and Empty Predictions
# ---------------------------------------------------------------------------

def test_evaluation_empty_inputs():
    evaluator = PairwiseEvaluator()
    metrics = evaluator.evaluate({}, {})
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.true_negatives == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0

# ---------------------------------------------------------------------------
# 4. Zero-Denominator Metric Cases
# ---------------------------------------------------------------------------

def test_evaluation_zero_denominator_safeties():
    # TP=0, FP=0, FN=2 (no positive predictions made, all true matches missed)
    m1 = EvaluationMetrics(tp=0, fp=0, fn=2, true_negatives=2)
    assert m1.precision == 0.0  # Denominator (TP + FP) == 0
    assert m1.recall == 0.0
    assert m1.f1 == 0.0

    # TP=0, FP=2, FN=0 (no true matches exist, model predicted 2 matches)
    m2 = EvaluationMetrics(tp=0, fp=2, fn=0, true_negatives=2)
    assert m2.precision == 0.0
    assert m2.recall == 0.0     # Denominator (TP + FN) == 0
    assert m2.f1 == 0.0

# ---------------------------------------------------------------------------
# 5 & 6. REVIEW on True Match and True Non-Match
# ---------------------------------------------------------------------------

def test_evaluation_review_handling():
    ground_truth = {
        "cp_match": True,
        "cp_non_match": False
    }
    predictions = {
        "cp_match": DecisionType.REVIEW,
        "cp_non_match": DecisionType.REVIEW
    }

    # Config 1: Default (review_is_match=False, review_is_error=True)
    eval1 = PairwiseEvaluator(review_is_match=False, review_is_error=True)
    m1 = eval1.evaluate(predictions, ground_truth)
    assert m1.true_positives == 0
    assert m1.false_negatives == 1  # Missed match
    assert m1.false_positives == 1  # False alarm on non-match

    # Config 2: Review is treated as match (review_is_match=True)
    eval2 = PairwiseEvaluator(review_is_match=True)
    m2 = eval2.evaluate(predictions, ground_truth)
    assert m2.true_positives == 1
    assert m2.false_positives == 1

    # Config 3: Review is not match, but non-match review not penalized (review_is_error=False)
    eval3 = PairwiseEvaluator(review_is_match=False, review_is_error=False)
    m3 = eval3.evaluate(predictions, ground_truth)
    assert m3.false_positives == 0
    assert m3.true_negatives == 1
    assert m3.false_negatives == 1

# ---------------------------------------------------------------------------
# 7. Duplicate Logical Candidate Pairs
# ---------------------------------------------------------------------------

def test_evaluation_duplicate_predictions_rejected():
    decisions = [
        MatchDecision(candidate_pair_id="cp_1", decision=DecisionType.MATCH, model_version="v1"),
        MatchDecision(candidate_pair_id="cp_1", decision=DecisionType.NON_MATCH, model_version="v1")
    ]
    evaluator = PairwiseEvaluator()
    ground_truth = {"cp_1": True}

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(decisions, ground_truth)
    assert "Duplicate predictions detected" in str(exc_info.value)
    assert "cp_1" in str(exc_info.value)

def test_validate_and_build_predictions_tuples():
    pairs = [
        ("cp_1", DecisionType.MATCH),
        ("cp_1", DecisionType.REVIEW)
    ]
    with pytest.raises(ValueError) as exc_info:
        PairwiseEvaluator.validate_and_build_predictions(pairs)
    assert "Duplicate predictions detected" in str(exc_info.value)

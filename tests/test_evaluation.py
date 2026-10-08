import pytest
from src.core.evaluation import PairwiseEvaluator
from src.core.models import DecisionType

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
    
    # F1 = 2 * (P * R) / (P + R) = 2 * (4/9) / (4/3) = (8/9) * (3/4) = 24 / 36 = 2/3
    assert abs(metrics.f1 - 2/3) < 1e-5

def test_evaluation_review_handling():
    # Test how REVIEW is handled based on config
    ground_truth = {
        "cp_match": True,
        "cp_non_match": False
    }
    
    predictions = {
        "cp_match": DecisionType.REVIEW,
        "cp_non_match": DecisionType.REVIEW
    }
    
    # Config 1: Default (Review is neither match nor error-free, review on true match is FN, review on true non-match is FP)
    eval1 = PairwiseEvaluator(review_is_match=False, review_is_error=True)
    m1 = eval1.evaluate(predictions, ground_truth)
    assert m1.true_positives == 0
    assert m1.false_negatives == 1  # Missed the true match
    assert m1.false_positives == 1  # False alarm on non-match
    
    # Config 2: Review is treated as match
    eval2 = PairwiseEvaluator(review_is_match=True)
    m2 = eval2.evaluate(predictions, ground_truth)
    assert m2.true_positives == 1
    assert m2.false_positives == 1
    
    # Config 3: Review is not match, but we don't penalize review on non-match
    eval3 = PairwiseEvaluator(review_is_match=False, review_is_error=False)
    m3 = eval3.evaluate(predictions, ground_truth)
    assert m3.false_positives == 0
    assert m3.true_negatives == 1

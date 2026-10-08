from typing import List, Dict, Tuple
from src.core.models import DecisionType

class EvaluationMetrics:
    def __init__(self, tp: int, fp: int, fn: int, true_negatives: int = 0):
        self.true_positives = tp
        self.false_positives = fp
        self.false_negatives = fn
        self.true_negatives = true_negatives

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

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
    
    Configuration for REVIEW state:
    By default, a REVIEW decision is not considered a MATCH.
    If review_is_error is True, a REVIEW decision on a true positive pair
    counts as a false negative (missed opportunity), and a REVIEW decision 
    on a true negative pair counts as a false positive (false alarm).
    This can be configured explicitly based on how the system treats REVIEWs.
    """
    def __init__(self, review_is_match: bool = False, review_is_error: bool = True):
        self.review_is_match = review_is_match
        self.review_is_error = review_is_error

    def evaluate(self, predictions: Dict[str, DecisionType], ground_truth: Dict[str, bool]) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth.
        predictions: candidate_pair_id -> DecisionType
        ground_truth: candidate_pair_id -> boolean (True if they actually match)
        """
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for pair_id, is_actual_match in ground_truth.items():
            if pair_id not in predictions:
                # Missing prediction counts as a false negative if it was a match.
                if is_actual_match:
                    fn += 1
                else:
                    tn += 1
                continue
                
            decision = predictions[pair_id]
            
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
                    # If it's a review and review doesn't count as match, it's a False Negative 
                    # (we failed to confidently match them).
                    # If review_is_error is False, we might just ignore it, but standard metric 
                    # requires it to be a FN if not TP.
                    fn += 1
            else:
                if is_pred_match:
                    fp += 1
                else:
                    # If it's a review and review isn't a match.
                    # Should it be penalized as a FP if review_is_error is True? 
                    # Usually, a REVIEW on a Non-Match is an uncertainty false alarm.
                    if is_review and self.review_is_error:
                        fp += 1
                    else:
                        tn += 1
                    
        return EvaluationMetrics(tp, fp, fn, tn)

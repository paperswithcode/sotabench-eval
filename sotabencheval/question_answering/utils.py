from sotabencheval.question_answering.evaluate_v11 import evaluate as evaluate_v11
from sotabencheval.question_answering.evaluate_v20 import get_raw_scores as get_raw_scores_v20

__all__ = ["evaluate_v11", "evaluate_v20"]


def evaluate_v20(dataset, predictions):
    exact_match, f1 = get_raw_scores_v20(dataset, predictions)
    return {'exact_match': exact_match, 'f1': f1}

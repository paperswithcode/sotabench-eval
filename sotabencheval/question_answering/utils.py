from sotabencheval.question_answering.evaluate_v11 import evaluate as evaluate_v11
from sotabencheval.question_answering.evaluate_v20 import get_raw_scores

__all__ = ["evaluate_v11", "evaluate_v20"]


def evaluate_v20(dataset, predictions):
    exact_scores, f1_scores = get_raw_scores(dataset, predictions)
    total = len(dataset)
    exact_match = 100.0 * sum(exact_scores.values()) / total
    f1 = 100.0 * sum(f1_scores.values()) / total
    return {'exact_match': exact_match, 'f1': f1_scores}

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def score_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Accuracy plus macro and weighted precision, recall and F1."""
    scores = {"accuracy": accuracy_score(labels, predictions)}

    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        scores[f"precision_{average}"] = precision
        scores[f"recall_{average}"] = recall
        scores[f"f1_{average}"] = f1

    return scores


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Trainer entry point: argmax the logits, then score."""
    logits, labels = eval_pred

    return score_predictions(labels, np.argmax(logits, axis=-1))

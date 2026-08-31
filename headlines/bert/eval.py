"""Metric computation for BERT headline classification."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy and weighted F1 from Trainer predictions.

    Args:
        eval_pred: tuple of raw model logits and integer label ids.

    Returns:
        mapping of metric name to score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

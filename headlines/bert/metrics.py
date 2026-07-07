"""Classification metrics and inference helpers for the BERT pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Accuracy in ``[0.0, 1.0]``.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1_score(
    y_true: Sequence[int], y_pred: Sequence[int], average: str = "weighted"
) -> float:
    """Compute the F1 score with a configurable averaging strategy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        average: Averaging strategy passed to sklearn (``"weighted"``,
            ``"macro"``, ``"micro"``).

    Returns:
        F1 score in ``[0.0, 1.0]``.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        labels: Optional full label set; pass it to keep the matrix a fixed
            size even when a class is absent from a small evaluation split.

    Returns:
        2-D ndarray confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=list(labels) if labels is not None else None)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Compute a bundle of classification metrics.

    Reports accuracy and both weighted and macro precision/recall/F1 so class
    imbalance (the Neutral class dominates this dataset) is visible at a glance.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Mapping of metric name to score.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def classification_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: Sequence[str] | None = None,
) -> str:
    """Return sklearn's per-class precision/recall/F1 text report.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        target_names: Optional class names, indexed by integer label.

    Returns:
        A formatted multi-line report string.
    """
    names = list(target_names) if target_names is not None else None
    # Pin ``labels`` to the full class set so the report stays complete even
    # when a class is missing from a small evaluation split.
    labels = list(range(len(names))) if names is not None else None
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        zero_division=0,
    )


def get_predictions(
    model: torch.nn.Module, dataloader, device
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``dataloader`` and collect predictions and labels.

    Args:
        model: Trained BERT classifier.
        dataloader: DataLoader yielding the dict produced by ``HeadlineDataset``.
        device: Torch device used for inference.

    Returns:
        Tuple of (headlines, y_pred, y_true) where ``headlines`` is a list of
        raw strings and the prediction/label tensors are 1-D ``long`` tensors.
    """
    model.eval()

    headlines: list[str] = []
    y_pred: list[int] = []
    y_true: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            headlines.extend(batch["text"])
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    return headlines, torch.tensor(y_pred, dtype=torch.long), torch.tensor(y_true, dtype=torch.long)

import numpy as np
import pytest

from headlines.bert.metrics import (
    classification_summary,
    compute_accuracy,
    compute_confusion_matrix,
    compute_f1_score,
    compute_metrics,
)


@pytest.fixture
def labels_fixture():
    """Dummy labels for metric tests."""
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 2, 2, 1, 0]
    return y_true, y_pred


class TestMetrics:
    """Metric functions for the BERT classifier."""

    def test_accuracy(self, labels_fixture):
        y_true, y_pred = labels_fixture
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.8)

    def test_confusion_matrix(self, labels_fixture):
        y_true, y_pred = labels_fixture
        expected = np.array([[2, 0, 0], [0, 1, 1], [0, 0, 1]])
        assert np.array_equal(compute_confusion_matrix(y_true, y_pred), expected)

    def test_f1_weighted(self, labels_fixture):
        y_true, y_pred = labels_fixture
        assert compute_f1_score(y_true, y_pred) == pytest.approx(0.8)

    def test_f1_macro_average_arg(self, labels_fixture):
        y_true, y_pred = labels_fixture
        macro = compute_f1_score(y_true, y_pred, average="macro")
        assert 0.0 <= macro <= 1.0

    def test_compute_metrics_bundle(self, labels_fixture):
        y_true, y_pred = labels_fixture
        metrics = compute_metrics(y_true, y_pred)
        expected_keys = {
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        }
        assert set(metrics) == expected_keys
        assert all(isinstance(v, float) for v in metrics.values())
        assert metrics["accuracy"] == pytest.approx(0.8)

    def test_classification_summary_is_text(self, labels_fixture):
        y_true, y_pred = labels_fixture
        report = classification_summary(
            y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
        assert "Negative" in report
        assert "precision" in report

import numpy as np

from headlines.bert.eval import compute_metrics


LABELS = np.array([0, 1, 2])

CONFIDENT_LOGITS = np.array(
    [
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
)


class TestComputeMetrics:
    """test scores classifier predictions against their labels"""

    def test_reports_accuracy_and_f1(self):
        """test the Trainer gets exactly the two metrics the config names"""
        assert set(compute_metrics((CONFIDENT_LOGITS, LABELS))) == {"accuracy", "f1"}

    def test_perfect_predictions_score_one(self):
        """test logits that pick every label correctly score at the top"""
        scores = compute_metrics((CONFIDENT_LOGITS, LABELS))

        assert scores["accuracy"] == 1.0
        assert scores["f1"] == 1.0

    def test_all_wrong_predictions_score_zero(self):
        """test shifting every winning column away drops both metrics to zero"""
        scores = compute_metrics((np.roll(CONFIDENT_LOGITS, 1, axis=1), LABELS))

        assert scores["accuracy"] == 0.0
        assert scores["f1"] == 0.0

    def test_reads_the_largest_logit_not_the_first(self):
        """test the argmax runs across labels rather than across the batch"""
        half_right = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

        assert compute_metrics((half_right, LABELS))["accuracy"] == 2 / 3

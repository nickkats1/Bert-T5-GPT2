import numpy as np

from bert.metrics import compute_metrics, score_predictions


class TestScorePredictions:
    def test_perfect_predictions_score_one(self):
        labels = np.array([0, 1, 2, 1])

        assert score_predictions(labels, labels)["accuracy"] == 1.0

    def test_reports_macro_and_weighted_variants(self):
        labels = np.array([0, 1, 2])
        scores = score_predictions(labels, labels)

        assert set(scores) == {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        }

    def test_macro_punishes_ignoring_a_minority_class(self):
        labels = np.array([1, 1, 1, 1, 0, 2])
        always_neutral = np.array([1, 1, 1, 1, 1, 1])
        scores = score_predictions(labels, always_neutral)

        assert scores["f1_macro"] < scores["f1_weighted"]

    def test_every_score_is_a_fraction(self):
        labels = np.array([0, 1, 2, 0])
        scores = score_predictions(labels, np.array([0, 1, 1, 2]))

        assert all(0.0 <= value <= 1.0 for value in scores.values())


class TestComputeMetrics:
    def test_argmaxes_logits_before_scoring(self):
        logits = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
        labels = np.array([0, 1, 2])

        assert compute_metrics((logits, labels))["accuracy"] == 1.0

    def test_returns_the_same_keys_as_score_predictions(self):
        logits = np.zeros((3, 3))
        labels = np.array([0, 1, 2])

        assert set(compute_metrics((logits, labels))) == set(score_predictions(labels, labels))

import csv

import numpy as np
import pytest

from headlines.bert.config import BertCFG
from headlines.bert.predictions import (
    load_trained,
    predict,
    predict_logits,
    score_rows,
    write_predictions,
)


HEADLINES = [
    "Sunak unveils a budget aimed at growth",
    "Rail strikes to disrupt travel this weekend",
]


class TestLoadTrained:
    """test loads the fine-tuned checkpoint from disk"""

    def test_missing_checkpoint_names_the_directory(self, tmp_path):
        """test the error tells you training has not been run"""
        missing = tmp_path / "never-trained"

        with pytest.raises(FileNotFoundError, match=str(missing)):
            load_trained(output_dir=str(missing))


class TestPredictLogits:
    """test runs raw headlines through the model and keeps the scores"""

    @pytest.fixture
    def logits(self, bert_model, bert_tokenizer):
        """Raw scores from the untrained head."""
        return predict_logits(HEADLINES, bert_model, bert_tokenizer)

    def test_one_row_per_headline_and_one_column_per_label(self, logits):
        """test the array lines up with what score_rows expects"""
        assert logits.shape == (len(HEADLINES), BertCFG.num_labels)

    def test_batching_does_not_lose_rows(self, bert_model, bert_tokenizer):
        """test a batch smaller than the input still returns every headline"""
        batched = predict_logits(HEADLINES, bert_model, bert_tokenizer, batch_size=1)

        assert batched.shape == (len(HEADLINES), BertCFG.num_labels)


class TestScoreRows:
    """test turns raw scores into a label and a confidence"""

    @pytest.fixture
    def rows(self):
        """Rows scored from handmade logits."""
        logits = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

        return score_rows(HEADLINES, logits)

    def test_picks_the_largest_logit(self, rows):
        """test the winning column becomes the predicted label"""
        assert [row["pred_label"] for row in rows] == [0, 2]

    def test_scores_are_probabilities(self, rows):
        """test confidences fall inside a valid probability range"""
        assert all(0.0 <= row["score"] <= 1.0 for row in rows)

    def test_keeps_the_headline(self, rows):
        """test each row can be traced back to its text"""
        assert [row["Headlines"] for row in rows] == HEADLINES


class TestPredict:
    """test classifies raw headlines with a model"""

    @pytest.fixture
    def rows(self, bert_model, bert_tokenizer):
        """Predictions over sample headlines from an untrained head."""
        return predict(HEADLINES, bert_model, bert_tokenizer)

    def test_one_row_per_headline(self, rows):
        """test nothing is dropped or duplicated"""
        assert len(rows) == len(HEADLINES)

    def test_labels_are_in_range(self, rows):
        """test predictions land inside the configured label set"""
        assert all(row["pred_label"] in range(3) for row in rows)


class TestWritePredictions:
    """test writes scored headlines to csv"""

    def test_writes_expected_columns(self, tmp_path):
        """test the csv carries true and predicted labels side by side"""
        path = tmp_path / "predictions.csv"
        rows = [{"Headlines": HEADLINES[0], "true_label": 2, "pred_label": 2, "score": 0.9}]

        write_predictions(rows, path=str(path))

        with open(path, newline="", encoding="utf-8") as f:
            written = list(csv.DictReader(f))

        assert written[0]["Headlines"] == HEADLINES[0]
        assert written[0]["true_label"] == "2"
        assert written[0]["pred_label"] == "2"

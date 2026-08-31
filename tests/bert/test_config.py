from dataclasses import FrozenInstanceError

import pytest

from headlines.bert.config import BertCFG


class TestBertCFG:
    """test the classification settings every other bert module reads"""

    def test_fields_cannot_be_reassigned(self):
        """test the config is a shared constant rather than mutable state"""
        config = BertCFG()

        with pytest.raises(FrozenInstanceError):
            config.num_labels = 5

    def test_num_labels_matches_the_data(self, guardian_rows):
        """test the head is sized to the labels build_label_maps will find"""
        assert BertCFG.num_labels == len({label for _, label in guardian_rows})

    def test_best_model_tracks_rising_accuracy(self):
        """test the winning checkpoint is the most accurate one, not the least"""
        assert BertCFG.metric_for_best_model == "accuracy"
        assert BertCFG.greater_is_better

    def test_data_path_points_at_a_csv(self):
        """test load_csv is handed something it can read"""
        assert BertCFG.data_path.endswith(".csv")

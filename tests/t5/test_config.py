from dataclasses import FrozenInstanceError

import pytest

from headlines.t5.config import T5CFG


class TestT5CFG:
    """test the summarization settings every other t5 module reads"""

    def test_fields_cannot_be_reassigned(self):
        """test the config is a shared constant rather than mutable state"""
        config = T5CFG()

        with pytest.raises(FrozenInstanceError):
            config.max_target_length = 8

    def test_headlines_are_shorter_than_descriptions(self):
        """test the target budget leaves room for a headline, not an article"""
        assert T5CFG.max_target_length < T5CFG.max_source_length

    def test_source_prefix_separates_from_the_text(self):
        """test the task prefix does not run into the first word"""
        assert T5CFG.source_prefix.endswith(" ")

    def test_evaluation_scores_generated_text(self):
        """test rouge sees decoded headlines rather than raw logits"""
        assert T5CFG.predict_with_generate
        assert T5CFG.metric_for_best_model == "rouge1"
        assert T5CFG.greater_is_better

    def test_data_path_points_at_a_csv(self):
        """test load_csv is handed something it can read"""
        assert T5CFG.data_path.endswith(".csv")

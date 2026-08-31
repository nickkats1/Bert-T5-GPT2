from dataclasses import FrozenInstanceError

import pytest

from headlines.gpt2.config import Gpt2CFG


class TestGpt2CFG:
    """test the generation settings every other gpt2 module reads"""

    def test_fields_cannot_be_reassigned(self):
        """test the config is a shared constant rather than mutable state"""
        config = Gpt2CFG()

        with pytest.raises(FrozenInstanceError):
            config.max_new_tokens = 8

    def test_pad_token_is_not_native_to_gpt2(self, gpt2_tokenizer):
        """test the pad token had to be added, which is why build_model resizes"""
        assert Gpt2CFG.pad_token in gpt2_tokenizer.get_added_vocab()

    def test_best_model_tracks_falling_loss(self):
        """test the winning checkpoint is the one with the lowest loss"""
        assert Gpt2CFG.metric_for_best_model == "loss"
        assert not Gpt2CFG.greater_is_better

    def test_data_path_points_at_a_csv(self):
        """test load_csv is handed something it can read"""
        assert Gpt2CFG.data_path.endswith(".csv")

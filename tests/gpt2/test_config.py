import dataclasses

import pytest

from gpt2.config import CLM, ClmDataArguments, ClmModelArguments, training_arguments


class TestClmModelArguments:
    def test_defaults_to_a_gpt2_checkpoint(self):
        assert ClmModelArguments().model_name_or_path == "gpt2"

    def test_carries_a_pad_token_distinct_from_eos(self):
        assert ClmModelArguments().pad_token == "<|pad|>"

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ClmModelArguments().model_name_or_path = "other"


class TestClmDataArguments:
    def test_points_at_the_reuters_csv(self):
        assert ClmDataArguments.data_path == "data/reuters_headlines.csv"

    def test_trains_on_headlines_only(self):
        assert ClmDataArguments.text_column == "Headlines"


class TestTrainingArguments:
    def test_every_setting_is_a_real_field(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path))

        assert all(hasattr(args, name) for name in CLM)

    def test_settings_reach_the_arguments(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path))
        normalised = {"report_to"}

        for name, value in CLM.items():
            if name not in normalised:
                assert getattr(args, name) == value

    def test_reporting_is_switched_off(self, tmp_path):
        assert training_arguments(output_dir=str(tmp_path)).report_to == []

    def test_selects_the_lowest_loss_checkpoint(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path))

        assert args.metric_for_best_model == "loss"
        assert args.greater_is_better is False

    def test_never_asks_for_fp16(self, tmp_path):
        assert training_arguments(output_dir=str(tmp_path)).fp16 is False

    def test_overrides_win_over_settings(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path), num_train_epochs=1, max_steps=5)

        assert args.num_train_epochs == 1
        assert args.max_steps == 5

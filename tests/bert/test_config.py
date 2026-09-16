import dataclasses

import pytest

from bert.config import (
    CLASSIFICATION,
    ClassificationDataArguments,
    ClassificationModelArguments,
    training_arguments,
)


class TestClassificationModelArguments:
    def test_defaults_to_a_bert_checkpoint(self):
        assert ClassificationModelArguments().model_name_or_path == "bert-base-uncased"

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ClassificationModelArguments().model_name_or_path = "other"


class TestClassificationDataArguments:
    def test_points_at_the_guardian_csv(self):
        assert ClassificationDataArguments.data_path == "data/guardian_headlines.csv"

    def test_splits_leave_eighty_percent_for_training(self):
        assert ClassificationDataArguments.test_size == 0.2


class TestTrainingArguments:
    def test_every_setting_is_a_real_field(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path))

        assert all(hasattr(args, name) for name in CLASSIFICATION)

    def test_settings_reach_the_arguments(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path))
        normalised = {"report_to"}

        for name, value in CLASSIFICATION.items():
            if name not in normalised:
                assert getattr(args, name) == value

    def test_reporting_is_switched_off(self, tmp_path):
        assert training_arguments(output_dir=str(tmp_path)).report_to == []

    def test_selects_the_weighted_f1_checkpoint(self, tmp_path):
        assert training_arguments(output_dir=str(tmp_path)).metric_for_best_model == "f1_weighted"

    def test_trains_in_fp16(self, tmp_path):
        assert training_arguments(output_dir=str(tmp_path)).fp16 is True

    def test_overrides_win_over_settings(self, tmp_path):
        args = training_arguments(output_dir=str(tmp_path), num_train_epochs=1, max_steps=5)

        assert args.num_train_epochs == 1
        assert args.max_steps == 5

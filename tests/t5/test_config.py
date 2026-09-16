import dataclasses

import pytest

from t5.config import (
    SUMMARIZATION,
    SummarizationDataArguments,
    SummarizationModelArguments,
    seq2seq_arguments,
)


class TestSummarizationModelArguments:
    def test_defaults_to_a_t5_checkpoint(self):
        assert SummarizationModelArguments().model_name_or_path == "t5-base"

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SummarizationModelArguments().model_name_or_path = "other"


class TestSummarizationDataArguments:
    def test_points_at_the_reuters_csv(self):
        assert SummarizationDataArguments.data_path == "data/reuters_headlines.csv"

    def test_summarizes_descriptions_into_headlines(self):
        assert SummarizationDataArguments.source_column == "Description"
        assert SummarizationDataArguments.target_column == "Headlines"

    def test_carries_the_task_prefix_t5_expects(self):
        assert SummarizationDataArguments.source_prefix == "summarize: "

    def test_targets_are_shorter_than_sources(self):
        assert SummarizationDataArguments.max_target_length < SummarizationDataArguments.max_source_length


class TestSeq2SeqArguments:
    def test_every_setting_is_a_real_field(self, tmp_path):
        args = seq2seq_arguments(output_dir=str(tmp_path))

        assert all(hasattr(args, name) for name in SUMMARIZATION)

    def test_settings_reach_the_arguments(self, tmp_path):
        args = seq2seq_arguments(output_dir=str(tmp_path))
        normalised = {"report_to"}

        for name, value in SUMMARIZATION.items():
            if name not in normalised:
                assert getattr(args, name) == value

    def test_reporting_is_switched_off(self, tmp_path):
        assert seq2seq_arguments(output_dir=str(tmp_path)).report_to == []

    def test_generates_during_evaluation(self, tmp_path):
        assert seq2seq_arguments(output_dir=str(tmp_path)).predict_with_generate is True

    def test_selects_on_rouge(self, tmp_path):
        assert seq2seq_arguments(output_dir=str(tmp_path)).metric_for_best_model == "rouge1"

    def test_never_asks_for_fp16(self, tmp_path):
        assert seq2seq_arguments(output_dir=str(tmp_path)).fp16 is False

    def test_generation_length_matches_the_target_length(self, tmp_path):
        args = seq2seq_arguments(output_dir=str(tmp_path))

        assert args.generation_max_length == SummarizationDataArguments.max_target_length

    def test_overrides_win_over_settings(self, tmp_path):
        args = seq2seq_arguments(output_dir=str(tmp_path), num_train_epochs=1, generation_num_beams=4)

        assert args.num_train_epochs == 1
        assert args.generation_num_beams == 4

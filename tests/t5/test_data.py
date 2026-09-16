import dataclasses

import pandas as pd
import pytest

from t5.data import build_datasets, load_csv, split_frame


class TestLoadCSV:
    def test_keeps_only_the_requested_columns(self, temp_reuters_file):
        frame = load_csv(temp_reuters_file, ["Description", "Headlines"])

        assert list(frame.columns) == ["Description", "Headlines"]

    def test_drops_rows_missing_either_side(self, tmp_path):
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"Description": ["text", None], "Headlines": ["head", "head2"]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Description", "Headlines"])) == 1

    def test_drops_duplicate_pairs(self, tmp_path):
        path = tmp_path / "dupes.csv"
        pd.DataFrame({"Description": ["a", "a", "b"], "Headlines": ["x", "x", "y"]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Description", "Headlines"])) == 2

    def test_an_absent_column_is_rejected(self, temp_reuters_file):
        with pytest.raises(KeyError):
            load_csv(temp_reuters_file, ["Nonexistent"])


class TestSplitFrame:
    def test_produces_two_named_splits(self, t5_data_args):
        assert set(split_frame(t5_data_args, seed=42)) == {"train", "validation"}

    def test_leaves_both_sides_untokenized(self, t5_data_args):
        splits = split_frame(t5_data_args, seed=42)

        assert t5_data_args.source_column in splits["train"].column_names
        assert t5_data_args.target_column in splits["train"].column_names
        assert "input_ids" not in splits["train"].column_names


@pytest.fixture
def t5_splits(t5_data_args, t5_tokenizer):
    """The two splits built from the temporary Reuters CSV."""
    return build_datasets(t5_data_args, t5_tokenizer, seed=42)


class TestBuildDatasets:
    def test_produces_two_named_splits(self, t5_splits):
        assert set(t5_splits) == {"train", "validation"}

    def test_keeps_every_row(self, t5_data_args, t5_splits):
        expected = len(load_csv(t5_data_args.data_path, [t5_data_args.source_column, t5_data_args.target_column]))

        assert sum(len(split) for split in t5_splits.values()) == expected

    def test_drops_the_raw_text_columns(self, t5_splits):
        columns = t5_splits["train"].column_names

        assert "Description" not in columns
        assert "Headlines" not in columns

    def test_carries_inputs_and_labels(self, t5_splits):
        columns = t5_splits["train"].column_names

        assert "input_ids" in columns
        assert "labels" in columns

    def test_respects_the_source_truncation_length(self, t5_data_args, t5_splits):
        assert all(len(ids) <= t5_data_args.max_source_length for ids in t5_splits["train"]["input_ids"])

    def test_respects_the_target_truncation_length(self, t5_data_args, t5_splits):
        assert all(len(ids) <= t5_data_args.max_target_length for ids in t5_splits["train"]["labels"])

    def test_max_eval_samples_caps_the_validation_split(self, t5_data_args, t5_tokenizer):
        capped = dataclasses.replace(t5_data_args, max_eval_samples=1)
        splits = build_datasets(capped, t5_tokenizer, seed=42)

        assert len(splits["validation"]) == 1

    def test_the_task_prefix_reaches_the_encoded_source(self, t5_data_args, t5_tokenizer, t5_splits):
        decoded = t5_tokenizer.decode(t5_splits["train"][0]["input_ids"], skip_special_tokens=True)

        assert decoded.startswith(t5_data_args.source_prefix.strip())

    def test_is_reproducible_for_a_seed(self, t5_data_args, t5_tokenizer):
        first = build_datasets(t5_data_args, t5_tokenizer, seed=42)
        second = build_datasets(t5_data_args, t5_tokenizer, seed=42)

        assert first["validation"]["labels"] == second["validation"]["labels"]

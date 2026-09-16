import pandas as pd
import pytest

from bert.config import ClassificationDataArguments
from bert.data import build_datasets, load_csv, split_frame
from bert.utils_bert import ID2LABEL


class TestLoadCSV:
    def test_keeps_only_the_requested_columns(self, temp_guardian_file):
        frame = load_csv(temp_guardian_file, ["Headlines"])

        assert list(frame.columns) == ["Headlines"]

    def test_drops_duplicate_headlines(self, tmp_path):
        path = tmp_path / "dupes.csv"
        pd.DataFrame({"Headlines": ["same", "same", "other"]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Headlines"])) == 2

    def test_drops_rows_with_missing_text(self, tmp_path):
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"Headlines": ["kept", None]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Headlines"])) == 1

    def test_index_is_reset(self, temp_guardian_file):
        frame = load_csv(temp_guardian_file, ["Headlines"])

        assert list(frame.index) == list(range(len(frame)))

    def test_an_absent_column_is_rejected(self, temp_guardian_file):
        with pytest.raises(KeyError):
            load_csv(temp_guardian_file, ["Nonexistent"])

    def test_a_headerless_file_yields_no_rows(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame({"Headlines": []}).to_csv(path, index=False)

        assert len(load_csv(path, ["Headlines"])) == 0


class TestSplitFrame:
    def test_produces_two_named_splits(self, bert_data_args):
        assert set(split_frame(bert_data_args, seed=42)) == {"train", "validation"}

    def test_labels_before_tokenizing(self, bert_data_args):
        splits = split_frame(bert_data_args, seed=42)

        assert "labels" in splits["train"].column_names
        assert "input_ids" not in splits["train"].column_names

    def test_headlines_of_a_single_sentiment_cannot_be_stratified(self, tmp_path):
        path = tmp_path / "neutral.csv"
        neutral = ["the meeting is at noon", "a report was filed", "the door is closed", "data was recorded"]
        pd.DataFrame({"Headlines": neutral}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="least populated class"):
            split_frame(ClassificationDataArguments(data_path=str(path)), seed=42)


@pytest.fixture
def bert_splits(bert_data_args, bert_tokenizer):
    """The two splits built from the temporary Guardian CSV."""
    return build_datasets(bert_data_args, bert_tokenizer, seed=42)


class TestBuildDatasets:
    def test_produces_two_named_splits(self, bert_splits):
        assert set(bert_splits) == {"train", "validation"}

    def test_keeps_every_row(self, bert_data_args, bert_splits):
        expected = len(load_csv(bert_data_args.data_path, [bert_data_args.text_column]))

        assert sum(len(split) for split in bert_splits.values()) == expected

    def test_train_is_the_largest_split(self, bert_splits):
        assert len(bert_splits["train"]) > len(bert_splits["validation"])

    def test_drops_the_raw_text_column(self, bert_splits):
        assert "Headlines" not in bert_splits["train"].column_names

    def test_carries_tokenized_inputs_and_labels(self, bert_splits):
        columns = bert_splits["train"].column_names

        assert "input_ids" in columns
        assert "attention_mask" in columns
        assert "labels" in columns

    def test_labels_stay_within_the_known_ids(self, bert_splits):
        assert set(bert_splits["train"]["labels"]).issubset(ID2LABEL)

    def test_respects_the_truncation_length(self, bert_data_args, bert_splits):
        assert all(len(ids) <= bert_data_args.max_length for ids in bert_splits["train"]["input_ids"])

    def test_is_reproducible_for_a_seed(self, bert_data_args, bert_tokenizer):
        first = build_datasets(bert_data_args, bert_tokenizer, seed=42)
        second = build_datasets(bert_data_args, bert_tokenizer, seed=42)

        assert first["validation"]["labels"] == second["validation"]["labels"]

import pandas as pd
import pytest

from gpt2.config import ClmDataArguments
from gpt2.data import build_datasets, load_csv, split_frame


class TestLoadCSV:
    def test_keeps_only_the_requested_columns(self, temp_reuters_file):
        frame = load_csv(temp_reuters_file, ["Headlines"])

        assert list(frame.columns) == ["Headlines"]

    def test_drops_duplicate_headlines(self, tmp_path):
        path = tmp_path / "dupes.csv"
        pd.DataFrame({"Headlines": ["same", "same", "other"]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Headlines"])) == 2

    def test_drops_rows_with_missing_text(self, tmp_path):
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"Headlines": ["kept", None]}).to_csv(path, index=False)

        assert len(load_csv(path, ["Headlines"])) == 1

    def test_an_absent_column_is_rejected(self, temp_reuters_file):
        with pytest.raises(KeyError):
            load_csv(temp_reuters_file, ["Nonexistent"])


class TestSplitFrame:
    def test_produces_two_named_splits(self, gpt2_data_args):
        assert set(split_frame(gpt2_data_args, seed=42)) == {"train", "validation"}

    def test_leaves_the_text_untokenized(self, gpt2_data_args):
        splits = split_frame(gpt2_data_args, seed=42)

        assert gpt2_data_args.text_column in splits["train"].column_names
        assert "input_ids" not in splits["train"].column_names


@pytest.fixture
def gpt2_splits(gpt2_data_args, gpt2_tokenizer):
    """The two splits built from the temporary Reuters CSV."""
    return build_datasets(gpt2_data_args, gpt2_tokenizer, seed=42)


class TestBuildDatasets:
    def test_produces_two_named_splits(self, gpt2_splits):
        assert set(gpt2_splits) == {"train", "validation"}

    def test_keeps_every_row(self, gpt2_data_args, gpt2_splits):
        expected = len(load_csv(gpt2_data_args.data_path, [gpt2_data_args.text_column]))

        assert sum(len(split) for split in gpt2_splits.values()) == expected

    def test_drops_the_raw_text_column(self, gpt2_splits):
        assert "Headlines" not in gpt2_splits["train"].column_names

    def test_carries_tokenized_inputs_and_labels(self, gpt2_splits):
        columns = gpt2_splits["train"].column_names

        assert "input_ids" in columns
        assert "attention_mask" in columns
        assert "labels" in columns

    def test_pads_every_row_to_the_same_length(self, gpt2_data_args, gpt2_splits):
        assert all(len(ids) == gpt2_data_args.max_length for ids in gpt2_splits["train"]["input_ids"])

    def test_masks_the_loss_on_padding_only(self, gpt2_splits):
        for row in gpt2_splits["train"]:
            for token, attend, label in zip(row["input_ids"], row["attention_mask"], row["labels"], strict=True):
                assert label == (token if attend else -100)

    def test_every_headline_ends_with_eos(self, gpt2_splits, gpt2_tokenizer):
        for row in gpt2_splits["train"]:
            last = sum(row["attention_mask"]) - 1

            assert row["input_ids"][last] == gpt2_tokenizer.eos_token_id

    def test_headlines_too_long_to_fit_still_end_with_eos(self, gpt2_tokenizer, tmp_path):
        path = tmp_path / "long.csv"
        overlong = " ".join(["extremely"] * 40)
        pd.DataFrame({"Headlines": [overlong, "short one", "another short", "a fourth headline"]}).to_csv(
            path, index=False
        )
        splits = build_datasets(ClmDataArguments(data_path=str(path), max_length=16), gpt2_tokenizer, seed=42)

        for split in splits.values():
            for row in split:
                last = sum(row["attention_mask"]) - 1

                assert row["input_ids"][last] == gpt2_tokenizer.eos_token_id

    def test_is_reproducible_for_a_seed(self, gpt2_data_args, gpt2_tokenizer):
        first = build_datasets(gpt2_data_args, gpt2_tokenizer, seed=42)
        second = build_datasets(gpt2_data_args, gpt2_tokenizer, seed=42)

        assert first["validation"]["input_ids"] == second["validation"]["input_ids"]

import pytest

from headlines.t5.config import T5CFG
from headlines.t5.data import load_csv, split_csv, tokenize


@pytest.fixture
def reuters_dataset(temp_reuters_headlines):
    """Reuters sample CSV loaded as a dataset."""
    return load_csv(file_path=str(temp_reuters_headlines))


class TestLoadCSV:
    """test loads csv from file path"""

    def test_keeps_both_text_columns(self, reuters_dataset):
        """test the description and its headline both survive loading"""
        assert "Description" in reuters_dataset.column_names
        assert "Headlines" in reuters_dataset.column_names


class TestSplitCSV:
    """test splits a dataset into train and test"""

    def test_keeps_every_row(self, reuters_dataset, reuters_rows):
        """test no row is lost or duplicated across the splits"""
        train, test = split_csv(reuters_dataset)

        assert len(train) + len(test) == len(reuters_rows)

    def test_holds_out_a_test_split(self, reuters_dataset):
        """test both splits carry rows to work with"""
        train, test = split_csv(reuters_dataset)

        assert len(train) > len(test) > 0

    def test_splits_do_not_overlap(self, reuters_dataset):
        """test a headline lands in exactly one split"""
        train, test = split_csv(reuters_dataset)

        assert set(train["Headlines"]).isdisjoint(test["Headlines"])

    def test_is_reproducible(self, reuters_dataset):
        """test the same rows land in the same split on every call"""
        first = split_csv(reuters_dataset)
        second = split_csv(reuters_dataset)

        assert [s["Headlines"] for s in first] == [s["Headlines"] for s in second]


class TestTokenize:
    """test tokenizes descriptions as source and headlines as target"""

    @pytest.fixture
    def tokenized(self, reuters_dataset, t5_tokenizer):
        """Reuters sample dataset after tokenization."""
        return tokenize(reuters_dataset, t5_tokenizer)

    def test_replaces_text_columns(self, tokenized):
        """test raw text is dropped for model inputs"""
        assert "Description" not in tokenized.column_names
        assert "Headlines" not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "labels" in tokenized.column_names

    def test_pads_source_to_config_length(self, tokenized):
        """test every input matches the configured source length"""
        assert all(len(row) == T5CFG.max_source_length for row in tokenized["input_ids"])

    def test_pads_target_to_config_length(self, tokenized):
        """test every label row matches the configured target length"""
        assert all(len(row) == T5CFG.max_target_length for row in tokenized["labels"])

    def test_masks_label_padding(self, tokenized):
        """test padded label positions are ignored by the loss"""
        assert any(-100 in row for row in tokenized["labels"])

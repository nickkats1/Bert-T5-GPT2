import pytest

from headlines.gpt2.config import Gpt2CFG
from headlines.gpt2.data import load_csv, split_csv, tokenize


@pytest.fixture
def headline_dataset(temp_reuters_headlines):
    """Reuters sample CSV loaded as a headline-only dataset."""
    return load_csv(file_path=str(temp_reuters_headlines))


class TestLoadCSV:
    """test loads csv from file path and drops the unused columns"""

    def test_keeps_only_headlines(self, headline_dataset):
        """test time and description are dropped before training"""
        assert headline_dataset.column_names == ["Headlines"]

    def test_keeps_every_row(self, headline_dataset, reuters_rows):
        """test dropping columns does not drop rows"""
        assert len(headline_dataset) == len(reuters_rows)

    def test_drops_any_other_column(self, temp_reuters_with_labels):
        """test a labels column in the CSV never reaches training"""
        assert load_csv(file_path=str(temp_reuters_with_labels)).column_names == ["Headlines"]


class TestSplitCSV:
    """test splits a dataset into train and test"""

    def test_keeps_every_row(self, headline_dataset, reuters_rows):
        """test no row is lost or duplicated across the splits"""
        train, test = split_csv(headline_dataset)

        assert len(train) + len(test) == len(reuters_rows)

    def test_holds_out_a_test_split(self, headline_dataset):
        """test both splits carry rows to work with"""
        train, test = split_csv(headline_dataset)

        assert len(train) > len(test) > 0

    def test_splits_do_not_overlap(self, headline_dataset):
        """test a headline lands in exactly one split"""
        train, test = split_csv(headline_dataset)

        assert set(train["Headlines"]).isdisjoint(test["Headlines"])

    def test_is_reproducible(self, headline_dataset):
        """test the same rows land in the same split on every call"""
        first = split_csv(headline_dataset)
        second = split_csv(headline_dataset)

        assert [s["Headlines"] for s in first] == [s["Headlines"] for s in second]


class TestTokenize:
    """test tokenizes headlines for causal language modelling"""

    @pytest.fixture
    def tokenized(self, headline_dataset, gpt2_tokenizer):
        """Reuters sample headlines after tokenization."""
        return tokenize(headline_dataset, gpt2_tokenizer)

    def test_replaces_headlines_column(self, tokenized):
        """test raw text is dropped for model inputs"""
        assert "Headlines" not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names

    def test_pads_to_config_length(self, tokenized):
        """test every row matches the configured max length"""
        assert all(len(row) == Gpt2CFG.max_length for row in tokenized["input_ids"])

    def test_leaves_labels_to_the_collator(self, tokenized):
        """test nothing here presets labels the collator would overwrite"""
        assert "labels" not in tokenized.column_names

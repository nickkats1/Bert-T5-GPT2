import pytest

from headlines.bert.config import BertCFG
from headlines.bert.data import build_label_maps, load_csv, split_csv, tokenize


class TestLoadCSV:
    """test loads csv from file path"""

    def test_is_instance_csv(self, temp_guardian_file):
        """test if temp file is instance of dataset"""
        ds = load_csv(file_path=temp_guardian_file)

        assert "Headlines" in ds.column_names


@pytest.fixture
def guardian_dataset(temp_guardian_file):
    """Guardian sample CSV loaded as a dataset."""
    return load_csv(file_path=str(temp_guardian_file))


class TestSplitCSV:
    """test splits a dataset into train, test and val"""

    def test_keeps_every_row(self, guardian_dataset, guardian_rows):
        """test no row is lost or duplicated across the splits"""
        train, test, val = split_csv(guardian_dataset)

        assert len(train) + len(test) + len(val) == len(guardian_rows)

    def test_splits_do_not_overlap(self, guardian_dataset):
        """test a headline lands in exactly one split"""
        train, test, val = split_csv(guardian_dataset)

        assert set(train["Headlines"]).isdisjoint(test["Headlines"])
        assert set(train["Headlines"]).isdisjoint(val["Headlines"])
        assert set(test["Headlines"]).isdisjoint(val["Headlines"])

    def test_is_reproducible(self, guardian_dataset):
        """test the same rows land in the same split on every call"""
        first = split_csv(guardian_dataset)
        second = split_csv(guardian_dataset)

        assert [s["Headlines"] for s in first] == [s["Headlines"] for s in second]


class TestBuildLabelMaps:
    """test builds label and id lookups from a dataset"""

    def test_maps_every_label(self, guardian_dataset):
        """test one entry per distinct label"""
        label_to_id, id_to_label = build_label_maps(guardian_dataset)

        assert len(label_to_id) == 3
        assert len(id_to_label) == 3

    def test_maps_are_inverses(self, guardian_dataset):
        """test id_to_label reverses label_to_id"""
        label_to_id, id_to_label = build_label_maps(guardian_dataset)

        assert {i: label for label, i in label_to_id.items()} == id_to_label


class TestTokenize:
    """test tokenizes headlines and encodes labels"""

    @pytest.fixture
    def tokenized(self, guardian_dataset, bert_tokenizer):
        """Guardian sample dataset after tokenization."""
        label_to_id, _ = build_label_maps(guardian_dataset)
        return tokenize(guardian_dataset, bert_tokenizer, label_to_id)

    def test_replaces_headlines_column(self, tokenized):
        """test raw text is dropped for model inputs"""
        assert "Headlines" not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names

    def test_pads_to_config_length(self, tokenized):
        """test every row matches the configured max length"""
        assert all(len(row) == BertCFG.max_length for row in tokenized["input_ids"])

    def test_labels_are_ids(self, tokenized):
        """test labels land inside the range the model expects"""
        assert all(label in range(3) for label in tokenized["labels"])

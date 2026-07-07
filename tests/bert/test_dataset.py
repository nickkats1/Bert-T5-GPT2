import pytest
import torch
from transformers import BertTokenizer

from headlines.bert.config import CONFIG
from headlines.bert.dataset import HeadlineDataset


@pytest.fixture
def tokenizer():
    """instance of bert tokenizer"""
    return BertTokenizer.from_pretrained(CONFIG.model_name)


@pytest.fixture
def sample_dataset(tokenizer):
    """small HeadlineDataset for testing."""
    headlines = [
        "Markets rally on strong earnings",
        "Tech stocks tumble amid fears",
        "Housing market shows strength",
    ]
    targets = [2, 0, 1]
    return HeadlineDataset(headlines, targets, tokenizer, max_length=32)


class TestHeadlineDataset:
    """Test Custom Dataset"""

    def test_stores_headlines(self, sample_dataset):
        """Test if headlines is stored in constructor as attribute"""

        assert len(sample_dataset.headlines) == 3

    def test_stores_targets(self, sample_dataset):
        """test if targets are store in constructor as attribute"""
        assert (len(sample_dataset.targets)) == 3

    def test_stores_tokenizer(self, sample_dataset, tokenizer):
        """test tokenizer is attribute"""

        assert (sample_dataset.tokenizer) == tokenizer

        assert isinstance(tokenizer, BertTokenizer)

    def test_stores_max_length(self, sample_dataset):
        """test max length is attribute"""
        assert sample_dataset.max_length == 32

    def test_len(self, sample_dataset, tokenizer):
        """Test to see if length of 0 is returned when no dataset is found"""
        assert len(sample_dataset.headlines) == 3

        dataset = HeadlineDataset([], [], tokenizer=tokenizer, max_length=32)

        assert len(dataset.headlines) == 0

    def test_getitem(self, sample_dataset):
        """tests for __getitem__ method"""

        item = sample_dataset[0]

        # test if required keys are returned

        assert "text" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "targets" in item

        # test if item is returned as dict
        assert isinstance(item, dict)

        # test if correct instance are returned from dict
        assert isinstance(item["text"], str)
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["targets"], torch.Tensor)
        assert item["targets"].dtype == torch.long

    def test_tokenized_shape6(self, sample_dataset, tokenizer):
        """test tokenizer output."""
        item = sample_dataset[0]

        # assert that input ids and attention mask has correct length
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)

        # assert that tokenized output starts with [CLS]

        assert item["input_ids"][0].item() == tokenizer.cls_token_id

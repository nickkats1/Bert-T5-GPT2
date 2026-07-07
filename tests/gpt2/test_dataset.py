import pytest
from transformers import GPT2Tokenizer

from headlines.gpt2.config import CONFIG
from headlines.gpt2.dataset import DescriptionDataset


@pytest.fixture(scope="module")
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens(
        {
            "pad_token": CONFIG.pad_token,
            "bos_token": CONFIG.bos_token,
            "eos_token": CONFIG.eos_token,
        }
    )
    return tok


@pytest.fixture
def dataset():
    return DescriptionDataset(
        ["markets rally", "tech stocks tumble", "housing gains"], tokenizer=None, max_length=32
    )


class TestDescriptionDataset:
    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem_wraps_bos_eos(self, tokenizer):
        ds = DescriptionDataset(["hello world"], tokenizer=tokenizer, max_length=16)
        item = ds[0]
        assert item.startswith(tokenizer.bos_token)
        assert item.endswith(tokenizer.eos_token)

    def test_collate_fn_returns_tensors(self, tokenizer):
        ds = DescriptionDataset(["a b c", "d e"], tokenizer=tokenizer, max_length=16)
        batch = ds.collate_fn([ds[0], ds[1]])
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2

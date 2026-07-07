import pandas as pd
import pytest
from transformers import GPT2Tokenizer

from headlines.gpt2.config import CONFIG
from headlines.gpt2.utils import build_dataloaders, load_data, split_data


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


class TestLoadData:
    def test_none_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data(None)

    def test_keeps_only_description(self, temp_reuters_headlines):
        df = load_data(temp_reuters_headlines)
        assert list(df.columns) == ["Description"]
        assert len(df) == 3


class TestSplitData:
    def test_split_lengths(self):
        df = pd.DataFrame({"Description": [f"row {i}" for i in range(10)]})
        train, val = split_data(df, test_size=0.2, random_state=0)
        assert len(train) == 8
        assert len(val) == 2
        assert isinstance(train, list)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            split_data(pd.DataFrame({"x": [1]}), test_size=0.2, random_state=0)


class TestBuildDataloaders:
    def test_yields_tokenized_batches(self, tokenizer):
        train = [f"train description {i}" for i in range(6)]
        val = [f"val description {i}" for i in range(2)]
        train_loader, val_loader = build_dataloaders(
            train, val, tokenizer, batch_size=3, max_length=16
        )
        assert len(train_loader.dataset) == 6
        assert len(val_loader.dataset) == 2

        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 3

"""Loading, splitting and tokenizing the Reuters headlines for causal language modelling."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from gpt2.config import ClmDataArguments


def load_csv(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read the requested columns, dropping missing and duplicate rows."""
    return pd.read_csv(path)[columns].dropna().drop_duplicates().reset_index(drop=True)


def split_frame(data_args: ClmDataArguments, seed: int) -> DatasetDict:
    """Split the raw headlines into train and validation."""
    frame = load_csv(data_args.data_path, [data_args.text_column])
    train, validation = train_test_split(frame, test_size=data_args.test_size, random_state=seed)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(validation, preserve_index=False),
        }
    )


def build_datasets(data_args: ClmDataArguments, tokenizer: PreTrainedTokenizerBase, seed: int) -> DatasetDict:
    """Tokenize every headline to a fixed length ending in EOS, with the loss masked on padding."""

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch[data_args.text_column], truncation=True, max_length=data_args.max_length - 1)
        encoded["input_ids"] = [[*ids, tokenizer.eos_token_id] for ids in encoded["input_ids"]]
        encoded["attention_mask"] = [[*mask, 1] for mask in encoded["attention_mask"]]
        padded = tokenizer.pad(encoded, padding="max_length", max_length=data_args.max_length)
        padded["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids] for ids in padded["input_ids"]
        ]
        return padded

    return split_frame(data_args, seed).map(tokenize, batched=True, remove_columns=[data_args.text_column])

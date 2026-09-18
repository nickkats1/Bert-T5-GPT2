from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from t5.config import SummarizationDataArguments


def load_csv(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read the requested columns, dropping missing and duplicate rows."""
    return pd.read_csv(path)[columns].dropna().drop_duplicates().reset_index(drop=True)


def split_frame(data_args: SummarizationDataArguments, seed: int) -> DatasetDict:
    """Split the raw pairs into train and validation."""
    frame = load_csv(data_args.data_path, [data_args.source_column, data_args.target_column])
    train, validation = train_test_split(frame, test_size=data_args.test_size, random_state=seed)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(validation, preserve_index=False),
        }
    )


def build_datasets(
    data_args: SummarizationDataArguments, tokenizer: PreTrainedTokenizerBase, seed: int
) -> DatasetDict:
    """Prefix and tokenize the sources, tokenize the targets as labels, and cap validation if asked."""

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(
            [data_args.source_prefix + text for text in batch[data_args.source_column]],
            truncation=True,
            max_length=data_args.max_source_length,
        )
        encoded["labels"] = tokenizer(
            text_target=batch[data_args.target_column], truncation=True, max_length=data_args.max_target_length
        )["input_ids"]
        return encoded

    columns = [data_args.source_column, data_args.target_column]
    datasets = split_frame(data_args, seed).map(tokenize, batched=True, remove_columns=columns)

    if data_args.max_eval_samples is not None:
        datasets["validation"] = datasets["validation"].select(range(data_args.max_eval_samples))

    return datasets

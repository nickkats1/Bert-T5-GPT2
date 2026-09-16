"""Loading, labelling, splitting and tokenizing the Guardian headlines."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from bert.config import ClassificationDataArguments
from bert.utils_bert import label_headlines


def load_csv(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read the requested columns, dropping missing and duplicate rows."""
    return pd.read_csv(path)[columns].dropna().drop_duplicates().reset_index(drop=True)


def split_frame(data_args: ClassificationDataArguments, seed: int) -> DatasetDict:
    """Label the headlines and split them, stratified so every split sees every sentiment."""
    frame = label_headlines(load_csv(data_args.data_path, [data_args.text_column]), data_args.text_column)
    train, validation = train_test_split(
        frame, test_size=data_args.test_size, random_state=seed, stratify=frame["labels"]
    )

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(validation, preserve_index=False),
        }
    )


def build_datasets(
    data_args: ClassificationDataArguments, tokenizer: PreTrainedTokenizerBase, seed: int
) -> DatasetDict:
    """Tokenize both splits, keeping only model inputs and labels."""

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch[data_args.text_column], truncation=True, max_length=data_args.max_length)

    return split_frame(data_args, seed).map(tokenize, batched=True, remove_columns=[data_args.text_column])

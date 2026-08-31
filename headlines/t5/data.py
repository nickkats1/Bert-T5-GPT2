"""Loading and splitting of the Reuters article CSV for T5 fine-tuning."""

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from headlines.t5.config import T5CFG


def load_csv(file_path: str) -> Dataset:
    """Load CSV file from file path and returns dataset.

    Args:
        file_path: where the file is located.

    Returns:
        dataset: CSV file converted as dataset instance.
    """
    return Dataset.from_csv(str(file_path))


def split_csv(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Split loaded dataset into train and test splits.

    Args:
        dataset: Dataset wrapper for CSV data to be trained.

    Returns:
        tuple of the training split and the held-out test split.
    """
    splits = dataset.train_test_split(test_size=0.2, seed=42)

    return splits["train"], splits["test"]


def drop_empty_rows(dataset: Dataset) -> Dataset:
    """Drop rows missing either side of the description/headline pair.

    Args:
        dataset: split carrying "Description" and "Headlines" columns.

    Returns:
        dataset with only rows the tokenizer can encode.
    """
    return dataset.filter(lambda row: bool(row["Description"]) and bool(row["Headlines"]))


def tokenize(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Tokenize descriptions as source text and headlines as target text.

    Args:
        dataset: split to tokenize.
        tokenizer: tokenizer matching the model being fine-tuned.

    Returns:
        dataset with input_ids, attention_mask and padded labels masked to -100.
    """

    def preprocess(batch):
        encoded = tokenizer(
            [T5CFG.source_prefix + text for text in batch["Description"]],
            truncation=True,
            max_length=T5CFG.max_source_length,
            padding="max_length",
        )
        targets = tokenizer(
            text_target=batch["Headlines"],
            truncation=True,
            max_length=T5CFG.max_target_length,
            padding="max_length",
        )
        encoded["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in ids] for ids in targets["input_ids"]
        ]
        return encoded

    return drop_empty_rows(dataset).map(preprocess, batched=True, remove_columns=["Headlines", "Time", "Description"])

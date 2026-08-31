"""Loading and splitting of the Reuters headline CSV for GPT-2 fine-tuning."""

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from headlines.gpt2.config import Gpt2CFG


def load_csv(file_path: str) -> Dataset:
    """Load CSV file from file path and returns dataset of headlines only.

    Args:
        file_path: where the file is located.

    Returns:
        dataset: CSV file converted as dataset instance, keeping "Headlines".
    """
    return Dataset.from_csv(str(file_path)).select_columns(["Headlines"])


def split_csv(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Split loaded dataset into train and test splits.

    Args:
        dataset: Dataset wrapper for CSV data to be trained.

    Returns:
        tuple of the training split and the held-out test split.
    """
    splits = dataset.train_test_split(test_size=0.2, seed=42)

    return splits["train"], splits["test"]


def tokenize(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Tokenize headlines for causal language modelling.

    Args:
        dataset: split to tokenize.
        tokenizer: tokenizer matching the model being fine-tuned.

    Returns:
        dataset with input_ids and attention_mask; the collator builds the labels.
    """

    def preprocess(batch):
        return tokenizer(
            batch["Headlines"],
            truncation=True,
            max_length=Gpt2CFG.max_length,
            padding="max_length",
        )

    return dataset.filter(lambda row: bool(row["Headlines"])).map(
        preprocess, batched=True, remove_columns=["Headlines"]
    )

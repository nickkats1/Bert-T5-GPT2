"""PyTorch ``Dataset`` and ``DataLoader`` construction for the BERT pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from headlines.bert.config import CONFIG


class HeadlineDataset(Dataset):
    """Tokenizes headlines and pairs them with integer sentiment targets.

    Attributes:
        headlines: Sequence of raw headline strings.
        targets: Integer sentiment labels aligned with ``headlines``.
        tokenizer: HuggingFace BERT tokenizer.
        max_length: Maximum token length used for padding/truncation.
    """

    def __init__(
        self,
        headlines: Sequence[str],
        targets: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            headlines: Sequence (list, ndarray, ...) of headline strings.
            targets: Integer sentiment labels of the same length as headlines.
            tokenizer: HuggingFace BERT tokenizer.
            max_length: Maximum token length used by the tokenizer.

        Raises:
            ValueError: If ``headlines`` and ``targets`` have different lengths.
        """
        if len(headlines) != len(targets):
            raise ValueError(
                f"headlines ({len(headlines)}) and targets ({len(targets)}) "
                f"must have the same length."
            )
        self.headlines = headlines
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.headlines)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Tokenize and return the example at ``index``.

        Args:
            index: Integer index of the example.

        Returns:
            A dictionary with keys ``text`` (str), ``input_ids``,
            ``attention_mask`` (both 1-D tensors of length ``max_length``),
            and ``targets`` (scalar long tensor).
        """
        headline = str(self.headlines[index])
        target = self.targets[index]

        encoded = self.tokenizer(
            headline,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": headline,
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


def create_data_loader(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = CONFIG.max_length,
    batch_size: int = CONFIG.batch_size,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` over the headlines in ``df``.

    Args:
        df: DataFrame with a ``Headlines`` column and an integer-encoded
            ``sentiment`` label column (see :mod:`headlines.bert.utils`).
        tokenizer: HuggingFace BERT tokenizer.
        max_length: Max token length for padding / truncation.
        batch_size: Number of examples per batch.
        shuffle: Whether to reshuffle the data every epoch.
        generator: Optional seeded ``torch.Generator`` for reproducible
            shuffling (see :func:`headlines.common.make_generator`).

    Returns:
        A PyTorch ``DataLoader`` yielding tokenized batches.
    """
    dataset = HeadlineDataset(
        headlines=df["Headlines"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )

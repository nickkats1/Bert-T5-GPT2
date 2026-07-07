"""PyTorch ``Dataset`` for T5 sequence-to-sequence summarization."""

from __future__ import annotations

from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    """Tokenizes source-target text pairs for T5 fine-tuning.

    Attributes:
        tokenizer: HuggingFace T5 tokenizer.
        data: DataFrame containing source and target text columns.
        source_len: Maximum token length for source sequences.
        target_len: Maximum token length for target sequences.
        source_text: Series of source text strings.
        target_text: Series of target text strings.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        source_len: int,
        target_len: int,
        source_col: str,
        target_col: str,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataframe: DataFrame containing both source and target columns.
            tokenizer: HuggingFace T5 tokenizer.
            source_len: Maximum token length for source sequences.
            target_len: Maximum token length for target sequences.
            source_col: Column name for the source (input) text.
            target_col: Column name for the target (output) text.

        Raises:
            KeyError: If ``source_col`` or ``target_col`` is missing.
        """
        for col in (source_col, target_col):
            if col not in dataframe.columns:
                raise KeyError(f"Column {col!r} missing from DataFrame.")

        self.tokenizer = tokenizer
        self.data = dataframe.reset_index(drop=True)
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_col]
        self.target_text = self.data[target_col]

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.target_text)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Tokenize and return the source-target pair at ``index``.

        Args:
            index: Integer index of the example.

        Returns:
            A dict with:

            - ``source_ids``: 1-D long tensor of length ``source_len``.
            - ``source_mask``: 1-D attention mask of length ``source_len``.
            - ``target_ids``: 1-D long tensor of length ``target_len``.
        """
        source_text = str(self.source_text.iloc[index])
        target_text = str(self.target_text.iloc[index])

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "source_ids": source["input_ids"].squeeze(0),
            "source_mask": source["attention_mask"].squeeze(0),
            "target_ids": target["input_ids"].squeeze(0),
        }

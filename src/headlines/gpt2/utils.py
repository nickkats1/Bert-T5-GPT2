"""Data loading and DataLoader helpers for the GPT-2 pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from headlines.gpt2.dataset import DescriptionDataset


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load the Reuters CSV and return only the ``Description`` column.

    Drops the ``Time`` and ``Headlines`` columns (if present), removes rows
    with missing descriptions, and de-duplicates.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with a single ``Description`` column.

    Raises:
        FileNotFoundError: If ``file_path`` is None or does not exist.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(path, delimiter=",")

    columns_to_drop = [c for c in ("Time", "Headlines") if c in dataframe.columns]
    if columns_to_drop:
        dataframe = dataframe.drop(columns=columns_to_drop)

    if "Description" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Description"])

    return dataframe.drop_duplicates().reset_index(drop=True)


def split_data(
    text: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """Split a description DataFrame into train/validation string lists.

    Args:
        text: DataFrame containing a ``Description`` column.
        test_size: Fraction of rows reserved for validation.
        random_state: RNG seed for the split.

    Returns:
        Tuple ``(train_descriptions, val_descriptions)`` of Python lists.

    Raises:
        KeyError: If the ``Description`` column is missing.
    """
    if "Description" not in text.columns:
        raise KeyError("Input DataFrame must contain a 'Description' column.")

    # Imported here to keep sklearn out of the import path of pure-data callers.
    from sklearn.model_selection import train_test_split

    df_train, df_val = train_test_split(text, test_size=test_size, random_state=random_state)
    train_description = df_train["Description"].reset_index(drop=True).tolist()
    val_description = df_val["Description"].reset_index(drop=True).tolist()
    return train_description, val_description


def build_dataloaders(
    train_description: list[str],
    val_description: list[str],
    tokenizer,
    batch_size: int,
    max_length: int = 128,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation ``DataLoader``s from description string lists.

    Args:
        train_description: Training description strings.
        val_description: Validation description strings.
        tokenizer: HuggingFace GPT-2 tokenizer with BOS/EOS/pad tokens set.
        batch_size: Number of examples per batch.
        max_length: Maximum token length used during collation.
        generator: Optional seeded ``torch.Generator`` for reproducible
            shuffling of the training loader.

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    train_set = DescriptionDataset(train_description, tokenizer, max_length=max_length)
    val_set = DescriptionDataset(val_description, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=val_set.collate_fn
    )
    return train_loader, val_loader

"""Data loading for the T5 Description → Headline summarization pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load the Reuters CSV with both ``Headlines`` and ``Description`` columns.

    Drops the ``Time`` column, fills NaNs with empty strings, de-duplicates,
    and resets the index so callers can use positional indexing.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with at least ``Headlines`` and ``Description`` columns.

    Raises:
        FileNotFoundError: If ``file_path`` is None or does not exist.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, delimiter=",")

    missing = {"Headlines", "Description"} - set(df.columns)
    if missing:
        raise KeyError(f"CSV is missing required column(s): {sorted(missing)}")

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    return df.fillna("").drop_duplicates().reset_index(drop=True)

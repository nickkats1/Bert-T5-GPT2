"""Data loading, cleaning, and weak-supervision labeling for the BERT pipeline.

The Guardian headlines dataset ships without sentiment labels, so we derive
them with TextBlob polarity (a lightweight, deterministic heuristic) and then
integer-encode the result for training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob

LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
REVERSE_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
#: Ordered class names, indexed by integer label — used by metric reports.
CLASS_NAMES = [REVERSE_LABEL_MAP[i] for i in range(len(REVERSE_LABEL_MAP))]


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A DataFrame containing the CSV contents.

    Raises:
        FileNotFoundError: If ``file_path`` is None or does not exist.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, delimiter=",")


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Guardian headlines DataFrame.

    Steps:
        1. Remove the optional ``Time`` column if present.
        2. Drop rows missing the ``Headlines`` value.
        3. Drop duplicate rows and reset the index.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A cleaned copy of the DataFrame with a fresh ``0..N-1`` index.
    """
    cleaned = dataframe.copy()

    if "Time" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Time"])

    if "Headlines" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Headlines"])

    return cleaned.drop_duplicates().reset_index(drop=True)


def polarity(text: str) -> float:
    """Compute the TextBlob polarity score for a string.

    Args:
        text: Free-form text. Falsy or non-string input returns ``0.0``.

    Returns:
        Polarity score in ``[-1.0, 1.0]``.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    return float(TextBlob(text).polarity)


def sentiment(score: float) -> str:
    """Map a polarity score to a discrete sentiment label.

    Args:
        score: Polarity score in ``[-1.0, 1.0]``.

    Returns:
        One of ``"Negative"``, ``"Neutral"``, ``"Positive"``.
    """
    if score == 0:
        return "Neutral"
    if score < 0:
        return "Negative"
    return "Positive"


def label_encode_sentiments(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Encode the ``sentiment`` column as integers and drop helper columns.

    Maps ``Negative`` → 0, ``Neutral`` → 1, ``Positive`` → 2 and drops the
    ``polarity`` helper column if present. Removes duplicate rows.

    Args:
        dataframe: DataFrame containing a ``sentiment`` column with string
            labels.

    Returns:
        A copy of ``dataframe`` with integer-encoded sentiment, the
        ``polarity`` column removed, duplicates dropped, and the index reset.

    Raises:
        KeyError: If the ``sentiment`` column is missing.
    """
    if "sentiment" not in dataframe.columns:
        raise KeyError("DataFrame must contain a 'sentiment' column.")

    df = dataframe.copy()
    df["sentiment"] = df["sentiment"].map(LABEL_MAP).astype("int64")

    if "polarity" in df.columns:
        df = df.drop(columns=["polarity"])

    return df.drop_duplicates().reset_index(drop=True)


def _split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that degrades gracefully.

    Falls back to a plain (non-stratified) split when the requested test slice
    is smaller than the number of classes — sklearn cannot stratify in that
    case, which happens on very small subsets.
    """
    try:
        return train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["sentiment"]
        )
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=seed)


def split_train_val_test(
    df: pd.DataFrame,
    holdout_size: float,
    test_size_from_holdout: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labeled DataFrame into stratified train / val / test partitions.

    Args:
        df: Labeled DataFrame with an integer ``sentiment`` column.
        holdout_size: Fraction reserved for validation + test.
        test_size_from_holdout: Fraction of the holdout used for the test set.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple ``(df_train, df_val, df_test)``.
    """
    df_train, df_holdout = _split(df, holdout_size, seed)
    df_val, df_test = _split(df_holdout, test_size_from_holdout, seed)
    return df_train, df_val, df_test

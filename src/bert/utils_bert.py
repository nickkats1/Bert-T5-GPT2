"""TextBlob-based sentiment labels for headlines that ship without a label column."""

import pandas as pd
from textblob import TextBlob


ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}


def polarity(text: str) -> float:
    """Score a headline between -1 (negative) and 1 (positive)."""
    return TextBlob(text).polarity


def sentiment(score: float) -> int:
    """Map a polarity score to a label id."""
    if score == 0:
        return LABEL2ID["Neutral"]
    if score < 0:
        return LABEL2ID["Negative"]
    return LABEL2ID["Positive"]


def label_headlines(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Return a copy of the frame with a labels column derived from the headline text."""
    return frame.assign(labels=frame[text_column].map(lambda text: sentiment(polarity(text))))

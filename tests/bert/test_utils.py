import pandas as pd
import pytest

from headlines.bert.utils import (
    LABEL_MAP,
    clean_data,
    label_encode_sentiments,
    load_data,
    polarity,
    sentiment,
    split_train_val_test,
)


class TestLoadAndClean:
    """Tests for load_data / clean_data."""

    def test_load_data_none_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data(None)

    def test_load_data_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("does/not/exist.csv")

    def test_load_data_reads_file(self, temp_guardian_file):
        df = load_data(temp_guardian_file)
        assert "Headlines" in df.columns
        assert len(df) == 4

    def test_clean_data_drops_time_and_dupes(self):
        raw = pd.DataFrame(
            {
                "Time": ["a", "b", "b"],
                "Headlines": ["x", "y", "y"],
            }
        )
        cleaned = clean_data(raw)
        assert "Time" not in cleaned.columns
        assert len(cleaned) == 2
        assert list(cleaned.index) == [0, 1]


class TestPolarity:
    def test_positive(self):
        assert polarity("I love this movie") > 0.0

    def test_negative(self):
        assert polarity("I hate this movie") < 0.0

    def test_neutral(self):
        assert polarity("Do you want to see a movie?") == 0.0

    def test_non_string_returns_zero(self):
        assert polarity(None) == 0.0  # type: ignore[arg-type]


class TestSentiment:
    def test_positive(self):
        assert sentiment(0.5) == "Positive"

    def test_negative(self):
        assert sentiment(-0.5) == "Negative"

    def test_neutral(self):
        assert sentiment(0.0) == "Neutral"


class TestLabelEncode:
    def test_encodes_and_resets(self):
        df = pd.DataFrame(
            {
                "Headlines": ["a", "b", "c"],
                "polarity": [0.5, -0.5, 0.0],
                "sentiment": ["Positive", "Negative", "Neutral"],
            }
        )
        out = label_encode_sentiments(df)
        assert "polarity" not in out.columns
        assert out["sentiment"].tolist() == [
            LABEL_MAP["Positive"],
            LABEL_MAP["Negative"],
            LABEL_MAP["Neutral"],
        ]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            label_encode_sentiments(pd.DataFrame({"Headlines": ["a"]}))


class TestSplitTrainValTest:
    def _labeled(self, n_per_class: int) -> pd.DataFrame:
        rows = []
        for label in (0, 1, 2):
            rows += [{"Headlines": f"h{label}_{i}", "sentiment": label} for i in range(n_per_class)]
        return pd.DataFrame(rows)

    def test_partitions_are_disjoint_and_complete(self):
        df = self._labeled(20)
        train, val, test = split_train_val_test(df, 0.5, 0.2, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        idx = set(train["Headlines"]) | set(val["Headlines"]) | set(test["Headlines"])
        assert len(idx) == len(df)  # no overlap

    def test_falls_back_on_tiny_data(self):
        # 6 rows, 3 classes: the nested test slice is smaller than the class
        # count, so stratification must gracefully fall back instead of raising.
        df = self._labeled(2)
        train, val, test = split_train_val_test(df, 0.5, 0.2, seed=0)
        assert len(train) + len(val) + len(test) == len(df)

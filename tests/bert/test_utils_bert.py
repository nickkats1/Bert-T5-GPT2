import pandas as pd

from bert.utils_bert import ID2LABEL, LABEL2ID, label_headlines, polarity, sentiment


class TestIdLabelMaps:
    def test_covers_three_sentiments(self):
        assert ID2LABEL == {0: "Negative", 1: "Neutral", 2: "Positive"}

    def test_label_to_id_is_the_inverse(self):
        assert {label: index for index, label in ID2LABEL.items()} == LABEL2ID


class TestPolarity:
    def test_returns_a_score_in_range(self):
        assert -1.0 <= polarity("Record profits lift the index to a new high") <= 1.0

    def test_is_deterministic(self):
        headline = "Ministers face fresh criticism over a delayed inquiry"

        assert polarity(headline) == polarity(headline)


class TestSentiment:
    def test_positive_polarity_maps_to_positive(self):
        assert sentiment(0.5) == LABEL2ID["Positive"]

    def test_negative_polarity_maps_to_negative(self):
        assert sentiment(-0.5) == LABEL2ID["Negative"]

    def test_exactly_zero_maps_to_neutral(self):
        assert sentiment(0.0) == LABEL2ID["Neutral"]


class TestLabelHeadlines:
    def test_adds_a_labels_column(self, temp_guardian_file):
        frame = label_headlines(pd.read_csv(temp_guardian_file), "Headlines")

        assert "labels" in frame.columns

    def test_labels_are_known_ids(self, temp_guardian_file):
        frame = label_headlines(pd.read_csv(temp_guardian_file), "Headlines")

        assert set(frame["labels"]).issubset(ID2LABEL)

    def test_does_not_mutate_the_input(self, temp_guardian_file):
        original = pd.read_csv(temp_guardian_file)
        label_headlines(original, "Headlines")

        assert "labels" not in original.columns

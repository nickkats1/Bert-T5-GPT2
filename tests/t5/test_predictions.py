import csv

import pytest

from headlines.t5.predictions import generate_headlines, load_trained, score_rows, write_predictions


DESCRIPTIONS = [
    "The Bank of England said the economy would take longer to recover than forecast.",
    "Oil prices held steady as demand worries offset the production cuts agreed by OPEC.",
]

HEADLINES = [
    "Bank of England warns of slower recovery",
    "Oil prices steady as demand concerns offset supply cuts",
]


class TestLoadTrained:
    """test loads the fine-tuned checkpoint from disk"""

    def test_missing_checkpoint_names_the_directory(self, tmp_path):
        """test the error tells you training has not been run"""
        missing = tmp_path / "never-trained"

        with pytest.raises(FileNotFoundError, match=str(missing)):
            load_trained(output_dir=str(missing))


class TestScoreRows:
    """test pairs each description with its generated headline"""

    @pytest.fixture
    def rows(self):
        """Rows built from handmade generations."""
        return score_rows(DESCRIPTIONS, HEADLINES)

    def test_one_row_per_description(self, rows):
        """test nothing is dropped or duplicated"""
        assert len(rows) == len(DESCRIPTIONS)

    def test_keeps_the_description(self, rows):
        """test each row can be traced back to its source text"""
        assert [row["Description"] for row in rows] == DESCRIPTIONS

    def test_carries_the_generated_headline(self, rows):
        """test the model output lands in its own column"""
        assert [row["pred_headline"] for row in rows] == HEADLINES


@pytest.mark.integration
class TestGenerateHeadlines:
    """test summarizes descriptions into headlines with a model"""

    @pytest.fixture
    def generated(self, t5_model, t5_tokenizer):
        """Headlines written by the untrained model for each description."""
        return generate_headlines(DESCRIPTIONS, t5_model, t5_tokenizer)

    def test_one_headline_per_description(self, generated):
        """test nothing is dropped or duplicated"""
        assert len(generated) == len(DESCRIPTIONS)

    def test_returns_decoded_text(self, generated):
        """test the caller gets strings rather than token ids"""
        assert all(isinstance(headline, str) for headline in generated)

    def test_batching_does_not_lose_rows(self, t5_model, t5_tokenizer):
        """test a batch smaller than the input still covers every description"""
        assert len(generate_headlines(DESCRIPTIONS, t5_model, t5_tokenizer, batch_size=1)) == len(DESCRIPTIONS)


class TestWritePredictions:
    """test writes generated headlines to csv"""

    def test_writes_expected_columns(self, tmp_path):
        """test the csv carries true and predicted headlines side by side"""
        path = tmp_path / "t5_predictions.csv"
        rows = [
            {
                "Description": DESCRIPTIONS[0],
                "true_headline": HEADLINES[0],
                "pred_headline": HEADLINES[1],
            }
        ]

        write_predictions(rows, path=str(path))

        with open(path, newline="", encoding="utf-8") as f:
            written = list(csv.DictReader(f))

        assert written[0]["Description"] == DESCRIPTIONS[0]
        assert written[0]["true_headline"] == HEADLINES[0]
        assert written[0]["pred_headline"] == HEADLINES[1]

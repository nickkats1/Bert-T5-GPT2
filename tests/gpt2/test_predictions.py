import csv

import pytest

from headlines.gpt2.predictions import generate_headlines, load_trained, score_rows, write_predictions


PROMPTS = [
    "Bank of England",
    "Oil prices steady",
]

GENERATED = [
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


@pytest.mark.integration
class TestGenerateHeadlines:
    """test continues prompts into full headlines with a model"""

    @pytest.fixture
    def generated(self, gpt2_model, gpt2_tokenizer):
        """Headlines continued by the untrained model from each prompt."""
        return generate_headlines(PROMPTS, gpt2_model, gpt2_tokenizer)

    def test_one_headline_per_prompt(self, generated):
        """test nothing is dropped or duplicated"""
        assert len(generated) == len(PROMPTS)

    def test_returns_decoded_text(self, generated):
        """test the caller gets strings rather than token ids"""
        assert all(isinstance(headline, str) for headline in generated)

    def test_batching_does_not_lose_rows(self, gpt2_model, gpt2_tokenizer):
        """test a batch smaller than the input still covers every prompt"""
        assert len(generate_headlines(PROMPTS, gpt2_model, gpt2_tokenizer, batch_size=1)) == len(PROMPTS)


class TestScoreRows:
    """test pairs each prompt with its generated headline"""

    @pytest.fixture
    def rows(self):
        """Rows built from handmade generations."""
        return score_rows(PROMPTS, GENERATED)

    def test_one_row_per_prompt(self, rows):
        """test nothing is dropped or duplicated"""
        assert len(rows) == len(PROMPTS)

    def test_keeps_the_prompt(self, rows):
        """test each row can be traced back to what started it"""
        assert [row["prompt"] for row in rows] == PROMPTS

    def test_carries_the_generated_headline(self, rows):
        """test the model output lands in its own column"""
        assert [row["generated"] for row in rows] == GENERATED


class TestWritePredictions:
    """test writes generated headlines to csv"""

    def test_writes_expected_columns(self, tmp_path):
        """test the csv carries the prompt next to what it produced"""
        path = tmp_path / "gpt2_predictions.csv"
        rows = [{"prompt": PROMPTS[0], "generated": GENERATED[0]}]

        write_predictions(rows, path=str(path))

        with open(path, newline="", encoding="utf-8") as f:
            written = list(csv.DictReader(f))

        assert written[0]["prompt"] == PROMPTS[0]
        assert written[0]["generated"] == GENERATED[0]

import math

import pytest

from headlines.gpt2.data import load_csv, split_csv, tokenize
from headlines.gpt2.eval import evaluate_perplexity, perplexity
from headlines.gpt2.train import build_trainer


class TestPerplexity:
    """test converts a mean loss into perplexity"""

    def test_zero_loss_is_one(self):
        """test a model that never guesses wrong has perplexity one"""
        assert perplexity(0.0) == 1.0

    def test_matches_the_exponential_of_loss(self):
        """test the score is exp of the loss it came from"""
        assert perplexity(2.0) == pytest.approx(math.exp(2.0))

    def test_rises_with_loss(self):
        """test a worse model scores higher"""
        assert perplexity(3.0) > perplexity(1.0)

    def test_a_diverged_loss_is_infinite(self):
        """test a loss too large for a float reports infinity rather than raising"""
        assert perplexity(1000.0) == float("inf")


@pytest.mark.integration
class TestEvaluatePerplexity:
    """test scores a split by running the trainer over it"""

    @pytest.fixture
    def eval_dataset(self, temp_reuters_headlines, gpt2_tokenizer):
        """Tokenized held-out split of the Reuters sample."""
        _, test = split_csv(load_csv(file_path=str(temp_reuters_headlines)))

        return tokenize(test, gpt2_tokenizer)

    @pytest.fixture
    def metrics(self, eval_dataset, gpt2_model, gpt2_tokenizer, tmp_path):
        """Scores from one evaluation pass over the untrained model."""
        trainer = build_trainer(
            gpt2_model,
            gpt2_tokenizer,
            eval_dataset,
            eval_dataset,
            output_dir=str(tmp_path),
        )

        return evaluate_perplexity(trainer, eval_dataset)

    def test_reports_loss_and_perplexity(self, metrics):
        """test the caller gets the raw loss alongside its readable form"""
        assert set(metrics) == {"loss", "perplexity"}

    def test_perplexity_follows_the_loss(self, metrics):
        """test the reported score is exp of the loss reported next to it"""
        assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss"]))

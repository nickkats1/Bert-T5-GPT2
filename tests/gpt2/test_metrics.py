import math

from gpt2.metrics import perplexity


class TestPerplexity:
    def test_zero_loss_is_perplexity_one(self):
        assert perplexity(0.0) == 1.0

    def test_matches_the_exponential_of_the_loss(self):
        assert perplexity(2.0) == math.exp(2.0)

    def test_grows_with_the_loss(self):
        assert perplexity(1.0) < perplexity(2.0)

    def test_overflow_becomes_infinity(self):
        assert perplexity(1e6) == float("inf")

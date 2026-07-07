import math

import torch
import torch.nn as nn

from headlines.gpt2.metrics import generate_samples, perplexity


class TestPerplexity:
    def test_zero_loss_is_one(self):
        assert perplexity(0.0) == 1.0

    def test_matches_exp(self):
        assert perplexity(1.5) == math.exp(1.5)

    def test_monotonic_in_loss(self):
        assert perplexity(0.5) < perplexity(2.0)

    def test_overflow_returns_inf(self):
        assert perplexity(1e9) == float("inf")


class _FakeEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    bos_token = "<bos>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None):
        return _FakeEncoding(input_ids=torch.tensor([[2]]))

    def decode(self, seq, skip_special_tokens=True):
        return "  generated headline  "


class _FakeGenModel(nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def generate(self, input_ids=None, num_return_sequences=1, **kwargs):
        return torch.zeros((num_return_sequences, 4), dtype=torch.long)


class TestGenerateSamples:
    def test_returns_requested_number_of_stripped_strings(self):
        samples = generate_samples(
            _FakeGenModel(), _FakeTokenizer(), torch.device("cpu"), num_samples=3
        )
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)
        assert samples[0] == "generated headline"  # whitespace stripped

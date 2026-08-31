import numpy as np
import pytest

from headlines.t5.eval import build_compute_metrics


HEADLINES = [
    "Bank of England warns of slower recovery",
    "Oil prices steady as demand concerns offset supply cuts",
]

UNRELATED = [
    "Chipmaker shares slide after weak quarterly outlook",
    "Trade talks stall over fishing rights and state aid",
]


def encode(tokenizer, texts):
    """Pad a list of headlines into the label id array the Trainer would hand over."""
    encoded = tokenizer(texts, padding="max_length", max_length=32, truncation=True)

    return np.array(encoded["input_ids"])


class TestBuildComputeMetrics:
    """test scores generated headlines against their references"""

    @pytest.fixture
    def compute_metrics(self, t5_tokenizer):
        """ROUGE scorer bound to the tiny tokenizer."""
        return build_compute_metrics(t5_tokenizer)

    def test_identical_text_scores_perfectly(self, compute_metrics, t5_tokenizer):
        """test rouge reaches 1.0 when prediction and reference match"""
        ids = encode(t5_tokenizer, HEADLINES)

        assert compute_metrics((ids, ids))["rouge1"] == 1.0

    def test_unrelated_text_scores_below_perfect(self, compute_metrics, t5_tokenizer):
        """test the score actually discriminates rather than always passing"""
        predictions = encode(t5_tokenizer, UNRELATED)
        references = encode(t5_tokenizer, HEADLINES)

        assert compute_metrics((predictions, references))["rouge1"] < 1.0

    def test_masked_labels_are_restored_before_decoding(self, compute_metrics, t5_tokenizer):
        """test the -100 the collator writes never reaches the tokenizer"""
        ids = encode(t5_tokenizer, HEADLINES)
        masked = np.where(ids == t5_tokenizer.pad_token_id, -100, ids)

        assert compute_metrics((ids, masked))["rouge1"] == 1.0

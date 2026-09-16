import numpy as np

from t5.metrics import build_compute_metrics, clean, clean_batch


class TestClean:
    def test_collapses_repeated_spaces(self):
        assert clean("a    b") == "a b"

    def test_strips_newlines(self):
        assert clean("a\nb\n") == "a b"

    def test_leaves_a_tidy_headline_alone(self):
        assert clean("Bank of England warns of slower recovery") == "Bank of England warns of slower recovery"


class TestCleanBatch:
    def test_returns_one_string_per_input(self):
        assert len(clean_batch(["a  b", "c\nd"])) == 2

    def test_cleans_every_entry(self):
        assert clean_batch(["a  b", "c\nd"]) == ["a b", "c d"]


class TestBuildComputeMetrics:
    def test_returns_rouge_scores(self, t5_tokenizer):
        compute_metrics = build_compute_metrics(t5_tokenizer)
        ids = np.array(t5_tokenizer(["a short headline"], padding=True)["input_ids"])

        assert "rouge1" in compute_metrics((ids, ids))

    def test_identical_text_scores_perfectly(self, t5_tokenizer):
        compute_metrics = build_compute_metrics(t5_tokenizer)
        ids = np.array(t5_tokenizer(["a short headline"], padding=True)["input_ids"])

        assert compute_metrics((ids, ids))["rouge1"] == 1.0

    def test_masked_labels_are_restored_before_decoding(self, t5_tokenizer):
        compute_metrics = build_compute_metrics(t5_tokenizer)
        ids = np.array(t5_tokenizer(["a short headline"], padding=True)["input_ids"])
        masked = np.where(ids == t5_tokenizer.pad_token_id, -100, ids)

        assert compute_metrics((ids, masked))["rouge1"] == 1.0

import pytest

from headlines.t5.metrics import (
    ROUGE_KEYS,
    compute_meteor,
    compute_metrics,
    compute_rouge,
    save_predictions,
)


class TestRouge:
    def test_perfect_match_is_one(self):
        scores = compute_rouge(["the cat sat"], ["the cat sat"])
        assert set(scores) == set(ROUGE_KEYS)
        assert scores["rouge1"] == pytest.approx(1.0)

    def test_empty_returns_zeros(self):
        scores = compute_rouge([], [])
        assert scores == dict.fromkeys(ROUGE_KEYS, 0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_rouge(["a"], ["a", "b"])


class TestMeteor:
    def test_perfect_match_high(self):
        assert compute_meteor(["the cat sat"], ["the cat sat"]) > 0.9

    def test_empty_returns_zero(self):
        assert compute_meteor([], []) == 0.0


class TestComputeMetrics:
    def test_keys_present(self):
        metrics = compute_metrics(["the cat sat"], ["the cat sat"])
        assert {"rouge1", "rouge2", "rougeL", "meteor", "gen_len"} <= set(metrics)
        assert metrics["gen_len"] == pytest.approx(3.0)

    def test_no_bertscore_by_default(self):
        metrics = compute_metrics(["a b"], ["a b"])
        assert "bertscore_f1" not in metrics


class TestSavePredictions:
    def test_writes_csv(self, tmp_path):
        path = save_predictions(["pred one"], ["actual one"], path=tmp_path / "preds.csv")
        assert path.exists()
        content = path.read_text()
        assert "Generated Text" in content
        assert "pred one" in content

    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError):
            save_predictions(["a"], ["a", "b"], path=tmp_path / "x.csv")

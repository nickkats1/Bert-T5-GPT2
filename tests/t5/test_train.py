import pytest

from headlines.t5.config import T5CFG
from headlines.t5.data import load_csv, split_csv, tokenize
from headlines.t5.train import build_trainer, build_training_arguments


@pytest.fixture
def reuters_dataset(temp_reuters_headlines):
    """Reuters sample CSV loaded as a dataset."""
    return load_csv(file_path=str(temp_reuters_headlines))


class TestBuildTrainingArguments:
    """test translates the config dataclass into Trainer settings"""

    @pytest.fixture
    def arguments(self, tmp_path):
        """Training arguments pointed at a throwaway output directory."""
        return build_training_arguments(output_dir=str(tmp_path))

    def test_carries_config_values(self, arguments):
        """test the fields come from T5CFG rather than library defaults"""
        assert arguments.num_train_epochs == T5CFG.num_train_epochs
        assert arguments.learning_rate == T5CFG.learning_rate
        assert arguments.weight_decay == T5CFG.weight_decay
        assert arguments.seed == T5CFG.seed

    def test_generates_during_evaluation(self, arguments):
        """test rouge scoring receives generated text rather than logits"""
        assert arguments.predict_with_generate
        assert arguments.generation_max_length == T5CFG.max_target_length

    def test_best_model_has_a_criterion(self, arguments):
        """test load_best_model_at_end knows which metric decides"""
        assert arguments.load_best_model_at_end
        assert arguments.metric_for_best_model == T5CFG.metric_for_best_model
        assert arguments.greater_is_better

    def test_output_dir_is_overridable(self, arguments, tmp_path):
        """test smoke runs can write somewhere other than the real checkpoint"""
        assert arguments.output_dir == str(tmp_path)


class TestBuildTrainer:
    """test assembles the Trainer that runs the fine-tune"""

    @pytest.fixture
    def trainer(self, reuters_dataset, t5_model, t5_tokenizer, tmp_path):
        """Trainer built over the sample dataset."""
        train, test = split_csv(reuters_dataset)

        return build_trainer(
            t5_model,
            t5_tokenizer,
            tokenize(train, t5_tokenizer),
            tokenize(test, t5_tokenizer),
            output_dir=str(tmp_path),
        )

    def test_uses_processing_class(self, trainer, t5_tokenizer):
        """test the tokenizer is passed the way transformers v5 expects"""
        assert trainer.processing_class is t5_tokenizer

    def test_wires_both_splits(self, trainer):
        """test train and eval datasets are distinct and tokenized"""
        assert "input_ids" in trainer.train_dataset.column_names
        assert "input_ids" in trainer.eval_dataset.column_names
        assert len(trainer.train_dataset) > len(trainer.eval_dataset)

    def test_scores_with_project_metrics(self, trainer):
        """test evaluation reports rouge"""
        assert trainer.compute_metrics is not None

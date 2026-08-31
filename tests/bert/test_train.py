import pytest

from headlines.bert.config import BertCFG
from headlines.bert.data import build_label_maps, load_csv, split_csv, tokenize
from headlines.bert.eval import compute_metrics
from headlines.bert.train import build_trainer, build_training_arguments


@pytest.fixture
def guardian_dataset(temp_guardian_file):
    """Guardian sample CSV loaded as a dataset."""
    return load_csv(file_path=str(temp_guardian_file))


class TestBuildTrainingArguments:
    """test translates the config dataclass into Trainer settings"""

    @pytest.fixture
    def arguments(self, tmp_path):
        """Training arguments pointed at a throwaway output directory."""
        return build_training_arguments(output_dir=str(tmp_path))

    def test_carries_config_values(self, arguments):
        """test the fields come from BertCFG rather than library defaults"""
        assert arguments.num_train_epochs == BertCFG.num_train_epochs
        assert arguments.learning_rate == BertCFG.learning_rate
        assert arguments.weight_decay == BertCFG.weight_decay
        assert arguments.seed == BertCFG.seed

    def test_best_model_has_a_criterion(self, arguments):
        """test load_best_model_at_end knows which metric decides"""
        assert arguments.load_best_model_at_end
        assert arguments.metric_for_best_model == BertCFG.metric_for_best_model
        assert arguments.greater_is_better

    def test_output_dir_is_overridable(self, arguments, tmp_path):
        """test smoke runs can write somewhere other than the real checkpoint"""
        assert arguments.output_dir == str(tmp_path)


class TestBuildTrainer:
    """test assembles the Trainer that runs the fine-tune"""

    @pytest.fixture
    def trainer(self, guardian_dataset, bert_model, bert_tokenizer, tmp_path):
        """Trainer built over the sample dataset."""
        label_to_id, _ = build_label_maps(guardian_dataset)
        train, _, val = split_csv(guardian_dataset)

        return build_trainer(
            bert_model,
            bert_tokenizer,
            tokenize(train, bert_tokenizer, label_to_id),
            tokenize(val, bert_tokenizer, label_to_id),
            output_dir=str(tmp_path),
        )

    def test_uses_processing_class(self, trainer, bert_tokenizer):
        """test the tokenizer is passed the way transformers v5 expects"""
        assert trainer.processing_class is bert_tokenizer

    def test_wires_both_splits(self, trainer):
        """test train and eval datasets are distinct and tokenized"""
        assert "input_ids" in trainer.train_dataset.column_names
        assert "input_ids" in trainer.eval_dataset.column_names
        assert len(trainer.train_dataset) > len(trainer.eval_dataset)

    def test_scores_with_project_metrics(self, trainer):
        """test evaluation reports accuracy and weighted f1"""
        assert trainer.compute_metrics is compute_metrics

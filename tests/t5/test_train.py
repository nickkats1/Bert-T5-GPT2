import pytest
from transformers import Seq2SeqTrainer

from t5.config import seq2seq_arguments
from t5.train import build_model, build_trainer


class TestBuildTokenizer:
    def test_encodes_a_description(self, t5_tokenizer):
        assert len(t5_tokenizer("summarize: an article")["input_ids"]) > 0


class TestBuildModel:
    def test_can_generate(self, t5_model):
        assert t5_model.can_generate()

    def test_a_second_model_is_independent(self, t5_model_args):
        assert build_model(t5_model_args) is not build_model(t5_model_args)


@pytest.fixture
def t5_trainer(t5_model_args, t5_data_args, tmp_path):
    """Trainer built on the tiny checkpoint and the temporary CSV."""
    args = seq2seq_arguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
    )

    return build_trainer(t5_model_args, t5_data_args, args)


class TestBuildTrainer:
    def test_returns_a_seq2seq_trainer(self, t5_trainer):
        assert isinstance(t5_trainer, Seq2SeqTrainer)

    def test_holds_a_training_split(self, t5_trainer):
        assert len(t5_trainer.train_dataset) > 0

    def test_scores_with_rouge(self, t5_trainer):
        assert t5_trainer.compute_metrics is not None

    def test_stops_only_at_the_last_epoch(self, t5_trainer):
        assert not any(
            type(callback).__name__ == "EarlyStoppingCallback" for callback in t5_trainer.callback_handler.callbacks
        )

    @pytest.mark.integration
    def test_trains(self, t5_trainer):
        assert t5_trainer.train().training_loss is not None

import pytest
from transformers import Trainer

from gpt2.config import training_arguments
from gpt2.metrics import evaluate_perplexity
from gpt2.train import build_model, build_trainer


class TestBuildModel:
    def test_embeddings_cover_the_pad_token(self, gpt2_model, gpt2_tokenizer):
        assert gpt2_model.get_input_embeddings().num_embeddings >= len(gpt2_tokenizer)

    def test_can_generate(self, gpt2_model):
        assert gpt2_model.can_generate()

    def test_a_second_model_is_independent(self, gpt2_model_args, gpt2_tokenizer):
        first = build_model(gpt2_model_args, gpt2_tokenizer)
        second = build_model(gpt2_model_args, gpt2_tokenizer)

        assert first is not second


@pytest.fixture
def gpt2_trainer(gpt2_model_args, gpt2_data_args, tmp_path):
    """Trainer built on the tiny checkpoint and the temporary CSV."""
    args = training_arguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
    )

    return build_trainer(gpt2_model_args, gpt2_data_args, args)


class TestBuildTrainer:
    def test_returns_a_trainer(self, gpt2_trainer):
        assert isinstance(gpt2_trainer, Trainer)

    def test_holds_a_training_split(self, gpt2_trainer):
        assert len(gpt2_trainer.train_dataset) > 0

    def test_evaluates_on_the_validation_split(self, gpt2_trainer):
        assert len(gpt2_trainer.eval_dataset) > 0

    def test_stops_only_at_the_last_epoch(self, gpt2_trainer):
        assert not any(
            type(callback).__name__ == "EarlyStoppingCallback" for callback in gpt2_trainer.callback_handler.callbacks
        )

    @pytest.mark.integration
    def test_trains(self, gpt2_trainer):
        assert gpt2_trainer.train().training_loss is not None

    def test_scores_perplexity_on_the_validation_split(self, gpt2_trainer):
        scores = evaluate_perplexity(gpt2_trainer, gpt2_trainer.eval_dataset)

        assert scores["perplexity"] > 0

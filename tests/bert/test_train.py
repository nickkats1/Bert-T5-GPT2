import pytest
from transformers import Trainer

from bert.config import training_arguments
from bert.train import build_model, build_trainer
from bert.utils_bert import ID2LABEL, LABEL2ID


class TestBuildTokenizer:
    def test_loads_the_requested_checkpoint(self, bert_tokenizer):
        assert bert_tokenizer.pad_token is not None

    def test_encodes_a_headline(self, bert_tokenizer):
        assert len(bert_tokenizer("a headline")["input_ids"]) > 0


class TestBuildModel:
    def test_head_is_sized_to_the_label_set(self, bert_model):
        assert bert_model.config.num_labels == len(ID2LABEL)

    def test_carries_the_id_to_label_mapping(self, bert_model):
        assert bert_model.config.id2label == ID2LABEL

    def test_carries_the_label_to_id_mapping(self, bert_model):
        assert bert_model.config.label2id == LABEL2ID

    def test_a_second_model_is_independent(self, bert_model_args):
        assert build_model(bert_model_args) is not build_model(bert_model_args)


@pytest.fixture
def bert_trainer(bert_model_args, bert_data_args, tmp_path):
    """Trainer built on the tiny checkpoint and the temporary CSV."""
    args = training_arguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )

    return build_trainer(bert_model_args, bert_data_args, args)


class TestBuildTrainer:
    def test_returns_a_trainer(self, bert_trainer):
        assert isinstance(bert_trainer, Trainer)

    def test_holds_a_training_split(self, bert_trainer):
        assert len(bert_trainer.train_dataset) > 0

    def test_evaluates_on_the_validation_split(self, bert_trainer):
        assert len(bert_trainer.eval_dataset) > 0

    def test_scores_predictions_during_evaluation(self, bert_trainer):
        assert bert_trainer.compute_metrics is not None

    def test_stops_only_at_the_last_epoch(self, bert_trainer):
        assert not any(
            type(callback).__name__ == "EarlyStoppingCallback" for callback in bert_trainer.callback_handler.callbacks
        )

    @pytest.mark.integration
    def test_trains_and_predicts(self, bert_trainer):
        bert_trainer.train()
        metrics = bert_trainer.predict(bert_trainer.eval_dataset).metrics

        assert "test_f1_weighted" in metrics

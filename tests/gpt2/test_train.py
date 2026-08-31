import pytest
from transformers import DataCollatorForLanguageModeling

from headlines.gpt2.config import Gpt2CFG
from headlines.gpt2.data import load_csv, split_csv, tokenize
from headlines.gpt2.train import build_trainer, build_training_arguments


@pytest.fixture
def headline_dataset(temp_reuters_headlines):
    """Reuters sample CSV loaded as a headline-only dataset."""
    return load_csv(file_path=str(temp_reuters_headlines))


class TestBuildTokenizer:
    """test gives gpt2 the pad token it ships without"""

    def test_has_a_pad_token(self, gpt2_tokenizer):
        """test padding to max length is possible at all"""
        assert gpt2_tokenizer.pad_token == Gpt2CFG.pad_token


class TestBuildModel:
    """test sizes the model to the padded tokenizer"""

    def test_embeddings_cover_the_pad_token(self, gpt2_model, gpt2_tokenizer):
        """test the added token id does not index past the embedding matrix"""
        assert gpt2_model.get_input_embeddings().num_embeddings >= len(gpt2_tokenizer)


class TestBuildTrainingArguments:
    """test translates the config dataclass into Trainer settings"""

    @pytest.fixture
    def arguments(self, tmp_path):
        """Training arguments pointed at a throwaway output directory."""
        return build_training_arguments(output_dir=str(tmp_path))

    def test_carries_config_values(self, arguments):
        """test the fields come from Gpt2CFG rather than library defaults"""
        assert arguments.num_train_epochs == Gpt2CFG.num_train_epochs
        assert arguments.learning_rate == Gpt2CFG.learning_rate
        assert arguments.weight_decay == Gpt2CFG.weight_decay
        assert arguments.seed == Gpt2CFG.seed

    def test_best_model_tracks_falling_loss(self, arguments):
        """test the winning checkpoint is the one with the lowest loss"""
        assert arguments.load_best_model_at_end
        assert arguments.metric_for_best_model == Gpt2CFG.metric_for_best_model
        assert not arguments.greater_is_better

    def test_output_dir_is_overridable(self, arguments, tmp_path):
        """test smoke runs can write somewhere other than the real checkpoint"""
        assert arguments.output_dir == str(tmp_path)


class TestBuildTrainer:
    """test assembles the Trainer that runs the fine-tune"""

    @pytest.fixture
    def trainer(self, headline_dataset, gpt2_model, gpt2_tokenizer, tmp_path):
        """Trainer built over the sample dataset."""
        train, test = split_csv(headline_dataset)

        return build_trainer(
            gpt2_model,
            gpt2_tokenizer,
            tokenize(train, gpt2_tokenizer),
            tokenize(test, gpt2_tokenizer),
            output_dir=str(tmp_path),
        )

    def test_uses_processing_class(self, trainer, gpt2_tokenizer):
        """test the tokenizer is passed the way transformers v5 expects"""
        assert trainer.processing_class is gpt2_tokenizer

    def test_wires_both_splits(self, trainer):
        """test train and eval datasets are distinct and tokenized"""
        assert "input_ids" in trainer.train_dataset.column_names
        assert "input_ids" in trainer.eval_dataset.column_names
        assert len(trainer.train_dataset) > len(trainer.eval_dataset)

    def test_collates_without_masking(self, trainer):
        """test the collator trains causally rather than like bert"""
        assert isinstance(trainer.data_collator, DataCollatorForLanguageModeling)
        assert not trainer.data_collator.mlm

    def test_collator_ignores_padding_in_the_loss(self, trainer, gpt2_tokenizer):
        """test padded positions are masked to -100 and real tokens are kept"""
        rows = [dict(row) for row in trainer.train_dataset.select(range(2))]
        batch = trainer.data_collator(rows)

        padding = batch["input_ids"] == gpt2_tokenizer.pad_token_id

        assert (batch["labels"][padding] == -100).all()
        assert (batch["labels"][~padding] == batch["input_ids"][~padding]).all()

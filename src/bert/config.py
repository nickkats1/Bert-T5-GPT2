from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass(frozen=True)
class ClassificationModelArguments:
    """Which checkpoint to fine-tune."""

    model_name_or_path: str = "bert-base-uncased"


@dataclass(frozen=True)
class ClassificationDataArguments:
    """Where the headlines live and how they are split and truncated."""

    data_path: str = "data/guardian_headlines.csv"
    text_column: str = "Headlines"
    max_length: int = 64
    test_size: float = 0.2


CLASSIFICATION = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_weighted",
    "greater_is_better": True,
    "fp16": True,
    "logging_steps": 20,
    "seed": 42,
    "report_to": "none",
}


def training_arguments(output_dir: str = "bert-headlines", **overrides) -> TrainingArguments:
    """Build Trainer settings from CLASSIFICATION, letting keyword overrides win."""
    return TrainingArguments(output_dir=output_dir, **{**CLASSIFICATION, **overrides})

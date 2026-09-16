"""Arguments for fine-tuning T5 to summarize descriptions into headlines."""

from dataclasses import dataclass

from transformers import Seq2SeqTrainingArguments


@dataclass(frozen=True)
class SummarizationModelArguments:
    """Which checkpoint to fine-tune."""

    model_name_or_path: str = "t5-base"


@dataclass(frozen=True)
class SummarizationDataArguments:
    """Where the articles live, which columns form each pair, and how they are truncated."""

    data_path: str = "data/reuters_headlines.csv"
    source_column: str = "Description"
    target_column: str = "Headlines"
    source_prefix: str = "summarize: "
    max_source_length: int = 128
    max_target_length: int = 32
    test_size: float = 0.2
    max_eval_samples: int | None = None


SUMMARIZATION = {
    "num_train_epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "eval_strategy": "steps",
    "eval_steps": 250,
    "save_strategy": "steps",
    "save_steps": 250,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "rouge1",
    "greater_is_better": True,
    "predict_with_generate": True,
    "generation_max_length": SummarizationDataArguments.max_target_length,
    "generation_num_beams": 1,
    "fp16": False,
    "logging_steps": 100,
    "seed": 42,
    "report_to": "none",
}


def seq2seq_arguments(output_dir: str = "t5-headlines", **overrides) -> Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainer settings from SUMMARIZATION, letting keyword overrides win."""
    return Seq2SeqTrainingArguments(output_dir=output_dir, **{**SUMMARIZATION, **overrides})

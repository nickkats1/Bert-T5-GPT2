"""Fine-tuning of T5 for headline summarization."""

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from headlines.t5.config import T5CFG
from headlines.t5.data import load_csv, split_csv, tokenize
from headlines.t5.eval import build_compute_metrics


def build_tokenizer(model_name: str = T5CFG.model_name) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the configured checkpoint, overridable for tests."""
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str = T5CFG.model_name) -> PreTrainedModel:
    """Load the sequence-to-sequence model to fine-tune, overridable for tests."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def build_training_arguments(
    output_dir: str = T5CFG.output_dir,
) -> Seq2SeqTrainingArguments:
    """Translate the config dataclass into Trainer settings.

    Args:
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        arguments describing the fine-tuning run.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=T5CFG.num_train_epochs,
        learning_rate=T5CFG.learning_rate,
        weight_decay=T5CFG.weight_decay,
        per_device_train_batch_size=T5CFG.per_device_train_batch_size,
        per_device_eval_batch_size=T5CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=T5CFG.gradient_accumulation_steps,
        optim=T5CFG.optim,
        logging_strategy=T5CFG.logging_strategy,
        logging_steps=T5CFG.logging_steps,
        eval_strategy=T5CFG.eval_strategy,
        save_strategy=T5CFG.save_strategy,
        save_total_limit=T5CFG.save_total_limit,
        load_best_model_at_end=T5CFG.load_best_model_at_end,
        metric_for_best_model=T5CFG.metric_for_best_model,
        greater_is_better=T5CFG.greater_is_better,
        remove_unused_columns=T5CFG.remove_unused_columns,
        predict_with_generate=T5CFG.predict_with_generate,
        generation_max_length=T5CFG.max_target_length,
        generation_num_beams=T5CFG.generation_num_beams,
        fp16=T5CFG.fp16,
        seed=T5CFG.seed,
        report_to=T5CFG.report_to,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = T5CFG.output_dir,
) -> Seq2SeqTrainer:
    """Assemble the Trainer that runs the fine-tune.

    Args:
        model: summarization model from build_model.
        tokenizer: tokenizer from build_tokenizer.
        train_dataset: tokenized training split.
        eval_dataset: tokenized test split.
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        trainer ready for train and predict.
    """
    return Seq2SeqTrainer(
        model=model,
        args=build_training_arguments(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_compute_metrics(tokenizer),
    )


def main() -> None:
    """Fine-tune T5 on the Reuters descriptions and save the checkpoint."""
    dataset = load_csv(T5CFG.data_path)
    train, test = split_csv(dataset)

    tokenizer = build_tokenizer()
    model = build_model()

    trainer = build_trainer(
        model,
        tokenizer,
        tokenize(train, tokenizer),
        tokenize(test, tokenizer),
    )
    trainer.train()

    model.save_pretrained(T5CFG.output_dir)
    tokenizer.save_pretrained(T5CFG.output_dir)
    print(f"Saved checkpoint to {T5CFG.output_dir}")


if __name__ == "__main__":
    main()

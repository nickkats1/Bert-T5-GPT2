"""Fine-tuning of BERT for headline classification."""

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from headlines.bert.config import BertCFG
from headlines.bert.data import build_label_maps, load_csv, split_csv, tokenize
from headlines.bert.eval import compute_metrics


def build_tokenizer(model_name: str = BertCFG.model_name) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the configured checkpoint, overridable for tests."""
    return AutoTokenizer.from_pretrained(model_name)


def build_model(
    label_to_id: dict[object, int],
    id_to_label: dict[int, object],
    model_name: str = BertCFG.model_name,
) -> PreTrainedModel:
    """Load the classification head sized to the labels found in the data.

    Args:
        label_to_id: mapping produced by build_label_maps.
        id_to_label: inverse of label_to_id, stored on the model so predictions
            report real class names instead of LABEL_0.
        model_name: checkpoint to load, overridable for tests.

    Returns:
        model ready to fine-tune.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label={i: str(label) for i, label in id_to_label.items()},
        label2id={str(label): i for label, i in label_to_id.items()},
    )


def build_training_arguments(output_dir: str = BertCFG.output_dir) -> TrainingArguments:
    """Translate the config dataclass into Trainer settings.

    Args:
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        arguments describing the fine-tuning run.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=BertCFG.num_train_epochs,
        learning_rate=BertCFG.learning_rate,
        weight_decay=BertCFG.weight_decay,
        per_device_train_batch_size=BertCFG.per_device_train_batch_size,
        per_device_eval_batch_size=BertCFG.per_device_eval_batch_size,
        gradient_accumulation_steps=BertCFG.gradient_accumulation_steps,
        optim=BertCFG.optim,
        logging_strategy=BertCFG.logging_strategy,
        logging_steps=BertCFG.logging_steps,
        eval_strategy=BertCFG.eval_strategy,
        save_strategy=BertCFG.save_strategy,
        save_total_limit=BertCFG.save_total_limit,
        load_best_model_at_end=BertCFG.load_best_model_at_end,
        metric_for_best_model=BertCFG.metric_for_best_model,
        greater_is_better=BertCFG.greater_is_better,
        remove_unused_columns=BertCFG.remove_unused_columns,
        fp16=BertCFG.fp16,
        seed=BertCFG.seed,
        report_to=BertCFG.report_to,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = BertCFG.output_dir,
) -> Trainer:
    """Assemble the Trainer that runs the fine-tune.

    Args:
        model: classification model from build_model.
        tokenizer: tokenizer from build_tokenizer.
        train_dataset: tokenized training split.
        eval_dataset: tokenized validation split.
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        trainer ready for train and predict.
    """
    return Trainer(
        model=model,
        args=build_training_arguments(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def main() -> None:
    """Fine-tune BERT on the Guardian headlines and save the checkpoint."""
    dataset = load_csv(BertCFG.data_path)
    train, _, val = split_csv(dataset)

    label_to_id, id_to_label = build_label_maps(dataset)
    tokenizer = build_tokenizer()
    model = build_model(label_to_id, id_to_label)

    trainer = build_trainer(
        model,
        tokenizer,
        tokenize(train, tokenizer, label_to_id),
        tokenize(val, tokenizer, label_to_id),
    )
    trainer.train()

    model.save_pretrained(BertCFG.output_dir)
    tokenizer.save_pretrained(BertCFG.output_dir)
    print(f"Saved checkpoint to {BertCFG.output_dir}")


if __name__ == "__main__":
    main()

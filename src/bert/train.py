"""Fine-tune BERT to classify headline sentiment."""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from bert.config import ClassificationDataArguments, ClassificationModelArguments, training_arguments
from bert.data import build_datasets
from bert.metrics import compute_metrics
from bert.utils_bert import ID2LABEL, LABEL2ID


def build_tokenizer(model_args: ClassificationModelArguments) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the checkpoint."""
    return AutoTokenizer.from_pretrained(model_args.model_name_or_path)


def build_model(model_args: ClassificationModelArguments) -> PreTrainedModel:
    """Load the checkpoint with a classification head sized to the three sentiments."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )


def build_trainer(
    model_args: ClassificationModelArguments,
    data_args: ClassificationDataArguments,
    training_args: TrainingArguments,
) -> Trainer:
    """Assemble the tokenizer, model and datasets into a Trainer."""
    tokenizer = build_tokenizer(model_args)
    datasets = build_datasets(data_args, tokenizer, seed=training_args.seed)

    return Trainer(
        model=build_model(model_args),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def main() -> None:
    """Fine-tune on the Guardian headlines and save the best checkpoint."""
    training_args = training_arguments()
    trainer = build_trainer(ClassificationModelArguments(), ClassificationDataArguments(), training_args)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

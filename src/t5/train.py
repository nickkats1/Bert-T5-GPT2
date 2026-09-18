from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from t5.config import SummarizationDataArguments, SummarizationModelArguments, seq2seq_arguments
from t5.data import build_datasets
from t5.metrics import build_compute_metrics


def build_tokenizer(model_args: SummarizationModelArguments) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the checkpoint."""
    return AutoTokenizer.from_pretrained(model_args.model_name_or_path)


def build_model(model_args: SummarizationModelArguments) -> PreTrainedModel:
    """Load the encoder-decoder checkpoint."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)


def build_trainer(
    model_args: SummarizationModelArguments,
    data_args: SummarizationDataArguments,
    training_args: Seq2SeqTrainingArguments,
) -> Seq2SeqTrainer:
    """Assemble the tokenizer, model and datasets into a Seq2SeqTrainer."""
    tokenizer = build_tokenizer(model_args)
    model = build_model(model_args)
    datasets = build_datasets(data_args, tokenizer, seed=training_args.seed)

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_compute_metrics(tokenizer),
    )


def main() -> None:
    """Fine-tune on the Reuters pairs and save the best checkpoint."""
    training_args = seq2seq_arguments()
    trainer = build_trainer(SummarizationModelArguments(), SummarizationDataArguments(), training_args)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

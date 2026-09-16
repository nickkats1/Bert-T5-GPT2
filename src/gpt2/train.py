"""Fine-tune GPT-2 to generate headlines."""

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from gpt2.config import ClmDataArguments, ClmModelArguments, training_arguments
from gpt2.data import build_datasets
from gpt2.tokenizer import load_tokenizer


def build_model(model_args: ClmModelArguments, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
    """Load the checkpoint and grow its embeddings to cover the added pad token."""
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    return model


def build_trainer(
    model_args: ClmModelArguments, data_args: ClmDataArguments, training_args: TrainingArguments
) -> Trainer:
    """Assemble the tokenizer, model and datasets into a Trainer."""
    tokenizer = load_tokenizer(model_args)
    datasets = build_datasets(data_args, tokenizer, seed=training_args.seed)

    return Trainer(
        model=build_model(model_args, tokenizer),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )


def main() -> None:
    """Fine-tune on the Reuters headlines and save the best checkpoint."""
    training_args = training_arguments()
    trainer = build_trainer(ClmModelArguments(), ClmDataArguments(), training_args)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

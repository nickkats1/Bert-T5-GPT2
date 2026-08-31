"""Fine-tuning of GPT-2 for headline generation."""

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from headlines.gpt2.config import Gpt2CFG
from headlines.gpt2.data import load_csv, split_csv, tokenize


def build_tokenizer(model_name: str = Gpt2CFG.model_name) -> PreTrainedTokenizerBase:
    """Load the tokenizer and give it the pad token GPT-2 ships without."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": Gpt2CFG.pad_token})

    return tokenizer


def build_model(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str = Gpt2CFG.model_name,
) -> PreTrainedModel:
    """Load the causal model and grow its embeddings to fit the pad token.

    Args:
        tokenizer: tokenizer from build_tokenizer, already carrying the pad token.
        model_name: checkpoint to load, overridable for tests.

    Returns:
        model ready to fine-tune.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model


def build_training_arguments(output_dir: str = Gpt2CFG.output_dir) -> TrainingArguments:
    """Translate the config dataclass into Trainer settings.

    Args:
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        arguments describing the fine-tuning run.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Gpt2CFG.num_train_epochs,
        learning_rate=Gpt2CFG.learning_rate,
        weight_decay=Gpt2CFG.weight_decay,
        per_device_train_batch_size=Gpt2CFG.per_device_train_batch_size,
        per_device_eval_batch_size=Gpt2CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=Gpt2CFG.gradient_accumulation_steps,
        optim=Gpt2CFG.optim,
        logging_strategy=Gpt2CFG.logging_strategy,
        logging_steps=Gpt2CFG.logging_steps,
        eval_strategy=Gpt2CFG.eval_strategy,
        save_strategy=Gpt2CFG.save_strategy,
        save_total_limit=Gpt2CFG.save_total_limit,
        load_best_model_at_end=Gpt2CFG.load_best_model_at_end,
        metric_for_best_model=Gpt2CFG.metric_for_best_model,
        greater_is_better=Gpt2CFG.greater_is_better,
        remove_unused_columns=Gpt2CFG.remove_unused_columns,
        fp16=Gpt2CFG.fp16,
        seed=Gpt2CFG.seed,
        report_to=Gpt2CFG.report_to,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = Gpt2CFG.output_dir,
) -> Trainer:
    """Assemble the Trainer that runs the fine-tune.

    Args:
        model: causal model from build_model.
        tokenizer: tokenizer from build_tokenizer.
        train_dataset: tokenized training split.
        eval_dataset: tokenized test split.
        output_dir: where checkpoints are written, overridable for smoke runs.

    Returns:
        trainer ready for train and evaluate.
    """
    return Trainer(
        model=model,
        args=build_training_arguments(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


def main() -> None:
    """Fine-tune GPT-2 on the Reuters headlines and save the checkpoint."""
    dataset = load_csv(Gpt2CFG.data_path)
    train, test = split_csv(dataset)

    tokenizer = build_tokenizer()
    model = build_model(tokenizer)

    trainer = build_trainer(
        model,
        tokenizer,
        tokenize(train, tokenizer),
        tokenize(test, tokenizer),
    )
    trainer.train()

    model.save_pretrained(Gpt2CFG.output_dir)
    tokenizer.save_pretrained(Gpt2CFG.output_dir)
    print(f"Saved checkpoint to {Gpt2CFG.output_dir}")


if __name__ == "__main__":
    main()

"""Inference against the fine-tuned GPT-2 headline generator."""

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from headlines.gpt2.config import Gpt2CFG
from headlines.gpt2.data import load_csv, split_csv, tokenize
from headlines.gpt2.eval import evaluate_perplexity
from headlines.gpt2.train import build_trainer


PREDICTIONS_PATH = "gpt2_predictions.csv"

SAMPLE_SIZE = 20


def load_trained(
    output_dir: str = Gpt2CFG.output_dir,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the fine-tuned model and tokenizer from disk.

    Args:
        output_dir: checkpoint directory written by train.main.

    Returns:
        tuple of model in eval mode and its tokenizer.

    Raises:
        FileNotFoundError: if the checkpoint does not exist yet.
    """
    if not Path(output_dir).is_dir():
        raise FileNotFoundError(f"No checkpoint at {output_dir}. Run python -m headlines.gpt2.train first.")

    model = AutoModelForCausalLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.eval()

    return model, tokenizer


def generate_headlines(
    prompts: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[str]:
    """Continue each prompt into a full headline.

    Args:
        prompts: opening words to continue.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many prompts to push through at once.

    Returns:
        one generated headline per prompt, prompt text included.
    """
    device = model.device
    generated = []

    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            list(prompts[start : start + batch_size]),
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            tokens = model.generate(
                **encoded,
                max_new_tokens=Gpt2CFG.max_new_tokens,
                num_return_sequences=Gpt2CFG.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))

    return generated


def score_rows(prompts: Sequence[str], generated: Sequence[str]) -> list[dict[str, object]]:
    """Pair each prompt with the headline the model wrote from it.

    Args:
        prompts: text the headlines were generated from.
        generated: output of generate_headlines.

    Returns:
        one mapping per prompt with its generated headline.
    """
    return [{"prompt": prompt, "generated": headline} for prompt, headline in zip(prompts, generated, strict=True)]


def predict(
    prompts: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    """Generate headlines from prompts and report each alongside its prompt.

    Args:
        prompts: opening words to continue.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many prompts to push through at once.

    Returns:
        one mapping per prompt with its generated headline.
    """
    return score_rows(prompts, generate_headlines(prompts, model, tokenizer, batch_size))


def write_predictions(rows: list[dict[str, object]], path: str = PREDICTIONS_PATH) -> None:
    """Write generated headlines to CSV.

    Args:
        rows: mappings produced by predict.
        path: destination file.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Score the held-out test split and sample headlines from the checkpoint."""
    model, tokenizer = load_trained()

    _, test = split_csv(load_csv(Gpt2CFG.data_path))

    tokenized = tokenize(test, tokenizer)
    trainer = build_trainer(model, tokenizer, None, tokenized)
    scores = evaluate_perplexity(trainer, tokenized)

    for name, value in scores.items():
        print(f"{name:<12} {value:.4f}")

    prompts = [" ".join(headline.split()[:3]) for headline in test["Headlines"][:SAMPLE_SIZE]]
    rows = predict(prompts, model, tokenizer)

    write_predictions(rows)
    print(f"Wrote {len(rows)} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

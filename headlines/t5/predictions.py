"""Inference against the fine-tuned T5 headline summarizer."""

import csv
from collections.abc import Sequence
from pathlib import Path

import evaluate
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from headlines.t5.config import T5CFG
from headlines.t5.data import drop_empty_rows, load_csv, split_csv


PREDICTIONS_PATH = "t5_predictions.csv"


def load_trained(
    output_dir: str = T5CFG.output_dir,
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
        raise FileNotFoundError(f"No checkpoint at {output_dir}. Run python -m headlines.t5.train first.")

    model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.eval()

    return model, tokenizer


def generate_headlines(
    descriptions: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[str]:
    """Summarize article descriptions into headlines.

    Args:
        descriptions: article text to summarize.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many descriptions to push through at once.

    Returns:
        one generated headline per description.
    """
    device = model.device
    generated = []

    for start in range(0, len(descriptions), batch_size):
        encoded = tokenizer(
            [T5CFG.source_prefix + text for text in descriptions[start : start + batch_size]],
            truncation=True,
            max_length=T5CFG.max_source_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            tokens = model.generate(
                **encoded,
                max_new_tokens=T5CFG.max_target_length,
                num_beams=T5CFG.generation_num_beams,
            )

        generated.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))

    return generated


def score_rows(descriptions: Sequence[str], generated: Sequence[str]) -> list[dict[str, object]]:
    """Pair each description with the headline the model wrote for it.

    Args:
        descriptions: text the headlines were generated from.
        generated: output of generate_headlines.

    Returns:
        one mapping per description with its predicted headline.
    """
    return [
        {"Description": description, "pred_headline": headline}
        for description, headline in zip(descriptions, generated, strict=True)
    ]


def predict(
    descriptions: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    """Summarize descriptions and report the headline written for each.

    Args:
        descriptions: article text to summarize.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many descriptions to push through at once.

    Returns:
        one mapping per description with its predicted headline.
    """
    return score_rows(descriptions, generate_headlines(descriptions, model, tokenizer, batch_size))


def write_predictions(rows: list[dict[str, object]], path: str = PREDICTIONS_PATH) -> None:
    """Write generated headlines to CSV.

    Args:
        rows: mappings produced by main, carrying the true headline as well.
        path: destination file.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Description", "true_headline", "pred_headline"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Score the held-out test split with the saved checkpoint."""
    model, tokenizer = load_trained()

    _, test = split_csv(load_csv(T5CFG.data_path))
    test = drop_empty_rows(test)

    generated = generate_headlines(test["Description"], model, tokenizer)
    rows = score_rows(test["Description"], generated)

    for row, true_headline in zip(rows, test["Headlines"], strict=True):
        row["true_headline"] = true_headline

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=generated, references=test["Headlines"])

    for name, value in scores.items():
        print(f"{name:<12} {value:.4f}")

    write_predictions(rows)
    print(f"Wrote {len(rows)} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

"""Inference against the fine-tuned BERT headline classifier."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from headlines.bert.config import BertCFG
from headlines.bert.data import load_csv, split_csv
from headlines.bert.eval import compute_metrics


PREDICTIONS_PATH = "predictions.csv"


def load_trained(
    output_dir: str = BertCFG.output_dir,
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
        raise FileNotFoundError(f"No checkpoint at {output_dir}. Run python -m headlines.bert.train first.")

    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.eval()

    return model, tokenizer


def predict_logits(
    headlines: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> np.ndarray:
    """Run the model over raw headlines and collect the raw scores.

    Args:
        headlines: text to classify.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many headlines to push through at once.

    Returns:
        array of shape (len(headlines), num_labels).
    """
    device = model.device
    batches = []

    for start in range(0, len(headlines), batch_size):
        encoded = tokenizer(
            list(headlines[start : start + batch_size]),
            truncation=True,
            max_length=BertCFG.max_length,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            batches.append(model(**encoded).logits.float().cpu().numpy())

    return np.concatenate(batches)


def score_rows(headlines: Sequence[str], logits: np.ndarray) -> list[dict[str, object]]:
    """Turn raw scores into the winning label and its probability.

    Args:
        headlines: text the logits were produced from.
        logits: array from predict_logits.

    Returns:
        one mapping per headline with the predicted label and its probability.
    """
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predicted = probabilities.argmax(axis=-1)

    return [
        {
            "Headlines": headline,
            "pred_label": int(label),
            "score": float(probabilities[row, label]),
        }
        for row, (headline, label) in enumerate(zip(headlines, predicted, strict=True))
    ]


def predict(
    headlines: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    """Classify headlines and report the winning label with its confidence.

    Args:
        headlines: text to classify.
        model: model returned by load_trained or build_model.
        tokenizer: tokenizer matching the model.
        batch_size: how many headlines to push through at once.

    Returns:
        one mapping per headline with the predicted label and its probability.
    """
    return score_rows(headlines, predict_logits(headlines, model, tokenizer, batch_size))


def write_predictions(rows: list[dict[str, object]], path: str = PREDICTIONS_PATH) -> None:
    """Write scored headlines to CSV.

    Args:
        rows: mappings produced by main, carrying the true label as well.
        path: destination file.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Headlines", "true_label", "pred_label", "score"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Score the held-out test split with the saved checkpoint."""
    model, tokenizer = load_trained()

    _, test, _ = split_csv(load_csv(BertCFG.data_path))

    logits = predict_logits(test["Headlines"], model, tokenizer)
    rows = score_rows(test["Headlines"], logits)

    for row, true_label in zip(rows, test["labels"], strict=True):
        row["true_label"] = true_label

    scores = compute_metrics((logits, np.array(test["labels"])))

    for name, value in scores.items():
        print(f"{name:<12} {value:.4f}")

    write_predictions(rows)
    print(f"Wrote {len(rows)} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

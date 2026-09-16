"""Perplexity of the language model on held-out headlines."""

import math

from datasets import Dataset
from transformers import Trainer


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss into perplexity, saturating at infinity."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluate_perplexity(trainer: Trainer, dataset: Dataset) -> dict[str, float]:
    """Evaluate the trainer's model on the dataset and report loss and perplexity."""
    loss = trainer.evaluate(dataset)["eval_loss"]

    return {"loss": loss, "perplexity": perplexity(loss)}

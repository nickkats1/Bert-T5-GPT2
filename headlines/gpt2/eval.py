"""Metric computation for GPT-2 headline generation."""

import math

from datasets import Dataset
from transformers import Trainer


MAX_EXPONENT = 709.0


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss into perplexity.

    Args:
        loss: mean cross-entropy over the evaluated tokens.

    Returns:
        perplexity, the exponential of the loss, or infinity when the loss is
        too large for a float to hold.
    """
    if loss > MAX_EXPONENT:
        return float("inf")

    return math.exp(loss)


def evaluate_perplexity(trainer: Trainer, dataset: Dataset) -> dict[str, float]:
    """Run evaluation over a split and report loss alongside its perplexity.

    Perplexity comes from the loss the Trainer already averages rather than from
    compute_metrics, which would need the full logit tensor held in memory.

    Args:
        trainer: trainer wrapping the model to score.
        dataset: tokenized split to evaluate.

    Returns:
        mapping of metric name to score.
    """
    metrics = trainer.evaluate(eval_dataset=dataset)
    loss = metrics["eval_loss"]

    return {"loss": loss, "perplexity": perplexity(loss)}

"""Metric computation for T5 headline summarization."""

from collections.abc import Callable

import evaluate
import numpy as np
from transformers import PreTrainedTokenizerBase


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:
    """Build the ROUGE metric function bound to a tokenizer.

    Generated headlines arrive as token ids, so scoring needs the tokenizer that
    produced them to turn both sides back into text.

    Args:
        tokenizer: tokenizer matching the model being evaluated.

    Returns:
        function the Trainer calls with its generated ids and label ids.
    """
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """Compute ROUGE scores from generated headlines and references."""
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        return rouge.compute(
            predictions=tokenizer.batch_decode(predictions, skip_special_tokens=True),
            references=tokenizer.batch_decode(labels, skip_special_tokens=True),
        )

    return compute_metrics

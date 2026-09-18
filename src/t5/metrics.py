from collections.abc import Callable

import numpy as np
from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizerBase


ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def clean(text: str) -> str:
    """Collapse whitespace and newlines into single spaces."""
    return " ".join(text.split())


def clean_batch(texts: list[str]) -> list[str]:
    """Clean every string in the batch."""
    return [clean(text) for text in texts]


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable[[tuple], dict[str, float]]:
    """Close over the tokenizer so the Seq2SeqTrainer can decode ids before scoring."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)

    def decode(ids: np.ndarray) -> list[str]:
        restored = np.where(ids != -100, ids, tokenizer.pad_token_id)
        return clean_batch(tokenizer.batch_decode(restored, skip_special_tokens=True))

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        scores = [
            scorer.score(reference, prediction)
            for prediction, reference in zip(decode(predictions), decode(labels), strict=True)
        ]

        return {key: float(np.mean([score[key].fmeasure for score in scores])) for key in ROUGE_KEYS}

    return compute_metrics

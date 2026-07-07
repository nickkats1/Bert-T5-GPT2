"""Summarization metrics and prediction-saving helpers for the T5 pipeline.

Summaries are scored on three complementary levels:

* **ROUGE-1/2/L** — lexical n-gram/longest-subsequence overlap (the de-facto
  standard for summarization).
* **METEOR** — stem- and synonym-aware unigram matching with a fragmentation
  penalty; correlates with human judgment better than BLEU on short texts.
* **BERTScore** *(opt-in)* — contextual-embedding cosine similarity, the metric
  that best tracks semantic adequacy. Enabled via :func:`compute_bertscore`;
  requires the optional ``bert_score`` dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def _ensure_wordnet() -> None:
    """Best-effort ensure WordNet is present so METEOR can match synonyms."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:  # pragma: no cover - offline; METEOR still runs degraded
            pass


def compute_rouge(predictions: Sequence[str], actuals: Sequence[str]) -> dict[str, float]:
    """Compute average ROUGE-1/2/L F1 over prediction-reference pairs.

    Args:
        predictions: Generated headline strings.
        actuals: Ground-truth headline strings.

    Returns:
        Mapping ``{"rouge1": ..., "rouge2": ..., "rougeL": ...}`` of average
        F1 scores, all zero when ``predictions`` is empty.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    _check_lengths(predictions, actuals)

    n = len(predictions)
    if n == 0:
        return dict.fromkeys(ROUGE_KEYS, 0.0)

    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    totals = dict.fromkeys(ROUGE_KEYS, 0.0)

    for pred, actual in zip(predictions, actuals, strict=True):
        scores = scorer.score(str(actual), str(pred))
        for key in ROUGE_KEYS:
            totals[key] += scores[key].fmeasure

    return {key: totals[key] / n for key in ROUGE_KEYS}


def compute_meteor(predictions: Sequence[str], actuals: Sequence[str]) -> float:
    """Compute the mean METEOR score over prediction-reference pairs.

    Args:
        predictions: Generated headline strings.
        actuals: Ground-truth headline strings.

    Returns:
        Mean METEOR score in ``[0.0, 1.0]``; ``0.0`` for empty input.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    _check_lengths(predictions, actuals)
    if not predictions:
        return 0.0

    _ensure_wordnet()
    total = 0.0
    for pred, actual in zip(predictions, actuals, strict=True):
        total += meteor_score([str(actual).split()], str(pred).split())
    return float(total / len(predictions))


def compute_bertscore(
    predictions: Sequence[str],
    actuals: Sequence[str],
    model_type: str = "roberta-large",
    device: str | None = None,
) -> float:
    """Compute mean BERTScore F1 (opt-in semantic metric).

    Args:
        predictions: Generated headline strings.
        actuals: Ground-truth headline strings.
        model_type: Backbone passed to ``bert_score`` (downloaded on first use).
        device: Torch device string; defaults to bert_score's auto-detection.

    Returns:
        Mean BERTScore F1 in ``[0.0, 1.0]``; ``0.0`` for empty input.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
        ImportError: If the optional ``bert_score`` dependency is missing.
    """
    _check_lengths(predictions, actuals)
    if not predictions:
        return 0.0

    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "compute_bertscore requires the optional 'bert_score' dependency. "
            "Install it with: pip install '.[bertscore]'"
        ) from exc

    _, _, f1 = bert_score_fn(
        list(predictions),
        list(actuals),
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return float(f1.mean().item())


def compute_metrics(
    predictions: Sequence[str],
    actuals: Sequence[str],
    include_bertscore: bool = False,
) -> dict[str, float]:
    """Compute ROUGE, METEOR, mean generation length, and optionally BERTScore.

    Args:
        predictions: Generated headline strings.
        actuals: Ground-truth headline strings.
        include_bertscore: When ``True`` also compute BERTScore F1 (slow; pulls
            the optional ``bert_score`` dependency and downloads a model).

    Returns:
        Mapping with ROUGE-1/2/L, ``meteor``, ``gen_len`` (mean prediction
        length in whitespace tokens), and optionally ``bertscore_f1``.
    """
    metrics: dict[str, float] = {
        **compute_rouge(predictions, actuals),
        "meteor": compute_meteor(predictions, actuals),
        "gen_len": (
            sum(len(str(p).split()) for p in predictions) / len(predictions) if predictions else 0.0
        ),
    }
    if include_bertscore:
        metrics["bertscore_f1"] = compute_bertscore(predictions, actuals)
    return metrics


def save_predictions(
    predictions: Sequence[str],
    actuals: Sequence[str],
    path: str | Path = "predictions.csv",
) -> Path:
    """Save generated and reference texts side-by-side to a CSV file.

    Args:
        predictions: Generated text strings.
        actuals: Ground-truth text strings.
        path: Destination CSV path.

    Returns:
        The resolved ``Path`` written to.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    _check_lengths(predictions, actuals)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Generated Text": list(predictions), "Actual Text": list(actuals)}).to_csv(
        out_path, index=False
    )
    return out_path


def _check_lengths(predictions: Sequence[str], actuals: Sequence[str]) -> None:
    """Raise ``ValueError`` if the two sequences differ in length."""
    if len(predictions) != len(actuals):
        raise ValueError(
            f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
            f"must have the same length."
        )

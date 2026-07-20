"""Configuration for the BERT sentiment-classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BertConfig:
    """Hyperparameters for the BERT sentiment-classification pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
        data_path: Path (relative to repo root) to the input CSV.
        output_dir: Directory for saved metrics and artifacts.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        max_length: Token length for padding/truncation.
        batch_size: Mini-batch size.
        device: Preferred device string (resolved at runtime).
        holdout_size: Fraction of the dataset reserved for val + test.
        test_size_from_holdout: Fraction of the holdout used for the test set.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of total steps spent in LR warmup.
        patience: Early-stopping patience in epochs (monitors val loss).
        seed: RNG seed for reproducibility.
        num_classes: Number of sentiment classes.
        dropout: Dropout probability before the classification head.
        label_smoothing: Cross-entropy label smoothing to curb overfitting.
    """

    model_name: str = "bert-base-uncased"
    data_path: str = "data/guardian_headlines.csv"
    output_dir: str = "artifacts/bert/"
    epochs: int = 4
    learning_rate: float = 1e-5
    max_length: int = 80
    batch_size: int = 12
    device: str = "cuda"
    holdout_size: float = 0.50
    test_size_from_holdout: float = 0.20
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    seed: int = 42
    num_classes: int = 3
    dropout: float = 0.1
    label_smoothing: float = 0.1


CONFIG = BertConfig()

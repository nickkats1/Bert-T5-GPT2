"""Configuration for the T5 summarization pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class T5Config:
    """Hyperparameters for the T5 summarization pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
        data_path: Path (relative to repo root) to the input CSV.
        output_dir: Directory for saved model files and predictions.
        source_length: Max source token length.
        target_length: Max target token length.
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        device: Preferred device string (resolved at runtime).
        seed: RNG seed for reproducibility.
        test_size: Fraction of the dataset used for validation.
        generate_max_length: ``model.generate`` max length.
        generate_num_beams: ``model.generate`` beam-search width.
        repetition_penalty: ``model.generate`` repetition penalty.
        length_penalty: ``model.generate`` length penalty.
        source_prefix: Task prefix prepended to each source string.
        report_bertscore: Whether evaluation also computes BERTScore.
    """

    model_name: str = "t5-base"
    data_path: str = "data/reuters_headlines.csv"
    output_dir: str = "artifacts/t5/"
    source_length: int = 128
    target_length: int = 32
    batch_size: int = 12
    epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    device: str = "cuda:0"
    seed: int = 42
    test_size: float = 0.20
    generate_max_length: int = 128
    generate_num_beams: int = 2
    repetition_penalty: float = 2.5
    length_penalty: float = 1.0
    source_prefix: str = "summarize: "
    report_bertscore: bool = False


CONFIG = T5Config()

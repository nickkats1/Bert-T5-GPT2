"""Configuration for the GPT-2 fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Hyperparameters for the GPT-2 causal-language-modeling pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
        data_path: Path (relative to repo root) to the input CSV.
        output_dir: Directory for saved model files and metrics.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Mini-batch size.
        max_length: Token length for padding/truncation.
        accum_steps: Gradient-accumulation steps (effective batch multiplier).
        warmup_ratio: Fraction of total steps spent in LR warmup.
        max_grad_norm: Max gradient norm for clipping.
        use_amp: Enable automatic mixed precision on CUDA.
        device: Preferred device string (resolved at runtime).
        seed: RNG seed for reproducibility.
        test_size: Fraction of the dataset used for validation.
        pad_token: Pad token added to the GPT-2 tokenizer.
        bos_token: Beginning-of-sequence token added to the tokenizer.
        eos_token: End-of-sequence token added to the tokenizer.
    """

    model_name: str = "gpt2"
    data_path: str = "data/reuters_headlines.csv"
    output_dir: str = "artifacts/gpt2/"
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 12
    max_length: int = 128
    accum_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42
    test_size: float = 0.20
    patience: int = 2
    pad_token: str = "<|pad|>"
    bos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"


CONFIG = GPT2Config()

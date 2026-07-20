"""Shared utilities used across the BERT, GPT-2, and T5 pipelines.

This module centralizes cross-cutting concerns (reproducibility, device
resolution, logging, and artifact I/O) so the three model packages stay small
and focused on their task-specific logic.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured, non-propagating logger.

    Using the ``logging`` module (instead of bare ``print``) lets callers
    control verbosity, redirect output, and keep timestamps without touching
    the training code.

    Args:
        name: Logger name, conventionally the module ``__name__``.
        level: Minimum level to emit.

    Returns:
        A ``logging.Logger`` with a single stream handler attached.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU & CUDA) RNGs.

    Args:
        seed: Non-negative integer used to seed every RNG.
        deterministic: When ``True`` also request deterministic cuDNN/torch
            algorithms (``warn_only`` so unsupported ops fall back instead of
            raising). Combine with a seeded DataLoader ``generator`` (see
            :func:`make_generator`) for reproducible shuffling.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed for deterministic CUBLAS GEMMs; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker's NumPy/Python RNGs from its torch seed.

    Pass as ``worker_init_fn`` so multi-worker shuffling is reproducible.

    Args:
        worker_id: Worker index supplied by the DataLoader (unused; the base
            seed already differs per worker via ``torch.initial_seed()``).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded for reproducible shuffling.

    Args:
        seed: Seed for the generator.

    Returns:
        A seeded ``torch.Generator`` to pass as a DataLoader's ``generator``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable.

    Args:
        preferred: Optional preferred device string (``"cuda"``, ``"cuda:0"``,
            ``"cpu"``, ...). If ``None`` (default) CUDA is selected when
            available, otherwise CPU.

    Returns:
        ``torch.device`` instance suitable for ``.to(...)``. A requested CUDA
        device silently degrades to CPU when no GPU is present, so the same
        config runs unchanged on a laptop or a GPU box.
    """
    if preferred is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(preferred)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the parameters of a model.

    Args:
        model: Any ``torch.nn.Module``.
        trainable_only: If ``True`` count only parameters with
            ``requires_grad``; otherwise count all parameters.

    Returns:
        Total number of (optionally trainable) scalar parameters.
    """
    params = (
        (p for p in model.parameters() if p.requires_grad) if trainable_only else model.parameters()
    )
    return sum(p.numel() for p in params)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Serialize a metrics/config dict to a pretty-printed JSON file.

    Args:
        data: JSON-serializable mapping (e.g. a metrics dict).
        path: Destination file path; parent directories are created.

    Returns:
        The resolved ``Path`` written to.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


class EarlyStopping:
    """Track a monitored metric and signal when training should stop.

    Report each epoch's metric via :meth:`step`; it returns whether the value
    improved on the best seen so far (so the caller can checkpoint the best
    model). After ``patience`` consecutive non-improving epochs, ``should_stop``
    becomes ``True``.

    Attributes:
        best: Best metric value observed so far (``None`` before the first step).
        should_stop: ``True`` once patience is exhausted.
    """

    def __init__(self, patience: int = 2, mode: str = "min", min_delta: float = 0.0) -> None:
        """Initialize the early-stopping tracker.

        Args:
            patience: Consecutive non-improving epochs tolerated before stopping.
            mode: ``"min"`` (lower is better, e.g. loss) or ``"max"`` (higher is
                better, e.g. accuracy).
            min_delta: Minimum change over the best to count as an improvement.

        Raises:
            ValueError: If ``mode`` is not ``"min"`` or ``"max"``.
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}.")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """Record a metric value and report whether it improved.

        Args:
            value: The monitored metric for the just-finished epoch.

        Returns:
            ``True`` if ``value`` is a new best (caller should checkpoint),
            else ``False``.
        """
        if self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False

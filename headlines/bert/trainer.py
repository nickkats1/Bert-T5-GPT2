"""Training and evaluation loops for the BERT sentiment classifier.

Both loops share the same per-batch shape (``input_ids``, ``attention_mask``,
``targets``) and return ``(accuracy, mean_loss)`` so the caller can log and
early-stop on either quantity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train(
    model: nn.Module,
    loader,
    optimizer,
    device,
    *,
    loss_fn,
    n_examples: int,
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model: BERT classifier.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        loss_fn: Classification loss function (e.g., ``nn.CrossEntropyLoss``).
        n_examples: Total number of training examples (for accuracy).
        scheduler: Optional LR scheduler stepped per batch.
        max_grad_norm: Max gradient norm for clipping.

    Returns:
        Tuple ``(accuracy, mean_loss)``.
    """
    model.train()
    losses: list[float] = []
    correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct += int(torch.sum(preds == targets).item())
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    accuracy = correct / n_examples if n_examples else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return accuracy, mean_loss


def validate(
    model: nn.Module,
    loader,
    device,
    *,
    loss_fn,
    n_examples: int,
) -> tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Args:
        model: BERT classifier.
        loader: Validation or test DataLoader.
        device: Torch device.
        loss_fn: Classification loss function.
        n_examples: Total number of evaluation examples.

    Returns:
        Tuple ``(accuracy, mean_loss)``.
    """
    model.eval()
    losses: list[float] = []
    correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct += int(torch.sum(preds == targets).item())
            losses.append(loss.item())

    accuracy = correct / n_examples if n_examples else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return accuracy, mean_loss

"""Training and evaluation loops for GPT-2 causal-language-model fine-tuning.

Unlike a plain loop, ``train`` honours the knobs exposed by ``GPT2Config``:
mixed-precision (AMP), gradient accumulation, gradient clipping, and an
optional warmup scheduler. Pad tokens are masked out of the language-modeling
loss so padding never contributes to the gradient or the perplexity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from headlines.common import get_logger
from headlines.gpt2.metrics import perplexity

logger = get_logger(__name__)


def _labels_from(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a batch to ``device`` and build pad-masked LM labels."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # ignore padding in the LM loss
    return input_ids, attention_mask, labels


def train(
    model: nn.Module,
    loader,
    optimizer,
    device,
    *,
    epoch: int = 0,
    total_epochs: int = 1,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    accum_steps: int = 1,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model: ``GPT2LMHeadModel``.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        epoch: Current epoch index (0-based) for logging.
        total_epochs: Total number of epochs (for log formatting).
        scheduler: Optional LR scheduler stepped once per optimizer step.
        scaler: Optional shared ``GradScaler``; created (disabled off-CUDA) when
            not supplied. Pass one from the caller so its adaptive loss scale
            persists across epochs.
        accum_steps: Gradient-accumulation steps; the optimizer steps every
            ``accum_steps`` batches (effective batch = batch_size * accum_steps).
        use_amp: Enable automatic mixed precision (only active on CUDA).
        max_grad_norm: Max gradient norm for clipping.
        log_every: Log progress every N steps; 0 disables.

    Returns:
        Tuple ``(avg_loss, perplexity)``.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError("Training loader is empty.")

    accum_steps = max(1, accum_steps)
    amp_enabled = use_amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids, attention_mask, labels = _labels_from(batch, device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        total_loss += loss.item()

        # Average the gradient over the batches actually in this accumulation
        # group, so the trailing partial group is scaled correctly too.
        group_start = (i // accum_steps) * accum_steps
        group_size = min(accum_steps, n_batches - group_start)
        scaler.scale(loss / group_size).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        if log_every and i % log_every == 0:
            running = total_loss / (i + 1)
            logger.info(
                "Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f",
                epoch + 1,
                total_epochs,
                i,
                n_batches,
                running,
            )

    avg_loss = total_loss / n_batches
    ppl = perplexity(avg_loss)
    logger.info(
        "Epoch [%d/%d] | Avg Loss: %.4f | Perplexity: %.4f",
        epoch + 1,
        total_epochs,
        avg_loss,
        ppl,
    )
    return avg_loss, ppl


def validate(model: nn.Module, loader, device) -> tuple[float, float]:
    """Evaluate the model on a validation/test loader.

    Args:
        model: ``GPT2LMHeadModel``.
        loader: Validation or test DataLoader.
        device: Torch device.

    Returns:
        Tuple ``(avg_loss, perplexity)``.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError("Validation loader is empty.")

    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = _labels_from(batch, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.item()

    avg_eval_loss = eval_loss / n_batches
    ppl = perplexity(avg_eval_loss)
    logger.info("Val Loss: %.4f | Perplexity: %.4f", avg_eval_loss, ppl)
    return avg_eval_loss, ppl

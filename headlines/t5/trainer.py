"""Training and generation loops for T5 summarization fine-tuning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from headlines.common import get_logger
from headlines.t5.config import CONFIG

logger = get_logger(__name__)


def train(
    model: nn.Module,
    loader,
    optimizer,
    device,
    *,
    tokenizer,
    epoch: int = 0,
    log_every: int = 50,
) -> float:
    """Run one training epoch for the T5 model.

    Passes the target IDs as ``labels`` (pad tokens masked with ``-100`` so they
    are ignored by the loss) and lets T5 build the decoder inputs via its
    internal right-shift, keeping training consistent with generation.

    Args:
        model: ``T5ForConditionalGeneration``.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        tokenizer: T5Tokenizer (used for the pad token ID).
        epoch: Current epoch index (0-based) for logging.
        log_every: Log progress every N steps; 0 disables logging.

    Returns:
        Mean training loss for the epoch.

    Raises:
        ValueError: If ``loader`` is empty.
    """
    if len(loader) == 0:
        raise ValueError("Training loader is empty.")

    model.train()
    losses: list[float] = []

    for step, batch in enumerate(loader):
        # Pass ``labels`` only and let T5 build ``decoder_input_ids`` via its
        # internal right-shift (which prepends ``decoder_start_token_id``). This
        # keeps training consistent with ``model.generate`` at inference time.
        labels = batch["target_ids"].to(device, dtype=torch.long).clone()
        labels[labels == tokenizer.pad_token_id] = -100

        input_ids = batch["source_ids"].to(device, dtype=torch.long)
        attention_mask = batch["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        losses.append(loss.item())

        if log_every and step % log_every == 0:
            logger.info("Epoch: %d | Step: %d | Loss: %.4f", epoch, step, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(np.mean(losses))


def validate(
    model: nn.Module,
    loader,
    device,
    *,
    tokenizer,
    max_length: int = CONFIG.generate_max_length,
    num_beams: int = CONFIG.generate_num_beams,
    repetition_penalty: float = CONFIG.repetition_penalty,
    length_penalty: float = CONFIG.length_penalty,
    log_every: int = 10,
) -> tuple[list[str], list[str]]:
    """Generate predictions on a validation/test loader.

    Args:
        model: ``T5ForConditionalGeneration``.
        loader: Validation or test DataLoader.
        device: Torch device.
        tokenizer: T5Tokenizer for decoding.
        max_length: Generation max length.
        num_beams: Beam-search width.
        repetition_penalty: Generation repetition penalty.
        length_penalty: Generation length penalty.
        log_every: Log progress every N steps; 0 disables logging.

    Returns:
        Tuple of ``(predictions, actuals)`` as lists of decoded strings.
    """
    model.eval()
    predictions: list[str] = []
    actuals: list[str] = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["source_ids"].to(device, dtype=torch.long)
            attention_mask = batch["source_mask"].to(device, dtype=torch.long)
            target_ids = batch["target_ids"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

            if log_every and step % log_every == 0:
                logger.info("Validation step: %d", step)

            predictions.extend(preds)
            actuals.extend(targets)

    return predictions, actuals

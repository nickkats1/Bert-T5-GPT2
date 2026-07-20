"""Entry point for fine-tuning GPT-2 on Reuters descriptions.

Any config field can be overridden from the CLI, e.g.::

    python -m headlines.gpt2.run --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

from headlines.common import (
    EarlyStopping,
    count_parameters,
    get_logger,
    make_generator,
    resolve_device,
    save_json,
    seed_everything,
)
from headlines.gpt2.config import CONFIG, GPT2Config
from headlines.gpt2.metrics import generate_samples
from headlines.gpt2.trainer import train, validate
from headlines.gpt2.utils import build_dataloaders, load_data, split_data

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> GPT2Config:
    """Parse CLI overrides on top of the default ``GPT2Config``."""
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 on Reuters descriptions.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--max-length", type=int, default=CONFIG.max_length)
    p.add_argument("--accum-steps", type=int, default=CONFIG.accum_steps)
    p.add_argument("--device", default=CONFIG.device)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    args = p.parse_args(argv)
    return dataclasses.replace(
        CONFIG,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        accum_steps=args.accum_steps,
        device=args.device,
        seed=args.seed,
        use_amp=CONFIG.use_amp and not args.no_amp,
    )


def main(cfg: GPT2Config | None = None) -> dict[str, float]:
    """Run the GPT-2 fine-tuning pipeline end-to-end.

    Returns:
        A metrics dict (also written to ``output_dir/metrics.json``).
    """
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Device: %s | AMP: %s", device, cfg.use_amp and device.type == "cuda")

    description_df = load_data(file_path=cfg.data_path)
    train_description, val_description = split_data(
        description_df, test_size=cfg.test_size, random_state=cfg.seed
    )
    logger.info("Split -> train=%d val=%d", len(train_description), len(val_description))

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens(
        {"pad_token": cfg.pad_token, "bos_token": cfg.bos_token, "eos_token": cfg.eos_token}
    )
    # Right padding is correct for TRAINING a causal LM: GPT-2 derives position
    # ids from a plain arange, so left padding would shift every real token's
    # positional embedding. (Generation here uses a single unpadded prompt.)
    tokenizer.padding_side = "right"

    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    train_loader, val_loader = build_dataloaders(
        train_description,
        val_description,
        tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        generator=make_generator(cfg.seed),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # Optimizer steps per epoch = ceil(batches / accum_steps) — must match the
    # trainer, which also steps on the trailing partial group.
    steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.accum_steps))
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    # Validate every epoch so we can early-stop on val loss and keep the best
    # weights, rather than blindly returning the final (possibly overfit) model.
    early_stopping = EarlyStopping(patience=cfg.patience, mode="min")
    best_state = None
    train_loss = train_perplexity = 0.0
    val_loss = val_perplexity = 0.0
    for epoch in range(cfg.epochs):
        train_loss, train_perplexity = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=cfg.epochs,
            scheduler=scheduler,
            scaler=scaler,
            accum_steps=cfg.accum_steps,
            use_amp=cfg.use_amp,
            max_grad_norm=cfg.max_grad_norm,
        )
        val_loss, val_perplexity = validate(model, val_loader, device)
        if early_stopping.step(val_loss):
            best_state = copy.deepcopy(model.state_dict())
        if early_stopping.should_stop:
            logger.info(
                "Early stopping at epoch %d (best val loss %.4f)", epoch + 1, early_stopping.best
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    samples = generate_samples(model, tokenizer, device)
    for idx, sample in enumerate(samples, 1):
        logger.info("Sample %d: %s", idx, sample)

    metrics = {
        "train_loss": train_loss,
        "train_perplexity": train_perplexity,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
    }
    save_json(metrics, os.path.join(cfg.output_dir, "metrics.json"))
    logger.info("Saved metrics to %s", os.path.join(cfg.output_dir, "metrics.json"))
    return metrics


if __name__ == "__main__":
    main()

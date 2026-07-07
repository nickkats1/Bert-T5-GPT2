"""Entry point for fine-tuning T5 on Reuters Description -> Headlines.

Any config field can be overridden from the CLI, e.g.::

    python -m headlines.t5.run --epochs 1 --device cpu --model-name t5-small
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from headlines.common import (
    count_parameters,
    get_logger,
    make_generator,
    resolve_device,
    save_json,
    seed_everything,
)
from headlines.t5.config import CONFIG, T5Config
from headlines.t5.dataset import SummarizationDataset
from headlines.t5.metrics import compute_metrics, save_predictions
from headlines.t5.trainer import train, validate
from headlines.t5.utils import load_data

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> T5Config:
    """Parse CLI overrides on top of the default ``T5Config``."""
    p = argparse.ArgumentParser(description="Fine-tune T5 on description→headline summarization.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--source-length", type=int, default=CONFIG.source_length)
    p.add_argument("--target-length", type=int, default=CONFIG.target_length)
    p.add_argument("--device", default=CONFIG.device)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument(
        "--bertscore",
        action="store_true",
        help="Also compute BERTScore (requires the optional 'bert_score' extra).",
    )
    args = p.parse_args(argv)
    return dataclasses.replace(
        CONFIG,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        source_length=args.source_length,
        target_length=args.target_length,
        device=args.device,
        seed=args.seed,
        report_bertscore=CONFIG.report_bertscore or args.bertscore,
    )


def main(cfg: T5Config | None = None) -> dict[str, float]:
    """Run the T5 fine-tuning + evaluation pipeline end-to-end.

    Returns:
        The summarization metrics dict (also written to
        ``output_dir/metrics.json``).
    """
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Device: %s", device)

    df = load_data(file_path=cfg.data_path)
    df["Description"] = cfg.source_prefix + df["Description"].astype(str)

    df_train, df_val = train_test_split(df, test_size=cfg.test_size, random_state=cfg.seed)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    logger.info("Split -> train=%d val=%d", len(df_train), len(df_val))

    tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    train_set = SummarizationDataset(
        df_train,
        tokenizer,
        source_len=cfg.source_length,
        target_len=cfg.target_length,
        source_col="Description",
        target_col="Headlines",
    )
    val_set = SummarizationDataset(
        df_val,
        tokenizer,
        source_len=cfg.source_length,
        target_len=cfg.target_length,
        source_col="Description",
        target_col="Headlines",
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, generator=make_generator(cfg.seed)
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.epochs):
        train(model, train_loader, optimizer, device, tokenizer=tokenizer, epoch=epoch)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    predictions, actuals = validate(
        model,
        val_loader,
        device,
        tokenizer=tokenizer,
        max_length=cfg.generate_max_length,
        num_beams=cfg.generate_num_beams,
        repetition_penalty=cfg.repetition_penalty,
        length_penalty=cfg.length_penalty,
    )
    save_predictions(predictions, actuals, path=os.path.join(cfg.output_dir, "predictions.csv"))

    metrics = compute_metrics(predictions, actuals, include_bertscore=cfg.report_bertscore)
    logger.info("Summarization metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    save_json(metrics, os.path.join(cfg.output_dir, "metrics.json"))
    logger.info("Saved metrics to %s", os.path.join(cfg.output_dir, "metrics.json"))
    return metrics


if __name__ == "__main__":
    main()

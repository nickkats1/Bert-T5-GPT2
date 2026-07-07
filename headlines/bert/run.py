"""Entry point for fine-tuning BERT on Guardian headline sentiment.

Any config field can be overridden from the CLI, e.g.::

    python -m headlines.bert.run --epochs 1 --device cpu --batch-size 4
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os

import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from headlines.bert.config import CONFIG, BertConfig
from headlines.bert.dataset import create_data_loader
from headlines.bert.metrics import (
    classification_summary,
    compute_confusion_matrix,
    compute_metrics,
    get_predictions,
)
from headlines.bert.model import BertClassifier
from headlines.bert.trainer import train, validate
from headlines.bert.utils import (
    CLASS_NAMES,
    clean_data,
    label_encode_sentiments,
    load_data,
    polarity,
    sentiment,
    split_train_val_test,
)
from headlines.common import (
    EarlyStopping,
    count_parameters,
    get_logger,
    make_generator,
    resolve_device,
    save_json,
    seed_everything,
)

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> BertConfig:
    """Parse CLI overrides on top of the default ``BertConfig``."""
    p = argparse.ArgumentParser(description="Fine-tune BERT for sentiment classification.")
    p.add_argument("--model-name", default=CONFIG.model_name)
    p.add_argument("--data-path", default=CONFIG.data_path)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    p.add_argument("--max-length", type=int, default=CONFIG.max_length)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--device", default=CONFIG.device)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    args = p.parse_args(argv)
    return dataclasses.replace(
        CONFIG,
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


def main(cfg: BertConfig | None = None) -> dict[str, float]:
    """Run the BERT sentiment-classification pipeline end-to-end.

    Returns:
        The test-set metrics dict (also written to ``output_dir/metrics.json``).
    """
    cfg = cfg if cfg is not None else parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Device: %s", device)

    df = clean_data(load_data(file_path=cfg.data_path))
    df["polarity"] = df["Headlines"].apply(polarity)
    df["sentiment"] = df["polarity"].apply(sentiment)
    df = label_encode_sentiments(df)
    logger.info("Loaded %d labeled headlines", len(df))

    df_train, df_val, df_test = split_train_val_test(
        df, cfg.holdout_size, cfg.test_size_from_holdout, cfg.seed
    )
    logger.info("Split -> train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    train_loader = create_data_loader(
        df_train,
        tokenizer,
        cfg.max_length,
        cfg.batch_size,
        True,
        generator=make_generator(cfg.seed),
    )
    val_loader = create_data_loader(df_val, tokenizer, cfg.max_length, cfg.batch_size, False)
    test_loader = create_data_loader(df_test, tokenizer, cfg.max_length, cfg.batch_size, False)

    model = BertClassifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # Early-stop on validation accuracy and keep the best-performing weights.
    early_stopping = EarlyStopping(patience=cfg.patience, mode="max")
    best_state = None
    for epoch in range(cfg.epochs):
        train_acc, train_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            n_examples=len(df_train),
            scheduler=scheduler,
        )
        val_acc, val_loss = validate(
            model, val_loader, device, loss_fn=loss_fn, n_examples=len(df_val)
        )
        logger.info(
            "Epoch %d/%d | train acc %.4f loss %.4f | val acc %.4f loss %.4f",
            epoch + 1,
            cfg.epochs,
            train_acc,
            train_loss,
            val_acc,
            val_loss,
        )
        if early_stopping.step(val_acc):
            best_state = copy.deepcopy(model.state_dict())
        if early_stopping.should_stop:
            logger.info(
                "Early stopping at epoch %d (best val acc %.4f)", epoch + 1, early_stopping.best
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_pred, y_true = get_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
    cm = compute_confusion_matrix(
        y_true.tolist(), y_pred.tolist(), labels=list(range(len(CLASS_NAMES)))
    )
    report = classification_summary(y_true.tolist(), y_pred.tolist(), target_names=CLASS_NAMES)

    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(
        {**metrics, "confusion_matrix": cm.tolist()},
        os.path.join(cfg.output_dir, "metrics.json"),
    )
    logger.info("Saved metrics to %s", os.path.join(cfg.output_dir, "metrics.json"))
    return metrics


if __name__ == "__main__":
    main()

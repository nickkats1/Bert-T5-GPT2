from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from headlines.t5.trainer import train, validate


class _ToySeq2SeqDataset(Dataset):
    """Yields the {source_ids, source_mask, target_ids} shape the trainer expects."""

    def __init__(self, n: int = 6, src_len: int = 8, tgt_len: int = 5, vocab: int = 20):
        torch.manual_seed(0)
        # Targets start at 1 so none collide with the fake pad id (0).
        self.source_ids = torch.randint(1, vocab, (n, src_len))
        self.source_mask = torch.ones((n, src_len), dtype=torch.long)
        self.target_ids = torch.randint(1, vocab, (n, tgt_len))

    def __len__(self) -> int:
        return self.source_ids.size(0)

    def __getitem__(self, idx):
        return {
            "source_ids": self.source_ids[idx],
            "source_mask": self.source_mask[idx],
            "target_ids": self.target_ids[idx],
        }


class _ToySeq2Seq(nn.Module):
    """Minimal stand-in: differentiable ``.loss`` for train, scripted ``.generate``."""

    def __init__(self, vocab: int = 20, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.embed(input_ids))
        loss = logits.mean() if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return torch.zeros((input_ids.size(0), 3), dtype=torch.long)


class _FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["decoded"] * len(ids)


def _build():
    loader = DataLoader(_ToySeq2SeqDataset(), batch_size=3)
    model = _ToySeq2Seq()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    return model, loader, optimizer


class TestT5Trainer:
    def test_train_returns_float_loss(self):
        model, loader, optimizer = _build()
        loss = train(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            tokenizer=_FakeTokenizer(),
            log_every=0,
        )
        assert isinstance(loss, float)
        assert loss == loss  # not NaN

    def test_train_empty_loader_raises(self):
        empty = DataLoader(_ToySeq2SeqDataset(n=0), batch_size=3)
        model, _, optimizer = _build()
        with pytest.raises(ValueError):
            train(
                model,
                empty,
                optimizer,
                torch.device("cpu"),
                tokenizer=_FakeTokenizer(),
                log_every=0,
            )

    def test_validate_returns_aligned_predictions(self):
        model, loader, _ = _build()
        preds, actuals = validate(
            model,
            loader,
            torch.device("cpu"),
            tokenizer=_FakeTokenizer(),
            log_every=0,
        )
        assert len(preds) == len(actuals) == len(loader.dataset)
        assert all(isinstance(p, str) for p in preds)

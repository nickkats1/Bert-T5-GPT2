from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from headlines.gpt2.trainer import train, validate


class _ToyLMDataset(Dataset):
    """Yields the {input_ids, attention_mask} batch shape the trainer expects."""

    def __init__(self, n: int = 8, seq_len: int = 6, vocab: int = 50):
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, vocab, (n, seq_len))
        self.attention_mask = torch.ones((n, seq_len), dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}


class _ToyLM(nn.Module):
    """Minimal LM head returning an object with a ``.loss`` attribute."""

    def __init__(self, vocab: int = 50, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _build():
    loader = DataLoader(_ToyLMDataset(), batch_size=4)
    model = _ToyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    return model, loader, optimizer


class TestGPT2Trainer:
    def test_train_returns_loss_and_perplexity(self):
        model, loader, optimizer = _build()
        loss, ppl = train(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            accum_steps=2,
            use_amp=False,
        )
        assert isinstance(loss, float)
        assert ppl >= 1.0  # perplexity = exp(loss) >= 1 for non-negative loss

    def test_validate_runs(self):
        model, loader, _ = _build()
        loss, ppl = validate(model, loader, device=torch.device("cpu"))
        assert loss >= 0.0
        assert ppl >= 1.0

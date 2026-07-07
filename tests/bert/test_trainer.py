import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from headlines.bert.trainer import train, validate


class _ToyDataset(Dataset):
    """In-memory dataset producing the dict shape consumed by the trainer."""

    def __init__(self, n: int = 8, seq_len: int = 4, num_classes: int = 3):
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, 100, (n, seq_len))
        self.attention_mask = torch.ones((n, seq_len), dtype=torch.long)
        self.targets = torch.randint(0, num_classes, (n,))

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "targets": self.targets[idx],
        }


class _ToyModel(nn.Module):
    """Tiny stand-in for BertClassifier with the same call signature."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.embed = nn.Embedding(100, 8)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(input_ids).mean(dim=1))


class TestTrainer:
    """Smoke tests for the BERT trainer functions."""

    def _build(self):
        dataset = _ToyDataset()
        loader = DataLoader(dataset, batch_size=4)
        model = _ToyModel()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        return model, loader, loss_fn, optimizer, len(dataset)

    def test_train_returns_floats_in_range(self):
        model, loader, loss_fn, optimizer, n = self._build()

        accuracy, loss = train(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            loss_fn=loss_fn,
            n_examples=n,
        )

        assert isinstance(accuracy, float)
        assert isinstance(loss, float)
        assert 0.0 <= accuracy <= 1.0
        assert loss == loss  # not NaN

    def test_validate_runs_without_scheduler(self):
        model, loader, loss_fn, _, n = self._build()
        accuracy, loss = validate(
            model,
            loader,
            device=torch.device("cpu"),
            loss_fn=loss_fn,
            n_examples=n,
        )
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0

    def test_train_works_with_scheduler_none(self):
        model, loader, loss_fn, optimizer, n = self._build()
        train(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            loss_fn=loss_fn,
            n_examples=n,
            scheduler=None,
        )

"""BERT classifier with a dropout + linear head for sentiment classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from headlines.bert.config import CONFIG


class BertClassifier(nn.Module):
    """BERT with a dropout and linear classification head.

    Architecture:
        BERT encoder (hidden_size) -> Dropout(dropout) -> Linear(hidden_size, num_classes)

    The head width is read from the loaded encoder's ``config.hidden_size``, so
    swapping in ``bert-large-uncased`` (1024-dim) just works.
    """

    def __init__(
        self,
        model_name: str = CONFIG.model_name,
        num_classes: int = CONFIG.num_classes,
        dropout: float = CONFIG.dropout,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.out(self.drop(pooled_output))

"""Keeps the suite on CPU: pytest imports this before conftest.py pulls in torch."""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

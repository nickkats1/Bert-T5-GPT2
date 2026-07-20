# Headlines: BERT · T5 · GPT-2

Fine-tuning three transformer architectures on news-headline data — one task per
architecture, each in its own self-contained package with a runnable CLI.

| Model | Task | Data | Reported metrics |
| ----- | ---- | ---- | ---------------- |
| **BERT** | Sentiment classification (3 classes) | Guardian headlines | accuracy, macro/weighted precision·recall·F1, MCC, confusion matrix |
| **GPT-2** | Causal language modeling | Reuters descriptions | loss, perplexity, sample generations |
| **T5** | Summarization (Description → Headline) | Reuters headlines + descriptions | ROUGE-1/2/L, METEOR, gen-length, optional BERTScore |

## Project layout

```text
bert-t5-gpt2/
├── data/                     # Raw CSV inputs (Guardian + Reuters)
├── src/
│   └── headlines/            # Importable package (src layout)
│       ├── common.py         # seed_everything, resolve_device, logging, JSON I/O
│       ├── bert/             # BERT sentiment classifier
│       ├── gpt2/             # GPT-2 fine-tuning
│       └── t5/               # T5 summarization
├── tests/                    # Pytest suite
├── artifacts/                # Saved models / predictions / metrics (git-ignored)
├── pyproject.toml            # Packaging, pytest, and ruff config
├── requirements.txt
└── .github/workflows/ci.yml  # ruff + pytest on Python 3.11–3.13
```

Every model package follows the same shape:

```text
src/headlines/<model>/
├── config.py     # Frozen @dataclass of hyperparameters (CONFIG)
├── utils.py      # Data loading / cleaning / splitting helpers
├── dataset.py    # PyTorch Dataset
├── model.py      # (BERT only — GPT-2/T5 use HuggingFace heads directly)
├── trainer.py    # train() / validate() loops
├── metrics.py    # Task-appropriate evaluation
└── run.py        # End-to-end entry point with an argparse CLI
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes the src-layout `headlines` package importable
# dev extras (pytest, ruff):  pip install -e ".[dev]"

# One-time NLTK data for the T5 METEOR metric:
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The CSV inputs are tracked under `data/`; no external download is required.

## Quick start

Run any pipeline from the repo root:

```bash
python -m headlines.bert.run     # BERT sentiment classifier
python -m headlines.gpt2.run     # GPT-2 fine-tuning
python -m headlines.t5.run       # T5 summarization
```

Or via the installed entry points (`bert-run`, `gpt2-run`, `t5-run`).

The common hyperparameters (model, epochs, learning rate, batch size, device,
seed, …) are exposed as CLI flags, so most experiments need no file edits — run
any pipeline with `--help` for its full list:

```bash
python -m headlines.bert.run --epochs 1 --device cpu --batch-size 4
python -m headlines.t5.run   --model-name t5-small --epochs 1 --bertscore
python -m headlines.gpt2.run --epochs 1 --no-amp
```

Pipelines auto-detect CUDA via `headlines.common.resolve_device` and fall back
to CPU when no GPU is present. Artifacts (saved model, predictions,
`metrics.json`) land under `artifacts/<model>/`.

## What each pipeline does

- **BERT** derives weak sentiment labels from TextBlob polarity, stratifies a
  train/val/test split, fine-tunes `bert-base-uncased` with a dropout +
  linear head, and reports a full classification report on the held-out test
  set. Note: because the labels come from TextBlob, the reported scores measure
  how well BERT reproduces TextBlob's rule — a distillation/weak-supervision
  setup, not gold-standard sentiment.
- **GPT-2** fine-tunes with **mixed precision (AMP)**, **gradient accumulation**,
  gradient clipping, and a **linear warmup** schedule — every knob in
  `GPT2Config` is actually honored by the trainer — then samples a few
  generations for a qualitative sanity check.
- **T5** frames summarization as `Description → Headline`, trains with
  teacher forcing (pad tokens masked from the loss), and evaluates with ROUGE,
  METEOR, and an optional semantic BERTScore.

## Testing & linting

```bash
pytest                          # full suite
pytest tests/bert -v            # one package
ruff check src/ tests/          # lint
ruff format src/ tests/         # auto-format
```

CI runs the same checks on every push / PR (`.github/workflows/ci.yml`).

## License

[MIT](LICENSE).

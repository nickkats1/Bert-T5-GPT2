# Headlines: Transformers

This code base trains three popular pre-trained transformers: Bert(encoder-only), T5(seq-to-seq) and gpt2(text-generation) on a dataset consisting of article headlines.

## Overview

The purpose of this repository is to fully fine tune the three parts of the transformer based on popular pretrained models for different tasks and evaluation metrics

### Tasks

- **Text Classification**: Bert, Encoder-only part of transformer
- **Text Summarization**: T5, encoder-decoder part of the transformer
- **Text Generation**: GPT2: Decoder-Only, autoregressive part of the transformer.

## Getting Started

```bash
git clone git@github.com:nickkats1/Bert-T5-GPT2.git
cd Bert-T5-GPT2

pip install -r requirements.txt
pip install -e .
```

**Note**: make sure you have a gpu or can run this repo where you can rent a gpu.

## Running

Each model is its own package under `headlines/`, carrying its own `config.py`, `data.py`,
`train.py`, `eval.py`, and `predictions.py`. Installing the package puts three commands on your
path:

```bash
train-bert     # BERT sentiment classification of Guardian headlines
train-t5       # T5 summarization of Reuters descriptions into headlines
train-gpt2     # GPT-2 causal language modelling on Reuters descriptions
```

Training settings come from `transformers.TrainingArguments`, filled in from a frozen dataclass per
model. To change a run, edit that dataclass — `headlines/t5/config.py` for T5, and so on.

Each model also has a `predictions.py` that reloads the saved checkpoint and scores the held-out
test split:

```bash
python -m headlines.bert.predictions
```

`examples/bert.ipynb` is the notebook that matches this code. The other four
(`train_bert.ipynb`, `train_t5.ipynb`, `train_gpt2.ipynb`, `save_and_score.ipynb`) target an older
API and do not currently run.

### Development

```bash
make test        # full suite
make test-fast   # skips tests marked integration
make quality     # ruff check + format --check
make style       # apply fixes and formatting
```

The suite builds its models from tiny randomly-initialised checkpoints on CPU, so it runs in
seconds and never downloads or trains a real model. No test is currently marked `integration`, so
`make test-fast` runs the same set as `make test`.

## Background

The Transformer has three different parts: the encoder, the encoder-decoder (with cross-attention) and the decoder-only part.

The decoder-only part is what most LLM's are train/based on.

### Bert (Bi-directional)

Every token attends to every other token in both directions, so the model builds one representation
of the whole sentence. That is the right shape for classification: a head reads a single label off
that representation. It is not a generator — it has no notion of continuing text. This repo uses
`bert-base-uncased`, which needs `transformers>=4.48`.

### T5 (encoder-decoder)

The encoder reads the input, and a separate decoder writes the output while attending back to the
encoder through cross-attention. Input and output are both text and do not have to be the same
length, which is what summarization needs: a long description in, a short headline out.

### GPT-2 (decoder-only)

Each token attends only to the tokens before it, and the model is trained to predict the next one.
That is why it generates text, and why its score is perplexity rather than accuracy or ROUGE —
there is no single right answer to compare against, only how surprised the model is by real text.

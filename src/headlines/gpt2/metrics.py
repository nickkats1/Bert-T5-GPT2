"""Evaluation metrics and qualitative sampling for the GPT-2 pipeline."""

from __future__ import annotations

import math

import torch


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss into perplexity.

    Args:
        loss: Mean per-token cross-entropy loss (natural log).

    Returns:
        ``exp(loss)``, or ``inf`` on overflow.
    """
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    device,
    num_samples: int = 3,
    max_new_tokens: int = 40,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> list[str]:
    """Generate a few unconditioned samples for a qualitative sanity check.

    Sampling from the fine-tuned model (seeded only with the BOS token) is a
    cheap way to eyeball whether training produced fluent, on-domain text —
    perplexity alone can hide degenerate repetition.

    Args:
        model: Fine-tuned ``GPT2LMHeadModel``.
        tokenizer: Matching tokenizer with ``bos_token``/``eos_token`` set.
        device: Torch device.
        num_samples: Number of sequences to generate.
        max_new_tokens: Max tokens to generate per sample.
        top_k: Top-k sampling cutoff.
        top_p: Nucleus (top-p) sampling cutoff.
        temperature: Sampling temperature.

    Returns:
        A list of decoded strings (special tokens stripped).
    """
    model.eval()
    prompt = tokenizer(tokenizer.bos_token, return_tensors="pt").to(device)

    generated = model.generate(
        **prompt,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in generated]

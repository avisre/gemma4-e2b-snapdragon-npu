"""WikiText-2 calibration data loader — mirrors qai_hub_models LLM recipes.

Per Agent 28's verification of `qai_hub_models/datasets/wikitext.py`:
- Dataset: wikitext-2-raw-v1, TRAIN split, joined with BOS separator
- Default samples: ceil(80000 / context_length) = 20 for ctx 4096
- Per-sample length: context_length tokens (4096)
- NO chat template wrapping — plain prose calibrates better
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
from datasets import load_dataset


def get_wikitext_calibration(
    tokenizer,
    context_length: int = 4096,
    num_samples: int | None = None,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Return a list of `input_ids` tensors of shape (1, context_length).

    Args:
        tokenizer: HF tokenizer (must have .bos_token_id).
        context_length: chunk size for calibration prompts.
        num_samples: total chunks; defaults to ceil(80000/context_length).
        seed: ignored (qai_hub_models is deterministic).

    Returns:
        List of long-tensor calibration inputs ready for forward.
    """
    if num_samples is None:
        num_samples = math.ceil(80000 / context_length)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty rows (wikitext has many)
    texts = [row["text"] for row in ds if row["text"].strip()]
    # Join with BOS separator (qai_hub_models pattern)
    bos = tokenizer.bos_token or "<bos>"
    joined = bos.join(texts)

    tokens = tokenizer(
        joined, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    samples = []
    for i in range(num_samples):
        chunk = tokens[i * context_length : (i + 1) * context_length]
        if chunk.size(0) < context_length:
            break
        samples.append(chunk.unsqueeze(0))  # (1, context_length)

    return samples


def calibration_data_reader(samples: list[torch.Tensor]) -> Iterator[dict[str, torch.Tensor]]:
    """ONNX Runtime CalibrationDataReader interface."""
    for s in samples:
        yield {"input_ids": s.numpy()}

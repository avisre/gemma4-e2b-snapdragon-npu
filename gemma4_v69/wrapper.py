"""Gemma 4 E2B text decoder wrapper for QNN export.

Strategy (consolidated from 32-agent swarm research):
1. Reuse the wrapper from ExecuTorch PR #19166 at
   `executorch/examples/models/gemma4/text_decoder/gemma4_model.py`.
   It's already a clean, backend-agnostic torch.export-compatible module.
2. PR #19166 wires XNNPACK. We re-wire for QNN partitioner downstream.
3. Per-Layer Embeddings (PLE) stay as plain `nn.Embedding` + `nn.Linear`
   — fully decomposable to QNN-native ops (per Agent 02).
4. KV cache: 35 layers total, 15 unique cache pairs (layers 15-34 share KV
   from layers 13/14 — per Agent 06). We hand-wire as explicit tensor IO.
5. RoPE: precomputed cos/sin tables per layer-type (sliding θ=10k vs
   global θ=1M with partial_rotary_factor=0.25 NoPE tail) — per Agent 08.

This module is intentionally thin: it imports the PR #19166 wrapper as the
source of truth and exposes our preferred forward signature for QNN export.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Path injection: ExecuTorch lives next to this package after `setup.sh`
_HERE = Path(__file__).resolve().parent.parent
_EXECUTORCH = _HERE / "executorch"
if _EXECUTORCH.exists():
    sys.path.insert(0, str(_EXECUTORCH))

# These imports require setup.sh to have cloned ExecuTorch
try:
    from examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model as _ETGemma4
    from examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
    from examples.models.gemma4.text_decoder.convert_weights import convert_hf_to_executorch
    _ET_AVAILABLE = True
except ImportError as e:
    _ET_AVAILABLE = False
    _IMPORT_ERR = e


def load_gemma4_e2b_for_qnn(
    hf_checkpoint_dir: str | Path,
    max_seq_len: int = 2048,
    prefill_ar_len: int = 128,
    dtype: torch.dtype = torch.float16,
) -> "Gemma4ForQNN":
    """Load Gemma 4 E2B from HF safetensors, return QNN-ready wrapper."""
    if not _ET_AVAILABLE:
        raise RuntimeError(
            f"ExecuTorch not on path. Run ./setup.sh first.\nOriginal import error: {_IMPORT_ERR}"
        )

    cfg = Gemma4Config.from_e2b()  # ships with the PR #19166 wrapper
    cfg.max_seq_len = max_seq_len
    cfg.prefill_ar_len = prefill_ar_len

    state_dict = convert_hf_to_executorch(Path(hf_checkpoint_dir))
    base = _ETGemma4(cfg)
    base.load_state_dict(state_dict, strict=False)
    base.to(dtype=dtype).eval()

    return Gemma4ForQNN(base, cfg)


class Gemma4ForQNN(nn.Module):
    """Forward signature that matches the AI Hub LLMBase contract.

    Per Agent 23, the AI Hub Llama-3.2-1B template uses:
        forward(input_tokens, attention_mask, *args) -> [logits, *kv_out]

    where `*args` is `[position_ids_cos, position_ids_sin, past_k_0_in, past_v_0_in, ...]`.
    We replicate that exactly so the existing `qnn_llama_runner` consumes the output.

    Per Agent 06, Gemma 4 E2B has 35 layers but only 15 unique KV pairs (last 20
    layers share KV from layers 13/14). The KV IO matches those 15.
    """

    def __init__(self, base: nn.Module, cfg: "Gemma4Config"):
        super().__init__()
        self.base = base
        self.cfg = cfg
        # Layer indices that own their own KV (per Agent 06)
        self.kv_owning_layers = list(range(15))  # 0..14
        # Sliding-window cache layers: 0-3, 5-8, 10-13 (12 layers, 256 head_dim, 511 cache len)
        # Full-attn cache layers:      4, 9, 14    (3 layers, 512 head_dim, max_seq_len)
        self.sliding_idx = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
        self.full_idx = [4, 9, 14]

    def forward(
        self,
        input_ids: torch.Tensor,                # [1, S]
        attention_mask: torch.Tensor,           # [1, 1, S, S+kv]
        *kv_state: torch.Tensor,                # past_k_sliding_*, past_v_sliding_*, past_k_full_*, past_v_full_*
    ) -> list[torch.Tensor]:
        # Reshape kv_state into the dict the base model expects
        # Order from AI Hub convention: (cos, sin, past_k_0, past_v_0, past_k_1, past_v_1, ...)
        # For Gemma 4 with dual RoPE, we have cos_sliding, sin_sliding, cos_global, sin_global
        # Then 12 (k,v) sliding pairs + 3 (k,v) full pairs = 30 KV tensors
        cos_sliding, sin_sliding, cos_global, sin_global = kv_state[:4]
        kv_pairs = kv_state[4:]
        assert len(kv_pairs) == 30, f"expected 30 KV tensors, got {len(kv_pairs)}"

        past_k_sliding = list(kv_pairs[0:12])
        past_v_sliding = list(kv_pairs[12:24])
        past_k_full = list(kv_pairs[24:27])
        past_v_full = list(kv_pairs[27:30])

        # Call base model
        logits, present_k_sliding, present_v_sliding, present_k_full, present_v_full = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cos_sliding=cos_sliding,
            sin_sliding=sin_sliding,
            cos_global=cos_global,
            sin_global=sin_global,
            past_k_sliding=past_k_sliding,
            past_v_sliding=past_v_sliding,
            past_k_full=past_k_full,
            past_v_full=past_v_full,
        )

        out: list[torch.Tensor] = [logits]
        out.extend(present_k_sliding)
        out.extend(present_v_sliding)
        out.extend(present_k_full)
        out.extend(present_v_full)
        return out

    def get_static_input_specs(self, sequence_length: int) -> dict[str, tuple]:
        """Return AI Hub `input_specs` dict (shape + dtype per input).

        sequence_length=128 -> prefill graph
        sequence_length=1   -> decode graph
        """
        S = sequence_length
        max_ctx = self.cfg.max_seq_len  # 2048

        sliding_head_dim = 256
        full_head_dim = 512
        sliding_kv_len = 511   # sliding_window - 1
        full_kv_len = max_ctx  # full context

        # RoPE half-dims
        sliding_embed = sliding_head_dim // 2
        full_embed = full_head_dim // 2

        spec: dict[str, tuple] = {
            "input_ids": ((1, S), "int32"),
            "attention_mask": ((1, 1, S, max_ctx), "float32"),
            "position_ids_cos_sliding": ((1, 1, S, sliding_embed), "float32"),
            "position_ids_sin_sliding": ((1, 1, S, sliding_embed), "float32"),
            "position_ids_cos_global": ((1, 1, S, full_embed), "float32"),
            "position_ids_sin_global": ((1, 1, S, full_embed), "float32"),
        }

        # Sliding KV (12 layers)
        for layer_idx in self.sliding_idx:
            spec[f"past_k_sliding_{layer_idx}_in"] = (
                (1, 1, sliding_head_dim, sliding_kv_len), "float32"
            )
        for layer_idx in self.sliding_idx:
            spec[f"past_v_sliding_{layer_idx}_in"] = (
                (1, 1, sliding_kv_len, sliding_head_dim), "float32"
            )
        # Full KV (3 layers)
        for layer_idx in self.full_idx:
            spec[f"past_k_full_{layer_idx}_in"] = (
                (1, 1, full_head_dim, full_kv_len), "float32"
            )
        for layer_idx in self.full_idx:
            spec[f"past_v_full_{layer_idx}_in"] = (
                (1, 1, full_kv_len, full_head_dim), "float32"
            )
        return spec

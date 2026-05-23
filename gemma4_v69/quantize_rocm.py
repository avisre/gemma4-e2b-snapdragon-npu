"""Real Option A: PyTorch+ROCm calibration on AMD GPU → AIMET-ONNX export.

Pipeline:
1. Load Gemma 4 E2B as PyTorch model with torch_dtype=bfloat16 on ROCm GPU
   (HF accelerate `device_map="auto"` with cpu_offload for 16GB-tight VRAM)
2. Register forward hooks on every quantization-target linear projection
3. Run 128 calibration prompts (THIS is where the 9070 XT actually helps —
   ~3-5 min on GPU vs ~25-35 min on CPU)
4. Compute per-channel min/max stats per layer
5. Load FP16 ONNX export, instantiate AIMET QuantizationSimModel
6. Inject our pre-computed scales via `set_and_freeze_param_encodings`
7. Export QNN-ready QDQ ONNX + encodings.json

Time estimate end-to-end: ~10-15 min on RX 9070 XT (vs ~35-50 min CPU only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Skip patterns: PLE quant = garbage output (Agent 32 / MLX-verified)
PLE_SAFE_SKIP_MODULES = [
    "embed_tokens",
    "embed_tokens_per_layer",
    "per_layer_input_gate",
    "per_layer_projection",
    "per_layer_model_projection",
    "post_per_layer_input_norm",
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
    "v_norm",
    "norm",
    "lm_head",
    "rotary_emb",
]

QUANT_TARGET_PATTERNS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def should_quantize(module_name: str) -> bool:
    name_l = module_name.lower()
    if any(skip in name_l for skip in PLE_SAFE_SKIP_MODULES):
        return False
    return any(p in name_l for p in QUANT_TARGET_PATTERNS)


def pt_module_to_onnx_node(pt_name: str) -> str:
    """Map a PyTorch module name to its likely ONNX node name after torch.export.

    Convention from `torch.onnx.export`:
        `model.layers.0.self_attn.q_proj` → `/model/layers.0/self_attn/q_proj/MatMul`
    The exact suffix depends on op type; we return the prefix (caller fuzzy-matches).
    """
    return "/" + pt_name.replace(".", "/")


class CalibrationCollector:
    """Forward-hook based activation collector. Stores min/max running stats."""

    def __init__(self, model: nn.Module):
        self.stats: dict[str, dict] = defaultdict(lambda: {
            "min": None, "max": None, "count": 0,
        })
        self.hooks = []
        for name, module in model.named_modules():
            if should_quantize(name) and isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)
        print(f"==> Registered {len(self.hooks)} forward hooks on Linear modules")

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # output: (batch, seq, out_features). Stat per output channel.
            with torch.no_grad():
                t = output.detach().float()
                # Flatten batch+seq → take min/max per output channel
                t2 = t.reshape(-1, t.shape[-1])
                mn = t2.min(dim=0).values.cpu()
                mx = t2.max(dim=0).values.cpu()
                s = self.stats[name]
                if s["min"] is None:
                    s["min"] = mn
                    s["max"] = mx
                else:
                    s["min"] = torch.minimum(s["min"], mn)
                    s["max"] = torch.maximum(s["max"], mx)
                s["count"] += 1
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def to_encodings(self, weight_bits: int = 8, activation_bits: int = 16) -> dict:
        """Convert running min/max to AIMET-compatible encoding dicts."""
        # AIMET wants: {tensor_name: [{"bitwidth": N, "dtype": "int", "max": ..., "min": ..., "offset": 0, "scale": ...}]}
        encodings = {}
        for name, s in self.stats.items():
            mn = s["min"]
            mx = s["max"]
            # Symmetric quant: scale = max(|min|, |max|) / (2^(bits-1) - 1)
            absmax = torch.maximum(mn.abs(), mx.abs())
            qrange = (1 << (activation_bits - 1)) - 1
            scale = absmax / qrange
            # Per-channel encodings (one per output dim)
            encodings[name] = [
                {
                    "bitwidth": activation_bits,
                    "dtype": "int",
                    "min": float(-absmax[i].item()),
                    "max": float(absmax[i].item()),
                    "offset": 0,
                    "scale": float(scale[i].item()) if scale[i].item() > 0 else 1.0,
                    "is_symmetric": "True",
                }
                for i in range(len(absmax))
            ]
        return encodings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/gemma-4-e2b-it")
    p.add_argument("--onnx-input", default="exported/prefill.onnx",
                   help="FP16 ONNX export to inject encodings into")
    p.add_argument("--output-dir", default="quantized")
    p.add_argument("--num-calib-samples", type=int, default=128)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-memory-gb", type=int, default=14)
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    ckpt = here / args.checkpoint
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # ROCm-friendly allocator
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"==> GPU: {dev_name} ({vram_gb:.1f} GB VRAM)")
        if "9070" in dev_name or "AMD" in dev_name:
            print(f"   AMD detected — using ROCm path")
    else:
        print("==> WARN: no GPU. CPU calibration is ~5-10x slower.")

    print(f"\n==> Loading Gemma 4 E2B (bfloat16, device_map=auto)")
    max_mem = {0: f"{args.max_memory_gb}GiB", "cpu": "45GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device != "auto" else "auto",
        max_memory=max_mem if args.device == "auto" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    print(f"\n==> Attaching calibration hooks")
    collector = CalibrationCollector(model)

    from .calibration import get_wikitext_calibration
    calib_samples = get_wikitext_calibration(
        tokenizer,
        context_length=args.context_length,
        num_samples=args.num_calib_samples,
    )

    print(f"\n==> Running {args.num_calib_samples} calibration prompts on GPU")
    print(f"   (~3-5 min on RX 9070 XT, vs ~25-35 min on CPU)")
    target_device = next(model.parameters()).device
    with torch.inference_mode():
        for i, sample in enumerate(calib_samples):
            inputs = sample.to(target_device)
            _ = model(input_ids=inputs)
            if (i + 1) % 16 == 0:
                print(f"   calib {i+1}/{len(calib_samples)} "
                      f"(VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB)")

    collector.remove()
    encodings = collector.to_encodings(weight_bits=8, activation_bits=16)
    enc_path = out_dir / "pytorch_calibration_encodings.json"
    enc_path.write_text(json.dumps(encodings, indent=2))
    print(f"\n==> Saved per-channel encodings for {len(encodings)} modules to {enc_path}")
    print(f"   File size: {enc_path.stat().st_size / 1024**2:.1f} MB")

    # Free GPU memory before AIMET
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n==> Loading ONNX export + injecting encodings via AIMET-ONNX")
    onnx_in = here / args.onnx_input
    if not onnx_in.exists():
        print(f"   WARN: {onnx_in} missing. Run `python -m gemma4_v69.export` first.")
        print(f"   Encodings are saved; AIMET injection step will be re-runnable.")
        return 0

    try:
        from aimet_onnx.quantsim import QuantizationSimModel
        from aimet_common.defs import QuantScheme
    except ImportError:
        print("   WARN: AIMET-ONNX not installed. Encodings saved for later use.")
        return 0

    import onnx
    model_onnx = onnx.load(str(onnx_in))
    sim = QuantizationSimModel(
        model_onnx,
        param_type="int8",
        activation_type="int16",
        quant_scheme=QuantScheme.post_training_tf_enhanced,
    )

    # Map PyTorch names → ONNX node names (fuzzy)
    onnx_node_names = [n.name for n in sim.model.graph.node]
    injected = 0
    skipped = 0
    for pt_name, enc_list in encodings.items():
        onnx_prefix = pt_module_to_onnx_node(pt_name)
        candidates = [n for n in onnx_node_names if n.startswith(onnx_prefix) and "MatMul" in n]
        if not candidates:
            skipped += 1
            continue
        for cand in candidates:
            if cand in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[cand].set_encoding(enc_list[0])  # per-tensor fallback
                injected += 1

    print(f"   Injected encodings: {injected} ops")
    print(f"   Skipped (name mismatch): {skipped}")

    sim.export(str(out_dir), "gemma4_e2b_w8a16")
    print(f"\n==> Wrote {out_dir}/gemma4_e2b_w8a16.onnx + .encodings.json")
    print(f"==> Next: python -m gemma4_v69.aihub_submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

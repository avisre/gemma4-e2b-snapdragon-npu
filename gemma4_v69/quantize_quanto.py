"""Gemma 4 E2B → w8a16 quantization via Optimum-Quanto on AMD ROCm.

Per Agent 4 + user's pick (Option B), use HuggingFace Optimum-Quanto with
PyTorch's ROCm backend. Optimum-Quanto:
- Native PyTorch quantization (uses CUDA / ROCm kernels for fast calibration)
- Supports `weights=qint8, activations=qint16` for w8a16
- Outputs torch state — we convert to QNN-compatible ONNX after

This path requires:
1. PyTorch 2.9+ with ROCm 7.0.2+ (per Agent 2)
2. RX 9070 XT or other RDNA4 GPU
3. 16 GB VRAM (tight for E2B — uses HF `device_map="auto"` with cpu_offload)

Wall-clock estimate: ~5-10 min calibration on 9070 XT vs ~25-35 min CPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Skip patterns for PLE-sensitive modules (per Agent 32)
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


def should_quantize(module_name: str) -> bool:
    """Predicate: True if this module should be quantized (False = keep fp16)."""
    name_l = module_name.lower()
    for skip in PLE_SAFE_SKIP_MODULES:
        if skip in name_l:
            return False
    # Only quantize linear projections (q/k/v/o, gate/up/down)
    return any(p in name_l for p in ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/gemma-4-e2b-it")
    p.add_argument("--output-dir", default="quantized")
    p.add_argument("--num-calib-samples", type=int, default=128)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cpu. ROCm aliases as 'cuda'")
    p.add_argument("--max-memory-gb", type=int, default=14,
                   help="Reserve ~2GB headroom on 16GB VRAM")
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    ckpt = here / args.checkpoint
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # Set ROCm-friendly env BEFORE torch import (already imported but no harm)
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    # Verify GPU available
    if torch.cuda.is_available():
        print(f"==> GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        print("==> WARN: no GPU detected. Falling back to CPU (will be slow).")

    # Import optimum-quanto lazily
    try:
        from optimum.quanto import quantize, freeze, qint8, qint16, Calibration
    except ImportError as e:
        raise RuntimeError(
            "Optimum-Quanto not installed. Install:\n"
            "  pip install optimum-quanto\n"
            "Then verify torch CUDA (ROCm aliased) works:\n"
            "  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"\n"
            f"Original error: {e}"
        )

    from .calibration import get_wikitext_calibration

    print(f"==> Loading Gemma 4 E2B (bfloat16) with device_map=auto")
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

    # Identify modules to skip (PLE-sensitive)
    skip_modules = []
    for name, _module in model.named_modules():
        if not should_quantize(name) and name != "":
            skip_modules.append(name)
    print(f"==> Skipping quantization on {len(skip_modules)} modules "
          f"(PLE/embeddings/norms — fp16)")
    print(f"   Examples: {skip_modules[:5]}")

    print(f"\n==> Applying Optimum-Quanto w8a16 (weights=qint8, activations=qint16)")
    quantize(
        model,
        weights=qint8,
        activations=qint16,
        exclude=skip_modules,
    )

    print(f"\n==> Calibration ({args.num_calib_samples} samples × {args.context_length} tokens)")
    calib_samples = get_wikitext_calibration(
        tokenizer,
        context_length=args.context_length,
        num_samples=args.num_calib_samples,
    )

    with Calibration(streamline=False, debug=False):
        with torch.inference_mode():
            for i, sample in enumerate(calib_samples):
                inputs = sample.to(model.device if hasattr(model, "device") else "cuda")
                _ = model(input_ids=inputs)
                if (i + 1) % 16 == 0:
                    print(f"   calib {i+1}/{len(calib_samples)}")

    print("\n==> Freezing quantization scales (calibration done)")
    freeze(model)

    out_pt = out_dir / "gemma4_e2b_w8a16.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "quantization": {"weights": "qint8", "activations": "qint16"},
        "skip_modules": skip_modules,
    }, out_pt)
    print(f"   wrote {out_pt} ({out_pt.stat().st_size / 1024**3:.2f} GB)")

    # Export to ONNX for AI Hub (next pipeline step)
    print(f"\n==> Exporting to ONNX (for AI Hub submission)")
    print(f"   Note: Optimum-Quanto → ONNX requires `optimum-cli export onnx` or")
    print(f"   manual torch.onnx.export with quanto QTensor support.")
    print(f"   This is a known rough edge — see https://github.com/huggingface/optimum-quanto/issues")
    print(f"   For now, use Path A (PyTorch GPU calibration → AIMET ONNX export) as fallback.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

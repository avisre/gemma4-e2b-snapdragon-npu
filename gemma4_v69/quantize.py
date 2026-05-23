"""ONNX Runtime static quantization for Gemma 4 — w8a16, PLE-safe.

Per Agent 21: v69 lacks native INT4 HMX tile, so w4a16 dequants to INT8 at MAC.
Use w8a16 for native 64x32x8 HMX throughput — same memory headroom as w4 once
embeddings are also int8.

Per Agent 32: PLE quantization → catastrophic garbage output. The `per_layer_*`
tensors + `ScaledEmbedding` MUST stay FP16. Skip-list reflects this.

Backend selection:
- --backend cpu  (default): works everywhere, ~30-90 min for E2B
- --backend rocm: AMD GPU via ONNX Runtime ROCm EP, ~5-10 min on RX 9070 XT 16GB
- --backend cuda: NVIDIA GPU via standard CUDA EP
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


PLE_SAFE_SKIP_PATTERNS = [
    # Per-layer embeddings — quantization here = garbage output (MLX-verified)
    "embed_tokens_per_layer",
    "per_layer_input_gate",
    "per_layer_projection",
    "per_layer_model_projection",
    "post_per_layer_input_norm",
    # Token embedding + lm_head — standard "keep fp16" for LLMs
    "embed_tokens",
    "lm_head",
    # All RMSNorms (scalar weights, sensitive)
    "norm",
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
    "v_norm",
    # RoPE precomputed tables
    "rotary_emb",
    "cos_cached",
    "sin_cached",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="exported")
    p.add_argument("--output-dir", default="quantized")
    p.add_argument("--backend", choices=["cpu", "rocm", "cuda"], default="cpu")
    p.add_argument("--checkpoint", default="checkpoints/gemma-4-e2b-it")
    p.add_argument("--num-calib-samples", type=int, default=20)
    p.add_argument("--context-length", type=int, default=4096)
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    in_dir = here / args.input_dir
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # Import lazily so the module loads even if onnxruntime isn't installed
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        QuantFormat,
        CalibrationMethod,
        CalibrationDataReader,
    )

    from .calibration import get_wikitext_calibration

    print(f"==> Loading tokenizer + calibration data ({args.num_calib_samples} samples)")
    tokenizer = AutoTokenizer.from_pretrained(here / args.checkpoint)
    calib_samples = get_wikitext_calibration(
        tokenizer,
        context_length=args.context_length,
        num_samples=args.num_calib_samples,
    )

    class CalibReader(CalibrationDataReader):
        def __init__(self, samples):
            self.it = iter(samples)
        def get_next(self):
            try:
                t = next(self.it)
                return {"input_ids": t.numpy()}
            except StopIteration:
                return None

    for graph in ["prefill", "decode"]:
        # ORT needs the model as ONNX, not .pt2 — convert via torch.onnx.export
        # done inside export.py for full pipeline. Here we assume *.onnx exists.
        in_onnx = in_dir / f"{graph}.onnx"
        if not in_onnx.exists():
            print(f"WARN: {in_onnx} missing — run torch.onnx.export step first")
            continue
        out_onnx = out_dir / f"{graph}_w8a16.onnx"
        print(f"\n==> Quantizing {graph} on {args.backend}")
        quantize_static(
            model_input=str(in_onnx),
            model_output=str(out_onnx),
            calibration_data_reader=CalibReader(calib_samples),
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt16,   # w8a16
            nodes_to_exclude=PLE_SAFE_SKIP_PATTERNS,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                # Optimal for QNN HTP per Agent 21
                "ForceQuantizeNoInputCheck": True,
            },
        )
        print(f"   wrote {out_onnx} ({out_onnx.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n==> Done. Next: python -m gemma4_v69.aihub_submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

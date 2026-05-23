"""AIMET-ONNX static quantization for Gemma 4 — w8a16, PLE-safe.

Updated per Agent 4's verdict:
- ORT `quantize_static` only does w8a8 — wrong for v69 (Agent 21 said use w8a16)
- AIMET-ONNX CPU wheel supports w8a16 natively + outputs QNN-compatible encodings
- No GPU benefit on AMD or NVIDIA for the quantization step (Agents 3, 4, 7)
- Calibration runs CPU-only, ~20-40 min on modern CPU

PLE-safe skip-list per Agent 32: quantizing PLE → garbage output.
Keep all `per_layer_*` tensors + ScaledEmbedding + norms at FP16.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


PLE_SAFE_SKIP_OPS = [
    # Per-layer embeddings — quantization here = garbage output (MLX-verified)
    "embed_tokens_per_layer",
    "per_layer_input_gate",
    "per_layer_projection",
    "per_layer_model_projection",
    "post_per_layer_input_norm",
    # Token embedding + lm_head — standard "keep fp16" for LLMs
    "embed_tokens",
    "lm_head",
    # All RMSNorms (sensitive scalar weights)
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "q_norm",
    "k_norm",
    "v_norm",
    # RoPE precomputed tables
    "rotary_emb",
]

PLE_SAFE_SKIP_OP_TYPES = [
    "LayerNormalization",  # All RMSNorms
    "RMSNormalization",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="exported")
    p.add_argument("--output-dir", default="quantized")
    p.add_argument("--checkpoint", default="checkpoints/gemma-4-e2b-it")
    p.add_argument("--num-calib-samples", type=int, default=128)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument(
        "--htp-config",
        default=None,
        help="AIMET config JSON for QNN HTP v69. Defaults to bundled.",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    in_dir = here / args.input_dir
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # Import AIMET lazily (heavy + has system deps)
    try:
        from aimet_onnx.quantsim import QuantizationSimModel
        from aimet_common.defs import QuantScheme
    except ImportError as e:
        raise RuntimeError(
            "AIMET-ONNX not installed. Run:\n"
            "  pip install https://github.com/quic/aimet/releases/download/2.26.0/"
            "aimet_onnx-2.26.0.cpu-cp310-cp310-manylinux_2_34_x86_64.whl\n"
            "(Requires Python 3.10 + apt deps from reqs_deb_onnx_common.txt)\n"
            f"Original error: {e}"
        )

    import onnx
    from .calibration import get_wikitext_calibration

    print(f"==> Loading tokenizer + calibration data ({args.num_calib_samples} samples)")
    tokenizer = AutoTokenizer.from_pretrained(here / args.checkpoint)
    calib_samples = get_wikitext_calibration(
        tokenizer,
        context_length=args.context_length,
        num_samples=args.num_calib_samples,
    )

    # AIMET config for w8a16 with HTP v69 target
    if args.htp_config:
        config_file = args.htp_config
    else:
        # Write a default HTP v69 config
        config_file = out_dir / "htp_quantsim_config_v69.json"
        config_file.write_text(json.dumps({
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True", "is_symmetric": "True"},
                "strict_symmetric": "False",
                "unsigned_symmetric": "False",
                "per_channel_quantization": "True",
            },
            "params": {
                "weight": {"is_quantized": "True"},
                "bias": {"is_quantized": "False"},
            },
            "op_type": {
                op: {"is_quantized": "False"} for op in PLE_SAFE_SKIP_OP_TYPES
            },
            "supergroups": [
                {"op_list": ["MatMul", "Add"]},
                {"op_list": ["MatMul", "BiasAdd"]},
            ],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }, indent=2))
        print(f"   wrote AIMET config to {config_file}")

    for graph in ["prefill", "decode"]:
        in_onnx = in_dir / f"{graph}.onnx"
        if not in_onnx.exists():
            print(f"WARN: {in_onnx} missing — run torch.onnx.export step in export.py first")
            continue
        out_onnx = out_dir / f"{graph}_w8a16"

        print(f"\n==> Quantizing {graph} (CPU, w8a16, ~20-40 min)")
        model = onnx.load(str(in_onnx))

        sim = QuantizationSimModel(
            model,
            param_type="int8",          # weights w8
            activation_type="int16",     # activations a16 (w8a16!)
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            config_file=str(config_file),
        )

        # Disable quantizers on PLE-sensitive ops by name pattern
        for op_name, qsim_op in sim.qc_quantize_op_dict.items():
            if any(skip_pat in op_name for skip_pat in PLE_SAFE_SKIP_OPS):
                qsim_op.enabled = False
                print(f"   skipping quant for {op_name}")

        # Calibration forward pass
        def forward_pass(session, _):
            for sample in calib_samples:
                _ = session.run(None, {"input_ids": sample.numpy()})

        sim.compute_encodings(forward_pass, None)
        sim.export(str(out_dir), graph + "_w8a16")
        print(f"   wrote {out_dir}/{graph}_w8a16.onnx + .encodings")

    print("\n==> Done. Next: python -m gemma4_v69.aihub_submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

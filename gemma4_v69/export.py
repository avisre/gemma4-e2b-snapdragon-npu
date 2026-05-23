"""torch.export Gemma 4 wrapper to ONNX, two graphs (prefill + decode).

Static-shape ONNX is required by Qualcomm AI Hub QNN compile. Per Agent 24:
    prefill graph: sequence_length=128 (efficient batched matmul)
    decode  graph: sequence_length=1   (latency-optimal per-token)

Both share weights at AI Hub link time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .wrapper import load_gemma4_e2b_for_qnn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/gemma-4-e2b-it")
    p.add_argument("--output-dir", default="exported")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--prefill-ar-len", type=int, default=128)
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    ckpt = here / args.checkpoint
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    print(f"==> Loading wrapper from {ckpt}")
    model = load_gemma4_e2b_for_qnn(
        ckpt, max_seq_len=args.max_seq_len, prefill_ar_len=args.prefill_ar_len
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    for graph_name, seq_len in [("prefill", args.prefill_ar_len), ("decode", 1)]:
        print(f"\n==> torch.export {graph_name} (seq_len={seq_len})")
        spec = model.get_static_input_specs(seq_len)
        # Materialize example inputs in spec order
        example_inputs = []
        for name, (shape, dtype) in spec.items():
            tt = getattr(torch, dtype.replace("32", "32").replace("16", "16"))
            if dtype == "int32":
                example_inputs.append(torch.zeros(shape, dtype=torch.int32))
            else:
                example_inputs.append(torch.zeros(shape, dtype=torch.float16 if dtype == "float16" else torch.float32))

        with torch.inference_mode():
            exported = torch.export.export(model, tuple(example_inputs), strict=False)

        out_path = out_dir / f"{graph_name}.pt2"
        torch.export.save(exported, out_path)
        print(f"   wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Also dump the input_specs JSON for aihub_submit
        import json
        spec_path = out_dir / f"{graph_name}_input_specs.json"
        spec_path.write_text(json.dumps({k: {"shape": v[0], "dtype": v[1]} for k, v in spec.items()}, indent=2))

    print(f"\n==> Saving tokenizer copy at {out_dir}/tokenizer.json")
    tokenizer.save_pretrained(out_dir)

    print("\n==> Done. Next: python -m gemma4_v69.quantize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

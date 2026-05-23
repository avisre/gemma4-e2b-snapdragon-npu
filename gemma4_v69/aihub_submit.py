"""Submit quantized ONNX to Qualcomm AI Hub for QNN context binary compile.

Per Agent 19, AI Hub LLM submit pattern:
1. Submit each split (component) as a separate `submit_compile_job`
2. Each split has prefill_part_N + token_part_N graphs sharing weights
3. After all parts compile, run `submit_link_job` to assemble shards
4. Download `.bin` files + tokenizer + Genie configs → genie_bundle/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import qai_hub as hub


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="quantized")
    p.add_argument("--output-dir", default="genie_bundle")
    p.add_argument(
        "--device",
        default="Samsung Galaxy S22 5G",
        help="SM8450/v69 target. Other valid: 'Xiaomi 12', 'OnePlus 10 Pro', etc.",
    )
    p.add_argument("--device-os", default="13")
    p.add_argument("--num-splits", type=int, default=4,
                   help="Shard the model into N parts (v69 needs ~4 for E2B)")
    p.add_argument("--token", default=None, help="Override QAI_HUB_API_TOKEN env")
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent
    in_dir = here / args.input_dir
    out_dir = here / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # Verify device
    devs = [d for d in hub.get_devices() if d.name == args.device]
    if not devs:
        print(f"ERROR: device '{args.device}' not in AI Hub catalog. Try `qai-hub list-devices`")
        return 2
    device = devs[0]
    print(f"==> Target: {device.name} ({device.attributes})")

    # For each split, submit prefill + decode graphs
    # In a real pipeline, you'd partition the model into N splits and
    # have N prefill.onnx + N decode.onnx files. Here, single-graph for clarity.
    job_ids: dict[str, str] = {}
    for graph in ["prefill", "decode"]:
        in_onnx = in_dir / f"{graph}_w8a16.onnx"
        if not in_onnx.exists():
            print(f"ERROR: {in_onnx} missing — run quantize.py first")
            return 3
        spec_path = (here / "exported" / f"{graph}_input_specs.json")
        input_specs = {
            k: (tuple(v["shape"]), v["dtype"]) for k, v in json.loads(spec_path.read_text()).items()
        }

        options_str = (
            "--target_runtime qnn_context_binary "
            "--compute_unit npu "
            "--quantize_full_type w8a16 "
            "--quantize_io "
            f"--qnn_graph_name {graph}_part1 "
            f"--num_sharding {args.num_splits}"
        )

        print(f"\n==> Submitting {graph} compile job to AI Hub (num_sharding={args.num_splits})")
        job = hub.submit_compile_job(
            model=str(in_onnx),
            device=device,
            name=f"gemma4_e2b_v69_{graph}",
            input_specs=input_specs,
            options=options_str,
        )
        print(f"   Job ID: {job.job_id} ({job.url})")
        job_ids[graph] = job.job_id

    # Save job IDs for polling
    (out_dir / "job_ids.json").write_text(json.dumps(job_ids, indent=2))

    print("\n==> Polling for completion (compiles usually 30-60 min for LLMs)")
    results = {}
    for graph, jid in job_ids.items():
        job = hub.get_job(jid)
        print(f"   Waiting on {graph}...")
        while job.get_status().code in ("CREATED", "PROVISIONING", "RUNNING"):
            time.sleep(60)
            job = hub.get_job(jid)
        status = job.get_status()
        print(f"   {graph}: {status.code}")
        if status.code != "SUCCESS":
            print(f"   ERROR — see {job.url}")
            return 4
        results[graph] = job.get_target_model()

    # Download .bin shards
    print(f"\n==> Downloading .bin shards to {out_dir}")
    for graph, target in results.items():
        for i, model in enumerate(target if isinstance(target, list) else [target]):
            dst = out_dir / f"{graph}_part{i+1}.bin"
            model.download(str(dst))
            print(f"   {dst.name}: {dst.stat().st_size / 1024 / 1024:.1f} MB")

    # Write htp_backend_ext_config.json for v69
    htp_cfg = {
        "devices": [{
            "soc_model": 36,  # SM8450 — per Agent 5 (Qualcomm SoC ID, not the SM number)
            "dsp_arch": "v69",
            "cores": [{
                "core_id": 0,
                "perf_profile": "burst",
                "rpc_control_latency": 100,
            }],
        }],
        "memory": {"mem_type": "shared_buffer"},
        "context": {"weight_sharing_enabled": True},
    }
    (out_dir / "htp_backend_ext_config.json").write_text(json.dumps(htp_cfg, indent=2))

    # Genie config skeleton
    bin_files = sorted([p.name for p in out_dir.glob("*.bin")])
    genie_cfg = {
        "dialog": {
            "engine": {
                "model": {
                    "ctx-bins": bin_files,
                    "graphs": ["prefill_part1", "token_part1"],
                    "size": 2048,
                    "n-vocab": 262144,
                },
                "tokenizer": {"path": "<tokenizer_path>"},
                "htp_backend_ext": {"path": "<htp_backend_ext_path>"},
            },
            "sampler": {"version": 1, "temp": 0.7, "top-k": 40, "top-p": 0.95},
        },
        "extensions": [{"path": "<extensions_path>"}],
    }
    (out_dir / "genie_config.json").write_text(json.dumps(genie_cfg, indent=2))

    print(f"\n==> Done. Bundle ready at {out_dir}")
    print(f"   Next: cp {here}/checkpoints/gemma-4-e2b-it/tokenizer.json {out_dir}/")
    print(f"   Then: ./push_and_run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

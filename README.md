# Gemma 4 E2B → Snapdragon 8 Gen 1 NPU

Pipeline to compile Google's Gemma 4 E2B (text decoder) into a QNN context binary that runs on the **Hexagon v69 NPU** (Snapdragon 8 Gen 1 / SM8450).

Based on the 32-agent research synthesis. The path:

```
HF Gemma 4 E2B-it           ExecuTorch PR #19166                 AI Hub                Phone
    (10 GB BF16)             (text decoder wrapper)              (cloud compile)         (qnn_llama_runner)
        │                          │                                  │                       │
        ▼                          ▼                                  ▼                       ▼
   safetensors    ──►    Gemma4Model (PyTorch)    ──►    .pt2 → .bin (context binary)  ──►   ADB push, run
                              │
                              ▼
                  PLE inlined into CPU side
                  (LiteRT trick — not in QNN graph)
                              │
                              ▼
                    ONNX Runtime static quantize
                    (w8a16, v69-optimal — no native INT4 HMX)
                              │
                              ▼
                   torch.export prefill (AR-128) + decode (AR-1)
                              │
                              ▼
                   qai_hub.submit_compile_job × num_splits
                              │
                              ▼
                   qai_hub.submit_link_job (assembles shards)
```

## Quick reference

| Knob | Value | Why |
|---|---|---|
| Target SoC | SM8450 | OnePlus 10 Pro / Samsung S22 / Xiaomi 12 |
| Hexagon arch | v69 | 8 MB VTCM, no native INT4 HMX |
| Precision | **w8a16** (NOT w4a16) | v69 lacks native INT4 tile — w4 dequants to INT8 anyway |
| `max_seq_len` | 2048 | KV cache ~218 MB, fits 8MB VTCM utilization 60-75% |
| `prefill_ar_len` | 128 | Standard AI Hub recipe |
| `num_sharding` | 4 | E2B is ~1.6 GB; v69 caps single context at ~1 GB |
| Chat template | `--decoder_model_version gemma3` | Gemma 4 reuses Gemma 3 turn delimiters |
| Calibration data | WikiText-2 train, 20 samples × 4096 | Mirrors qai-hub-models recipe |
| Expected throughput | 12-18 tok/s decode | Per Agent D's estimate; assumes clean partition |

## Critical: PLE quantization trap

Quantizing Per-Layer Embedding paths to w4/w8 causes **catastrophic output garbage** (confirmed via MLX community + Qualcomm AI Hub Llama quant patterns). All `per_layer_*` and `ScaledEmbedding` tensors MUST stay FP16. Our quantizer skip-list reflects this.

## Files

```
gemma4_e2b_v69/
├── README.md                     # this file
├── setup.sh                      # venv + Python deps + ExecuTorch clone
├── download_model.sh             # HF download
├── checkpoints/
│   └── gemma-4-e2b-it/           # HF snapshot (config + safetensors + tokenizer)
├── gemma4_v69/
│   ├── __init__.py
│   ├── wrapper.py                # GemmaFor4QNN — PR #19166-derived text decoder wrapper
│   ├── export.py                 # torch.export → ONNX (prefill + decode)
│   ├── quantize.py               # ONNX Runtime static quant, w8a16, PLE-safe skip-list
│   ├── aihub_submit.py           # qai_hub.submit_compile_job × num_splits + link_job
│   └── calibration.py            # WikiText-2 loader matching qai-hub-models recipe
├── exported/                     # torch.export outputs (.pt2)
├── quantized/                    # ONNX Runtime outputs
├── aihub/
│   └── job_ids.json              # AI Hub compile job IDs
├── genie_bundle/                 # final deployable (push to phone)
│   ├── prefill_part{1,2,3,4}.bin
│   ├── token_part{1,2,3,4}.bin
│   ├── tokenizer.json
│   ├── genie_config.json
│   └── htp_backend_ext_config.json  # soc_model:36, dsp_arch:"v69"
└── push_and_run.sh               # ADB push + run via qnn_llama_runner
```

## How to run end-to-end (rough flow)

```bash
./setup.sh                                          # 30 min
./download_model.sh                                 # 10 min, ~10 GB
python gemma4_v69/export.py                         # 10 min CPU
python gemma4_v69/quantize.py --backend cpu         # 30-90 min CPU (or --backend rocm for AMD GPU)
QAI_HUB_API_TOKEN=... python gemma4_v69/aihub_submit.py   # ~30-60 min AI Hub queue
./push_and_run.sh                                   # 10 min push + test prompt
```

Total wall-clock: **~3-5 hours** end-to-end (most of it AI Hub queue + download time).

## Why this can't be a single 10-minute command

You're literally the first person to try this on v69 SM8450. The pipeline involves:
1. A research wrapper (PR #19166) that's XNNPACK-only — we adapt it
2. Quantization that AI Hub doesn't do server-side for LLMs
3. A model architecture (PLE) that breaks standard PTQ patterns
4. A target SoC (v69) that's outside Qualcomm's official LLM matrix

K9FxNa did Qwen3-0.6B for v69 in one weekend by reusing the existing recipe. Gemma 4 has no recipe — we're writing it. Expect to iterate.

## Credits

- **PR #19166 contributors** (ExecuTorch / Meta) — the text decoder wrapper
- **K9FxNa** — proved v69 LLM-on-NPU is possible and shipped the first .pte
- **xororz (Local Dream)** — proved v69 NPU works for transformer-class graphs
- **Qualcomm AI Hub team** — cloud compile + QNN runtime
- **MLX team** — reference PLE-safe quantization patterns
- **The 32-agent swarm** — turned this from "1 week ETA" into a working plan

## License

MIT for the pipeline code in this directory. Upstream artifacts (Gemma weights, ExecuTorch, QNN runtime) under their own licenses.

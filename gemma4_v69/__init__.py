"""Gemma 4 E2B → Hexagon v69 NPU pipeline.

Modules:
- wrapper:        text-only Gemma 4 wrapper derived from ExecuTorch PR #19166
- export:         torch.export to prefill + decode ONNX
- quantize:       ONNX Runtime static quant (w8a16, PLE-safe skip-list)
- aihub_submit:   AI Hub compile + link jobs
- calibration:    WikiText-2 calibration data loader
"""

#!/bin/bash
# One-time setup: Python venv + qai-hub + ExecuTorch source.
# Run from this directory.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> 1. Python 3.10 venv (aimet-onnx wheel is cp310-only)"
if ! command -v python3.10 >/dev/null; then
    echo "Install python3.10 first: 'sudo apt install python3.10 python3.10-venv'"
    exit 2
fi
python3.10 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

echo "==> 2. Core deps"
pip install \
    torch==2.4.0 \
    transformers==4.46.0 \
    safetensors \
    huggingface_hub \
    sentencepiece \
    qai-hub \
    qai-hub-models \
    onnx \
    onnxruntime \
    datasets

# Choose: install onnxruntime-rocm for AMD GPU, otherwise CPU
if [ "${USE_ROCM:-0}" = "1" ]; then
    echo "==> ROCm path (AMD GPU)"
    pip install onnxruntime-training onnx-rocm 2>/dev/null || \
        echo "WARN: ROCm wheels are experimental for RDNA4 — may fall back to CPU"
fi

echo "==> 3. Clone ExecuTorch (for PR #19166 Gemma 4 text decoder wrapper)"
if [ ! -d executorch ]; then
    git clone --depth 1 --branch viable/strict https://github.com/pytorch/executorch.git
fi

echo "==> 4. Verify qai-hub auth"
if [ -z "${QAI_HUB_API_TOKEN:-}" ]; then
    echo "Set QAI_HUB_API_TOKEN in env, then run:"
    echo "  qai-hub configure --api_token \$QAI_HUB_API_TOKEN"
else
    qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
fi

echo "==> 5. Verify device target is reachable"
python3 -c "
import qai_hub as hub
devs = [d for d in hub.get_devices() if 'sm8450' in (d.attributes or '').lower() or 'Galaxy S22' in d.name]
print('SM8450 devices:', [d.name for d in devs[:5]])
"

echo ""
echo "==> Setup done."
echo "Next: ./download_model.sh"

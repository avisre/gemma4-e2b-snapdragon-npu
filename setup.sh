#!/bin/bash
# One-time setup: Python 3.10 venv + AIMET-ONNX + qai-hub + ExecuTorch.
# Optional: build llama.cpp with Vulkan to validate FP16 Gemma 4 on 9070 XT.
#
# Verified by 8-agent swarm (2026-05-23):
# - Quantization is CPU-only by design regardless of GPU vendor.
# - AIMET-ONNX (CPU wheel) is the correct quantizer for w8a16 + QNN target.
# - 9070 XT is only useful for FP16 inference validation via Vulkan.
# - ROCm 7.2.3 supports gfx1201 natively but adds no value here.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> 1. Python 3.10 venv (AIMET-ONNX wheel is cp310-only)"
if ! command -v python3.10 >/dev/null; then
    echo "Install python3.10 first:"
    echo "  sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 2
fi
python3.10 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

echo "==> 2. Core deps (CPU PyTorch — quantize is CPU-bound regardless of GPU)"
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

echo "==> 3. AIMET-ONNX (the right quantizer for w8a16 + QNN target per Agent 4)"
AIMET_WHL="https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cpu-cp310-cp310-manylinux_2_34_x86_64.whl"
pip install "$AIMET_WHL" || echo "WARN: AIMET wheel install failed — check version + apt deps from reqs_deb_onnx_common.txt"

echo "==> 4. Clone ExecuTorch (for PR #19166 Gemma 4 text decoder wrapper)"
if [ ! -d executorch ]; then
    git clone --depth 1 --branch viable/strict https://github.com/pytorch/executorch.git
fi

echo "==> 5. (Optional) llama.cpp Vulkan — to validate FP16 Gemma 4 on your AMD GPU"
if [ "${BUILD_LLAMA_CPP_VULKAN:-0}" = "1" ]; then
    if ! command -v vulkaninfo >/dev/null; then
        echo "Install Vulkan first: sudo apt install vulkan-tools libvulkan-dev"
        echo "Skipping llama.cpp build."
    else
        if [ ! -d llama.cpp ]; then
            git clone https://github.com/ggerganov/llama.cpp.git
        fi
        cd llama.cpp
        cmake -B build -DGGML_VULKAN=ON
        cmake --build build -j --target llama-cli llama-bench
        cd ..
        echo "==> Validate Gemma 4 on 9070 XT (after download_model.sh):"
        echo "    ./llama.cpp/build/bin/llama-cli -m <gguf_path> -ngl 99 -p 'hello' -n 32"
    fi
else
    echo "    Skipping llama.cpp Vulkan build. Re-run with BUILD_LLAMA_CPP_VULKAN=1 if wanted."
fi

echo "==> 6. Verify qai-hub auth"
if [ -z "${QAI_HUB_API_TOKEN:-}" ]; then
    echo "Set QAI_HUB_API_TOKEN in env, then:"
    echo "  qai-hub configure --api_token \$QAI_HUB_API_TOKEN"
else
    qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
fi

echo "==> 7. Verify SM8450 device is reachable"
python3 -c "
import qai_hub as hub
devs = [d for d in hub.get_devices() if 'Galaxy S22' in d.name or 'sm8450' in (d.attributes or '').lower()]
print('SM8450 devices:', [d.name for d in devs[:5]])
" || echo "WARN: qai-hub auth issue"

echo ""
echo "==> Setup done."
echo "Next: ./download_model.sh (~10 GB, 10-30 min depending on bandwidth)"

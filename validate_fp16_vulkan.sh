#!/bin/bash
# Use AMD RX 9070 XT via Vulkan to sanity-check FP16 Gemma 4 BEFORE quantizing.
# This is the ONE place the AMD GPU actually helps in this pipeline.
#
# Per Agent 5: llama.cpp Vulkan is mature on RDNA4; ROCm setup isn't needed
# for this validation step.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "$HERE/llama.cpp/build/bin/llama-cli" ]; then
    echo "llama.cpp Vulkan not built. Run:"
    echo "  BUILD_LLAMA_CPP_VULKAN=1 ./setup.sh"
    exit 1
fi

# Convert Gemma 4 E2B to GGUF FP16 (one-time, ~2 min)
GGUF="$HERE/checkpoints/gemma-4-e2b-it.f16.gguf"
if [ ! -f "$GGUF" ]; then
    echo "==> Converting HF checkpoint to GGUF FP16"
    if [ ! -d "$HERE/llama.cpp/convert_hf_to_gguf.py" ]; then
        cd "$HERE/llama.cpp" && ls convert*.py
    fi
    cd "$HERE/llama.cpp"
    python3 convert_hf_to_gguf.py "$HERE/checkpoints/gemma-4-e2b-it" \
        --outtype f16 \
        --outfile "$GGUF"
    cd "$HERE"
fi

# Verify GPU + run a test prompt
echo ""
echo "==> Vulkan device check"
vulkaninfo --summary 2>/dev/null | grep -i "AMD Radeon" | head -3 || echo "(no AMD Vulkan device found)"

PROMPT="${1:-The capital of France is}"
echo ""
echo "==> Running FP16 Gemma 4 on Vulkan (your 9070 XT)"
"$HERE/llama.cpp/build/bin/llama-cli" \
    -m "$GGUF" \
    -ngl 99 \
    -p "$PROMPT" \
    -n 64 \
    --temp 0.1

echo ""
echo "==> If output is coherent, FP16 model is healthy."
echo "==> Save this baseline; compare it against post-quantize output."

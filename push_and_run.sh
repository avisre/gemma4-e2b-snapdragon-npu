#!/bin/bash
# Push genie_bundle to phone and run 1 prompt.
# Per Agent 25, Gemma 4 reuses Gemma 3's chat template, so
# `--decoder_model_version gemma3` works with the existing qnn_llama_runner.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEVICE_SERIAL="${DEVICE_SERIAL:-a5523839}"
DEVICE_DIR="/data/local/tmp/gemma4_e2b_v69"

cd "$HERE/genie_bundle"

# Copy tokenizer if not already present
[ -f tokenizer.json ] || cp "$HERE/checkpoints/gemma-4-e2b-it/tokenizer.json" .

echo "==> Pushing genie_bundle to phone"
adb -s "$DEVICE_SERIAL" shell "mkdir -p $DEVICE_DIR"
adb -s "$DEVICE_SERIAL" push . "$DEVICE_DIR/"

# Reuse the QNN runtime libs already on the phone from the Qwen3 setup
PHONE_QNN_DIR=/data/local/tmp/executorch_qualcomm_tutorial
echo "==> Linking Qwen3 setup's QNN libs (V69Stub/Skel/system) into bundle dir"
for lib in libQnnHtp.so libQnnSystem.so libQnnHtpV69Stub.so libQnnHtpV69Skel.so libqnn_executorch_backend.so qnn_llama_runner; do
    adb -s "$DEVICE_SERIAL" shell "[ -f $DEVICE_DIR/$lib ] || cp $PHONE_QNN_DIR/$lib $DEVICE_DIR/"
done
adb -s "$DEVICE_SERIAL" shell "chmod +x $DEVICE_DIR/qnn_llama_runner"

PROMPT="${1:-Capital of France}"
echo "==> Running prompt: $PROMPT"
adb -s "$DEVICE_SERIAL" shell << ADBEOF
cd $DEVICE_DIR
export LD_LIBRARY_PATH=\$PWD
export ADSP_LIBRARY_PATH=\$PWD
./qnn_llama_runner \\
    --decoder_model_version gemma3 \\
    --tokenizer_path tokenizer.json \\
    --model_path prefill_part1.bin \\
    --prompt "$PROMPT" \\
    --seq_len 256 \\
    --kv_updater SmartMask \\
    --eval_mode 1 \\
    --temperature 0.7 2>&1 | grep -E "tokens/second|Time to first|Total inference|Generated"
echo "--- Response ---"
cat outputs.txt
ADBEOF

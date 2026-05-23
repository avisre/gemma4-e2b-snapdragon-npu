#!/bin/bash
# Download Gemma 4 E2B-it from HuggingFace. Apache 2.0, not gated.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Optional: set HF_TOKEN for faster downloads + resumability
# huggingface-cli login

python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='google/gemma-4-E2B-it',
    local_dir='$HERE/checkpoints/gemma-4-e2b-it',
    allow_patterns=['*.safetensors', '*.safetensors.index.json', '*.json', '*.txt', '*.md'],
    max_workers=4,
)
print('Downloaded to:', path)
"

echo ""
echo "==> Verifying"
ls -lh "$HERE/checkpoints/gemma-4-e2b-it/" | head -20
echo ""
total=$(du -sh "$HERE/checkpoints/gemma-4-e2b-it/" | awk '{print $1}')
echo "Total: $total"

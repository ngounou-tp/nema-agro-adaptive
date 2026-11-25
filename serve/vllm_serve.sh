#!/usr/bin/env bash
# Serve the base model + both LoRA adapters behind one OpenAI-compatible API (port 8000).
# Needs a Linux machine with an NVIDIA GPU (24 GB+ for a 9B model in bf16; e.g. L4 or A10G), Python 3.10+.
#   pip install vllm
#   export HF_TOKEN=hf_...            (only if a repo is private)
#   export VLLM_API_KEY=choose-a-long-random-secret   (clients must send it as a Bearer token)
#   bash serve/vllm_serve.sh
set -euo pipefail
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"            # the exact base used by AutoScientist
AGRO_ADAPTER="${AGRO_ADAPTER:-YOUR_HF_USER/nema-agro-lora}"
WASTE_ADAPTER="${WASTE_ADAPTER:-YOUR_HF_USER/revalo-waste-lora}"
LORA_RANK="${LORA_RANK:-16}"                           # 'r' in adapter_config.json (use the larger of the two)

vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules nema-agro="$AGRO_ADAPTER" revalo-waste="$WASTE_ADAPTER" \
  --max-lora-rank "$LORA_RANK" \
  --max-model-len 4096 \
  --api-key "${VLLM_API_KEY:?set VLLM_API_KEY}" \
  --host 0.0.0.0 --port 8000

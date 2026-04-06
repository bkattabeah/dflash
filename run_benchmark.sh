#!/bin/bash
# Unified benchmark runner.
#
# Usage:
#   BACKEND=transformers bash run_benchmark.sh
#   BACKEND=sglang BASE_URL=http://127.0.0.1:30000 MODEL=Qwen/Qwen3.5-9B bash run_benchmark.sh
#   BACKEND=vllm   BASE_URL=http://127.0.0.1:8000   MODEL=Qwen/Qwen3.5-27B bash run_benchmark.sh

BACKEND="${BACKEND:-transformers}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"

mkdir -p logs

datasets=(gsm8k math500 humaneval mbpp mt-bench)
# max_samples per dataset (transformers only)
declare -A samples=([gsm8k]=128 [math500]=128 [humaneval]=164 [mbpp]=128 [mt-bench]=80)

for ds in "${datasets[@]}"; do
  echo "========================================================"
  echo "Running ${BACKEND} benchmark: ${ds}"
  echo "========================================================"

  if [ "$BACKEND" = "transformers" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    torchrun \
      --nproc_per_node=8 \
      --master_port=29600 \
      benchmark.py \
      --backend transformers \
      --model "$MODEL" \
      --draft-model "$DRAFT_MODEL" \
      --dataset "$ds" \
      --max-samples "${samples[$ds]}" \
      --max-new-tokens 2048 \
      --temperature 0.0 \
      2>&1 | tee "logs/${BACKEND}_${ds}.log"
  else
    python benchmark.py \
      --backend "$BACKEND" \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset "$ds" \
      --num-prompts 1024 \
      --max-new-tokens 4096 \
      --concurrency 32 \
      2>&1 | tee "logs/${BACKEND}_${ds}.log"
  fi
done

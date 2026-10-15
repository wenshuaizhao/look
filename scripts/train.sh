#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WORK="${WORK:-$ROOT/work}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TOKENIZERS_PARALLELISM=false
ARM=${ARM:-2}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DATA=("$WORK/data/mix/"*.jsonl)
[[ -f "${DATA[0]}" ]] || { echo 'Build the instruction mix first (see README).' >&2; exit 1; }
case "$ARM" in
  1) ;;
  2) DATA+=("$WORK/data/look/look.jsonl");;
  3) DATA+=("$WORK/data/look/look.jsonl#${LOOK_N:-4600}");;
  *) echo 'ARM must be 1 (SFT), 2 (LOOK 1:1), or 3 (LOOK subset)' >&2; exit 1;;
esac
swift sft --model "$MODEL" \
  --template qwen3_vl_look --custom_register_path "$ROOT/looktask/template.py" \
  --dataset "${DATA[@]}" --split_dataset_ratio 0.01 \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa \
  --num_train_epochs "${EPOCHS:-2}" --learning_rate "${LR:-1e-4}" \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 --max_length 4096 --warmup_ratio 0.05 \
  --eval_steps 200 --save_steps 200 --save_total_limit 40 --logging_steps 10 \
  --dataset_num_proc "${DATASET_NUM_PROC:-16}" --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
  --output_dir "$WORK/runs/${RUN:-arm$ARM}" "$@"

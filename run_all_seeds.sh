#!/usr/bin/env bash

set -euo pipefail

SCRIPT="runWeightedRankBased.py"

DATASET="mnist"
ALPHA="0.01"

SEEDS=(2026 13007 49211 71023 98029 600011)

MAX_PARALLEL=6

LOG_DIR="./logs/${DATASET}}"
mkdir -p "$LOG_DIR"

run_job () {
    seed="$1"

    log_file="${LOG_DIR}/${seed}/execution_log.log"

    echo "Starting seed ${seed}"

    python3 "$SCRIPT" \
        --seed "$seed" \
        --dataset "$DATASET" \
        --alpha "$ALPHA" \
        > "$log_file" 2>&1

    echo "Finished seed ${seed}"
}

running=0

for seed in "${SEEDS[@]}"; do

    run_job "$seed" &
    ((running+=1))

    if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
        wait -n
        ((running-=1))
    fi

done

wait

echo "All experiments completed."
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEMO_DATASET_ROOT="$PROJECT_ROOT/demo_assets/mvtec_ad/hazelnut"

required=()
for normal_index in {000..014}; do
    required+=("train/good/${normal_index}.png")
done
required+=(
    "test/crack/000.png" "test/hole/000.png" "test/print/000.png" "test/cut/000.png"
    "ground_truth/crack/000_mask.png" "ground_truth/hole/000_mask.png"
    "ground_truth/print/000_mask.png" "ground_truth/cut/000_mask.png"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "$DEMO_DATASET_ROOT/$relative_path" ]]; then
        echo "ERROR: missing bundled demo asset: $DEMO_DATASET_ROOT/$relative_path" >&2
        exit 1
    fi
done

export OBJECT_NAME=hazelnut
export DATASET_ROOT="$DEMO_DATASET_ROOT"
export ANOMALIES_STR="${ANOMALIES_STR:-crack hole print cut}"
export REF_IDS_STR="${REF_IDS_STR:-000 000 000 000}"
export SAMPLES_PER_ANOMALY="${SAMPLES_PER_ANOMALY:-1}"
export SAMPLES_PER_PAIR_STR="${SAMPLES_PER_PAIR_STR:-}"
export SEED="${SEED:-309}"
export RUN_NAME="${RUN_NAME:-hazelnut_in_context_demo_seed${SEED}}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/$RUN_NAME}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

echo "models:   local_files_only=$LOCAL_FILES_ONLY"
exec bash "$PROJECT_ROOT/scripts/run_hazelnut_t2r.sh"

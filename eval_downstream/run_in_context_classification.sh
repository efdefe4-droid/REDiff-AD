#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

OBJ="${OBJ:-hazelnut}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-$REPO_ROOT/outputs}"
RESULT_ROOT="${RESULT_ROOT:-$PROPOSAL_ROOT/hazelnut_rediff_ad}"
GENERATED_LAYOUT="${GENERATED_LAYOUT:-in_context}"
IMAGE_NAME="${IMAGE_NAME:-edit.png}"

DEFAULT_ANOMALIES=(
    crack
    cut
    hole
    print
)

if [ -n "${ANOMALIES:-}" ]; then
    ANOMALIES_NORMALIZED="${ANOMALIES//,/ }"
    read -r -a ANOMALY_LIST <<< "$ANOMALIES_NORMALIZED"
elif [ "$GENERATED_LAYOUT" = "tf-idg" ] && [ -d "$RESULT_ROOT/test" ]; then
    mapfile -t ANOMALY_LIST < <(find "$RESULT_ROOT/test" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
elif [ -d "$RESULT_ROOT" ]; then
    mapfile -t ANOMALY_LIST < <(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
else
    ANOMALY_LIST=("${DEFAULT_ANOMALIES[@]}")
fi

GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-rediff-ad}"
EPOCHS="${EPOCHS:-30}"
BS="${BS:-8}"
LR="${LR:-0.00001}"
SEED="${SEED:-2026}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
TEST_SPLIT="${TEST_SPLIT:-last_two_thirds}"
CLASSIFICATION_TRAIN_REPEAT="${CLASSIFICATION_TRAIN_REPEAT:-3}"
MAX_IMAGES="${MAX_IMAGES:-250}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
CLEAN_PREPARED="${CLEAN_PREPARED:-1}"
case "$GENERATED_LAYOUT" in
    anomaly-diffusion|seas|anostyle|dualanodiff|self-anomalydiffusion|o2mag-flat|tf-idg)
        MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-0.0}"
        MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-1.0}"
        ;;
    *)
        MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-0.001}"
        MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-0.20}"
        ;;
esac
MASK_NAME="${MASK_NAME:-contour_refined_mask.png}"
LINK_FILES="${LINK_FILES:-1}"

MVTEC_PATH="${MVTEC_PATH:-${MVTEC_ROOT:-$REPO_ROOT/datasets}}"
RUN_TAG="$(basename "$RESULT_ROOT")"
DATASET_NAME="${DATASET_NAME:-$(basename "$PROPOSAL_ROOT")}"
RESULT_TAG="${RESULT_TAG:-$RUN_TAG}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$RESULT_TAG}"
CLASSIFICATION_RESULT_CSV="${CLASSIFICATION_RESULT_CSV:-$RESULTS_DIR/${OBJ}_classification.csv}"
GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/eval_downstream/generated_data/classification/$RESULT_TAG}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/eval_downstream/checkpoints/classification/$RESULT_TAG/$OBJ}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/eval_downstream/logs/classification/$RESULT_TAG}"
LOG_FILE="$LOG_DIR/${OBJ}.log"

if [ ! -d "$RESULT_ROOT" ]; then
    echo "Missing RESULT_ROOT: $RESULT_ROOT"
    exit 1
fi

if [ ! -d "$MVTEC_PATH/$OBJ/train/good" ]; then
    echo "Missing MVTec train/good path: $MVTEC_PATH/$OBJ/train/good"
    exit 1
fi

if [ ! -d "$MVTEC_PATH/$OBJ/test" ]; then
    echo "Missing MVTec test path: $MVTEC_PATH/$OBJ/test"
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found."
    exit 1
fi

mkdir -p "$LOG_DIR" "$CHECKPOINT_PATH" "$RESULTS_DIR"
: > "$LOG_FILE"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PREPARE_ARGS=(
    --input-layout "$GENERATED_LAYOUT"
    --in-context-results-root "$RESULT_ROOT"
    --sample-name "$OBJ"
    --anomalies "${ANOMALY_LIST[@]}"
    --output-root "$GENERATED_DATA_ROOT"
    --image-name "$IMAGE_NAME"
    --mask-name "$MASK_NAME"
    --max-images "$MAX_IMAGES"
    --min-mask-area-ratio "$MIN_MASK_AREA_RATIO"
    --max-mask-area-ratio "$MAX_MASK_AREA_RATIO"
    --require-mask-file
)

if [ "$CLEAN_PREPARED" = "1" ]; then
    PREPARE_ARGS+=(--clean)
fi

if [ "$LINK_FILES" = "1" ]; then
    PREPARE_ARGS+=(--link-files)
fi

if [ "$SKIP_MISSING" = "1" ]; then
    PREPARE_ARGS+=(--skip-missing)
fi

log "============================================================"
log "In-Context downstream classification via eval_downstream"
log "Object:       $OBJ"
log "Anomalies:    ${ANOMALY_LIST[*]}"
log "Result root:  $RESULT_ROOT"
log "Layout:       $GENERATED_LAYOUT"
log "MVTec path:   $MVTEC_PATH"
log "Data root:    $GENERATED_DATA_ROOT"
log "Checkpoint:   $CHECKPOINT_PATH"
log "Result CSV:   $CLASSIFICATION_RESULT_CSV"
log "Log file:     $LOG_FILE"
log "Conda env:    $CONDA_ENV"
log "Image size:   $IMAGE_SIZE"
log "Test split:   $TEST_SPLIT"
log "Train repeat: $CLASSIFICATION_TRAIN_REPEAT"
log "Seed:         $SEED"
log "Max images:   $MAX_IMAGES"
log "Mask name:    $MASK_NAME"
log "Mask area:    [$MIN_MASK_AREA_RATIO, $MAX_MASK_AREA_RATIO]"
log "Link files:   $LINK_FILES"
log "SKIP_PREPARE: $SKIP_PREPARE"
log "============================================================"

if [ "$SKIP_PREPARE" = "1" ]; then
    log "SKIP_PREPARE=1, using existing prepared data."
else
    log
    log "============================================================"
    log "Preparing In-Context outputs for classification"
    log "============================================================"
    python -u eval_downstream/prepare_reflex_classification_data.py "${PREPARE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

MANIFEST_PATH="$GENERATED_DATA_ROOT/$OBJ/manifest.csv"
if [ ! -s "$MANIFEST_PATH" ]; then
    log "ERROR: missing or empty prepared manifest: $MANIFEST_PATH"
    log "       Check RESULT_ROOT=$RESULT_ROOT and MASK_NAME=$MASK_NAME"
    exit 1
fi
PREPARED_COUNT="$(awk 'NR > 1 {count++} END {print count + 0}' "$MANIFEST_PATH")"
if [ "$PREPARED_COUNT" -eq 0 ]; then
    log "ERROR: prepared data has 0 images: $MANIFEST_PATH"
    log "       Check RESULT_ROOT=$RESULT_ROOT and MASK_NAME=$MASK_NAME"
    exit 1
fi
log "Prepared images: $PREPARED_COUNT"

if [ "$PREPARE_ONLY" = "1" ]; then
    log "PREPARE_ONLY=1, skip classification training."
    exit 0
fi

log
log "============================================================"
log "Training classification downstream model"
log "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" python -u eval_downstream/train-classification.py \
    --sample_name "$OBJ" \
    --mvtec_path "$MVTEC_PATH" \
    --generated_data_path "$GENERATED_DATA_ROOT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --lr "$LR" \
    --image_size "$IMAGE_SIZE" \
    --test_split "$TEST_SPLIT" \
    --train_repeat "$CLASSIFICATION_TRAIN_REPEAT" \
    --seed "$SEED" \
    --result_csv "$CLASSIFICATION_RESULT_CSV" \
    --dataset_name "$DATASET_NAME" 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

OBJ="${OBJ:-hazelnut}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-$REPO_ROOT/outputs}"
RESULT_ROOT="${RESULT_ROOT:-$PROPOSAL_ROOT/hazelnut_rediff_ad}"
GENERATED_LAYOUT="${GENERATED_LAYOUT:-insert-anything}"

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
LR="${LR:-0.0001}"
SEED="${SEED:-2026}"
SELECTION_SPLIT="${SELECTION_SPLIT:-val}"
FINAL_TEST_SPLIT="${FINAL_TEST_SPLIT:-test}"
LOCALIZATION_SELECTION_METRIC="${LOCALIZATION_SELECTION_METRIC:-pixel}"
EXTRA_REAL_ANOMALY_PROB="${EXTRA_REAL_ANOMALY_PROB:-0.1}"
MAX_IMAGES="${MAX_IMAGES:-250}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
RUN_BEST_CKPT_TEST="${RUN_BEST_CKPT_TEST:-1}"
BEST_CKPT_TEST_SPLIT="${BEST_CKPT_TEST_SPLIT:-test}"
CLEAN_PREPARED="${CLEAN_PREPARED:-1}"
COPY_RAW_PREPARE="${COPY_RAW_PREPARE:-0}"
MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-0.0}"
MAX_MASK_AREA_RATIO="${MAX_MASK_AREA_RATIO:-1.0}"
IMAGE_NAME="${IMAGE_NAME:-edit.png}"
MASK_NAME="${MASK_NAME:-contour_refined_mask.png}"
LINK_FILES="${LINK_FILES:-1}"
RUN_TAG="$(basename "$RESULT_ROOT")"
DATASET_NAME="${DATASET_NAME:-$(basename "$PROPOSAL_ROOT")}"
RESULT_TAG="${RESULT_TAG:-$RUN_TAG}"
EXP_NAME="${EXP_NAME:-${RESULT_TAG}_${OBJ}_localization}"

MVTEC_PATH="${MVTEC_PATH:-${MVTEC_ROOT:-$REPO_ROOT/datasets}}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$RESULT_TAG}"
LOCALIZATION_RESULT_CSV="${LOCALIZATION_RESULT_CSV:-$RESULTS_DIR/${OBJ}_localization.csv}"
GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/eval_downstream/generated_data/localization/$RESULT_TAG}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/eval_downstream/checkpoints/localization/$RESULT_TAG/$OBJ}"
BEST_CKPT_TEST_DIR="${BEST_CKPT_TEST_DIR:-$REPO_ROOT/eval_downstream/test_results/localization/$RESULT_TAG/$OBJ}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/eval_downstream/logs/localization/$RESULT_TAG}"
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

if [ ! -d "$MVTEC_PATH/$OBJ/ground_truth" ]; then
    echo "Missing MVTec ground_truth path: $MVTEC_PATH/$OBJ/ground_truth"
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

PREPARE_ARGS=(
    --input-layout "$GENERATED_LAYOUT"
    --insert-anything-results-root "$RESULT_ROOT"
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
elif [ "$COPY_RAW_PREPARE" = "1" ]; then
    PREPARE_ARGS+=(--copy-raw)
fi

if [ "$SKIP_MISSING" = "1" ]; then
    PREPARE_ARGS+=(--skip-missing)
fi

log "============================================================"
log "Insert-Anything downstream localization fine tune"
log "Object:       $OBJ"
log "Anomalies:    ${ANOMALY_LIST[*]}"
log "Result root:  $RESULT_ROOT"
log "Layout:       $GENERATED_LAYOUT"
log "Experiment:   $EXP_NAME"
log "Image name:   $IMAGE_NAME"
log "GT mask name: $MASK_NAME"
log "Max images:   $MAX_IMAGES"
log "Link files:   $LINK_FILES"
log "Copy raw:     $COPY_RAW_PREPARE"
log "Data root:    $GENERATED_DATA_ROOT"
log "Checkpoint:   $CHECKPOINT_PATH"
log "Log file:     $LOG_FILE"
log "MVTec path:   $MVTEC_PATH"
log "Conda env:    $CONDA_ENV"
log "GPU visible:  $GPU"
log "Epochs:       $EPOCHS"
log "Batch size:   $BS"
log "Seed:         $SEED"
log "Select split: $SELECTION_SPLIT"
log "Final split:  $FINAL_TEST_SPLIT"
log "Selection metric: $LOCALIZATION_SELECTION_METRIC"
log "Extra real anomaly prob: $EXTRA_REAL_ANOMALY_PROB"
log "Best ckpt test: $RUN_BEST_CKPT_TEST split=$BEST_CKPT_TEST_SPLIT"
log "Test result dir: $BEST_CKPT_TEST_DIR"
log "Result CSV: $LOCALIZATION_RESULT_CSV"
log "============================================================"

if [ "$SKIP_PREPARE" = "1" ]; then
    log "SKIP_PREPARE=1, using existing prepared data."
else
    log
    log "============================================================"
    log "Preparing Insert-Anything outputs for localization"
    log "============================================================"
    conda run --no-capture-output -n "$CONDA_ENV" python eval_downstream/prepare_reflex_classification_data.py "${PREPARE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
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
    log "PREPARE_ONLY=1, skip localization fine tune."
    exit 0
fi

log
log "============================================================"
log "Training localization downstream model"
log "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" python eval_downstream/train-localization.py \
    --sample_name "$OBJ" \
    --mvtec_path "$MVTEC_PATH" \
    --generated_data_path "$GENERATED_DATA_ROOT" \
    --save_path "$CHECKPOINT_PATH" \
    --mask_root "$MVTEC_PATH/$OBJ/ground_truth" \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --lr "$LR" \
    --log_path "$LOG_DIR" \
    --seed "$SEED" \
    --selection_split "$SELECTION_SPLIT" \
    --final_test_split "$FINAL_TEST_SPLIT" \
    --selection_metric "$LOCALIZATION_SELECTION_METRIC" \
    --extra_real_anomaly_prob "$EXTRA_REAL_ANOMALY_PROB" \
    --result_csv "$LOCALIZATION_RESULT_CSV" \
    --dataset_name "$DATASET_NAME" \
    --gpu_id 0 \
    2>&1 | tee -a "$LOG_FILE"

if [ "$RUN_BEST_CKPT_TEST" = "1" ]; then
    BEST_CHECKPOINT="$CHECKPOINT_PATH/$OBJ.pckl"
    if [ ! -f "$BEST_CHECKPOINT" ]; then
        log "ERROR: missing val-best checkpoint: $BEST_CHECKPOINT"
        exit 1
    fi

    log
    log "============================================================"
    log "Testing val-best localization checkpoint"
    log "============================================================"
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" python eval_downstream/test-localization.py \
        --sample_name "$OBJ" \
        --mvtec_path "$MVTEC_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --split "$BEST_CKPT_TEST_SPLIT" \
        --result_dir "$BEST_CKPT_TEST_DIR" \
        2>&1 | tee -a "$LOG_FILE"
fi

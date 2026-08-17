#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
OBJECT_NAME="${OBJECT_NAME:-hazelnut}"
OBJECT_PROMPT_DEFAULT="a ${OBJECT_NAME//_/ }"

# Prefer an explicit dataset path. A standalone clone defaults to ./datasets;
# no parent-workspace layout is assumed.
if [[ -z "${MVTEC_ROOT:-}" ]]; then
    if [[ -n "${DATASET_ROOT:-}" ]]; then
        MVTEC_ROOT="$(cd "$(dirname "$DATASET_ROOT")" && pwd -P)"
    else
        MVTEC_ROOT="$PROJECT_ROOT/datasets"
    fi
fi
DATASET_ROOT="${DATASET_ROOT:-$MVTEC_ROOT/$OBJECT_NAME}"

T2R_CSV="${T2R_CSV:-$PROJECT_ROOT/configs/block_frequency_t2r.csv}"
T2R_TOP10="${T2R_TOP10:-$PROJECT_ROOT/configs/top10_t2r_blocks.txt}"
ANOMALIES_STR="${ANOMALIES_STR:-crack hole print cut}"
SAMPLES_PER_ANOMALY="${SAMPLES_PER_ANOMALY:-6}"
SAMPLES_PER_PAIR_STR="${SAMPLES_PER_PAIR_STR:-}"
SEED="${SEED:-309}"
RUN_NAME="${RUN_NAME:-${OBJECT_NAME}_rediff_ad}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/$RUN_NAME}"
CONDA_ENV="${CONDA_ENV:-${FLUX_ENV:-rediff-ad}}"

if [[ ! -d "$DATASET_ROOT/train/good" || ! -d "$DATASET_ROOT/test" || ! -d "$DATASET_ROOT/ground_truth" ]]; then
    echo "ERROR: $OBJECT_NAME dataset is incomplete: $DATASET_ROOT" >&2
    echo "Set MVTEC_ROOT=/path/to/mvtec or DATASET_ROOT=/path/to/$OBJECT_NAME." >&2
    exit 1
fi
if [[ ! -s "$T2R_CSV" || ! -s "$T2R_TOP10" ]]; then
    echo "ERROR: missing T2R attention config: CSV=$T2R_CSV Top10=$T2R_TOP10" >&2
    exit 1
fi

read -r -a ANOMALIES <<< "$ANOMALIES_STR"
if [[ -n "${REF_IDS_STR:-}" ]]; then
    read -r -a REF_IDS <<< "$REF_IDS_STR"
else
    REF_IDS=()
    for _anomaly in "${ANOMALIES[@]}"; do
        REF_IDS+=("000")
    done
fi
if [[ "${#ANOMALIES[@]}" -ne "${#REF_IDS[@]}" ]]; then
    echo "ERROR: ANOMALIES_STR and REF_IDS_STR must have the same number of entries." >&2
    exit 1
fi
for pair_index in "${!ANOMALIES[@]}"; do
    _anomaly="${ANOMALIES[$pair_index]}"
    _ref_id="${REF_IDS[$pair_index]}"
    if [[ ! -f "$DATASET_ROOT/test/$_anomaly/${_ref_id}.png" ]]; then
        echo "ERROR: missing reference image: $DATASET_ROOT/test/$_anomaly/${_ref_id}.png" >&2
        exit 1
    fi
    if [[ ! -f "$DATASET_ROOT/ground_truth/$_anomaly/${_ref_id}_mask.png" ]]; then
        echo "ERROR: missing reference mask: $DATASET_ROOT/ground_truth/$_anomaly/${_ref_id}_mask.png" >&2
        exit 1
    fi
done
REF_IDS_STR="${REF_IDS[*]}"
if [[ -n "$SAMPLES_PER_PAIR_STR" ]]; then
    read -r -a SAMPLES_PER_PAIR <<< "$SAMPLES_PER_PAIR_STR"
    if [[ "${#SAMPLES_PER_PAIR[@]}" -ne "${#ANOMALIES[@]}" ]]; then
        echo "ERROR: SAMPLES_PER_PAIR_STR must have one count per anomaly/reference pair." >&2
        exit 1
    fi
fi

shopt -s nullglob
SOURCE_IMAGES=("$DATASET_ROOT/train/good"/*.png "$DATASET_ROOT/train/good"/*.jpg "$DATASET_ROOT/train/good"/*.jpeg)
shopt -u nullglob
if [[ "${#SOURCE_IMAGES[@]}" -eq 0 ]]; then
    echo "ERROR: no source images under $DATASET_ROOT/train/good" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
if [[ "${LOG_TO_FILE:-1}" == "1" ]]; then
    exec > >(tee -a "$OUT_ROOT/run_console.log") 2>&1
fi

echo "[$(date '+%F %T')] REDiff-AD $OBJECT_NAME run"
echo "project:  $PROJECT_ROOT"
echo "dataset:  $DATASET_ROOT"
echo "defects:  $ANOMALIES_STR"
echo "refs:     $REF_IDS_STR"
if [[ -n "$SAMPLES_PER_PAIR_STR" ]]; then
    echo "samples:  per-pair counts=$SAMPLES_PER_PAIR_STR"
else
    echo "samples:  $SAMPLES_PER_ANOMALY per defect"
fi
echo "seed:     $SEED"
echo "output:   $OUT_ROOT"
echo "design:   direct=T2R; adaptive=T2R; Shape-K=middle/all 36; Q80+contour=on"
echo "runtime:  quantize=${FULL_FLUX_QUANTIZE:-int4}; LoRA=${LORA_PATH:-WensongSong/Insert-Anything}/${LORA_WEIGHT_NAME:-20250321_steps5000_pytorch_lora_weights.safetensors}"

# The canonical INT4 random-object profile uses a tighter target-mask area
# (0.5-0.9). Object launchers may explicitly select another audited policy.
exec env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CONDA_ENV="$CONDA_ENV" \
    MVTEC_ROOT="$MVTEC_ROOT" \
    OBJ="$OBJECT_NAME" \
    OBJECT_NAME="$OBJECT_NAME" \
    RUN_OBJECT_SEQUENCE=single \
    RUN_UNLISTED_DEFECTS=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    SOURCE_IMAGE_ROOT="$DATASET_ROOT/train/good" \
    REF_IMAGE_ROOT="$DATASET_ROOT/test" \
    REF_MASK_ROOT="$DATASET_ROOT/ground_truth" \
    RESULT_ROOT="$PROJECT_ROOT/outputs" \
    RUN_NAME="$RUN_NAME" \
    OUT_ROOT="$OUT_ROOT" \
    RUN_LOG="$OUT_ROOT/run_log.csv" \
    ADAPTIVE_LOG="$OUT_ROOT/adaptive_log.csv" \
    ANOMALIES_STR="$ANOMALIES_STR" \
    REF_IDS_STR="$REF_IDS_STR" \
    SAMPLES_PER_ANOMALY="$SAMPLES_PER_ANOMALY" \
    SAMPLES_PER_PAIR_STR="$SAMPLES_PER_PAIR_STR" \
    SEED="$SEED" \
    START_INDEX="${START_INDEX:-0}" \
    SIZE=512 \
    STEPS="${STEPS:-30}" \
    SAVE_STEPS_STR="${SAVE_STEPS_STR:-1 2 3 4 5 6 8 10 12 15 20 25 30}" \
    AGGREGATE_STEPS_STR="${AGGREGATE_STEPS_STR:-10 15 20 25 27 28 29}" \
    ADAPTIVE_CHECK_STEPS_STR="${ADAPTIVE_CHECK_STEPS_STR:-1 2 3 4 5 6 8 10 12 15 20 25 30}" \
    OVERWRITE="${OVERWRITE:-0}" \
    SAVE_DEBUG_FIRST="${SAVE_DEBUG_FIRST:-1}" \
    SAVE_MASK_DEBUG=1 \
    EMPTY_CACHE_EACH_SAMPLE=1 \
    LOG_ATTENTION_STEPS="${LOG_ATTENTION_STEPS:-0}" \
    DRY_RUN="${DRY_RUN:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    ALLOW_CPU_ONLY="${ALLOW_CPU_ONLY:-0}" \
    LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}" \
    FULL_FLUX_QUANTIZE="${FULL_FLUX_QUANTIZE:-int4}" \
    CPU_OFFLOAD="${CPU_OFFLOAD:-1}" \
    SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}" \
    FLUX_FILL_PATH="${FLUX_FILL_PATH:-black-forest-labs/FLUX.1-Fill-dev}" \
    FLUX_REDUX_PATH="${FLUX_REDUX_PATH:-black-forest-labs/FLUX.1-Redux-dev}" \
    LORA_PATH="${LORA_PATH:-${INSERT_ANYTHING_LORA_PATH:-WensongSong/Insert-Anything}}" \
    LORA_WEIGHT_NAME="${LORA_WEIGHT_NAME:-${INSERT_ANYTHING_LORA_WEIGHT:-20250321_steps5000_pytorch_lora_weights.safetensors}}" \
    DIRECT_AGGREGATE_KIND=target_to_ref_image \
    BLOCK_FREQUENCY_CSV="$T2R_CSV" \
    TOP_BLOCKS_FILE="$T2R_TOP10" \
    TOP_K=10 \
    POLARITY=dominant \
    ROI=initial_mask \
    HIST_THRESHOLD_SCALE=0.85 \
    HIST_THRESHOLD_OFFSET=0.0 \
    COMPONENT_MODE=all \
    FILL_HOLES=1 \
    CLOSE_ITERATIONS=1 \
    DILATE_ITERATIONS=1 \
    RUN_PAMR_REFINE=0 \
    PAMR_ITER=500 \
    PAMR_SEED_THRESHOLD=0.30 \
    PAMR_THRESHOLD=0.50 \
    MGAC_ITER=120 \
    MGAC_SMOOTHING=0 \
    MGAC_BALLOON=0.0 \
    MGAC_ROI_DILATE=17 \
    MGAC_EDGE_ALPHA=5.0 \
    MGAC_SCHARR_WEIGHT=0.55 \
    MGAC_WAVELET_WEIGHT=0.45 \
    MGAC_GATE_PERCENTILE=70 \
    MGAC_GATE_DILATE=2 \
    MGAC_INIT_ERODE=2 \
    MGAC_KEEP_COARSE=0 \
    MGAC_USE_EDGE_GATE_AS_ROI=0 \
    MGAC_FINAL_CLOSE=0 \
    MGAC_FINAL_MIN_AREA=80 \
    MGAC_FINAL_FILL_HOLES=0 \
    MGAC_OUTPUT_MODE=edge_gate \
    RUN_Q80_APPEARANCE_REFINE=1 \
    Q80_APPEARANCE_PERCENTILE=80.0 \
    Q80_APPEARANCE_GATE_DILATE=2 \
    Q80_APPEARANCE_MIN_AREA=150 \
    Q80_APPEARANCE_FG_ERODE=1 \
    Q80_APPEARANCE_BG_RING_DILATE=8 \
    Q80_APPEARANCE_KEEP_MARGIN=0.25 \
    Q80_APPEARANCE_GROW_RADIUS=1 \
    Q80_APPEARANCE_ADD_MARGIN=0.15 \
    Q80_APPEARANCE_MAX_FG_DIST=3.0 \
    Q80_APPEARANCE_ROI_DILATE=17 \
    RUN_CONTOUR_REFINE=1 \
    REFINE_COARSE_OPEN=1 \
    REFINE_COARSE_OPEN_KERNEL=3 \
    REFINE_COARSE_OPEN_ITER=1 \
    CONTOUR_REFINE_INNER_ERODE=1 \
    CONTOUR_REFINE_CLIP_TO_COARSE=1 \
    CONTOUR_REFINE_EDGE_DILATE=2 \
    CONTOUR_REFINE_CLOSE=1 \
    CONTOUR_REFINE_FILL_HOLES=1 \
    CONTOUR_REFINE_COMPONENT_MODE=all \
    SAVE_ACTIVE_CONTOUR_MASK=0 \
    SAVE_ACTIVE_CONTOUR_DEBUG=0 \
    SAVE_ACTIVE_CONTOUR_EDGE_MAP=1 \
    ADAPTIVE_REF_INJECTION="${ADAPTIVE_REF_INJECTION:-1}" \
    ADAPTIVE_ROI=initial_mask \
    ADAPTIVE_AGGREGATE_KIND="${ADAPTIVE_AGGREGATE_KIND:-target_to_ref_image}" \
    ADAPTIVE_BLOCK_FREQUENCY_CSV="${ADAPTIVE_BLOCK_FREQUENCY_CSV:-$T2R_CSV}" \
    ADAPTIVE_TOP_BLOCKS_FILE="${ADAPTIVE_TOP_BLOCKS_FILE:-$T2R_TOP10}" \
    ADAPTIVE_POLARITY=dominant \
    ADAPTIVE_SCORE_MODE=aggregate_mask \
    ADAPTIVE_AGGREGATE_SCORE_KIND=inside_outside_contrast \
    ADAPTIVE_AGGREGATE_MIN_INSIDE_RATIO=0.7 \
    ADAPTIVE_AGGREGATE_MIN_INSIDE_MEAN=0.28 \
    ADAPTIVE_AGGREGATE_MIN_AREA_RATIO=0.85 \
    ADAPTIVE_AGGREGATE_MIN_CONTRAST_RATIO=1.5 \
    ADAPTIVE_AGGREGATE_MIN_CONTRAST_INSIDE_MEAN=0.20 \
    ADAPTIVE_AGGREGATE_MIN_CONTRAST_MARGIN=0.10 \
    ADAPTIVE_AGGREGATE_OUTSIDE_RING_DILATE=32 \
    ADAPTIVE_REF_TOKEN_START=512 \
    ADAPTIVE_REF_ATTENTION_THRESHOLD=0.12 \
    ADAPTIVE_REF_BASE_SCALE=3.00 \
    ADAPTIVE_REF_MAX_SCALE=10.0 \
    ADAPTIVE_REF_BOOST=1.7 \
    ADAPTIVE_REF_TRIGGER_MIN_SCALE=8.0 \
    ADAPTIVE_REF_DECAY=0.80 \
    ADAPTIVE_REF_DECAY_MIN_SCORE=1.60 \
    REF_TOKEN_NOISE_STD=0.00 \
    REF_TOKEN_DROPOUT=0.00 \
    REF_TOKEN_SCALE_JITTER=0.00 \
    REF_TOKEN_SPAN_DROPOUT=0.00 \
    REF_TOKEN_SPAN_LEN=2 \
    REF_TOKEN_PERTURB_SEED_OFFSET=700000 \
    REF_AUGMENT_BANK_SIZE=1 \
    REF_AUGMENT_ROTATE=20 \
    REF_AUGMENT_SCALE_JITTER=0.20 \
    REF_AUGMENT_TRANSLATE_RATIO=0.08 \
    REF_AUGMENT_BRIGHTNESS=0.08 \
    REF_AUGMENT_CONTRAST=0.12 \
    REF_AUGMENT_SEED_OFFSET=800000 \
    SHAPE_K_REMOVAL="${SHAPE_K_REMOVAL:-1}" \
    SHAPE_K_ETA=0.30 \
    SHAPE_K_START_RATIO=0.4 \
    SHAPE_K_END_RATIO=0.70 \
    SHAPE_K_START_STEP=-1 \
    SHAPE_K_END_STEP=-1 \
    SHAPE_K_EDGE_METHOD=foreground \
    SHAPE_K_FOREGROUND_THRESHOLD=250 \
    SHAPE_K_BLOCK_SCOPE=middle \
    SHAPE_K_MODE=both \
    SHAPE_K_SUPPRESS_SCALE=0.60 \
    SHAPE_K_BLOCKS_STR= \
    TARGET_MASK_SOURCE="${TARGET_MASK_SOURCE:-random_object}" \
    FIXED_TARGET_MASK="${FIXED_TARGET_MASK:-}" \
    REFERENCE_MASK_DILATE_ITERATIONS="${REFERENCE_MASK_DILATE_ITERATIONS:-5}" \
    REFERENCE_MASK_VERTICAL_SHIFT_RATIO="${REFERENCE_MASK_VERTICAL_SHIFT_RATIO:-0.05}" \
    OBJECT_PROMPT="${OBJECT_PROMPT:-$OBJECT_PROMPT_DEFAULT}" \
    OBJECT_SUPPORT_EROSION=8 \
    RANDOM_MASK_AREA_MIN_RATIO="${RANDOM_MASK_AREA_MIN_RATIO:-0.5}" \
    RANDOM_MASK_AREA_MAX_RATIO="${RANDOM_MASK_AREA_MAX_RATIO:-0.9}" \
    RANDOM_MASK_ROTATE=45 \
    RANDOM_MASK_ATTEMPTS=160 \
    RANDOM_MASK_DOUBLE_PROB="${RANDOM_MASK_DOUBLE_PROB:-0.07}" \
    bash "$PROJECT_ROOT/scripts/run_attention_direct_top10.sh"

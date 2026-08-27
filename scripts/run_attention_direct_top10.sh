#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

# Bash reads long scripts incrementally. Sequence runs can last long enough that
# editing this file mid-run makes the parent process read a mixed old/new file.
# Run sequence mode from a temp snapshot so active jobs are stable.
if [[ "${RUN_SEQUENCE_CHILD:-0}" != "1" && "${RUN_SEQUENCE_SNAPSHOT:-0}" != "1" ]]; then
    SNAPSHOT_PATH="$(mktemp "${TMPDIR:-/tmp}/run_attention_direct_top10.XXXXXX.sh")"
    cp "${BASH_SOURCE[0]}" "$SNAPSHOT_PATH"
    chmod +x "$SNAPSHOT_PATH"
    exec env         PROJECT_ROOT="$PROJECT_ROOT"         RUN_SEQUENCE_SNAPSHOT=1         RUN_SEQUENCE_TEMP_SCRIPT="$SNAPSHOT_PATH"         bash "$SNAPSHOT_PATH"
fi

if [[ "${RUN_SEQUENCE_SNAPSHOT:-0}" == "1" && "${RUN_SEQUENCE_CHILD:-0}" != "1" && -n "${RUN_SEQUENCE_TEMP_SCRIPT:-}" ]]; then
    cleanup_sequence_snapshot() {
        rm -f "$RUN_SEQUENCE_TEMP_SCRIPT"
    }
    trap cleanup_sequence_snapshot EXIT
fi

# ---------------------------------------------------------------------------
# Object presets
# ---------------------------------------------------------------------------
MVTEC_ROOT="${MVTEC_ROOT:-$PROJECT_ROOT/datasets}"
SUPPORTED_OBJECTS_STR="cable capsule grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
DEFAULT_REF_ID="${DEFAULT_REF_ID:-000}"
RUN_UNLISTED_DEFECTS="${RUN_UNLISTED_DEFECTS:-0}"

# Reference image ids for each "<object>/<defect>". Comment out an entry to skip
# that defect. Set RUN_UNLISTED_DEFECTS=1 to fall back to DEFAULT_REF_ID.
declare -A REF_ID_BY_DEFECT=(
    ["capsule/crack"]="001"
    ["capsule/faulty_imprint"]="004"
    ["capsule/poke"]="000"
    ["capsule/scratch"]="019"
    ["capsule/squeeze"]="000"
    ["hazelnut/crack"]="000"
    ["hazelnut/cut"]="000"
    ["hazelnut/hole"]="000"
    ["hazelnut/print"]="000"
    ["leather/color"]="001"
    ["leather/cut"]="000"
    ["leather/fold"]="000"
    ["leather/glue"]="006"
    ["leather/poke"]="007"
    ["metal_nut/bent"]="003"
    ["metal_nut/color"]="000"
    # ["metal_nut/flip"]="000"
    ["metal_nut/scratch"]="006"
    # ["screw/manipulated_front"]="000"
    # ["screw/scratch_head"]="000"
    # ["screw/scratch_neck"]="001"
    # ["screw/thread_side"]="001"
    # ["screw/thread_top"]="000"
    ["tile/crack"]="000"
    ["tile/glue_strip"]="010"
    ["tile/gray_stroke"]="004"
    ["tile/oil"]="001"
    ["tile/rough"]="003"
    ["toothbrush/defective"]="005"
    ["transistor/bent_lead"]="002"
    ["transistor/cut_lead"]="001"
    ["transistor/damaged_case"]="000"
    ["transistor/misplaced"]="002"
    ["wood/color"]="001"
    ["wood/combined"]="001"
    ["wood/hole"]="000"
    ["wood/liquid"]="000"
    ["wood/scratch"]="003"
)

# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------
object_prompt_from_name() {
    local object_name="$1"
    printf 'a %s' "${object_name//_/ }"
}

resolve_object_config() {
    local object_name="$1"
    case " $SUPPORTED_OBJECTS_STR " in
        *" $object_name "*) ;;
        *)
            echo "ERROR: unsupported OBJ: $object_name" >&2
            echo "       Supported: $SUPPORTED_OBJECTS_STR" >&2
            exit 1
            ;;
    esac

    OBJECT_DATASET_ROOT="$MVTEC_ROOT/$object_name"
    OBJECT_RUN_NAME="250ps-result_${object_name}-final"
    OBJECT_PROMPT_DEFAULT="$(object_prompt_from_name "$object_name")"

    local test_root="$OBJECT_DATASET_ROOT/test"
    if [[ ! -d "$test_root" ]]; then
        echo "ERROR: missing MVTec test directory for OBJ=$object_name: $test_root" >&2
        exit 1
    fi

    local anomalies=()
    local ref_ids=()
    local anomaly_name key
    while IFS= read -r anomaly_name; do
        key="$object_name/$anomaly_name"
        if [[ -n "${REF_ID_BY_DEFECT[$key]+_}" ]]; then
            anomalies+=("$anomaly_name")
            ref_ids+=("${REF_ID_BY_DEFECT[$key]}")
        elif [[ "$RUN_UNLISTED_DEFECTS" == "1" || "$RUN_UNLISTED_DEFECTS" == "true" ]]; then
            anomalies+=("$anomaly_name")
            ref_ids+=("$DEFAULT_REF_ID")
        else
            echo "skip unlisted defect: $key" >&2
        fi
    done < <(find "$test_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | grep -v '^good$')

    if [[ "${#anomalies[@]}" -eq 0 ]]; then
        echo "skip object with no enabled defects: $object_name" >&2
        return 2
    fi

    OBJECT_ANOMALIES_STR="${anomalies[*]}"
    OBJECT_REF_IDS_STR="${ref_ids[*]}"
}

run_sequence_object() {
    local sequence_object="$1"
    local script_path="$2"

    if ! resolve_object_config "$sequence_object"; then
        return 0
    fi
    echo "============================================================"
    echo "Sequence item: $sequence_object"
    echo "============================================================"
    env \
        RUN_SEQUENCE_CHILD=1 \
        OBJ="$sequence_object" \
        DATASET_ROOT="$OBJECT_DATASET_ROOT" \
        RUN_NAME="$OBJECT_RUN_NAME" \
        ANOMALIES_STR="$OBJECT_ANOMALIES_STR" \
        REF_IDS_STR="$OBJECT_REF_IDS_STR" \
        OBJECT_PROMPT="$OBJECT_PROMPT_DEFAULT" \
        bash "$script_path"
}

# ---------------------------------------------------------------------------
# Object sequence driver
# ---------------------------------------------------------------------------
OBJ="${OBJ:-all}"
RUN_OBJECT_SEQUENCE="${RUN_OBJECT_SEQUENCE:-${OBJECTS_STR:-$OBJ}}"
if [[ "$RUN_OBJECT_SEQUENCE" == "all" ]]; then
    RUN_OBJECT_SEQUENCE="$SUPPORTED_OBJECTS_STR"
fi
if [[ "${RUN_SEQUENCE_CHILD:-0}" != "1" && "$RUN_OBJECT_SEQUENCE" != "single" ]]; then
    SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
    read -r -a SEQUENCE_OBJECTS <<< "$RUN_OBJECT_SEQUENCE"
    for SEQUENCE_OBJECT in "${SEQUENCE_OBJECTS[@]}"; do
        run_sequence_object "$SEQUENCE_OBJECT" "$SCRIPT_PATH"
    done
    exit 0
fi

OBJECT_NAME="${OBJECT_NAME:-${OBJ%% *}}"
if ! resolve_object_config "$OBJECT_NAME"; then
    exit 1
fi

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
CLEAR_CUDA_VISIBLE_DEVICES="${CLEAR_CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CLEAR_CUDA_VISIBLE_DEVICES" == "1" || "$CLEAR_CUDA_VISIBLE_DEVICES" == "true" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

CONDA_ENV="${CONDA_ENV:-${FLUX_ENV:-rediff-ad}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-false}"
if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
    LOCAL_FILES_ARGS=(--local-files-only)
else
    LOCAL_FILES_ARGS=(--no-local-files-only)
fi

DEVICE="${DEVICE:-cuda}"
ALLOW_CPU_ONLY="${ALLOW_CPU_ONLY:-0}"
FULL_FLUX_QUANTIZE="${FULL_FLUX_QUANTIZE:-int4}"
QUANT_ARGS=(--full-flux-quantize "$FULL_FLUX_QUANTIZE")
FLUX_FILL_PATH="${FLUX_FILL_PATH:-black-forest-labs/FLUX.1-Fill-dev}"
FLUX_REDUX_PATH="${FLUX_REDUX_PATH:-black-forest-labs/FLUX.1-Redux-dev}"
LORA_PATH="${LORA_PATH:-${IN_CONTEXT_LORA_PATH:-WensongSong/Insert-Anything}}"
LORA_WEIGHT_NAME="${LORA_WEIGHT_NAME:-${IN_CONTEXT_LORA_WEIGHT:-20250321_steps5000_pytorch_lora_weights.safetensors}}"

# ---------------------------------------------------------------------------
# Dataset and outputs
# ---------------------------------------------------------------------------
DATASET_ROOT="${DATASET_ROOT:-$OBJECT_DATASET_ROOT}"
SOURCE_IMAGE_ROOT="${SOURCE_IMAGE_ROOT:-$DATASET_ROOT/train/good}"
REF_IMAGE_ROOT="${REF_IMAGE_ROOT:-$DATASET_ROOT/test}"
REF_MASK_ROOT="${REF_MASK_ROOT:-$DATASET_ROOT/ground_truth}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/result}"
RUN_NAME="${RUN_NAME:-$OBJECT_RUN_NAME}"
OUT_ROOT="${OUT_ROOT:-$RESULT_ROOT/proposal_result/$RUN_NAME}"
RUN_LOG="${RUN_LOG:-$OUT_ROOT/run_log.csv}"
ADAPTIVE_LOG="${ADAPTIVE_LOG:-$OUT_ROOT/adaptive_log.csv}"

# 異常種類
ANOMALIES_STR="${ANOMALIES_STR:-$OBJECT_ANOMALIES_STR}"
REF_IDS_STR="${REF_IDS_STR:-$OBJECT_REF_IDS_STR}"
read -r -a ANOMALIES <<< "$ANOMALIES_STR"
read -r -a REF_IDS <<< "$REF_IDS_STR"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SEED="${SEED:-714}"
SIZE="${SIZE:-512}"
SAMPLES_PER_ANOMALY="${SAMPLES_PER_ANOMALY:-250}"
SAMPLES_PER_PAIR_STR="${SAMPLES_PER_PAIR_STR:-}"
SAMPLES_PER_PAIR_ARGS=()
if [[ -n "$SAMPLES_PER_PAIR_STR" ]]; then
    read -r -a SAMPLES_PER_PAIR <<< "$SAMPLES_PER_PAIR_STR"
    if [[ "${#SAMPLES_PER_PAIR[@]}" -ne "${#ANOMALIES[@]}" ]]; then
        echo "ERROR: SAMPLES_PER_PAIR_STR must match the anomaly/reference pair count." >&2
        exit 1
    fi
    SAMPLES_PER_PAIR_ARGS=(--samples-per-pair "${SAMPLES_PER_PAIR[@]}")
fi
START_INDEX="${START_INDEX:-0}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_DEBUG_FIRST="${SAVE_DEBUG_FIRST:-0}"
EMPTY_CACHE_EACH_SAMPLE="${EMPTY_CACHE_EACH_SAMPLE:-1}"
LOG_ATTENTION_STEPS="${LOG_ATTENTION_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
STEPS="${STEPS:-30}"
SAVE_STEPS_STR="${SAVE_STEPS_STR:-1 2 3 4 5 6 8 10 12 15 20 25 30}"
AGGREGATE_STEPS_STR="${AGGREGATE_STEPS_STR:-10 15 20 25 27 28 29}"
ADAPTIVE_CHECK_STEPS_STR="${ADAPTIVE_CHECK_STEPS_STR:-$SAVE_STEPS_STR}"
read -r -a SAVE_STEPS <<< "$SAVE_STEPS_STR"
read -r -a AGGREGATE_STEPS <<< "$AGGREGATE_STEPS_STR"
read -r -a ADAPTIVE_CHECK_STEPS <<< "$ADAPTIVE_CHECK_STEPS_STR"

# ---------------------------------------------------------------------------
# Attention aggregation
# ---------------------------------------------------------------------------
BLOCK_FREQUENCY_CSV="${BLOCK_FREQUENCY_CSV:-$PROJECT_ROOT/configs/block_frequency_t2r.csv}"
TOP_BLOCKS_FILE="${TOP_BLOCKS_FILE:-}"
TOP_K="${TOP_K:-10}"
DIRECT_AGGREGATE_KIND="${DIRECT_AGGREGATE_KIND:-target_to_ref_image}"
POLARITY="${POLARITY:-dominant}"
ROI="${ROI:-initial_mask}"
HIST_THRESHOLD_SCALE="${HIST_THRESHOLD_SCALE:-0.85}"
HIST_THRESHOLD_OFFSET="${HIST_THRESHOLD_OFFSET:-0.0}"
COMPONENT_MODE="${COMPONENT_MODE:-all}"
FILL_HOLES="${FILL_HOLES:-1}"
CLOSE_ITERATIONS="${CLOSE_ITERATIONS:-1}"
DILATE_ITERATIONS="${DILATE_ITERATIONS:-1}"

# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------
RUN_PAMR_REFINE="${RUN_PAMR_REFINE:-0}"
PAMR_ITER="${PAMR_ITER:-500}"
PAMR_SEED_THRESHOLD="${PAMR_SEED_THRESHOLD:-0.30}"
PAMR_THRESHOLD="${PAMR_THRESHOLD:-0.50}"
MGAC_ITER="${MGAC_ITER:-120}"
MGAC_SMOOTHING="${MGAC_SMOOTHING:-0}"
MGAC_BALLOON="${MGAC_BALLOON:-0.0}"
MGAC_ROI_DILATE="${MGAC_ROI_DILATE:-17}"
MGAC_EDGE_ALPHA="${MGAC_EDGE_ALPHA:-5.0}"
MGAC_SCHARR_WEIGHT="${MGAC_SCHARR_WEIGHT:-0.55}"
MGAC_WAVELET_WEIGHT="${MGAC_WAVELET_WEIGHT:-0.45}"
MGAC_GATE_PERCENTILE="${MGAC_GATE_PERCENTILE:-70}"
MGAC_GATE_DILATE="${MGAC_GATE_DILATE:-2}"
MGAC_INIT_ERODE="${MGAC_INIT_ERODE:-2}"
MGAC_KEEP_COARSE="${MGAC_KEEP_COARSE:-0}"
MGAC_USE_EDGE_GATE_AS_ROI="${MGAC_USE_EDGE_GATE_AS_ROI:-0}"
MGAC_FINAL_CLOSE="${MGAC_FINAL_CLOSE:-0}"
MGAC_FINAL_MIN_AREA="${MGAC_FINAL_MIN_AREA:-80}"
MGAC_FINAL_FILL_HOLES="${MGAC_FINAL_FILL_HOLES:-0}"
MGAC_OUTPUT_MODE="${MGAC_OUTPUT_MODE:-edge_gate}"
RUN_Q80_APPEARANCE_REFINE="${RUN_Q80_APPEARANCE_REFINE:-1}"
Q80_APPEARANCE_PERCENTILE="${Q80_APPEARANCE_PERCENTILE:-80.0}"
Q80_APPEARANCE_GATE_DILATE="${Q80_APPEARANCE_GATE_DILATE:-2}"
Q80_APPEARANCE_MIN_AREA="${Q80_APPEARANCE_MIN_AREA:-150}"
Q80_APPEARANCE_FG_ERODE="${Q80_APPEARANCE_FG_ERODE:-1}"
Q80_APPEARANCE_BG_RING_DILATE="${Q80_APPEARANCE_BG_RING_DILATE:-8}"
Q80_APPEARANCE_KEEP_MARGIN="${Q80_APPEARANCE_KEEP_MARGIN:-0.25}"
Q80_APPEARANCE_GROW_RADIUS="${Q80_APPEARANCE_GROW_RADIUS:-1}"
Q80_APPEARANCE_ADD_MARGIN="${Q80_APPEARANCE_ADD_MARGIN:-0.15}"
Q80_APPEARANCE_MAX_FG_DIST="${Q80_APPEARANCE_MAX_FG_DIST:-3.0}"
Q80_APPEARANCE_ROI_DILATE="${Q80_APPEARANCE_ROI_DILATE:-$MGAC_ROI_DILATE}"
RUN_CONTOUR_REFINE="${RUN_CONTOUR_REFINE:-1}"
REFINE_COARSE_OPEN="${REFINE_COARSE_OPEN:-1}"
REFINE_COARSE_OPEN_KERNEL="${REFINE_COARSE_OPEN_KERNEL:-3}"
REFINE_COARSE_OPEN_ITER="${REFINE_COARSE_OPEN_ITER:-1}"
CONTOUR_REFINE_INNER_ERODE="${CONTOUR_REFINE_INNER_ERODE:-1}"
CONTOUR_REFINE_CLIP_TO_COARSE="${CONTOUR_REFINE_CLIP_TO_COARSE:-1}"
CONTOUR_REFINE_EDGE_DILATE="${CONTOUR_REFINE_EDGE_DILATE:-2}"
CONTOUR_REFINE_CLOSE="${CONTOUR_REFINE_CLOSE:-1}"
CONTOUR_REFINE_FILL_HOLES="${CONTOUR_REFINE_FILL_HOLES:-1}"
CONTOUR_REFINE_COMPONENT_MODE="${CONTOUR_REFINE_COMPONENT_MODE:-all}"
SAVE_ACTIVE_CONTOUR_MASK="${SAVE_ACTIVE_CONTOUR_MASK:-0}"
SAVE_ACTIVE_CONTOUR_DEBUG="${SAVE_ACTIVE_CONTOUR_DEBUG:-0}"
SAVE_ACTIVE_CONTOUR_EDGE_MAP="${SAVE_ACTIVE_CONTOUR_EDGE_MAP:-1}"
SAVE_MASK_DEBUG="${SAVE_MASK_DEBUG:-1}"
export RUN_PAMR_REFINE PAMR_ITER PAMR_SEED_THRESHOLD PAMR_THRESHOLD MGAC_ITER MGAC_SMOOTHING MGAC_BALLOON MGAC_ROI_DILATE MGAC_EDGE_ALPHA MGAC_SCHARR_WEIGHT MGAC_WAVELET_WEIGHT MGAC_GATE_PERCENTILE MGAC_GATE_DILATE MGAC_INIT_ERODE MGAC_KEEP_COARSE MGAC_USE_EDGE_GATE_AS_ROI MGAC_FINAL_CLOSE MGAC_FINAL_MIN_AREA MGAC_FINAL_FILL_HOLES MGAC_OUTPUT_MODE RUN_Q80_APPEARANCE_REFINE Q80_APPEARANCE_PERCENTILE Q80_APPEARANCE_GATE_DILATE Q80_APPEARANCE_MIN_AREA Q80_APPEARANCE_FG_ERODE Q80_APPEARANCE_BG_RING_DILATE Q80_APPEARANCE_KEEP_MARGIN Q80_APPEARANCE_GROW_RADIUS Q80_APPEARANCE_ADD_MARGIN Q80_APPEARANCE_MAX_FG_DIST Q80_APPEARANCE_ROI_DILATE RUN_CONTOUR_REFINE REFINE_COARSE_OPEN REFINE_COARSE_OPEN_KERNEL REFINE_COARSE_OPEN_ITER CONTOUR_REFINE_INNER_ERODE CONTOUR_REFINE_CLIP_TO_COARSE CONTOUR_REFINE_EDGE_DILATE CONTOUR_REFINE_CLOSE CONTOUR_REFINE_FILL_HOLES CONTOUR_REFINE_COMPONENT_MODE SAVE_ACTIVE_CONTOUR_MASK SAVE_ACTIVE_CONTOUR_DEBUG SAVE_ACTIVE_CONTOUR_EDGE_MAP

# ---------------------------------------------------------------------------
# Adaptive reference injection
# ---------------------------------------------------------------------------
ADAPTIVE_REF_INJECTION="${ADAPTIVE_REF_INJECTION:-1}"
ADAPTIVE_ROI="${ADAPTIVE_ROI:-initial_mask}"
ADAPTIVE_AGGREGATE_KIND="${ADAPTIVE_AGGREGATE_KIND:-target_to_ref_image}"
ADAPTIVE_BLOCK_FREQUENCY_CSV="${ADAPTIVE_BLOCK_FREQUENCY_CSV:-$PROJECT_ROOT/configs/block_frequency_t2r.csv}"
ADAPTIVE_TOP_BLOCKS_FILE="${ADAPTIVE_TOP_BLOCKS_FILE:-$PROJECT_ROOT/configs/top10_t2r_blocks.txt}"
ADAPTIVE_POLARITY="${ADAPTIVE_POLARITY:-dominant}"
ADAPTIVE_SCORE_MODE="${ADAPTIVE_SCORE_MODE:-aggregate_mask}" # ref_condition_mass
ADAPTIVE_AGGREGATE_SCORE_KIND="${ADAPTIVE_AGGREGATE_SCORE_KIND:-inside_outside_contrast}" # inside_coverage, inside_mean, coarse_area_ratio, shape_strict
ADAPTIVE_AGGREGATE_MIN_INSIDE_RATIO="${ADAPTIVE_AGGREGATE_MIN_INSIDE_RATIO:-0.7}"
ADAPTIVE_AGGREGATE_MIN_INSIDE_MEAN="${ADAPTIVE_AGGREGATE_MIN_INSIDE_MEAN:-0.28}"
ADAPTIVE_AGGREGATE_MIN_AREA_RATIO="${ADAPTIVE_AGGREGATE_MIN_AREA_RATIO:-0.85}"
ADAPTIVE_AGGREGATE_MIN_CONTRAST_RATIO="${ADAPTIVE_AGGREGATE_MIN_CONTRAST_RATIO:-1.5}"
ADAPTIVE_AGGREGATE_MIN_CONTRAST_INSIDE_MEAN="${ADAPTIVE_AGGREGATE_MIN_CONTRAST_INSIDE_MEAN:-0.20}"
ADAPTIVE_AGGREGATE_MIN_CONTRAST_MARGIN="${ADAPTIVE_AGGREGATE_MIN_CONTRAST_MARGIN:-0.10}"
ADAPTIVE_AGGREGATE_OUTSIDE_RING_DILATE="${ADAPTIVE_AGGREGATE_OUTSIDE_RING_DILATE:-32}"
ADAPTIVE_REF_TOKEN_START="${ADAPTIVE_REF_TOKEN_START:-512}"
ADAPTIVE_REF_ATTENTION_THRESHOLD="${ADAPTIVE_REF_ATTENTION_THRESHOLD:-0.12}"
ADAPTIVE_REF_BASE_SCALE="${ADAPTIVE_REF_BASE_SCALE:-3.00}"
ADAPTIVE_REF_MAX_SCALE="${ADAPTIVE_REF_MAX_SCALE:-10.0}"
ADAPTIVE_REF_BOOST="${ADAPTIVE_REF_BOOST:-1.7}"
ADAPTIVE_REF_TRIGGER_MIN_SCALE="${ADAPTIVE_REF_TRIGGER_MIN_SCALE:-8.0}"
ADAPTIVE_REF_DECAY="${ADAPTIVE_REF_DECAY:-0.80}"
ADAPTIVE_REF_DECAY_MIN_SCORE="${ADAPTIVE_REF_DECAY_MIN_SCORE:-1.60}"

# ---------------------------------------------------------------------------
# Reference diversity controls
# ---------------------------------------------------------------------------
REF_TOKEN_NOISE_STD="${REF_TOKEN_NOISE_STD:-0.00}"
REF_TOKEN_DROPOUT="${REF_TOKEN_DROPOUT:-0.00}"
REF_TOKEN_SCALE_JITTER="${REF_TOKEN_SCALE_JITTER:-0.00}"
REF_TOKEN_SPAN_DROPOUT="${REF_TOKEN_SPAN_DROPOUT:-0.00}"
REF_TOKEN_SPAN_LEN="${REF_TOKEN_SPAN_LEN:-2}"
REF_TOKEN_PERTURB_SEED_OFFSET="${REF_TOKEN_PERTURB_SEED_OFFSET:-700000}"

REF_AUGMENT_BANK_SIZE="${REF_AUGMENT_BANK_SIZE:-1}"
REF_AUGMENT_ROTATE="${REF_AUGMENT_ROTATE:-20}"
REF_AUGMENT_SCALE_JITTER="${REF_AUGMENT_SCALE_JITTER:-0.20}"
REF_AUGMENT_TRANSLATE_RATIO="${REF_AUGMENT_TRANSLATE_RATIO:-0.08}"
REF_AUGMENT_BRIGHTNESS="${REF_AUGMENT_BRIGHTNESS:-0.08}"
REF_AUGMENT_CONTRAST="${REF_AUGMENT_CONTRAST:-0.12}"
REF_AUGMENT_SEED_OFFSET="${REF_AUGMENT_SEED_OFFSET:-800000}"

# ---------------------------------------------------------------------------
# Shape-K diversity
# ---------------------------------------------------------------------------
SHAPE_K_REMOVAL="${SHAPE_K_REMOVAL:-1}"
SHAPE_K_ETA="${SHAPE_K_ETA:-0.4}"
SHAPE_K_START_RATIO="${SHAPE_K_START_RATIO:-0.4}"
SHAPE_K_END_RATIO="${SHAPE_K_END_RATIO:-0.70}"
SHAPE_K_START_STEP="${SHAPE_K_START_STEP:--1}"
SHAPE_K_END_STEP="${SHAPE_K_END_STEP:--1}"
SHAPE_K_EDGE_METHOD="${SHAPE_K_EDGE_METHOD:-foreground}"
SHAPE_K_FOREGROUND_THRESHOLD="${SHAPE_K_FOREGROUND_THRESHOLD:-250}"
SHAPE_K_BLOCK_SCOPE="${SHAPE_K_BLOCK_SCOPE:-middle}"
SHAPE_K_MODE="${SHAPE_K_MODE:-both}"
SHAPE_K_SUPPRESS_SCALE="${SHAPE_K_SUPPRESS_SCALE:-0.70}"
SHAPE_K_BLOCKS_STR="${SHAPE_K_BLOCKS_STR:-}"

# ---------------------------------------------------------------------------
# Target mask
# ---------------------------------------------------------------------------
TARGET_MASK_SOURCE="${TARGET_MASK_SOURCE:-random_object}"
FIXED_TARGET_MASK="${FIXED_TARGET_MASK:-}"
REFERENCE_MASK_DILATE_ITERATIONS="${REFERENCE_MASK_DILATE_ITERATIONS:-5}"
REFERENCE_MASK_VERTICAL_SHIFT_RATIO="${REFERENCE_MASK_VERTICAL_SHIFT_RATIO:-0.05}"
OBJECT_PROMPT="${OBJECT_PROMPT:-$OBJECT_PROMPT_DEFAULT}"
OBJECT_SUPPORT_EROSION="${OBJECT_SUPPORT_EROSION:-8}"
RANDOM_MASK_AREA_MIN_RATIO="${RANDOM_MASK_AREA_MIN_RATIO:-0.40}"
RANDOM_MASK_AREA_MAX_RATIO="${RANDOM_MASK_AREA_MAX_RATIO:-2.5}"
RANDOM_MASK_ROTATE="${RANDOM_MASK_ROTATE:-45}"
RANDOM_MASK_ATTEMPTS="${RANDOM_MASK_ATTEMPTS:-160}"
RANDOM_MASK_DOUBLE_PROB="${RANDOM_MASK_DOUBLE_PROB:-0.07}"

# ---------------------------------------------------------------------------
# Argument assembly
# ---------------------------------------------------------------------------
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}"
OFFLOAD_ARGS=()
if [[ "$SEQUENTIAL_CPU_OFFLOAD" == "1" ]]; then
    OFFLOAD_ARGS+=(--no-cpu-offload --sequential-cpu-offload)
elif [[ "$CPU_OFFLOAD" == "1" ]]; then
    OFFLOAD_ARGS+=(--cpu-offload)
else
    OFFLOAD_ARGS+=(--no-cpu-offload)
fi

FILL_ARGS=(--direct-fill-holes)
if [[ "$FILL_HOLES" == "0" ]]; then
    FILL_ARGS=(--no-direct-fill-holes)
fi

TARGET_MASK_ARGS=(--target-mask-source random_object)
case "$TARGET_MASK_SOURCE" in
    random_object)
        ;;
    reference_vertical_mixed)
        TARGET_MASK_ARGS=(
            --target-mask-source reference_vertical_mixed
            --reference-mask-dilate-iterations "$REFERENCE_MASK_DILATE_ITERATIONS"
            --reference-mask-vertical-shift-ratio "$REFERENCE_MASK_VERTICAL_SHIFT_RATIO"
        )
        ;;
    fixed)
        if [[ -z "$FIXED_TARGET_MASK" || ! -f "$FIXED_TARGET_MASK" ]]; then
            echo "ERROR: TARGET_MASK_SOURCE=fixed requires an existing FIXED_TARGET_MASK file." >&2
            exit 1
        fi
        TARGET_MASK_ARGS=(--target-mask-source fixed --fixed-target-mask "$FIXED_TARGET_MASK")
        ;;
    *)
        echo "ERROR: unsupported TARGET_MASK_SOURCE=$TARGET_MASK_SOURCE" >&2
        exit 1
        ;;
esac

ADAPTIVE_ARGS=(
    --adaptive-score-mode "$ADAPTIVE_SCORE_MODE"
    --adaptive-aggregate-score-kind "$ADAPTIVE_AGGREGATE_SCORE_KIND"
    --adaptive-aggregate-min-inside-ratio "$ADAPTIVE_AGGREGATE_MIN_INSIDE_RATIO"
    --adaptive-aggregate-min-inside-mean "$ADAPTIVE_AGGREGATE_MIN_INSIDE_MEAN"
    --adaptive-aggregate-min-area-ratio "$ADAPTIVE_AGGREGATE_MIN_AREA_RATIO"
    --adaptive-aggregate-min-contrast-ratio "$ADAPTIVE_AGGREGATE_MIN_CONTRAST_RATIO"
    --adaptive-aggregate-min-contrast-inside-mean "$ADAPTIVE_AGGREGATE_MIN_CONTRAST_INSIDE_MEAN"
    --adaptive-aggregate-min-contrast-margin "$ADAPTIVE_AGGREGATE_MIN_CONTRAST_MARGIN"
    --adaptive-aggregate-outside-ring-dilate "$ADAPTIVE_AGGREGATE_OUTSIDE_RING_DILATE"
    --adaptive-ref-token-start "$ADAPTIVE_REF_TOKEN_START"
    --adaptive-ref-attention-threshold "$ADAPTIVE_REF_ATTENTION_THRESHOLD"
    --adaptive-ref-base-scale "$ADAPTIVE_REF_BASE_SCALE"
    --adaptive-ref-max-scale "$ADAPTIVE_REF_MAX_SCALE"
    --adaptive-ref-boost "$ADAPTIVE_REF_BOOST"
    --adaptive-ref-trigger-min-scale "$ADAPTIVE_REF_TRIGGER_MIN_SCALE"
    --adaptive-ref-decay "$ADAPTIVE_REF_DECAY"
    --adaptive-ref-decay-min-score "$ADAPTIVE_REF_DECAY_MIN_SCORE"
)
if [[ "$ADAPTIVE_REF_INJECTION" == "1" || "$ADAPTIVE_REF_INJECTION" == "true" ]]; then
    ADAPTIVE_ARGS+=(--adaptive-ref-injection)
else
    ADAPTIVE_ARGS+=(--no-adaptive-ref-injection)
fi

REF_TOKEN_PERTURB_ARGS=(
    --ref-token-noise-std "$REF_TOKEN_NOISE_STD"
    --ref-token-dropout "$REF_TOKEN_DROPOUT"
    --ref-token-scale-jitter "$REF_TOKEN_SCALE_JITTER"
    --ref-token-span-dropout "$REF_TOKEN_SPAN_DROPOUT"
    --ref-token-span-len "$REF_TOKEN_SPAN_LEN"
    --ref-token-perturb-seed-offset "$REF_TOKEN_PERTURB_SEED_OFFSET"
)

REF_AUGMENT_ARGS=(
    --ref-augment-bank-size "$REF_AUGMENT_BANK_SIZE"
    --ref-augment-rotate "$REF_AUGMENT_ROTATE"
    --ref-augment-scale-jitter "$REF_AUGMENT_SCALE_JITTER"
    --ref-augment-translate-ratio "$REF_AUGMENT_TRANSLATE_RATIO"
    --ref-augment-brightness "$REF_AUGMENT_BRIGHTNESS"
    --ref-augment-contrast "$REF_AUGMENT_CONTRAST"
    --ref-augment-seed-offset "$REF_AUGMENT_SEED_OFFSET"
)

SHAPE_K_ARGS=(
    --shape-k-eta "$SHAPE_K_ETA"
    --shape-k-start-ratio "$SHAPE_K_START_RATIO"
    --shape-k-end-ratio "$SHAPE_K_END_RATIO"
    --shape-k-start-step "$SHAPE_K_START_STEP"
    --shape-k-end-step "$SHAPE_K_END_STEP"
    --shape-k-block-scope "$SHAPE_K_BLOCK_SCOPE"
    --shape-k-mode "$SHAPE_K_MODE"
    --shape-k-suppress-scale "$SHAPE_K_SUPPRESS_SCALE"
    --shape-k-edge-method "$SHAPE_K_EDGE_METHOD"
    --shape-k-foreground-threshold "$SHAPE_K_FOREGROUND_THRESHOLD"
)
if [[ "$SHAPE_K_REMOVAL" == "1" || "$SHAPE_K_REMOVAL" == "true" ]]; then
    SHAPE_K_ARGS+=(--shape-k-removal)
else
    SHAPE_K_ARGS+=(--no-shape-k-removal)
fi
if [[ -n "$SHAPE_K_BLOCKS_STR" ]]; then
    read -r -a SHAPE_K_BLOCKS <<< "$SHAPE_K_BLOCKS_STR"
    SHAPE_K_ARGS+=(--shape-k-blocks "${SHAPE_K_BLOCKS[@]}")
fi


OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

DEBUG_ARGS=()
if [[ "$SAVE_DEBUG_FIRST" == "1" ]]; then
    DEBUG_ARGS+=(--save-debug-first)
fi
if [[ "$SAVE_MASK_DEBUG" == "1" || "$SAVE_MASK_DEBUG" == "true" ]]; then
    DEBUG_ARGS+=(--save-mask-debug)
else
    DEBUG_ARGS+=(--no-save-mask-debug)
fi
if [[ "$LOG_ATTENTION_STEPS" == "1" ]]; then
    DEBUG_ARGS+=(--log-attention-steps)
fi
if [[ "$EMPTY_CACHE_EACH_SAMPLE" == "1" ]]; then
    DEBUG_ARGS+=(--empty-cache-each-sample)
else
    DEBUG_ARGS+=(--no-empty-cache-each-sample)
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
CUDA_AVAILABLE="not_checked"
CUDA_REASON="not checked (DRY_RUN)"
if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" ]]; then
    CUDA_CHECK_OUTPUT="$(conda run --no-capture-output -n "$CONDA_ENV" python -c 'import torch; ok=False; reason=""
try:
    if not torch.cuda.is_available():
        reason="torch.cuda.is_available() is False"
    else:
        torch.cuda.current_device(); torch.empty(1, device="cuda"); ok=True; reason="ok"
except Exception as exc:
    reason=f"{type(exc).__name__}: {exc}"
print("1" if ok else "0")
print(reason)' 2>/dev/null || printf '0\nconda CUDA check failed\n')"
    CUDA_AVAILABLE="$(printf '%s
' "$CUDA_CHECK_OUTPUT" | sed -n '1p')"
    CUDA_REASON="$(printf '%s
' "$CUDA_CHECK_OUTPUT" | sed -n '2p')"
    if [[ "$DEVICE" == cuda* && "$CUDA_AVAILABLE" != "1" ]]; then
        if [[ "$ALLOW_CPU_ONLY" == "1" || "$ALLOW_CPU_ONLY" == "true" ]]; then
            echo "WARN: PyTorch in conda env '$CONDA_ENV' cannot initialize CUDA; forcing CPU smoke-test mode." >&2
            DEVICE="cpu"
            FULL_FLUX_QUANTIZE="none"
            QUANT_ARGS=(--full-flux-quantize "$FULL_FLUX_QUANTIZE")
            OFFLOAD_ARGS=(--no-cpu-offload)
        else
            echo "ERROR: PyTorch in conda env '$CONDA_ENV' cannot initialize CUDA." >&2
            echo "       PyTorch CUDA check: $CUDA_REASON" >&2
            echo "       Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} clear_cuda_visible_devices=$CLEAR_CUDA_VISIBLE_DEVICES" >&2
            echo "       Check with:" >&2
            echo "         conda run --no-capture-output -n $CONDA_ENV python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'" >&2
            echo "       For CPU smoke tests only, use: ALLOW_CPU_ONLY=1 DEVICE=cpu FULL_FLUX_QUANTIZE=none CPU_OFFLOAD=0 SEQUENTIAL_CPU_OFFLOAD=0" >&2
            exit 1
        fi
    fi
    if [[ "$FULL_FLUX_QUANTIZE" != "none" && "$CUDA_AVAILABLE" != "1" ]]; then
        echo "ERROR: FULL_FLUX_QUANTIZE=$FULL_FLUX_QUANTIZE requires a working PyTorch CUDA runtime. Check: $CUDA_REASON" >&2
        exit 1
    fi
fi

if [[ ! -d "$SOURCE_IMAGE_ROOT" || ! -d "$REF_IMAGE_ROOT" || ! -d "$REF_MASK_ROOT" ]]; then
    echo "ERROR: missing dataset path(s) under $DATASET_ROOT" >&2
    exit 1
fi
if [[ ! -f "$BLOCK_FREQUENCY_CSV" ]]; then
    echo "ERROR: missing block frequency CSV: $BLOCK_FREQUENCY_CSV" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
RESET_LOGS="${RESET_LOGS:-$OVERWRITE}"
if [[ "$RESET_LOGS" == "1" || "$RESET_LOGS" == "true" ]]; then
    : > "$RUN_LOG"
    : > "$ADAPTIVE_LOG"
else
    touch "$RUN_LOG" "$ADAPTIVE_LOG"
fi
if [[ -z "$TOP_BLOCKS_FILE" ]]; then
    TOP_BLOCKS_FILE="$OUT_ROOT/top${TOP_K}_blocks_from_block_frequency.txt"
    awk -F, -v top="$TOP_K" 'NR > 1 && $2 != "" {print $2; count++; if (count >= top) exit}' "$BLOCK_FREQUENCY_CSV" > "$TOP_BLOCKS_FILE"
fi
if [[ ! -s "$TOP_BLOCKS_FILE" ]]; then
    echo "ERROR: no top layers found in $TOP_BLOCKS_FILE" >&2
    exit 1
fi
if [[ ! -f "$ADAPTIVE_BLOCK_FREQUENCY_CSV" ]]; then
    echo "ERROR: missing adaptive block frequency CSV: $ADAPTIVE_BLOCK_FREQUENCY_CSV" >&2
    exit 1
fi
if [[ -z "$ADAPTIVE_TOP_BLOCKS_FILE" ]]; then
    ADAPTIVE_TOP_BLOCKS_FILE="$OUT_ROOT/adaptive_top${TOP_K}_blocks_from_block_frequency.txt"
    awk -F, -v top="$TOP_K" 'NR > 1 && $2 != "" {print $2; count++; if (count >= top) exit}' "$ADAPTIVE_BLOCK_FREQUENCY_CSV" > "$ADAPTIVE_TOP_BLOCKS_FILE"
fi
if [[ ! -s "$ADAPTIVE_TOP_BLOCKS_FILE" ]]; then
    echo "ERROR: no adaptive layers found in $ADAPTIVE_TOP_BLOCKS_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Batch full-FLUX top-${TOP_K} attention generation"
echo "anomalies:   ${ANOMALIES[*]}"
echo "ref ids:     ${REF_IDS[*]}"
if [[ -n "$SAMPLES_PER_PAIR_STR" ]]; then
    echo "samples:     per-pair=${SAMPLES_PER_PAIR[*]}"
else
    echo "samples:     $SAMPLES_PER_ANOMALY per anomaly"
fi
echo "result root: $RESULT_ROOT"
echo "run name:    $RUN_NAME"
echo "out root:    $OUT_ROOT"
echo "size/steps:  ${SIZE}/${STEPS}"
echo "save steps:  ${SAVE_STEPS[*]}"
echo "aggregate steps: ${AGGREGATE_STEPS[*]}"
echo "adaptive check steps: ${ADAPTIVE_CHECK_STEPS[*]}"
echo "attention:   kind=$DIRECT_AGGREGATE_KIND polarity=$POLARITY roi=$ROI top_k=$TOP_K"
echo "PAMR:        run=$RUN_PAMR_REFINE iter=$PAMR_ITER seed_threshold=$PAMR_SEED_THRESHOLD threshold=$PAMR_THRESHOLD"
echo "MGAC:        mode=$MGAC_OUTPUT_MODE iter=$MGAC_ITER smoothing=$MGAC_SMOOTHING balloon=$MGAC_BALLOON roi_dilate=$MGAC_ROI_DILATE edge_alpha=$MGAC_EDGE_ALPHA weights=$MGAC_SCHARR_WEIGHT/$MGAC_WAVELET_WEIGHT gate_p=$MGAC_GATE_PERCENTILE gate_dilate=$MGAC_GATE_DILATE init_erode=$MGAC_INIT_ERODE keep_coarse=$MGAC_KEEP_COARSE edge_gate_roi=$MGAC_USE_EDGE_GATE_AS_ROI final_close=$MGAC_FINAL_CLOSE final_min_area=$MGAC_FINAL_MIN_AREA fill_holes=$MGAC_FINAL_FILL_HOLES"
echo "Q80 refine:  run=$RUN_Q80_APPEARANCE_REFINE percentile=$Q80_APPEARANCE_PERCENTILE gate_dilate=$Q80_APPEARANCE_GATE_DILATE min_area=$Q80_APPEARANCE_MIN_AREA grow=$Q80_APPEARANCE_GROW_RADIUS fg_erode=$Q80_APPEARANCE_FG_ERODE bg_ring=$Q80_APPEARANCE_BG_RING_DILATE keep_margin=$Q80_APPEARANCE_KEEP_MARGIN add_margin=$Q80_APPEARANCE_ADD_MARGIN max_fg_dist=$Q80_APPEARANCE_MAX_FG_DIST roi_dilate=$Q80_APPEARANCE_ROI_DILATE"
echo "coarse seed: open=$REFINE_COARSE_OPEN kernel=$REFINE_COARSE_OPEN_KERNEL iter=$REFINE_COARSE_OPEN_ITER internal_only=1"
echo "Contour:     run=$RUN_CONTOUR_REFINE mode=q80_outer_fill inner_erode=$CONTOUR_REFINE_INNER_ERODE clip_to_coarse=$CONTOUR_REFINE_CLIP_TO_COARSE edge_dilate=$CONTOUR_REFINE_EDGE_DILATE close=$CONTOUR_REFINE_CLOSE fill_holes=$CONTOUR_REFINE_FILL_HOLES component=$CONTOUR_REFINE_COMPONENT_MODE output=contour_refined_mask.png"
echo "save refine: mask_debug=$SAVE_MASK_DEBUG active_mask=$SAVE_ACTIVE_CONTOUR_MASK active_debug=$SAVE_ACTIVE_CONTOUR_DEBUG active_edge_map=$SAVE_ACTIVE_CONTOUR_EDGE_MAP"
echo "blocks file: $TOP_BLOCKS_FILE"
echo "adaptive blocks: kind=$ADAPTIVE_AGGREGATE_KIND polarity=$ADAPTIVE_POLARITY file=$ADAPTIVE_TOP_BLOCKS_FILE"
echo "device:      $DEVICE cuda_available=$CUDA_AVAILABLE clear_cuda_visible_devices=$CLEAR_CUDA_VISIBLE_DEVICES"
echo "offload:     cpu=$CPU_OFFLOAD sequential=$SEQUENTIAL_CPU_OFFLOAD"
echo "quantize:    $FULL_FLUX_QUANTIZE"
echo "model:       fill=$FLUX_FILL_PATH redux=$FLUX_REDUX_PATH"
echo "LoRA:        path=$LORA_PATH weight=$LORA_WEIGHT_NAME"
echo "adaptive:    ref_injection=$ADAPTIVE_REF_INJECTION mode=$ADAPTIVE_SCORE_MODE kind=$ADAPTIVE_AGGREGATE_SCORE_KIND inside_ratio=$ADAPTIVE_AGGREGATE_MIN_INSIDE_RATIO inside_mean=$ADAPTIVE_AGGREGATE_MIN_INSIDE_MEAN aggregate_min_area=$ADAPTIVE_AGGREGATE_MIN_AREA_RATIO contrast_ratio=$ADAPTIVE_AGGREGATE_MIN_CONTRAST_RATIO contrast_inside=$ADAPTIVE_AGGREGATE_MIN_CONTRAST_INSIDE_MEAN contrast_margin=$ADAPTIVE_AGGREGATE_MIN_CONTRAST_MARGIN outside_ring=$ADAPTIVE_AGGREGATE_OUTSIDE_RING_DILATE ref_threshold=$ADAPTIVE_REF_ATTENTION_THRESHOLD scale=${ADAPTIVE_REF_BASE_SCALE}-${ADAPTIVE_REF_MAX_SCALE} boost=$ADAPTIVE_REF_BOOST trigger_min=$ADAPTIVE_REF_TRIGGER_MIN_SCALE decay=$ADAPTIVE_REF_DECAY decay_min_score=$ADAPTIVE_REF_DECAY_MIN_SCORE"
echo "target mask: source=$TARGET_MASK_SOURCE double_prob=$RANDOM_MASK_DOUBLE_PROB area_ratio=${RANDOM_MASK_AREA_MIN_RATIO}-${RANDOM_MASK_AREA_MAX_RATIO} rotate=$RANDOM_MASK_ROTATE attempts=$RANDOM_MASK_ATTEMPTS ref_dilate=$REFERENCE_MASK_DILATE_ITERATIONS ref_vertical_shift=$REFERENCE_MASK_VERTICAL_SHIFT_RATIO"
echo "ref perturb: noise_std=$REF_TOKEN_NOISE_STD dropout=$REF_TOKEN_DROPOUT scale_jitter=$REF_TOKEN_SCALE_JITTER span_dropout=$REF_TOKEN_SPAN_DROPOUT span_len=$REF_TOKEN_SPAN_LEN seed_offset=$REF_TOKEN_PERTURB_SEED_OFFSET"
echo "ref augment: bank=$REF_AUGMENT_BANK_SIZE rotate=$REF_AUGMENT_ROTATE scale_jitter=$REF_AUGMENT_SCALE_JITTER translate=$REF_AUGMENT_TRANSLATE_RATIO brightness=$REF_AUGMENT_BRIGHTNESS contrast=$REF_AUGMENT_CONTRAST seed_offset=$REF_AUGMENT_SEED_OFFSET"
echo "shape-K:     enabled=$SHAPE_K_REMOVAL mode=$SHAPE_K_MODE eta=$SHAPE_K_ETA suppress=$SHAPE_K_SUPPRESS_SCALE ratio=${SHAPE_K_START_RATIO}-${SHAPE_K_END_RATIO} steps=${SHAPE_K_START_STEP}-${SHAPE_K_END_STEP} scope=$SHAPE_K_BLOCK_SCOPE edge=$SHAPE_K_EDGE_METHOD blocks=${SHAPE_K_BLOCKS_STR:-scope_default}"
echo "log:         $RUN_LOG"
echo "adaptive csv:$ADAPTIVE_LOG"
echo "============================================================"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "DRY_RUN=1: configuration validated; generation was not started."
    exit 0
fi

conda run --no-capture-output -n "$CONDA_ENV" python -u generation_attention/batch_visualize_flux_attention.py \
    --anomalies "${ANOMALIES[@]}" \
    --ref-ids "${REF_IDS[@]}" \
    --samples-per-anomaly "$SAMPLES_PER_ANOMALY" \
    "${SAMPLES_PER_PAIR_ARGS[@]}" \
    --source-root "$SOURCE_IMAGE_ROOT" \
    --ref-image-root "$REF_IMAGE_ROOT" \
    --ref-mask-root "$REF_MASK_ROOT" \
    --out-root "$OUT_ROOT" \
    --seed "$SEED" \
    --start-index "$START_INDEX" \
    --device "$DEVICE" \
    --size "$SIZE" \
    --num-inference-steps "$STEPS" \
    --save-steps "${SAVE_STEPS[@]}" \
    --direct-aggregate-steps "${AGGREGATE_STEPS[@]}" \
    --adaptive-check-steps "${ADAPTIVE_CHECK_STEPS[@]}" \
    "${TARGET_MASK_ARGS[@]}" \
    --object-prompt "$OBJECT_PROMPT" \
    --object-support-erosion "$OBJECT_SUPPORT_EROSION" \
    --random-mask-area-min-ratio "$RANDOM_MASK_AREA_MIN_RATIO" \
    --random-mask-area-max-ratio "$RANDOM_MASK_AREA_MAX_RATIO" \
    --random-mask-rotate "$RANDOM_MASK_ROTATE" \
    --random-mask-attempts "$RANDOM_MASK_ATTEMPTS" \
    --random-mask-double-prob "$RANDOM_MASK_DOUBLE_PROB" \
    --direct-aggregate-kind "$DIRECT_AGGREGATE_KIND" \
    --direct-selected-blocks-file "$TOP_BLOCKS_FILE" \
    --direct-top-k "$TOP_K" \
    --direct-block-frequency-csv "$BLOCK_FREQUENCY_CSV" \
    --direct-polarity "$POLARITY" \
    --direct-roi "$ROI" \
    --direct-hist-threshold-scale "$HIST_THRESHOLD_SCALE" \
    --direct-hist-threshold-offset "$HIST_THRESHOLD_OFFSET" \
    --direct-component-mode "$COMPONENT_MODE" \
    "${FILL_ARGS[@]}" \
    --direct-close-iterations "$CLOSE_ITERATIONS" \
    --direct-dilate-iterations "$DILATE_ITERATIONS" \
    --adaptive-roi "$ADAPTIVE_ROI" \
    --adaptive-aggregate-kind "$ADAPTIVE_AGGREGATE_KIND" \
    --adaptive-selected-blocks-file "$ADAPTIVE_TOP_BLOCKS_FILE" \
    --adaptive-block-frequency-csv "$ADAPTIVE_BLOCK_FREQUENCY_CSV" \
    --adaptive-polarity "$ADAPTIVE_POLARITY" \
    "${ADAPTIVE_ARGS[@]}" \
    "${REF_TOKEN_PERTURB_ARGS[@]}" \
    "${REF_AUGMENT_ARGS[@]}" \
    "${SHAPE_K_ARGS[@]}" \
    --log-file "$RUN_LOG" \
    --adaptive-log-file "$ADAPTIVE_LOG" \
    --flux-fill-path "$FLUX_FILL_PATH" \
    --flux-redux-path "$FLUX_REDUX_PATH" \
    --lora-path "$LORA_PATH" \
    --lora-weight-name "$LORA_WEIGHT_NAME" \
    "${LOCAL_FILES_ARGS[@]}" \
    "${QUANT_ARGS[@]}" \
    "${OFFLOAD_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    "${DEBUG_ARGS[@]}"

EXAMPLE_DIR="$OUT_ROOT/${ANOMALIES[0]}/ref_${REF_IDS[0]}/000"
echo "done:"
echo "  out root: $OUT_ROOT"
echo "  log:      $RUN_LOG"
echo "  example:  $EXAMPLE_DIR/edit.png"
echo "  coarse:   $EXAMPLE_DIR/coarse_mask.png"
echo "  contour mask: $EXAMPLE_DIR/contour_refined_mask.png"

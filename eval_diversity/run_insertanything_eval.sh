#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

OBJ="${OBJ:-hazelnut}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-$REPO_ROOT/outputs}"
RESULT_ROOT="${RESULT_ROOT:-$PROPOSAL_ROOT/hazelnut_rediff_ad}"
GENERATED_LAYOUT="${GENERATED_LAYOUT:-insert-anything}"
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

REAL_ROOT="${REAL_ROOT:-${MVTEC_ROOT:-$REPO_ROOT/datasets}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval_diversity/result}"
TMP_DIR="${TMP_DIR:-$REPO_ROOT/eval_diversity/tmp}"
CONDA_ENV="${CONDA_ENV:-rediff-ad}"
GPU="${GPU:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
CLEAN_LINKS="${CLEAN_LINKS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LPIPS_CLUSTER_SIZE="${LPIPS_CLUSTER_SIZE:-50}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-16}"
LPIPS_PAIR_BATCH_SIZE="${LPIPS_PAIR_BATCH_SIZE:-16}"
LPIPS_SEED="${LPIPS_SEED:-2026}"
MAX_IMAGES="${MAX_IMAGES:-250}"
KID_SUBSAMPLE_SIZE="${KID_SUBSAMPLE_SIZE:-50}"
KID_NUM_SUBSETS="${KID_NUM_SUBSETS:-200}"
KID_SEED="${KID_SEED:-2026}"
KID_BATCH_SIZE="${KID_BATCH_SIZE:-32}"
ISC_SPLITS="${ISC_SPLITS:-10}"

RUN_KID="${RUN_KID:-1}"
RUN_IS="${RUN_IS:-1}"
RUN_LPIPS="${RUN_LPIPS:-1}"
USE_CPU_FOR_IS="${USE_CPU_FOR_IS:-0}"

RUN_TAG="$(basename "$RESULT_ROOT")"
DATASET_NAME="${DATASET_NAME:-$(basename "$PROPOSAL_ROOT")}"
RESULT_TAG="${RESULT_TAG:-$RUN_TAG}"
TMP_RUN_DIR="$TMP_DIR/$RUN_TAG"
METRIC_TMP_DIR="${METRIC_TMP_DIR:-/tmp/eval_diversity_metrics/$RESULT_TAG/$OBJ}"
RESULT_DIR="${RESULT_DIR:-${RESULTS_DIR:-$OUT_DIR/$RESULT_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/${OBJ}_summary.csv}"

run_python() {
    if [ -n "$CONDA_ENV" ]; then
        conda run -n "$CONDA_ENV" python "$@"
    else
        python "$@"
    fi
}

ensure_python_module() {
    local module="$1"
    local package="$2"

    if ! run_python -c "import ${module}" >/dev/null 2>&1; then
        echo "Missing Python module '$module' (package '$package')." >&2
        echo "Install the pinned evaluation dependencies before running metrics:" >&2
        echo "  pip install -r requirements-eval.txt" >&2
        return 1
    fi
}

csv_kid_value() {
    local csv_path="$1"
    local category="$2"
    local defect="$3"
    local column="$4"

    if [ ! -s "$csv_path" ]; then
        return 0
    fi

    awk -F, -v category="$category" -v defect="$defect" -v column="$column" \
        'NR > 1 && $1 == category && $2 == defect {print $column; exit}' "$csv_path"
}

csv_category_score() {
    local csv_path="$1"
    local category="$2"
    local column="${3:-2}"

    if [ ! -s "$csv_path" ]; then
        return 0
    fi

    awk -F, -v category="$category" -v column="$column" '$1 == category {print $column; exit}' "$csv_path"
}

append_summary_mean() {
    local csv_path="$1"
    local tmp_path="${csv_path}.mean.tmp"

    awk -F, '
        BEGIN {
            OFS = ","
        }
        NR == 1 {
            next
        }
        $3 == "mean" || $3 == "overall" {
            next
        }
        {
            dataset = $1
            obj = $2
            for (i = 5; i <= 9; i++) {
                if ($i ~ /^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$/) {
                    sum[i] += $i
                    count[i] += 1
                }
            }
        }
        function avg(i) {
            return count[i] > 0 ? sprintf("%.10g", sum[i] / count[i]) : ""
        }
        END {
            if (dataset == "") {
                exit
            }
            printf "%s,%s,mean,,%s,%s,%s,%s,%s\n", dataset, obj, avg(5), avg(6), avg(7), avg(8), avg(9)
        }
    ' "$csv_path" > "$tmp_path"

    if [ -s "$tmp_path" ]; then
        cat "$tmp_path" >> "$csv_path"
    fi
    rm -f "$tmp_path"
}

if [ ! -d "$RESULT_ROOT" ]; then
    echo "Missing RESULT_ROOT: $RESULT_ROOT"
    exit 1
fi

if { [ "$RUN_KID" -eq 1 ] || [ "$RUN_LPIPS" -eq 1 ]; } && [ ! -d "$REAL_ROOT/$OBJ/test" ]; then
    echo "Missing real dataset path: $REAL_ROOT/$OBJ/test"
    exit 1
fi

if [ -n "$CONDA_ENV" ] && ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Set CONDA_ENV= to use the current python environment."
    exit 1
fi

mkdir -p "$TMP_RUN_DIR" "$METRIC_TMP_DIR" "$RESULT_DIR"

if [ "$DRY_RUN" != "1" ]; then
    printf "dataset,obj,defect,n_generated,kid_x1000,kid_std_x1000,is_mean,ic_lpips_mean,ic_lpips_std\n" > "$SUMMARY_CSV"
fi

for ANO in "${ANOMALY_LIST[@]}"; do
    if [ "$GENERATED_LAYOUT" = "tf-idg" ]; then
        ANO_DIR="$RESULT_ROOT/test/$ANO"
    else
        ANO_DIR="$RESULT_ROOT/$ANO"
    fi
    if [ ! -d "$ANO_DIR" ]; then
        if [ "$SKIP_MISSING" = "1" ]; then
            echo "[skip] missing anomaly dir: $ANO_DIR"
            continue
        fi
        echo "Missing anomaly dir: $ANO_DIR"
        exit 1
    fi

    IMAGE_DIR="$TMP_RUN_DIR/$ANO"
    KID_CSV="$METRIC_TMP_DIR/${ANO}_kid.csv"
    IS_CSV="$METRIC_TMP_DIR/${ANO}_is.csv"
    LPIPS_CSV="$METRIC_TMP_DIR/${ANO}_ic_lpips.csv"

    mkdir -p "$IMAGE_DIR"
    if [ "$CLEAN_LINKS" = "1" ]; then
        find "$IMAGE_DIR" -maxdepth 1 -type l -name "*.png" -delete
    fi

    count=0
    case "$GENERATED_LAYOUT" in
        anomaly-diffusion|seas|anostyle|dualanodiff|self-anomalydiffusion)
            SOURCE_IMAGE_DIR="$ANO_DIR/image"
            if [ -d "$SOURCE_IMAGE_DIR" ]; then
                while IFS= read -r -d '' img; do
                    if [ "$count" -ge "$MAX_IMAGES" ]; then
                        continue
                    fi
                    name="$(printf "%03d.png" "$count")"
                    img_abs="$(realpath "$img")"

                    ln -sf "$img_abs" "$IMAGE_DIR/$name"
                    count=$((count + 1))
                done < <(find "$SOURCE_IMAGE_DIR" -maxdepth 1 \( -type f -o -type l \) \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)
            fi
            ;;
        o2mag-flat)
            while IFS= read -r -d '' img; do
                if [ "$count" -ge "$MAX_IMAGES" ]; then
                    continue
                fi
                name="$(printf "%03d.png" "$count")"
                img_abs="$(realpath "$img")"

                ln -sf "$img_abs" "$IMAGE_DIR/$name"
                count=$((count + 1))
            done < <(find "$ANO_DIR" -maxdepth 1 \( -type f -o -type l \) \( -iname "*_edit.png" -o -iname "*_edit.jpg" -o -iname "*_edit.jpeg" \) -print0 | sort -z)
            ;;
        tf-idg)
            while IFS= read -r -d '' img; do
                if [ "$count" -ge "$MAX_IMAGES" ]; then
                    continue
                fi
                name="$(printf "%03d.png" "$count")"
                img_abs="$(realpath "$img")"

                ln -sf "$img_abs" "$IMAGE_DIR/$name"
                count=$((count + 1))
            done < <(find "$ANO_DIR" -maxdepth 1 \( -type f -o -type l \) \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)
            ;;
        insert-anything)
            while IFS= read -r -d '' img; do
                sample_dir="$(basename "$(dirname "$img")")"
                ref_dir="$(basename "$(dirname "$(dirname "$img")")")"
                if [[ ! "$sample_dir" =~ ^[0-9]+$ || "$ref_dir" != ref_* ]]; then
                    continue
                fi
                if [ "$count" -ge "$MAX_IMAGES" ]; then
                    continue
                fi
                name="$(printf "%03d.png" "$count")"
                img_abs="$(realpath "$img")"

                ln -sf "$img_abs" "$IMAGE_DIR/$name"
                count=$((count + 1))
            done < <(find "$ANO_DIR" -mindepth 3 -maxdepth 3 \( -type f -o -type l \) -name "$IMAGE_NAME" -print0 | sort -z)
            ;;
        *)
            echo "Unsupported GENERATED_LAYOUT: $GENERATED_LAYOUT"
            exit 1
            ;;
    esac

    echo "============================================================"
    echo "OBJ=$OBJ"
    echo "ANO=$ANO"
    echo "RESULT_ROOT=$RESULT_ROOT"
    echo "GENERATED_LAYOUT=$GENERATED_LAYOUT"
    echo "RESULT_DIR=$RESULT_DIR"
    echo "REAL_ROOT=$REAL_ROOT"
    echo "TMP_IMAGE_DIR=$IMAGE_DIR"
    echo "images=$count"
    echo "MAX_IMAGES=$MAX_IMAGES"
    echo "SUMMARY_CSV=$SUMMARY_CSV"
    echo "============================================================"

    if [ "$count" -eq 0 ]; then
        if [ "$SKIP_MISSING" = "1" ]; then
            echo "[skip] no $IMAGE_NAME found under $ANO_DIR"
            continue
        fi
        echo "No $IMAGE_NAME found under $ANO_DIR"
        exit 1
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] prepared image links only; metrics skipped"
        continue
    fi

    kid_value=""
    kid_std_value=""
    is_value=""
    lpips_value=""
    lpips_std_value=""

    if [ "$RUN_KID" -eq 1 ]; then
        echo
        echo "[KID] $OBJ/$ANO"
        run_python "$SCRIPT_DIR/compute-kid.py" \
            --real_path "$REAL_ROOT" \
            --generated_path "$TMP_RUN_DIR" \
            --categories "$OBJ" \
            --defects "$ANO" \
            --kid-subsample-size "$KID_SUBSAMPLE_SIZE" \
            --kid-num-subsets "$KID_NUM_SUBSETS" \
            --kid-seed "$KID_SEED" \
            --kid-batch-size "$KID_BATCH_SIZE" \
            --output_csv "$KID_CSV"
        kid_value="$(csv_kid_value "$KID_CSV" "$OBJ" "$ANO" 3)"
        kid_std_value="$(csv_kid_value "$KID_CSV" "$OBJ" "$ANO" 4)"
    fi

    if [ "$RUN_IS" -eq 1 ]; then
        echo
        echo "[IS] $OBJ/$ANO"
        ensure_python_module torch_fidelity torch-fidelity

        if [ "$USE_CPU_FOR_IS" -eq 1 ]; then
            run_python "$SCRIPT_DIR/compute-is.py" \
                --generated_path "$TMP_RUN_DIR" \
                --categories "$OBJ" \
                --defects "$ANO" \
                --isc-splits "$ISC_SPLITS" \
                --cpu \
                --output_csv "$IS_CSV"
        else
            run_python "$SCRIPT_DIR/compute-is.py" \
                --generated_path "$TMP_RUN_DIR" \
                --categories "$OBJ" \
                --defects "$ANO" \
                --isc-splits "$ISC_SPLITS" \
                --gpu "$GPU" \
                --output_csv "$IS_CSV"
        fi
        is_value="$(csv_category_score "$IS_CSV" "$OBJ")"
    fi

    if [ "$RUN_LPIPS" -eq 1 ]; then
        echo
        echo "[IC-LPIPS] $OBJ/$ANO"

        ensure_python_module lpips lpips

        run_python "$SCRIPT_DIR/compute-ic-lpipis.py" \
            --real_path "$REAL_ROOT" \
            --generated_path "$TMP_RUN_DIR" \
            --categories "$OBJ" \
            --defects "$ANO" \
            --cluster-size "$LPIPS_CLUSTER_SIZE" \
            --lpips-batch-size "$LPIPS_BATCH_SIZE" \
            --lpips-pair-batch-size "$LPIPS_PAIR_BATCH_SIZE" \
            --seed "$LPIPS_SEED" \
            --output_csv "$LPIPS_CSV"
        lpips_value="$(csv_category_score "$LPIPS_CSV" "$OBJ" 2)"
        lpips_std_value="$(csv_category_score "$LPIPS_CSV" "$OBJ" 3)"
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$DATASET_NAME" "$OBJ" "$ANO" "$count" "$kid_value" "$kid_std_value" "$is_value" "$lpips_value" "$lpips_std_value" >> "$SUMMARY_CSV"
done

if [ "$DRY_RUN" != "1" ]; then
    append_summary_mean "$SUMMARY_CSV"
fi

echo
echo "summary: $SUMMARY_CSV"
echo "done"

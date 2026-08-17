#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

export RUN_NAME="${RUN_NAME:-smoke_rediff_ad_seed309}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/$RUN_NAME}"
export ANOMALIES_STR="${ANOMALIES_STR:-crack}"
export SAMPLES_PER_ANOMALY="${SAMPLES_PER_ANOMALY:-1}"
export SEED="${SEED:-309}"
export OVERWRITE="${OVERWRITE:-1}"
export FULL_FLUX_QUANTIZE="${FULL_FLUX_QUANTIZE:-int4}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
export SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}"

exec bash "$PROJECT_ROOT/scripts/run_hazelnut_t2r.sh"

#!/usr/bin/env bash
# Shared stage runner. CPU/GPU resources are selected by the calling sbatch template.
set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT to the checkout on /scratch/NetID}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON to a CUDA Python or container wrapper}"
: "${DEMO_RUN:?Set DEMO_RUN to a persistent wall-analysis directory}"
DEMO_MODE="${DEMO_MODE:-2d}"
DEMO_STAGE="${DEMO_STAGE:-analyze}"
if [[ "${DEMO_MODE}" != 2d && "${DEMO_MODE}" != 3d ]]; then
  echo "DEMO_MODE must be 2d or 3d" >&2; exit 2
fi
export PYTHONPATH="${DEMO_PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_CACHE="${DEMO_SCRATCH:-/scratch/${USER}}/hold-cache/huggingface/hub"
export YOLO_CONFIG_DIR="${DEMO_SCRATCH:-/scratch/${USER}}/hold-cache/yolo26"
export MPLCONFIGDIR="${DEMO_SCRATCH:-/scratch/${USER}}/hold-cache/matplotlib"
export OMP_NUM_THREADS=4
export QT_QPA_PLATFORM=offscreen
cd "${DEMO_PROJECT}"
mkdir -p "${DEMO_RUN}" "${HF_HUB_CACHE}" "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}"
reconstruction=()
if [[ "${DEMO_MODE}" == 3d ]]; then
  reconstruction=(--reconstruction-path "${DEMO_RUN}/reconstruction/reconstruction.json")
fi
scale=()
if [[ -n "${DEMO_SCALE_M_PER_UNIT:-}" ]]; then scale=(--scale-m-per-unit "${DEMO_SCALE_M_PER_UNIT}"); fi
if [[ -n "${DEMO_SCALE_REFERENCE:-}" ]]; then scale+=(--scale-reference "${DEMO_SCALE_REFERENCE}"); fi
types=()
if [[ -n "${DEMO_TYPE_WEIGHTS:-}" ]]; then
  types=(--type-weights "${DEMO_TYPE_WEIGHTS}" --type-imgsz "${DEMO_TYPE_IMGSZ:-1024}"
         --type-min-score "${DEMO_TYPE_MIN_SCORE:-0.5}")
fi
annotations=()
if [[ -n "${DEMO_ANNOTATIONS:-}" ]]; then annotations=(--annotations-path "${DEMO_ANNOTATIONS}"); fi
case "${DEMO_STAGE}" in
  prepare)
    : "${DEMO_SOURCE:?Set DEMO_SOURCE to one photo, a photo folder/ZIP or a video}"
    exec "${DEMO_PYTHON}" -m hold_detection.cli prepare-capture "${DEMO_SOURCE}" \
      --output "${DEMO_RUN}/capture" --mode "${DEMO_MODE}" --max-frames "${DEMO_MAX_FRAMES:-48}"
    ;;
  fetch-sam3)
    : "${DEMO_SAM_WEIGHTS:?Set DEMO_SAM_WEIGHTS to the desired sam3.pt path}"
    exec "${DEMO_PYTHON}" -m hold_detection.cli fetch-sam3 --output "$(dirname "${DEMO_SAM_WEIGHTS}")"
    ;;
  reconstruct)
    exec "${DEMO_PYTHON}" -m hold_detection.cli reconstruct-wall "${DEMO_RUN}/capture" \
      --output "${DEMO_RUN}/reconstruction" --colmap "${DEMO_COLMAP:-colmap}" \
      --quality "${DEMO_RECON_QUALITY:-medium}" "${scale[@]}"
    ;;
  segment)
    : "${DEMO_SAM_WEIGHTS:?Set DEMO_SAM_WEIGHTS to the downloaded checkpoint}"
    exec "${DEMO_PYTHON}" -m hold_detection.cli segment-capture "${DEMO_RUN}/capture" \
      --run "${DEMO_RUN}" --sam-weights "${DEMO_SAM_WEIGHTS}" --device 0 \
      --stop-file "${DEMO_RUN}/STOP_REQUESTED" --confidence "${DEMO_CONFIDENCE:-0.5}" \
      "${reconstruction[@]}" "${types[@]}"
    ;;
  export)
    exec "${DEMO_PYTHON}" -m hold_detection.cli export-wall "${DEMO_RUN}" \
      --mode "${DEMO_MODE}" "${reconstruction[@]}" "${annotations[@]}"
    ;;
  match)
    : "${DEMO_ANNOTATED_GLB:?Set DEMO_ANNOTATED_GLB}"
    : "${DEMO_ANNOTATION_TRANSFORM:?Set DEMO_ANNOTATION_TRANSFORM to a 4x4 matrix JSON}"
    exec "${DEMO_PYTHON}" -m hold_detection.cli match-annotated-glb "${DEMO_RUN}/scene.json" \
      --annotated-glb "${DEMO_ANNOTATED_GLB}" --transform "${DEMO_ANNOTATION_TRANSFORM}" \
      --output "${DEMO_RUN}/scene_matched.json"
    ;;
  analyze)
    : "${DEMO_SOURCE:?Set DEMO_SOURCE}"
    : "${DEMO_SAM_WEIGHTS:?Set DEMO_SAM_WEIGHTS}"
    edits=()
    if [[ -n "${DEMO_ANNOTATIONS:-}" ]]; then edits=(--annotations "${DEMO_ANNOTATIONS}"); fi
    exec "${DEMO_PYTHON}" -m hold_detection.cli analyze-wall "${DEMO_SOURCE}" \
      --output "${DEMO_RUN}" --mode "${DEMO_MODE}" --sam-weights "${DEMO_SAM_WEIGHTS}" \
      --device 0 --max-frames "${DEMO_MAX_FRAMES:-48}" --colmap "${DEMO_COLMAP:-colmap}" \
      --quality "${DEMO_RECON_QUALITY:-medium}" --confidence "${DEMO_CONFIDENCE:-0.5}" \
      "${scale[@]}" "${types[@]}" "${edits[@]}"
    ;;
  *) echo "Unknown DEMO_STAGE: ${DEMO_STAGE}" >&2; exit 2 ;;
esac

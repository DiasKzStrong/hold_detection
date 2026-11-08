#!/usr/bin/env bash
# Submit only when the human explicitly runs this script on the HPC login node.
set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT}"
: "${DEMO_SOURCE:?Set DEMO_SOURCE}"
: "${DEMO_RUN:?Set DEMO_RUN}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON}"
: "${DEMO_SAM_WEIGHTS:?Set DEMO_SAM_WEIGHTS to a predownloaded checkpoint}"
DEMO_MODE="${DEMO_MODE:-2d}"
if [[ "${DEMO_MODE}" != 2d && "${DEMO_MODE}" != 3d ]]; then
  echo "DEMO_MODE must be 2d or 3d" >&2; exit 2
fi
export DEMO_MODE
cpu="${DEMO_PROJECT}/scripts/slurm/wall-cpu.sbatch"
gpu="${DEMO_PROJECT}/scripts/slurm/wall-gpu.sbatch"
prepare=$(sbatch --parsable --export=ALL,DEMO_STAGE=prepare "${cpu}")
previous="${prepare%%;*}"
if [[ "${DEMO_MODE}" == 3d ]]; then
  reconstruct=$(sbatch --parsable --dependency="afterok:${previous}" --export=ALL,DEMO_STAGE=reconstruct "${gpu}")
  previous="${reconstruct%%;*}"
fi
segment=$(sbatch --parsable --dependency="afterok:${previous}" --export=ALL,DEMO_STAGE=segment "${gpu}")
export_job=$(sbatch --parsable --dependency="afterok:${segment%%;*}" --export=ALL,DEMO_STAGE=export "${cpu}")
echo "Preparation: ${prepare}; segmentation: ${segment}; JSON export: ${export_job}"

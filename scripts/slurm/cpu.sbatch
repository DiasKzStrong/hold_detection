#!/usr/bin/env bash
# Dataset preparation and a cached SigLIP linear head do not need GPU allocation.
#SBATCH --job-name=hold-cpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=holds-%x-%j.log

set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON to Python or a container-wrapper executable}"
: "${DEMO_RUN:?Set DEMO_RUN to the output directory}"
DEMO_STAGE="${DEMO_STAGE:-siglip-head}"
DEMO_SCRATCH="${DEMO_SCRATCH:-/scratch/${USER}}"
export PYTHONPATH="${DEMO_PROJECT}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export DEMO_CPU_ONLY=1
export TORCH_HOME="${DEMO_SCRATCH}/hold-cache/torch"
export HF_HUB_CACHE="${DEMO_SCRATCH}/hold-cache/huggingface/hub"
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="${DEMO_SCRATCH}/hold-cache/matplotlib"
export OMP_NUM_THREADS=8
cd "${DEMO_PROJECT}"
mkdir -p "${TORCH_HOME}" "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

case "${DEMO_STAGE}" in
  prepare)
    : "${DEMO_SOURCE:?Set DEMO_SOURCE to a labeled YOLO/COCO export directory or ZIP}"
    srun "${DEMO_PYTHON}" -m hold_detection.cli prepare-crops "${DEMO_SOURCE}" --output "${DEMO_RUN}"
    ;;
  fetch-encoders)
    srun "${DEMO_PYTHON}" -m hold_detection.cli fetch-encoders
    ;;
  siglip-head)
    : "${DEMO_FEATURES:?Set DEMO_FEATURES to a completed SigLIP feature cache}"
    resume=()
    if [[ -f "${DEMO_RUN}/last.pt" ]]; then resume=(--resume); fi
    srun "${DEMO_PYTHON}" -m hold_detection.cli train-classifier \
      --architecture siglip --features "${DEMO_FEATURES}" --output "${DEMO_RUN}" \
      --epochs "${DEMO_EPOCHS:-30}" --batch-size "${DEMO_BATCH_SIZE:-512}" \
      --device cpu --workers 0 --threads 8 --checkpoint-every 100 \
      --time-budget-seconds "${DEMO_TIME_BUDGET_SECONDS:-6900}" "${resume[@]}"
    ;;
  *) echo "DEMO_STAGE must be prepare, fetch-encoders or siglip-head" >&2; exit 2 ;;
esac

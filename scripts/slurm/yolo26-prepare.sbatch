#!/usr/bin/env bash
# Prepare the existing YOLO export without allocating an A100.
#SBATCH --job-name=holds-yolo26-data
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=holds-%x-%j.log

set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT to the repository checkout on /scratch/NetID}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON to the YOLO26 Python or container wrapper}"
: "${DEMO_SOURCE:?Set DEMO_SOURCE to the Roboflow YOLO export ZIP or folder}"
: "${DEMO_DATA:?Set DEMO_DATA to a NEW prepared dataset directory}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 DEMO_CPU_ONLY=1
export PYTHONPATH="${DEMO_PROJECT}/src"
DEMO_SCRATCH="${DEMO_SCRATCH:-/scratch/${USER}}"
export YOLO_CONFIG_DIR="${DEMO_SCRATCH}/hold-cache/yolo26"
export MPLCONFIGDIR="${DEMO_SCRATCH}/hold-cache/matplotlib"
mkdir -p "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}"
cd "${DEMO_PROJECT}"
if [[ ! -e "${DEMO_SOURCE}" && -n "${DEMO_ROBOFLOW_VERSION:-}" ]]; then
  srun "${DEMO_PYTHON}" -m hold_detection.cli download-yolo --output "${DEMO_SOURCE}" \
    --version "${DEMO_ROBOFLOW_VERSION}" --project "${DEMO_ROBOFLOW_PROJECT:-holdclassificatore_v3_mix}"
fi
srun "${DEMO_PYTHON}" scripts/train_yolo26.py prepare "${DEMO_SOURCE}" --output "${DEMO_DATA}"
# Cache the official initialization checkpoint on a network-enabled CPU node.
srun "${DEMO_PYTHON}" -c 'import sys; from ultralytics import YOLO; YOLO(sys.argv[1])' "${DEMO_MODEL:-yolo26m.pt}"

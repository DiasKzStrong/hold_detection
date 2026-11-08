#!/usr/bin/env bash
#SBATCH --job-name=holds-yolo26-eval
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=holds-%x-%j.log
set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON}"
: "${DEMO_DATA:?Set DEMO_DATA}"
: "${DEMO_RUN:?Set DEMO_RUN to the YOLO training run}"
export PYTHONPATH="${DEMO_PROJECT}/src"
cd "${DEMO_PROJECT}"
srun "${DEMO_PYTHON}" scripts/train_yolo26.py evaluate --data "${DEMO_DATA}/data.yaml" \
  --output "${DEMO_RUN}" --split "${DEMO_SPLIT:-val}" --device 0 --imgsz "${DEMO_IMAGE_SIZE:-1024}"
